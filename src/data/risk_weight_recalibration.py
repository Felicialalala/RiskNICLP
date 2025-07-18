"""
RiskNICLP-5G 风险权重重新校准模块
解决Flow节点风险虚高和孤立高风险节点问题

基于实际数据特征：
- Flow节点平均风险0.586（虚高）
- 高风险节点仅占3.3%但主导负采样
- 负样本风险(0.609) > 正样本风险(0.302)的倒挂问题
"""

import torch
import dgl
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class RiskRecalibrationConfig:
    """风险重新校准配置"""
    # 节点类型权重调整
    flow_risk_decay_factor: float = 0.7  # Flow节点风险衰减系数
    alert_risk_boost_factor: float = 1.5  # Alert节点风险增强系数
    ue_risk_baseline: float = 0.1  # UE节点基础风险
    
    # 连接性权重
    isolated_penalty: float = 0.3  # 孤立节点惩罚
    propagation_boost: float = 0.2  # 传播链增强
    
    # 风险阈值重新定义
    high_risk_threshold: float = 0.6  # 降低高风险阈值
    medium_risk_threshold: float = 0.3
    low_risk_threshold: float = 0.1
    
    # 归一化参数
    enable_type_wise_normalization: bool = True
    target_distribution: Dict[str, float] = None
    
    def __post_init__(self):
        if self.target_distribution is None:
            # 目标风险分布：高风险20%，中风险40%，低风险40%
            self.target_distribution = {
                'high_risk': 0.2,
                'medium_risk': 0.4, 
                'low_risk': 0.4
            }

class RiskWeightRecalibrator:
    """风险权重重新校准器"""
    
    def __init__(self, config: RiskRecalibrationConfig = None):
        self.config = config or RiskRecalibrationConfig()
        self.logger = logging.getLogger(__name__)
        
        # 统计信息
        self.calibration_stats = {
            'original_distribution': {},
            'calibrated_distribution': {},
            'adjustment_summary': {}
        }
    
    def recalibrate_graph_risks(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        重新校准图中所有节点的风险分数
        
        Args:
            graph: 输入的异质图
            
        Returns:
            calibrated_graph: 校准后的图
        """
        self.logger.info("🔧 开始风险权重重新校准...")
        
        # 复制图以避免修改原图
        calibrated_graph = graph.clone()
        
        # 分析原始风险分布
        original_stats = self._analyze_risk_distribution(graph)
        self.calibration_stats['original_distribution'] = original_stats
        
        # 逐节点类型校准
        for node_type in graph.ntypes:
            if 'risk_score' in graph.nodes[node_type].data:
                calibrated_risks = self._recalibrate_node_type_risks(
                    graph, node_type
                )
                calibrated_graph.nodes[node_type].data['risk_score'] = calibrated_risks
                
                self.logger.info(f"✅ {node_type}节点风险校准完成")
        
        # 分析校准后分布
        calibrated_stats = self._analyze_risk_distribution(calibrated_graph)
        self.calibration_stats['calibrated_distribution'] = calibrated_stats
        
        # 生成调整摘要
        self._generate_adjustment_summary()
        
        self.logger.info("🎉 风险权重重新校准完成")
        return calibrated_graph
    
    def _recalibrate_node_type_risks(self, graph: dgl.DGLGraph, node_type: str) -> torch.Tensor:
        """重新校准特定节点类型的风险分数"""
        original_risks = graph.nodes[node_type].data['risk_score'].clone()
        num_nodes = len(original_risks)
        
        if num_nodes == 0:
            return original_risks
        
        self.logger.info(f"🎯 校准{node_type}节点风险 (共{num_nodes}个节点)")
        
        # 步骤1：基于节点类型的基础调整
        adjusted_risks = self._apply_node_type_adjustment(
            original_risks, node_type
        )
        
        # 步骤2：基于连接性的动态调整
        connectivity_adjusted_risks = self._apply_connectivity_adjustment(
            graph, node_type, adjusted_risks
        )
        
        # 步骤3：分布归一化
        if self.config.enable_type_wise_normalization:
            normalized_risks = self._apply_distribution_normalization(
                connectivity_adjusted_risks, node_type
            )
        else:
            normalized_risks = connectivity_adjusted_risks
        
        # 确保风险分数在[0,1]范围内
        normalized_risks = torch.clamp(normalized_risks, 0.0, 1.0)
        
        return normalized_risks
    
    def _apply_node_type_adjustment(self, risks: torch.Tensor, node_type: str) -> torch.Tensor:
        """基于节点类型的基础风险调整"""
        if node_type == 'Flow':
            # Flow节点风险衰减：解决虚高问题
            adjusted_risks = risks * self.config.flow_risk_decay_factor
            self.logger.info(f"   Flow节点风险衰减: {self.config.flow_risk_decay_factor}")
            
        elif node_type == 'Alert':
            # Alert节点风险增强：提升告警重要性
            adjusted_risks = torch.clamp(
                risks * self.config.alert_risk_boost_factor, 0.0, 1.0
            )
            self.logger.info(f"   Alert节点风险增强: {self.config.alert_risk_boost_factor}")
            
        elif node_type == 'UE':
            # UE节点风险基线调整：避免过低
            baseline = self.config.ue_risk_baseline
            adjusted_risks = torch.maximum(risks, torch.full_like(risks, baseline))
            self.logger.info(f"   UE节点基线风险: {baseline}")
            
        else:
            # 其他节点类型保持不变
            adjusted_risks = risks
            
        return adjusted_risks
    
    def _apply_connectivity_adjustment(self, graph: dgl.DGLGraph, node_type: str, 
                                     risks: torch.Tensor) -> torch.Tensor:
        """基于连接性的动态风险调整"""
        adjusted_risks = risks.clone()
        num_nodes = len(risks)
        
        if num_nodes == 0:
            return adjusted_risks
        
        # 获取节点的连接信息
        node_degrees = self._get_node_connectivity_info(graph, node_type)
        
        for i in range(num_nodes):
            degree_info = node_degrees[i]
            original_risk = risks[i].item()
            
            # 孤立节点惩罚
            if degree_info['total_degree'] == 0:
                penalty = self.config.isolated_penalty
                adjusted_risks[i] = original_risk * (1 - penalty)
                
            # 传播链增强
            elif degree_info['high_risk_neighbors'] > 0:
                boost = self.config.propagation_boost
                adjusted_risks[i] = min(1.0, original_risk * (1 + boost))
        
        return adjusted_risks
    
    def _get_node_connectivity_info(self, graph: dgl.DGLGraph, node_type: str) -> List[Dict]:
        """获取节点连接信息"""
        num_nodes = graph.number_of_nodes(node_type)
        connectivity_info = []
        
        for i in range(num_nodes):
            info = {
                'total_degree': 0,
                'high_risk_neighbors': 0,
                'in_degree': 0,
                'out_degree': 0
            }
            
            # 计算各种边类型的度数
            for etype in graph.etypes:
                src_type, _, dst_type = graph.to_canonical_etype(etype)
                
                if src_type == node_type:
                    # 出度
                    out_edges = graph.out_edges(i, etype=etype)
                    info['out_degree'] += len(out_edges[0])
                    
                if dst_type == node_type:
                    # 入度
                    in_edges = graph.in_edges(i, etype=etype)
                    info['in_degree'] += len(in_edges[0])
            
            info['total_degree'] = info['in_degree'] + info['out_degree']
            connectivity_info.append(info)
        
        return connectivity_info
    
    def _apply_distribution_normalization(self, risks: torch.Tensor, node_type: str) -> torch.Tensor:
        """应用分布归一化"""
        if len(risks) == 0:
            return risks
        
        # 计算当前分布
        current_dist = self._calculate_risk_distribution(risks)
        target_dist = self.config.target_distribution
        
        # 分位数映射归一化
        sorted_risks, sort_indices = torch.sort(risks)
        normalized_risks = risks.clone()
        
        # 重新分配风险值以匹配目标分布
        n_total = len(risks)
        n_low = int(n_total * target_dist['low_risk'])
        n_medium = int(n_total * target_dist['medium_risk'])
        n_high = n_total - n_low - n_medium
        
        # 低风险区间 [0, 0.3]
        low_values = torch.linspace(0.0, self.config.medium_risk_threshold, n_low)
        # 中风险区间 [0.3, 0.6]
        medium_values = torch.linspace(self.config.medium_risk_threshold, 
                                     self.config.high_risk_threshold, n_medium)
        # 高风险区间 [0.6, 1.0]
        high_values = torch.linspace(self.config.high_risk_threshold, 1.0, n_high)
        
        # 重新分配
        new_values = torch.cat([low_values, medium_values, high_values])
        normalized_risks[sort_indices] = new_values
        
        return normalized_risks
    
    def _analyze_risk_distribution(self, graph: dgl.DGLGraph) -> Dict:
        """分析图的风险分布"""
        distribution = {}
        
        for node_type in graph.ntypes:
            if 'risk_score' in graph.nodes[node_type].data:
                risks = graph.nodes[node_type].data['risk_score']
                distribution[node_type] = self._calculate_risk_distribution(risks)
        
        return distribution
    
    def _calculate_risk_distribution(self, risks: torch.Tensor) -> Dict:
        """计算风险分布统计"""
        if len(risks) == 0:
            return {}
        
        risks_np = risks.detach().cpu().numpy()
        
        high_risk = np.sum(risks_np > self.config.high_risk_threshold)
        medium_risk = np.sum((risks_np >= self.config.medium_risk_threshold) & 
                           (risks_np <= self.config.high_risk_threshold))
        low_risk = np.sum(risks_np < self.config.medium_risk_threshold)
        total = len(risks_np)
        
        return {
            'total_nodes': total,
            'mean': float(np.mean(risks_np)),
            'std': float(np.std(risks_np)),
            'min': float(np.min(risks_np)),
            'max': float(np.max(risks_np)),
            'high_risk_count': int(high_risk),
            'medium_risk_count': int(medium_risk),
            'low_risk_count': int(low_risk),
            'high_risk_ratio': float(high_risk / total),
            'medium_risk_ratio': float(medium_risk / total),
            'low_risk_ratio': float(low_risk / total)
        }
    
    def _generate_adjustment_summary(self):
        """生成调整摘要"""
        original = self.calibration_stats['original_distribution']
        calibrated = self.calibration_stats['calibrated_distribution']
        
        summary = {}
        
        for node_type in original.keys():
            if node_type in calibrated:
                orig_stats = original[node_type]
                calib_stats = calibrated[node_type]
                
                summary[node_type] = {
                    'mean_change': calib_stats['mean'] - orig_stats['mean'],
                    'high_risk_ratio_change': (calib_stats['high_risk_ratio'] - 
                                             orig_stats['high_risk_ratio']),
                    'distribution_improvement': self._calculate_distribution_improvement(
                        orig_stats, calib_stats
                    )
                }
        
        self.calibration_stats['adjustment_summary'] = summary
    
    def _calculate_distribution_improvement(self, original: Dict, calibrated: Dict) -> float:
        """计算分布改善程度"""
        target = self.config.target_distribution
        
        # 计算与目标分布的距离
        orig_distance = (
            abs(original['high_risk_ratio'] - target['high_risk']) +
            abs(original['medium_risk_ratio'] - target['medium_risk']) +
            abs(original['low_risk_ratio'] - target['low_risk'])
        )
        
        calib_distance = (
            abs(calibrated['high_risk_ratio'] - target['high_risk']) +
            abs(calibrated['medium_risk_ratio'] - target['medium_risk']) +
            abs(calibrated['low_risk_ratio'] - target['low_risk'])
        )
        
        # 改善程度 = (原始距离 - 校准距离) / 原始距离
        if orig_distance > 0:
            improvement = (orig_distance - calib_distance) / orig_distance
        else:
            improvement = 0.0
        
        return improvement
    
    def get_calibration_report(self) -> Dict:
        """获取校准报告"""
        return self.calibration_stats
    
    def save_calibration_report(self, output_path: str):
        """保存校准报告"""
        report = self.get_calibration_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📄 校准报告已保存: {output_path}")


def create_risk_recalibrator(config_dict: Dict = None) -> RiskWeightRecalibrator:
    """创建风险重新校准器的工厂函数"""
    if config_dict is None:
        config_dict = {}
    
    config = RiskRecalibrationConfig(
        flow_risk_decay_factor=config_dict.get('flow_risk_decay_factor', 0.7),
        alert_risk_boost_factor=config_dict.get('alert_risk_boost_factor', 1.5),
        ue_risk_baseline=config_dict.get('ue_risk_baseline', 0.1),
        isolated_penalty=config_dict.get('isolated_penalty', 0.3),
        propagation_boost=config_dict.get('propagation_boost', 0.2),
        high_risk_threshold=config_dict.get('high_risk_threshold', 0.6),
        medium_risk_threshold=config_dict.get('medium_risk_threshold', 0.3),
        low_risk_threshold=config_dict.get('low_risk_threshold', 0.1),
        enable_type_wise_normalization=config_dict.get('enable_type_wise_normalization', True)
    )
    
    return RiskWeightRecalibrator(config)
