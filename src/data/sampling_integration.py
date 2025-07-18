"""
RiskNICLP-5G 采样重构集成模块
将风险权重重构和语义感知负采样集成到主训练流程

集成了验证成功的优化参数：
- 风险权重重构：解决Flow节点风险虚高问题
- 语义感知负采样：解决负样本风险倒挂问题
- 优化采样比例：从4:1降低到1.2:1
"""

import torch
import dgl
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set, Union
from pathlib import Path
import json

# 导入重构模块
from .risk_weight_recalibration import RiskWeightRecalibrator, RiskRecalibrationConfig
from .semantic_negative_sampler import SemanticNegativeSampler, SemanticSamplingConfig

logger = logging.getLogger(__name__)

class SamplingIntegrator:
    """采样重构集成器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化采样集成器
        
        Args:
            config: 配置字典，来自riskniclp_config.yaml
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.risk_recalibrator = self._init_risk_recalibrator()
        self.semantic_sampler = self._init_semantic_sampler()
        
        self.logger.info("🚀 采样重构集成器初始化完成")
    
    def _init_risk_recalibrator(self) -> RiskWeightRecalibrator:
        """初始化风险权重重构器"""
        recalib_config = self.config.get('risk_recalibration', {})
        
        if not recalib_config.get('enabled', True):
            self.logger.info("⚠️ 风险权重重构已禁用")
            return None
        
        recalib_params = {
            'flow_risk_decay_factor': recalib_config.get('flow_decay_factor', 0.5),
            'alert_risk_boost_factor': recalib_config.get('alert_boost_factor', 2.0),
            'ue_risk_baseline': recalib_config.get('ue_baseline', 0.2),
            'enable_type_wise_normalization': recalib_config.get('enable_type_wise_normalization', True)
        }
        
        self.logger.info(f"✅ 初始化风险权重重构器: {recalib_params}")
        return RiskWeightRecalibrator(RiskRecalibrationConfig(**recalib_params))
    
    def _init_semantic_sampler(self) -> SemanticNegativeSampler:
        """初始化语义感知负采样器"""
        # 提取难度分布
        difficulty_dist = self.config.get('difficulty_distribution', {})
        if not difficulty_dist:
            difficulty_dist = {
                'easy': 0.7,
                'medium': 0.25,
                'hard': 0.05
            }
        
        # 提取语义约束配置
        semantic_constraints = self.config.get('semantic_constraints', {})
        
        # 构建采样配置
        sampling_params = {
            'neg_ratio': self.config.get('neg_ratio', 1.2),
            'seed': self.config.get('seed', 42),
            'difficulty_distribution': difficulty_dist,
            'temporal_constraint_enabled': semantic_constraints.get('temporal_constraint_enabled', True),
            'spatial_constraint_enabled': semantic_constraints.get('spatial_constraint_enabled', True),
            'min_temporal_gap': semantic_constraints.get('min_temporal_gap', 5.0),
            'max_attempts_multiplier': self.config.get('max_attempts_multiplier', 15),
            'quality_threshold': self.config.get('quality_threshold', 0.8)
        }
        
        self.logger.info(f"✅ 初始化语义感知负采样器: {sampling_params}")
        return SemanticNegativeSampler(SemanticSamplingConfig(**sampling_params))
    
    def process_graph(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        处理图：应用风险权重重构
        
        Args:
            graph: 输入的异质图
            
        Returns:
            processed_graph: 处理后的图
        """
        if self.risk_recalibrator is None:
            return graph
        
        self.logger.info("🔧 应用风险权重重构...")
        processed_graph = self.risk_recalibrator.recalibrate_graph_risks(graph)
        
        return processed_graph
    
    def sample_negatives(self, graph: dgl.DGLGraph, positive_edges: torch.Tensor,
                        risk_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        负采样接口：应用语义感知负采样
        
        Args:
            graph: DGL异质图
            positive_edges: 正样本边 [num_pos, 2]
            risk_scores: 节点风险分数（可选）
            
        Returns:
            negative_edges: 负样本边 [num_neg, 2]
        """
        self.logger.info("🎯 应用语义感知负采样...")
        
        # 先处理图（如果需要）
        processed_graph = self.process_graph(graph)
        
        # 应用语义感知负采样
        negative_edges = self.semantic_sampler.sample_negatives(
            processed_graph, positive_edges, risk_scores
        )
        
        self.logger.info(f"✅ 负采样完成: {len(negative_edges)}个负样本")
        return negative_edges
    
    def get_sampling_stats(self) -> Dict:
        """获取采样统计"""
        stats = {}
        
        if self.risk_recalibrator:
            stats['risk_calibration'] = self.risk_recalibrator.get_calibration_report()
        
        if self.semantic_sampler:
            stats['semantic_sampling'] = self.semantic_sampler.get_sampling_stats()
        
        return stats


def create_sampling_integrator(config: Dict = None) -> SamplingIntegrator:
    """创建采样集成器的工厂函数"""
    # 提取负采样配置
    if config is None:
        config = {}
    
    sampling_config = config.get('phase2', {}).get('negative_sampling', {})
    
    return SamplingIntegrator(sampling_config)


# 兼容性函数：用于替换原有的负采样器创建
def create_negative_sampler_from_config(config: Dict) -> Union[SemanticNegativeSampler, object]:
    """
    从配置创建负采样器（兼容性函数）

    Args:
        config: 采样器配置

    Returns:
        sampler: 负采样器实例
    """
    sampler_type = config.get('type', 'semantic_aware')

    if sampler_type == 'semantic_aware':
        # 使用新的语义感知采样器
        # 需要将配置包装成正确的格式
        wrapped_config = {'phase2': {'negative_sampling': config}}
        integrator = create_sampling_integrator(wrapped_config)
        return integrator.semantic_sampler
    else:
        # 兼容旧的采样器类型
        from .negative_sampler import create_negative_sampler
        return create_negative_sampler(config)
