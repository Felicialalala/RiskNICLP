"""
RiskNICLP-5G 正样本采样优化器
专门针对攻击链路预测任务的正样本比例优化

从异常检测(3-5%正样本)转换为攻击预测(25-35%正样本)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# 攻击链路预测的正样本比例配置
ATTACK_PROPAGATION_RATIOS = {
    'triggers': 0.35,      # Flow->Alert: 35% 攻击告警触发预测
    'initiates': 0.30,     # UE->Flow: 30% 攻击发起预测  
    'temporal_adj': 0.25,  # Flow->Flow: 25% 攻击传播预测
    'default': 0.20        # 其他边类型: 20% 基础攻击关联
}

# 异常检测的正样本比例配置（用于对比）
ANOMALY_DETECTION_RATIOS = {
    'triggers': 0.05,      # Flow->Alert: 5% 异常告警
    'initiates': 0.03,     # UE->Flow: 3% 异常发起
    'temporal_adj': 0.02,  # Flow->Flow: 2% 异常传播
    'default': 0.01        # 其他边类型: 1% 异常边
}

def optimize_positive_sampling(edge_type: str, 
                              task_type: str = 'attack_propagation',
                              attack_density: Optional[float] = None) -> float:
    """
    优化正样本采样比例
    
    Args:
        edge_type: 边类型 ('triggers', 'initiates', 'temporal_adj', 'default')
        task_type: 任务类型 ('attack_propagation' 或 'anomaly_detection')
        attack_density: 攻击密度 (可选，用于动态调整)
        
    Returns:
        优化后的正样本比例
    """
    
    if task_type == 'attack_propagation':
        base_ratio = ATTACK_PROPAGATION_RATIOS.get(edge_type, ATTACK_PROPAGATION_RATIOS['default'])
        
        # 基于攻击密度动态调整
        if attack_density is not None:
            if attack_density > 0.6:  # 高攻击密度环境 (如5G-NIDD)
                adjustment_factor = 1.0 + (attack_density - 0.6) * 0.5
                base_ratio = min(base_ratio * adjustment_factor, 0.45)  # 最高不超过45%
            elif attack_density < 0.3:  # 低攻击密度环境
                adjustment_factor = 0.7 + attack_density
                base_ratio = base_ratio * adjustment_factor
                
        logger.info(f"攻击预测模式 - {edge_type}: {base_ratio:.3f} (密度调整: {attack_density})")
        
    elif task_type == 'anomaly_detection':
        base_ratio = ANOMALY_DETECTION_RATIOS.get(edge_type, ANOMALY_DETECTION_RATIOS['default'])
        logger.info(f"异常检测模式 - {edge_type}: {base_ratio:.3f}")
        
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")
    
    return base_ratio

def get_edge_type_ratios(task_type: str = 'attack_propagation') -> Dict[str, float]:
    """
    获取所有边类型的正样本比例配置
    
    Args:
        task_type: 任务类型
        
    Returns:
        边类型到正样本比例的映射
    """
    if task_type == 'attack_propagation':
        return ATTACK_PROPAGATION_RATIOS.copy()
    elif task_type == 'anomaly_detection':
        return ANOMALY_DETECTION_RATIOS.copy()
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")

def calculate_expected_positive_samples(graph_stats: Dict[str, int], 
                                      task_type: str = 'attack_propagation') -> Dict[str, int]:
    """
    计算预期的正样本数量
    
    Args:
        graph_stats: 图统计信息 {'triggers': 边数, 'initiates': 边数, ...}
        task_type: 任务类型
        
    Returns:
        预期正样本数量
    """
    ratios = get_edge_type_ratios(task_type)
    expected_samples = {}
    
    for edge_type, num_edges in graph_stats.items():
        ratio = ratios.get(edge_type, ratios['default'])
        expected_samples[edge_type] = int(num_edges * ratio)
    
    return expected_samples

def validate_positive_ratio_consistency(config_ratios: Dict[str, float],
                                      task_type: str = 'attack_propagation') -> bool:
    """
    验证配置的正样本比例是否与任务类型一致
    
    Args:
        config_ratios: 配置中的正样本比例
        task_type: 任务类型
        
    Returns:
        是否一致
    """
    expected_ratios = get_edge_type_ratios(task_type)
    
    for edge_type, expected_ratio in expected_ratios.items():
        config_ratio = config_ratios.get(edge_type, 0.0)
        
        # 允许10%的误差范围
        if abs(config_ratio - expected_ratio) > expected_ratio * 0.1:
            logger.warning(
                f"正样本比例不一致 - {edge_type}: "
                f"配置={config_ratio:.3f}, 期望={expected_ratio:.3f}"
            )
            return False
    
    return True

def get_sampling_strategy_description(task_type: str = 'attack_propagation') -> str:
    """
    获取采样策略的描述
    
    Args:
        task_type: 任务类型
        
    Returns:
        策略描述
    """
    if task_type == 'attack_propagation':
        return """
攻击链路预测采样策略:
- 基于5G-NIDD数据集60.8%攻击流量的特性
- 正样本比例25-35%，反映攻击密集环境
- 重点预测攻击发起、告警触发、攻击传播三类边
- 评估指标关注预警效果而非异常检测准确率
        """.strip()
    elif task_type == 'anomaly_detection':
        return """
异常检测采样策略:
- 基于正常网络环境的假设
- 正样本比例1-5%，反映异常稀少性
- 重点识别偏离正常模式的边
- 评估指标关注AUC/AUPR等异常检测指标
        """.strip()
    else:
        return "未知任务类型"

class PositiveSamplingOptimizer:
    """正样本采样优化器类"""
    
    def __init__(self, task_type: str = 'attack_propagation'):
        """
        初始化优化器
        
        Args:
            task_type: 任务类型
        """
        self.task_type = task_type
        self.ratios = get_edge_type_ratios(task_type)
        
        logger.info(f"正样本采样优化器初始化 - 任务类型: {task_type}")
        logger.info(f"采样比例配置: {self.ratios}")
    
    def get_ratio(self, edge_type: str, attack_density: Optional[float] = None) -> float:
        """获取指定边类型的正样本比例"""
        return optimize_positive_sampling(edge_type, self.task_type, attack_density)
    
    def update_task_type(self, new_task_type: str):
        """更新任务类型"""
        self.task_type = new_task_type
        self.ratios = get_edge_type_ratios(new_task_type)
        logger.info(f"任务类型已更新为: {new_task_type}")
    
    def get_all_ratios(self) -> Dict[str, float]:
        """获取所有边类型的正样本比例"""
        return self.ratios.copy()
    
    def validate_config(self, config_ratios: Dict[str, float]) -> bool:
        """验证配置的正样本比例"""
        return validate_positive_ratio_consistency(config_ratios, self.task_type)

# 全局优化器实例
_global_optimizer = PositiveSamplingOptimizer('attack_propagation')

def set_global_task_type(task_type: str):
    """设置全局任务类型"""
    global _global_optimizer
    _global_optimizer.update_task_type(task_type)

def get_global_optimizer() -> PositiveSamplingOptimizer:
    """获取全局优化器实例"""
    return _global_optimizer
