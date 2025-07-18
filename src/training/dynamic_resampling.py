"""
动态重采样机制
根据模型训练反馈动态调整负采样策略
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class ResamplingMetrics:
    """重采样指标"""
    epoch: int
    auc: float
    negative_sample_quality: float  # 负样本预测分数的平均值（应该接近0）
    positive_sample_quality: float  # 正样本预测分数的平均值（应该接近1）
    separation_score: float  # 正负样本分离度
    risk_bias_activation: float  # 风险偏置参数的变化程度

@dataclass
class ResamplingStrategy:
    """重采样策略"""
    neg_ratio: int
    hard_ratio: float
    risk_thresholds: List[float]
    quality_threshold: float

class DynamicResamplingManager:
    """动态重采样管理器"""
    
    def __init__(
        self,
        initial_strategy: ResamplingStrategy,
        adaptation_patience: int = 3,
        min_improvement: float = 0.01,
        max_neg_ratio: int = 8,
        min_neg_ratio: int = 2
    ):
        """
        初始化动态重采样管理器
        
        Args:
            initial_strategy: 初始采样策略
            adaptation_patience: 适应耐心（多少个epoch无改善后调整）
            min_improvement: 最小改善阈值
            max_neg_ratio: 最大负采样比例
            min_neg_ratio: 最小负采样比例
        """
        self.current_strategy = initial_strategy
        self.adaptation_patience = adaptation_patience
        self.min_improvement = min_improvement
        self.max_neg_ratio = max_neg_ratio
        self.min_neg_ratio = min_neg_ratio
        
        # 历史记录
        self.metrics_history: List[ResamplingMetrics] = []
        self.strategy_history: List[ResamplingStrategy] = [initial_strategy]
        
        # 适应状态
        self.epochs_without_improvement = 0
        self.best_auc = 0.0
        self.adaptation_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    def should_adapt(self, current_metrics: ResamplingMetrics) -> bool:
        """判断是否需要调整采样策略"""
        
        # 记录当前指标
        self.metrics_history.append(current_metrics)
        
        # 检查是否有改善
        if current_metrics.auc > self.best_auc + self.min_improvement:
            self.best_auc = current_metrics.auc
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
        
        # 检查是否需要适应
        if self.epochs_without_improvement >= self.adaptation_patience:
            return True
        
        # 检查负样本质量是否过差
        if current_metrics.negative_sample_quality > 0.7:  # 负样本预测分数过高
            self.logger.warning(f"负样本质量过差: {current_metrics.negative_sample_quality:.3f}")
            return True
        
        # 检查正负样本分离度
        if current_metrics.separation_score < 0.1:  # 分离度过低
            self.logger.warning(f"正负样本分离度过低: {current_metrics.separation_score:.3f}")
            return True
        
        return False
    
    def adapt_strategy(self, current_metrics: ResamplingMetrics) -> ResamplingStrategy:
        """根据当前指标调整采样策略"""
        
        new_strategy = ResamplingStrategy(
            neg_ratio=self.current_strategy.neg_ratio,
            hard_ratio=self.current_strategy.hard_ratio,
            risk_thresholds=self.current_strategy.risk_thresholds.copy(),
            quality_threshold=self.current_strategy.quality_threshold
        )
        
        # 策略1: 负样本质量过差 -> 增加负采样比例，降低难度
        if current_metrics.negative_sample_quality > 0.7:
            if new_strategy.neg_ratio < self.max_neg_ratio:
                new_strategy.neg_ratio += 1
                new_strategy.hard_ratio = max(0.3, new_strategy.hard_ratio - 0.1)
                self.logger.info(f"负样本质量过差，增加负采样比例: {new_strategy.neg_ratio}")
        
        # 策略2: AUC过低且分离度差 -> 调整风险阈值
        elif current_metrics.auc < 0.6 and current_metrics.separation_score < 0.2:
            # 降低高风险阈值，增加中等风险样本
            new_strategy.risk_thresholds[0] = max(0.8, new_strategy.risk_thresholds[0] - 0.05)
            new_strategy.risk_thresholds[1] = max(0.5, new_strategy.risk_thresholds[1] - 0.05)
            self.logger.info(f"调整风险阈值: {new_strategy.risk_thresholds}")
        
        # 策略3: 风险偏置未激活 -> 降低质量阈值，增加困难样本
        elif current_metrics.risk_bias_activation < 0.1:
            new_strategy.quality_threshold = max(0.5, new_strategy.quality_threshold - 0.1)
            new_strategy.hard_ratio = min(0.8, new_strategy.hard_ratio + 0.1)
            self.logger.info(f"风险偏置未激活，增加困难样本比例: {new_strategy.hard_ratio}")
        
        # 策略4: 模型过拟合 -> 增加负样本多样性
        elif len(self.metrics_history) >= 3:
            recent_aucs = [m.auc for m in self.metrics_history[-3:]]
            if max(recent_aucs) - min(recent_aucs) < 0.01:  # AUC变化很小
                if new_strategy.neg_ratio > self.min_neg_ratio:
                    new_strategy.neg_ratio -= 1
                    new_strategy.quality_threshold += 0.1
                    self.logger.info(f"可能过拟合，减少负采样比例增加质量: {new_strategy.neg_ratio}")
        
        # 记录策略变化
        if self._strategy_changed(new_strategy):
            self.strategy_history.append(new_strategy)
            self.current_strategy = new_strategy
            self.adaptation_count += 1
            self.epochs_without_improvement = 0
            
            self.logger.info(f"🔄 采样策略已调整 (第{self.adaptation_count}次)")
            self.logger.info(f"   负采样比例: {new_strategy.neg_ratio}:1")
            self.logger.info(f"   困难样本比例: {new_strategy.hard_ratio:.2f}")
            self.logger.info(f"   风险阈值: {new_strategy.risk_thresholds}")
        
        return new_strategy
    
    def _strategy_changed(self, new_strategy: ResamplingStrategy) -> bool:
        """检查策略是否发生变化"""
        current = self.current_strategy
        return (
            new_strategy.neg_ratio != current.neg_ratio or
            abs(new_strategy.hard_ratio - current.hard_ratio) > 0.05 or
            any(abs(a - b) > 0.01 for a, b in zip(new_strategy.risk_thresholds, current.risk_thresholds)) or
            abs(new_strategy.quality_threshold - current.quality_threshold) > 0.05
        )
    
    def get_current_strategy(self) -> ResamplingStrategy:
        """获取当前采样策略"""
        return self.current_strategy
    
    def get_adaptation_summary(self) -> Dict:
        """获取适应过程摘要"""
        if not self.metrics_history:
            return {}
        
        return {
            'total_adaptations': self.adaptation_count,
            'best_auc': self.best_auc,
            'current_auc': self.metrics_history[-1].auc if self.metrics_history else 0.0,
            'auc_improvement': self.metrics_history[-1].auc - self.metrics_history[0].auc if len(self.metrics_history) > 1 else 0.0,
            'current_neg_ratio': self.current_strategy.neg_ratio,
            'current_hard_ratio': self.current_strategy.hard_ratio,
            'epochs_trained': len(self.metrics_history)
        }

def create_resampling_manager(config: Dict) -> DynamicResamplingManager:
    """创建动态重采样管理器"""
    
    # 从配置中提取初始策略
    neg_config = config.get('phase2', {}).get('negative_sampling', {})
    
    initial_strategy = ResamplingStrategy(
        neg_ratio=neg_config.get('neg_ratio', 4),
        hard_ratio=neg_config.get('hard_ratio', 0.6),
        risk_thresholds=neg_config.get('risk_thresholds', [0.9, 0.6, 0.3]),
        quality_threshold=neg_config.get('quality_threshold', 0.7)
    )
    
    return DynamicResamplingManager(
        initial_strategy=initial_strategy,
        adaptation_patience=config.get('adaptation_patience', 3),
        min_improvement=config.get('min_improvement', 0.01),
        max_neg_ratio=config.get('max_neg_ratio', 8),
        min_neg_ratio=config.get('min_neg_ratio', 2)
    )
