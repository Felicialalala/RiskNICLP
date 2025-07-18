#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险链路预测专用评估指标
包含Hit@K、NDCG、风险传播准确率等专业指标

Created: 2025-07-13
Author: RiskNICLP-5G Team
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings

logger = logging.getLogger(__name__)

class RiskLinkPredictionMetrics:
    """风险链路预测专用评估指标类"""
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10, 20]):
        """
        初始化评估指标
        
        Args:
            k_values: Hit@K和NDCG@K的K值列表
        """
        self.k_values = k_values
        
    def compute_all_metrics(self, 
                           predictions: torch.Tensor,
                           labels: torch.Tensor,
                           risk_scores: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        计算所有风险链路预测指标
        
        Args:
            predictions: 预测分数 [N]
            labels: 真实标签 [N] (1=攻击链路, 0=正常链路)
            risk_scores: 风险分数 [N] (可选，用于风险加权指标)
            
        Returns:
            metrics: 指标字典
        """
        metrics = {}
        
        # 基础分类指标
        basic_metrics = self._compute_basic_metrics(predictions, labels)
        metrics.update(basic_metrics)
        
        # Hit@K 指标
        hit_metrics = self._compute_hit_at_k(predictions, labels)
        metrics.update(hit_metrics)
        
        # NDCG@K 指标
        ndcg_metrics = self._compute_ndcg_at_k(predictions, labels)
        metrics.update(ndcg_metrics)
        
        # 风险传播指标
        if risk_scores is not None:
            risk_metrics = self._compute_risk_propagation_metrics(predictions, labels, risk_scores)
            metrics.update(risk_metrics)
        
        # 攻击链路发现指标
        discovery_metrics = self._compute_attack_discovery_metrics(predictions, labels)
        metrics.update(discovery_metrics)
        
        return metrics
    
    def _compute_basic_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """计算基础分类指标"""
        predictions_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        metrics = {}
        
        try:
            # AUC指标
            if len(np.unique(labels_np)) > 1:
                metrics['auc_roc'] = roc_auc_score(labels_np, predictions_np)
                metrics['auc_pr'] = average_precision_score(labels_np, predictions_np)
            else:
                metrics['auc_roc'] = 0.5
                metrics['auc_pr'] = labels_np.mean()
            
            # 准确率（使用最优阈值）
            optimal_threshold = self._find_optimal_threshold(predictions_np, labels_np)
            pred_binary = (predictions_np >= optimal_threshold).astype(int)
            
            metrics['accuracy'] = (pred_binary == labels_np).mean()
            metrics['optimal_threshold'] = optimal_threshold
            
            # 精确率、召回率、F1
            tp = ((pred_binary == 1) & (labels_np == 1)).sum()
            fp = ((pred_binary == 1) & (labels_np == 0)).sum()
            fn = ((pred_binary == 0) & (labels_np == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1
            
        except Exception as e:
            logger.warning(f"基础指标计算失败: {e}")
            metrics.update({
                'auc_roc': 0.5, 'auc_pr': 0.5, 'accuracy': 0.0,
                'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0
            })
        
        return metrics
    
    def _compute_hit_at_k(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """计算Hit@K指标 - 衡量Top-K预测中包含真实攻击链路的比例"""
        metrics = {}
        
        # 按预测分数排序
        sorted_indices = torch.argsort(predictions, descending=True)
        sorted_labels = labels[sorted_indices]
        
        total_positives = labels.sum().item()
        if total_positives == 0:
            for k in self.k_values:
                metrics[f'hit_at_{k}'] = 0.0
            return metrics
        
        for k in self.k_values:
            if k > len(predictions):
                metrics[f'hit_at_{k}'] = 1.0 if total_positives > 0 else 0.0
                continue
                
            # Top-K中的正样本数量
            top_k_positives = sorted_labels[:k].sum().item()
            hit_rate = top_k_positives / min(total_positives, k)
            metrics[f'hit_at_{k}'] = hit_rate
        
        return metrics
    
    def _compute_ndcg_at_k(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """计算NDCG@K指标 - 考虑排序质量的归一化折扣累积增益"""
        metrics = {}
        
        # 按预测分数排序
        sorted_indices = torch.argsort(predictions, descending=True)
        sorted_labels = labels[sorted_indices].float()
        
        for k in self.k_values:
            if k > len(predictions):
                k = len(predictions)
            
            # DCG@K
            dcg = self._compute_dcg(sorted_labels[:k])
            
            # IDCG@K (理想情况下的DCG)
            ideal_labels = torch.sort(labels, descending=True)[0].float()
            idcg = self._compute_dcg(ideal_labels[:k])
            
            # NDCG@K
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f'ndcg_at_{k}'] = ndcg
        
        return metrics
    
    def _compute_dcg(self, relevance_scores: torch.Tensor) -> float:
        """计算DCG (Discounted Cumulative Gain)"""
        if len(relevance_scores) == 0:
            return 0.0
        
        # DCG = sum(rel_i / log2(i+1)) for i in [1, k]
        positions = torch.arange(1, len(relevance_scores) + 1, dtype=torch.float)
        discounts = torch.log2(positions + 1)
        dcg = (relevance_scores / discounts).sum().item()
        
        return dcg
    
    def _compute_risk_propagation_metrics(self, 
                                        predictions: torch.Tensor, 
                                        labels: torch.Tensor,
                                        risk_scores: torch.Tensor) -> Dict[str, float]:
        """计算风险传播相关指标"""
        metrics = {}
        
        # 风险加权准确率 - 高风险节点的预测错误惩罚更重
        pred_binary = (predictions > 0.5).float()
        correct_predictions = (pred_binary == labels).float()
        
        # 归一化风险分数
        risk_weights = risk_scores / (risk_scores.sum() + 1e-8)
        weighted_accuracy = (correct_predictions * risk_weights).sum().item()
        metrics['risk_weighted_accuracy'] = weighted_accuracy
        
        # 高风险链路召回率 - 高风险区域的攻击链路发现率
        high_risk_mask = risk_scores > risk_scores.quantile(0.8)  # Top 20%高风险
        if high_risk_mask.sum() > 0:
            high_risk_labels = labels[high_risk_mask]
            high_risk_preds = pred_binary[high_risk_mask]
            
            if high_risk_labels.sum() > 0:
                high_risk_recall = (high_risk_preds * high_risk_labels).sum() / high_risk_labels.sum()
                metrics['high_risk_recall'] = high_risk_recall.item()
            else:
                metrics['high_risk_recall'] = 0.0
        else:
            metrics['high_risk_recall'] = 0.0
        
        # 风险传播一致性 - 预测分数与风险分数的相关性
        correlation = torch.corrcoef(torch.stack([predictions, risk_scores]))[0, 1]
        metrics['risk_prediction_correlation'] = correlation.item() if not torch.isnan(correlation) else 0.0
        
        return metrics
    
    def _compute_attack_discovery_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """计算攻击发现相关指标"""
        metrics = {}
        
        # 攻击链路发现率 - 在不同阈值下的发现率
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        total_attacks = labels.sum().item()
        if total_attacks == 0:
            for thresh in thresholds:
                metrics[f'discovery_rate_at_{thresh}'] = 0.0
            return metrics
        
        for thresh in thresholds:
            discovered = ((predictions >= thresh) & (labels == 1)).sum().item()
            discovery_rate = discovered / total_attacks
            metrics[f'discovery_rate_at_{thresh:.1f}'] = discovery_rate
        
        # 早期发现能力 - Top 10%预测中的攻击发现率
        top_10_percent = int(len(predictions) * 0.1)
        if top_10_percent > 0:
            sorted_indices = torch.argsort(predictions, descending=True)
            top_predictions = labels[sorted_indices[:top_10_percent]]
            early_discovery_rate = top_predictions.sum().item() / total_attacks
            metrics['early_discovery_rate'] = early_discovery_rate
        else:
            metrics['early_discovery_rate'] = 0.0
        
        return metrics
    
    def _find_optimal_threshold(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """找到最优分类阈值（最大化F1分数）"""
        thresholds = np.linspace(0.1, 0.9, 9)
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            pred_binary = (predictions >= thresh).astype(int)
            
            tp = ((pred_binary == 1) & (labels == 1)).sum()
            fp = ((pred_binary == 1) & (labels == 0)).sum()
            fn = ((pred_binary == 0) & (labels == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        return best_threshold
    
    def print_metrics_summary(self, metrics: Dict[str, float]) -> None:
        """打印指标摘要"""
        print("\n" + "="*60)
        print("🎯 风险链路预测评估结果")
        print("="*60)
        
        # 基础指标
        print(f"📊 基础分类指标:")
        print(f"   准确率: {metrics.get('accuracy', 0):.4f}")
        print(f"   精确率: {metrics.get('precision', 0):.4f}")
        print(f"   召回率: {metrics.get('recall', 0):.4f}")
        print(f"   F1分数: {metrics.get('f1_score', 0):.4f}")
        print(f"   AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
        print(f"   AUC-PR: {metrics.get('auc_pr', 0):.4f}")
        
        # Hit@K指标
        print(f"\n🎯 Hit@K指标 (Top-K命中率):")
        for k in self.k_values:
            hit_k = metrics.get(f'hit_at_{k}', 0)
            print(f"   Hit@{k}: {hit_k:.4f}")
        
        # NDCG@K指标
        print(f"\n📈 NDCG@K指标 (排序质量):")
        for k in self.k_values:
            ndcg_k = metrics.get(f'ndcg_at_{k}', 0)
            print(f"   NDCG@{k}: {ndcg_k:.4f}")
        
        # 风险传播指标
        if 'risk_weighted_accuracy' in metrics:
            print(f"\n⚠️ 风险传播指标:")
            print(f"   风险加权准确率: {metrics.get('risk_weighted_accuracy', 0):.4f}")
            print(f"   高风险召回率: {metrics.get('high_risk_recall', 0):.4f}")
            print(f"   风险预测相关性: {metrics.get('risk_prediction_correlation', 0):.4f}")
        
        # 攻击发现指标
        print(f"\n🔍 攻击发现指标:")
        print(f"   早期发现率(Top10%): {metrics.get('early_discovery_rate', 0):.4f}")
        for thresh in [0.3, 0.5, 0.7]:
            discovery = metrics.get(f'discovery_rate_at_{thresh:.1f}', 0)
            print(f"   发现率@{thresh}: {discovery:.4f}")
        
        print("="*60)
