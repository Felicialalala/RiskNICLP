"""
Attack Propagation Prediction Metrics for RiskNICLP-5G

Provides specialized metrics for attack propagation prediction tasks
Replaces anomaly detection metrics (AUC) with security-oriented evaluation
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import time


class RiskAwareMetrics:
    """Risk-aware evaluation metrics calculator"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset metric accumulators"""
        self.predictions = []
        self.targets = []
        self.risk_weights = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray, risk_weights: np.ndarray):
        """Update metrics with new batch"""
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())
        self.risk_weights.extend(risk_weights.flatten())
    
    def compute_risk_weighted_precision(self) -> float:
        """Compute risk-weighted precision"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        weights = np.array(self.risk_weights)
        
        positive_mask = predictions > 0.5
        if not np.any(positive_mask):
            return 0.0
        
        # Fix boolean operation issue
        true_positive_mask = (targets == 1) & positive_mask
        false_positive_mask = (targets == 0) & positive_mask
        
        weighted_tp = np.sum(true_positive_mask * weights)
        weighted_fp = np.sum(false_positive_mask * weights) 
        
        return weighted_tp / (weighted_tp + weighted_fp + 1e-6)
    
    def compute_risk_weighted_recall(self) -> float:
        """Compute risk-weighted recall"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        weights = np.array(self.risk_weights)
        
        positive_targets = targets == 1
        if not np.any(positive_targets):
            return 0.0
        
        positive_mask = predictions > 0.5
        
        # Fix boolean operation issue
        true_positive_mask = positive_targets & positive_mask
        false_negative_mask = positive_targets & ~positive_mask
        
        weighted_tp = np.sum(true_positive_mask * weights)
        weighted_fn = np.sum(false_negative_mask * weights)
        
        return weighted_tp / (weighted_tp + weighted_fn + 1e-6)


class DirectionalMetrics:
    """Directional prediction metrics calculator"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset metric accumulators"""
        self.predictions = []
        self.directions = []
    
    def update(self, predictions: np.ndarray, directions: np.ndarray):
        """Update metrics with new batch"""
        self.predictions.extend(predictions.flatten())
        self.directions.extend(directions.flatten())
    
    def compute_directional_hit_k(self, k: int = 100) -> float:
        """Compute directional Hit@K metric"""
        predictions = np.array(self.predictions)
        directions = np.array(self.directions)
        
        if len(predictions) < k:
            k = len(predictions)
        
        if k == 0:
            return 0.0
        
        # Get top-k predictions
        top_k_indices = np.argsort(predictions)[-k:]
        top_k_directions = directions[top_k_indices]
        
        # Count correct directions (assuming 1 = correct, 0 = incorrect)
        correct_directions = np.sum(top_k_directions)
        
        return correct_directions / k
    
    def compute_direction_error_rate(self) -> float:
        """Compute overall direction error rate"""
        directions = np.array(self.directions)
        
        if len(directions) == 0:
            return 1.0
        
        wrong_directions = np.sum(directions == 0)  # Assuming 0 = wrong direction
        
        return wrong_directions / len(directions)


class TemporalMetrics:
    """Temporal evaluation metrics for lead-time analysis"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset metric accumulators"""
        self.predictions = []
        self.targets = []
        self.timestamps = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray, timestamps: np.ndarray):
        """Update metrics with new batch"""
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())
        self.timestamps.extend(timestamps.flatten())
    
    def compute_lead_time_stats(self) -> Dict[str, float]:
        """Compute lead time statistics"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        timestamps = np.array(self.timestamps)
        
        lead_times = self._calculate_lead_times(predictions, targets, timestamps)
        
        if len(lead_times) == 0:
            return {'mean': 0.0, 'std': 0.0, 'median': 0.0}
        
        return {
            'mean': float(np.mean(lead_times)),
            'std': float(np.std(lead_times)),
            'median': float(np.median(lead_times))
        }
    
    def compute_mttd(self, threshold_seconds: int = 30) -> float:
        """Compute Mean Time To Detection"""
        predictions = np.array(self.predictions)
        timestamps = np.array(self.timestamps)
        
        # High confidence predictions
        high_confidence_mask = predictions > 0.8
        
        if not np.any(high_confidence_mask):
            return float(threshold_seconds)  # Default to max acceptable
        
        detection_times = timestamps[high_confidence_mask]
        valid_detections = detection_times[detection_times <= threshold_seconds]
        
        if len(valid_detections) == 0:
            return float(threshold_seconds)
        
        return float(np.mean(valid_detections))
    
    def _calculate_lead_times(self, predictions: np.ndarray, targets: np.ndarray, 
                            timestamps: np.ndarray) -> np.ndarray:
        """Calculate lead times between predictions and actual events"""
        lead_times = []
        
        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        sorted_preds = predictions[sorted_indices]
        sorted_targets = targets[sorted_indices]
        sorted_times = timestamps[sorted_indices]
        
        # Find prediction-attack pairs
        for i in range(len(sorted_preds) - 1):
            if sorted_preds[i] > 0.5:  # Positive prediction
                # Look for actual attack in next few timestamps
                for j in range(i + 1, min(i + 10, len(sorted_targets))):
                    if sorted_targets[j] == 1:  # Actual attack
                        lead_time = sorted_times[j] - sorted_times[i]
                        if lead_time > 0:
                            lead_times.append(lead_time)
                        break
        
        return np.array(lead_times)


class SecurityMetrics:
    """Security-specific evaluation metrics"""
    
    @staticmethod
    def compute_attack_detection_rate(predictions: np.ndarray, targets: np.ndarray, 
                                    attack_types: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute attack detection rate by attack type"""
        results = {}
        
        # Overall detection rate
        positive_mask = predictions > 0.5
        attack_mask = targets == 1
        
        if np.any(attack_mask):
            detected_attacks = np.sum(positive_mask & attack_mask)
            total_attacks = np.sum(attack_mask)
            results['overall'] = detected_attacks / total_attacks
        else:
            results['overall'] = 0.0
        
        # Per-attack-type detection rates
        if attack_types is not None:
            unique_types = np.unique(attack_types)
            for attack_type in unique_types:
                type_mask = attack_types == attack_type
                type_attack_mask = attack_mask & type_mask
                
                if np.any(type_attack_mask):
                    type_detected = np.sum(positive_mask & type_attack_mask)
                    type_total = np.sum(type_attack_mask)
                    results[f'type_{attack_type}'] = type_detected / type_total
                else:
                    results[f'type_{attack_type}'] = 0.0
        
        return results
    
    @staticmethod
    def compute_false_positive_rate(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute false positive rate"""
        positive_mask = predictions > 0.5
        negative_mask = targets == 0
        
        if not np.any(negative_mask):
            return 0.0
        
        false_positives = np.sum(positive_mask & negative_mask)
        total_negatives = np.sum(negative_mask)
        
        return false_positives / total_negatives


class AttackPropagationMetrics:
    """
    攻击链路预测专用评估指标
    替代传统异常检测的AUC指标，专注于攻击预测的实际效果
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置指标累积器"""
        self.predictions = []
        self.targets = []
        self.edge_types = []
        self.timestamps = []
        self.attack_types = []
        self.risk_scores = []

    def update(self, predictions: np.ndarray, targets: np.ndarray,
               edge_types: Optional[np.ndarray] = None,
               timestamps: Optional[np.ndarray] = None,
               attack_types: Optional[np.ndarray] = None,
               risk_scores: Optional[np.ndarray] = None):
        """更新指标数据"""
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())

        if edge_types is not None:
            self.edge_types.extend(edge_types.flatten())
        if timestamps is not None:
            self.timestamps.extend(timestamps.flatten())
        if attack_types is not None:
            self.attack_types.extend(attack_types.flatten())
        if risk_scores is not None:
            self.risk_scores.extend(risk_scores.flatten())

    def compute_attack_prediction_accuracy(self, threshold: float = 0.5) -> float:
        """
        计算攻击边预测准确率
        主要指标：正确预测攻击边的比例
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        predicted_positive = predictions > threshold
        actual_positive = targets == 1

        # 计算准确率
        correct_predictions = (predicted_positive == actual_positive)
        return np.mean(correct_predictions)

    def compute_early_warning_effectiveness(self, warning_threshold: float = 0.7,
                                          min_lead_time: float = 30.0) -> Dict[str, float]:
        """
        计算提前预警有效性
        关键指标：成功提前预警的攻击比例
        """
        if len(self.timestamps) == 0:
            return {'effectiveness': 0.0, 'avg_lead_time': 0.0, 'warning_count': 0}

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        timestamps = np.array(self.timestamps)

        # 高置信度预警
        warning_mask = predictions > warning_threshold
        attack_mask = targets == 1

        successful_warnings = 0
        total_attacks = np.sum(attack_mask)
        lead_times = []

        if total_attacks == 0:
            return {'effectiveness': 0.0, 'avg_lead_time': 0.0, 'warning_count': 0}

        # 计算每个攻击的预警效果
        attack_indices = np.where(attack_mask)[0]
        warning_indices = np.where(warning_mask)[0]

        for attack_idx in attack_indices:
            attack_time = timestamps[attack_idx]

            # 查找该攻击之前的预警
            prior_warnings = warning_indices[timestamps[warning_indices] < attack_time]

            if len(prior_warnings) > 0:
                # 找到最近的预警
                latest_warning_idx = prior_warnings[np.argmax(timestamps[prior_warnings])]
                lead_time = attack_time - timestamps[latest_warning_idx]

                if lead_time >= min_lead_time:
                    successful_warnings += 1
                    lead_times.append(lead_time)

        effectiveness = successful_warnings / total_attacks
        avg_lead_time = np.mean(lead_times) if lead_times else 0.0

        return {
            'effectiveness': effectiveness,
            'avg_lead_time': avg_lead_time,
            'warning_count': len(warning_indices),
            'successful_warnings': successful_warnings,
            'total_attacks': total_attacks
        }

    def compute_attack_chain_completeness(self) -> Dict[str, float]:
        """
        计算攻击链完整性
        衡量完整预测攻击传播路径的能力
        """
        if len(self.edge_types) == 0:
            return {'completeness': 0.0, 'chain_coverage': 0.0}

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        edge_types = np.array(self.edge_types)

        # 按边类型分组计算
        edge_type_performance = {}
        for edge_type in ['initiates', 'triggers', 'temporal_adj']:
            type_mask = edge_types == edge_type
            if np.any(type_mask):
                type_predictions = predictions[type_mask]
                type_targets = targets[type_mask]

                # 计算该边类型的召回率
                type_recall = self._compute_recall(type_predictions, type_targets)
                edge_type_performance[edge_type] = type_recall

        # 攻击链完整性 = 所有边类型召回率的几何平均
        if edge_type_performance:
            recalls = list(edge_type_performance.values())
            completeness = np.prod(recalls) ** (1.0 / len(recalls))
        else:
            completeness = 0.0

        # 链路覆盖率 = 至少预测到一种边类型的攻击比例
        chain_coverage = max(edge_type_performance.values()) if edge_type_performance else 0.0

        return {
            'completeness': completeness,
            'chain_coverage': chain_coverage,
            'edge_type_performance': edge_type_performance
        }

    def compute_threat_escalation_precision(self, escalation_threshold: float = 0.2) -> float:
        """
        计算威胁升级预测精度
        衡量准确预测攻击强度升级的能力
        """
        if len(self.risk_scores) == 0:
            return 0.0

        predictions = np.array(self.predictions)
        risk_scores = np.array(self.risk_scores)

        # 识别威胁升级：风险分数显著增加
        risk_increases = np.diff(risk_scores)
        escalation_mask = risk_increases > escalation_threshold

        if not np.any(escalation_mask):
            return 0.0

        # 计算预测到威胁升级的精度
        escalation_predictions = predictions[1:][escalation_mask]
        correct_escalation_predictions = np.sum(escalation_predictions > 0.5)

        return correct_escalation_predictions / len(escalation_predictions)

    def compute_false_alarm_rate(self, threshold: float = 0.5) -> float:
        """
        计算误报率 (SOC关键指标)
        错误攻击预警的比例
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        predicted_positive = predictions > threshold
        actual_negative = targets == 0

        if not np.any(actual_negative):
            return 0.0

        false_alarms = np.sum(predicted_positive & actual_negative)
        total_negatives = np.sum(actual_negative)

        return false_alarms / total_negatives

    def compute_attack_type_discrimination(self) -> Dict[str, float]:
        """
        计算攻击类型区分能力
        不同攻击类型的预测准确率
        """
        if len(self.attack_types) == 0:
            return {}

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        attack_types = np.array(self.attack_types)

        type_performance = {}
        unique_types = np.unique(attack_types)

        for attack_type in unique_types:
            type_mask = attack_types == attack_type
            if np.any(type_mask):
                type_predictions = predictions[type_mask]
                type_targets = targets[type_mask]

                # 计算F1分数
                f1_score = self._compute_f1_score(type_predictions, type_targets)
                type_performance[str(attack_type)] = f1_score

        return type_performance

    def compute_temporal_prediction_stability(self, window_size: int = 10) -> float:
        """
        计算时序预测稳定性
        连续时间窗口预测结果的一致性
        """
        if len(self.timestamps) < window_size * 2:
            return 0.0

        predictions = np.array(self.predictions)
        timestamps = np.array(self.timestamps)

        # 按时间排序
        sorted_indices = np.argsort(timestamps)
        sorted_predictions = predictions[sorted_indices]

        # 计算滑动窗口的预测一致性
        correlations = []
        for i in range(len(sorted_predictions) - window_size):
            window1 = sorted_predictions[i:i+window_size]
            window2 = sorted_predictions[i+1:i+1+window_size]

            # 计算相关性
            correlation = np.corrcoef(window1, window2)[0, 1]
            if not np.isnan(correlation):
                correlations.append(correlation)

        return np.mean(correlations) if correlations else 0.0

    def _compute_recall(self, predictions: np.ndarray, targets: np.ndarray,
                       threshold: float = 0.5) -> float:
        """计算召回率"""
        predicted_positive = predictions > threshold
        actual_positive = targets == 1

        if not np.any(actual_positive):
            return 0.0

        true_positives = np.sum(predicted_positive & actual_positive)
        return true_positives / np.sum(actual_positive)

    def _compute_f1_score(self, predictions: np.ndarray, targets: np.ndarray,
                         threshold: float = 0.5) -> float:
        """计算F1分数"""
        predicted_positive = predictions > threshold
        actual_positive = targets == 1

        if not np.any(actual_positive) and not np.any(predicted_positive):
            return 1.0
        if not np.any(actual_positive) or not np.any(predicted_positive):
            return 0.0

        true_positives = np.sum(predicted_positive & actual_positive)
        false_positives = np.sum(predicted_positive & ~actual_positive)
        false_negatives = np.sum(~predicted_positive & actual_positive)

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)

        return 2 * precision * recall / (precision + recall + 1e-8)

    def compute_comprehensive_report(self) -> Dict[str, Union[float, Dict]]:
        """
        计算综合评估报告
        包含所有攻击预测专用指标
        """
        report = {
            # 主要指标
            'attack_prediction_accuracy': self.compute_attack_prediction_accuracy(),
            'early_warning_effectiveness': self.compute_early_warning_effectiveness(),
            'attack_chain_completeness': self.compute_attack_chain_completeness(),
            'threat_escalation_precision': self.compute_threat_escalation_precision(),

            # 运营指标
            'false_alarm_rate': self.compute_false_alarm_rate(),
            'attack_type_discrimination': self.compute_attack_type_discrimination(),
            'temporal_prediction_stability': self.compute_temporal_prediction_stability(),

            # 统计信息
            'total_samples': len(self.predictions),
            'positive_ratio': np.mean(self.targets) if self.targets else 0.0,
            'prediction_confidence': np.mean(self.predictions) if self.predictions else 0.0
        }

        return report