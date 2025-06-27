"""
Temporal Evaluation Metrics for RiskNICLP-5G

Implements temporal-specific evaluation metrics:
- TemporalEvaluator: 时序预警评估器
- Lead-Time计算
- 风险加权指标
- 提前预警能力评估
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from ..common.logger import get_logger


class TemporalEvaluator:
    """时序预警评估器"""
    
    def __init__(self, lead_time_thresholds: List[int] = [5, 10, 15]):
        """
        Args:
            lead_time_thresholds: Lead-Time阈值（秒）
        """
        self.lead_time_thresholds = lead_time_thresholds
        self.logger = get_logger()
        
    def evaluate_lead_time(
        self, 
        predictions: torch.Tensor, 
        true_events: torch.Tensor, 
        timestamps: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Lead-Time指标计算
        
        Args:
            predictions: 预测分数 [num_samples, future_steps]
            true_events: 真实事件时间 [num_samples]
            timestamps: 预测时间戳 [num_samples]
            threshold: 预测阈值
            
        Returns:
            Lead-Time指标字典
        """
        results = {}
        
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(true_events, torch.Tensor):
            true_events = true_events.cpu().numpy()
        if isinstance(timestamps, torch.Tensor):
            timestamps = timestamps.cpu().numpy()
            
        # 二值化预测
        binary_predictions = (predictions > threshold).astype(int)
        
        total_samples = len(predictions)
        lead_times = []
        
        for i in range(total_samples):
            pred_steps = binary_predictions[i]  # [future_steps]
            true_event_time = true_events[i]
            current_time = timestamps[i]
            
            # 找到第一个预测为正的时间步
            positive_steps = np.where(pred_steps == 1)[0]
            
            if len(positive_steps) > 0:
                # 计算Lead-Time（秒）
                first_positive_step = positive_steps[0]
                predicted_event_time = current_time + first_positive_step
                
                if true_event_time > current_time:
                    # 真实事件在未来
                    lead_time = true_event_time - predicted_event_time
                    lead_times.append(lead_time)
                    
        if lead_times:
            results['mean_lead_time'] = float(np.mean(lead_times))
            results['median_lead_time'] = float(np.median(lead_times))
            results['std_lead_time'] = float(np.std(lead_times))
            results['min_lead_time'] = float(np.min(lead_times))
            results['max_lead_time'] = float(np.max(lead_times))
            
            # 统计不同阈值下的预警成功率
            for threshold_sec in self.lead_time_thresholds:
                success_count = np.sum(np.array(lead_times) >= threshold_sec)
                success_rate = success_count / len(lead_times)
                results[f'lead_time_{threshold_sec}s_success_rate'] = float(success_rate)
        else:
            # 没有成功预警
            for key in ['mean_lead_time', 'median_lead_time', 'std_lead_time', 'min_lead_time', 'max_lead_time']:
                results[key] = 0.0
            for threshold_sec in self.lead_time_thresholds:
                results[f'lead_time_{threshold_sec}s_success_rate'] = 0.0
                
        results['total_predictions'] = total_samples
        results['successful_predictions'] = len(lead_times)
        
        return results
        
    def evaluate_early_warning(
        self, 
        model,
        temporal_sequences: List[Dict[str, Any]],
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """
        提前预警能力评估
        
        Args:
            model: 时序预测模型
            temporal_sequences: 时序序列数据
            device: 计算设备
            
        Returns:
            预警评估结果
        """
        model.eval()
        all_predictions = []
        all_true_events = []
        all_timestamps = []
        
        with torch.no_grad():
            for seq in temporal_sequences:
                historical_graphs = seq['historical_graphs']
                current_graph = seq['current_graph']
                future_graph = seq.get('future_graph', None)
                time_index = seq['time_index']
                
                try:
                    # 模型预测
                    predictions = model(historical_graphs, current_graph)
                    
                    # 计算真实事件（简化版本）
                    if future_graph is not None:
                        # 使用未来图的风险信息作为真实事件
                        true_event = self._compute_true_event(current_graph, future_graph)
                    else:
                        true_event = 0.0  # 没有未来事件
                        
                    all_predictions.append(predictions.cpu())
                    all_true_events.append(true_event)
                    all_timestamps.append(time_index)
                    
                except Exception as e:
                    self.logger.warning(f"预测失败: {e}")
                    continue
                    
        if not all_predictions:
            return {'error': 'No valid predictions'}
            
        # 转换为张量
        predictions_tensor = torch.stack(all_predictions)
        true_events_tensor = torch.tensor(all_true_events)
        timestamps_tensor = torch.tensor(all_timestamps)
        
        # 计算Lead-Time指标
        lead_time_results = self.evaluate_lead_time(
            predictions_tensor, true_events_tensor, timestamps_tensor
        )
        
        return lead_time_results
        
    def _compute_true_event(self, current_graph, future_graph) -> float:
        """计算真实事件时间（简化实现）"""
        # 比较当前图和未来图的风险变化
        current_risk = self._compute_graph_risk(current_graph)
        future_risk = self._compute_graph_risk(future_graph)
        
        # 如果未来风险显著增加，认为有事件发生
        risk_increase = future_risk - current_risk
        if risk_increase > 0.1:  # 风险阈值
            return 1.0  # 有事件
        else:
            return 0.0  # 无事件
            
    def _compute_graph_risk(self, graph) -> float:
        """计算图的整体风险水平"""
        # 简化实现：使用节点度的均值作为风险指标
        if graph.number_of_nodes() == 0:
            return 0.0
            
        degrees = graph.in_degrees().float()
        return float(degrees.mean())


def compute_risk_weighted_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    risk_scores: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    风险加权指标计算
    
    Args:
        predictions: 预测分数 [num_samples]
        targets: 真实标签 [num_samples]
        risk_scores: 风险分数 [num_samples]
        threshold: 二值化阈值
        
    Returns:
        风险加权指标字典
    """
    # 转换为numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(risk_scores, torch.Tensor):
        risk_scores = risk_scores.cpu().numpy()
        
    # 二值化预测
    binary_predictions = (predictions > threshold).astype(int)
    
    # 标准指标
    accuracy = accuracy_score(targets, binary_predictions)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = precision_score(targets, binary_predictions, average='binary', zero_division=0)
        recall = recall_score(targets, binary_predictions, average='binary', zero_division=0)
        f1 = f1_score(targets, binary_predictions, average='binary', zero_division=0)
    
    # 风险加权指标
    # 使用风险分数作为样本权重
    sample_weights = risk_scores / (risk_scores.sum() + 1e-8)
    
    # 风险加权准确率
    weighted_accuracy = np.sum(sample_weights * (targets == binary_predictions))
    
    # 风险加权精确率
    positive_predictions = binary_predictions == 1
    if np.sum(positive_predictions) > 0:
        weighted_precision_num = np.sum(sample_weights[positive_predictions] * targets[positive_predictions])
        weighted_precision_den = np.sum(sample_weights[positive_predictions])
        weighted_precision = weighted_precision_num / (weighted_precision_den + 1e-8)
    else:
        weighted_precision = 0.0
        
    # 风险加权召回率
    positive_targets = targets == 1
    if np.sum(positive_targets) > 0:
        weighted_recall_num = np.sum(sample_weights[positive_targets] * binary_predictions[positive_targets])
        weighted_recall_den = np.sum(sample_weights[positive_targets])
        weighted_recall = weighted_recall_num / (weighted_recall_den + 1e-8)
    else:
        weighted_recall = 0.0
        
    # 风险加权F1
    if weighted_precision + weighted_recall > 0:
        weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
    else:
        weighted_f1 = 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'risk_weighted_accuracy': float(weighted_accuracy),
        'risk_weighted_precision': float(weighted_precision),
        'risk_weighted_recall': float(weighted_recall),
        'risk_weighted_f1': float(weighted_f1),
        'mean_risk_score': float(np.mean(risk_scores)),
        'std_risk_score': float(np.std(risk_scores))
    }


def directional_accuracy(
    src_scores: torch.Tensor, 
    dst_scores: torch.Tensor, 
    true_directions: torch.Tensor
) -> Dict[str, float]:
    """
    方向性预测准确率
    
    Args:
        src_scores: 源节点分数 [num_edges]
        dst_scores: 目标节点分数 [num_edges]
        true_directions: 真实方向 [num_edges] (1: src>dst, 0: src<dst)
        
    Returns:
        方向性准确率指标
    """
    # 转换为numpy
    if isinstance(src_scores, torch.Tensor):
        src_scores = src_scores.cpu().numpy()
    if isinstance(dst_scores, torch.Tensor):
        dst_scores = dst_scores.cpu().numpy()
    if isinstance(true_directions, torch.Tensor):
        true_directions = true_directions.cpu().numpy()
        
    # 预测方向
    predicted_directions = (src_scores > dst_scores).astype(int)
    
    # 计算准确率
    accuracy = np.mean(predicted_directions == true_directions)
    
    # 计算混淆矩阵
    true_positives = np.sum((predicted_directions == 1) & (true_directions == 1))
    false_positives = np.sum((predicted_directions == 1) & (true_directions == 0))
    true_negatives = np.sum((predicted_directions == 0) & (true_directions == 0))
    false_negatives = np.sum((predicted_directions == 0) & (true_directions == 1))
    
    total = len(predicted_directions)
    
    return {
        'directional_accuracy': float(accuracy),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'true_negatives': int(true_negatives),
        'false_negatives': int(false_negatives),
        'total_samples': int(total),
        'direction_error_rate': float(1 - accuracy)
    }


def compute_temporal_stability(
    predictions_sequence: List[torch.Tensor],
    window_size: int = 5
) -> Dict[str, float]:
    """
    计算时序预测稳定性
    
    Args:
        predictions_sequence: 时序预测序列
        window_size: 滑动窗口大小
        
    Returns:
        稳定性指标
    """
    if len(predictions_sequence) < 2:
        return {'temporal_stability': 0.0, 'prediction_variance': 0.0}
        
    # 计算相邻时间步预测的差异
    prediction_diffs = []
    for i in range(1, len(predictions_sequence)):
        prev_pred = predictions_sequence[i-1].cpu().numpy()
        curr_pred = predictions_sequence[i].cpu().numpy()
        
        # 计算L2差异
        diff = np.mean((prev_pred - curr_pred) ** 2)
        prediction_diffs.append(diff)
        
    # 稳定性指标
    mean_diff = np.mean(prediction_diffs)
    stability = 1.0 / (1.0 + mean_diff)  # 差异越小，稳定性越高
    
    # 预测方差
    all_predictions = torch.stack(predictions_sequence).cpu().numpy()
    prediction_variance = np.var(all_predictions, axis=0).mean()
    
    return {
        'temporal_stability': float(stability),
        'prediction_variance': float(prediction_variance),
        'mean_prediction_diff': float(mean_diff),
        'num_time_steps': len(predictions_sequence)
    }


def create_temporal_evaluator(config: Dict[str, Any]) -> TemporalEvaluator:
    """
    根据配置创建时序评估器
    
    Args:
        config: 评估器配置
        
    Returns:
        时序评估器实例
    """
    return TemporalEvaluator(
        lead_time_thresholds=config.get('lead_time_thresholds', [5, 10, 15])
    ) 