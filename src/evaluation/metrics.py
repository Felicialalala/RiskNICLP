"""
Risk-aware evaluation metrics for RiskNICLP-5G

Provides specialized metrics for security-oriented evaluation
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score


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