"""
RiskNICLP-5G Loss Functions for Attack Propagation Prediction

Includes specialized loss functions for attack propagation prediction tasks
Replaces anomaly detection losses with attack-focused objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class RiskAwareBCELoss(nn.Module):
    """Risk-aware BCE loss with positive weight balancing"""
    
    def __init__(self, pos_weight_factor: float = 1.0):
        super().__init__()
        self.pos_weight_factor = pos_weight_factor
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions [N]
            targets: Ground truth labels [N] 
            pos_weight: Positive class weight (optional)
        """
        if pos_weight is not None:
            pos_weight = pos_weight * self.pos_weight_factor
            
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return loss_fn(predictions, targets)


class FocalLoss(nn.Module):
    """Focal Loss for addressing extreme class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions [N]
            targets: Ground truth labels [N]
        """
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class InfoNCEContrastiveLoss(nn.Module):
    """✨ 增益螺丝2: InfoNCE对比损失 - 轻量版"""
    
    def __init__(self, temperature: float = 0.07, risk_threshold: float = 0.8):
        super().__init__()
        self.temperature = temperature
        self.risk_threshold = risk_threshold
        
    def forward(self, embeddings: torch.Tensor, risk_scores: torch.Tensor,
                edge_pairs: torch.Tensor, edge_labels: torch.Tensor) -> torch.Tensor:
        """
        轻量级InfoNCE：对每个batch收集高风险正边及其负样本进行对比学习
        
        Args:
            embeddings: 节点嵌入 [N, D]
            risk_scores: 风险分数 [N]
            edge_pairs: 边对 [E, 2]
            edge_labels: 边标签 [E]
            
        Returns:
            InfoNCE对比损失
        """
        
        # 找到高风险正边作为query
        pos_mask = edge_labels == 1
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)
            
        pos_edges = edge_pairs[pos_mask]  # [P, 2]
        neg_edges = edge_pairs[~pos_mask]  # [N, 2]
        
        # 计算正边的平均风险
        pos_u_risk = risk_scores[pos_edges[:, 0]]
        pos_v_risk = risk_scores[pos_edges[:, 1]]
        pos_avg_risk = (pos_u_risk + pos_v_risk) / 2
        
        # 选择高风险正边
        high_risk_pos_mask = pos_avg_risk > self.risk_threshold
        if high_risk_pos_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)
            
        high_risk_pos_edges = pos_edges[high_risk_pos_mask]  # [H, 2]
        
        # 对每个高风险正边计算InfoNCE
        contrastive_losses = []
        
        for i in range(min(len(high_risk_pos_edges), 10)):  # 最多处理10个样本
            query_edge = high_risk_pos_edges[i]  # [2]
            
            # Query嵌入：正边节点嵌入的平均
            query_u = embeddings[query_edge[0]]
            query_v = embeddings[query_edge[1]]
            query_embed = (query_u + query_v) / 2  # [D]
            
            # 正样本：其他高风险正边
            positive_embeds = []
            for j in range(len(high_risk_pos_edges)):
                if i != j:
                    pos_u = embeddings[high_risk_pos_edges[j, 0]]
                    pos_v = embeddings[high_risk_pos_edges[j, 1]]
                    positive_embeds.append((pos_u + pos_v) / 2)
            
            # 负样本：随机选择5个负边
            neg_sample_size = min(5, len(neg_edges))
            if neg_sample_size > 0:
                neg_indices = torch.randperm(len(neg_edges))[:neg_sample_size]
                negative_embeds = []
                for idx in neg_indices:
                    neg_u = embeddings[neg_edges[idx, 0]]
                    neg_v = embeddings[neg_edges[idx, 1]]
                    negative_embeds.append((neg_u + neg_v) / 2)
            else:
                negative_embeds = []
            
            # 计算相似度
            if positive_embeds:
                pos_sims = [F.cosine_similarity(query_embed, pos_embed, dim=0) for pos_embed in positive_embeds]
                pos_sim = torch.stack(pos_sims).mean()  # 平均正样本相似度
            else:
                continue
                
            if negative_embeds:
                neg_sims = [F.cosine_similarity(query_embed, neg_embed, dim=0) for neg_embed in negative_embeds]
                neg_sim_stack = torch.stack(neg_sims)
            else:
                neg_sim_stack = torch.tensor([])
            
            # InfoNCE损失: -log(exp(pos_sim/τ) / (exp(pos_sim/τ) + Σexp(neg_sim/τ)))
            pos_exp = torch.exp(pos_sim / self.temperature)
            if len(neg_sim_stack) > 0:
                neg_exp = torch.exp(neg_sim_stack / self.temperature).sum()
                loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))
            else:
                loss = torch.tensor(0.0)
            
            contrastive_losses.append(loss)
        
        if contrastive_losses:
            return torch.stack(contrastive_losses).mean()
        else:
            return torch.tensor(0.0, requires_grad=True)


class CombinedLoss(nn.Module):
    """Combined loss: BCE + Focal + InfoNCE"""
    
    def __init__(self, 
                 bce_weight: float = 1.0,
                 focal_weight: float = 0.5, 
                 infonce_weight: float = 0.2):
        super().__init__()
        
        self.bce_loss = RiskAwareBCELoss()
        self.focal_loss = FocalLoss()
        self.infonce_loss = InfoNCEContrastiveLoss()
        
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.infonce_weight = infonce_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                embeddings: Optional[torch.Tensor] = None,
                risk_scores: Optional[torch.Tensor] = None,
                edge_pairs: Optional[torch.Tensor] = None,
                pos_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Combined loss computation
        
        Returns:
            total_loss, loss_dict
        """
        
        # BCE Loss
        bce_loss = self.bce_loss(predictions, targets, pos_weight)
        
        # Focal Loss  
        focal_loss = self.focal_loss(predictions, targets)
        
        # InfoNCE Loss (if embeddings available)
        if embeddings is not None and risk_scores is not None and edge_pairs is not None:
            infonce_loss = self.infonce_loss(embeddings, risk_scores, edge_pairs, targets)
        else:
            infonce_loss = torch.tensor(0.0)
        
        # Combined loss
        total_loss = (self.bce_weight * bce_loss + 
                     self.focal_weight * focal_loss + 
                     self.infonce_weight * infonce_loss)
        
        loss_dict = {
            'bce_loss': bce_loss.item(),
            'focal_loss': focal_loss.item(), 
            'infonce_loss': infonce_loss.item() if isinstance(infonce_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


class AttackPropagationLoss(nn.Module):
    """
    攻击传播预测专用损失函数
    专门针对攻击链路预测任务设计，替代传统的异常检测损失
    """

    def __init__(self,
                 base_weight: float = 1.0,
                 temporal_weight: float = 0.3,
                 propagation_weight: float = 0.4,
                 early_warning_weight: float = 0.5,
                 false_alarm_penalty: float = 2.0):
        """
        初始化攻击传播损失函数

        Args:
            base_weight: 基础预测损失权重
            temporal_weight: 时序一致性损失权重
            propagation_weight: 传播链完整性损失权重
            early_warning_weight: 提前预警损失权重
            false_alarm_penalty: 误报惩罚权重
        """
        super().__init__()

        self.base_weight = base_weight
        self.temporal_weight = temporal_weight
        self.propagation_weight = propagation_weight
        self.early_warning_weight = early_warning_weight
        self.false_alarm_penalty = false_alarm_penalty

        # 基础损失函数
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_loss = FocalLoss(alpha=0.35, gamma=1.5)  # 调整为攻击预测参数

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                edge_types: Optional[torch.Tensor] = None,
                timestamps: Optional[torch.Tensor] = None,
                risk_scores: Optional[torch.Tensor] = None,
                edge_pairs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算攻击传播损失

        Args:
            predictions: 模型预测 [N]
            targets: 真实标签 [N]
            edge_types: 边类型 [N] (可选)
            timestamps: 时间戳 [N] (可选)
            risk_scores: 风险分数 [N] (可选)
            edge_pairs: 边对 [N, 2] (可选)

        Returns:
            total_loss, loss_components
        """

        # 1. 基础攻击预测损失
        base_loss = self._compute_base_attack_loss(predictions, targets, edge_types)

        # 2. 时序一致性损失
        temporal_loss = self._compute_temporal_consistency_loss(
            predictions, targets, timestamps, edge_pairs
        ) if timestamps is not None else torch.tensor(0.0)

        # 3. 攻击传播链完整性损失
        propagation_loss = self._compute_propagation_chain_loss(
            predictions, targets, edge_types, risk_scores
        ) if edge_types is not None else torch.tensor(0.0)

        # 4. 提前预警损失
        early_warning_loss = self._compute_early_warning_loss(
            predictions, targets, timestamps
        ) if timestamps is not None else torch.tensor(0.0)

        # 5. 误报惩罚
        false_alarm_penalty = self._compute_false_alarm_penalty(predictions, targets)

        # 组合损失
        total_loss = (
            self.base_weight * base_loss +
            self.temporal_weight * temporal_loss +
            self.propagation_weight * propagation_loss +
            self.early_warning_weight * early_warning_loss +
            self.false_alarm_penalty * false_alarm_penalty
        )

        loss_components = {
            'base_loss': base_loss.item(),
            'temporal_loss': temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else 0.0,
            'propagation_loss': propagation_loss.item() if isinstance(propagation_loss, torch.Tensor) else 0.0,
            'early_warning_loss': early_warning_loss.item() if isinstance(early_warning_loss, torch.Tensor) else 0.0,
            'false_alarm_penalty': false_alarm_penalty.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_components

    def _compute_base_attack_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                                edge_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算基础攻击预测损失"""

        # 使用Focal Loss处理类别不平衡（攻击预测中正样本比例25-35%）
        focal_loss = self.focal_loss(predictions, targets)

        # 如果有边类型信息，对不同类型的边使用不同权重
        if edge_types is not None:
            type_weights = self._get_edge_type_weights(edge_types)
            bce_losses = self.bce_loss(predictions, targets)
            weighted_bce = (bce_losses * type_weights).mean()

            return 0.7 * focal_loss + 0.3 * weighted_bce
        else:
            return focal_loss

    def _get_edge_type_weights(self, edge_types: torch.Tensor) -> torch.Tensor:
        """获取边类型权重"""
        weights = torch.ones_like(edge_types, dtype=torch.float)

        # 根据边类型的重要性分配权重
        type_weight_map = {
            'triggers': 1.2,      # Flow->Alert: 高权重（关键告警）
            'initiates': 1.1,     # UE->Flow: 中高权重（攻击发起）
            'temporal_adj': 1.0,  # Flow->Flow: 标准权重（攻击传播）
        }

        for i, edge_type in enumerate(edge_types):
            if isinstance(edge_type, str):
                weights[i] = type_weight_map.get(edge_type, 1.0)
            else:
                weights[i] = 1.0  # 默认权重

        return weights

    def _compute_temporal_consistency_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                                         timestamps: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        """计算时序一致性损失"""
        if len(timestamps) < 2:
            return torch.tensor(0.0)

        # 按时间排序
        sorted_indices = torch.argsort(timestamps)
        sorted_predictions = predictions[sorted_indices]
        sorted_targets = targets[sorted_indices]

        # 计算相邻时间步的预测一致性
        consistency_loss = 0.0
        count = 0

        for i in range(len(sorted_predictions) - 1):
            # 如果连续两个时间步都是攻击，预测应该保持一致性
            if sorted_targets[i] == 1 and sorted_targets[i + 1] == 1:
                pred_diff = torch.abs(sorted_predictions[i] - sorted_predictions[i + 1])
                consistency_loss += pred_diff
                count += 1

        return consistency_loss / max(count, 1)

    def _compute_propagation_chain_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                                      edge_types: torch.Tensor, risk_scores: torch.Tensor) -> torch.Tensor:
        """计算攻击传播链完整性损失"""
        if edge_types is None or risk_scores is None:
            return torch.tensor(0.0)

        # 识别攻击传播链：initiates -> triggers -> temporal_adj
        chain_loss = 0.0
        chain_count = 0

        # 简化的链完整性检查
        initiate_mask = (edge_types == 'initiates') if hasattr(edge_types, '__iter__') else torch.zeros_like(targets, dtype=torch.bool)
        trigger_mask = (edge_types == 'triggers') if hasattr(edge_types, '__iter__') else torch.zeros_like(targets, dtype=torch.bool)

        # 如果有攻击发起但没有对应的告警触发，增加损失
        if torch.any(initiate_mask) and torch.any(trigger_mask):
            initiate_predictions = predictions[initiate_mask]
            trigger_predictions = predictions[trigger_mask]

            # 攻击发起的预测应该与告警触发的预测相关
            if len(initiate_predictions) > 0 and len(trigger_predictions) > 0:
                initiate_mean = initiate_predictions.mean()
                trigger_mean = trigger_predictions.mean()

                # 如果攻击发起预测高但告警触发预测低，增加损失
                chain_inconsistency = F.relu(initiate_mean - trigger_mean - 0.1)
                chain_loss += chain_inconsistency
                chain_count += 1

        return chain_loss / max(chain_count, 1)

    def _compute_early_warning_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                                  timestamps: torch.Tensor) -> torch.Tensor:
        """计算提前预警损失"""
        if len(timestamps) < 2:
            return torch.tensor(0.0)

        # 奖励提前预警：在攻击发生前的预测
        warning_loss = 0.0
        warning_count = 0

        # 按时间排序
        sorted_indices = torch.argsort(timestamps)
        sorted_predictions = predictions[sorted_indices]
        sorted_targets = targets[sorted_indices]
        sorted_timestamps = timestamps[sorted_indices]

        for i in range(len(sorted_targets) - 1):
            # 如果当前是预警（高预测值），下一个时间步是攻击
            if sorted_predictions[i] > 0.7 and sorted_targets[i + 1] == 1:
                # 计算提前预警时间
                lead_time = sorted_timestamps[i + 1] - sorted_timestamps[i]

                # 奖励更长的提前预警时间（负损失）
                if lead_time > 0:
                    warning_bonus = -torch.log(1 + lead_time / 30.0)  # 30秒为基准
                    warning_loss += warning_bonus
                    warning_count += 1

        return warning_loss / max(warning_count, 1)

    def _compute_false_alarm_penalty(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算误报惩罚"""
        # 识别误报：高预测值但实际为负样本
        false_alarm_mask = (predictions > 0.5) & (targets == 0)

        if torch.any(false_alarm_mask):
            false_alarm_predictions = predictions[false_alarm_mask]
            # 误报惩罚与预测置信度成正比
            penalty = (false_alarm_predictions - 0.5).mean()
            return penalty
        else:
            return torch.tensor(0.0)