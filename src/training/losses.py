"""
RiskNICLP-5G Loss Functions

Includes BCE loss, focal loss, and InfoNCE contrastive loss for risk-aware training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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