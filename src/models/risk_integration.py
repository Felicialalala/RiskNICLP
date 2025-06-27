"""
RiskNICLP-5G Risk Integration Module

Provides risk integration components for risk-aware modeling with heterogeneous graph support
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class RiskIntegrator(nn.Module):
    """Risk integration module for RiskNICLP-5G with heterogeneous node support"""
    
    def __init__(self, hidden_dim: int = 64, risk_embed_dim: int = 16):
        """
        Initialize risk integrator
        
        Args:
            hidden_dim: Hidden dimension size
            risk_embed_dim: Risk embedding dimension
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.risk_embed_dim = risk_embed_dim
        
        # Risk integration layers
        self.risk_projection = nn.Linear(risk_embed_dim, hidden_dim)
        self.integration_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        print(f"🔧 RiskIntegrator初始化:")
        print(f"  隐藏维度: {hidden_dim}, 风险嵌入维度: {risk_embed_dim}")
        
    def forward(self, 
                node_embeddings: Dict[str, torch.Tensor], 
                risk_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        按节点类型集成风险信息
        
        Args:
            node_embeddings: {节点类型: 嵌入张量}
            risk_scores: {节点类型: 风险分数}
            
        Returns:
            {节点类型: 风险增强嵌入}
        """
        
        # 减少调试输出频率
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        debug_enabled = self._debug_counter % 100 == 0  # 每100次调用输出一次
        
        if debug_enabled:
            print(f"🔍 RiskIntegrator处理:")
        
        risk_enhanced_embeddings = {}
        
        for ntype, embeddings in node_embeddings.items():
            if len(embeddings) == 0:
                risk_enhanced_embeddings[ntype] = embeddings
                continue
                
            if debug_enabled:
                print(f"  处理 {ntype}: 嵌入{embeddings.shape}")
            
            # 获取对应的风险分数
            if ntype in risk_scores and len(risk_scores[ntype]) > 0:
                risk_vals = risk_scores[ntype]
                if debug_enabled:
                    print(f"    风险分数: {risk_vals.shape}")
                
                # 简单的风险增强：风险分数作为权重
                risk_weights = torch.sigmoid(risk_vals).unsqueeze(-1)  # [N, 1]
                enhanced = embeddings * (1.0 + risk_weights)  # 基于风险放大特征
                
                if debug_enabled:
                    print(f"    ✅ 风险增强完成: {enhanced.shape}")
                risk_enhanced_embeddings[ntype] = enhanced
            else:
                # 无风险信息时直接传递
                if debug_enabled:
                    print(f"    ⚠️ 无风险信息，直接传递")
                risk_enhanced_embeddings[ntype] = embeddings
        
        if debug_enabled:
            print(f"📊 RiskIntegrator输出: {list(risk_enhanced_embeddings.keys())}")
        
        return risk_enhanced_embeddings
        
    def forward_legacy(self, node_embeddings: torch.Tensor, risk_embeddings: torch.Tensor) -> torch.Tensor:
        """
        传统的风险集成 (兼容性)
        
        Args:
            node_embeddings: Node embeddings tensor
            risk_embeddings: Risk embeddings tensor
            
        Returns:
            Risk-integrated embeddings
        """
        # Project risk embeddings to hidden dimension
        risk_proj = self.risk_projection(risk_embeddings)
        
        # Concatenate and integrate
        combined = torch.cat([node_embeddings, risk_proj], dim=-1)
        integrated = self.integration_layer(combined)
        
        # Residual connection
        output = node_embeddings + integrated
        
        return output
        
    def compute_risk_bias(self, risk_scores: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Compute risk-based attention bias
        
        Args:
            risk_scores: Risk scores tensor
            temperature: Temperature for attention scaling
            
        Returns:
            Risk bias tensor
        """
        # Normalize risk scores
        risk_norm = torch.softmax(risk_scores / temperature, dim=-1)
        
        return risk_norm 