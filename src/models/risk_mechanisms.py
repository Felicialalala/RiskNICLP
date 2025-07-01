"""
简化风险机制模块 - 用于缓解过拟合
替代复杂的风险偏置注意力机制，使用简单的线性风险集成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimplifiedRiskIntegration(nn.Module):
    """简化的风险集成机制 - 线性组合而非复杂注意力"""
    
    def __init__(self, feature_dim: int, risk_weight: float = 0.1):
        """
        Args:
            feature_dim: 特征维度
            risk_weight: 风险权重 (降低以减少噪声)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.risk_weight = risk_weight
        
        # 简单的线性投影而非复杂的注意力机制
        self.risk_projection = nn.Linear(1, feature_dim)
        self.feature_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, node_features: torch.Tensor, risk_scores: torch.Tensor) -> torch.Tensor:
        """
        简化前向传播：线性风险集成
        
        Args:
            node_features: [N, feature_dim] 节点特征
            risk_scores: [N] 风险分数
            
        Returns:
            enhanced_features: [N, feature_dim] 风险增强特征
        """
        batch_size, feature_dim = node_features.shape
        
        # 风险分数投影到特征空间
        risk_expanded = risk_scores.unsqueeze(-1)  # [N, 1]
        risk_features = self.risk_projection(risk_expanded)  # [N, feature_dim]
        
        # 简单的线性组合（而非复杂的注意力机制）
        enhanced_features = node_features + self.risk_weight * risk_features
        
        # 标准化输出
        enhanced_features = self.feature_norm(enhanced_features)
        
        return enhanced_features


class SimplifiedRiskGATConv(nn.Module):
    """简化的风险感知图注意力层"""
    
    def __init__(self, in_feats: int, out_feats: int, num_heads: int = 4, 
                 dropout: float = 0.3, risk_weight: float = 0.1):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.dropout = dropout
        self.risk_weight = risk_weight
        
        # 标准GAT组件
        self.W = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * out_feats))
        
        # 简化的风险集成
        self.risk_integration = SimplifiedRiskIntegration(
            feature_dim=out_feats * num_heads,
            risk_weight=risk_weight
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
        
    def forward(self, graph, node_features, risk_scores: Optional[torch.Tensor] = None):
        """
        简化前向传播
        
        Args:
            graph: DGL图
            node_features: 节点特征
            risk_scores: 风险分数（可选）
        """
        h = self.W(node_features)  # [N, num_heads * out_feats]
        h = h.view(-1, self.num_heads, self.out_feats)  # [N, num_heads, out_feats]
        
        # 标准GAT注意力计算（简化版）
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self._edge_attention)
            e = graph.edata['e']  # [E, num_heads]
            
            # 应用dropout和softmax
            e = self.dropout_layer(e)
            graph.edata['a'] = edge_softmax(graph, e)
            
            # 消息传递
            graph.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'h'))
            h_out = graph.ndata['h']  # [N, num_heads, out_feats]
            
        # 合并多头输出
        h_out = h_out.view(-1, self.num_heads * self.out_feats)  # [N, num_heads * out_feats]
        
        # 🔧 关键简化：只在提供风险分数时应用风险集成
        if risk_scores is not None:
            h_out = self.risk_integration(h_out, risk_scores)
        
        return h_out
    
    def _edge_attention(self, edges):
        """计算边注意力（简化版）"""
        # 拼接源节点和目标节点特征
        concat_features = torch.cat([edges.src['h'], edges.dst['h']], dim=-1)  # [E, num_heads, 2*out_feats]
        
        # 计算注意力分数
        e = torch.einsum('ehd,hd->eh', concat_features, self.a)  # [E, num_heads]
        return {'e': self.leaky_relu(e)}


# 为了兼容性，提供原始接口的简化实现
class SimplifiedTrueRiskBiasGATConv(SimplifiedRiskGATConv):
    """兼容原始TrueRiskBiasGATConv接口的简化实现"""
    
    def __init__(self, in_feats: int, out_feats: int, num_heads: int = 4,
                 dropout: float = 0.3, beta_src: float = 1.2, beta_dst: float = 1.2,
                 temperature: float = 0.5, training_stage: str = 'basic'):
        # 将复杂的beta参数转化为简单的风险权重
        risk_weight = (beta_src + beta_dst - 2.0) * 0.1  # 简化映射
        
        super().__init__(
            in_feats=in_feats,
            out_feats=out_feats, 
            num_heads=num_heads,
            dropout=dropout,
            risk_weight=risk_weight
        )
        
        self.training_stage = training_stage
        
        # 为了兼容性保留原始参数（但不使用）
        self.beta_src = nn.Parameter(torch.full((num_heads,), beta_src))
        self.beta_dst = nn.Parameter(torch.full((num_heads,), beta_dst))
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # 在basic阶段冻结参数
        if training_stage == 'basic':
            self.beta_src.requires_grad_(False)
            self.beta_dst.requires_grad_(False)
            self.temperature.requires_grad_(False)
            
    def set_training_stage(self, stage: str):
        """设置训练阶段"""
        self.training_stage = stage
        
        if stage == 'basic':
            # Basic阶段：冻结风险参数
            self.beta_src.requires_grad_(False)
            self.beta_dst.requires_grad_(False)
            self.temperature.requires_grad_(False)
        else:
            # Risk-aware阶段：解冻参数
            self.beta_src.requires_grad_(True)
            self.beta_dst.requires_grad_(True)
            self.temperature.requires_grad_(True)


# 导入必要的DGL函数（如果可用）
try:
    from dgl.nn.functional import edge_softmax
    import dgl.function as fn
except ImportError:
    # 备用实现
    def edge_softmax(graph, edge_scores):
        """备用边softmax实现"""
        return F.softmax(edge_scores, dim=-1)
    
    class fn:
        @staticmethod
        def u_mul_e(u, e, out):
            return lambda edges: {out: edges.src[u] * edges.data[e].unsqueeze(-1)}
        
        @staticmethod  
        def sum(msg, out):
            return lambda nodes: {out: torch.sum(nodes.mailbox[msg], dim=1)} 