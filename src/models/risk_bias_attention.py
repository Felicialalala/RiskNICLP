import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
import dgl


class SimplifiedRiskBiasGAT(nn.Module):
    """简化版风险偏置GAT (Phase 2.1优先实现)"""
    
    def __init__(self, in_feats, out_feats, num_heads=4, dropout=0.1):
        super(SimplifiedRiskBiasGAT, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        
        # GAT基础层
        self.gat_conv = GATConv(in_feats, out_feats, num_heads, dropout=dropout, allow_zero_in_degree=True)
        
        # 简化版：单β参数，便于快速验证
        self.beta = nn.Parameter(torch.tensor(0.7))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, graph, features, risk_scores=None):
        """
        Args:
            graph: DGL图
            features: 节点特征 [num_nodes, in_feats]
            risk_scores: 风险分数 [num_nodes] (可选)
        """
        # 基础GAT计算
        h = self.gat_conv(graph, features)  # [num_nodes, num_heads, out_feats]
        
        # 如果提供风险分数，应用风险偏置
        if risk_scores is not None:
            # 简化风险增强：h = h * (1 + β * risk_scores)
            risk_enhancement = 1 + self.beta * risk_scores.view(-1, 1, 1)
            h = h * risk_enhancement
            
        # 平均多头输出
        h = h.mean(dim=1)  # [num_nodes, out_feats]
        h = self.dropout(h)
        
        return h


class RiskBiasGATConv(nn.Module):
    """完整版风险偏置图注意力卷积层"""
    
    def __init__(self, in_feats, out_feats, num_heads=4, dropout=0.1):
        super(RiskBiasGATConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        
        # GAT基础层
        self.gat_conv = GATConv(in_feats, out_feats, num_heads, dropout=dropout, allow_zero_in_degree=True)
        
        # 风险偏置参数
        self.risk_projector = nn.Linear(1, num_heads)
        self.temperature = nn.Parameter(torch.tensor(0.1))
        self.beta_1 = nn.Parameter(torch.tensor(0.5))  # 源节点风险权重
        self.beta_2 = nn.Parameter(torch.tensor(0.5))  # 目标节点风险权重
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, graph, features, risk_scores):
        """
        实现β₁·r_i + β₂·r_j风险偏置注意力
        
        Args:
            graph: DGL图
            features: 节点特征 [num_nodes, in_feats]
            risk_scores: 风险分数 [num_nodes]
        """
        # 获取边的源节点和目标节点
        src_nodes, dst_nodes = graph.edges()
        
        # 提取边对应的风险分数
        src_risk = risk_scores[src_nodes].unsqueeze(-1)  # [num_edges, 1]
        dst_risk = risk_scores[dst_nodes].unsqueeze(-1)  # [num_edges, 1]
        
        # 计算风险偏置: β₁·r_i + β₂·r_j
        risk_bias = self.beta_1 * src_risk + self.beta_2 * dst_risk  # [num_edges, 1]
        risk_bias = self.risk_projector(risk_bias)  # [num_edges, num_heads]
        risk_bias = risk_bias / self.temperature
        
        # 基础GAT注意力计算
        h = self.gat_conv(graph, features)  # [num_nodes, num_heads, out_feats]
        
        # TODO: 完整实现需要修改DGL GAT内部注意力计算
        # 当前阶段使用简化的风险增强
        if risk_scores is not None:
            risk_enhancement = 1 + torch.tanh(risk_bias.mean(dim=0).mean()) * risk_scores.view(-1, 1, 1)
            h = h * risk_enhancement
            
        # 平均多头输出
        h = h.mean(dim=1)  # [num_nodes, out_feats]
        h = self.dropout(h)
        
        return h


class RiskBiasAttentionWrapper(nn.Module):
    """风险偏置注意力包装器，支持不同实现的切换"""
    
    def __init__(self, in_feats, out_feats, num_heads=4, dropout=0.1, method='simplified'):
        super(RiskBiasAttentionWrapper, self).__init__()
        self.method = method
        
        if method == 'simplified':
            self.attention_layer = SimplifiedRiskBiasGAT(in_feats, out_feats, num_heads, dropout)
        elif method == 'full':
            self.attention_layer = RiskBiasGATConv(in_feats, out_feats, num_heads, dropout)
        else:
            # 回退到标准GAT
            self.attention_layer = GATConv(in_feats, out_feats, num_heads, dropout=dropout, allow_zero_in_degree=True)
            
    def forward(self, graph, features, risk_scores=None):
        if self.method in ['simplified', 'full']:
            return self.attention_layer(graph, features, risk_scores)
        else:
            # 标准GAT不使用风险分数
            h = self.attention_layer(graph, features)
            return h.mean(dim=1) if len(h.shape) == 3 else h 