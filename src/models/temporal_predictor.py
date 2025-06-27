"""
Temporal Prediction Module for RiskNICLP-5G

Implements graph temporal modeling approaches:
- GraphRNNCell: Graph-RNN基础单元
- TGATLayer: 时序图注意力层
- GraphTemporalPredictor: 图时序预测器主类
- TemporalDataLoader: 时序数据加载器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import dgl
from dgl.nn import GraphConv
import numpy as np
from ..common.logger import get_logger


class GraphRNNCell(nn.Module):
    """Graph-RNN基础单元"""
    
    def __init__(self, hidden_dim: int, edge_dim: Optional[int] = None):
        super(GraphRNNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim or hidden_dim
        
        # 节点级GRU
        self.node_gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 边级GRU (可选)
        self.edge_gru = nn.GRUCell(self.edge_dim, hidden_dim) if edge_dim else None
        
        # 图卷积层
        self.graph_conv = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        
        # 消息传递层
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.logger = get_logger()
        
    def forward(self, graph: dgl.DGLGraph, node_features: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Graph-RNN前向传播
        
        Args:
            graph: DGL图
            node_features: 节点特征 [num_nodes, hidden_dim]
            hidden_state: 隐藏状态 [num_nodes, hidden_dim]
            
        Returns:
            updated_hidden: 更新后的隐藏状态 [num_nodes, hidden_dim]
        """
        num_nodes = graph.number_of_nodes()
        
        # 初始化隐藏状态
        if hidden_state is None:
            hidden_state = torch.zeros(num_nodes, self.hidden_dim, device=node_features.device)
        
        # 图卷积获取邻居信息
        neighbor_info = self.graph_conv(graph, hidden_state)
        
        # 组合当前特征和邻居信息
        combined_input = self.message_mlp(torch.cat([node_features, neighbor_info], dim=-1))
        
        # GRU更新
        updated_hidden = self.node_gru(combined_input, hidden_state)
        
        return updated_hidden


class TGATLayer(nn.Module):
    """时序图注意力层"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, time_dim: int = 16):
        super(TGATLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.time_dim = time_dim
        
        # 时序注意力
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # 时间编码器
        self.time_encoder = nn.Linear(1, time_dim)
        
        # 节点Transformer
        self.node_transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        
        # 时间位置编码
        self.temporal_position_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim))  # 最多100个时间步
        
        self.logger = get_logger()
        
    def forward(self, historical_graphs: List[dgl.DGLGraph], current_graph: dgl.DGLGraph, timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        TGAT前向传播
        
        Args:
            historical_graphs: 历史图序列
            current_graph: 当前图
            timestamps: 时间戳 [num_time_steps]
            
        Returns:
            temporal_embeddings: 时序嵌入 [num_nodes, hidden_dim]
        """
        all_graphs = historical_graphs + [current_graph]
        num_time_steps = len(all_graphs)
        
        if num_time_steps == 0:
            raise ValueError("至少需要一个图")
        
        # 获取节点数量（假设所有图具有相同的节点数）
        num_nodes = current_graph.number_of_nodes()
        
        # 提取每个时间步的节点特征
        temporal_node_features = []
        for i, graph in enumerate(all_graphs):
            # 简化版本：使用节点度作为特征
            if 'feat' in graph.ndata:
                node_feat = graph.ndata['feat']
            else:
                # 使用节点度作为默认特征
                degrees = graph.in_degrees().float().unsqueeze(-1)
                node_feat = degrees.expand(-1, self.hidden_dim)
                
            temporal_node_features.append(node_feat)
        
        # 堆叠时序特征 [num_time_steps, num_nodes, hidden_dim]
        temporal_features = torch.stack(temporal_node_features, dim=0)
        
        # 添加时间位置编码
        if num_time_steps <= self.temporal_position_encoding.size(1):
            pos_encoding = self.temporal_position_encoding[:, :num_time_steps, :]
            temporal_features = temporal_features + pos_encoding.transpose(0, 1)
        
        # 对每个节点应用时序注意力
        node_embeddings = []
        for node_idx in range(num_nodes):
            # 获取该节点的时序特征 [num_time_steps, hidden_dim]
            node_temporal_feat = temporal_features[:, node_idx, :].unsqueeze(0)  # [1, num_time_steps, hidden_dim]
            
            # 应用时序注意力
            attended_feat, _ = self.temporal_attention(
                node_temporal_feat, node_temporal_feat, node_temporal_feat
            )
            
            # 应用Transformer编码器
            encoded_feat = self.node_transformer(attended_feat)
            
            # 取最后时间步的嵌入
            final_embedding = encoded_feat[0, -1, :]  # [hidden_dim]
            node_embeddings.append(final_embedding)
        
        # 组合所有节点嵌入
        temporal_embeddings = torch.stack(node_embeddings, dim=0)  # [num_nodes, hidden_dim]
        
        return temporal_embeddings


class GraphTemporalPredictor(nn.Module):
    """图时序预测器主类 - 优化架构设计"""
    
    def __init__(self, hidden_dim: int = 64, future_steps: int = 3, method: str = 'hetero_lstm', shared_encoder=None):
        super(GraphTemporalPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.future_steps = future_steps
        self.method = method
        
        # 核心架构选择
        if method == 'hetero_lstm':
            # P0 版本: HeteroGNN + LSTM (推荐)
            self.spatial_encoder = shared_encoder  # 复用现有SharedEncoder
            self.temporal_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.aggregation_method = 'graph_level'  # 图级聚合
        elif method == 'tgat_risk':
            # P1 版本: TGAT + Risk-Bias (目标)
            self.temporal_encoder = TGATLayer(hidden_dim)
            self.risk_bias_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
            self.aggregation_method = 'edge_level'  # 边级事件流
        elif method == 'graph_rnn':
            # P2 版本: Graph-RNN (消融对比)
            self.temporal_encoder = GraphRNNCell(hidden_dim)
            self.aggregation_method = 'node_level'  # 节点级生成
        elif method == 'lstm_baseline':
            # 纯LSTM基线
            self.temporal_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.aggregation_method = 'sequence_only'
        else:
            raise ValueError(f"未知的时序建模方法: {method}")
        
        # 风险预测器
        self.risk_predictor = nn.Linear(hidden_dim, future_steps)
        
        # 特征投影层
        self.feature_projection = nn.Linear(1, hidden_dim)
        
        # 图级聚合器 (用于hetero_lstm)
        self.graph_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.logger = get_logger()
        
    def forward(self, historical_data: Any, current_data: Any) -> torch.Tensor:
        """
        支持四种时序建模方法的统一接口
        
        Args:
            historical_data: 历史数据（格式取决于方法）
            current_data: 当前数据
            
        Returns:
            future_risks: 未来风险预测 [num_nodes, future_steps] 或 [batch_size, future_steps]
        """
        if self.method == 'hetero_lstm':
            return self._hetero_lstm_forward(historical_data, current_data)
        elif self.method == 'tgat_risk':
            return self._tgat_risk_forward(historical_data, current_data)
        elif self.method == 'graph_rnn':
            return self._graph_rnn_forward(historical_data, current_data)
        elif self.method == 'lstm_baseline':
            return self._lstm_forward(historical_data, current_data)
        else:
            raise ValueError(f"未支持的方法: {self.method}")
    
    def _hetero_lstm_forward(self, historical_graphs: List[dgl.DGLGraph], current_graph: dgl.DGLGraph) -> torch.Tensor:
        """HeteroGNN + LSTM前向传播 - P0优先实施方案"""
        try:
            # Step 1: 空间编码 - 复用现有SharedEncoder处理每个图
            temporal_embeddings = []
            all_graphs = historical_graphs + [current_graph]
            
            for graph in all_graphs:
                if self.spatial_encoder is not None:
                    # 准备SharedEncoder需要的输入格式
                    node_features = {}
                    for ntype in graph.ntypes:
                        if 'feat' in graph.nodes[ntype].data:
                            node_features[ntype] = graph.nodes[ntype].data['feat']
                    
                    # 准备风险分数 - 修复类型错误
                    all_risk_scores = []
                    for ntype in graph.ntypes:
                        if 'risk_scores' in graph.nodes[ntype].data:
                            all_risk_scores.append(graph.nodes[ntype].data['risk_scores'])
                        else:
                            # 为没有风险分数的节点类型生成默认值
                            num_nodes = graph.number_of_nodes(ntype)
                            if ntype == 'Alert':
                                default_risk = 0.8
                            elif ntype == 'Flow':
                                default_risk = 0.3
                            elif ntype == 'UE':
                                default_risk = 0.1
                            else:
                                default_risk = 0.2
                            all_risk_scores.append(torch.full((num_nodes,), default_risk, dtype=torch.float32))
                    
                    # 合并所有节点类型的风险分数为单个tensor
                    if all_risk_scores:
                        risk_scores = torch.cat(all_risk_scores, dim=0)
                    else:
                        # 兜底：生成与图节点总数匹配的默认风险分数
                        total_nodes = sum([graph.number_of_nodes(ntype) for ntype in graph.ntypes])
                        risk_scores = torch.full((total_nodes,), 0.3, dtype=torch.float32)
                    
                    # 使用SharedEncoder获得图嵌入
                    spatial_embeddings = self.spatial_encoder(graph, node_features, risk_scores)
                    
                    # 图级聚合：将所有节点类型的嵌入合并
                    all_embeddings = []
                    for ntype, embedding in spatial_embeddings.items():
                        all_embeddings.append(embedding.mean(dim=0))  # 每种节点类型的平均嵌入
                    
                    combined_embedding = torch.stack(all_embeddings, dim=0).mean(dim=0)  # 所有类型的平均
                    graph_embedding = self.graph_aggregator(combined_embedding.unsqueeze(0))
                else:
                    # 兜底方案：使用节点度统计作为图特征
                    degrees = graph.in_degrees().float()
                    graph_stats = torch.stack([
                        degrees.mean(),
                        degrees.std(),
                        degrees.max().float(),
                        graph.number_of_edges() / max(graph.number_of_nodes(), 1)
                    ])
                    graph_embedding = self.graph_aggregator(graph_stats.unsqueeze(0))
                
                temporal_embeddings.append(graph_embedding.squeeze(0))
            
            # Step 2: 时序建模 - LSTM处理图序列
            temporal_sequence = torch.stack(temporal_embeddings, dim=0).unsqueeze(0)  # [1, seq_len, hidden_dim]
            lstm_output, (hidden, cell) = self.temporal_encoder(temporal_sequence)
            
            # Step 3: 风险预测 - 使用最后时间步的隐状态
            future_risks = self.risk_predictor(hidden[-1])  # [1, future_steps]
            
            return future_risks.squeeze(0)  # [future_steps]
            
        except Exception as e:
            self.logger.logger.error(f"HeteroLSTM前向传播失败: {e}")
            return torch.zeros(self.future_steps)
    
    def _tgat_risk_forward(self, historical_graphs: List[dgl.DGLGraph], current_graph: dgl.DGLGraph) -> torch.Tensor:
        """TGAT + Risk-Bias前向传播 - P1目标方案"""
        try:
            # 使用TGAT处理时序图
            temporal_embeddings = self.temporal_encoder(historical_graphs, current_graph)
            
            # 添加风险偏置注意力
            if hasattr(self, 'risk_bias_attention'):
                # 简化版风险偏置：使用节点度作为风险代理
                risk_proxy = current_graph.in_degrees().float().unsqueeze(-1)
                risk_weights = torch.softmax(risk_proxy, dim=0)
                
                # 风险加权的注意力
                weighted_embeddings = temporal_embeddings * risk_weights
                attended_features, _ = self.risk_bias_attention(
                    weighted_embeddings.unsqueeze(0),
                    weighted_embeddings.unsqueeze(0), 
                    weighted_embeddings.unsqueeze(0)
                )
                temporal_embeddings = attended_features.squeeze(0)
            
            # 图级聚合 + 风险预测
            graph_embedding = torch.mean(temporal_embeddings, dim=0)
            future_risks = self.risk_predictor(graph_embedding)
            
            return future_risks
            
        except Exception as e:
            self.logger.logger.error(f"TGAT-Risk前向传播失败: {e}")
            return torch.zeros(self.future_steps)

    def _graph_rnn_forward(self, historical_graphs: List[dgl.DGLGraph], current_graph: dgl.DGLGraph) -> torch.Tensor:
        """Graph-RNN前向传播 - P2消融实验方案"""
        # 初始化隐藏状态
        num_nodes = current_graph.number_of_nodes()
        hidden_state = None
        
        # 获取当前图的节点特征
        if 'feat' in current_graph.ndata:
            current_features = current_graph.ndata['feat']
        else:
            # 使用节点度作为特征
            degrees = current_graph.in_degrees().float().unsqueeze(-1)
            current_features = self.feature_projection(degrees)
        
        # 遍历历史图序列
        for graph_t in historical_graphs:
            # 获取节点特征
            if 'feat' in graph_t.ndata:
                node_features = graph_t.ndata['feat']
            else:
                degrees = graph_t.in_degrees().float().unsqueeze(-1)
                node_features = self.feature_projection(degrees)
            
            # 更新隐藏状态
            hidden_state = self.temporal_encoder(graph_t, node_features, hidden_state)
        
        # 处理当前时间步
        final_hidden = self.temporal_encoder(current_graph, current_features, hidden_state)
        
        # 预测未来风险
        future_risks = self.risk_predictor(final_hidden)
        
        return future_risks
    
    def _tgat_forward(self, historical_graphs: List[dgl.DGLGraph], current_graph: dgl.DGLGraph) -> torch.Tensor:
        """TGAT前向传播"""
        # 使用TGAT处理图序列
        temporal_embeddings = self.temporal_encoder(historical_graphs, current_graph)
        
        # 预测未来风险
        future_risks = self.risk_predictor(temporal_embeddings)
        
        return future_risks
    
    def _lstm_forward(self, historical_embeddings: torch.Tensor, current_embedding: torch.Tensor) -> torch.Tensor:
        """LSTM基线前向传播"""
        # 组合历史嵌入和当前嵌入
        if len(historical_embeddings.shape) == 2:
            # [seq_len, hidden_dim] -> [1, seq_len, hidden_dim]
            historical_embeddings = historical_embeddings.unsqueeze(0)
        if len(current_embedding.shape) == 1:
            # [hidden_dim] -> [1, 1, hidden_dim]
            current_embedding = current_embedding.unsqueeze(0).unsqueeze(0)
        elif len(current_embedding.shape) == 2:
            # [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
            current_embedding = current_embedding.unsqueeze(1)
        
        # 连接序列
        sequence = torch.cat([historical_embeddings, current_embedding], dim=1)
        
        # LSTM编码
        lstm_out, _ = self.temporal_encoder(sequence)
        
        # 使用最后时刻的输出进行预测
        final_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # 预测未来风险
        future_risks = self.risk_predictor(final_output)
        
        return future_risks


class TemporalDataLoader:
    """时序数据加载器"""
    
    def __init__(self, graphs: List[dgl.DGLGraph], window_size: int = 5, stride: int = 1):
        """
        Args:
            graphs: 图序列
            window_size: 历史窗口大小
            stride: 滑动步长
        """
        self.graphs = graphs
        self.window_size = window_size
        self.stride = stride
        self.logger = get_logger()
        
        # 构建时序序列
        self.sequences = self._build_sequences()
        
    def _build_sequences(self) -> List[Dict[str, Any]]:
        """构建(历史图序列, 当前图, 未来图)三元组"""
        sequences = []
        
        for i in range(0, len(self.graphs) - self.window_size, self.stride):
            # 历史窗口
            historical_graphs = self.graphs[i:i+self.window_size-1]
            
            # 当前图
            current_graph = self.graphs[i+self.window_size-1]
            
            # 未来图（用于计算目标）
            future_graph = None
            if i + self.window_size < len(self.graphs):
                future_graph = self.graphs[i+self.window_size]
            
            sequences.append({
                'historical_graphs': historical_graphs,
                'current_graph': current_graph,
                'future_graph': future_graph,
                'time_index': i + self.window_size - 1
            })
            
        self.logger.info(f"构建了 {len(sequences)} 个时序序列")
        return sequences
    
    def get_temporal_sequences(self) -> List[Dict[str, Any]]:
        """获取时序序列数据"""
        return self.sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of range")
        return self.sequences[idx]


def create_temporal_predictor(config: Dict[str, Any]) -> GraphTemporalPredictor:
    """
    根据配置创建时序预测器
    
    Args:
        config: 预测器配置
        
    Returns:
        时序预测器实例
    """
    return GraphTemporalPredictor(
        hidden_dim=config.get('hidden_dim', 64),
        future_steps=config.get('future_steps', 3),
        method=config.get('method', 'graph_rnn')
    ) 