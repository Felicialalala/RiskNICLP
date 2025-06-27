"""
RiskNICLP-5G Main Model

Main model implementation for RiskNICLP-5G with heterogeneous node support
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from .encoders import SharedEncoder
from .decoders import RiskGravityDecoder
from .risk_integration import RiskIntegrator


class RiskNICLPModel(nn.Module):
    """RiskNICLP-5G Main Model with Heterogeneous Node Support"""
    
    def __init__(self, node_feature_dims: Dict[str, int], feature_dim: int, 
                 hidden_dim: int, risk_embed_dim: int, num_risk_bins: int, 
                 num_heads: int, num_layers: int, dropout: float, 
                 temperature: float, gravity_epsilon: float):
        """
        Initialize RiskNICLP-5G model with heterogeneous node support
        
        Args:
            node_feature_dims: Dict mapping node types to their feature dimensions
                               e.g., {'Alert': 9, 'Flow': 17, 'UE': 16}
            feature_dim: Unified projection dimension (45)
            hidden_dim: Hidden dimension (64)
            risk_embed_dim: Risk embedding dimension (16)
            num_risk_bins: Number of risk bins (8)
            num_heads: Number of attention heads (4)
            num_layers: Number of layers (2)
            dropout: Dropout rate
            temperature: Temperature parameter
            gravity_epsilon: Gravity epsilon parameter
        """
        super().__init__()
        
        self.node_feature_dims = node_feature_dims
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.risk_embed_dim = risk_embed_dim
        
        # 异质图节点类型专属投影层 (核心改进)
        self.node_projections = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)  # 轻量dropout
            )
            for node_type, input_dim in node_feature_dims.items()
        })
        
        # Risk embedding
        self.risk_embedding = nn.Embedding(num_risk_bins, risk_embed_dim)
        
        # Shared encoder
        self.shared_encoder = SharedEncoder(
            input_dim=feature_dim + risk_embed_dim,  # 45 + 16 = 61
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Risk integrator
        self.risk_integrator = RiskIntegrator(
            hidden_dim=hidden_dim,
            risk_embed_dim=risk_embed_dim
        )
        
        # Decoder
        self.decoder = RiskGravityDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            temperature=temperature,
            epsilon=gravity_epsilon
        )
        
        # Late fusion layer
        self.late_fusion = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 初始化日志
        print(f"  RiskNICLP模型初始化:")
        print(f"  节点类型投影: {node_feature_dims} → {feature_dim}")
        print(f"  隐藏维度: {hidden_dim}, 风险嵌入: {risk_embed_dim}")
        
    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with heterogeneous node processing
        
        Args:
            batch: Batch data dictionary containing:
                - 'graph': DGL heterogeneous graph (with node features)
                - 'risk_scores': Risk scores tensor
                - 'edge_pairs': Edge pairs for link prediction (optional)
                - 'scale_type': 'fine' or 'coarse'
                
        Returns:
            Tuple of (fine_scores, coarse_scores)
        """
        graph = batch['graph']
        risk_scores = batch['risk_scores']
        edge_pairs = batch.get('edge_pairs', None)
        
        # 1. 异质图节点特征投影 (按类型)
        node_features = self._extract_and_project_features(graph)
        
        # 2. 风险向量化
        risk_bins = self._discretize_risk_scores(risk_scores)
        risk_embeddings = self.risk_embedding(risk_bins)
        
        # 3. 特征-风险集成 (按节点类型)
        integrated_features = self._integrate_features_and_risk(
            node_features, risk_embeddings, graph
        )
        
        # 4. 准备异质图输入
        risk_type_features = self._prepare_hetero_risk(graph, risk_scores)
        
        # 5. 共享编码
        embeddings = self.shared_encoder(
            graph=graph,
            node_features=integrated_features,
            risk_scores=risk_type_features
        )
        
        # 6. 风险增强
        risk_enhanced_embeddings = self.risk_integrator(embeddings, risk_type_features)
        
        # 7. 链路预测
        if edge_pairs is not None:
            link_scores = self._predict_edge_pairs(
                risk_enhanced_embeddings, risk_scores, edge_pairs
            )
        else:
            # 🔧 修复：区分训练和评估模式
            # 检查batch中是否有edge_labels，如果有，说明是训练模式
            if 'edge_labels' in batch and len(batch['edge_labels']) > 0:
                # 训练模式：只计算真实边的分数，确保与edge_labels维度匹配
                if graph.number_of_edges() > 0:
                    # 🔧 修复异质图边处理 - 正确处理多种边类型
                    all_src_nodes = []
                    all_dst_nodes = []
                    
                    if hasattr(graph, 'etypes') and len(graph.etypes) > 1:
                        # 异质图：遍历所有边类型
                        for etype in graph.etypes:
                            try:
                                src, dst = graph.edges(etype=etype)
                                all_src_nodes.append(src)
                                all_dst_nodes.append(dst)
                            except Exception as e:
                                print(f"⚠️ 处理边类型 {etype} 时出错: {e}")
                                continue
                        
                        if all_src_nodes and all_dst_nodes:
                            # 合并所有边类型的边
                            combined_src = torch.cat(all_src_nodes)
                            combined_dst = torch.cat(all_dst_nodes)
                            edge_pairs = torch.stack([combined_src, combined_dst], dim=1)
                        else:
                            # 没有有效边，使用空边对
                            edge_pairs = torch.empty((0, 2), dtype=torch.long)
                    else:
                        # 同质图：直接获取边
                        try:
                            src_nodes, dst_nodes = graph.edges()
                            edge_pairs = torch.stack([src_nodes, dst_nodes], dim=1)
                        except Exception as e:
                            print(f"⚠️ 同质图边获取失败: {e}")
                            edge_pairs = torch.empty((0, 2), dtype=torch.long)
                    
                    if len(edge_pairs) > 0:
                        link_scores = self._predict_edge_pairs(
                            risk_enhanced_embeddings, risk_scores, edge_pairs
                        )
                    else:
                        # 空边情况
                        link_scores = torch.tensor([0.0], requires_grad=True)
                else:
                    # 空图情况
                    link_scores = torch.tensor([0.0], requires_grad=True)
            else:
                # 评估模式：生成所有节点对分数
                link_scores = self._compute_all_pairs_scores(
                    risk_enhanced_embeddings, risk_scores
                )
        
        # 8. 双尺度输出
        scale_type = batch.get('scale_type', ['fine'] * len(batch.get('scale_types', ['fine'])))
        if isinstance(scale_type, list):
            scale_type = scale_type[0] if scale_type else 'fine'
            
        if scale_type == 'fine':
            fine_scores = link_scores
            coarse_scores = torch.zeros_like(link_scores)
        else:
            fine_scores = torch.zeros_like(link_scores)
            coarse_scores = link_scores
        
        return fine_scores, coarse_scores
    
    def _extract_and_project_features(self, graph: 'dgl.DGLGraph') -> Dict[str, torch.Tensor]:
        """按节点类型提取并投影特征"""
        projected_features = {}
        
        for ntype in graph.ntypes:
            if 'feat' in graph.nodes[ntype].data:
                raw_features = graph.nodes[ntype].data['feat']
                
                if ntype in self.node_projections:
                    # 应用类型专属投影
                    projected = self.node_projections[ntype](raw_features)
                    projected_features[ntype] = projected
                else:
                    # 兜底：如果没有对应投影层，使用线性映射
                    input_dim = raw_features.shape[1]
                    fallback_proj = nn.Linear(input_dim, self.feature_dim).to(raw_features.device)
                    projected_features[ntype] = fallback_proj(raw_features)
                    print(f"⚠️ 使用兜底投影: {ntype} ({input_dim} → {self.feature_dim})")
        
        return projected_features
    
    def _integrate_features_and_risk(self, node_features: Dict[str, torch.Tensor], 
                                   risk_embeddings: torch.Tensor, 
                                   graph: 'dgl.DGLGraph') -> Dict[str, torch.Tensor]:
        """按节点类型集成特征和风险嵌入"""
        integrated_features = {}
        start_idx = 0
        
        for ntype in graph.ntypes:
            if ntype in node_features:
                num_nodes = graph.number_of_nodes(ntype)
                
                # 获取该类型节点的特征和风险嵌入
                node_feat = node_features[ntype]  # [num_nodes, feature_dim]
                node_risk = risk_embeddings[start_idx:start_idx + num_nodes]  # [num_nodes, risk_embed_dim]
                
                # 拼接特征和风险嵌入
                integrated = torch.cat([node_feat, node_risk], dim=-1)  # [num_nodes, feature_dim + risk_embed_dim]
                integrated_features[ntype] = integrated
                
                start_idx += num_nodes
        
        return integrated_features
    
    def _discretize_risk_scores(self, risk_scores: torch.Tensor) -> torch.Tensor:
        """离散化风险分数到bin索引"""
        if len(risk_scores) == 0:
            return torch.tensor([], dtype=torch.long)
        
        # 分位数分箱
        num_bins = 8
        try:
            quantiles = torch.quantile(risk_scores, torch.linspace(0, 1, num_bins + 1))
            bin_indices = torch.searchsorted(quantiles[1:], risk_scores)
            bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
        except:
            # 兜底：均匀分箱
            min_val, max_val = risk_scores.min(), risk_scores.max()
            normalized = (risk_scores - min_val) / (max_val - min_val + 1e-8)
            bin_indices = torch.clamp((normalized * num_bins).long(), 0, num_bins - 1)
        
        return bin_indices
    
    def _prepare_hetero_risk(self, graph: 'dgl.DGLGraph', risk_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """按节点类型准备风险分数 - 优先使用图文件中的数据"""
        type_risk = {}
        
        # 优先尝试从图文件中直接获取风险分数
        if not graph.is_homogeneous:
            # 异质图：尝试从节点数据中获取risk_score
            has_saved_risk = True
            for ntype in graph.ntypes:
                node_data = graph.nodes[ntype].data
                if 'risk_score' in node_data and torch.is_tensor(node_data['risk_score']):
                    # 直接使用图文件中保存的风险分数
                    type_risk[ntype] = node_data['risk_score']
                else:
                    has_saved_risk = False
                    break
            
            # 如果所有节点类型都有保存的风险分数，直接返回
            if has_saved_risk:
                return type_risk
        
        # 降级方案：使用传入的risk_scores或默认值
        if len(risk_scores) == 0:
            # 使用默认风险分数
            for ntype in graph.ntypes:
                num_nodes = graph.number_of_nodes(ntype)
                default_risk = {'Alert': 0.8, 'Flow': 0.3, 'UE': 0.1}.get(ntype, 0.2)
                type_risk[ntype] = torch.full((num_nodes,), default_risk, dtype=torch.float32)
        else:
            # 按节点类型分割传入的risk_scores
            start_idx = 0
            for ntype in graph.ntypes:
                num_nodes = graph.number_of_nodes(ntype)
                if num_nodes > 0:
                    type_risk[ntype] = risk_scores[start_idx:start_idx + num_nodes]
                    start_idx += num_nodes
                else:
                    type_risk[ntype] = torch.tensor([])
        
        return type_risk
    
    def _predict_edge_pairs(self, embeddings: Dict[str, torch.Tensor], 
                          risk_scores: torch.Tensor, edge_pairs) -> torch.Tensor:
        """预测指定边对的分数 - 修复异质图索引问题"""
        # 简化实现：将异质嵌入拼接成统一表示
        all_embeddings = []
        node_offset_map = {}  # 记录每种节点类型的偏移量
        current_offset = 0
        
        for ntype in embeddings:
            if len(embeddings[ntype]) > 0:
                node_offset_map[ntype] = current_offset
                all_embeddings.append(embeddings[ntype])
                current_offset += embeddings[ntype].shape[0]
        
        if not all_embeddings:
            return torch.tensor([])
        
        combined_embeddings = torch.cat(all_embeddings, dim=0)
        
        # 🔧 安全的边索引处理
        u_nodes, v_nodes = edge_pairs[:, 0], edge_pairs[:, 1]
        
        # 确保索引在有效范围内
        max_node_idx = combined_embeddings.shape[0] - 1
        u_nodes = torch.clamp(u_nodes, 0, max_node_idx)
        v_nodes = torch.clamp(v_nodes, 0, max_node_idx)
        
        # 检查风险分数维度匹配
        if len(risk_scores) <= max_node_idx:
            # 风险分数不足，扩展到匹配节点数
            extended_risk = torch.full((max_node_idx + 1,), 0.3, dtype=risk_scores.dtype, device=risk_scores.device)
            extended_risk[:len(risk_scores)] = risk_scores
            risk_scores = extended_risk
        
        # 使用解码器预测
        u_embeddings = combined_embeddings[u_nodes]
        v_embeddings = combined_embeddings[v_nodes]
        u_risk = risk_scores[u_nodes].unsqueeze(-1)
        v_risk = risk_scores[v_nodes].unsqueeze(-1)
        
        link_scores = self.decoder(
            embeddings_u=u_embeddings,
            embeddings_v=v_embeddings,
            risk_u=u_risk,
            risk_v=v_risk
        ).squeeze(-1)
        
        return link_scores
    
    def _compute_all_pairs_scores(self, embeddings: Dict[str, torch.Tensor], 
                                risk_scores: torch.Tensor) -> torch.Tensor:
        """计算所有节点对的分数（评估用）- 内存优化版本"""
        # 拼接所有节点嵌入
        all_embeddings = []
        for ntype in embeddings:
            if len(embeddings[ntype]) > 0:
                all_embeddings.append(embeddings[ntype])
        
        if not all_embeddings:
            return torch.tensor([])
        
        combined_embeddings = torch.cat(all_embeddings, dim=0)
        
        # 创建节点对 - 内存优化策略
        num_nodes = combined_embeddings.shape[0]
        if num_nodes == 0:
            return torch.tensor([])
        
        # 🚨 关键修复：严格限制计算规模，避免OOM
        max_nodes_for_full_computation = 1000  # 降低阈值
        max_pairs = 5000  # 最大计算5000个节点对
        
        if num_nodes > max_nodes_for_full_computation:
            # 对于大图，只计算有限的节点对
            # 策略1：随机采样节点
            selected_indices = torch.randperm(num_nodes)[:max_nodes_for_full_computation]
            selected_embeddings = combined_embeddings[selected_indices]
            selected_risk_scores = risk_scores[selected_indices]
            
            # 策略2：计算有限的节点对
            num_selected = len(selected_indices)
            if num_selected * num_selected > max_pairs:
                # 进一步随机采样节点对
                pair_indices = torch.randperm(num_selected * num_selected)[:max_pairs]
                u_indices = pair_indices // num_selected
                v_indices = pair_indices % num_selected
            else:
                u_indices = torch.arange(num_selected).unsqueeze(1).repeat(1, num_selected).flatten()
                v_indices = torch.arange(num_selected).unsqueeze(0).repeat(num_selected, 1).flatten()
            
            u_embeddings = selected_embeddings[u_indices]
            v_embeddings = selected_embeddings[v_indices]
            u_risk = selected_risk_scores[u_indices].unsqueeze(-1)
            v_risk = selected_risk_scores[v_indices].unsqueeze(-1)
            
        else:
            # 小图：计算所有节点对，但仍然限制最大数量
            total_pairs = num_nodes * num_nodes
            if total_pairs > max_pairs:
                indices = torch.randperm(total_pairs)[:max_pairs]
                u_indices = indices // num_nodes
                v_indices = indices % num_nodes
            else:
                u_indices = torch.arange(num_nodes).unsqueeze(1).repeat(1, num_nodes).flatten()
                v_indices = torch.arange(num_nodes).unsqueeze(0).repeat(num_nodes, 1).flatten()
            
            u_embeddings = combined_embeddings[u_indices]
            v_embeddings = combined_embeddings[v_indices]
            u_risk = risk_scores[u_indices].unsqueeze(-1)
            v_risk = risk_scores[v_indices].unsqueeze(-1)
        
        # 批量计算分数 - 进一步分块避免内存峰值
        chunk_size = 1000  # 每次处理1000个节点对
        all_scores = []
        
        for i in range(0, len(u_embeddings), chunk_size):
            end_idx = min(i + chunk_size, len(u_embeddings))
            chunk_u = u_embeddings[i:end_idx]
            chunk_v = v_embeddings[i:end_idx]
            chunk_u_risk = u_risk[i:end_idx]
            chunk_v_risk = v_risk[i:end_idx]
            
            chunk_scores = self.decoder(chunk_u, chunk_v, chunk_u_risk, chunk_v_risk).squeeze(-1)
            all_scores.append(chunk_scores)
        
        if all_scores:
            scores = torch.cat(all_scores, dim=0)
        else:
            scores = torch.tensor([0.0])  # 兜底分数
        
        return scores

    def get_node_embeddings(self, graph: 'dgl.DGLGraph', risk_scores: torch.Tensor) -> torch.Tensor:
        """
        获取节点嵌入表示，用于链路预测
        
        Args:
            graph: DGL异质图
            risk_scores: 风险分数向量
            
        Returns:
            torch.Tensor: 所有节点的嵌入表示 [total_nodes, embedding_dim]
        """
        # 1. 提取并投影各类型节点特征
        projected_features = self._extract_and_project_features(graph)
        
        # 2. 获取风险嵌入
        risk_bins = self._discretize_risk_scores(risk_scores)
        risk_embeddings = self.risk_embedding(risk_bins)
        
        # 3. 按节点类型集成特征和风险
        integrated_features = self._integrate_features_and_risk(
            projected_features, risk_embeddings, graph
        )
        
        # 4. 通过图编码器获取最终嵌入
        final_embeddings = self.shared_encoder(graph, integrated_features)
        
        # 5. 拼接所有节点类型的嵌入为统一张量
        all_embeddings = []
        for ntype in graph.ntypes:
            if ntype in final_embeddings and len(final_embeddings[ntype]) > 0:
                all_embeddings.append(final_embeddings[ntype])
        
        if all_embeddings:
            combined_embeddings = torch.cat(all_embeddings, dim=0)
        else:
            # 兜底：创建零嵌入
            total_nodes = sum([graph.number_of_nodes(ntype) for ntype in graph.ntypes])
            combined_embeddings = torch.zeros(total_nodes, self.hidden_dim, device=graph.device)
        
        return combined_embeddings 