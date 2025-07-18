"""
GraphSAINT Baseline for RiskNICLP-5G Attack Link Prediction

Implementation of GraphSAINT (Graph Sampling Based Inductive Learning Method)
as a strong baseline for attack propagation prediction in 5G networks.

Reference: GraphSAINT: Graph Sampling Based Inductive Learning Method (ICLR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import GraphConv, SAGEConv
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GraphSAINTSampler:
    """
    GraphSAINT sampling strategy for large-scale heterogeneous graphs
    Supports node sampling, edge sampling, and random walk sampling
    """

    def __init__(
        self,
        sampling_method: str = "node",  # "node", "edge", "rw"
        sample_coverage: float = 0.1,   # Fraction of nodes/edges to sample
        num_walks: int = 1000,          # For random walk sampling
        walk_length: int = 10,          # For random walk sampling
        batch_size: int = 1000,         # Subgraph batch size
        num_workers: int = 4
    ):
        self.sampling_method = sampling_method
        self.sample_coverage = sample_coverage
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def node_sampling(self, graph: dgl.DGLGraph) -> List[dgl.DGLGraph]:
        """Node-based sampling for GraphSAINT"""
        subgraphs = []
        total_nodes = graph.number_of_nodes()
        sample_size = int(total_nodes * self.sample_coverage)

        # Generate multiple subgraphs for training
        num_subgraphs = max(1, total_nodes // self.batch_size)

        for _ in range(num_subgraphs):
            # Random node sampling
            sampled_nodes = torch.randperm(total_nodes)[:sample_size]

            # Extract subgraph
            subgraph = dgl.node_subgraph(graph, sampled_nodes)

            # Ensure subgraph has enough nodes for training
            if subgraph.number_of_nodes() > 10:
                subgraphs.append(subgraph)

        return subgraphs

    def edge_sampling(self, graph: dgl.DGLGraph) -> List[dgl.DGLGraph]:
        """Edge-based sampling for GraphSAINT"""
        subgraphs = []
        total_edges = graph.number_of_edges()
        sample_size = int(total_edges * self.sample_coverage)

        num_subgraphs = max(1, total_edges // self.batch_size)

        for _ in range(num_subgraphs):
            # Random edge sampling
            edge_ids = torch.randperm(total_edges)[:sample_size]

            # Get nodes from sampled edges
            src, dst = graph.edges()
            sampled_src = src[edge_ids]
            sampled_dst = dst[edge_ids]
            sampled_nodes = torch.unique(torch.cat([sampled_src, sampled_dst]))

            # Extract subgraph
            subgraph = dgl.node_subgraph(graph, sampled_nodes)

            if subgraph.number_of_nodes() > 10:
                subgraphs.append(subgraph)

        return subgraphs

    def random_walk_sampling(self, graph: dgl.DGLGraph) -> List[dgl.DGLGraph]:
        """Random walk sampling for GraphSAINT"""
        subgraphs = []
        total_nodes = graph.number_of_nodes()

        # Start random walks from random nodes
        start_nodes = torch.randint(0, total_nodes, (self.num_walks,))

        # Perform random walks
        traces, _ = dgl.sampling.random_walk(
            graph, start_nodes, length=self.walk_length
        )

        # Extract unique nodes from walks
        sampled_nodes = torch.unique(traces.flatten())
        sampled_nodes = sampled_nodes[sampled_nodes >= 0]  # Remove padding

        # Create subgraph
        if len(sampled_nodes) > 10:
            subgraph = dgl.node_subgraph(graph, sampled_nodes)
            subgraphs.append(subgraph)

        return subgraphs

    def sample(self, graph: dgl.DGLGraph) -> List[dgl.DGLGraph]:
        """Main sampling interface"""
        if self.sampling_method == "node":
            return self.node_sampling(graph)
        elif self.sampling_method == "edge":
            return self.edge_sampling(graph)
        elif self.sampling_method == "rw":
            return self.random_walk_sampling(graph)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")


class GraphSAINTEncoder(nn.Module):
    """
    GraphSAINT encoder with multi-layer GraphConv
    Supports heterogeneous node types through projection
    """

    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()

        self.node_feature_dims = node_feature_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # 统一特征维度处理
        self.max_dim = max(node_feature_dims.values())

        # Node type projections - 统一输入维度
        self.node_projections = nn.ModuleDict()
        for node_type, input_dim in node_feature_dims.items():
            self.node_projections[node_type] = nn.Linear(self.max_dim, hidden_dim)

        # GraphConv layers with allow_zero_in_degree=True
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                GraphConv(hidden_dim, hidden_dim, activation=F.relu, allow_zero_in_degree=True)
            )

        # Dropout and normalization
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through GraphSAINT encoder

        Args:
            graph: DGL graph (can be subgraph from sampling)
            node_features: Dict of node features by type

        Returns:
            Node embeddings [N, hidden_dim]
        """
        # 处理异质图：转换为同质图并统一特征
        if hasattr(graph, 'ntypes') and len(graph.ntypes) > 1:
            # 异质图转同质图
            homo_graph = dgl.to_homogeneous(graph)

            # 统一特征维度并投影
            unified_features = []

            # 按照节点类型顺序处理特征
            for node_type in graph.ntypes:
                if node_type in node_features:
                    feat = node_features[node_type]
                    # 填充到最大维度
                    if feat.shape[1] < self.max_dim:
                        feat = torch.nn.functional.pad(feat, (0, self.max_dim - feat.shape[1]))
                    # 投影到隐藏维度
                    projected = self.node_projections[node_type](feat)
                    unified_features.append(projected)

            h = torch.cat(unified_features, dim=0)
            working_graph = homo_graph
        else:
            # 同质图处理
            # 假设只有一种节点类型
            node_type = list(node_features.keys())[0]
            feat = node_features[node_type]
            # 填充到最大维度
            if feat.shape[1] < self.max_dim:
                feat = torch.nn.functional.pad(feat, (0, self.max_dim - feat.shape[1]))
            h = self.node_projections[node_type](feat)
            working_graph = graph

        # 添加自环以避免0入度问题
        working_graph = dgl.add_self_loop(working_graph)

        # Multi-layer graph convolution
        for i, conv_layer in enumerate(self.conv_layers):
            h_new = conv_layer(working_graph, h)
            h_new = self.layer_norms[i](h_new)
            h_new = self.dropout_layer(h_new)

            # Residual connection
            h = h + h_new

        return h


class GraphSAINTDecoder(nn.Module):
    """
    Enhanced decoder for link prediction with multiple scoring methods
    """

    def __init__(self, hidden_dim: int, temperature: float = 1.0, decoder_type: str = "mlp"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.decoder_type = decoder_type

        if decoder_type == "mlp":
            # MLP decoder for better expressiveness
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        elif decoder_type == "bilinear":
            # Bilinear decoder
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(
        self,
        embeddings: torch.Tensor,
        edge_pairs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute link prediction scores

        Args:
            embeddings: Node embeddings [N, hidden_dim]
            edge_pairs: Edge pairs to predict [E, 2]

        Returns:
            Link prediction scores [E]
        """
        src_embeddings = embeddings[edge_pairs[:, 0]]
        dst_embeddings = embeddings[edge_pairs[:, 1]]

        if self.decoder_type == "dot":
            # Dot product similarity
            scores = torch.sum(src_embeddings * dst_embeddings, dim=1)
            scores = scores / self.temperature
            return torch.sigmoid(scores)

        elif self.decoder_type == "mlp":
            # MLP decoder
            combined = torch.cat([src_embeddings, dst_embeddings], dim=1)
            scores = self.mlp(combined).squeeze(-1)
            return torch.sigmoid(scores)

        elif self.decoder_type == "bilinear":
            # Bilinear decoder
            scores = self.bilinear(src_embeddings, dst_embeddings).squeeze(-1)
            return torch.sigmoid(scores)

        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")


class GraphSAINTBaseline(nn.Module):
    """
    Complete GraphSAINT baseline model for attack link prediction
    """

    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        temperature: float = 1.0,
        sampling_config: Optional[Dict] = None
    ):
        super().__init__()

        self.node_feature_dims = node_feature_dims
        self.hidden_dim = hidden_dim

        # Initialize sampler
        if sampling_config is None:
            sampling_config = {
                "sampling_method": "node",
                "sample_coverage": 0.1,
                "batch_size": 1000
            }
        self.sampler = GraphSAINTSampler(**sampling_config)

        # Initialize encoder and decoder
        self.encoder = GraphSAINTEncoder(
            node_feature_dims=node_feature_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = GraphSAINTDecoder(
            hidden_dim=hidden_dim,
            temperature=temperature,
            decoder_type="mlp"  # 使用更强的MLP解码器
        )

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_features: Dict[str, torch.Tensor],
        edge_pairs: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for link prediction

        Args:
            graph: Full graph or subgraph
            node_features: Node features by type
            edge_pairs: Edge pairs to predict

        Returns:
            Link prediction scores
        """
        # Encode nodes
        embeddings = self.encoder(graph, node_features)

        # Decode links
        scores = self.decoder(embeddings, edge_pairs)

        return scores

    def training_step(
        self,
        graph: dgl.DGLGraph,
        node_features: Dict[str, torch.Tensor],
        positive_edges: torch.Tensor,
        negative_edges: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Training step with GraphSAINT sampling

        Returns:
            loss: Training loss
            metrics: Training metrics
        """
        # Sample subgraphs
        subgraphs = self.sampler.sample(graph)

        total_loss = 0.0
        num_subgraphs = len(subgraphs)

        if num_subgraphs == 0:
            # Fallback: use full graph
            subgraphs = [graph]
            num_subgraphs = 1

        for subgraph in subgraphs:
            # Map edges to subgraph nodes
            subgraph_pos_edges = self._map_edges_to_subgraph(
                positive_edges, subgraph, graph
            )
            subgraph_neg_edges = self._map_edges_to_subgraph(
                negative_edges, subgraph, graph
            )

            if len(subgraph_pos_edges) == 0 or len(subgraph_neg_edges) == 0:
                continue

            # Extract subgraph node features
            subgraph_features = self._extract_subgraph_features(
                node_features, subgraph
            )

            # Forward pass
            pos_scores = self.forward(subgraph, subgraph_features, subgraph_pos_edges)
            neg_scores = self.forward(subgraph, subgraph_features, subgraph_neg_edges)

            # Compute loss
            pos_loss = F.binary_cross_entropy(
                pos_scores, torch.ones_like(pos_scores)
            )
            neg_loss = F.binary_cross_entropy(
                neg_scores, torch.zeros_like(neg_scores)
            )

            subgraph_loss = (pos_loss + neg_loss) / 2
            total_loss += subgraph_loss

        avg_loss = total_loss / num_subgraphs

        metrics = {
            "train_loss": avg_loss.item(),
            "num_subgraphs": num_subgraphs
        }

        return avg_loss, metrics

    def _map_edges_to_subgraph(
        self,
        edges: torch.Tensor,
        subgraph: dgl.DGLGraph,
        full_graph: dgl.DGLGraph
    ) -> torch.Tensor:
        """Map edges from full graph to subgraph node IDs"""
        if len(edges) == 0:
            return torch.empty(0, 2, dtype=torch.long)

        # Get subgraph node mapping
        subgraph_nodes = subgraph.ndata[dgl.NID] if dgl.NID in subgraph.ndata else torch.arange(subgraph.number_of_nodes())

        # Create mapping from full graph node ID to subgraph node ID
        node_mapping = {node.item(): i for i, node in enumerate(subgraph_nodes)}

        # Filter and map edges
        mapped_edges = []
        for edge in edges:
            src, dst = edge.tolist()
            if src in node_mapping and dst in node_mapping:
                mapped_edges.append([node_mapping[src], node_mapping[dst]])

        return torch.tensor(mapped_edges, dtype=torch.long) if mapped_edges else torch.empty(0, 2, dtype=torch.long)

    def _extract_subgraph_features(
        self,
        node_features: Dict[str, torch.Tensor],
        subgraph: dgl.DGLGraph
    ) -> Dict[str, torch.Tensor]:
        """Extract node features for subgraph"""
        subgraph_features = {}

        # Get subgraph node mapping
        subgraph_nodes = subgraph.ndata[dgl.NID] if dgl.NID in subgraph.ndata else torch.arange(subgraph.number_of_nodes())

        for node_type, features in node_features.items():
            if len(features) > 0:
                # Map features to subgraph nodes
                # This is a simplified mapping - in practice, you'd need proper node type tracking
                max_idx = min(len(features) - 1, subgraph_nodes.max().item())
                valid_indices = subgraph_nodes[subgraph_nodes <= max_idx]

                if len(valid_indices) > 0:
                    subgraph_features[node_type] = features[valid_indices]

        return subgraph_features