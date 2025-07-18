"""
TGAT Baseline for RiskNICLP-5G Attack Link Prediction

Implementation of Temporal Graph Attention Network (TGAT) as a baseline
for temporal attack propagation prediction in 5G networks.

Reference: Inductive Representation Learning on Temporal Graphs (ICLR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import dgl
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)


class TimeEncoder(nn.Module):
    """
    Time encoding module for TGAT
    Encodes time differences using learnable basis functions
    """

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float().reshape(self.time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time differences

        Args:
            t: Time differences [batch_size, 1] or [batch_size]

        Returns:
            Time encodings [batch_size, time_dim]
        """
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)

        # Apply time encoding
        output = torch.cos(self.w(t))
        return output


class TemporalAttentionLayer(nn.Module):
    """
    Temporal attention layer for TGAT
    Computes attention over temporal neighbors
    """

    def __init__(
        self,
        node_dim: int,
        time_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.node_dim = node_dim
        self.time_dim = time_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.query_proj = nn.Linear(node_dim, node_dim)
        self.key_proj = nn.Linear(node_dim + time_dim, node_dim)
        self.value_proj = nn.Linear(node_dim + time_dim, node_dim)

        # Output projection
        self.out_proj = nn.Linear(node_dim, node_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer norm
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        query_node: torch.Tensor,
        neighbor_nodes: torch.Tensor,
        time_encodings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Temporal attention computation

        Args:
            query_node: Query node features [batch_size, node_dim]
            neighbor_nodes: Neighbor node features [batch_size, num_neighbors, node_dim]
            time_encodings: Time encodings [batch_size, num_neighbors, time_dim]
            mask: Attention mask [batch_size, num_neighbors]

        Returns:
            Updated node features [batch_size, node_dim]
        """
        batch_size, num_neighbors, _ = neighbor_nodes.shape

        # Prepare query
        query = self.query_proj(query_node)  # [batch_size, node_dim]
        query = query.view(batch_size, 1, self.num_heads, self.head_dim)

        # Prepare key and value (concatenate node features with time encodings)
        neighbor_time_concat = torch.cat([neighbor_nodes, time_encodings], dim=-1)
        key = self.key_proj(neighbor_time_concat)  # [batch_size, num_neighbors, node_dim]
        value = self.value_proj(neighbor_time_concat)  # [batch_size, num_neighbors, node_dim]

        key = key.view(batch_size, num_neighbors, self.num_heads, self.head_dim)
        value = value.view(batch_size, num_neighbors, self.num_heads, self.head_dim)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.squeeze(2)  # [batch_size, num_heads, num_neighbors]

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights.unsqueeze(-2), value)  # [batch_size, num_heads, 1, head_dim]
        attended = attended.squeeze(-2)  # [batch_size, num_heads, head_dim]
        attended = attended.view(batch_size, -1)  # [batch_size, node_dim]

        # Output projection
        output = self.out_proj(attended)

        # Residual connection and layer norm
        output = self.layer_norm(output + query_node)

        return output


class TGATLayer(nn.Module):
    """
    Single TGAT layer combining temporal attention and message passing
    """

    def __init__(
        self,
        node_dim: int,
        time_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_neighbors: int = 20
    ):
        super().__init__()

        self.node_dim = node_dim
        self.time_dim = time_dim
        self.num_neighbors = num_neighbors

        # Temporal attention layer
        self.temporal_attention = TemporalAttentionLayer(
            node_dim=node_dim,
            time_dim=time_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # MLP for final transformation
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 2, node_dim)
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        time_features: torch.Tensor,
        graph: dgl.DGLGraph
    ) -> torch.Tensor:
        """
        Forward pass through TGAT layer

        Args:
            node_features: Node features [num_nodes, node_dim]
            edge_features: Edge features [num_edges, edge_dim] (can be None)
            time_features: Time features [num_edges, time_dim]
            graph: DGL graph

        Returns:
            Updated node features [num_nodes, node_dim]
        """
        num_nodes = node_features.shape[0]
        updated_features = []

        for node_id in range(num_nodes):
            # Get neighbors and their features
            try:
                # 对于异质图，先转换为同质图
                if hasattr(graph, 'ntypes') and len(graph.ntypes) > 1:
                    homo_graph = dgl.to_homogeneous(graph)
                    neighbors = homo_graph.successors(node_id)
                else:
                    neighbors = graph.successors(node_id)
            except:
                # Fallback: 使用边信息获取邻居
                src, dst = graph.edges()
                neighbors = dst[src == node_id]
                if len(neighbors) == 0:
                    neighbors = src[dst == node_id]

            if len(neighbors) == 0:
                # No neighbors, keep original features
                updated_features.append(node_features[node_id])
                continue

            # Sample neighbors if too many
            if len(neighbors) > self.num_neighbors:
                sampled_indices = torch.randperm(len(neighbors))[:self.num_neighbors]
                neighbors = neighbors[sampled_indices]

            # Get neighbor features and time encodings
            neighbor_features = node_features[neighbors]  # [num_neighbors, node_dim]

            # Get edge information for time encoding
            try:
                edge_ids = graph.edge_ids(node_id, neighbors)
                neighbor_time_features = time_features[edge_ids]  # [num_neighbors, time_dim]
            except:
                # Fallback: use zero time features
                neighbor_time_features = torch.zeros(len(neighbors), self.time_dim, device=node_features.device)

            # Pad if necessary
            if len(neighbors) < self.num_neighbors:
                pad_size = self.num_neighbors - len(neighbors)
                neighbor_features = F.pad(neighbor_features, (0, 0, 0, pad_size))
                neighbor_time_features = F.pad(neighbor_time_features, (0, 0, 0, pad_size))

                # Create mask
                mask = torch.zeros(self.num_neighbors, dtype=torch.bool, device=node_features.device)
                mask[:len(neighbors)] = True
            else:
                mask = torch.ones(self.num_neighbors, dtype=torch.bool, device=node_features.device)

            # Apply temporal attention
            query_node = node_features[node_id].unsqueeze(0)  # [1, node_dim]
            neighbor_features = neighbor_features.unsqueeze(0)  # [1, num_neighbors, node_dim]
            neighbor_time_features = neighbor_time_features.unsqueeze(0)  # [1, num_neighbors, time_dim]
            mask = mask.unsqueeze(0)  # [1, num_neighbors]

            attended_features = self.temporal_attention(
                query_node, neighbor_features, neighbor_time_features, mask
            )

            updated_features.append(attended_features.squeeze(0))

        # Stack all updated features
        updated_node_features = torch.stack(updated_features, dim=0)

        # Apply MLP and residual connection
        mlp_output = self.mlp(updated_node_features)
        output = self.layer_norm(mlp_output + node_features)

        return output


class TGATBaseline(nn.Module):
    """
    Complete TGAT baseline model for temporal attack link prediction
    """

    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        hidden_dim: int = 128,
        time_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_neighbors: int = 20
    ):
        super().__init__()

        self.node_feature_dims = node_feature_dims
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_layers = num_layers

        # Node type projections
        self.node_projections = nn.ModuleDict()
        for node_type, input_dim in node_feature_dims.items():
            self.node_projections[node_type] = nn.Linear(input_dim, hidden_dim)

        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)

        # TGAT layers
        self.tgat_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = TGATLayer(
                node_dim=hidden_dim,
                time_dim=time_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_neighbors=num_neighbors
            )
            self.tgat_layers.append(layer)

        # Link prediction decoder
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_features: Dict[str, torch.Tensor],
        edge_times: torch.Tensor,
        edge_pairs: torch.Tensor,
        current_time: float = None
    ) -> torch.Tensor:
        """
        Forward pass for temporal link prediction

        Args:
            graph: DGL graph with temporal edges
            node_features: Node features by type
            edge_times: Edge timestamps [num_edges]
            edge_pairs: Edge pairs to predict [num_pairs, 2]
            current_time: Current timestamp for time encoding

        Returns:
            Link prediction scores [num_pairs]
        """
        # Project heterogeneous node features
        projected_features = []
        for node_type, features in node_features.items():
            if node_type in self.node_projections and len(features) > 0:
                projected = self.node_projections[node_type](features)
                projected_features.append(projected)

        if not projected_features:
            # Fallback: create zero features
            total_nodes = graph.number_of_nodes()
            node_embeddings = torch.zeros(total_nodes, self.hidden_dim, device=graph.device)
        else:
            node_embeddings = torch.cat(projected_features, dim=0)

        # Encode edge times
        if current_time is None:
            current_time = edge_times.max().item()

        time_diffs = current_time - edge_times
        time_encodings = self.time_encoder(time_diffs)

        # Apply TGAT layers
        for tgat_layer in self.tgat_layers:
            node_embeddings = tgat_layer(
                node_features=node_embeddings,
                edge_features=None,  # Not used in this simplified version
                time_features=time_encodings,
                graph=graph
            )

        # Predict links
        src_embeddings = node_embeddings[edge_pairs[:, 0]]
        dst_embeddings = node_embeddings[edge_pairs[:, 1]]

        # Concatenate source and destination embeddings
        pair_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

        # Predict link probabilities
        link_scores = self.link_predictor(pair_embeddings).squeeze(-1)

        return link_scores

    def training_step(
        self,
        graph: dgl.DGLGraph,
        node_features: Dict[str, torch.Tensor],
        edge_times: torch.Tensor,
        positive_edges: torch.Tensor,
        negative_edges: torch.Tensor,
        current_time: float = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Training step for TGAT

        Returns:
            loss: Training loss
            metrics: Training metrics
        """
        # Forward pass for positive edges
        pos_scores = self.forward(
            graph, node_features, edge_times, positive_edges, current_time
        )

        # Forward pass for negative edges
        neg_scores = self.forward(
            graph, node_features, edge_times, negative_edges, current_time
        )

        # Compute loss
        pos_loss = F.binary_cross_entropy(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy(
            neg_scores, torch.zeros_like(neg_scores)
        )

        total_loss = (pos_loss + neg_loss) / 2

        # Compute metrics
        with torch.no_grad():
            pos_acc = (pos_scores > 0.5).float().mean()
            neg_acc = (neg_scores <= 0.5).float().mean()
            overall_acc = (pos_acc + neg_acc) / 2

        metrics = {
            "train_loss": total_loss.item(),
            "pos_accuracy": pos_acc.item(),
            "neg_accuracy": neg_acc.item(),
            "overall_accuracy": overall_acc.item()
        }

        return total_loss, metrics