"""
Shared Encoder Architecture for RiskNICLP-5G

Implements:
- SharedEncoder: Parameter sharing between fine/coarse scales
- UEEncoder: GAT-based UE↔Flow encoder with risk-bias attention
- FlowEncoder: GraphSAGE + GRU for Flow→Alert and temporal modeling
- Risk-aware attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import GATConv, SAGEConv
import warnings

from ..common.logger import get_logger


class RiskBiasGATConv(nn.Module):
    """GAT with risk-bias attention mechanism"""
    
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_heads: int = 4,
        feat_drop: float = 0.0,
        attn_drop: float = 0.0,
        residual: bool = True,
        allow_zero_in_degree: bool = True
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.out_feats = out_feats
        
        # Base GAT layer
        self.gat = GATConv(
            in_feats=in_feats,
            out_feats=out_feats,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=residual,
            allow_zero_in_degree=allow_zero_in_degree
        )
        
        # Risk bias parameters
        self.risk_bias_transform = nn.Linear(1, num_heads)
        self.beta_1 = nn.Parameter(torch.tensor(1.0))  # UE→Flow bias weight
        self.beta_2 = nn.Parameter(torch.tensor(0.5))  # Flow→UE bias weight
        
        # Freeze risk parameters initially for progressive training
        self.freeze_risk_params()
        
    def freeze_risk_params(self):
        """Freeze risk-related parameters"""
        self.risk_bias_transform.requires_grad_(False)
        self.beta_1.requires_grad_(False)
        self.beta_2.requires_grad_(False)
        
    def unfreeze_risk_params(self):
        """Unfreeze risk-related parameters"""
        self.risk_bias_transform.requires_grad_(True)
        self.beta_1.requires_grad_(True)
        self.beta_2.requires_grad_(True)
    
    def forward(
        self, 
        graph: dgl.DGLGraph, 
        feat: torch.Tensor,
        risk_scores: Optional[torch.Tensor] = None,
        edge_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        Forward pass with risk-bias attention
        
        Args:
            graph: DGL graph
            feat: Node features [N, in_feats]
            risk_scores: Risk scores [N, 1] 
            edge_type: Edge type for beta selection
            
        Returns:
            Output features [N, num_heads * out_feats] or [N, out_feats]
        """
        
        # Standard GAT forward
        if risk_scores is None:
            return self.gat(graph, feat)
        
        # Compute risk bias
        risk_bias = self.risk_bias_transform(risk_scores)  # [N, num_heads]
        
        # Select beta based on edge type
        if edge_type == "ue_to_flow":
            beta = self.beta_1
        elif edge_type == "flow_to_ue":
            beta = self.beta_2
        else:
            beta = 1.0
            
        # Apply risk bias to attention weights
        # This is a simplified implementation - in practice would modify GAT attention computation
        h = self.gat(graph, feat)
        
        # Apply risk bias as a multiplicative factor
        if h.dim() == 3:  # Multi-head output [N, num_heads, out_feats]
            risk_bias_expanded = risk_bias.unsqueeze(-1)  # [N, num_heads, 1]
            h = h * (1 + beta * risk_bias_expanded)
            h = h.mean(dim=1)  # Average heads: [N, out_feats]
        else:  # Single head output [N, out_feats]
            h = h * (1 + beta * risk_bias.mean(dim=-1, keepdim=True))
            
        return h


class UEEncoder(nn.Module):
    """UE encoder using risk-bias GAT for UE↔Flow interactions"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        residual: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers with risk bias
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = RiskBiasGATConv(
                in_feats=hidden_dim,
                out_feats=hidden_dim,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=residual
            )
            self.gat_layers.append(layer)
        
        # Output normalization
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.logger = get_logger()
        
    def freeze_risk_params(self):
        """Freeze risk parameters in all layers"""
        for layer in self.gat_layers:
            layer.freeze_risk_params()
            
    def unfreeze_risk_params(self):
        """Unfreeze risk parameters in all layers"""
        for layer in self.gat_layers:
            layer.unfreeze_risk_params()
    
    def forward(
        self,
        graph: dgl.DGLGraph,
        ue_features: torch.Tensor,
        flow_features: torch.Tensor,
        ue_risk: Optional[torch.Tensor] = None,
        flow_risk: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode UE and Flow nodes with risk-aware attention
        
        Args:
            graph: Heterogeneous graph with UE↔Flow edges
            ue_features: UE node features [N_ue, input_dim]
            flow_features: Flow node features [N_flow, input_dim]
            ue_risk: UE risk scores [N_ue, 1]
            flow_risk: Flow risk scores [N_flow, 1]
            
        Returns:
            Tuple of (ue_embeddings, flow_embeddings)
        """
        
        # Project input features
        ue_h = self.input_proj(ue_features)
        flow_h = self.input_proj(flow_features)
        
        # Combine node features for homogeneous processing
        all_features = torch.cat([ue_h, flow_h], dim=0)
        
        # Combine risk scores
        if ue_risk is not None and flow_risk is not None:
            all_risk = torch.cat([ue_risk, flow_risk], dim=0)
        else:
            all_risk = None
        
        # Apply GAT layers
        h = all_features
        for i, gat_layer in enumerate(self.gat_layers):
            h_new = gat_layer(graph, h, all_risk)
            h_new = self.dropout(h_new)
            
            # Residual connection
            if h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new
                
            h = self.norm(h)
        
        # Split back to UE and Flow embeddings
        n_ue = ue_features.shape[0]
        ue_embeddings = h[:n_ue]
        flow_embeddings = h[n_ue:]
        
        return ue_embeddings, flow_embeddings


class FlowEncoder(nn.Module):
    """Flow encoder using GraphSAGE + GRU for Flow→Alert and temporal modeling"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        aggregator_type: str = "mean",
        dropout: float = 0.1,
        temporal_dim: int = 32
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temporal_dim = temporal_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = SAGEConv(
                in_feats=hidden_dim,
                out_feats=hidden_dim,
                aggregator_type=aggregator_type,
                #dropout=dropout 在forward方法中应用dropout
            )
            self.sage_layers.append(layer)
        
        # Temporal modeling with GRU
        self.temporal_proj = nn.Linear(hidden_dim, temporal_dim)
        self.gru = nn.GRU(
            input_size=temporal_dim,
            hidden_size=temporal_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Time difference encoding (log-binned)
        self.time_bins = 20
        self.time_embedding = nn.Embedding(self.time_bins, temporal_dim // 4)
        
        # Output layers
        self.norm = nn.LayerNorm(hidden_dim + temporal_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.logger = get_logger()
    
    def encode_time_differences(self, time_diffs: torch.Tensor) -> torch.Tensor:
        """
        Encode time differences using log-binning
        
        Args:
            time_diffs: Time differences in seconds [N]
            
        Returns:
            Time embeddings [N, temporal_dim // 4]
        """
        # Log-binning for time differences
        log_diffs = torch.log1p(torch.clamp(time_diffs, min=0))
        
        # Quantize to bins
        max_log_diff = 10.0  # ~22000 seconds
        bin_indices = torch.clamp(
            (log_diffs / max_log_diff * self.time_bins).long(),
            0, self.time_bins - 1
        )
        
        return self.time_embedding(bin_indices)
    
    def forward(
        self,
        graph: dgl.DGLGraph,
        flow_features: torch.Tensor,
        alert_features: torch.Tensor,
        flow_times: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode Flow and Alert nodes with temporal modeling
        
        Args:
            graph: Heterogeneous graph with Flow→Alert edges
            flow_features: Flow node features [N_flow, input_dim]
            alert_features: Alert node features [N_alert, input_dim]
            flow_times: Flow timestamps [N_flow]
            sequence_lengths: Sequence lengths for GRU [batch_size]
            
        Returns:
            Tuple of (flow_embeddings, alert_embeddings)
        """
        
        # Project input features
        flow_h = self.input_proj(flow_features)
        alert_h = self.input_proj(alert_features)
        
        # Combine for homogeneous processing
        all_features = torch.cat([flow_h, alert_h], dim=0)
        
        # Apply GraphSAGE layers
        h = all_features
        for sage_layer in self.sage_layers:
            h = sage_layer(graph, h)
            h = F.relu(h)
            h = self.dropout(h)
        
        # Split back
        n_flow = flow_features.shape[0]
        flow_embeddings = h[:n_flow]
        alert_embeddings = h[n_flow:]
        
        # Temporal modeling for flows
        if flow_times is not None:
            # Time encoding
            if len(flow_times) > 1:
                time_diffs = flow_times[1:] - flow_times[:-1]
                time_diffs = torch.cat([torch.zeros(1, device=flow_times.device), time_diffs])
            else:
                time_diffs = torch.zeros_like(flow_times)
            
            time_embeds = self.encode_time_differences(time_diffs)
            
            # Prepare for GRU
            temporal_input = self.temporal_proj(flow_embeddings)  # [N_flow, temporal_dim]
            
            # Add time embeddings
            if time_embeds.shape[-1] == temporal_input.shape[-1] // 4:
                # Repeat time embeddings to match temporal_dim
                time_embeds = time_embeds.repeat(1, 4)[:, :temporal_input.shape[-1]]
            
            temporal_input = temporal_input + time_embeds
            
            # GRU processing (simplified - assumes single sequence)
            temporal_input = temporal_input.unsqueeze(0)  # [1, N_flow, temporal_dim]
            temporal_output, _ = self.gru(temporal_input)
            temporal_output = temporal_output.squeeze(0)  # [N_flow, temporal_dim]
            
            # Combine spatial and temporal features
            flow_embeddings = torch.cat([flow_embeddings, temporal_output], dim=-1)
            flow_embeddings = self.norm(flow_embeddings)
        
        return flow_embeddings, alert_embeddings


class SharedEncoder(nn.Module):
    """Shared encoder architecture with parameter sharing between scales"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        temporal_dim: int = 32
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Shared encoders
        self.ue_encoder = UEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.flow_encoder = FlowEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            temporal_dim=temporal_dim
        )
        
        # Output dimension
        self.output_dim = hidden_dim
        self.flow_output_dim = hidden_dim + temporal_dim
        
        self.logger = get_logger()
        
    def freeze_risk_params(self):
        """Freeze risk parameters for progressive training"""
        self.ue_encoder.freeze_risk_params()
        
    def unfreeze_risk_params(self):
        """Unfreeze risk parameters for progressive training"""
        self.ue_encoder.unfreeze_risk_params()
    
    def forward(
        self,
        graph: dgl.DGLGraph,
        node_features: Dict[str, torch.Tensor],
        risk_scores: Optional[Dict[str, torch.Tensor]] = None,
        temporal_info: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through shared encoders
        
        Args:
            graph: Heterogeneous graph
            node_features: Dict of node type -> features
            risk_scores: Dict of node type -> risk scores  
            temporal_info: Dict with temporal information
            
        Returns:
            Dict of node type -> embeddings
        """
        
        embeddings = {}
        
        # UE and Flow encoding
        if 'UE' in node_features and 'Flow' in node_features:
            ue_risk = risk_scores.get('UE') if risk_scores else None
            flow_risk = risk_scores.get('Flow') if risk_scores else None
            
            ue_emb, flow_emb_ue = self.ue_encoder(
                graph=graph,
                ue_features=node_features['UE'],
                flow_features=node_features['Flow'],
                ue_risk=ue_risk,
                flow_risk=flow_risk
            )
            
            embeddings['UE'] = ue_emb
            embeddings['Flow_UE'] = flow_emb_ue
        
        # Flow and Alert encoding
        if 'Flow' in node_features and 'Alert' in node_features:
            flow_times = temporal_info.get('flow_times') if temporal_info else None
            
            flow_emb_alert, alert_emb = self.flow_encoder(
                graph=graph,
                flow_features=node_features['Flow'],
                alert_features=node_features['Alert'],
                flow_times=flow_times
            )
            
            # Combine Flow embeddings if both exist
            if 'Flow_UE' in embeddings:
                # Simple concatenation or learned fusion
                embeddings['Flow'] = torch.cat([
                    embeddings['Flow_UE'], 
                    flow_emb_alert
                ], dim=-1)
            else:
                embeddings['Flow'] = flow_emb_alert
                
            embeddings['Alert'] = alert_emb
        
        return embeddings 