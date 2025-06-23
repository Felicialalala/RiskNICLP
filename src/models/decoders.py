"""
RiskGravity Decoder for RiskNICLP-5G

Implements:
- Physics-inspired gravity decoder with risk-weighted "mass"
- Directional link prediction with margin ranking loss
- Late fusion for multi-scale integration
- Learnable distance function with epsilon regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union
import math

from ..common.logger import get_logger


class RiskGravityDecoder(nn.Module):
    """
    Physics-inspired decoder using gravitational analogy
    
    F(u,v) = σ(G · (m_u · m_v) / (||h_u - h_v||² + ε))
    
    Where:
    - m_u, m_v: Risk-weighted "masses" learned from embeddings
    - G: Learnable gravitational constant
    - ε: Learnable epsilon for numerical stability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        epsilon: float = 1e-3,
        temperature: float = 0.1,
        learnable_epsilon: bool = True,
        use_risk_weighting: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.use_risk_weighting = use_risk_weighting
        
        # Mass computation networks
        self.mass_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive mass
        )
        
        # Risk weighting for mass
        if use_risk_weighting:
            self.risk_mass_transform = nn.Sequential(
                nn.Linear(1, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )
        
        # Learnable parameters
        self.gravity_constant = nn.Parameter(torch.tensor(1.0))
        
        # Convert epsilon to float if it's a string
        epsilon_value = float(epsilon) if isinstance(epsilon, str) else epsilon
        
        if learnable_epsilon:
            self.epsilon = nn.Parameter(torch.tensor(epsilon_value))
        else:
            self.register_buffer('epsilon', torch.tensor(epsilon_value))
        
        # Distance function enhancement
        self.distance_transform = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # Directional bias
        self.directional_bias = nn.Parameter(torch.zeros(1))
        
        self.logger = get_logger()
        
    def compute_masses(
        self, 
        embeddings: torch.Tensor,
        risk_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute risk-weighted masses from embeddings
        
        Args:
            embeddings: Node embeddings [N, input_dim]
            risk_scores: Risk scores [N, 1]
            
        Returns:
            Masses [N, 1]
        """
        # Base mass from embeddings
        masses = self.mass_network(embeddings)
        
        # Risk weighting
        if self.use_risk_weighting and risk_scores is not None:
            risk_weights = self.risk_mass_transform(risk_scores)
            masses = masses * (1 + risk_weights)
        
        return masses
    
    def compute_enhanced_distance(
        self,
        embeddings_u: torch.Tensor,
        embeddings_v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute enhanced distance with learned transformation
        
        Args:
            embeddings_u: Source embeddings [N, input_dim]
            embeddings_v: Target embeddings [N, input_dim]
            
        Returns:
            Enhanced distances [N, 1]
        """
        # Euclidean distance
        euclidean_dist = torch.norm(embeddings_u - embeddings_v, p=2, dim=-1, keepdim=True)
        
        # Enhanced distance with learned transformation
        combined_features = torch.cat([embeddings_u, embeddings_v], dim=-1)
        distance_enhancement = self.distance_transform(combined_features)
        
        # Combine distances
        enhanced_distance = euclidean_dist + distance_enhancement
        
        return enhanced_distance
    
    def forward(
        self,
        embeddings_u: torch.Tensor,
        embeddings_v: torch.Tensor,
        risk_u: Optional[torch.Tensor] = None,
        risk_v: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute gravitational link scores
        
        Args:
            embeddings_u: Source node embeddings [N, input_dim]
            embeddings_v: Target node embeddings [N, input_dim]
            risk_u: Source risk scores [N, 1]
            risk_v: Target risk scores [N, 1]
            return_components: Whether to return score components
            
        Returns:
            Link scores [N, 1] or (scores, components_dict)
        """
        
        # Compute masses
        mass_u = self.compute_masses(embeddings_u, risk_u)
        mass_v = self.compute_masses(embeddings_v, risk_v)
        
        # Compute enhanced distance
        distance = self.compute_enhanced_distance(embeddings_u, embeddings_v)
        
        # Gravitational force computation
        # F = G * (m_u * m_v) / (distance² + ε)
        numerator = self.gravity_constant * mass_u * mass_v
        denominator = distance ** 2 + torch.abs(self.epsilon)
        
        force = numerator / denominator
        
        # Apply temperature scaling and sigmoid
        scores = torch.sigmoid(force / self.temperature)
        
        if return_components:
            components = {
                'mass_u': mass_u,
                'mass_v': mass_v,
                'distance': distance,
                'force': force,
                'gravity_constant': self.gravity_constant,
                'epsilon': self.epsilon
            }
            return scores, components
        
        return scores
    
    def compute_directional_scores(
        self,
        embeddings_u: torch.Tensor,
        embeddings_v: torch.Tensor,
        risk_u: Optional[torch.Tensor] = None,
        risk_v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bidirectional link scores for direction prediction
        
        Args:
            embeddings_u: Node u embeddings [N, input_dim]
            embeddings_v: Node v embeddings [N, input_dim]  
            risk_u: Node u risk scores [N, 1]
            risk_v: Node v risk scores [N, 1]
            
        Returns:
            Tuple of (scores_u_to_v, scores_v_to_u)
        """
        
        # u → v direction
        scores_uv = self.forward(embeddings_u, embeddings_v, risk_u, risk_v)
        
        # v → u direction  
        scores_vu = self.forward(embeddings_v, embeddings_u, risk_v, risk_u)
        
        # Apply directional bias
        scores_uv = scores_uv + self.directional_bias
        scores_vu = scores_vu - self.directional_bias
        
        return scores_uv, scores_vu


class LateFusionDecoder(nn.Module):
    """Late fusion decoder for multi-scale integration"""
    
    def __init__(
        self,
        fine_input_dim: int,
        coarse_input_dim: int,
        hidden_dim: int = 64,
        fusion_method: str = "mlp",
        epsilon: float = 1e-3
    ):
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # Individual decoders for each scale
        self.fine_decoder = RiskGravityDecoder(
            input_dim=fine_input_dim,
            hidden_dim=hidden_dim,
            epsilon=epsilon
        )
        
        self.coarse_decoder = RiskGravityDecoder(
            input_dim=coarse_input_dim,
            hidden_dim=hidden_dim,
            epsilon=epsilon
        )
        
        # Fusion network
        if fusion_method == "mlp":
            self.fusion_network = nn.Sequential(
                nn.Linear(2, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        elif fusion_method == "weighted":
            self.alpha = nn.Parameter(torch.tensor(0.7))  # Fine scale weight
        elif fusion_method == "attention":
            self.attention_network = nn.Sequential(
                nn.Linear(2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 2),
                nn.Softmax(dim=-1)
            )
        
        self.logger = get_logger()
    
    def forward(
        self,
        fine_embeddings_u: torch.Tensor,
        fine_embeddings_v: torch.Tensor,
        coarse_embeddings_u: torch.Tensor,
        coarse_embeddings_v: torch.Tensor,
        fine_risk_u: Optional[torch.Tensor] = None,
        fine_risk_v: Optional[torch.Tensor] = None,
        coarse_risk_u: Optional[torch.Tensor] = None,
        coarse_risk_v: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with multi-scale fusion
        
        Args:
            fine_embeddings_u/v: Fine scale embeddings
            coarse_embeddings_u/v: Coarse scale embeddings
            fine_risk_u/v: Fine scale risk scores
            coarse_risk_u/v: Coarse scale risk scores
            
        Returns:
            Fused link scores [N, 1]
        """
        
        # Compute scores from both scales
        fine_scores = self.fine_decoder(
            fine_embeddings_u, fine_embeddings_v,
            fine_risk_u, fine_risk_v
        )
        
        coarse_scores = self.coarse_decoder(
            coarse_embeddings_u, coarse_embeddings_v,
            coarse_risk_u, coarse_risk_v
        )
        
        # Fusion
        if self.fusion_method == "mlp":
            combined_scores = torch.cat([fine_scores, coarse_scores], dim=-1)
            fused_scores = self.fusion_network(combined_scores)
            
        elif self.fusion_method == "weighted":
            alpha = torch.sigmoid(self.alpha)
            fused_scores = alpha * fine_scores + (1 - alpha) * coarse_scores
            
        elif self.fusion_method == "attention":
            combined_scores = torch.cat([fine_scores, coarse_scores], dim=-1)
            attention_weights = self.attention_network(combined_scores)
            
            fused_scores = (
                attention_weights[:, 0:1] * fine_scores +
                attention_weights[:, 1:2] * coarse_scores
            )
            
        else:
            # Simple average fallback
            fused_scores = 0.5 * (fine_scores + coarse_scores)
        
        return fused_scores
    
    def compute_directional_scores(
        self,
        fine_embeddings_u: torch.Tensor,
        fine_embeddings_v: torch.Tensor,
        coarse_embeddings_u: torch.Tensor,
        coarse_embeddings_v: torch.Tensor,
        fine_risk_u: Optional[torch.Tensor] = None,
        fine_risk_v: Optional[torch.Tensor] = None,
        coarse_risk_u: Optional[torch.Tensor] = None,
        coarse_risk_v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute directional scores with multi-scale fusion"""
        
        # u → v direction
        scores_uv = self.forward(
            fine_embeddings_u, fine_embeddings_v,
            coarse_embeddings_u, coarse_embeddings_v,
            fine_risk_u, fine_risk_v,
            coarse_risk_u, coarse_risk_v
        )
        
        # v → u direction
        scores_vu = self.forward(
            fine_embeddings_v, fine_embeddings_u,
            coarse_embeddings_v, coarse_embeddings_u,
            fine_risk_v, fine_risk_u,
            coarse_risk_v, coarse_risk_u
        )
        
        return scores_uv, scores_vu


class DirectionalLoss(nn.Module):
    """Directional margin ranking loss for link direction prediction"""
    
    def __init__(
        self,
        margin: float = 0.1,
        reduction: str = "mean"
    ):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(
        self,
        scores_uv: torch.Tensor,
        scores_vu: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute directional ranking loss
        
        Args:
            scores_uv: u→v link scores [N, 1]
            scores_vu: v→u link scores [N, 1]
            labels: Direction labels (1 for u→v, -1 for v→u) [N, 1]
            
        Returns:
            Directional loss
        """
        
        # Margin ranking loss: max(0, margin + wrong_direction - correct_direction)
        correct_scores = torch.where(labels > 0, scores_uv, scores_vu)
        wrong_scores = torch.where(labels > 0, scores_vu, scores_uv)
        
        loss = F.margin_ranking_loss(
            correct_scores.squeeze(),
            wrong_scores.squeeze(),
            torch.ones_like(correct_scores.squeeze()),
            margin=self.margin,
            reduction=self.reduction
        )
        
        return loss 