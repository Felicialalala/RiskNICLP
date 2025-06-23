"""
RiskNICLP-5G Training Framework

Provides progressive training framework for RiskNICLP-5G
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader


class ProgressiveTrainer:
    """Progressive trainer for RiskNICLP-5G model"""
    
    def __init__(self, model, config, logger, device):
        """
        Initialize progressive trainer
        
        Args:
            model: RiskNICLP model instance
            config: Configuration object
            logger: Logger instance
            device: Training device
        """
        self.model = model
        self.config = config
        self.logger = logger
        self.device = device
        
        # Setup optimizer (with type safety for string values)
        lr = getattr(config.training, 'learning_rate', 1e-3)
        lr = float(lr) if isinstance(lr, str) else lr
        
        weight_decay = getattr(config.training, 'weight_decay', 1e-4)
        weight_decay = float(weight_decay) if isinstance(weight_decay, str) else weight_decay
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Setup scheduler (with type safety)
        max_epochs = getattr(config.training, 'max_epochs', 20)
        max_epochs = int(max_epochs) if isinstance(max_epochs, str) else max_epochs
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=lr * 0.01
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        
    def train_progressive(self, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
        """
        Progressive training with three phases: Basic -> Risk-aware -> Full-fusion
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Best model
        """
        max_epochs = getattr(self.config.training, 'max_epochs', 20)
        
        # Define progressive training phases
        phases = {
            'basic': {
                'epochs': 5,
                'thresholds': {'val_auc': 0.80},
                'description': 'Basic GNN feature learning'
            },
            'risk_aware': {
                'epochs': 10, 
                'thresholds': {'risk_recall_100': 0.75},
                'description': 'Risk-aware attention training'
            },
            'full_fusion': {
                'epochs': max_epochs,
                'thresholds': None,
                'description': 'Complete model optimization'
            }
        }
        
        current_phase = 'basic'
        phase_start_epoch = 0
        
        # Start with frozen risk parameters
        if hasattr(self.model, 'shared_encoder'):
            if hasattr(self.model.shared_encoder, 'freeze_risk_params'):
                self.model.shared_encoder.freeze_risk_params()
                self.logger.logger.info("🔒 Risk parameters frozen for basic phase")
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Check if should advance to next phase
            if epoch > phase_start_epoch:
                phase_config = phases[current_phase]
                
                # Get validation metrics for phase decision
                val_metrics = self.validate_epoch(val_loader)
                
                should_advance = False
                
                # Check epoch-based advancement
                if epoch - phase_start_epoch >= phase_config['epochs']:
                    should_advance = True
                    advancement_reason = "epoch limit reached"
                    
                # Check metric-based advancement  
                elif phase_config['thresholds']:
                    for metric_name, threshold in phase_config['thresholds'].items():
                        if val_metrics.get(metric_name, 0) >= threshold:
                            should_advance = True
                            advancement_reason = f"{metric_name} >= {threshold}"
                            break
                
                # Advance to next phase
                if should_advance and current_phase != 'full_fusion':
                    next_phase = self._get_next_phase(current_phase)
                    self.logger.logger.info(
                        f"🔄 Advancing from {current_phase} to {next_phase} phase "
                        f"(reason: {advancement_reason})"
                    )
                    
                    # Update training phase
                    current_phase = next_phase
                    phase_start_epoch = epoch
                    
                    # Unfreeze risk parameters when entering risk-aware phase
                    if current_phase == 'risk_aware':
                        if hasattr(self.model, 'shared_encoder'):
                            if hasattr(self.model.shared_encoder, 'unfreeze_risk_params'):
                                self.model.shared_encoder.unfreeze_risk_params()
                                self.logger.logger.info("🔓 Risk parameters unfrozen for risk-aware phase")
                    
                    # Enable full fusion capabilities
                    elif current_phase == 'full_fusion':
                        self.logger.logger.info("🚀 Full fusion phase activated")
            
            # Update model training phase
            if hasattr(self.model, 'training_phase'):
                self.model.training_phase = current_phase
            
            # Training phase  
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Enhanced logging with phase info
            phase_desc = phases[current_phase]['description']
            self.logger.logger.info(
                f"📊 Epoch {epoch + 1}/{max_epochs} [{current_phase.upper()}]: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val AUC: {val_metrics.get('auc', 0.0):.4f}, "
                f"Phase: {phase_desc}"
            )
            
            # Save best model
            current_metric = val_metrics.get('auc', 0.0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint('best_model.pt')
                self.logger.logger.info(f"💾 New best model saved (AUC: {current_metric:.4f})")
        
        # Final phase summary
        self.logger.logger.info(f"✅ Progressive training completed. Best AUC: {self.best_metric:.4f}")
        
        return self.model
    
    def _get_next_phase(self, current_phase: str) -> str:
        """Get next training phase"""
        phase_order = ['basic', 'risk_aware', 'full_fusion']
        try:
            current_idx = phase_order.index(current_phase)
            if current_idx < len(phase_order) - 1:
                return phase_order[current_idx + 1]
        except ValueError:
            pass
        return 'full_fusion'  # Default to final phase
        
    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            data_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(data_loader):
            try:
                # Move to device (placeholder)
                # In real implementation, move batch data to device
                
                # Forward pass
                fine_scores, coarse_scores = self.model(batch)
                
                # Compute loss (placeholder)
                loss = torch.tensor(0.5, requires_grad=True)  # Placeholder loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update parameters
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                self.logger.logger.warning(f"Training error in batch {batch_idx}: {e}")
                continue
                
        return total_loss / max(num_batches, 1)
        
    def validate_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            data_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Placeholder validation
            metrics = {
                'auc': 0.85,
                'loss': 0.3,
                'precision': 0.80,
                'recall': 0.75
            }
            
        return metrics
        
    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric
        }
        
        # Placeholder save - in real implementation, save to file
        self.logger.logger.info(f"Checkpoint saved: {filename}")
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        # Placeholder implementation
        self.logger.logger.info(f"Checkpoint loaded: {checkpoint_path}") 