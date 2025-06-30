"""
RiskNICLP-5G Progressive Training Module

Implements progressive training strategy with phase-based parameter freezing/unfreezing
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import logging

from ..common.logger import get_logger


class TrainingPhase(Enum):
    """Training phases for progressive training"""
    BASIC = "basic"  # Basic GNN feature learning
    RISK_AWARE = "risk_aware"  # Risk-aware attention training
    FULL_FUSION = "full_fusion"  # Complete model optimization


class ProgressiveTrainer:
    """Progressive trainer with risk parameter management"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        basic_epochs: int = 5,
        risk_aware_epochs: int = 10,
        basic_auc_threshold: float = 0.8,
        risk_recall_threshold: float = 0.75,
        device: str = 'cpu'
    ):
        """
        Initialize progressive trainer
        
        Args:
            model: Model to train
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            basic_epochs: Epochs for basic phase
            risk_aware_epochs: Epochs for risk-aware phase  
            basic_auc_threshold: AUC threshold to advance from basic phase
            risk_recall_threshold: Risk recall threshold to advance phases
            device: Training device
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.basic_epochs = basic_epochs
        self.risk_aware_epochs = risk_aware_epochs
        self.basic_auc_threshold = basic_auc_threshold
        self.risk_recall_threshold = risk_recall_threshold
        self.device = device
        
        self.current_phase = TrainingPhase.BASIC
        self.phase_epoch = 0
        self.best_auc = 0.0
        self.best_model_state = None
        
        self.logger = get_logger()
        
    def train_epoch(
        self,
        train_loader,
        criterion,
        evaluator=None,
        val_loader=None
    ) -> Dict[str, float]:
        """
        Train one epoch with progressive strategy
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            evaluator: Evaluator for validation
            val_loader: Validation data loader
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        # Phase management
        self._manage_training_phase()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            fine_scores, coarse_scores = self.model(batch)
            
            # Compute loss
            edge_labels = batch['edge_labels']
            loss = criterion(fine_scores, edge_labels.float())
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 🔥 SOTA阶段性参数监控 + 梯度监控
            if batch_idx % 50 == 0:
                if hasattr(self.model, 'shared_encoder'):
                    self._log_risk_bias_parameters_brief()
                
                # 梯度监控
                grad_stats = self._monitor_gradients()
                
                # 检查风险偏置参数是否有梯度
                risk_has_grad = any('beta' in name or 'temperature' in name 
                                   for name in grad_stats.keys())
                if not risk_has_grad:
                    self.logger.warning("❌ 风险偏置参数没有梯度！")
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Validation
        val_metrics = {}
        if evaluator and val_loader:
            val_metrics = evaluator.evaluate(val_loader, split='val')
            
            # Update best model
            current_auc = val_metrics.get('auc', 0.0)
            if current_auc > self.best_auc:
                self.best_auc = current_auc
                self.best_model_state = self.model.state_dict().copy()
        
        # Phase progression
        self.phase_epoch += 1
        
        return {
            'loss': avg_loss,
            'phase': self.current_phase.value,
            'phase_epoch': self.phase_epoch,
            **val_metrics
        }
    
    def _manage_training_phase(self):
        """Manage training phase transitions"""
        
        if self.current_phase == TrainingPhase.BASIC:
            # Freeze risk parameters
            self._freeze_risk_parameters()
            
            # Check for phase advancement
            if (self.phase_epoch >= self.basic_epochs or 
                self.best_auc >= self.basic_auc_threshold):
                self._advance_to_risk_aware()
                
        elif self.current_phase == TrainingPhase.RISK_AWARE:
            # Unfreeze risk parameters
            self._unfreeze_risk_parameters()
            
            # Check for phase advancement
            if self.phase_epoch >= self.risk_aware_epochs:
                self._advance_to_full_fusion()
                
        elif self.current_phase == TrainingPhase.FULL_FUSION:
            # All parameters active
            self._unfreeze_all_parameters()
    
    def _advance_to_risk_aware(self):
        """Advance to risk-aware training phase"""
        reason = "epoch limit reached" if self.phase_epoch >= self.basic_epochs else "AUC threshold reached"
        self.logger.info(f"🔄 Advancing from basic to risk_aware phase (reason: {reason})")
        
        self.current_phase = TrainingPhase.RISK_AWARE
        self.phase_epoch = 0
        
        # Unfreeze risk parameters
        self._unfreeze_risk_parameters()
        self.logger.info("🔓 Risk parameters unfrozen for risk-aware phase")
    
    def _advance_to_full_fusion(self):
        """Advance to full fusion phase"""
        self.logger.info("🔄 Advancing from risk_aware to full_fusion phase (reason: epoch limit reached)")
        
        self.current_phase = TrainingPhase.FULL_FUSION
        self.phase_epoch = 0
        
        # Ensure all parameters are active
        self._unfreeze_all_parameters()
        self.logger.info("🚀 Full fusion phase activated")
    
    def _freeze_risk_parameters(self):
        """Freeze risk-related parameters"""
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if any(keyword in name.lower() for keyword in ['risk', 'beta', 'temperature']):
                param.requires_grad = False
                frozen_count += 1
        
        # Also freeze specific modules
        for name, module in self.model.named_modules():
            if hasattr(module, 'freeze_risk_params'):
                module.freeze_risk_params()
        
        if frozen_count > 0:
            self.logger.info(f"🔒 Risk parameters frozen for basic phase ({frozen_count} parameters)")
    
    def _unfreeze_risk_parameters(self):
        """Unfreeze risk-related parameters"""
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if any(keyword in name.lower() for keyword in ['risk', 'beta', 'temperature']):
                param.requires_grad = True
                unfrozen_count += 1
        
        # Also unfreeze specific modules
        for name, module in self.model.named_modules():
            if hasattr(module, 'unfreeze_risk_params'):
                module.unfreeze_risk_params()
                
        if unfrozen_count > 0:
            self.logger.info(f"🔓 Risk parameters unfrozen ({unfrozen_count} parameters)")
    
    def _unfreeze_all_parameters(self):
        """Unfreeze all model parameters"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def _log_risk_bias_parameters_brief(self):
        """🔥 简要记录风险偏置参数 (降低频率)"""
        try:
            for name, module in self.model.named_modules():
                if hasattr(module, 'beta_src') and hasattr(module, 'beta_dst'):
                    if 'true_risk_gat' in name.lower():  # 只监控关键模块
                        beta_src = float(module.beta_src.data)
                        beta_dst = float(module.beta_dst.data)
                        temperature = float(module.temperature.data) if hasattr(module, 'temperature') else 0.1
                        
                        self.logger.info(f"🔥 {name} 参数: β₁={beta_src:.3f}, β₂={beta_dst:.3f}, τ={temperature:.3f}")
                        break  # 只记录第一个找到的
        except Exception as e:
            pass  # 忽略错误，避免训练中断
    
    def _log_risk_bias_parameters(self):
        """🔥 记录风险偏置参数的学习情况"""
        try:
            risk_params = {}
            for name, module in self.model.named_modules():
                if hasattr(module, 'beta_src') and hasattr(module, 'beta_dst'):
                    beta_src = float(module.beta_src.data)
                    beta_dst = float(module.beta_dst.data)
                    temperature = float(module.temperature.data) if hasattr(module, 'temperature') else 0.1
                    
                    risk_params[f"{name}_beta_src"] = beta_src
                    risk_params[f"{name}_beta_dst"] = beta_dst
                    risk_params[f"{name}_temperature"] = temperature
                    
                    self.logger.info(f"🔥 {name} 风险偏置参数: β₁={beta_src:.3f}, β₂={beta_dst:.3f}, τ={temperature:.3f}")
            
            return risk_params
        except Exception as e:
            self.logger.warning(f"风险偏置参数记录失败: {e}")
            return {}
    
    def _get_risk_parameters(self) -> dict:
        """获取当前的风险偏置参数"""
        risk_params = {}
        try:
            for name, module in self.model.named_modules():
                if hasattr(module, 'beta_src') and hasattr(module, 'beta_dst'):
                    risk_params[f"{name}_beta_src"] = float(module.beta_src.data)
                    risk_params[f"{name}_beta_dst"] = float(module.beta_dst.data)
                    if hasattr(module, 'temperature'):
                        risk_params[f"{name}_temperature"] = float(module.temperature.data)
        except Exception as e:
            self.logger.warning(f"获取风险参数失败: {e}")
        return risk_params
    
    def _monitor_gradients(self):
        """SOTA梯度监控 - 监控梯度传播情况"""
        grad_stats = {}
        zero_grad_params = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats[name] = grad_norm
                
                # 检查梯度过小的参数
                if grad_norm < 1e-6:
                    zero_grad_params.append(name)
            else:
                zero_grad_params.append(f"{name} (None)")
        
        # 记录风险偏置参数的梯度
        risk_grad_norms = {k: v for k, v in grad_stats.items() 
                          if any(keyword in k.lower() for keyword in ['beta', 'temperature', 'risk'])}
        
        if risk_grad_norms:
            self.logger.info(f"🔥 风险偏置参数梯度: {risk_grad_norms}")
        
        if zero_grad_params:
            self.logger.warning(f"⚠️ 零梯度参数: {zero_grad_params[:5]}...")  # 只显示前5个
            
        return grad_stats
    
    def _validate_training_state(self):
        """SOTA训练状态验证"""
        issues = []
        
        # 检查风险偏置参数是否可训练
        risk_params_trainable = 0
        total_risk_params = 0
        
        for name, param in self.model.named_parameters():
            if any(keyword in name.lower() for keyword in ['beta', 'temperature', 'risk']):
                total_risk_params += 1
                if param.requires_grad:
                    risk_params_trainable += 1
                else:
                    issues.append(f"参数 {name} 未设置为可训练")
        
        if risk_params_trainable == 0:
            issues.append("❌ 没有风险偏置参数参与训练")
        else:
            self.logger.info(f"✅ {risk_params_trainable}/{total_risk_params} 风险偏置参数可训练")
        
        # 检查学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr < 1e-6:
            issues.append(f"学习率过小: {current_lr}")
        
        if issues:
            self.logger.warning("训练状态问题: " + "; ".join(issues))
        
        return len(issues) == 0
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_auc': self.best_auc,
            'current_phase': self.current_phase.value,
            'phase_epoch': self.phase_epoch,
            'metrics': metrics,
            'risk_params': self._get_risk_parameters()
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, filepath)
        self.logger.info(f"💾 Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.best_auc = checkpoint.get('best_auc', 0.0)
        self.current_phase = TrainingPhase(checkpoint.get('current_phase', 'basic'))
        self.phase_epoch = checkpoint.get('phase_epoch', 0)
        
        self.logger.info(f"📂 Checkpoint loaded: {filepath}")
        self.logger.info(f"   Phase: {self.current_phase.value}, Epoch: {self.phase_epoch}, Best AUC: {self.best_auc:.4f}")
        
        return checkpoint.get('metrics', {})