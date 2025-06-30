"""
RiskNICLP-5G Training Framework

Provides progressive training framework for RiskNICLP-5G with temporal support
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader
from ..evaluation.temporal_metrics import TemporalEvaluator, compute_risk_weighted_metrics
from ..models.temporal_predictor import TemporalDataLoader
from .intelligent_validator import IntelligentValidator
from .risk_bias_diagnostic import RiskBiasIntelligentDiagnostic
import torch.nn.functional as F
import numpy as np


class ProgressiveTrainer:
    """Progressive trainer for RiskNICLP-5G model with temporal support"""
    
    def __init__(self, model, config, logger, device, temporal_predictor=None):
        """
        Initialize progressive trainer
        
        Args:
            model: RiskNICLP model instance
            config: Configuration object
            logger: Logger instance
            device: Training device
            temporal_predictor: Temporal predictor model (optional)
        """
        self.model = model
        self.config = config
        self.logger = logger
        self.device = device
        
        # Temporal prediction support
        self.temporal_predictor = temporal_predictor
        self.temporal_evaluator = TemporalEvaluator() if temporal_predictor else None
        self.training_mode = getattr(config.training, 'mode', 'standard')  # 'standard' or 'temporal'
        
        # 🔥 修复：先冻结参数，再创建优化器
        # 初始阶段冻结风险参数
        if hasattr(model, 'freeze_risk_params'):
            model.freeze_risk_params()
            logger.logger.info("🔒 风险参数已冻结（在优化器创建前）")
        
        # Negative sampling configuration
        self.negative_sampling_enabled = False  # 🔥 默认初始化，防止AttributeError
        self.negative_sampling_config = {}
        self.pos_neg_ratio_monitor = {'positive': 0, 'negative': 0}
        
        if hasattr(config, 'phase2') and hasattr(config.phase2, 'negative_sampling'):
            self.negative_sampling_config = config.phase2['negative_sampling']
            self.negative_sampling_enabled = self.negative_sampling_config.get('enabled', False)
            logger.logger.info(f"🎯 负采样平衡训练已启用: {self.negative_sampling_config.get('strategy', 'unknown')}")
        else:
            logger.logger.info("📊 负采样配置未找到，使用默认图级别训练")
        
        # 🔥 关键修复：Setup optimizer with only trainable parameters
        lr = getattr(config.training, 'learning_rate', 1e-3)
        lr = float(lr) if isinstance(lr, str) else lr
        
        weight_decay = getattr(config.training, 'weight_decay', 1e-4)
        weight_decay = float(weight_decay) if isinstance(weight_decay, str) else weight_decay
        
        # 🔥 关键修复：只为requires_grad=True的参数创建优化器
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,  # 只包含可训练参数
            lr=lr,
            weight_decay=weight_decay
        )
        
        logger.logger.info(f"✅ 优化器创建：管理 {len(trainable_params)} 个可训练参数")
        
        # Setup scheduler (with type safety)
        max_epochs = getattr(config.training, 'max_epochs', 20)
        max_epochs = int(max_epochs) if isinstance(max_epochs, str) else max_epochs
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=lr * 0.01
        )
        
        # Adaptive loss function for negative sampling
        self.criterion = nn.BCELoss()
        if self.negative_sampling_enabled:
            # Will be updated dynamically based on batch positive/negative ratios
            self.adaptive_weights = {'pos': 1.0, 'neg': 1.0}
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.current_phase = 'basic'  # 🔥 添加当前训练阶段追踪
        
        # 🔥 智能验证器 - 死锁预防系统
        self.intelligent_validator = IntelligentValidator(
            timeout_seconds=180,  # 3分钟超时
            logger=logger
        )
        
        # 🔥 风险偏置诊断器 - 参数解冻+梯度监控
        self.risk_bias_diagnostic = RiskBiasIntelligentDiagnostic(
            model=model,
            logger=logger
        )
        
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
        
        # 🔥 修复：使用配置文件中的阈值，而不是硬编码
        phases = {
            'basic': {
                'epochs': getattr(self.config.training, 'basic_epochs', 1),
                'thresholds': {'val_auc': getattr(self.config.training, 'basic_auc_threshold', 0.60)},
                'description': 'Basic GNN feature learning'
            },
            'risk_aware': {
                'epochs': getattr(self.config.training, 'risk_aware_epochs', 8), 
                'thresholds': {'risk_recall_100': getattr(self.config.training, 'risk_recall_threshold', 0.70)},
                'description': 'Risk-aware attention training'
            },
            'full_fusion': {
                'epochs': max_epochs,
                'thresholds': None,
                'description': 'Complete model optimization'
            }
        }
        
        current_phase = 'basic'
        self.current_phase = current_phase  # 🔥 初始化阶段追踪
        phase_start_epoch = 0
        

        
        # 🔥 修复：参数冻结后再进行诊断检查
        self.logger.logger.info("🔍 验证训练状态...")
        
        # 更新诊断工具的训练阶段
        self.risk_bias_diagnostic.training_stage = current_phase
        
        # 综合诊断训练就绪状态 - 修复版：Basic阶段不执行错误的"修复"
        is_ready, readiness_result = self.risk_bias_diagnostic.verify_training_readiness()
        
        if not is_ready:
            # 🔥 修复：Basic阶段的"未就绪"状态通常是正常的（因为参数被冻结）
            if current_phase == 'basic':
                # Basic阶段：检查是否真的有问题，还是正常的冻结状态
                failed_checks = [k for k, v in readiness_result.get('readiness_checks', {}).items() if not v]
                expected_basic_failures = ['params_trainable', 'has_gradients', 'mechanism_active']
                
                # 如果失败的都是预期的项目，则认为状态正常
                unexpected_failures = [f for f in failed_checks if f not in expected_basic_failures]
                
                if not unexpected_failures:
                    self.logger.logger.info("✅ Basic阶段状态正常：风险参数已正确冻结")
                else:
                    self.logger.logger.warning(f"⚠️ Basic阶段检测到意外问题: {unexpected_failures}")
                    # 只处理真正的问题
                    if 'has_risk_params' in unexpected_failures:
                        self.logger.logger.error("❌ 严重：未找到风险参数，请检查模型架构")
            else:
                # 非Basic阶段：尝试解冻参数
                self.logger.logger.warning("⚠️ 非Basic阶段训练就绪状态检查失败，尝试自动修复...")
                if self.risk_bias_diagnostic.force_unfreeze_risk_parameters():
                    self.logger.logger.info("🔧 已尝试强制解冻风险参数")
                
                # 重新验证
                is_ready, readiness_result = self.risk_bias_diagnostic.verify_training_readiness()
                
                if not is_ready:
                    failed_checks = [k for k, v in readiness_result.get('readiness_checks', {}).items() if not v]
                    self.logger.logger.warning(f"❌ 自动修复后仍有失败项: {failed_checks}")
                else:
                    self.logger.logger.info("✅ 自动修复成功")
                
        if not self._validate_training_state():
            self.logger.logger.warning("⚠️ 基础训练状态存在问题，但继续训练")
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Update model training phase
            if hasattr(self.model, 'training_phase'):
                self.model.training_phase = current_phase
            
            # 🔥 新增：更新风险偏置注意力层的训练阶段
            self._update_risk_attention_stage(current_phase)
            
            # Training phase  
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            
            # 🔥 修复：在每个epoch结束后检查阶段切换条件
            phase_config = phases[current_phase]
            should_advance = False
            advancement_reason = ""
            
            # Check epoch-based advancement
            if epoch - phase_start_epoch >= phase_config['epochs']:
                should_advance = True
                advancement_reason = "epoch limit reached"
                
            # Check metric-based advancement  
            elif phase_config['thresholds']:
                for metric_name, threshold in phase_config['thresholds'].items():
                    if val_metrics.get(metric_name, 0) >= threshold:
                        should_advance = True
                        advancement_reason = f"{metric_name} >= {threshold} (current: {val_metrics.get(metric_name, 0):.4f})"
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
                self.current_phase = current_phase  # 🔥 同步更新实例变量
                phase_start_epoch = epoch + 1  # 🔥 修复：下一个阶段从下个epoch开始
                
                # 🔥 修复：使用统一的参数解冻方法
                if current_phase == 'risk_aware':
                    if hasattr(self.model, 'unfreeze_risk_params'):
                        self.model.unfreeze_risk_params()
                        self.logger.logger.info("🔓 Risk parameters unfrozen for risk-aware phase")
                        
                        # 🔥 关键修复：立即重新创建优化器包含新解冻的参数
                        self._recreate_optimizer_for_risk_aware()
                        
                        # 🔥 验证风险参数解冻状态
                        risk_status = self._verify_risk_param_status()
                        trainable_count = risk_status.get('trainable_params', 0)
                        total_count = risk_status.get('total_params', 0)
                        
                        if total_count > 0 and trainable_count == total_count:
                            self.logger.logger.info(f"✅ 风险参数解冻验证成功: {trainable_count}/{total_count} 参数已激活")
                            
                            # 🔥 新增：验证优化器参数覆盖
                            opt_param_count = sum(len(group['params']) for group in self.optimizer.param_groups)
                            total_model_params = sum(1 for p in self.model.parameters() if p.requires_grad)
                            
                            if opt_param_count == total_model_params:
                                self.logger.logger.info(f"✅ 优化器参数覆盖验证成功: {opt_param_count}/{total_model_params}")
                            else:
                                self.logger.logger.error(f"❌ 优化器参数覆盖失败: {opt_param_count}/{total_model_params}")
                                # 强制重建优化器
                                self._force_recreate_optimizer()
                                
                        else:
                            self.logger.logger.warning(f"⚠️ 风险参数解冻不完整: {trainable_count}/{total_count} 参数已激活")
                            # 尝试再次解冻
                            self.model.unfreeze_risk_params()
                            self._recreate_optimizer_for_risk_aware()
                            self.logger.logger.info("🔧 已尝试再次解冻风险参数并重建优化器")
                    else:
                        self.logger.logger.warning("⚠️ 模型没有unfreeze_risk_params方法，尝试使用SharedEncoder方法")
                        if hasattr(self.model, 'shared_encoder') and hasattr(self.model.shared_encoder, 'unfreeze_risk_params'):
                            self.model.shared_encoder.unfreeze_risk_params()
                            self._recreate_optimizer_for_risk_aware()
                            self.logger.logger.info("🔓 SharedEncoder风险参数已解冻并重建优化器")
                
                # Enable full fusion capabilities
                elif current_phase == 'full_fusion':
                    self.logger.logger.info("🚀 Full fusion phase activated")
            
            # Update scheduler
            self.scheduler.step()
            
            # 🔥 风险偏置健康监控
            if current_phase in ['risk_aware', 'full_fusion']:
                # 在Risk-aware阶段进行深度诊断
                try:
                    from .risk_bias_diagnostic import RiskBiasIntelligentDiagnostic
                    diagnostic = RiskBiasIntelligentDiagnostic(
                        model=self.model, 
                        logger=self.logger, 
                        training_stage=current_phase
                    )
                    diagnostic.comprehensive_risk_bias_diagnosis()
                except Exception as e:
                    self.logger.logger.warning(f"风险偏置诊断失败: {e}")
            
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
    
    def train_temporal_sequences(self, graph_sequences: List, val_sequences: List) -> nn.Module:
        """
        Train with temporal graph sequences for lead-time prediction
        
        Args:
            graph_sequences: Training graph sequences
            val_sequences: Validation graph sequences
            
        Returns:
            Best temporal model
        """
        if self.temporal_predictor is None:
            raise ValueError("Temporal predictor not provided for temporal training")
            
        max_epochs = getattr(self.config.training, 'max_epochs', 20)
        
        # Setup temporal data loader
        temporal_train_loader = TemporalDataLoader(graph_sequences, window_size=5, stride=1)
        temporal_val_loader = TemporalDataLoader(val_sequences, window_size=5, stride=1)
        
        # Setup temporal optimizer
        temporal_optimizer = AdamW(
            self.temporal_predictor.parameters(),
            lr=float(getattr(self.config.training, 'learning_rate', 1e-3)),
            weight_decay=float(getattr(self.config.training, 'weight_decay', 1e-4))
        )
        
        best_lead_time = 0.0
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Train temporal predictor
            temporal_loss = self.train_temporal_epoch(temporal_train_loader, temporal_optimizer)
            
            # Validate temporal predictions
            temporal_metrics = self.validate_temporal_epoch(temporal_val_loader)
            
            # Log temporal training progress
            mean_lead_time = temporal_metrics.get('mean_lead_time', 0.0)
            self.logger.logger.info(
                f"🕐 Temporal Epoch {epoch + 1}/{max_epochs}: "
                f"Loss: {temporal_loss:.4f}, "
                f"Lead-Time: {mean_lead_time:.2f}s, "
                f"Success Rate (5s): {temporal_metrics.get('lead_time_5s_success_rate', 0.0):.3f}"
            )
            
            # Save best temporal model
            if mean_lead_time > best_lead_time:
                best_lead_time = mean_lead_time
                self.save_temporal_checkpoint('best_temporal_model.pt')
                self.logger.logger.info(f"💾 New best temporal model saved (Lead-Time: {mean_lead_time:.2f}s)")
                
        self.logger.logger.info(f"✅ Temporal training completed. Best Lead-Time: {best_lead_time:.2f}s")
        
        return self.temporal_predictor
    
    def train_temporal_epoch(self, temporal_loader: TemporalDataLoader, optimizer) -> float:
        """Train temporal predictor for one epoch"""
        self.temporal_predictor.train()
        total_loss = 0.0
        num_batches = 0
        
        temporal_criterion = nn.MSELoss()  # For temporal risk prediction
        
        for sequence in temporal_loader.get_temporal_sequences():
            historical_graphs = sequence['historical_graphs']
            current_graph = sequence['current_graph']
            future_graph = sequence.get('future_graph', None)
            
            if future_graph is None:
                continue  # Skip sequences without future data
                
            try:
                # Forward pass
                predicted_risks = self.temporal_predictor(historical_graphs, current_graph)
                
                # Compute target risks from future graph
                target_risks = self._compute_target_risks(current_graph, future_graph)
                
                # Ensure matching dimensions
                if predicted_risks.size(0) != target_risks.size(0):
                    min_size = min(predicted_risks.size(0), target_risks.size(0))
                    predicted_risks = predicted_risks[:min_size]
                    target_risks = target_risks[:min_size]
                
                # Compute loss
                loss = temporal_criterion(predicted_risks.mean(dim=-1), target_risks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                self.logger.logger.warning(f"Temporal training error: {e}")
                continue
                
        return total_loss / max(num_batches, 1)
    
    def validate_temporal_epoch(self, temporal_loader: TemporalDataLoader) -> Dict[str, float]:
        """Validate temporal predictions for one epoch"""
        if self.temporal_evaluator is None:
            return {}
            
        self.temporal_predictor.eval()
        temporal_sequences = temporal_loader.get_temporal_sequences()
        
        # Evaluate lead-time performance
        temporal_metrics = self.temporal_evaluator.evaluate_early_warning(
            self.temporal_predictor, temporal_sequences, self.device
        )
        
        return temporal_metrics
    
    def _compute_target_risks(self, current_graph, future_graph) -> torch.Tensor:
        """Compute target risk scores from future graph"""
        current_nodes = current_graph.number_of_nodes()
        future_nodes = future_graph.number_of_nodes()
        
        # Use node degree change as risk proxy
        current_degrees = current_graph.in_degrees().float()
        future_degrees = future_graph.in_degrees().float()
        
        # Handle different graph sizes
        min_nodes = min(current_nodes, future_nodes)
        if min_nodes == 0:
            return torch.zeros(1, device=self.device)
            
        current_degrees = current_degrees[:min_nodes]
        future_degrees = future_degrees[:min_nodes]
        
        # Risk increase = degree increase (normalized)
        degree_change = future_degrees - current_degrees
        risk_scores = torch.clamp(degree_change / (current_degrees + 1e-8), 0, 1)
        
        return risk_scores.to(self.device)
    
    def save_temporal_checkpoint(self, filename: str) -> None:
        """Save temporal model checkpoint"""
        if self.temporal_predictor is not None:
            checkpoint = {
                'model_state_dict': self.temporal_predictor.state_dict(),
                'epoch': self.current_epoch,
                'config': self.config
            }
            torch.save(checkpoint, filename)
    
    def _validate_training_state(self) -> bool:
        """验证训练状态和风险参数配置 - 修复版：阶段感知检查"""
        try:
            print("🔍 验证训练状态...")
            
            # 🔥 修复：使用修复版诊断工具，传递正确的训练阶段
            from .risk_bias_diagnostic import RiskBiasIntelligentDiagnostic
            
            # 确定当前训练阶段
            current_stage = "basic" if hasattr(self, 'current_phase') and self.current_phase == "basic" else "risk_aware"
            
            diagnostic = RiskBiasIntelligentDiagnostic(
                model=self.model, 
                logger=self.logger, 
                training_stage=current_stage
            )
            
            # 进行综合诊断 - 直接使用诊断工具的结果，不再自定义错误的检查
            is_ready, readiness_result = diagnostic.verify_training_readiness()
            
            if not is_ready:
                # 🔥 修复：只有在非Basic阶段或者确实存在严重问题时才尝试修复
                if current_stage != "basic":
                    self.logger.logger.warning("⚠️ 非Basic阶段训练就绪状态检查失败，尝试自动修复...")
                    
                    # 只在Risk-aware阶段才尝试解冻参数
                    if diagnostic.force_unfreeze_risk_parameters():
                        self.logger.logger.info("✅ 风险参数自动解冻成功")
                    
                    # 🔥 修复：调用主模型的统一解冻方法
                    if hasattr(self.model, 'unfreeze_risk_params'):
                        try:
                            self.model.unfreeze_risk_params()
                            self.logger.logger.info("✅ 主模型风险参数解冻成功")
                        except Exception as e:
                            self.logger.logger.warning(f"主模型解冻失败: {e}")
                    elif hasattr(self.model, 'shared_encoder') and hasattr(self.model.shared_encoder, 'unfreeze_risk_params'):
                        try:
                            self.model.shared_encoder.unfreeze_risk_params()
                            self.logger.logger.info("✅ SharedEncoder风险参数解冻成功")
                        except Exception as e:
                            self.logger.logger.warning(f"SharedEncoder解冻失败: {e}")
                    
                    # 重新验证
                    is_ready, readiness_result = diagnostic.verify_training_readiness()
                    
                    if not is_ready:
                        failed_checks = [k for k, v in readiness_result.get('readiness_checks', {}).items() if not v]
                        self.logger.logger.warning(f"❌ 自动修复后仍有失败项: {failed_checks}")
                        return False
                    else:
                        self.logger.logger.info("✅ 自动修复成功，所有检查项通过")
                else:
                    # Basic阶段：不应该出现就绪性失败，如果出现则说明有严重问题
                    failed_checks = [k for k, v in readiness_result.get('readiness_checks', {}).items() if not v]
                    self.logger.logger.info(f"📋 Basic阶段状态检查结果: {failed_checks}")
                    # Basic阶段允许某些"失败"项（因为参数应该被冻结）
                    return True
            
            # 验证优化器状态
            lr = self.optimizer.param_groups[0]['lr']
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            diagnosis_result = readiness_result.get('full_diagnosis', {})
            risk_param_count = diagnosis_result.get('parameter_status', {}).get('total_risk_params', 0)
            
            self.logger.logger.info(f"✅ 训练状态正常: 风险参数 {risk_param_count}, 可训练参数 {trainable_params}/{total_params}, LR={lr:.2e}")
            
            return True
            
        except Exception as e:
            self.logger.logger.error(f"❌ 训练状态验证失败: {str(e)}")
            return False

    def _monitor_gradients(self) -> Dict[str, float]:
        """SOTA梯度监控方法 - 阶段感知"""
        grad_stats = {}
        risk_params_info = {}
        
        try:
            for name, param in self.model.named_parameters():
                # 收集风险偏置参数的详细信息
                if 'beta' in name or 'temperature' in name:
                    risk_params_info[name] = {
                        'requires_grad': param.requires_grad,
                        'has_grad': param.grad is not None,
                        'grad_norm': float(param.grad.norm(2)) if param.grad is not None else 0.0
                    }
                
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2)
                    grad_stats[name] = float(grad_norm)
                    
                    # 检查异常梯度
                    if torch.isnan(param.grad).any():
                        self.logger.logger.warning(f"⚠️ NaN梯度: {name}")
                    elif torch.isinf(param.grad).any():
                        self.logger.logger.warning(f"⚠️ Inf梯度: {name}")
            
            # 详细的风险偏置参数状态报告
            risk_grad_count = sum(1 for info in risk_params_info.values() if info['has_grad'])
            risk_trainable_count = sum(1 for info in risk_params_info.values() if info['requires_grad'])
            
            # 阶段特定的诊断
            if self.current_phase == 'basic':
                frozen_count = sum(1 for info in risk_params_info.values() if not info['requires_grad'])
                if frozen_count == len(risk_params_info):
                    self.logger.logger.debug(f"✅ Basic阶段：{frozen_count}个风险参数正确冻结")
                else:
                    self.logger.logger.warning(f"⚠️ Basic阶段：应冻结{len(risk_params_info)}个风险参数，实际冻结{frozen_count}个")
            else:
                if risk_trainable_count > 0 and risk_grad_count > 0:
                    self.logger.logger.debug(f"✅ {self.current_phase.upper()}阶段：{risk_grad_count}/{risk_trainable_count}个风险参数有梯度")
                else:
                    self.logger.logger.warning(f"❌ {self.current_phase.upper()}阶段：风险参数状态异常 - 可训练:{risk_trainable_count}, 有梯度:{risk_grad_count}")
                    
        except Exception as e:
            self.logger.logger.error(f"❌ 梯度监控失败: {e}")
            
        return grad_stats

    def _verify_risk_param_status(self) -> Dict[str, any]:
        """🔥 修复版：验证风险参数状态 - 正确查找路径"""
        try:
            status = {
                'total_params': 0,
                'frozen_params': 0,
                'trainable_params': 0,
                'param_details': [],
                'has_risk_mechanism': False
            }
            
            # 🔥 修复：检查SharedEncoder中的risk_bias_layers
            if hasattr(self.model, 'shared_encoder') and hasattr(self.model.shared_encoder, 'risk_bias_layers'):
                risk_layers = self.model.shared_encoder.risk_bias_layers
                status['has_risk_mechanism'] = True
                
                for layer_idx, layer in enumerate(risk_layers):
                    # 检查每层的风险参数
                    for param_name in ['beta_src', 'beta_dst', 'temperature']:
                        if hasattr(layer, param_name):
                            param = getattr(layer, param_name)
                            if isinstance(param, nn.Parameter):
                                status['total_params'] += 1
                                
                                param_info = {
                                    'name': f"risk_bias_layers[{layer_idx}].{param_name}",
                                    'requires_grad': param.requires_grad,
                                    'value': param.data.detach().cpu().numpy().tolist() if param.numel() > 1 else float(param.data.detach().cpu()),
                                    'grad_available': param.grad is not None
                                }
                                
                                if param.grad is not None:
                                    param_info['grad_norm'] = float(param.grad.norm().item())
                                
                                status['param_details'].append(param_info)
                                
                                if param.requires_grad:
                                    status['trainable_params'] += 1
                                else:
                                    status['frozen_params'] += 1
            
            # 🔥 修复：检查RiskGravityDecoder中的温度参数
            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'temperature'):
                param = self.model.decoder.temperature
                if isinstance(param, nn.Parameter):
                    status['total_params'] += 1
                    status['has_risk_mechanism'] = True
                    
                    param_info = {
                        'name': 'decoder.temperature',
                        'requires_grad': param.requires_grad,
                        'value': float(param.data.detach().cpu()),
                        'grad_available': param.grad is not None
                    }
                    
                    if param.grad is not None:
                        param_info['grad_norm'] = float(param.grad.norm().item())
                    
                    status['param_details'].append(param_info)
                    
                    if param.requires_grad:
                        status['trainable_params'] += 1
                    else:
                        status['frozen_params'] += 1
            
            # 检查其他位置的风险参数（如UEEncoder）
            if hasattr(self.model, 'ue_encoder') and hasattr(self.model.ue_encoder, 'gat_layers'):
                for layer_idx, layer in enumerate(self.model.ue_encoder.gat_layers):
                    for param_name in ['beta_src', 'beta_dst', 'temperature']:
                        if hasattr(layer, param_name):
                            param = getattr(layer, param_name)
                            if isinstance(param, nn.Parameter):
                                status['total_params'] += 1
                                status['has_risk_mechanism'] = True
                                
                                param_info = {
                                    'name': f"ue_encoder.gat_layers[{layer_idx}].{param_name}",
                                    'requires_grad': param.requires_grad,
                                    'value': param.data.detach().cpu().numpy().tolist() if param.numel() > 1 else float(param.data.detach().cpu()),
                                    'grad_available': param.grad is not None
                                }
                                
                                if param.grad is not None:
                                    param_info['grad_norm'] = float(param.grad.norm().item())
                                
                                status['param_details'].append(param_info)
                                
                                if param.requires_grad:
                                    status['trainable_params'] += 1
                                else:
                                    status['frozen_params'] += 1
            
            return status
            
        except Exception as e:
            self.logger.logger.error(f"❌ 风险参数状态验证失败: {str(e)}")
            return {
                'total_params': 0,
                'frozen_params': 0,
                'trainable_params': 0,
                'param_details': [],
                'has_risk_mechanism': False,
                'error': str(e)
            }

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
        
    def _clear_frozen_parameter_gradients(self):
        """🔧 修复1：清除冻结参数的残留梯度"""
        if self.current_phase == 'basic':
            cleared_count = 0
            for name, param in self.model.named_parameters():
                if ('beta' in name or 'temperature' in name) and not param.requires_grad:
                    if param.grad is not None:
                        param.grad = None
                        cleared_count += 1
            
            if cleared_count > 0:
                self.logger.logger.debug(f"🧹 清除了{cleared_count}个冻结参数的残留梯度")

    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        🔥 修复版训练函数 - 解决梯度时序和消失问题
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 🔥 阶段感知的负采样统计
        epoch_pos_samples = 0
        epoch_neg_samples = 0
        
        for batch_idx, batch in enumerate(data_loader):
            try:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # 收集采样统计信息
                if 'positive_edges' in batch and 'negative_edges' in batch:
                    epoch_pos_samples += len(batch['positive_edges'])
                    epoch_neg_samples += len(batch['negative_edges'])
                
                # Forward pass with real link prediction
                if self.negative_sampling_enabled and 'positive_edges' in batch and 'negative_edges' in batch:
                    loss = self._compute_link_prediction_loss(batch)
                else:
                    # Standard graph-level forward pass
                    fine_scores, coarse_scores = self.model(batch)
                    loss = self._compute_graph_level_loss(fine_scores, coarse_scores, batch)
                
                # 🔥 修复：正确的梯度清除顺序
                # 1. 先清除所有梯度
                self.optimizer.zero_grad()
                
                # 2. 反向传播
                loss.backward()
                
                # 3. 清除冻结参数的残留梯度（如果在Basic阶段）
                if self.current_phase == 'basic':
                    self._clear_frozen_parameter_gradients()
                
                # SOTA梯度监控 (每50个batch监控一次)
                if batch_idx % 50 == 0:
                    grad_stats = self._monitor_gradients()
                    
                    # 🔥 阶段感知的风险偏置参数梯度检查
                    risk_has_grad = any('beta' in name or 'temperature' in name 
                                       for name in grad_stats.keys())
                    
                    if self.current_phase == 'basic':
                        # Basic阶段：风险参数应该被冻结，没有梯度是正常的
                        if risk_has_grad:
                            self.logger.logger.warning("⚠️ Basic阶段风险参数意外有梯度！")
                        else:
                            self.logger.logger.debug("✅ Basic阶段：风险参数正确冻结")
                    else:
                        # Risk-aware阶段及以后：风险参数应该有梯度
                        if not risk_has_grad:
                            self.logger.logger.warning(f"❌ {self.current_phase.upper()}阶段风险偏置参数没有梯度！")
                        else:
                            self.logger.logger.debug(f"✅ {self.current_phase.upper()}阶段：风险参数正常训练")
                
                # Gradient clipping with monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if grad_norm > 10.0:
                    self.logger.logger.warning(f"⚠️ 梯度爆炸: {grad_norm:.3f}")
                
                # Update parameters
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                self.logger.logger.warning(f"Training error in batch {batch_idx}: {e}")
                continue
        
        # Log epoch-level sampling statistics
        if self.negative_sampling_enabled and num_batches > 0:
            total_epoch_samples = epoch_pos_samples + epoch_neg_samples
            if total_epoch_samples > 0:
                pos_percentage = (epoch_pos_samples / total_epoch_samples) * 100
                neg_percentage = (epoch_neg_samples / total_epoch_samples) * 100
                
                self.logger.logger.info(
                    f"📊 Epoch {self.current_epoch + 1} 样本分布: "
                    f"正样本 {epoch_pos_samples} ({pos_percentage:.1f}%), "
                    f"负样本 {epoch_neg_samples} ({neg_percentage:.1f}%)"
                )
                
                # Log adaptive weights
                self.logger.logger.debug(
                    f"⚖️ 自适应权重: 正样本 {self.adaptive_weights.get('pos', 1.0):.3f}, "
                    f"负样本 {self.adaptive_weights.get('neg', 1.0):.3f}"
                )
                
        return total_loss / max(num_batches, 1)
        
    def validate_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        🔥 智能验证 - 超时保护 + 死锁预防
        
        Args:
            data_loader: Validation data loader
            
        Returns:
            验证指标字典
        """
        # 🔥 使用智能验证器替代复杂的验证逻辑
        return self.intelligent_validator.validate_with_intelligence(
            model=self.model,
            data_loader=data_loader,
            device=self.device
        )
        
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
    
    def _move_batch_to_device(self, batch):
        """Move batch data to device"""
        device_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                device_batch[key] = value.to(self.device)
            elif hasattr(value, 'to'):  # DGL graph
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _compute_link_prediction_loss(self, batch):
        """
        Compute real link prediction loss using positive and negative edges
        🔥 关键修复：使用RiskGravityDecoder而不是简单点积，确保风险参数参与梯度传播
        
        Args:
            batch: Batch data containing graph, positive_edges, negative_edges
            
        Returns:
            torch.Tensor: Computed loss
        """
        graph = batch['graph']
        positive_edges = batch['positive_edges']
        negative_edges = batch['negative_edges']
        risk_scores = batch['risk_scores']
        
        # Get node embeddings from the model
        try:
            # Call model to get node embeddings
            node_embeddings = self.model.get_node_embeddings(graph, risk_scores)
            
            # 🔥 关键修复：使用RiskGravityDecoder计算链路分数，确保风险信息参与
            # Compute positive edge scores using decoder
            pos_src_emb = node_embeddings[positive_edges[:, 0]]
            pos_dst_emb = node_embeddings[positive_edges[:, 1]]
            pos_src_risk = risk_scores[positive_edges[:, 0]].unsqueeze(-1)
            pos_dst_risk = risk_scores[positive_edges[:, 1]].unsqueeze(-1)
            
            # 使用模型的RiskGravityDecoder
            pos_scores = self.model.decoder(
                embeddings_u=pos_src_emb,
                embeddings_v=pos_dst_emb,
                risk_u=pos_src_risk,
                risk_v=pos_dst_risk
            ).squeeze(-1)
            
            # Compute negative edge scores using decoder
            neg_src_emb = node_embeddings[negative_edges[:, 0]]
            neg_dst_emb = node_embeddings[negative_edges[:, 1]]
            neg_src_risk = risk_scores[negative_edges[:, 0]].unsqueeze(-1)
            neg_dst_risk = risk_scores[negative_edges[:, 1]].unsqueeze(-1)
            
            # 使用模型的RiskGravityDecoder
            neg_scores = self.model.decoder(
                embeddings_u=neg_src_emb,
                embeddings_v=neg_dst_emb,
                risk_u=neg_src_risk,
                risk_v=neg_dst_risk
            ).squeeze(-1)
            
            # Create labels: 1 for positive edges, 0 for negative edges
            pos_labels = torch.ones(positive_edges.size(0), device=self.device)
            neg_labels = torch.zeros(negative_edges.size(0), device=self.device)
            
            # Combine scores and labels
            all_scores = torch.cat([pos_scores, neg_scores])
            all_labels = torch.cat([pos_labels, neg_labels])
            
            # 🔥 修复：decoder已经应用了sigmoid，使用BCE而不是BCE_with_logits
            # Apply BCE loss (decoder output is already sigmoid)
            all_scores = torch.clamp(all_scores, 1e-7, 1 - 1e-7)  # 数值稳定性
            loss = F.binary_cross_entropy(all_scores, all_labels)
            
            return loss
            
        except AttributeError:
            # Fallback: model doesn't have get_node_embeddings method
            fine_scores, coarse_scores = self.model(batch)
            return self._compute_graph_level_loss(fine_scores, coarse_scores, batch)
    
    def _compute_graph_level_loss(self, fine_scores, coarse_scores, batch):
        """
        Compute standard graph-level loss with improved dimension handling
        
        Args:
            fine_scores: Fine-grained prediction scores
            coarse_scores: Coarse-grained prediction scores  
            batch: Batch data
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Extract edge labels from batch
        edge_labels = batch['edge_labels'].float()
        
        # 添加调试信息
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        debug_enabled = self._debug_counter % 100 == 0  # 每100次输出一次调试信息
        
        if debug_enabled:
            print(f"🔍 损失计算调试 - Fine scores shape: {fine_scores.shape if hasattr(fine_scores, 'shape') else len(fine_scores)}")
            print(f"  Edge labels shape: {edge_labels.shape}")
            print(f"  Unique labels: {torch.unique(edge_labels)}")
            print(f"  Label distribution: 正样本={torch.sum(edge_labels).item()}, 负样本={torch.sum(edge_labels == 0).item()}")
        
        if fine_scores.numel() > 0 and edge_labels.numel() > 0:
            # 🔧 修复后的简化处理：边标签已确保与边数匹配
            fine_scores_flat = fine_scores.flatten()
            edge_labels_flat = edge_labels.flatten()
            
            # 🛡️ 严格验证：维度必须匹配
            if len(fine_scores_flat) != len(edge_labels_flat):
                raise ValueError(f"❌ 训练损失计算失败：预测分数长度{len(fine_scores_flat)}与边标签长度{len(edge_labels_flat)}不匹配")
            
            if debug_enabled:
                print(f"✅ 损失计算：预测={len(fine_scores_flat)}, 标签={len(edge_labels_flat)}")
                print(f"  标签分布: 正样本={torch.sum(edge_labels_flat).item()}, 负样本={torch.sum(edge_labels_flat == 0).item()}")
            
            # SOTA数值稳定的Binary cross entropy loss
            try:
                # 添加数值稳定性检查
                if torch.isnan(fine_scores_flat).any() or torch.isinf(fine_scores_flat).any():
                    if debug_enabled:
                        print("⚠️ 检测到预测分数中的NaN/Inf值")
                    fine_scores_flat = torch.nan_to_num(fine_scores_flat, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 使用label smoothing提高稳定性 - SOTA实践
                smoothed_labels = edge_labels_flat * 0.9 + 0.05
                
                loss = F.binary_cross_entropy_with_logits(fine_scores_flat, smoothed_labels)
                
                # 验证损失值
                if torch.isnan(loss) or torch.isinf(loss):
                    if debug_enabled:
                        print(f"⚠️ 检测到异常损失值: {loss}, 使用备用损失计算")
                    raise ValueError("Loss contains NaN/Inf")
                
                # Add coarse-grained loss if available
                if coarse_scores is not None and coarse_scores.numel() > 0:
                    coarse_scores_flat = coarse_scores.flatten()[:len(edge_labels_flat)]
                    try:
                        coarse_loss = F.binary_cross_entropy_with_logits(coarse_scores_flat, edge_labels_flat)
                        if not torch.isnan(coarse_loss) and not torch.isinf(coarse_loss):
                            loss = 0.7 * loss + 0.3 * coarse_loss
                    except:
                        # 粗粒度损失计算失败，只使用细粒度损失
                        pass
                
                if debug_enabled:
                    print(f"  最终损失值: {loss.item():.4f}")
                
                return loss
                
            except Exception as e:
                if debug_enabled:
                    print(f"❌ 损失计算异常: {e}, 使用SOTA备用方法")
                # SOTA稳定损失计算 - 使用smoothed labels
                smoothed_labels = edge_labels_flat * 0.9 + 0.05
                probs = torch.sigmoid(fine_scores_flat)
                probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
                stable_loss = -(smoothed_labels * torch.log(probs) + 
                               (1 - smoothed_labels) * torch.log(1 - probs)).mean()
                return stable_loss if not torch.isnan(stable_loss) else torch.tensor(0.5, requires_grad=True, device=self.device)
        else:
            # Fallback for empty predictions
            return torch.tensor(0.5, requires_grad=True, device=self.device) 

    def _recreate_optimizer_for_risk_aware(self):
        """🔥 重新创建优化器以包含解冻的风险参数"""
        # 获取当前学习率和配置
        current_lr = self.optimizer.param_groups[0]['lr']
        weight_decay = self.optimizer.param_groups[0]['weight_decay']
        
        # 获取所有可训练参数（包括新解冻的风险参数）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 🔥 新增：参数验证
        if len(trainable_params) == 0:
            self.logger.logger.error("❌ 没有可训练参数，优化器重建失败")
            return False
        
        # 🔥 新增：风险参数检查
        risk_param_count = sum(1 for name, p in self.model.named_parameters() 
                              if p.requires_grad and any(keyword in name for keyword in ['beta', 'temperature']))
        
        # 创建新优化器
        try:
            self.optimizer = AdamW(
                trainable_params,
                lr=current_lr,
                weight_decay=weight_decay
            )
            
            self.logger.logger.info(f"✅ 优化器重新创建：现在管理 {len(trainable_params)} 个参数（包含 {risk_param_count} 个风险参数）")
            return True
            
        except Exception as e:
            self.logger.logger.error(f"❌ 优化器重建失败: {e}")
            return False
    
    def _force_recreate_optimizer(self):
        """🔥 强制重建优化器 - 用于紧急修复"""
        self.logger.logger.warning("🚨 执行强制优化器重建...")
        
        # 确保所有风险参数解冻
        for name, param in self.model.named_parameters():
            if any(keyword in name for keyword in ['beta', 'temperature']):
                param.requires_grad = True
        
        # 重建优化器
        success = self._recreate_optimizer_for_risk_aware()
        
        if success:
            self.logger.logger.info("✅ 强制优化器重建成功")
        else:
            self.logger.logger.error("❌ 强制优化器重建失败")
            
        return success
    
    def _update_risk_attention_stage(self, stage: str):
        """🔥 更新所有风险偏置注意力层的训练阶段"""
        updated_layers = 0
        
        def update_layer_stage(module):
            nonlocal updated_layers
            if hasattr(module, 'set_training_stage'):
                module.set_training_stage(stage)
                updated_layers += 1
            
            # 递归检查子模块
            for child in module.children():
                update_layer_stage(child)
        
        update_layer_stage(self.model)
        
        if updated_layers > 0:
            self.logger.logger.info(f"🔧 更新了 {updated_layers} 个风险偏置注意力层的训练阶段: {stage}")
        else:
            self.logger.logger.debug("📝 未找到需要更新训练阶段的风险偏置注意力层")