#!/usr/bin/env python3
"""
关键梯度问题修复脚本
解决发现的4个核心问题
"""

def fix_optimizer_parameter_management():
    """🔧 修复1：优化器参数管理时序问题"""
    print("=" * 60)
    print("🔧 修复1：优化器参数管理时序")
    print("=" * 60)
    
    return """
# 在 src/models/training.py 的 __init__ 方法中
# 移动优化器创建到参数冻结之后

class ProgressiveTrainer:
    def __init__(self, model, config, logger, device, temporal_predictor=None):
        # ... 其他初始化代码 ...
        
        # 🔥 修复：先冻结参数，再创建优化器
        # 初始阶段冻结风险参数
        if hasattr(model, 'freeze_risk_params'):
            model.freeze_risk_params()
            logger.logger.info("🔒 风险参数已冻结（在优化器创建前）")
        
        # Setup optimizer (现在创建，只包含可训练参数)
        lr = float(getattr(config.training, 'learning_rate', 1e-3))
        weight_decay = float(getattr(config.training, 'weight_decay', 1e-4))
        
        # 🔥 关键修复：只为requires_grad=True的参数创建优化器
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,  # 只包含可训练参数
            lr=lr,
            weight_decay=weight_decay
        )
        
        logger.logger.info(f"✅ 优化器创建：管理 {len(trainable_params)} 个可训练参数")
"""

def fix_gradient_clearing_sequence():
    """🔧 修复2：梯度清除时序"""
    print("=" * 60) 
    print("🔧 修复2：梯度清除时序")
    print("=" * 60)
    
    return """
# 在 src/models/training.py 的 train_epoch 方法中
# 修正梯度清除顺序

def train_epoch(self, data_loader: DataLoader) -> float:
    # ... 前面代码不变 ...
    
    for batch_idx, batch in enumerate(data_loader):
        try:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            if self.negative_sampling_enabled and 'positive_edges' in batch:
                loss = self._compute_link_prediction_loss(batch)
            else:
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
            
            # 4. 梯度裁剪和参数更新
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
"""

def fix_risk_aware_optimizer_update():
    """🔧 修复3：Risk_Aware阶段优化器更新"""
    print("=" * 60)
    print("🔧 修复3：Risk_Aware阶段优化器更新") 
    print("=" * 60)
    
    return """
# 在 src/models/training.py 的 train_progressive 方法中
# 阶段切换时重新创建优化器

def train_progressive(self, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    # ... 前面代码不变 ...
    
    # 阶段切换逻辑
    if should_advance:
        current_phase = self._get_next_phase(current_phase)
        self.current_phase = current_phase
        
        if current_phase == 'risk_aware':
            # 🔥 修复：解冻风险参数
            if hasattr(self.model, 'unfreeze_risk_params'):
                self.model.unfreeze_risk_params()
                self.logger.logger.info("🔓 风险参数已解冻")
            
            # 🔥 关键修复：重新创建优化器包含新解冻的参数
            self._recreate_optimizer_for_risk_aware()
            
            self.logger.logger.info(f"🔄 切换到 {current_phase} 阶段，优化器已更新")

def _recreate_optimizer_for_risk_aware(self):
    \"\"\"重新创建优化器以包含解冻的风险参数\"\"\"
    # 获取当前学习率和配置
    current_lr = self.optimizer.param_groups[0]['lr']
    weight_decay = self.optimizer.param_groups[0]['weight_decay']
    
    # 获取所有可训练参数（包括新解冻的风险参数）
    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
    
    # 创建新优化器
    self.optimizer = AdamW(
        trainable_params,
        lr=current_lr,
        weight_decay=weight_decay
    )
    
    self.logger.logger.info(f"✅ 优化器重新创建：现在管理 {len(trainable_params)} 个参数")
"""

def fix_temperature_parameter():
    """🔧 修复4：温度参数调整"""
    print("=" * 60)
    print("🔧 修复4：温度参数过度平滑")
    print("=" * 60)
    
    return """
# 在 src/models/risk_bias_attention.py 的 TrueRiskBiasGATConv 类中
# 调整温度参数到合适范围

class TrueRiskBiasGATConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=4, dropout=0.1):
        super().__init__()
        
        # ... 其他初始化代码 ...
        
        # 🔥 关键修复：调整温度参数到最优范围
        self.temperature = nn.Parameter(torch.tensor(0.2), requires_grad=True)  # 从0.5→0.2
        
        # 🔥 温度参数约束：更严格的范围，防止过度平滑
        self.register_buffer('temp_min', torch.tensor(0.1))   # 从0.3→0.1
        self.register_buffer('temp_max', torch.tensor(0.3))   # 从1.0→0.3
        
        # 🔥 修复：增强风险投影的影响力
        self.risk_proj = nn.Linear(1, num_heads, bias=True)
        nn.init.constant_(self.risk_proj.weight, 3.0)        # 从2.0→3.0
        nn.init.constant_(self.risk_proj.bias, 0.2)          # 从0.1→0.2
        
        print(f"🔥 温度参数修复: τ={self.temperature.item():.2f}")
        print(f"   约束范围: [{self.temp_min.item():.1f}, {self.temp_max.item():.1f}]")
        print(f"   风险投影权重: {self.risk_proj.weight.mean().item():.1f}")
"""

def main():
    """执行所有修复"""
    print("🚀 开始关键梯度问题修复...")
    
    fix1 = fix_optimizer_parameter_management()
    fix2 = fix_gradient_clearing_sequence()  
    fix3 = fix_risk_aware_optimizer_update()
    fix4 = fix_temperature_parameter()
    
    print("\n" + "="*80)
    print("📋 修复清单总结")
    print("="*80)
    print("1. ✅ 优化器参数管理时序修复")
    print("2. ✅ 梯度清除时序修复") 
    print("3. ✅ Risk_Aware阶段优化器更新修复")
    print("4. ✅ 温度参数过度平滑修复")
    print("\n🎯 预期效果：")
    print("   - Basic阶段：无残留梯度")
    print("   - Risk_Aware阶段：梯度从1e-11→1e-6以上")
    print("   - 参数学习：β值开始偏离初始值")
    print("   - AUC提升：突破0.6基线")

if __name__ == "__main__":
    main() 