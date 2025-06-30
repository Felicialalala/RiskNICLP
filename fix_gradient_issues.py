#!/usr/bin/env python3
"""
风险偏置梯度问题修复脚本
按照用户提供的排查流程逐步解决问题
"""

import torch
import torch.nn as nn

def fix_gradient_timing_issue():
    """🔧 修复1：解决Basic阶段意外梯度问题（时序问题）"""
    print("=" * 60)
    print("🔧 修复1：Basic阶段梯度时序问题")
    print("=" * 60)
    
    fix_code = '''
# 在 src/models/training.py 的 train_epoch 方法中
# 第753行 self.optimizer.zero_grad() 之前添加：

def _clear_frozen_parameter_gradients(self):
    """清除冻结参数的残留梯度"""
    if self.current_phase == 'basic':
        for name, param in self.model.named_parameters():
            if ('beta' in name or 'temperature' in name) and not param.requires_grad:
                if param.grad is not None:
                    param.grad = None
                    
# 在反向传播前调用：
if self.current_phase == 'basic':
    self._clear_frozen_parameter_gradients()
    
self.optimizer.zero_grad()
loss.backward()
'''
    
    print(fix_code)
    return fix_code


def fix_risk_bias_computation():
    """🔧 修复2：风险偏置计算中的梯度消失问题"""
    print("=" * 60)
    print("🔧 修复2：风险偏置计算梯度消失")
    print("=" * 60)
    
    fix_code = '''
# 在 src/models/risk_bias_attention.py 的 TrueRiskBiasGATConv 类中
# 修复风险偏置计算逻辑：

def __init__(self, in_feats, out_feats, num_heads=4, dropout=0.1):
    super().__init__()
    
    # 🔥 关键修复：增大初始beta值，防止梯度过小
    self.beta_src = nn.Parameter(torch.ones(num_heads) * 2.0)  # 从0.5→2.0
    self.beta_dst = nn.Parameter(torch.ones(num_heads) * 2.0)  # 从0.5→2.0
    
    # 🔥 关键修复：调整温度参数范围，防止过度锐化
    self.temperature = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 从0.3→0.5
    
    # 🔥 温度参数约束：更宽松的范围
    self.register_buffer('temp_min', torch.tensor(0.3))   # 从0.1→0.3
    self.register_buffer('temp_max', torch.tensor(1.0))   # 从0.5→1.0
    
    # 🔥 修复：风险投影使用更大的初始化
    self.risk_proj = nn.Linear(1, num_heads, bias=True)  # 添加bias
    nn.init.constant_(self.risk_proj.weight, 2.0)        # 从1.0→2.0
    nn.init.constant_(self.risk_proj.bias, 0.1)          # 添加小偏置

def _forward_homogeneous(self, graph, features, risk_scores):
    # ... 前面代码不变 ...
    
    # 🔥 关键修复：直接使用β参数，不进行额外的组合
    # 修复前：
    # beta_combined = beta_src_clamped.unsqueeze(0) + beta_dst_clamped.unsqueeze(0)
    # risk_bias = risk_bias_single * beta_combined
    
    # 修复后：
    beta_src_clamped = torch.clamp(self.beta_src, self.beta_min, self.beta_max)
    beta_dst_clamped = torch.clamp(self.beta_dst, self.beta_min, self.beta_max)
    
    # 🔥 直接计算风险偏置，避免相乘导致的梯度衰减
    risk_bias_src = beta_src_clamped.unsqueeze(0) * r_src  # [E, H]
    risk_bias_dst = beta_dst_clamped.unsqueeze(0) * r_dst  # [E, H]
    risk_bias = risk_bias_src + risk_bias_dst               # [E, H]
    
    # 融合注意力：e_ij = base_attention + risk_bias
    enhanced_attention = base_attention + risk_bias
'''
    
    print(fix_code)
    return fix_code


def fix_data_variance_issue():
    """🔧 修复3：数据方差不足问题"""
    print("=" * 60)
    print("🔧 修复3：数据风险分数方差不足")
    print("=" * 60)
    
    fix_code = '''
# 在数据预处理阶段，确保风险分数有足够的方差
# 在 data_preprocessing.py 或相关文件中添加：

def enhance_risk_score_variance(risk_scores):
    """增强风险分数的方差，防止梯度消失"""
    
    # 检查当前方差
    current_var = risk_scores.var().item()
    print(f"当前风险分数方差: {current_var:.6f}")
    
    if current_var < 0.01:  # 方差过小
        print("⚠️ 风险分数方差过小，进行增强")
        
        # 方法1：分位数拉伸
        q25, q75 = torch.quantile(risk_scores, 0.25), torch.quantile(risk_scores, 0.75)
        iqr = q75 - q25
        
        if iqr > 0:
            # 拉伸到更大范围
            risk_scores = (risk_scores - q25) / iqr * 0.8 + 0.1
        else:
            # 如果IQR为0，添加噪声
            noise = torch.randn_like(risk_scores) * 0.1
            risk_scores = risk_scores + noise
            
        risk_scores = torch.clamp(risk_scores, 0.0, 1.0)
        
        enhanced_var = risk_scores.var().item()
        print(f"增强后风险分数方差: {enhanced_var:.6f}")
    
    return risk_scores

# 在模型前向传播前调用：
risk_scores = enhance_risk_score_variance(risk_scores)
'''
    
    print(fix_code)
    return fix_code


def create_unit_test():
    """🔧 修复4：创建单元测试验证修复效果"""
    print("=" * 60)
    print("🔧 修复4：梯度验证单元测试")
    print("=" * 60)
    
    test_code = '''
def test_risk_bias_gradient_fix():
    """验证风险偏置梯度修复效果"""
    from src.models.risk_bias_attention import TrueRiskBiasGATConv
    import dgl
    
    # 创建测试图
    g = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])))
    
    # 高方差风险分数
    risk_scores = torch.tensor([0.1, 0.5, 0.9])  # 方差 = 0.16
    features = torch.randn(3, 64, requires_grad=True)
    
    # 创建层
    layer = TrueRiskBiasGATConv(64, 64, num_heads=4)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    
    print(f"初始β₁: {layer.beta_src}")
    print(f"初始β₂: {layer.beta_dst}")
    print(f"初始τ: {layer.temperature}")
    
    # 前向传播
    output = layer(g, features, risk_scores)
    
    # 损失：鼓励高风险节点有更大的表示
    loss = -(output[2] * risk_scores[2]).sum() + (output[0] * risk_scores[0]).sum()
    
    print(f"损失: {loss.item():.6f}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    beta_src_grad = layer.beta_src.grad
    beta_dst_grad = layer.beta_dst.grad
    temp_grad = layer.temperature.grad
    
    print(f"β₁梯度: {beta_src_grad}")
    print(f"β₂梯度: {beta_dst_grad}")
    print(f"τ梯度: {temp_grad}")
    
    # 验证梯度不为零
    assert beta_src_grad is not None, "β₁梯度为None"
    assert beta_dst_grad is not None, "β₂梯度为None"
    assert temp_grad is not None, "τ梯度为None"
    
    # 验证梯度有意义（不是极小值）
    assert beta_src_grad.abs().max() > 1e-6, f"β₁梯度过小: {beta_src_grad.abs().max()}"
    assert beta_dst_grad.abs().max() > 1e-6, f"β₂梯度过小: {beta_dst_grad.abs().max()}"
    assert temp_grad.abs() > 1e-6, f"τ梯度过小: {temp_grad.abs()}"
    
    print("✅ 所有梯度验证通过！")
    
    return True

# 运行测试
if __name__ == "__main__":
    test_risk_bias_gradient_fix()
'''
    
    print(test_code)
    return test_code


def main():
    """主修复流程"""
    print("🚀 风险偏置梯度问题修复方案")
    print("按照用户提供的排查流程逐步解决")
    print("=" * 80)
    
    # 1. 修复时序问题
    fix_gradient_timing_issue()
    
    # 2. 修复计算问题
    fix_risk_bias_computation()
    
    # 3. 修复数据问题  
    fix_data_variance_issue()
    
    # 4. 创建验证测试
    create_unit_test()
    
    print("=" * 80)
    print("🎯 修复总结：")
    print("1. ⏰ 时序修复：清除Basic阶段冻结参数的残留梯度")
    print("2. 🧮 计算修复：增大β初始值，调整温度范围，简化风险偏置计算")
    print("3. 📊 数据修复：增强风险分数方差，防止梯度消失")
    print("4. 🧪 测试验证：创建单元测试确保修复效果")
    print("=" * 80)


if __name__ == "__main__":
    main() 