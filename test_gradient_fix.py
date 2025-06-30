#!/usr/bin/env python3
"""
简单的梯度修复验证测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    import dgl
    from src.models.risk_bias_attention import TrueRiskBiasGATConv
    
    def test_gradient_fix():
        """验证风险偏置梯度修复效果"""
        print("🧪 测试风险偏置梯度修复...")
        
        # 创建测试图：3个节点，3条边
        g = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])))
        
        # 高方差风险分数
        risk_scores = torch.tensor([0.1, 0.5, 0.9])  # 方差 = 0.16
        features = torch.randn(3, 64, requires_grad=True)
        
        print(f"风险分数方差: {risk_scores.var().item():.6f}")
        print(f"风险分数范围: [{risk_scores.min().item():.1f}, {risk_scores.max().item():.1f}]")
        
        # 创建修复版GAT层
        layer = TrueRiskBiasGATConv(64, 64, num_heads=4)
        optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
        
        print(f"\n🔍 初始参数值:")
        print(f"β₁: {layer.beta_src.detach()}")
        print(f"β₂: {layer.beta_dst.detach()}")
        print(f"τ: {layer.temperature.item():.3f}")
        
        # 前向传播
        output = layer(g, features, risk_scores)
        
        # 损失：鼓励高风险节点有更大的表示
        loss = -(output[2] * risk_scores[2]).sum() + (output[0] * risk_scores[0]).sum()
        
        print(f"\n📊 前向传播结果:")
        print(f"输出形状: {output.shape}")
        print(f"损失值: {loss.item():.6f}")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 检查梯度
        beta_src_grad = layer.beta_src.grad
        beta_dst_grad = layer.beta_dst.grad
        temp_grad = layer.temperature.grad
        
        print(f"\n🔍 梯度检查:")
        print(f"β₁梯度: {beta_src_grad}")
        print(f"β₂梯度: {beta_dst_grad}")  
        print(f"τ梯度: {temp_grad}")
        
        # 验证结果
        success = True
        issues = []
        
        if beta_src_grad is None:
            issues.append("β₁梯度为None")
            success = False
        elif beta_src_grad.abs().max() < 1e-6:
            issues.append(f"β₁梯度过小: {beta_src_grad.abs().max():.2e}")
            success = False
            
        if beta_dst_grad is None:
            issues.append("β₂梯度为None") 
            success = False
        elif beta_dst_grad.abs().max() < 1e-6:
            issues.append(f"β₂梯度过小: {beta_dst_grad.abs().max():.2e}")
            success = False
            
        if temp_grad is None:
            issues.append("τ梯度为None")
            success = False
        elif temp_grad.abs() < 1e-6:
            issues.append(f"τ梯度过小: {temp_grad.abs():.2e}")
            success = False
        
        print(f"\n🎯 测试结果:")
        if success:
            print("✅ 所有梯度验证通过！修复成功")
            print(f"   β₁梯度范数: {beta_src_grad.norm().item():.2e}")
            print(f"   β₂梯度范数: {beta_dst_grad.norm().item():.2e}")
            print(f"   τ梯度绝对值: {temp_grad.abs().item():.2e}")
        else:
            print("❌ 梯度验证失败:")
            for issue in issues:
                print(f"   - {issue}")
                
        return success
    
    if __name__ == "__main__":
        test_gradient_fix()
        
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在正确的环境中运行测试")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc() 