#!/usr/bin/env python3
"""
快速梯度测试 - 验证RiskNICLP-5G梯度消失修复效果
"""

import torch
import torch.nn as nn
import dgl
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.models.riskniclp_model import RiskNICLPModel
from src.models.risk_bias_attention import TrueRiskBiasGATConv
from debug_risk_gradient import register_debug_hooks, list_risk_params


def test_risk_bias_gradient():
    """测试风险偏置GAT层的梯度传播"""
    print("🧪 测试风险偏置GAT层梯度传播...")
    
    # 创建简单测试图
    g = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])))
    node_features = torch.randn(3, 64, requires_grad=True)
    risk_scores = torch.tensor([0.1, 0.5, 0.9])
    
    # 创建GAT层
    gat_layer = TrueRiskBiasGATConv(in_feats=64, out_feats=64, num_heads=4)
    
    # 解冻参数
    for param in gat_layer.parameters():
        param.requires_grad = True
    
    print(f"  风险参数状态:")
    print(f"    beta_src: requires_grad={gat_layer.beta_src.requires_grad}, 值={gat_layer.beta_src.data}")
    print(f"    beta_dst: requires_grad={gat_layer.beta_dst.requires_grad}, 值={gat_layer.beta_dst.data}")
    print(f"    temperature: requires_grad={gat_layer.temperature.requires_grad}, 值={gat_layer.temperature.data}")
    
    # 前向传播
    output = gat_layer(g, node_features, risk_scores)
    loss = output.sum()
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print(f"  梯度检查:")
    print(f"    beta_src梯度: {gat_layer.beta_src.grad}")
    print(f"    beta_dst梯度: {gat_layer.beta_dst.grad}")
    print(f"    temperature梯度: {gat_layer.temperature.grad}")
    
    # 验证梯度非零
    has_beta_src_grad = gat_layer.beta_src.grad is not None and gat_layer.beta_src.grad.norm() > 1e-8
    has_beta_dst_grad = gat_layer.beta_dst.grad is not None and gat_layer.beta_dst.grad.norm() > 1e-8
    has_temp_grad = gat_layer.temperature.grad is not None and abs(gat_layer.temperature.grad.item()) > 1e-8
    
    if has_beta_src_grad and has_beta_dst_grad and has_temp_grad:
        print("  ✅ 风险偏置GAT层梯度传播正常")
        return True
    else:
        print("  ❌ 风险偏置GAT层梯度传播异常")
        return False


def test_full_model_gradient():
    """测试完整RiskNICLP模型的梯度传播"""
    print("🧪 测试完整RiskNICLP模型梯度传播...")
    
    # 创建模型
    model = RiskNICLPModel(
        input_dim=112, feature_dim=45, hidden_dim=64,
        risk_embed_dim=16, num_risk_bins=8, num_heads=4,
        num_layers=2, dropout=0.1, temperature=0.1, gravity_epsilon=1e-3
    )
    
    # 模拟阶段切换：Basic → Risk-aware
    print("  📍 Basic阶段：冻结风险参数")
    model.freeze_risk_params()
    
    # 检查冻结状态
    risk_params = list_risk_params(model)
    frozen_count = sum(1 for _, p in risk_params if not p.requires_grad)
    print(f"    冻结参数数量: {frozen_count}/{len(risk_params)}")
    
    # 切换到Risk-aware阶段
    print("  📍 切换到Risk-aware阶段：解冻风险参数")
    model.unfreeze_risk_params()
    
    # 检查解冻状态
    trainable_count = sum(1 for _, p in risk_params if p.requires_grad)
    print(f"    可训练参数数量: {trainable_count}/{len(risk_params)}")
    
    # 注册梯度钩子
    try:
        debug_hooks = register_debug_hooks(model)
        print(f"    梯度钩子注册成功: {len(debug_hooks)} 个")
    except Exception as e:
        print(f"    梯度钩子注册失败: {e}")
        return False
    
    # 创建模拟批次数据
    mock_batch = create_mock_batch()
    
    # 前向传播
    try:
        fine_scores, coarse_scores = model(mock_batch)
        loss = fine_scores.sum() + coarse_scores.sum()
        
        # 反向传播
        loss.backward()
        
        print("  ✅ 前向和反向传播成功")
        
        # 检查风险参数梯度
        risk_grad_count = 0
        for name, param in risk_params:
            if param.grad is not None and param.grad.norm() > 1e-8:
                risk_grad_count += 1
        
        print(f"    有效梯度参数: {risk_grad_count}/{len(risk_params)}")
        
        # 清理钩子
        for hook in debug_hooks:
            hook.remove()
        
        if risk_grad_count > 0:
            print("  ✅ 完整模型梯度传播正常")
            return True
        else:
            print("  ❌ 完整模型梯度传播异常：无有效梯度")
            return False
            
    except Exception as e:
        print(f"  ❌ 前向/反向传播失败: {e}")
        return False


def create_mock_batch():
    """创建模拟批次数据"""
    # 创建异质图
    num_ue = 3
    num_flow = 10
    num_alert = 2
    
    # UE-Flow edges
    ue_flow_edges = (torch.randint(0, num_ue, (5,)), torch.randint(0, num_flow, (5,)))
    # Flow-Alert edges
    flow_alert_edges = (torch.randint(0, num_flow, (3,)), torch.randint(0, num_alert, (3,)))
    
    graph_data = {
        ('UE', 'ue_to_flow', 'Flow'): ue_flow_edges,
        ('Flow', 'flow_to_alert', 'Alert'): flow_alert_edges,
        ('Flow', 'flow_to_ue', 'UE'): (ue_flow_edges[1], ue_flow_edges[0]),  # 反向边
    }
    
    hetero_graph = dgl.heterograph(graph_data)
    
    # 节点特征
    hetero_graph.nodes['UE'].data['feat'] = torch.randn(num_ue, 61)
    hetero_graph.nodes['Flow'].data['feat'] = torch.randn(num_flow, 61)
    hetero_graph.nodes['Alert'].data['feat'] = torch.randn(num_alert, 61)
    
    # 风险分数
    risk_scores = torch.cat([
        torch.rand(num_ue) * 0.3,      # UE: 低风险
        torch.rand(num_flow) * 0.8,    # Flow: 中等风险
        torch.rand(num_alert) * 0.5    # Alert: 中等风险
    ])
    
    # 边标签
    total_edges = ue_flow_edges[0].size(0) + flow_alert_edges[0].size(0)
    edge_labels = torch.randint(0, 2, (total_edges,)).float()
    
    return {
        'graph': hetero_graph,
        'risk_scores': risk_scores,
        'edge_labels': edge_labels
    }


def main():
    """主测试函数"""
    print("🚀 RiskNICLP-5G 梯度消失修复验证")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 测试1：风险偏置GAT层
    test1_result = test_risk_bias_gradient()
    print()
    
    # 测试2：完整模型
    test2_result = test_full_model_gradient()
    print()
    
    # 总结
    print("=" * 60)
    print("📊 测试结果总结:")
    print(f"  风险偏置GAT层: {'✅ 通过' if test1_result else '❌ 失败'}")
    print(f"  完整模型梯度: {'✅ 通过' if test2_result else '❌ 失败'}")
    
    if test1_result and test2_result:
        print("\n🎉 恭喜！梯度消失问题已修复")
        print("💡 建议：")
        print("  1. 使用修复后的配置文件重新训练")
        print("  2. 监控训练过程中的梯度变化")
        print("  3. 如果仍有问题，调整学习率和风险偏置强度")
    else:
        print("\n⚠️ 仍有问题需要进一步排查")
        print("💡 建议：")
        print("  1. 检查模型架构是否正确加载")
        print("  2. 验证风险偏置参数初始化")
        print("  3. 调试特定失败的测试")
    
    return test1_result and test2_result


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 