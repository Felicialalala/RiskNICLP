#!/usr/bin/env python3
"""
风险偏置梯度调试工具 - 实现用户提供的排查流程
"""

import torch
import torch.nn as nn
import itertools
from typing import Dict, List, Tuple, Optional


def list_risk_params(model):
    """列出所有风险相关参数"""
    return [(n, p) for n, p in model.named_parameters() if any(key in n for key in ['beta', 'temperature', 'risk'])]


def register_debug_hooks(model):
    """为所有风险参数注册梯度监控hook - 修复版：只为可训练参数注册"""
    hooks = []
    for name, p in list_risk_params(model):
        # 🔥 修复：只为requires_grad=True的参数注册钩子
        if p.requires_grad:
            hook = p.register_hook(lambda g, n=name: print(f"[GRAD] {n}: {g.norm().item():.6e} (范围: {g.min().item():.3e} - {g.max().item():.3e})"))
            hooks.append(hook)
        else:
            print(f"⚠️ 跳过冻结参数: {name} (requires_grad=False)")
    
    print(f"✅ 成功注册 {len(hooks)} 个梯度钩子")
    return hooks


def check_optimizer_coverage(model, optimizer):
    """1️⃣ 检查优化器是否管理了所有风险参数"""
    print("=" * 60)
    print("1️⃣ OPTIMIZER侧排查：参数覆盖检查")
    print("=" * 60)
    
    risk_names = {n for n, _ in list_risk_params(model)}
    
    # 从优化器param_groups中提取参数名
    opt_param_ids = set()
    for group in optimizer.param_groups:
        for param in group['params']:
            opt_param_ids.add(id(param))
    
    # 检查模型中的风险参数是否被优化器管理
    opt_names = set()
    missing_params = []
    
    for name, param in list_risk_params(model):
        if id(param) in opt_param_ids:
            opt_names.add(name)
        else:
            missing_params.append(name)
    
    print(f"🔍 风险参数总数: {len(risk_names)}")
    print(f"✅ 被优化器管理: {len(opt_names)}")
    print(f"❌ 未被优化器管理: {len(missing_params)}")
    
    if missing_params:
        print("⚠️ 未被优化器管理的风险参数:")
        for param in missing_params:
            print(f"  - {param}")
        return False
    else:
        print("✅ 所有风险参数都被优化器正确管理")
        return True


def analyze_risk_data_variance(batch):
    """4️⃣ 数据侧排查：风险方差检查"""
    print("=" * 60)
    print("4️⃣ 数据侧排查：风险方差 & 节点对差异")
    print("=" * 60)
    
    if 'risk_scores' not in batch:
        print("❌ batch中没有risk_scores")
        return False
    
    risk_scores = batch['risk_scores']
    
    print(f"🔍 风险分数统计:")
    print(f"  形状: {risk_scores.shape}")
    print(f"  最小值: {risk_scores.min().item():.6f}")
    print(f"  最大值: {risk_scores.max().item():.6f}")
    print(f"  均值: {risk_scores.mean().item():.6f}")
    print(f"  方差: {risk_scores.var().item():.6f}")
    print(f"  标准差: {risk_scores.std().item():.6f}")
    
    # 检查方差是否足够
    if risk_scores.var().item() < 1e-3:
        print("❌ 风险分数方差过小 (<1e-3)，可能导致梯度消失")
        return False
    
    # 检查边对的风险差异
    if 'positive_edges' in batch and len(batch['positive_edges']) > 0:
        pos_edges = batch['positive_edges']
        if len(pos_edges) > 0:
            src_risks = risk_scores[pos_edges[:, 0]]
            dst_risks = risk_scores[pos_edges[:, 1]]
            
            print(f"🔍 正样本边的风险分析:")
            print(f"  源节点风险方差: {src_risks.var().item():.6f}")
            print(f"  目标节点风险方差: {dst_risks.var().item():.6f}")
            print(f"  边风险差异: {(src_risks - dst_risks).abs().mean().item():.6f}")
            
            # 检查是否有风险差异
            risk_diff = (src_risks - dst_risks).abs().mean().item()
            if risk_diff < 1e-4:
                print("❌ 边的风险差异过小，可能无法产生有效梯度")
                return False
    
    print("✅ 风险数据方差检查通过")
    return True


def trace_forward_path(model, batch, target_param_name="beta_src"):
    """2️⃣ 前向图排查：追踪风险参数计算路径"""
    print("=" * 60)
    print("2️⃣ 前向图排查：风险参数计算路径追踪")
    print("=" * 60)
    
    # 找到目标参数
    target_param = None
    for name, param in model.named_parameters():
        if target_param_name in name:
            target_param = param
            print(f"🎯 追踪参数: {name}")
            break
    
    if target_param is None:
        print(f"❌ 未找到参数 {target_param_name}")
        return False
    
    # 启用梯度检查
    with torch.autograd.set_detect_anomaly(True):
        try:
            # 前向传播
            print("🔄 执行前向传播...")
            output = model(batch)
            
            if isinstance(output, tuple):
                loss = output[0].sum()  # 简化的loss计算
            else:
                loss = output.sum()
            
            print(f"✅ 前向传播成功，输出: {type(output)}")
            print(f"📊 Loss值: {loss.item():.6f}")
            
            # 反向传播
            print("🔄 执行反向传播...")
            loss.backward()
            
            # 检查目标参数的梯度
            if target_param.grad is not None:
                grad_norm = target_param.grad.norm().item()
                print(f"✅ {target_param_name} 梯度范数: {grad_norm:.6e}")
                
                if grad_norm < 1e-6:
                    print(f"⚠️ 梯度过小，可能存在数值问题")
                    return False
                else:
                    print(f"✅ 梯度正常")
                    return True
            else:
                print(f"❌ {target_param_name} 没有梯度")
                return False
                
        except Exception as e:
            print(f"❌ 前向/反向传播失败: {e}")
            return False


def minimal_risk_bias_test():
    """5️⃣ 极简单元测试：验证风险偏置机制"""
    print("=" * 60)  
    print("5️⃣ 极简单元测试：风险偏置机制验证")
    print("=" * 60)
    
    from src.models.risk_bias_attention import TrueRiskBiasGATConv
    import dgl
    
    # 创建简单的测试图：2个节点，1条边
    print("🔄 创建测试图: 2节点，1边")
    g = dgl.graph((torch.tensor([0]), torch.tensor([1])))
    
    # 节点特征：2个节点，64维
    node_features = torch.randn(2, 64, requires_grad=True)
    
    # 风险分数：node0=0.1 (低风险), node1=0.9 (高风险)
    risk_scores = torch.tensor([0.1, 0.9], requires_grad=False)
    
    print(f"📊 节点特征: {node_features.shape}")
    print(f"📊 风险分数: {risk_scores}")
    
    # 创建风险偏置GAT层
    gat_layer = TrueRiskBiasGATConv(in_feats=64, out_feats=64, num_heads=4)
    optimizer = torch.optim.Adam(gat_layer.parameters(), lr=1e-3)
    
    print(f"🔍 初始β₁: {gat_layer.beta_src}")
    print(f"🔍 初始β₂: {gat_layer.beta_dst}")
    print(f"🔍 初始τ: {gat_layer.temperature}")
    
    # 前向传播
    output = gat_layer(g, node_features, risk_scores)
    
    # 简单的损失：希望高风险节点的嵌入更大
    loss = -output[1].sum() + output[0].sum()  # 鼓励node1(高风险)输出更大
    
    print(f"📊 输出: {output.shape}")
    print(f"📊 损失: {loss.item():.6f}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    beta_src_grad = gat_layer.beta_src.grad
    beta_dst_grad = gat_layer.beta_dst.grad
    temp_grad = gat_layer.temperature.grad
    
    print(f"🔍 β₁梯度: {beta_src_grad}")
    print(f"🔍 β₂梯度: {beta_dst_grad}")
    print(f"🔍 τ梯度: {temp_grad}")
    
    # 验证梯度
    has_grad = (
        beta_src_grad is not None and beta_src_grad.abs().sum() > 1e-6 and
        beta_dst_grad is not None and beta_dst_grad.abs().sum() > 1e-6 and
        temp_grad is not None and temp_grad.abs() > 1e-6
    )
    
    if has_grad:
        print("✅ 极简测试通过：风险偏置参数接收到有效梯度")
        return True
    else:
        print("❌ 极简测试失败：风险偏置参数梯度异常")
        return False


def comprehensive_gradient_diagnosis(model, batch, optimizer):
    """综合梯度诊断流程"""
    print("\n" + "="*80)
    print("🔥 风险偏置梯度综合诊断开始")
    print("="*80)
    
    results = {}
    
    # 1. 优化器覆盖检查
    results['optimizer_coverage'] = check_optimizer_coverage(model, optimizer)
    
    # 2. 数据方差检查
    results['data_variance'] = analyze_risk_data_variance(batch)
    
    # 3. 前向路径追踪
    results['forward_path'] = trace_forward_path(model, batch)
    
    # 4. 极简测试
    results['minimal_test'] = minimal_risk_bias_test()
    
    # 汇总结果
    print("\n" + "="*80)
    print("🎯 诊断结果汇总")
    print("="*80)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20s}: {status}")
    
    print(f"\n📊 总体通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！梯度传播路径正常")
    else:
        print("⚠️ 部分测试失败，需要进一步排查")
        
        # 给出具体建议
        if not results['optimizer_coverage']:
            print("💡 建议：检查优化器参数组，确保包含所有风险参数")
        if not results['data_variance']:
            print("💡 建议：增加风险分数的多样性，使用分层采样")
        if not results['forward_path']:
            print("💡 建议：检查模型forward方法，确保风险参数参与计算")
        if not results['minimal_test']:
            print("💡 建议：检查TrueRiskBiasGATConv实现，可能存在数值计算问题")
    
    return results


if __name__ == "__main__":
    print("🔥 风险偏置梯度调试工具")
    print("使用方法：在train.py中导入并调用 comprehensive_gradient_diagnosis(model, batch, optimizer)") 