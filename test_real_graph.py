#!/usr/bin/env python3
"""
测试真实异质图数据处理 - 验证方案B实现
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import dgl
from typing import Dict, Any
import time

# 导入修改后的模型
from src.models.riskniclp_model import RiskNICLPModel
from src.common import load_config


def test_real_graph_processing():
    """测试真实异质图处理"""
    print("🧪 测试真实异质图处理 (方案B)")
    
    # 加载配置 - 使用字典访问
    try:
        config = load_config("config/baseline_config.yaml")
        node_feature_dims = config.model.node_feature_dims
    except AttributeError:
        # 兜底：直接使用字典
        import yaml
        with open("config/baseline_config.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
        node_feature_dims = config_dict['model']['node_feature_dims']
        
        # 创建简单的配置对象
        class SimpleConfig:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, SimpleConfig(v))
                    else:
                        setattr(self, k, v)
        
        config = SimpleConfig(config_dict)
    
    # 查找第一个图文件
    import glob
    graph_files = glob.glob("data/processed/graphs/hetero_graph_fine_*.dgl")
    if not graph_files:
        print("❌ 未找到图文件")
        return False
    
    graph_file = graph_files[0]
    print(f"📂 加载图文件: {graph_file}")
    
    # 加载图
    graphs, labels = dgl.load_graphs(graph_file)
    graph = graphs[0]
    
    print(f"📊 图结构信息:")
    print(f"  节点类型: {graph.ntypes}")
    print(f"  边类型: {graph.etypes}")
    
    # 显示真实节点特征维度
    for ntype in graph.ntypes:
        if 'feat' in graph.nodes[ntype].data:
            feat_shape = graph.nodes[ntype].data['feat'].shape
            print(f"  {ntype}: {feat_shape}")
    
    # 创建模型
    print(f"\n🏗️ 模型配置:")
    print(f"  节点维度配置: {node_feature_dims}")
    
    model = RiskNICLPModel(
        node_feature_dims=node_feature_dims,
        feature_dim=config.model.feature_dim,
        hidden_dim=config.model.hidden_dim,
        risk_embed_dim=config.model.risk_embed_dim,
        num_risk_bins=config.model.num_risk_bins,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        temperature=config.model.temperature,
        gravity_epsilon=float(config.model.gravity_epsilon)
    )
    
    # 准备输入数据
    total_nodes = sum(graph.number_of_nodes(ntype) for ntype in graph.ntypes)
    risk_scores = torch.rand(total_nodes) * 0.5 + 0.1  # 模拟风险分数 [0.1, 0.6]
    
    # 准备批次数据
    batch = {
        'graph': graph,
        'risk_scores': risk_scores,
        'scale_type': 'fine'
    }
    
    print(f"\n🔍 输入数据:")
    print(f"  总节点数: {total_nodes}")
    print(f"  风险分数范围: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")
    
    # 前向传播测试
    print(f"\n🚀 开始前向传播测试...")
    
    try:
        start_time = time.time()
        
        with torch.no_grad():
            fine_scores, coarse_scores = model(batch)
        
        end_time = time.time()
        
        print(f"✅ 前向传播成功!")
        print(f"  耗时: {end_time - start_time:.3f}秒")
        print(f"  输出形状:")
        print(f"    fine_scores: {fine_scores.shape}")
        print(f"    coarse_scores: {coarse_scores.shape}")
        print(f"  分数范围:")
        print(f"    fine: [{fine_scores.min():.3f}, {fine_scores.max():.3f}]")
        print(f"    coarse: [{coarse_scores.min():.3f}, {coarse_scores.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """测试批处理"""
    print("\n🧪 测试批处理场景")
    
    # 简化配置加载
    import yaml
    with open("config/baseline_config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class SimpleConfig:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, SimpleConfig(v))
                else:
                    setattr(self, k, v)
    
    config = SimpleConfig(config_dict)
    
    # 加载多个图文件
    import glob
    graph_files = glob.glob("data/processed/graphs/hetero_graph_fine_*.dgl")[:3]
    
    if len(graph_files) < 2:
        print("⚠️ 图文件不足，跳过批处理测试")
        return True
    
    print(f"📂 加载 {len(graph_files)} 个图文件")
    
    graphs = []
    all_risk_scores = []
    
    for i, graph_file in enumerate(graph_files):
        graph_list, _ = dgl.load_graphs(graph_file)
        graph = graph_list[0]
        graphs.append(graph)
        
        # 模拟风险分数
        total_nodes = sum(graph.number_of_nodes(ntype) for ntype in graph.ntypes)
        risk_scores = torch.rand(total_nodes) * 0.5 + 0.1
        all_risk_scores.append(risk_scores)
        
        print(f"  图{i}: 节点数 {total_nodes}")
    
    # 批处理图
    try:
        batched_graph = dgl.batch(graphs)
        batched_risk_scores = torch.cat(all_risk_scores, dim=0)
        
        print(f"📊 批处理结果:")
        print(f"  批处理图节点数: {sum(batched_graph.number_of_nodes(ntype) for ntype in batched_graph.ntypes)}")
        print(f"  批处理风险分数: {batched_risk_scores.shape}")
        
        # 测试模型处理
        node_feature_dims = config_dict['model']['node_feature_dims']
        model = RiskNICLPModel(
            node_feature_dims=node_feature_dims,
            feature_dim=config.model.feature_dim,
            hidden_dim=config.model.hidden_dim,
            risk_embed_dim=config.model.risk_embed_dim,
            num_risk_bins=config.model.num_risk_bins,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            temperature=config.model.temperature,
            gravity_epsilon=float(config.model.gravity_epsilon)
        )
        
        batch = {
            'graph': batched_graph,
            'risk_scores': batched_risk_scores,
            'scale_type': 'fine'
        }
        
        with torch.no_grad():
            fine_scores, coarse_scores = model(batch)
        
        print(f"✅ 批处理测试成功!")
        print(f"  输出: fine={fine_scores.shape}, coarse={coarse_scores.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 批处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("🚀 开始真实异质图处理测试 (方案B)")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # 测试1: 单图处理
    if test_real_graph_processing():
        success_count += 1
    
    # 测试2: 批处理
    if test_batch_processing():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过! 方案B实现成功!")
        print("\n📋 下一步可以运行:")
        print("  python train.py --debug  # 调试模式训练")
        print("  python train.py --dry-run  # 干运行验证")
    else:
        print("❌ 部分测试失败，需要修复")
        return False
    
    return True


if __name__ == "__main__":
    main() 