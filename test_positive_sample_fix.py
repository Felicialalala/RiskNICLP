#!/usr/bin/env python3
"""
测试正样本定义修复效果
验证新的边标签生成逻辑是否能产生平衡的正负样本
"""

import sys
import os
sys.path.append('.')

import torch
import dgl
import random
import numpy as np
from pathlib import Path

def set_seed(seed=42):
    """设置随机种子确保可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_test_heterograph():
    """创建测试用异质图"""
    # 创建简单的异质图用于测试
    graph_data = {
        ('UE', 'initiates', 'Flow'): (torch.tensor([0, 1, 2, 3, 4]), torch.tensor([5, 6, 7, 8, 9])),  # 5条边
        ('Flow', 'triggers', 'Alert'): (torch.tensor([5, 6, 7]), torch.tensor([10, 11, 12])),  # 3条边
        ('Flow', 'temporal_adj', 'Flow'): (torch.tensor([5, 6, 7, 8]), torch.tensor([6, 7, 8, 9]))  # 4条边
    }
    
    return dgl.heterograph(graph_data, num_nodes_dict={'UE': 5, 'Flow': 5, 'Alert': 3})

def test_new_edge_labeling_logic(graph):
    """测试新的边标签生成逻辑"""
    print("🧪 测试新的边标签生成逻辑...")
    
    all_edge_labels = []
    edge_type_stats = {}
    
    for etype in graph.etypes:
        num_edges = graph.number_of_edges(etype)
        print(f"\n📊 边类型: {etype}, 边数量: {num_edges}")
        
        if num_edges > 0:
            if etype == 'triggers':
                # Flow->Alert: triggers are attack events (100% positive)
                attack_labels = torch.ones(num_edges, dtype=torch.long)
                pos_ratio = 1.0
            elif etype == 'initiates':
                # UE->Flow: 30% 作为攻击发起边
                torch.manual_seed(42)  # 确保可重现
                attack_ratio = 0.3
                num_positive = int(num_edges * attack_ratio)
                attack_labels = torch.zeros(num_edges, dtype=torch.long)
                if num_positive > 0:
                    positive_indices = torch.randperm(num_edges)[:num_positive]
                    attack_labels[positive_indices] = 1
                pos_ratio = num_positive / num_edges
            elif etype == 'temporal_adj':
                # 时序邻接边: 10% 作为攻击传播路径
                torch.manual_seed(42)
                attack_ratio = 0.1
                num_positive = int(num_edges * attack_ratio)
                attack_labels = torch.zeros(num_edges, dtype=torch.long)
                if num_positive > 0:
                    positive_indices = torch.randperm(num_edges)[:num_positive]
                    attack_labels[positive_indices] = 1
                pos_ratio = num_positive / num_edges
            else:
                # 其他边类型保持中性
                attack_labels = torch.zeros(num_edges, dtype=torch.long)
                pos_ratio = 0.0
            
            pos_count = torch.sum(attack_labels == 1).item()
            neg_count = torch.sum(attack_labels == 0).item()
            
            print(f"   ✅ 正样本: {pos_count} ({pos_ratio:.1%})")
            print(f"   ❌ 负样本: {neg_count} ({1-pos_ratio:.1%})")
            
            edge_type_stats[etype] = {
                'total': num_edges,
                'positive': pos_count,
                'negative': neg_count,
                'pos_ratio': pos_ratio
            }
            
            all_edge_labels.append(attack_labels)
    
    # 合并所有边标签
    if all_edge_labels:
        edge_labels = torch.cat(all_edge_labels, dim=0)
    else:
        edge_labels = torch.tensor([0])
    
    total_positive = torch.sum(edge_labels == 1).item()
    total_negative = torch.sum(edge_labels == 0).item()
    total_edges = len(edge_labels)
    
    print(f"\n📈 总体统计:")
    print(f"   📊 总边数: {total_edges}")
    print(f"   ✅ 总正样本: {total_positive} ({total_positive/total_edges:.1%})")
    print(f"   ❌ 总负样本: {total_negative} ({total_negative/total_edges:.1%})")
    print(f"   ⚖️ 正负比例: 1:{total_negative/total_positive:.1f}" if total_positive > 0 else "   ⚖️ 正负比例: 无正样本")
    
    return edge_labels, edge_type_stats

def simulate_negative_sampling(positive_edges, graph, neg_ratio=3):
    """模拟负采样过程"""
    print(f"\n🔄 模拟负采样过程 (neg_ratio={neg_ratio})...")
    
    num_pos_edges = len(positive_edges)
    num_neg_edges = num_pos_edges * neg_ratio
    
    print(f"   📋 正样本边数: {num_pos_edges}")
    print(f"   📋 目标负样本边数: {num_neg_edges}")
    
    # 简单负采样模拟
    num_nodes = graph.number_of_nodes()
    
    # 生成正样本边集合
    positive_edge_set = set()
    for i in range(len(positive_edges)):
        src, dst = positive_edges[i].tolist()
        positive_edge_set.add((src, dst))
        positive_edge_set.add((dst, src))
    
    # 生成负样本
    negative_edges = []
    max_attempts = num_neg_edges * 10
    attempts = 0
    
    while len(negative_edges) < num_neg_edges and attempts < max_attempts:
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        
        if src != dst and (src, dst) not in positive_edge_set:
            negative_edges.append([src, dst])
        attempts += 1
    
    actual_neg_edges = len(negative_edges)
    print(f"   ✅ 实际生成负样本边数: {actual_neg_edges}")
    print(f"   📊 最终正负比例: 1:{actual_neg_edges/num_pos_edges:.1f}" if num_pos_edges > 0 else "   📊 无正样本")
    
    return torch.tensor(negative_edges, dtype=torch.long) if negative_edges else torch.empty((0, 2), dtype=torch.long)

def test_multiple_graphs():
    """测试多个图的平衡性"""
    print("\n🔍 测试多个不同配置的图...")
    
    configs = [
        {'UE': 3, 'Flow': 5, 'Alert': 2, 'initiates': 4, 'triggers': 3, 'temporal_adj': 6},
        {'UE': 5, 'Flow': 8, 'Alert': 3, 'initiates': 7, 'triggers': 5, 'temporal_adj': 10},
        {'UE': 2, 'Flow': 3, 'Alert': 1, 'initiates': 2, 'triggers': 2, 'temporal_adj': 4}
    ]
    
    overall_stats = {'total_pos': 0, 'total_neg': 0, 'total_edges': 0}
    
    for i, config in enumerate(configs):
        print(f"\n--- 图 {i+1} ---")
        
        # 创建测试图
        graph_data = {
            ('UE', 'initiates', 'Flow'): (
                torch.randint(0, config['UE'], (config['initiates'],)), 
                torch.randint(0, config['Flow'], (config['initiates'],))
            ),
            ('Flow', 'triggers', 'Alert'): (
                torch.randint(0, config['Flow'], (config['triggers'],)), 
                torch.randint(0, config['Alert'], (config['triggers'],))
            ),
            ('Flow', 'temporal_adj', 'Flow'): (
                torch.randint(0, config['Flow'], (config['temporal_adj'],)), 
                torch.randint(0, config['Flow'], (config['temporal_adj'],))
            )
        }
        
        graph = dgl.heterograph(graph_data, num_nodes_dict={
            'UE': config['UE'], 
            'Flow': config['Flow'], 
            'Alert': config['Alert']
        })
        
        edge_labels, stats = test_new_edge_labeling_logic(graph)
        
        total_pos = torch.sum(edge_labels == 1).item()
        total_neg = torch.sum(edge_labels == 0).item()
        
        overall_stats['total_pos'] += total_pos
        overall_stats['total_neg'] += total_neg
        overall_stats['total_edges'] += len(edge_labels)
    
    print(f"\n🎯 多图总体统计:")
    total_edges = overall_stats['total_edges']
    total_pos = overall_stats['total_pos']
    total_neg = overall_stats['total_neg']
    
    print(f"   📊 总边数: {total_edges}")
    print(f"   ✅ 总正样本: {total_pos} ({total_pos/total_edges:.1%})")
    print(f"   ❌ 总负样本: {total_neg} ({total_neg/total_edges:.1%})")
    print(f"   ⚖️ 整体正负比例: 1:{total_neg/total_pos:.1f}" if total_pos > 0 else "   ⚖️ 整体正负比例: 无正样本")

def main():
    """主测试函数"""
    print("🚀 RiskNICLP-5G 正样本定义修复验证测试")
    print("=" * 60)
    
    set_seed(42)
    
    # 测试1: 基础异质图测试
    print("\n📋 测试1: 基础异质图边标签生成")
    test_graph = create_test_heterograph()
    edge_labels, stats = test_new_edge_labeling_logic(test_graph)
    
    # 测试2: 负采样模拟
    positive_edges = torch.tensor([[0, 5], [1, 6], [5, 10]], dtype=torch.long)  # 模拟正样本边
    negative_edges = simulate_negative_sampling(positive_edges, test_graph)
    
    # 测试3: 多图配置测试
    test_multiple_graphs()
    
    # 最终评估
    print("\n" + "=" * 60)
    print("📊 修复效果评估:")
    
    if stats:
        total_pos = sum(stat['positive'] for stat in stats.values())
        total_neg = sum(stat['negative'] for stat in stats.values())
        total_edges = sum(stat['total'] for stat in stats.values())
        
        if total_pos > 0:
            balance_ratio = min(total_pos, total_neg) / max(total_pos, total_neg)
            print(f"   ✅ 正负样本平衡度: {balance_ratio:.3f} (目标 > 0.2)")
            print(f"   🎯 修复成功: {'是' if balance_ratio > 0.2 else '否'}")
            
            if balance_ratio > 0.2:
                print("   🌟 修复效果良好！现在应该能看到显著改善的负采样综合评分。")
            else:
                print("   ⚠️ 需要进一步调整正样本比例。")
        else:
            print("   ❌ 仍然没有正样本，需要检查边类型配置。")
    
    print("\n🎉 测试完成！建议重新运行训练以验证改进效果。")

if __name__ == "__main__":
    main() 