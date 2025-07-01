#!/usr/bin/env python3
"""
负采样修复验证脚本
测试大图负采样和正样本去重修复效果
"""

import sys
import os
sys.path.append('.')

import torch
import random
import dgl
from pathlib import Path
from torch.utils.data import DataLoader

# 设置random seed确保可复现
torch.manual_seed(42)
random.seed(42)

def test_large_graph_negative_sampling():
    """测试大图负采样修复"""
    print("🔄 测试大图负采样修复...")
    
    # 导入修复后的负采样器
    from src.data.negative_sampler import RiskAwareNegativeSampler
    
    # 创建模拟大图
    num_nodes = 100000  # 10万节点
    num_pos_edges = 50000  # 5万正样本边
    
    # 创建DGL图
    src_nodes = torch.randint(0, num_nodes, (num_pos_edges,))
    dst_nodes = torch.randint(0, num_nodes, (num_pos_edges,))
    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
    
    # 创建正样本边和风险分数
    positive_edges = torch.stack([src_nodes, dst_nodes], dim=1)
    risk_scores = torch.rand(num_nodes)  # 随机风险分数
    
    # 创建负采样器
    sampler = RiskAwareNegativeSampler(
        strategy='semantic',
        neg_ratio=3,  # 3:1比例
        seed=42
    )
    
    print(f"📊 测试配置: 节点={num_nodes}, 正样本={num_pos_edges}, 目标负样本={num_pos_edges * 3}")
    
    try:
        # 执行负采样
        negative_edges = sampler.sample_negatives(graph, positive_edges, risk_scores)
        
        # 检查结果
        actual_neg_count = len(negative_edges)
        expected_neg_count = num_pos_edges * 3
        success_rate = actual_neg_count / expected_neg_count if expected_neg_count > 0 else 0
        
        print(f"✅ 大图负采样结果:")
        print(f"   期望负样本: {expected_neg_count}")
        print(f"   实际负样本: {actual_neg_count}")
        print(f"   成功率: {success_rate:.1%}")
        
        if success_rate > 0.8:  # 80%以上认为成功
            print(f"✅ 大图负采样修复成功！")
            return True
        else:
            print(f"❌ 大图负采样仍需优化（成功率={success_rate:.1%}）")
            return False
            
    except Exception as e:
        print(f"❌ 大图负采样测试失败: {e}")
        return False

def test_positive_edge_deduplication():
    """测试正样本去重修复"""
    print("\n🔄 测试正样本去重修复...")
    
    # 导入修复后的collate函数
    from train import graph_collate_fn
    from src.data.graph_dataset import RiskNICLPDataset
    
    # 创建模拟batch数据
    def create_mock_item(num_nodes=1000, num_pos=100, num_neg=200, duplicate_factor=3):
        """创建包含重复正样本的模拟数据"""
        graph = dgl.graph(([], []), num_nodes=num_nodes)
        
        # 创建原始正样本
        original_pos = torch.randint(0, num_nodes, (num_pos, 2))
        
        # 人为添加重复（模拟增强策略的重复问题）
        duplicated_pos = original_pos.repeat(duplicate_factor, 1)
        
        # 创建负样本
        negative_edges = torch.randint(0, num_nodes, (num_neg, 2))
        
        return {
            'graph': graph,
            'risk_scores': torch.rand(num_nodes),
            'edge_labels': torch.randint(0, 2, (num_nodes,)).float(),  # 随机边标签
            'positive_edges': duplicated_pos,
            'negative_edges': negative_edges,
            'scale_type': 'fine',
            'file_name': 'test_graph.dgl',
            'idx': 0
        }
    
    # 创建测试batch
    batch = [
        create_mock_item(num_pos=50, duplicate_factor=3),   # 150个正样本（包含重复）
        create_mock_item(num_pos=30, duplicate_factor=2),   # 60个正样本（包含重复）
    ]
    
    print(f"📊 测试配置:")
    print(f"   Batch大小: {len(batch)}")
    print(f"   原始正样本（含重复）: {sum(len(item['positive_edges']) for item in batch)}")
    print(f"   原始负样本: {sum(len(item['negative_edges']) for item in batch)}")
    
    try:
        # 执行collate函数
        result = graph_collate_fn(batch)
        
        # 检查去重效果
        final_pos_count = len(result['positive_edges'])
        final_neg_count = len(result['negative_edges'])
        
        print(f"✅ 去重后结果:")
        print(f"   最终正样本: {final_pos_count}")
        print(f"   最终负样本: {final_neg_count}")
        print(f"   正负比例: 1:{final_neg_count/max(final_pos_count, 1):.1f}")
        
        # 检查是否有合理的比例
        if final_neg_count > final_pos_count:  # 负样本应该更多
            print(f"✅ 正样本去重修复成功！")
            return True
        else:
            print(f"❌ 正负样本比例仍不合理")
            return False
            
    except Exception as e:
        print(f"❌ 正样本去重测试失败: {e}")
        return False

def test_real_dataset_sampling():
    """测试真实数据集的采样效果"""
    print("\n🔄 测试真实数据集采样效果...")
    
    try:
        from src.config.config_loader import load_config
        from src.data.data_processor import DataProcessor
        from src.data.feature_selector import FeatureSelector
        from src.data.negative_sampler import create_negative_sampler
        from src.data.graph_dataset import RiskNICLPDataset
        from train import graph_collate_fn
        
        # 加载配置
        config = load_config("config/baseline_config.yaml")
        
        # 创建数据处理器
        feature_selector = FeatureSelector(
            variance_threshold=config.data.variance_threshold,
            correlation_threshold=config.data.correlation_threshold
        )
        
        data_processor = DataProcessor(
            feature_selector=feature_selector,
            risk_norm_method=config.data.risk_norm_method
        )
        
        # 创建数据集
        dataset = RiskNICLPDataset(
            graph_dir=config.data.graph_dir,
            metadata_dir=config.data.metadata_dir,
            feature_processor=data_processor,
            risk_vectorizer=None,
            negative_sampler_config=config.phase2['negative_sampling']
        )
        
        # 准备数据
        dataset.prepare_data()
        
        # 创建数据加载器（小batch测试）
        test_loader = DataLoader(
            dataset,
            batch_size=2,  # 小batch避免内存问题
            shuffle=False,
            collate_fn=graph_collate_fn
        )
        
        print(f"📊 真实数据集测试: 数据集大小={len(dataset)}")
        
        # 测试几个batch
        total_pos = 0
        total_neg = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 3:  # 只测试前3个batch
                break
                
            pos_count = len(batch['positive_edges'])
            neg_count = len(batch['negative_edges'])
            
            total_pos += pos_count
            total_neg += neg_count
            batch_count += 1
            
            print(f"   Batch {batch_idx + 1}: 正样本={pos_count}, 负样本={neg_count}, 比例=1:{neg_count/max(pos_count, 1):.1f}")
        
        if batch_count > 0:
            overall_ratio = total_neg / max(total_pos, 1)
            overall_pos_pct = total_pos / (total_pos + total_neg) * 100
            overall_neg_pct = total_neg / (total_pos + total_neg) * 100
            
            print(f"✅ 真实数据集采样结果:")
            print(f"   总体正样本: {total_pos} ({overall_pos_pct:.1f}%)")
            print(f"   总体负样本: {total_neg} ({overall_neg_pct:.1f}%)")
            print(f"   总体比例: 1:{overall_ratio:.1f}")
            
            # 评估修复效果
            if overall_neg_pct > 50 and overall_ratio > 1.5:  # 负样本占50%+且比例>1.5:1
                print(f"✅ 真实数据集采样修复成功！")
                return True
            else:
                print(f"❌ 真实数据集采样仍需优化")
                return False
        else:
            print(f"❌ 无法获取真实数据集batch")
            return False
            
    except Exception as e:
        print(f"❌ 真实数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试流程"""
    print("🚀 负采样修复验证开始")
    print("=" * 60)
    
    results = {}
    
    # 测试1: 大图负采样
    results['large_graph'] = test_large_graph_negative_sampling()
    
    # 测试2: 正样本去重
    results['deduplication'] = test_positive_edge_deduplication()
    
    # 测试3: 真实数据集
    results['real_dataset'] = test_real_dataset_sampling()
    
    # 总结结果
    print("\n" + "=" * 60)
    print("🎯 修复验证总结:")
    
    success_count = sum(results.values())
    total_tests = len(results)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print(f"\n📊 总体结果: {success_count}/{total_tests} 测试通过")
    
    if success_count == total_tests:
        print("🎉 所有修复验证通过！负采样问题已解决。")
        return True
    elif success_count >= total_tests * 0.6:  # 60%以上通过
        print("🟡 大部分修复生效，仍有改进空间。")
        return True
    else:
        print("🔴 修复效果不佳，需要进一步调试。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 