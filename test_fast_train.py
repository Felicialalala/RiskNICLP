#!/usr/bin/env python3
"""
快速负采样性能测试脚本
"""

import os
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import dgl
from torch.utils.data import DataLoader

from src.common import load_config, set_seed, configure_reproducible_environment
from src.data import RiskNICLPDataset

def test_negative_sampling_performance():
    """测试负采样性能"""
    print("🔧 开始负采样性能测试...")
    
    # 设置环境
    configure_reproducible_environment()
    set_seed(42, deterministic=True)
    
    # 加载配置
    config = load_config("config/baseline_config.yaml")
    config.data.num_workers = 0  # 单线程避免多进程问题
    
    # 创建数据集
    dataset = RiskNICLPDataset(
        graph_dir=config.data.graph_dir,
        metadata_dir=config.data.metadata_dir,
        feature_processor=None,  # 跳过特征处理
        risk_vectorizer=None,   # 跳过风险向量化
        cache_manager=None,
        negative_sampler_config=config.phase2['negative_sampling']
    )
    
    # 准备数据
    print("📊 准备数据...")
    dataset.prepare_data()
    
    # 创建数据分割
    train_dataset, _, _ = dataset.create_splits(
        train_ratio=0.8,
        val_ratio=0.1, 
        test_ratio=0.1,
        seed=42
    )
    
    print(f"✅ 训练集大小: {len(train_dataset)}")
    
    # 测试单个样本的负采样时间
    print("⏱️ 测试单个样本负采样时间...")
    
    start_time = time.time()
    
    # 测试前5个样本
    for i in range(min(5, len(train_dataset))):
        sample_start = time.time()
        try:
            sample = train_dataset[i]
            sample_end = time.time()
            sample_time = sample_end - sample_start
            
            print(f"  样本 {i}: {sample_time:.3f}s - "
                  f"正边:{len(sample.get('positive_edges', []))}, "
                  f"负边:{len(sample.get('negative_edges', []))}")
                  
        except Exception as e:
            print(f"  样本 {i}: 失败 - {e}")
    
    total_time = time.time() - start_time
    avg_time = total_time / min(5, len(train_dataset))
    
    print(f"📈 平均采样时间: {avg_time:.3f}s/样本")
    print(f"🔥 预计单epoch时间: {avg_time * len(train_dataset) / 60:.1f}分钟")
    
    # 测试DataLoader
    print("🔄 测试DataLoader迭代...")
    
    def simple_collate_fn(batch):
        """超简化的collate函数用于性能测试"""
        return {
            'graph': dgl.batch([item['graph'] for item in batch]),
            'positive_edges': torch.cat([item.get('positive_edges', torch.empty((0, 2), dtype=torch.long)) for item in batch], dim=0),
            'negative_edges': torch.cat([item.get('negative_edges', torch.empty((0, 2), dtype=torch.long)) for item in batch], dim=0),
            'file_names': [item.get('file_name', '') for item in batch]
        }
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate_fn
    )
    
    loader_start = time.time()
    batch_count = 0
    
    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:  # 只测试前3个批次
                break
                
            batch_count += 1
            print(f"  批次 {batch_idx}: 图节点数={batch['graph'].number_of_nodes()}, "
                  f"正边数={len(batch['positive_edges'])}, 负边数={len(batch['negative_edges'])}")
    
    except Exception as e:
        print(f"❌ DataLoader测试失败: {e}")
        return False
    
    loader_time = time.time() - loader_start
    print(f"✅ DataLoader测试完成: {batch_count}个批次，耗时{loader_time:.3f}s")
    
    return True

if __name__ == "__main__":
    try:
        success = test_negative_sampling_performance()
        if success:
            print("🎉 负采样性能测试通过！可以开始正式训练了。")
        else:
            print("❌ 负采样性能测试失败！需要进一步优化。")
    except Exception as e:
        print(f"💥 测试脚本异常: {e}")
        import traceback
        traceback.print_exc() 