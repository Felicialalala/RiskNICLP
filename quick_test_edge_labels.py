#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from src.data import RiskNICLPDataset, FeatureProcessor, RiskVectorizer
from src.common import load_config

print("🔍 快速测试edge_labels修复效果...")

# 加载配置
config = load_config("config/baseline_config.yaml")

# 创建数据处理器（最小配置）
feature_processor = FeatureProcessor(
    method="hybrid",
    target_dim=45,
)

risk_vectorizer = RiskVectorizer(
    num_bins=8,
    embed_dim=16,
)

# 创建数据集
dataset = RiskNICLPDataset(
    graph_dir=config.data.graph_dir,
    metadata_dir=config.data.metadata_dir,
    feature_processor=feature_processor,
    risk_vectorizer=risk_vectorizer,
    negative_sampler_config=config.phase2['negative_sampling']
)

# 准备数据
dataset.prepare_data()

print(f"📊 数据集大小: {len(dataset)}")

# 测试前3个样本
for i in range(min(3, len(dataset))):
    print(f"\n🔍 测试样本 {i}:")
    sample = dataset[i]
    
    edge_labels = sample['edge_labels']
    print(f"  ✅ edge_labels类型: {type(edge_labels)}")
    print(f"  ✅ 是否为tensor: {torch.is_tensor(edge_labels)}")
    if torch.is_tensor(edge_labels):
        print(f"  ✅ 形状: {edge_labels.shape}")
        print(f"  ✅ 数据类型: {edge_labels.dtype}")
        print(f"  ✅ 正样本数: {torch.sum(edge_labels).item()}")
    else:
        print(f"  ❌ 非tensor内容: {edge_labels}")

print(f"\n�� edge_labels测试完成！") 