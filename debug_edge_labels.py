#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data import RiskNICLPDataset, FeatureProcessor, RiskVectorizer
from src.common import load_config

# 加载配置
config = load_config("config/baseline_config.yaml")

# 创建数据处理器
feature_processor = FeatureProcessor(
    method=config.data.feature_selection_method,
    variance_threshold=config.data.variance_threshold,
    correlation_threshold=config.data.correlation_threshold,
    target_dim=config.model.feature_dim,
)

risk_vectorizer = RiskVectorizer(
    num_bins=config.model.num_risk_bins,
    embed_dim=config.model.risk_embed_dim,
    normalization=config.data.risk_norm_method,
    adaptive_binning=config.data.adaptive_binning,
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

print("🔍 检查前3个样本的edge_labels类型...")
for i in range(min(3, len(dataset))):
    sample = dataset[i]
    edge_labels = sample['edge_labels']
    print(f"样本{i}:")
    print(f"  edge_labels类型: {type(edge_labels)}")
    print(f"  edge_labels内容: {edge_labels}")
    if hasattr(edge_labels, '__len__'):
        print(f"  edge_labels长度: {len(edge_labels)}")
    print(f"  所有键: {sample.keys()}")
    print() 