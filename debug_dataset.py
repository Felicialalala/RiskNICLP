#!/usr/bin/env python3
"""调试数据集加载问题"""

import sys
sys.path.append('src')
import dgl
import torch
from src.data.graph_dataset import RiskNICLPDataset
from src.data import FeatureProcessor, RiskVectorizer

def debug_dataset():
    print('🔍 开始调试数据集加载...')
    
    # 创建数据集实例
    feature_processor = FeatureProcessor()
    risk_vectorizer = RiskVectorizer()
    
    dataset = RiskNICLPDataset(
        graph_dir='data/processed/graphs',
        metadata_dir='data/processed/metadata',
        feature_processor=feature_processor,
        risk_vectorizer=risk_vectorizer
    )
    
    # 准备数据集
    dataset.prepare_data()
    print(f'数据集大小: {len(dataset)}')
    
    # 尝试加载第一个样本并跟踪每一步
    try:
        print('📊 开始加载样本 0...')
        sample = dataset[0]
        print('✅ 样本加载成功!')
        print(f'图类型: {type(sample["graph"])}')
        print(f'风险分数类型: {type(sample["risk_scores"])}, 形状: {sample["risk_scores"].shape}')
        print(f'边标签类型: {type(sample["edge_labels"])}, 形状: {sample["edge_labels"].shape}')
        
    except Exception as e:
        print(f'❌ 样本加载失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dataset() 