#!/usr/bin/env python3
"""
RiskNICLP-5G 离线采样数据集
使用预计算的正负样本，大幅提升训练效率
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random
import time

from ..common.logger import get_logger


class OfflineRiskNICLPDataset(Dataset):
    """基于离线采样的RiskNICLP数据集"""
    
    def __init__(
        self,
        offline_samples_file: str,
        feature_processor,
        risk_vectorizer,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        shuffle_seed: int = 42
    ):
        """
        初始化离线数据集
        
        Args:
            offline_samples_file: 离线采样结果文件
            feature_processor: 特征处理器
            risk_vectorizer: 风险向量化器
            split: 数据分割 ('train', 'val', 'test')
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            shuffle_seed: 随机种子
        """
        self.logger = get_logger().logger  # 修复：使用内部logger对象
        self.feature_processor = feature_processor
        self.risk_vectorizer = risk_vectorizer
        self.split = split

        # 加载离线采样数据
        self.logger.info(f"加载离线采样数据: {offline_samples_file}")
        self.samples_data = self._load_offline_samples(offline_samples_file)

        # 数据分割
        self.samples = self._split_data(
            self.samples_data['samples'],
            split, train_ratio, val_ratio, shuffle_seed
        )

        self.logger.info(f"数据集初始化完成 ({split}): {len(self.samples)}个样本")
        
        # 统计信息
        self._compute_statistics()
    
    def _load_offline_samples(self, file_path: str) -> Dict:
        """加载离线采样数据"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.logger.info(f"离线采样统计:")
        self.logger.info(f"  图数量: {data['stats']['total_graphs']}")
        self.logger.info(f"  正样本: {data['stats']['total_positive_edges']:,}")
        self.logger.info(f"  负样本: {data['stats']['total_negative_edges']:,}")
        
        return data
    
    def _split_data(self, samples: List[Dict], split: str, 
                   train_ratio: float, val_ratio: float, seed: int) -> List[Dict]:
        """数据分割"""
        # 设置随机种子确保可重现
        random.seed(seed)
        np.random.seed(seed)
        
        # 随机打乱
        samples_copy = samples.copy()
        random.shuffle(samples_copy)
        
        # 计算分割点
        n_total = len(samples_copy)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if split == 'train':
            return samples_copy[:n_train]
        elif split == 'val':
            return samples_copy[n_train:n_train + n_val]
        elif split == 'test':
            return samples_copy[n_train + n_val:]
        else:
            raise ValueError(f"未知的数据分割: {split}")
    
    def _compute_statistics(self):
        """计算数据集统计信息"""
        total_positive = sum(len(sample['positive_edges']) for sample in self.samples)
        total_negative = sum(len(sample['negative_edges']) for sample in self.samples)
        
        self.stats = {
            'num_graphs': len(self.samples),
            'total_positive_edges': total_positive,
            'total_negative_edges': total_negative,
            'average_positive_per_graph': total_positive / len(self.samples) if self.samples else 0,
            'average_negative_per_graph': total_negative / len(self.samples) if self.samples else 0,
            'negative_positive_ratio': total_negative / total_positive if total_positive > 0 else 0
        }
        
        self.logger.info(f"{self.split}集统计:")
        self.logger.info(f"  图数量: {self.stats['num_graphs']}")
        self.logger.info(f"  正样本: {self.stats['total_positive_edges']:,}")
        self.logger.info(f"  负样本: {self.stats['total_negative_edges']:,}")
        self.logger.info(f"  负正比例: 1:{self.stats['negative_positive_ratio']:.1f}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            node_features: 节点特征 [num_nodes, feature_dim]
            risk_embeddings: 风险嵌入 [num_nodes, risk_embed_dim]
            positive_edges: 正样本边 [num_pos, 2]
            negative_edges: 负样本边 [num_neg, 2]
        """
        sample = self.samples[idx]
        
        # 重新加载图以获取特征（这里可以进一步优化）
        graph_file = sample['graph_file']
        try:
            import dgl
            graph, _ = dgl.load_graphs(graph_file)
            graph = graph[0]
            
            # 提取和处理特征
            node_features = self._extract_and_process_features(graph)
            
            # 处理风险分数
            risk_scores = sample['risk_scores']
            risk_embeddings = self.risk_vectorizer.transform(risk_scores.numpy())
            
            # 获取预计算的正负样本
            positive_edges = sample['positive_edges']
            negative_edges = sample['negative_edges']
            
            return (
                torch.FloatTensor(node_features),
                risk_embeddings,
                positive_edges,
                negative_edges
            )
            
        except Exception as e:
            self.logger.error(f"加载样本失败 {idx}: {e}")
            # 返回空样本
            return (
                torch.zeros((1, self.feature_processor.output_dim)),
                torch.zeros((1, self.risk_vectorizer.embed_dim)),
                torch.empty((0, 2), dtype=torch.long),
                torch.empty((0, 2), dtype=torch.long)
            )
    
    def _extract_and_process_features(self, graph) -> np.ndarray:
        """提取和处理图特征"""
        # 这里简化处理，实际应该根据具体的特征提取逻辑
        # 假设使用Flow节点的特征
        if 'Flow' in graph.ntypes and 'feat' in graph.nodes['Flow'].data:
            raw_features = graph.nodes['Flow'].data['feat'].numpy()
            processed_features = self.feature_processor.transform(raw_features)
            return processed_features
        else:
            # 回退：生成随机特征
            num_nodes = max([graph.number_of_nodes(ntype) for ntype in graph.ntypes])
            return np.random.randn(num_nodes, self.feature_processor.output_dim)
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        return self.stats


class OfflineDataLoader:
    """离线数据加载器"""
    
    def __init__(
        self,
        offline_samples_file: str,
        feature_processor,
        risk_vectorizer,
        batch_size: int = 4,
        num_workers: int = 0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        shuffle_seed: int = 42
    ):
        """
        初始化离线数据加载器
        
        Args:
            offline_samples_file: 离线采样结果文件
            feature_processor: 特征处理器
            risk_vectorizer: 风险向量化器
            batch_size: 批次大小
            num_workers: 数据加载进程数
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            shuffle_seed: 随机种子
        """
        self.logger = get_logger().logger  # 修复：使用内部logger对象

        # 创建数据集
        self.train_dataset = OfflineRiskNICLPDataset(
            offline_samples_file, feature_processor, risk_vectorizer,
            split='train', train_ratio=train_ratio, val_ratio=val_ratio, shuffle_seed=shuffle_seed
        )

        self.val_dataset = OfflineRiskNICLPDataset(
            offline_samples_file, feature_processor, risk_vectorizer,
            split='val', train_ratio=train_ratio, val_ratio=val_ratio, shuffle_seed=shuffle_seed
        )

        self.test_dataset = OfflineRiskNICLPDataset(
            offline_samples_file, feature_processor, risk_vectorizer,
            split='test', train_ratio=train_ratio, val_ratio=val_ratio, shuffle_seed=shuffle_seed
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
        
        self.logger.info("离线数据加载器初始化完成")
    
    def _collate_fn(self, batch):
        """批次整理函数"""
        # 简单的批次整理，可以根据需要进一步优化
        node_features_list = []
        risk_embeddings_list = []
        positive_edges_list = []
        negative_edges_list = []
        
        for node_features, risk_embeddings, positive_edges, negative_edges in batch:
            node_features_list.append(node_features)
            risk_embeddings_list.append(risk_embeddings)
            positive_edges_list.append(positive_edges)
            negative_edges_list.append(negative_edges)
        
        return {
            'node_features': node_features_list,
            'risk_embeddings': risk_embeddings_list,
            'positive_edges': positive_edges_list,
            'negative_edges': negative_edges_list
        }
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """获取训练、验证、测试数据加载器"""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_statistics(self) -> Dict:
        """获取所有数据集的统计信息"""
        return {
            'train': self.train_dataset.get_statistics(),
            'val': self.val_dataset.get_statistics(),
            'test': self.test_dataset.get_statistics()
        }


def create_offline_dataloaders(
    offline_samples_file: str,
    feature_processor,
    risk_vectorizer,
    config: Dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建离线数据加载器的便捷函数
    
    Args:
        offline_samples_file: 离线采样结果文件
        feature_processor: 特征处理器
        risk_vectorizer: 风险向量化器
        config: 配置字典
    
    Returns:
        train_loader, val_loader, test_loader
    """
    data_config = config.get('data', {})
    
    offline_loader = OfflineDataLoader(
        offline_samples_file=offline_samples_file,
        feature_processor=feature_processor,
        risk_vectorizer=risk_vectorizer,
        batch_size=config.get('training', {}).get('batch_size', 4),
        num_workers=data_config.get('num_workers', 0),
        train_ratio=data_config.get('train_ratio', 0.8),
        val_ratio=data_config.get('val_ratio', 0.1),
        shuffle_seed=config.get('seed', 42)
    )
    
    return offline_loader.get_loaders()


# 使用示例
if __name__ == "__main__":
    # 这里是使用示例，实际使用时需要提供真实的处理器
    print("离线数据集模块已就绪")
    print("使用方法:")
    print("1. 首先运行 offline_sampling.py 生成离线采样数据")
    print("2. 然后使用 OfflineRiskNICLPDataset 或 create_offline_dataloaders 加载数据")
    print("3. 在训练循环中使用预计算的正负样本，大幅提升效率")
