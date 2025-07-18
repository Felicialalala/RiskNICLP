#!/usr/bin/env python3
"""
数据子集加载器
专门为攻击链路预测任务优化的数据加载策略
"""

import torch
import dgl
import numpy as np
from pathlib import Path
import logging

class AttackPredictionDataLoader:
    """攻击链路预测数据加载器"""
    
    def __init__(self, config):
        self.config = config
        self.data_config = config.get('data', {})
        self.subset_ratio = self.data_config.get('subset_ratio', 0.1)
        self.max_edges_per_type = self.data_config.get('max_edges_per_type', 50000)
        self.chunk_size = self.data_config.get('chunk_size', 1000)
        
        logging.info(f"攻击预测数据加载器初始化:")
        logging.info(f"  数据子集比例: {self.subset_ratio:.1%}")
        logging.info(f"  每类边数限制: {self.max_edges_per_type:,}")
        logging.info(f"  分块大小: {self.chunk_size}")
    
    def load_graph_subset(self, graph_path):
        """加载图数据子集"""
        try:
            # 这里应该实现实际的图数据加载逻辑
            # 目前返回模拟数据
            logging.info("加载图数据子集...")
            
            # 模拟创建一个小规模图用于测试
            num_ue = int(1000 * self.subset_ratio)
            num_flow = int(5000 * self.subset_ratio) 
            num_alert = int(500 * self.subset_ratio)
            
            # 创建异质图
            graph_data = {
                ('UE', 'initiates', 'Flow'): (
                    torch.randint(0, num_ue, (min(self.max_edges_per_type, num_ue * 2),)),
                    torch.randint(0, num_flow, (min(self.max_edges_per_type, num_ue * 2),))
                ),
                ('Flow', 'triggers', 'Alert'): (
                    torch.randint(0, num_flow, (min(self.max_edges_per_type, num_flow // 2),)),
                    torch.randint(0, num_alert, (min(self.max_edges_per_type, num_flow // 2),))
                ),
                ('Flow', 'temporal_adj', 'Flow'): (
                    torch.randint(0, num_flow, (min(self.max_edges_per_type, num_flow),)),
                    torch.randint(0, num_flow, (min(self.max_edges_per_type, num_flow),))
                )
            }
            
            graph = dgl.heterograph(graph_data)
            
            logging.info(f"图数据子集加载完成:")
            logging.info(f"  UE节点: {num_ue}")
            logging.info(f"  Flow节点: {num_flow}")
            logging.info(f"  Alert节点: {num_alert}")
            for etype in graph.etypes:
                logging.info(f"  {etype}边: {graph.number_of_edges(etype)}")
            
            return graph
            
        except Exception as e:
            logging.error(f"图数据加载失败: {e}")
            raise
    
    def create_streaming_loader(self, graph):
        """创建流式数据加载器"""
        # 这里应该实现流式加载逻辑
        # 目前返回简单的批次迭代器
        
        total_edges = sum(graph.number_of_edges(etype) for etype in graph.etypes)
        num_batches = max(1, total_edges // self.chunk_size)
        
        logging.info(f"创建流式加载器: {num_batches} 个批次")
        
        for i in range(num_batches):
            # 模拟批次数据
            batch_size = min(self.chunk_size, total_edges - i * self.chunk_size)
            
            yield {
                'graph': graph,
                'batch_id': i,
                'batch_size': batch_size,
                'edge_indices': torch.arange(i * self.chunk_size, 
                                           min((i + 1) * self.chunk_size, total_edges))
            }

# 使用示例
if __name__ == "__main__":
    import yaml
    
    with open('config/riskniclp_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    loader = AttackPredictionDataLoader(config)
    print("✅ 攻击预测数据加载器创建成功")
