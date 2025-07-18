#!/usr/bin/env python3
"""
RiskNICLP-5G 离线采样系统
将采样与训练分离，符合SOTA实践，提升训练效率
"""

import torch
import dgl
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import time
import logging
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import yaml

from .negative_sampler import create_negative_sampler
from ..common.logger import get_logger


@dataclass
class OfflineSamplingConfig:
    """离线采样配置"""
    # 采样策略
    negative_ratio: float = 4.0
    hard_negative_ratio: float = 0.6
    sampling_strategy: str = 'stratified'
    
    # 性能优化
    batch_size: int = 10  # 每批处理的图数量
    num_workers: int = 4  # 并行进程数
    chunk_size: int = 50000  # 大图分块大小
    
    # 缓存配置
    cache_dir: str = 'data/offline_samples'
    enable_compression: bool = True
    max_cache_size_gb: float = 10.0
    
    # 质量控制
    quality_threshold: float = 0.7
    max_attempts_multiplier: int = 10
    
    # 输出格式
    save_format: str = 'pickle'  # 'pickle', 'hdf5', 'parquet'
    include_metadata: bool = True


class OfflineSampler:
    """离线负采样器"""
    
    def __init__(self, config: OfflineSamplingConfig):
        self.config = config
        self.logger = get_logger().logger  # 修复：使用内部logger对象
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'total_graphs': 0,
            'total_positive_edges': 0,
            'total_negative_edges': 0,
            'processing_time': 0,
            'failed_graphs': []
        }
    
    def process_single_graph(self, graph_file: Path) -> Optional[Dict]:
        """处理单个图文件"""
        try:
            # 加载图
            graph, _ = dgl.load_graphs(str(graph_file))
            graph = graph[0]
            
            # 提取正样本边
            positive_edges = self._extract_positive_edges(graph)
            if len(positive_edges) == 0:
                return None
            
            # 提取风险分数
            risk_scores = self._extract_risk_scores(graph)
            
            # 创建负采样器
            sampler_config = {
                'type': 'emergency',
                'strategy': self.config.sampling_strategy,
                'neg_ratio': self.config.negative_ratio,
                'hard_ratio': self.config.hard_negative_ratio
            }
            sampler = create_negative_sampler(sampler_config)
            
            # 执行采样
            start_time = time.time()
            
            # 大图分块处理
            if len(positive_edges) > self.config.chunk_size:
                negative_edges = self._chunked_sampling(
                    graph, positive_edges, risk_scores, sampler
                )
            else:
                negative_edges = sampler.sample_negatives(
                    graph, positive_edges, risk_scores
                )
            
            sampling_time = time.time() - start_time
            
            # 构建结果
            result = {
                'graph_file': str(graph_file),
                'positive_edges': positive_edges,
                'negative_edges': negative_edges,
                'risk_scores': risk_scores,
                'graph_metadata': self._extract_graph_metadata(graph),
                'sampling_metadata': {
                    'sampling_time': sampling_time,
                    'positive_count': len(positive_edges),
                    'negative_count': len(negative_edges),
                    'actual_ratio': len(negative_edges) / len(positive_edges) if len(positive_edges) > 0 else 0,
                    'sampler_type': sampler_config['type'],
                    'strategy': sampler_config['strategy']
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"处理图文件失败 {graph_file}: {e}")
            return None
    
    def _extract_positive_edges(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """提取正样本边"""
        positive_edges = []
        
        # 从不同边类型提取正样本
        for etype in graph.etypes:
            if etype in ['triggers', 'initiates', 'temporal_adj']:
                src, dst = graph.edges(etype=etype)
                if len(src) > 0:
                    edges = torch.stack([src, dst], dim=1)
                    positive_edges.append(edges)
        
        if positive_edges:
            all_edges = torch.cat(positive_edges, dim=0)
            # 去重
            unique_edges = torch.unique(all_edges, dim=0)
            return unique_edges
        else:
            return torch.empty((0, 2), dtype=torch.long)
    
    def _extract_risk_scores(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """提取风险分数"""
        # 优先使用Flow节点的风险分数
        if 'Flow' in graph.ntypes and 'risk_score' in graph.nodes['Flow'].data:
            return graph.nodes['Flow'].data['risk_score']
        
        # 回退到其他节点类型
        for ntype in graph.ntypes:
            if 'risk_score' in graph.nodes[ntype].data:
                return graph.nodes[ntype].data['risk_score']
        
        # 最后回退：生成随机风险分数
        num_nodes = max([graph.number_of_nodes(ntype) for ntype in graph.ntypes])
        return torch.rand(num_nodes)
    
    def _extract_graph_metadata(self, graph: dgl.DGLGraph) -> Dict:
        """提取图元数据"""
        metadata = {
            'node_types': graph.ntypes,
            'edge_types': graph.etypes,
            'node_counts': {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes},
            'edge_counts': {etype: graph.number_of_edges(etype) for etype in graph.etypes}
        }
        
        # 添加特征维度信息
        for ntype in graph.ntypes:
            if 'feat' in graph.nodes[ntype].data:
                metadata[f'{ntype}_feat_dim'] = graph.nodes[ntype].data['feat'].shape[1]
        
        return metadata
    
    def _chunked_sampling(self, graph: dgl.DGLGraph, positive_edges: torch.Tensor, 
                         risk_scores: torch.Tensor, sampler) -> torch.Tensor:
        """分块采样处理大图"""
        chunk_size = self.config.chunk_size
        num_chunks = (len(positive_edges) + chunk_size - 1) // chunk_size
        
        all_negative_edges = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(positive_edges))
            
            chunk_positive = positive_edges[start_idx:end_idx]
            chunk_negative = sampler.sample_negatives(
                graph, chunk_positive, risk_scores
            )
            
            all_negative_edges.append(chunk_negative)
            
            if i % 10 == 0:
                self.logger.info(f"分块采样进度: {i+1}/{num_chunks}")
        
        return torch.cat(all_negative_edges, dim=0)
    
    def process_graph_directory(self, graph_dir: str, output_file: str = None) -> str:
        """处理整个图目录"""
        graph_dir = Path(graph_dir)
        graph_files = list(graph_dir.glob('*.dgl'))
        
        if not graph_files:
            raise ValueError(f"未找到图文件: {graph_dir}")
        
        self.logger.info(f"开始离线采样: {len(graph_files)}个图文件")
        
        # 并行处理
        results = []
        failed_count = 0
        
        if self.config.num_workers > 1:
            # 多进程处理
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_single_graph, graph_file): graph_file 
                    for graph_file in graph_files
                }
                
                for future in as_completed(future_to_file):
                    graph_file = future_to_file[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                        else:
                            failed_count += 1
                    except Exception as e:
                        self.logger.error(f"处理失败 {graph_file}: {e}")
                        failed_count += 1
                        
                    # 进度报告
                    if len(results) % 10 == 0:
                        self.logger.info(f"处理进度: {len(results)}/{len(graph_files)}")
        else:
            # 单进程处理
            for i, graph_file in enumerate(graph_files):
                result = self.process_single_graph(graph_file)
                if result is not None:
                    results.append(result)
                else:
                    failed_count += 1
                
                if i % 10 == 0:
                    self.logger.info(f"处理进度: {i+1}/{len(graph_files)}")
        
        # 统计信息
        total_positive = sum(len(r['positive_edges']) for r in results)
        total_negative = sum(len(r['negative_edges']) for r in results)
        
        self.stats.update({
            'total_graphs': len(results),
            'total_positive_edges': total_positive,
            'total_negative_edges': total_negative,
            'failed_graphs': failed_count
        })
        
        self.logger.info(f"离线采样完成:")
        self.logger.info(f"  成功处理: {len(results)}个图")
        self.logger.info(f"  失败: {failed_count}个图")
        self.logger.info(f"  正样本总数: {total_positive:,}")
        self.logger.info(f"  负样本总数: {total_negative:,}")
        self.logger.info(f"  平均比例: 1:{total_negative/total_positive:.1f}")
        
        # 保存结果
        if output_file is None:
            output_file = self.cache_dir / f"offline_samples_{len(results)}graphs.pkl"
        
        output_path = self._save_results(results, output_file)
        
        return str(output_path)
    
    def _save_results(self, results: List[Dict], output_file: str) -> Path:
        """保存采样结果"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 添加统计信息
        final_data = {
            'samples': results,
            'stats': self.stats,
            'config': self.config
        }
        
        if self.config.save_format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise NotImplementedError(f"格式 {self.config.save_format} 暂未实现")
        
        self.logger.info(f"采样结果已保存: {output_path}")
        return output_path


def main():
    """主函数：执行离线采样"""
    # 加载配置
    with open('config/riskniclp_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建离线采样配置
    sampling_config = OfflineSamplingConfig(
        negative_ratio=4.0,
        hard_negative_ratio=0.6,
        sampling_strategy='stratified',
        num_workers=min(4, mp.cpu_count()),
        chunk_size=10000,  # 减小分块大小
        cache_dir='data/offline_samples'
    )
    
    # 执行离线采样
    sampler = OfflineSampler(sampling_config)
    
    graph_dir = config['data']['graph_dir']
    output_file = f"data/offline_samples/riskniclp_samples_{int(time.time())}.pkl"
    
    result_path = sampler.process_graph_directory(graph_dir, output_file)
    
    print(f"🎉 离线采样完成！结果保存在: {result_path}")


if __name__ == "__main__":
    main()
