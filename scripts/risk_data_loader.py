#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk-PPR数据加载器
用于将完整的Risk-PPR计算结果整合到RiskNICLP-Sec模型训练中

Created: 2025-01-27
Author: RiskNICLP-5G Pipeline
"""

import torch
import json
import numpy as np
import dgl
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RiskDataLoader:
    """Risk-PPR数据加载器类"""
    
    def __init__(self, 
                 risk_vectors_file: str = "data/processed/metadata/risk_vectors_complete.json",
                 training_data_file: str = "data/processed/metadata/training_risk_data.json"):
        """
        初始化数据加载器
        
        Args:
            risk_vectors_file: 风险向量文件路径
            training_data_file: 训练数据文件路径
        """
        self.risk_vectors_file = risk_vectors_file
        self.training_data_file = training_data_file
        
        # 加载数据
        self.risk_vectors = self._load_risk_vectors()
        self.training_data = self._load_training_data()
        
        logger.info(f"加载完成: {len(self.risk_vectors)} 个图的风险向量")
        logger.info(f"训练数据: {len(self.training_data)} 个图样本")
    
    def _load_risk_vectors(self) -> Dict:
        """加载风险向量数据"""
        try:
            with open(self.risk_vectors_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"风险向量文件不存在: {self.risk_vectors_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return {}
    
    def _load_training_data(self) -> List[Dict]:
        """加载训练数据"""
        try:
            with open(self.training_data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"训练数据文件不存在: {self.training_data_file}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return []
    
    def get_graph_risk_features(self, graph_id: str) -> Optional[Dict]:
        """
        获取指定图的风险特征
        
        Args:
            graph_id: 图ID (不含.dgl后缀)
            
        Returns:
            包含PPR分数和风险权重的字典
        """
        if graph_id in self.risk_vectors:
            return self.risk_vectors[graph_id]
        else:
            logger.warning(f"未找到图ID {graph_id} 的风险数据")
            return None
    
    def get_graph_with_risk_features(self, graph_path: str) -> Tuple[Optional[dgl.DGLHeteroGraph], Optional[Dict]]:
        """
        加载图文件并附带其风险特征
        
        Args:
            graph_path: 图文件路径
            
        Returns:
            (DGL图对象, 风险特征字典)
        """
        try:
            # 加载DGL图
            graphs, _ = dgl.load_graphs(graph_path)
            graph = graphs[0]
            
            # 获取对应的风险特征
            graph_filename = Path(graph_path).stem  # 获取不含扩展名的文件名
            risk_features = self.get_graph_risk_features(graph_filename)
            
            return graph, risk_features
            
        except Exception as e:
            logger.error(f"加载图文件 {graph_path} 失败: {e}")
            return None, None
    
    def enhance_graph_with_risk_scores(self, graph: dgl.DGLHeteroGraph, risk_features: Dict) -> dgl.DGLHeteroGraph:
        """
        将风险分数添加到图的节点特征中
        
        Args:
            graph: 原始DGL图
            risk_features: 风险特征字典
            
        Returns:
            增强后的DGL图
        """
        if not risk_features or 'ppr_scores' not in risk_features:
            logger.warning("风险特征缺失，返回原图")
            return graph
        
        try:
            # 为每种节点类型添加PPR分数特征
            for node_type in ['UE', 'Flow', 'Alert']:
                if node_type in risk_features['ppr_scores']:
                    ppr_scores = torch.tensor(risk_features['ppr_scores'][node_type], dtype=torch.float32)
                    
                    # 确保维度匹配
                    if len(ppr_scores) == graph.num_nodes(node_type):
                        # 添加PPR分数作为新特征
                        if 'feat' in graph.nodes[node_type].data:
                            # 已有特征，拼接PPR分数
                            existing_feat = graph.nodes[node_type].data['feat']
                            enhanced_feat = torch.cat([existing_feat, ppr_scores.unsqueeze(-1)], dim=-1)
                            graph.nodes[node_type].data['feat'] = enhanced_feat
                        else:
                            # 没有现有特征，直接使用PPR分数
                            graph.nodes[node_type].data['feat'] = ppr_scores.unsqueeze(-1)
                        
                        # 单独存储PPR分数
                        graph.nodes[node_type].data['ppr_score'] = ppr_scores
                        
                        logger.debug(f"为 {node_type} 节点添加PPR特征: {len(ppr_scores)} 个节点")
                    else:
                        logger.warning(f"{node_type} 节点数量不匹配: 图中{graph.num_nodes(node_type)}, PPR{len(ppr_scores)}")
            
            return graph
            
        except Exception as e:
            logger.error(f"增强图特征时出错: {e}")
            return graph
    
    def get_high_risk_nodes(self, graph_id: str, node_type: str = 'Flow', top_k: int = 5) -> List[int]:
        """
        获取指定图中的高风险节点
        
        Args:
            graph_id: 图ID
            node_type: 节点类型
            top_k: 返回top-k个高风险节点
            
        Returns:
            高风险节点ID列表
        """
        risk_features = self.get_graph_risk_features(graph_id)
        if not risk_features or 'top_risk_nodes' not in risk_features:
            return []
        
        if node_type in risk_features['top_risk_nodes']:
            return risk_features['top_risk_nodes'][node_type][:top_k]
        else:
            return []
    
    def get_risk_statistics_summary(self) -> Dict:
        """获取所有图的风险统计摘要"""
        if not self.training_data:
            return {}
        
        summary = {
            'total_graphs': len(self.training_data),
            'node_type_stats': {},
            'risk_score_ranges': {}
        }
        
        # 统计各节点类型的数量范围
        for node_type in ['UE', 'Flow', 'Alert']:
            node_counts = [data['num_nodes'][node_type] for data in self.training_data 
                          if node_type in data['num_nodes']]
            if node_counts:
                summary['node_type_stats'][node_type] = {
                    'min': min(node_counts),
                    'max': max(node_counts),
                    'mean': sum(node_counts) / len(node_counts)
                }
        
        # 统计风险分数范围
        for data in self.training_data:
            if 'risk_statistics' in data:
                for node_type, stats in data['risk_statistics'].items():
                    if node_type not in summary['risk_score_ranges']:
                        summary['risk_score_ranges'][node_type] = {
                            'min_score': float('inf'),
                            'max_score': float('-inf')
                        }
                    
                    summary['risk_score_ranges'][node_type]['min_score'] = min(
                        summary['risk_score_ranges'][node_type]['min_score'], 
                        stats['min']
                    )
                    summary['risk_score_ranges'][node_type]['max_score'] = max(
                        summary['risk_score_ranges'][node_type]['max_score'], 
                        stats['max']
                    )
        
        return summary
    
    def prepare_model_input_batch(self, graph_ids: List[str]) -> Tuple[List[dgl.DGLHeteroGraph], List[Dict]]:
        """
        为模型训练准备批次输入
        
        Args:
            graph_ids: 图ID列表
            
        Returns:
            (图列表, 风险特征列表)
        """
        graphs = []
        risk_features_list = []
        
        for graph_id in graph_ids:
            # 获取图路径
            if graph_id in self.risk_vectors:
                graph_path = self.risk_vectors[graph_id]['graph_path']
                graph, risk_features = self.get_graph_with_risk_features(graph_path)
                
                if graph is not None and risk_features is not None:
                    # 增强图特征
                    enhanced_graph = self.enhance_graph_with_risk_scores(graph, risk_features)
                    graphs.append(enhanced_graph)
                    risk_features_list.append(risk_features)
                else:
                    logger.warning(f"跳过图ID {graph_id}: 加载失败")
            else:
                logger.warning(f"跳过图ID {graph_id}: 风险数据不存在")
        
        return graphs, risk_features_list
    
    def get_available_graph_ids(self) -> List[str]:
        """获取所有可用的图ID"""
        return list(self.risk_vectors.keys())
    
    def filter_graphs_by_risk_level(self, min_avg_risk: float = 0.01) -> List[str]:
        """
        根据风险水平过滤图
        
        Args:
            min_avg_risk: 最小平均风险分数阈值
            
        Returns:
            符合条件的图ID列表
        """
        filtered_ids = []
        
        for data in self.training_data:
            if 'risk_statistics' in data:
                # 计算所有节点类型的平均风险
                avg_risks = []
                for node_type, stats in data['risk_statistics'].items():
                    avg_risks.append(stats['mean'])
                
                if avg_risks and np.mean(avg_risks) >= min_avg_risk:
                    filtered_ids.append(data['graph_id'])
        
        logger.info(f"风险过滤: {len(filtered_ids)}/{len(self.training_data)} 个图满足条件 (min_risk={min_avg_risk})")
        return filtered_ids


def demo_usage():
    """演示使用方法"""
    print("=" * 50)
    print("Risk-PPR数据加载器使用演示")
    print("=" * 50)
    
    # 初始化加载器
    loader = RiskDataLoader()
    
    # 获取摘要统计
    summary = loader.get_risk_statistics_summary()
    print(f"\n数据摘要:")
    print(f"  总图数量: {summary.get('total_graphs', 0)}")
    
    if summary.get('node_type_stats'):
        print(f"\n节点类型统计:")
        for node_type, stats in summary['node_type_stats'].items():
            print(f"  {node_type}: {stats['min']}-{stats['max']} (平均: {stats['mean']:.1f})")
    
    # 获取可用图ID
    available_ids = loader.get_available_graph_ids()
    print(f"\n可用图ID数量: {len(available_ids)}")
    
    if available_ids:
        # 演示加载单个图
        sample_id = available_ids[0]
        print(f"\n演示加载图: {sample_id}")
        
        graph, risk_features = loader.get_graph_with_risk_features(
            loader.risk_vectors[sample_id]['graph_path']
        )
        
        if graph is not None:
            print(f"  图节点数: {dict(graph.num_nodes_dict())}")
            print(f"  图边数: {dict(graph.num_edges_dict())}")
            
            # 增强图特征
            enhanced_graph = loader.enhance_graph_with_risk_scores(graph, risk_features)
            print(f"  增强后特征维度: {enhanced_graph.nodes['Flow'].data.get('feat', torch.tensor([])).shape}")
        
        # 获取高风险节点
        high_risk_flows = loader.get_high_risk_nodes(sample_id, 'Flow', top_k=3)
        print(f"  高风险Flow节点: {high_risk_flows}")
    
    # 风险过滤演示
    high_risk_graphs = loader.filter_graphs_by_risk_level(min_avg_risk=0.01)
    print(f"\n高风险图数量: {len(high_risk_graphs)}")
    
    print("\n演示完成!")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行演示
    demo_usage() 