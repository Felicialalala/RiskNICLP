"""
RiskNICLP-5G 统一采样框架
重构后的采样器，整合所有采样策略
"""

import torch
import dgl
import numpy as np
import random
import time
import logging
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class SamplingConfig:
    """统一采样配置"""
    # 基础配置
    negative_ratio: float = 1.0
    hard_negative_ratio: float = 0.3
    temporal_negative_ratio: float = 0.2
    seed: int = 42

    # 质量控制
    quality_threshold: float = 0.7
    max_attempts_multiplier: int = 10

    # 风险分层
    risk_thresholds: List[float] = None

    def __post_init__(self):
        if self.risk_thresholds is None:
            self.risk_thresholds = [0.8, 0.5, 0.2]

class BaseSampler(ABC):
    """采样器基类"""

    def __init__(self, config: SamplingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_random_seed()

    def _setup_random_seed(self):
        """设置随机种子"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    @abstractmethod
    def sample(self, graph: dgl.DGLGraph, positive_edges: torch.Tensor,
               risk_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """采样接口"""
        pass


class NegativeSampler(BaseSampler):
    """统一负采样器"""

    def __init__(self, config: SamplingConfig = None):
        super().__init__(config or SamplingConfig())

        # 采样统计
        self.stats = {
            'total_attempts': 0,
            'successful_samples': 0,
            'quality_scores': [],
            'sampling_times': []
        }

    def sample(self, graph: dgl.DGLGraph, positive_edges: torch.Tensor,
               risk_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        统一负采样接口

        Args:
            graph: DGL异质图
            positive_edges: 正样本边 [num_pos, 2]
            risk_scores: 节点风险分数

        Returns:
            negative_edges: 负样本边 [num_neg, 2]
        """
        start_time = time.time()

        # 计算需要的负样本数量
        num_pos = positive_edges.shape[0]
        num_neg = int(num_pos * self.config.negative_ratio)

        # 根据配置选择采样策略
        if risk_scores is not None:
            negative_edges = self._risk_aware_sampling(
                graph, positive_edges, risk_scores, num_neg
            )
        else:
            negative_edges = self._uniform_sampling(
                graph, positive_edges, num_neg
            )

        # 更新统计
        sampling_time = time.time() - start_time
        self.stats['sampling_times'].append(sampling_time)
        self.stats['successful_samples'] += negative_edges.shape[0]

        return negative_edges

    def sample_negatives(self, graph: dgl.DGLGraph, positive_edges: torch.Tensor,
                        risk_scores: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        兼容性方法：负采样接口
        这是训练代码期望的标准接口

        Args:
            graph: DGL图
            positive_edges: 正样本边 [N, 2]
            risk_scores: 节点风险分数 [num_nodes]
            **kwargs: 其他参数

        Returns:
            negative_edges: 负样本边 [M, 2]
        """
        return self.sample(graph, positive_edges, risk_scores)

    def _uniform_sampling(self, graph: dgl.DGLGraph, positive_edges: torch.Tensor,
                         num_neg: int) -> torch.Tensor:
        """均匀负采样"""
        # 获取所有可能的节点对
        num_nodes = graph.number_of_nodes()

        # 创建正样本边的集合用于快速查找
        pos_edge_set = set()
        for edge in positive_edges:
            pos_edge_set.add((edge[0].item(), edge[1].item()))

        negative_edges = []
        max_attempts = num_neg * self.config.max_attempts_multiplier
        attempts = 0

        while len(negative_edges) < num_neg and attempts < max_attempts:
            # 随机采样节点对
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)

            # 检查是否为有效负样本
            if src != dst and (src, dst) not in pos_edge_set:
                negative_edges.append([src, dst])

            attempts += 1

        self.stats['total_attempts'] += attempts

        if len(negative_edges) < num_neg:
            self.logger.warning(f"只采样到 {len(negative_edges)}/{num_neg} 个负样本")

        return torch.tensor(negative_edges, dtype=torch.long)

    def _risk_aware_sampling(self, graph: dgl.DGLGraph, positive_edges: torch.Tensor,
                           risk_scores: torch.Tensor, num_neg: int) -> torch.Tensor:
        """风险感知负采样"""
        # 按风险分数分层
        high_risk_nodes = torch.where(risk_scores > self.config.risk_thresholds[0])[0]
        med_risk_nodes = torch.where(
            (risk_scores > self.config.risk_thresholds[2]) &
            (risk_scores <= self.config.risk_thresholds[0])
        )[0]
        low_risk_nodes = torch.where(risk_scores <= self.config.risk_thresholds[2])[0]

        # 计算各层级的采样数量
        num_hard = int(num_neg * self.config.hard_negative_ratio)
        num_temporal = int(num_neg * self.config.temporal_negative_ratio)
        num_easy = num_neg - num_hard - num_temporal

        negative_edges = []

        # 硬负样本：高风险节点间的连接
        if len(high_risk_nodes) > 1:
            hard_negs = self._sample_from_nodes(
                high_risk_nodes, positive_edges, num_hard
            )
            negative_edges.extend(hard_negs)

        # 时序负样本：中等风险节点
        if len(med_risk_nodes) > 1:
            temporal_negs = self._sample_from_nodes(
                med_risk_nodes, positive_edges, num_temporal
            )
            negative_edges.extend(temporal_negs)

        # 简单负样本：低风险节点
        if len(low_risk_nodes) > 1:
            easy_negs = self._sample_from_nodes(
                low_risk_nodes, positive_edges, num_easy
            )
            negative_edges.extend(easy_negs)

        # 如果采样不足，用均匀采样补充
        if len(negative_edges) < num_neg:
            remaining = num_neg - len(negative_edges)
            uniform_negs = self._uniform_sampling(graph, positive_edges, remaining)
            negative_edges.extend(uniform_negs.tolist())

        return torch.tensor(negative_edges[:num_neg], dtype=torch.long)

    def _sample_from_nodes(self, candidate_nodes: torch.Tensor,
                          positive_edges: torch.Tensor, num_samples: int) -> List[List[int]]:
        """从候选节点中采样"""
        if len(candidate_nodes) < 2:
            return []

        # 创建正样本边集合
        pos_edge_set = set()
        for edge in positive_edges:
            pos_edge_set.add((edge[0].item(), edge[1].item()))

        negative_edges = []
        max_attempts = num_samples * self.config.max_attempts_multiplier
        attempts = 0

        while len(negative_edges) < num_samples and attempts < max_attempts:
            # 从候选节点中随机选择
            src_idx = random.randint(0, len(candidate_nodes) - 1)
            dst_idx = random.randint(0, len(candidate_nodes) - 1)

            src = candidate_nodes[src_idx].item()
            dst = candidate_nodes[dst_idx].item()

            if src != dst and (src, dst) not in pos_edge_set:
                negative_edges.append([src, dst])

            attempts += 1

        return negative_edges

    def get_stats(self) -> Dict:
        """获取采样统计信息"""
        if self.stats['sampling_times']:
            avg_time = np.mean(self.stats['sampling_times'])
            total_time = np.sum(self.stats['sampling_times'])
        else:
            avg_time = total_time = 0

        return {
            'total_attempts': self.stats['total_attempts'],
            'successful_samples': self.stats['successful_samples'],
            'average_sampling_time': avg_time,
            'total_sampling_time': total_time,
            'success_rate': (
                self.stats['successful_samples'] / max(self.stats['total_attempts'], 1)
            )
        }

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_attempts': 0,
            'successful_samples': 0,
            'quality_scores': [],
            'sampling_times': []
        }

    def get_config(self) -> Dict:
        """获取采样器配置 (兼容性方法)"""
        return {
            'type': 'unified',
            'neg_ratio': self.config.negative_ratio,
            'negative_ratio': self.config.negative_ratio,
            'hard_ratio': self.config.hard_negative_ratio,
            'temporal_ratio': self.config.temporal_negative_ratio,
            'quality_threshold': self.config.quality_threshold,
            'max_attempts_multiplier': self.config.max_attempts_multiplier,
            'seed': self.config.seed
        }


class PositiveSampler(BaseSampler):
    """正采样器"""

    def __init__(self, config: SamplingConfig = None):
        super().__init__(config or SamplingConfig())

    def sample(self, graph: dgl.DGLGraph, edge_types: List[str] = None,
               risk_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        正样本采样

        Args:
            graph: DGL异质图
            edge_types: 边类型列表
            risk_scores: 节点风险分数

        Returns:
            positive_edges: 正样本边 [num_pos, 2]
        """
        if edge_types is None:
            edge_types = graph.etypes

        all_positive_edges = []

        for etype in edge_types:
            if graph.number_of_edges(etype) > 0:
                src, dst = graph.edges(etype=etype)
                edges = torch.stack([src, dst], dim=1)
                all_positive_edges.append(edges)

        if all_positive_edges:
            positive_edges = torch.cat(all_positive_edges, dim=0)
        else:
            positive_edges = torch.empty((0, 2), dtype=torch.long)

        return positive_edges


class UnifiedSamplingFramework:
    """统一采样框架"""

    def __init__(self, config: SamplingConfig = None):
        self.config = config or SamplingConfig()
        self.positive_sampler = PositiveSampler(self.config)
        self.negative_sampler = NegativeSampler(self.config)
        self.logger = logging.getLogger(__name__)

    def sample_batch(self, graph: dgl.DGLGraph,
                    risk_scores: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量采样正负样本

        Args:
            graph: DGL异质图
            risk_scores: 节点风险分数

        Returns:
            edges: 所有边 [num_total, 2]
            labels: 边标签 [num_total] (1为正样本，0为负样本)
            edge_types: 边类型 [num_total]
        """
        # 采样正样本
        positive_edges = self.positive_sampler.sample(graph, risk_scores=risk_scores)

        # 采样负样本
        negative_edges = self.negative_sampler.sample(
            graph, positive_edges, risk_scores=risk_scores
        )

        # 合并正负样本
        all_edges = torch.cat([positive_edges, negative_edges], dim=0)
        labels = torch.cat([
            torch.ones(positive_edges.shape[0], dtype=torch.long),
            torch.zeros(negative_edges.shape[0], dtype=torch.long)
        ])

        # 随机打乱
        perm = torch.randperm(all_edges.shape[0])
        all_edges = all_edges[perm]
        labels = labels[perm]

        return all_edges, labels, perm

    def get_sampling_stats(self) -> Dict:
        """获取采样统计信息"""
        neg_stats = self.negative_sampler.get_stats()

        return {
            'negative_sampling': neg_stats,
            'config': {
                'negative_ratio': self.config.negative_ratio,
                'hard_negative_ratio': self.config.hard_negative_ratio,
                'temporal_negative_ratio': self.config.temporal_negative_ratio,
                'quality_threshold': self.config.quality_threshold
            }
        }

    def reset_stats(self):
        """重置所有统计信息"""
        self.negative_sampler.reset_stats()
        self.stats = {
            'total_calls': 0,
            'successful_samples': 0,
            'quality_scores': [],
            'timing': []
        }
        
        self.logger.info(f"🔧 统一负采样器初始化:")
        self.logger.info(f"   策略: {self.config.strategy}")
        self.logger.info(f"   负采样比例: {self.config.neg_ratio}:1")
        self.logger.info(f"   困难/中等/简单比例: {self.config.hard_ratio:.1f}/{self.config.medium_ratio:.1f}/{self.config.easy_ratio:.1f}")
    
    def sample(self, graph, positive_edges: torch.Tensor, 
               risk_scores: torch.Tensor = None) -> torch.Tensor:
        """
        统一负采样接口
        
        Args:
            graph: DGL图
            positive_edges: 正样本边 [N, 2]
            risk_scores: 节点风险分数 [num_nodes]
            
        Returns:
            negative_edges: 负样本边 [M, 2]
        """
        start_time = time.time()
        self.stats['total_calls'] += 1
        
        num_pos_edges = positive_edges.shape[0]
        num_neg_edges = num_pos_edges * self.config.neg_ratio
        
        self.logger.info(f"🎯 [统一负采样] 目标: {num_pos_edges}正样本 -> {num_neg_edges}负样本")
        
        # 构建正样本边集合
        positive_edge_set = self._build_positive_edge_set(positive_edges)
        
        # 根据策略选择采样方法
        if self.config.strategy == 'uniform':
            negative_edges = self._uniform_sampling(graph, num_neg_edges, positive_edge_set)
        elif self.config.strategy == 'stratified':
            negative_edges = self._stratified_sampling(graph, num_neg_edges, positive_edge_set, risk_scores)
        elif self.config.strategy == 'hard':
            negative_edges = self._hard_sampling(graph, num_neg_edges, positive_edge_set, risk_scores)
        else:
            self.logger.warning(f"未知策略 {self.config.strategy}，使用uniform")
            negative_edges = self._uniform_sampling(graph, num_neg_edges, positive_edge_set)
        
        # 质量评估
        quality_score = self._evaluate_quality(negative_edges, positive_edges, risk_scores)
        
        # 更新统计
        elapsed = time.time() - start_time
        self.stats['successful_samples'] += 1
        self.stats['quality_scores'].append(quality_score)
        self.stats['timing'].append(elapsed)
        
        self.logger.info(f"✅ [统一负采样完成] 生成 {len(negative_edges)}/{num_neg_edges} 负样本")
        self.logger.info(f"   质量分数: {quality_score:.3f}, 耗时: {elapsed:.2f}s")
        
        return torch.tensor(negative_edges, dtype=torch.long)
    
    def _build_positive_edge_set(self, positive_edges: torch.Tensor) -> Set[Tuple[int, int]]:
        """构建正样本边集合（包括反向边）"""
        positive_edge_set = set()
        for edge in positive_edges:
            u, v = edge[0].item(), edge[1].item()
            positive_edge_set.add((u, v))
            positive_edge_set.add((v, u))  # 添加反向边
        return positive_edge_set
    
    def _uniform_sampling(self, graph, num_neg_edges: int, 
                         positive_edge_set: Set[Tuple[int, int]]) -> List[List[int]]:
        """均匀随机采样"""
        negative_edges = []
        num_nodes = graph.number_of_nodes()
        max_attempts = num_neg_edges * self.config.max_attempts_multiplier
        attempts = 0
        
        while len(negative_edges) < num_neg_edges and attempts < max_attempts:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            
            if u != v and (u, v) not in positive_edge_set:
                negative_edges.append([u, v])
            
            attempts += 1
        
        return negative_edges
    
    def _stratified_sampling(self, graph, num_neg_edges: int,
                           positive_edge_set: Set[Tuple[int, int]],
                           risk_scores: torch.Tensor = None) -> List[List[int]]:
        """分层采样（基于风险分数）"""
        if risk_scores is None:
            return self._uniform_sampling(graph, num_neg_edges, positive_edge_set)
        
        # 分层节点
        num_nodes = graph.number_of_nodes()
        high_risk_nodes, medium_risk_nodes, low_risk_nodes = self._stratify_nodes(
            risk_scores, num_nodes
        )
        
        negative_edges = []
        
        # 困难负样本（高风险节点间）
        num_hard = int(num_neg_edges * self.config.hard_ratio)
        if num_hard > 0 and len(high_risk_nodes) > 1:
            hard_edges = self._sample_from_node_set(
                high_risk_nodes, num_hard, positive_edge_set
            )
            negative_edges.extend(hard_edges)
            self.logger.info(f"   困难负样本: {len(hard_edges)}/{num_hard}")
        
        # 中等负样本（高-中风险节点间）
        num_medium = int(num_neg_edges * self.config.medium_ratio)
        if num_medium > 0:
            medium_edges = self._sample_cross_risk(
                high_risk_nodes, medium_risk_nodes, num_medium, positive_edge_set
            )
            negative_edges.extend(medium_edges)
            self.logger.info(f"   中等负样本: {len(medium_edges)}/{num_medium}")
        
        # 简单负样本（低风险节点间）
        remaining = num_neg_edges - len(negative_edges)
        if remaining > 0:
            easy_edges = self._sample_from_node_set(
                low_risk_nodes, remaining, positive_edge_set
            )
            negative_edges.extend(easy_edges)
            self.logger.info(f"   简单负样本: {len(easy_edges)}/{remaining}")
        
        return negative_edges
    
    def _hard_sampling(self, graph, num_neg_edges: int,
                      positive_edge_set: Set[Tuple[int, int]],
                      risk_scores: torch.Tensor = None) -> List[List[int]]:
        """困难采样（主要采样高风险节点间的边）"""
        if risk_scores is None:
            return self._uniform_sampling(graph, num_neg_edges, positive_edge_set)
        
        num_nodes = graph.number_of_nodes()
        high_risk_nodes, _, _ = self._stratify_nodes(risk_scores, num_nodes)
        
        # 80%困难样本，20%随机样本
        num_hard = int(num_neg_edges * 0.8)
        num_random = num_neg_edges - num_hard
        
        negative_edges = []
        
        # 困难样本
        if len(high_risk_nodes) > 1:
            hard_edges = self._sample_from_node_set(
                high_risk_nodes, num_hard, positive_edge_set
            )
            negative_edges.extend(hard_edges)
        
        # 随机样本补充
        if num_random > 0:
            random_edges = self._uniform_sampling(graph, num_random, positive_edge_set)
            negative_edges.extend(random_edges)
        
        return negative_edges
    
    def _stratify_nodes(self, risk_scores: torch.Tensor, num_nodes: int) -> Tuple[List[int], List[int], List[int]]:
        """基于风险分数分层节点"""
        high_threshold, medium_threshold = self.config.risk_thresholds[0], self.config.risk_thresholds[1]
        
        high_risk_nodes = []
        medium_risk_nodes = []
        low_risk_nodes = []
        
        for i in range(num_nodes):
            if i < len(risk_scores):
                score = risk_scores[i].item()
                if score >= high_threshold:
                    high_risk_nodes.append(i)
                elif score >= medium_threshold:
                    medium_risk_nodes.append(i)
                else:
                    low_risk_nodes.append(i)
            else:
                low_risk_nodes.append(i)
        
        return high_risk_nodes, medium_risk_nodes, low_risk_nodes
    
    def _sample_from_node_set(self, node_set: List[int], num_edges: int,
                             positive_edge_set: Set[Tuple[int, int]]) -> List[List[int]]:
        """从节点集合中采样边"""
        negative_edges = []
        max_attempts = num_edges * self.config.max_attempts_multiplier
        attempts = 0
        
        if len(node_set) < 2:
            return negative_edges
        
        while len(negative_edges) < num_edges and attempts < max_attempts:
            u = random.choice(node_set)
            v = random.choice(node_set)
            
            if u != v and (u, v) not in positive_edge_set:
                negative_edges.append([u, v])
            
            attempts += 1
        
        return negative_edges
    
    def _sample_cross_risk(self, high_nodes: List[int], medium_nodes: List[int],
                          num_edges: int, positive_edge_set: Set[Tuple[int, int]]) -> List[List[int]]:
        """跨风险层级采样"""
        negative_edges = []
        max_attempts = num_edges * self.config.max_attempts_multiplier
        attempts = 0
        
        if len(high_nodes) == 0 or len(medium_nodes) == 0:
            # 回退到单一集合采样
            all_nodes = high_nodes + medium_nodes
            return self._sample_from_node_set(all_nodes, num_edges, positive_edge_set)
        
        while len(negative_edges) < num_edges and attempts < max_attempts:
            u = random.choice(high_nodes)
            v = random.choice(medium_nodes)
            
            if u != v and (u, v) not in positive_edge_set:
                negative_edges.append([u, v])
            
            attempts += 1
        
        return negative_edges
    
    def _evaluate_quality(self, negative_edges: List[List[int]], 
                         positive_edges: torch.Tensor,
                         risk_scores: torch.Tensor = None) -> float:
        """评估负采样质量"""
        if len(negative_edges) == 0:
            return 0.0
        
        # 基础质量：负样本数量比例
        target_count = positive_edges.shape[0] * self.config.neg_ratio
        count_quality = min(len(negative_edges) / target_count, 1.0)
        
        # 如果没有风险分数，只返回数量质量
        if risk_scores is None:
            return count_quality
        
        # 风险分离度：负样本的平均风险应该低于正样本
        neg_risks = []
        for edge in negative_edges:
            u, v = edge[0], edge[1]
            if u < len(risk_scores) and v < len(risk_scores):
                avg_risk = (risk_scores[u] + risk_scores[v]) / 2
                neg_risks.append(avg_risk.item())
        
        pos_risks = []
        for edge in positive_edges:
            u, v = edge[0].item(), edge[1].item()
            if u < len(risk_scores) and v < len(risk_scores):
                avg_risk = (risk_scores[u] + risk_scores[v]) / 2
                pos_risks.append(avg_risk.item())
        
        if len(neg_risks) > 0 and len(pos_risks) > 0:
            neg_mean = np.mean(neg_risks)
            pos_mean = np.mean(pos_risks)
            separation = max(0, pos_mean - neg_mean) / max(pos_mean, 0.1)
        else:
            separation = 0.5
        
        # 综合质量分数
        quality = 0.5 * count_quality + 0.5 * separation
        return quality
    
    def get_stats(self) -> Dict:
        """获取采样统计信息"""
        if len(self.stats['quality_scores']) > 0:
            avg_quality = np.mean(self.stats['quality_scores'])
            avg_timing = np.mean(self.stats['timing'])
        else:
            avg_quality = 0.0
            avg_timing = 0.0

        return {
            'total_calls': self.stats['total_calls'],
            'successful_samples': self.stats['successful_samples'],
            'average_quality': avg_quality,
            'average_timing': avg_timing,
            'config': {
                'strategy': self.config.strategy,
                'neg_ratio': self.config.neg_ratio,
                'hard_ratio': self.config.hard_ratio,
                'medium_ratio': self.config.medium_ratio,
                'easy_ratio': self.config.easy_ratio
            }
        }

    def get_config(self) -> Dict:
        """获取采样器配置（兼容性方法）"""
        return {
            'type': 'unified',
            'strategy': self.config.strategy,
            'neg_ratio': self.config.neg_ratio,
            'hard_ratio': self.config.hard_ratio,
            'medium_ratio': self.config.medium_ratio,
            'easy_ratio': self.config.easy_ratio,
            'risk_thresholds': self.config.risk_thresholds,
            'quality_threshold': self.config.quality_threshold,
            'max_attempts_multiplier': self.config.max_attempts_multiplier,
            'seed': self.config.seed
        }

    def sample_negatives(self, graph, positive_edges: torch.Tensor,
                        risk_scores: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        兼容性方法：负采样接口
        这是训练代码期望的标准接口

        Args:
            graph: DGL图
            positive_edges: 正样本边 [N, 2]
            risk_scores: 节点风险分数 [num_nodes]
            **kwargs: 其他参数

        Returns:
            negative_edges: 负样本边 [M, 2]
        """
        return self.sample(graph, positive_edges, risk_scores)

    @property
    def strategy(self) -> str:
        """兼容性属性：采样策略"""
        return self.config.strategy

    @property
    def neg_ratio(self) -> int:
        """兼容性属性：负采样比例"""
        return self.config.neg_ratio

    def get_sampling_stats(self) -> Dict:
        """兼容性方法：获取采样统计"""
        return self.get_stats()

    def reset_stats(self):
        """兼容性方法：重置统计信息"""
        self.stats = {
            'total_calls': 0,
            'successful_samples': 0,
            'quality_scores': [],
            'timing': []
        }

# 工厂函数
def create_sampling_framework(config_dict: Dict = None) -> UnifiedSamplingFramework:
    """创建统一采样框架"""
    if config_dict is None:
        config_dict = {}

    config = SamplingConfig(
        negative_ratio=config_dict.get('negative_ratio', 1.0),
        hard_negative_ratio=config_dict.get('hard_negative_ratio', 0.3),
        temporal_negative_ratio=config_dict.get('temporal_negative_ratio', 0.2),
        quality_threshold=config_dict.get('quality_threshold', 0.7),
        max_attempts_multiplier=config_dict.get('max_attempts_multiplier', 10),
        seed=config_dict.get('seed', 42)
    )

    return UnifiedSamplingFramework(config)
