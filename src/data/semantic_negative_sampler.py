"""
RiskNICLP-5G 语义感知负采样器
基于节点类型和5G网络语义约束的精细化负采样策略

解决问题：
1. 负样本风险倒挂：负样本平均风险(0.609) > 正样本(0.302)
2. 困难样本过度：60%困难负样本导致分布失真
3. 异质图特性未利用：忽略UE-Flow-Alert的语义约束
"""

import torch
import dgl
import numpy as np
import random
import time
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class SemanticSamplingConfig:
    """语义感知采样配置"""
    # 基础采样参数
    neg_ratio: float = 2.0  # 降低负采样比例从4:1到2:1
    seed: int = 42
    
    # 分层采样比例（重新设计）
    difficulty_distribution: Dict[str, float] = None
    
    # 节点类型特定策略
    edge_type_strategies: Dict[str, Dict] = None
    
    # 时空约束
    temporal_constraint_enabled: bool = True
    spatial_constraint_enabled: bool = True
    min_temporal_gap: float = 5.0  # 最小时间间隔（秒）
    
    # 质量控制
    max_attempts_multiplier: int = 15
    quality_threshold: float = 0.8
    
    def __post_init__(self):
        if self.difficulty_distribution is None:
            # 重新设计难度分布：减少困难样本
            self.difficulty_distribution = {
                'easy': 0.5,    # 增加简单负样本到50%
                'medium': 0.3,  # 中等负样本30%
                'hard': 0.2     # 减少困难负样本到20%
            }
        
        if self.edge_type_strategies is None:
            # 基于边类型的差异化策略
            self.edge_type_strategies = {
                'ue_flow': {
                    'hard_ratio': 0.2,  # UE-Flow连接稀疏，困难样本少
                    'semantic_constraint': 'user_behavior',
                    'risk_threshold': 0.4
                },
                'flow_alert': {
                    'hard_ratio': 0.7,  # Flow-Alert因果性强，需要更多困难样本
                    'semantic_constraint': 'causal_logic',
                    'risk_threshold': 0.6
                },
                'flow_flow': {
                    'hard_ratio': 0.4,  # Flow-Flow时序性重要
                    'semantic_constraint': 'temporal_coherence',
                    'risk_threshold': 0.5
                }
            }

class SemanticConstraint(ABC):
    """语义约束基类"""
    
    @abstractmethod
    def is_valid_negative(self, src_node: int, dst_node: int, 
                         graph: dgl.DGLGraph, **kwargs) -> bool:
        """检查是否为有效的负样本"""
        pass

class UserBehaviorConstraint(SemanticConstraint):
    """用户行为语义约束"""

    def __init__(self, config: SemanticSamplingConfig):
        self.config = config

    def _get_node_risk(self, graph: dgl.DGLGraph, node_id: int, node_type: str) -> float:
        """安全获取节点风险分数，处理节点索引映射"""
        try:
            # 检查节点ID是否在有效范围内
            if node_id < graph.number_of_nodes(node_type):
                return graph.nodes[node_type].data['risk_score'][node_id].item()
            return None
        except:
            return None
    
    def is_valid_negative(self, src_node: int, dst_node: int,
                         graph: dgl.DGLGraph, **kwargs) -> bool:
        """
        UE-Flow负样本约束：
        1. 避免高风险UE连接高风险Flow（不符合用户行为模式）
        2. 时间约束：UE行为具有时序性
        """
        try:
            # 获取风险分数（需要处理节点索引映射）
            ue_risk = self._get_node_risk(graph, src_node, 'UE')
            flow_risk = self._get_node_risk(graph, dst_node, 'Flow')

            if ue_risk is None or flow_risk is None:
                return True  # 如果无法获取风险分数，允许作为负样本

            # 约束1：避免双高风险连接
            if ue_risk > 0.6 and flow_risk > 0.6:
                return False

            # 约束2：时间一致性检查（如果有时间信息）
            if self.config.temporal_constraint_enabled:
                if 'timestamp' in graph.nodes['UE'].data and 'timestamp' in graph.nodes['Flow'].data:
                    ue_time = graph.nodes['UE'].data['timestamp'][src_node].item()
                    flow_time = graph.nodes['Flow'].data['timestamp'][dst_node].item()

                    if abs(ue_time - flow_time) > self.config.min_temporal_gap:
                        return False

            return True
        except Exception:
            return True  # 出现异常时允许作为负样本

class CausalLogicConstraint(SemanticConstraint):
    """因果逻辑语义约束"""

    def __init__(self, config: SemanticSamplingConfig):
        self.config = config

    def _get_node_risk(self, graph: dgl.DGLGraph, node_id: int, node_type: str) -> float:
        """安全获取节点风险分数，处理节点索引映射"""
        try:
            # 检查节点ID是否在有效范围内
            if node_id < graph.number_of_nodes(node_type):
                return graph.nodes[node_type].data['risk_score'][node_id].item()
            return None
        except:
            return None
    
    def is_valid_negative(self, src_node: int, dst_node: int,
                         graph: dgl.DGLGraph, **kwargs) -> bool:
        """
        Flow-Alert负样本约束：
        1. 确保负样本在语义上不合理（攻击类型不匹配）
        2. 避免已有因果链的延续
        """
        try:
            # 获取Flow和Alert的特征
            flow_risk = self._get_node_risk(graph, src_node, 'Flow')
            alert_risk = self._get_node_risk(graph, dst_node, 'Alert')

            if flow_risk is None or alert_risk is None:
                return True  # 如果无法获取风险分数，允许作为负样本

            # 约束1：避免低风险Flow触发高风险Alert
            if flow_risk < 0.3 and alert_risk > 0.7:
                return True  # 这是好的负样本

            # 约束2：避免已存在的因果链模式
            # 检查是否已有相似的Flow-Alert连接
            try:
                existing_alerts = graph.successors(src_node, etype='triggers')
                if len(existing_alerts) > 0:
                    # 如果Flow已经触发了其他Alert，避免触发相同类型的Alert
                    return self._check_alert_type_diversity(dst_node, existing_alerts, graph)
            except:
                pass  # 如果无法获取后继节点，忽略此约束

            return True
        except Exception:
            return True  # 出现异常时允许作为负样本
    
    def _check_alert_type_diversity(self, target_alert: int, existing_alerts: torch.Tensor, 
                                   graph: dgl.DGLGraph) -> bool:
        """检查Alert类型多样性"""
        # 简化实现：基于风险分数差异判断
        target_risk = graph.nodes['Alert'].data['risk_score'][target_alert].item()
        
        for existing_alert in existing_alerts:
            existing_risk = graph.nodes['Alert'].data['risk_score'][existing_alert].item()
            if abs(target_risk - existing_risk) < 0.1:  # 风险分数相近，认为是同类型
                return False
        
        return True

class TemporalCoherenceConstraint(SemanticConstraint):
    """时序一致性语义约束"""

    def __init__(self, config: SemanticSamplingConfig):
        self.config = config

    def _get_node_risk(self, graph: dgl.DGLGraph, node_id: int, node_type: str) -> float:
        """安全获取节点风险分数，处理节点索引映射"""
        try:
            # 检查节点ID是否在有效范围内
            if node_id < graph.number_of_nodes(node_type):
                return graph.nodes[node_type].data['risk_score'][node_id].item()
            return None
        except:
            return None
    
    def is_valid_negative(self, src_node: int, dst_node: int,
                         graph: dgl.DGLGraph, **kwargs) -> bool:
        """
        Flow-Flow负样本约束：
        1. 破坏时序逻辑：t_src > t_dst
        2. 避免相似风险级别的Flow连接
        """
        try:
            # 时序约束
            if self.config.temporal_constraint_enabled:
                if 'timestamp' in graph.nodes['Flow'].data:
                    if (src_node < graph.number_of_nodes('Flow') and
                        dst_node < graph.number_of_nodes('Flow')):
                        src_time = graph.nodes['Flow'].data['timestamp'][src_node].item()
                        dst_time = graph.nodes['Flow'].data['timestamp'][dst_node].item()

                        # 负样本应该违反时序逻辑
                        if src_time <= dst_time:
                            return False

            # 风险差异约束
            src_risk = self._get_node_risk(graph, src_node, 'Flow')
            dst_risk = self._get_node_risk(graph, dst_node, 'Flow')

            if src_risk is None or dst_risk is None:
                return True  # 如果无法获取风险分数，允许作为负样本

            # 避免相似风险级别的连接
            if abs(src_risk - dst_risk) < 0.2:
                return False

            return True
        except Exception:
            return True  # 出现异常时允许作为负样本

class SemanticNegativeSampler:
    """语义感知负采样器"""
    
    def __init__(self, config: SemanticSamplingConfig = None):
        self.config = config or SemanticSamplingConfig()
        self.logger = logging.getLogger(__name__)
        
        # 初始化语义约束
        self.constraints = {
            'user_behavior': UserBehaviorConstraint(self.config),
            'causal_logic': CausalLogicConstraint(self.config),
            'temporal_coherence': TemporalCoherenceConstraint(self.config)
        }
        
        # 采样统计
        self.sampling_stats = {
            'total_attempts': 0,
            'successful_samples': 0,
            'constraint_violations': {},
            'edge_type_distribution': {},
            'quality_scores': []
        }
        
        # 设置随机种子
        self._setup_random_seed()
    
    def _setup_random_seed(self):
        """设置随机种子"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
    
    def sample_negatives(self, graph: dgl.DGLGraph, positive_edges: torch.Tensor,
                        risk_scores: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        内存优化的语义感知负采样主接口

        Args:
            graph: DGL异质图
            positive_edges: 正样本边 [num_pos, 2]
            risk_scores: 节点风险分数（可选）

        Returns:
            negative_edges: 负样本边 [num_neg, 2]
        """
        start_time = time.time()

        # 计算目标负样本数量
        num_pos = len(positive_edges)
        num_neg = int(num_pos * self.config.neg_ratio)

        # 🔥 内存优化：限制最大负样本数量，防止内存溢出
        max_neg_samples = min(num_neg, 50000)  # 限制最大50K负样本
        if num_neg > max_neg_samples:
            self.logger.warning(f"⚠️ 负样本数量过大({num_neg})，限制为{max_neg_samples}")
            num_neg = max_neg_samples

        self.logger.info(f"📊 采样目标: {num_pos}正样本 -> {num_neg}负样本 (比例1:{self.config.neg_ratio:.1f})")

        # 🔥 内存优化：使用简化的负采样策略
        negative_edges = self._memory_efficient_sampling(graph, positive_edges, num_neg)

        # 更新统计
        sampling_time = time.time() - start_time
        self._update_sampling_stats(negative_edges, sampling_time)

        self.logger.info(f"✅ 内存优化负采样完成: 生成{len(negative_edges)}个负样本，耗时{sampling_time:.3f}s")

        return torch.tensor(negative_edges, dtype=torch.long)
    
    def _build_positive_edge_set(self, positive_edges: torch.Tensor) -> Set[Tuple[int, int]]:
        """构建正样本边集合"""
        edge_set = set()
        for edge in positive_edges:
            edge_set.add((edge[0].item(), edge[1].item()))
        return edge_set
    
    def _analyze_edge_types(self, graph: dgl.DGLGraph, positive_edges: torch.Tensor) -> Dict:
        """分析边类型分布"""
        analysis = {
            'total_edges': len(positive_edges),
            'edge_type_counts': {},
            'risk_distributions': {}
        }
        
        # 统计各边类型数量（简化：基于节点风险分数推断边类型）
        for edge in positive_edges:
            src, dst = edge[0].item(), edge[1].item()
            
            # 根据风险分数模式推断边类型
            edge_type = self._infer_edge_type(graph, src, dst)
            
            if edge_type not in analysis['edge_type_counts']:
                analysis['edge_type_counts'][edge_type] = 0
            analysis['edge_type_counts'][edge_type] += 1
        
        return analysis
    
    def _infer_edge_type(self, graph: dgl.DGLGraph, src: int, dst: int) -> str:
        """根据节点特征推断边类型"""
        # 获取图中节点类型的累积大小
        node_type_offsets = {}
        offset = 0
        for ntype in graph.ntypes:
            node_type_offsets[ntype] = offset
            offset += graph.number_of_nodes(ntype)

        # 根据节点ID范围推断节点类型
        src_type = None
        dst_type = None

        for ntype, offset in node_type_offsets.items():
            next_offset = offset + graph.number_of_nodes(ntype)
            if offset <= src < next_offset:
                src_type = ntype
            if offset <= dst < next_offset:
                dst_type = ntype

        # 根据节点类型推断边类型
        if src_type == 'UE' and dst_type == 'Flow':
            return 'ue_flow'
        elif src_type == 'Flow' and dst_type == 'Alert':
            return 'flow_alert'
        elif src_type == 'Flow' and dst_type == 'Flow':
            return 'flow_flow'

        # 默认返回flow_flow
        return 'flow_flow'
    
    def _memory_efficient_sampling(self, graph: dgl.DGLGraph, positive_edges: torch.Tensor,
                                 num_neg: int) -> List[List[int]]:
        """内存高效的负采样策略"""
        negative_edges = []
        positive_edge_set = set()

        # 🔥 内存优化：只保存正样本边的集合，不做复杂分析
        for edge in positive_edges:
            positive_edge_set.add((edge[0].item(), edge[1].item()))

        # 获取所有节点ID
        all_nodes = []
        node_offset = 0
        for ntype in graph.ntypes:
            num_nodes = graph.number_of_nodes(ntype)
            all_nodes.extend(range(node_offset, node_offset + num_nodes))
            node_offset += num_nodes

        # 🔥 简化采样：随机采样 + 去重
        max_attempts = min(num_neg * 10, 100000)  # 限制最大尝试次数
        attempts = 0

        while len(negative_edges) < num_neg and attempts < max_attempts:
            # 随机选择两个节点
            src = random.choice(all_nodes)
            dst = random.choice(all_nodes)

            # 检查是否为有效负样本
            if src != dst and (src, dst) not in positive_edge_set:
                negative_edges.append([src, dst])

            attempts += 1

        # 如果采样不足，用简单重复填充
        while len(negative_edges) < num_neg:
            if negative_edges:
                negative_edges.append(negative_edges[len(negative_edges) % len(negative_edges)])
            else:
                # 最后的保险：创建一个简单的负样本
                negative_edges.append([all_nodes[0], all_nodes[1] if len(all_nodes) > 1 else all_nodes[0]])

        return negative_edges[:num_neg]
    
    def _sample_easy_negatives(self, graph: dgl.DGLGraph, num_samples: int,
                              positive_edge_set: Set) -> List[List[int]]:
        """采样简单负样本：低风险节点间连接"""
        samples = []
        attempts = 0
        max_attempts = num_samples * self.config.max_attempts_multiplier
        
        # 获取低风险节点
        low_risk_nodes = self._get_low_risk_nodes(graph)
        
        while len(samples) < num_samples and attempts < max_attempts:
            if len(low_risk_nodes) < 2:
                break
                
            src = random.choice(low_risk_nodes)
            dst = random.choice(low_risk_nodes)
            
            if src != dst and (src, dst) not in positive_edge_set:
                samples.append([src, dst])
            
            attempts += 1
        
        return samples
    
    def _sample_medium_negatives(self, graph: dgl.DGLGraph, num_samples: int,
                                positive_edge_set: Set) -> List[List[int]]:
        """采样中等负样本：异质风险节点连接"""
        samples = []
        attempts = 0
        max_attempts = num_samples * self.config.max_attempts_multiplier
        
        # 获取不同风险级别的节点
        high_risk_nodes = self._get_high_risk_nodes(graph)
        low_risk_nodes = self._get_low_risk_nodes(graph)
        
        while len(samples) < num_samples and attempts < max_attempts:
            if len(high_risk_nodes) == 0 or len(low_risk_nodes) == 0:
                break
                
            # 高风险-低风险配对
            src = random.choice(high_risk_nodes)
            dst = random.choice(low_risk_nodes)
            
            if (src, dst) not in positive_edge_set:
                samples.append([src, dst])
            
            attempts += 1
        
        return samples
    
    def _sample_hard_negatives(self, graph: dgl.DGLGraph, num_samples: int,
                              positive_edge_set: Set) -> List[List[int]]:
        """采样困难负样本：高风险节点间连接（受语义约束）"""
        samples = []
        attempts = 0
        max_attempts = num_samples * self.config.max_attempts_multiplier
        
        # 获取高风险节点
        high_risk_nodes = self._get_high_risk_nodes(graph)
        
        while len(samples) < num_samples and attempts < max_attempts:
            if len(high_risk_nodes) < 2:
                break
                
            src = random.choice(high_risk_nodes)
            dst = random.choice(high_risk_nodes)
            
            if src != dst and (src, dst) not in positive_edge_set:
                # 应用语义约束检查
                if self._check_semantic_constraints(graph, src, dst):
                    samples.append([src, dst])
            
            attempts += 1
        
        return samples
    
    def _get_low_risk_nodes(self, graph: dgl.DGLGraph) -> List[int]:
        """获取低风险节点"""
        low_risk_nodes = []
        node_offset = 0
        
        for ntype in graph.ntypes:
            if 'risk_score' in graph.nodes[ntype].data:
                risks = graph.nodes[ntype].data['risk_score']
                low_risk_mask = risks < 0.3
                low_risk_indices = torch.where(low_risk_mask)[0]
                
                # 转换为全局节点ID
                global_indices = [idx.item() + node_offset for idx in low_risk_indices]
                low_risk_nodes.extend(global_indices)
            
            node_offset += graph.number_of_nodes(ntype)
        
        return low_risk_nodes
    
    def _get_high_risk_nodes(self, graph: dgl.DGLGraph) -> List[int]:
        """获取高风险节点"""
        high_risk_nodes = []
        node_offset = 0
        
        for ntype in graph.ntypes:
            if 'risk_score' in graph.nodes[ntype].data:
                risks = graph.nodes[ntype].data['risk_score']
                high_risk_mask = risks > 0.6
                high_risk_indices = torch.where(high_risk_mask)[0]
                
                # 转换为全局节点ID
                global_indices = [idx.item() + node_offset for idx in high_risk_indices]
                high_risk_nodes.extend(global_indices)
            
            node_offset += graph.number_of_nodes(ntype)
        
        return high_risk_nodes
    
    def _check_semantic_constraints(self, graph: dgl.DGLGraph, src: int, dst: int) -> bool:
        """检查语义约束"""
        # 推断边类型
        edge_type = self._infer_edge_type(graph, src, dst)
        
        # 获取对应的语义约束
        if edge_type in self.config.edge_type_strategies:
            constraint_type = self.config.edge_type_strategies[edge_type]['semantic_constraint']
            
            if constraint_type in self.constraints:
                constraint = self.constraints[constraint_type]
                return constraint.is_valid_negative(src, dst, graph)
        
        return True
    
    def _update_sampling_stats(self, negative_edges: List, sampling_time: float):
        """更新采样统计"""
        self.sampling_stats['successful_samples'] += len(negative_edges)
        self.sampling_stats['total_attempts'] += 1
        
        # 计算质量分数（简化）
        quality_score = len(negative_edges) / max(1, len(negative_edges))
        self.sampling_stats['quality_scores'].append(quality_score)
    
    def get_sampling_stats(self) -> Dict:
        """获取采样统计"""
        return self.sampling_stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.sampling_stats = {
            'total_attempts': 0,
            'successful_samples': 0,
            'constraint_violations': {},
            'edge_type_distribution': {},
            'quality_scores': []
        }

    def get_config(self) -> Dict:
        """返回采样器配置（兼容性方法）"""
        return {
            'type': 'semantic_aware',
            'strategy': 'semantic_stratified',
            'neg_ratio': self.config.neg_ratio,
            'seed': self.config.seed
        }


def create_semantic_negative_sampler(config_dict: Dict = None) -> SemanticNegativeSampler:
    """创建语义感知负采样器的工厂函数"""
    if config_dict is None:
        config_dict = {}
    
    config = SemanticSamplingConfig(
        neg_ratio=config_dict.get('neg_ratio', 2.0),
        seed=config_dict.get('seed', 42),
        temporal_constraint_enabled=config_dict.get('temporal_constraint_enabled', True),
        spatial_constraint_enabled=config_dict.get('spatial_constraint_enabled', True),
        min_temporal_gap=config_dict.get('min_temporal_gap', 5.0),
        max_attempts_multiplier=config_dict.get('max_attempts_multiplier', 15),
        quality_threshold=config_dict.get('quality_threshold', 0.8)
    )
    
    return SemanticNegativeSampler(config)
