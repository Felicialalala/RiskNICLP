"""
RiskNICLP-5G 攻击传播采样器
专门针对攻击链路预测任务的正负样本生成策略

替代异常检测采样器，专注于攻击传播模式的建模
"""

import torch
import dgl
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AttackPropagationConfig:
    """攻击传播采样配置"""
    # 正样本比例 (基于攻击密集环境)
    positive_ratios: Dict[str, float] = None
    
    # 负样本策略
    negative_ratio: float = 2.0  # 负正比例 2:1 (而非异常检测的10:1)
    
    # 攻击传播特征
    temporal_window: float = 60.0  # 攻击传播时间窗口(秒)
    propagation_depth: int = 3     # 攻击传播深度
    
    # 攻击类型权重
    attack_type_weights: Dict[str, float] = None
    
    # 采样质量控制
    min_attack_confidence: float = 0.6  # 最小攻击置信度
    max_benign_ratio: float = 0.4       # 最大良性样本比例
    
    def __post_init__(self):
        if self.positive_ratios is None:
            # 基于5G-NIDD数据集的攻击密度设计
            self.positive_ratios = {
                'triggers': 0.35,      # Flow->Alert: 35% 攻击告警触发
                'initiates': 0.30,     # UE->Flow: 30% 攻击发起
                'temporal_adj': 0.25,  # Flow->Flow: 25% 攻击传播
                'default': 0.20        # 其他边类型: 20% 基础攻击关联
            }
        
        if self.attack_type_weights is None:
            # 基于5G-NIDD攻击类型分布
            self.attack_type_weights = {
                'UDPFlood': 1.0,      # 37.5% - 主导攻击
                'HTTPFlood': 0.8,     # 11.5% - 次要攻击
                'SlowrateDoS': 0.9,   # 6.0% - 持续攻击
                'Other': 0.7          # 5.5% - 其他攻击
            }

class AttackPropagationSampler:
    """
    攻击传播采样器
    专门针对攻击链路预测任务优化的正负样本生成器
    """
    
    def __init__(self, config: AttackPropagationConfig = None, seed: int = 42):
        """
        初始化攻击传播采样器
        
        Args:
            config: 攻击传播采样配置
            seed: 随机种子
        """
        self.config = config or AttackPropagationConfig()
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 攻击模式缓存
        self._attack_patterns = {}
        self._benign_patterns = {}
        
        logger.info(f"攻击传播采样器初始化完成 - 正样本比例: {self.config.positive_ratios}")
    
    def sample_positive_edges(self, graph: dgl.DGLGraph, 
                            risk_scores: torch.Tensor,
                            attack_metadata: Optional[Dict] = None) -> torch.Tensor:
        """
        采样正样本边 - 基于攻击传播模式
        
        Args:
            graph: DGL异质图
            risk_scores: 节点风险分数
            attack_metadata: 攻击元数据
            
        Returns:
            正样本边张量 [num_positive, 2]
        """
        positive_edges = []
        
        if graph.is_homogeneous:
            # 同质图：基于风险分数采样
            positive_edges = self._sample_homogeneous_positive(graph, risk_scores)
        else:
            # 异质图：基于边类型和攻击语义采样
            positive_edges = self._sample_heterogeneous_positive(graph, risk_scores, attack_metadata)
        
        return torch.tensor(positive_edges, dtype=torch.long) if positive_edges else torch.empty((0, 2), dtype=torch.long)
    
    def sample_negative_edges(self, graph: dgl.DGLGraph,
                            positive_edges: torch.Tensor,
                            risk_scores: torch.Tensor,
                            attack_metadata: Optional[Dict] = None) -> torch.Tensor:
        """
        采样负样本边 - 基于攻击传播约束
        
        Args:
            graph: DGL异质图
            positive_edges: 正样本边
            risk_scores: 节点风险分数
            attack_metadata: 攻击元数据
            
        Returns:
            负样本边张量 [num_negative, 2]
        """
        if len(positive_edges) == 0:
            return torch.empty((0, 2), dtype=torch.long)
        
        num_negative = int(len(positive_edges) * self.config.negative_ratio)
        negative_edges = []
        
        if graph.is_homogeneous:
            negative_edges = self._sample_homogeneous_negative(graph, positive_edges, risk_scores, num_negative)
        else:
            negative_edges = self._sample_heterogeneous_negative(graph, positive_edges, risk_scores, num_negative, attack_metadata)
        
        return torch.tensor(negative_edges, dtype=torch.long) if negative_edges else torch.empty((0, 2), dtype=torch.long)
    
    def _sample_heterogeneous_positive(self, graph: dgl.DGLGraph, 
                                     risk_scores: torch.Tensor,
                                     attack_metadata: Optional[Dict] = None) -> List[Tuple[int, int]]:
        """异质图正样本采样"""
        positive_edges = []
        
        for etype in graph.etypes:
            src, dst = graph.edges(etype=etype)
            if len(src) == 0:
                continue
            
            # 获取该边类型的正样本比例
            pos_ratio = self.config.positive_ratios.get(etype, self.config.positive_ratios['default'])
            num_positive = max(1, int(len(src) * pos_ratio))
            
            # 基于攻击传播逻辑选择正样本
            edge_candidates = list(zip(src.tolist(), dst.tolist()))
            
            if etype == 'initiates':  # UE -> Flow (攻击发起)
                selected_edges = self._select_attack_initiation_edges(edge_candidates, graph, risk_scores, num_positive)
            elif etype == 'triggers':  # Flow -> Alert (告警触发)
                selected_edges = self._select_alert_triggering_edges(edge_candidates, graph, risk_scores, num_positive)
            elif etype == 'temporal_adj':  # Flow -> Flow (攻击传播)
                selected_edges = self._select_attack_propagation_edges(edge_candidates, graph, risk_scores, num_positive)
            else:
                # 其他边类型：基于风险分数选择
                selected_edges = self._select_risk_based_edges(edge_candidates, graph, risk_scores, num_positive)
            
            positive_edges.extend(selected_edges)
        
        return positive_edges
    
    def _select_attack_initiation_edges(self, candidates: List[Tuple[int, int]], 
                                      graph: dgl.DGLGraph, risk_scores: torch.Tensor,
                                      num_positive: int) -> List[Tuple[int, int]]:
        """选择攻击发起边 (UE -> Flow)"""
        # 优先选择：高风险UE -> 高风险Flow的连接
        scored_candidates = []
        
        for src, dst in candidates:
            try:
                # 获取UE和Flow的风险分数
                ue_risk = self._get_node_risk_safe(graph, src, 'UE', risk_scores)
                flow_risk = self._get_node_risk_safe(graph, dst, 'Flow', risk_scores)
                
                # 攻击发起评分：UE风险 + Flow风险 + 时序一致性
                attack_score = ue_risk * 0.6 + flow_risk * 0.4
                
                # 奖励高风险组合
                if ue_risk > 0.7 and flow_risk > 0.6:
                    attack_score += 0.2
                
                scored_candidates.append((attack_score, src, dst))
                
            except Exception:
                # 如果无法获取风险分数，使用默认评分
                scored_candidates.append((0.5, src, dst))
        
        # 按攻击评分排序，选择top-k
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [(src, dst) for _, src, dst in scored_candidates[:num_positive]]
    
    def _select_alert_triggering_edges(self, candidates: List[Tuple[int, int]], 
                                     graph: dgl.DGLGraph, risk_scores: torch.Tensor,
                                     num_positive: int) -> List[Tuple[int, int]]:
        """选择告警触发边 (Flow -> Alert)"""
        scored_candidates = []
        
        for src, dst in candidates:
            try:
                flow_risk = self._get_node_risk_safe(graph, src, 'Flow', risk_scores)
                alert_risk = self._get_node_risk_safe(graph, dst, 'Alert', risk_scores)
                
                # 告警触发评分：Flow风险主导，Alert响应能力
                trigger_score = flow_risk * 0.8 + alert_risk * 0.2
                
                # 奖励高风险Flow触发告警的模式
                if flow_risk > 0.8:
                    trigger_score += 0.15
                
                scored_candidates.append((trigger_score, src, dst))
                
            except Exception:
                scored_candidates.append((0.5, src, dst))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [(src, dst) for _, src, dst in scored_candidates[:num_positive]]
    
    def _select_attack_propagation_edges(self, candidates: List[Tuple[int, int]], 
                                       graph: dgl.DGLGraph, risk_scores: torch.Tensor,
                                       num_positive: int) -> List[Tuple[int, int]]:
        """选择攻击传播边 (Flow -> Flow)"""
        scored_candidates = []
        
        for src, dst in candidates:
            try:
                src_risk = self._get_node_risk_safe(graph, src, 'Flow', risk_scores)
                dst_risk = self._get_node_risk_safe(graph, dst, 'Flow', risk_scores)
                
                # 攻击传播评分：风险递增模式
                if dst_risk > src_risk:
                    # 攻击强度递增
                    propagation_score = (dst_risk - src_risk) * 0.5 + src_risk * 0.5
                else:
                    # 攻击强度维持
                    propagation_score = src_risk * 0.7
                
                # 奖励高风险传播链
                if src_risk > 0.6 and dst_risk > 0.6:
                    propagation_score += 0.1
                
                scored_candidates.append((propagation_score, src, dst))
                
            except Exception:
                scored_candidates.append((0.5, src, dst))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [(src, dst) for _, src, dst in scored_candidates[:num_positive]]
    
    def _select_risk_based_edges(self, candidates: List[Tuple[int, int]], 
                               graph: dgl.DGLGraph, risk_scores: torch.Tensor,
                               num_positive: int) -> List[Tuple[int, int]]:
        """基于风险分数选择边"""
        scored_candidates = []
        
        for src, dst in candidates:
            try:
                src_risk = risk_scores[src].item() if src < len(risk_scores) else 0.1
                dst_risk = risk_scores[dst].item() if dst < len(risk_scores) else 0.1
                
                # 简单的风险评分
                risk_score = (src_risk + dst_risk) / 2
                scored_candidates.append((risk_score, src, dst))
                
            except Exception:
                scored_candidates.append((0.1, src, dst))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [(src, dst) for _, src, dst in scored_candidates[:num_positive]]
    
    def _get_node_risk_safe(self, graph: dgl.DGLGraph, node_id: int,
                          node_type: str, risk_scores: torch.Tensor) -> float:
        """安全获取节点风险分数"""
        try:
            if node_id < len(risk_scores):
                return risk_scores[node_id].item()
            else:
                return 0.1  # 默认低风险
        except Exception:
            return 0.1

    def _sample_heterogeneous_negative(self, graph: dgl.DGLGraph,
                                     positive_edges: torch.Tensor,
                                     risk_scores: torch.Tensor,
                                     num_negative: int,
                                     attack_metadata: Optional[Dict] = None) -> List[Tuple[int, int]]:
        """异质图负样本采样 - 基于攻击传播约束"""
        negative_edges = []
        positive_edge_set = set((src.item(), dst.item()) for src, dst in positive_edges)

        # 按边类型分配负样本
        edge_type_counts = {}
        for etype in graph.etypes:
            edge_type_counts[etype] = graph.number_of_edges(etype)

        total_edges = sum(edge_type_counts.values())

        for etype in graph.etypes:
            if total_edges == 0:
                continue

            # 按边类型比例分配负样本
            type_ratio = edge_type_counts[etype] / total_edges
            type_negative_count = max(1, int(num_negative * type_ratio))

            if etype == 'initiates':  # UE -> Flow
                type_negatives = self._generate_attack_initiation_negatives(
                    graph, positive_edge_set, risk_scores, type_negative_count
                )
            elif etype == 'triggers':  # Flow -> Alert
                type_negatives = self._generate_alert_triggering_negatives(
                    graph, positive_edge_set, risk_scores, type_negative_count
                )
            elif etype == 'temporal_adj':  # Flow -> Flow
                type_negatives = self._generate_attack_propagation_negatives(
                    graph, positive_edge_set, risk_scores, type_negative_count
                )
            else:
                type_negatives = self._generate_generic_negatives(
                    graph, positive_edge_set, risk_scores, type_negative_count, etype
                )

            negative_edges.extend(type_negatives)

        return negative_edges[:num_negative]

    def _generate_attack_initiation_negatives(self, graph: dgl.DGLGraph,
                                            positive_edge_set: Set[Tuple[int, int]],
                                            risk_scores: torch.Tensor,
                                            count: int) -> List[Tuple[int, int]]:
        """生成攻击发起负样本 - 违反攻击发起逻辑"""
        negatives = []
        attempts = 0
        max_attempts = count * 10

        # 获取UE和Flow节点
        ue_nodes = list(range(graph.number_of_nodes('UE'))) if 'UE' in graph.ntypes else []
        flow_nodes = list(range(graph.number_of_nodes('Flow'))) if 'Flow' in graph.ntypes else []

        while len(negatives) < count and attempts < max_attempts:
            attempts += 1

            if not ue_nodes or not flow_nodes:
                break

            ue_node = random.choice(ue_nodes)
            flow_node = random.choice(flow_nodes)

            edge = (ue_node, flow_node)

            # 跳过已存在的正样本
            if edge in positive_edge_set:
                continue

            # 负样本约束：违反攻击发起逻辑
            ue_risk = self._get_node_risk_safe(graph, ue_node, 'UE', risk_scores)
            flow_risk = self._get_node_risk_safe(graph, flow_node, 'Flow', risk_scores)

            # 生成语义上不合理的连接
            is_valid_negative = (
                (ue_risk < 0.3 and flow_risk > 0.7) or  # 低风险UE -> 高风险Flow
                (ue_risk > 0.8 and flow_risk < 0.2) or  # 高风险UE -> 低风险Flow (不现实)
                (ue_risk < 0.2 and flow_risk < 0.2)     # 双低风险连接
            )

            if is_valid_negative:
                negatives.append(edge)

        return negatives

    def _generate_alert_triggering_negatives(self, graph: dgl.DGLGraph,
                                           positive_edge_set: Set[Tuple[int, int]],
                                           risk_scores: torch.Tensor,
                                           count: int) -> List[Tuple[int, int]]:
        """生成告警触发负样本 - 违反告警触发逻辑"""
        negatives = []
        attempts = 0
        max_attempts = count * 10

        flow_nodes = list(range(graph.number_of_nodes('Flow'))) if 'Flow' in graph.ntypes else []
        alert_nodes = list(range(graph.number_of_nodes('Alert'))) if 'Alert' in graph.ntypes else []

        while len(negatives) < count and attempts < max_attempts:
            attempts += 1

            if not flow_nodes or not alert_nodes:
                break

            flow_node = random.choice(flow_nodes)
            alert_node = random.choice(alert_nodes)

            edge = (flow_node, alert_node)

            if edge in positive_edge_set:
                continue

            flow_risk = self._get_node_risk_safe(graph, flow_node, 'Flow', risk_scores)
            alert_risk = self._get_node_risk_safe(graph, alert_node, 'Alert', risk_scores)

            # 负样本约束：违反告警触发逻辑
            is_valid_negative = (
                (flow_risk < 0.2) or  # 低风险Flow不应触发告警
                (flow_risk > 0.9 and alert_risk < 0.1)  # 高风险Flow但告警系统无响应
            )

            if is_valid_negative:
                negatives.append(edge)

        return negatives

    def _generate_attack_propagation_negatives(self, graph: dgl.DGLGraph,
                                             positive_edge_set: Set[Tuple[int, int]],
                                             risk_scores: torch.Tensor,
                                             count: int) -> List[Tuple[int, int]]:
        """生成攻击传播负样本 - 违反时序和风险传播逻辑"""
        negatives = []
        attempts = 0
        max_attempts = count * 10

        flow_nodes = list(range(graph.number_of_nodes('Flow'))) if 'Flow' in graph.ntypes else []

        while len(negatives) < count and attempts < max_attempts:
            attempts += 1

            if len(flow_nodes) < 2:
                break

            src_flow = random.choice(flow_nodes)
            dst_flow = random.choice(flow_nodes)

            if src_flow == dst_flow:
                continue

            edge = (src_flow, dst_flow)

            if edge in positive_edge_set:
                continue

            src_risk = self._get_node_risk_safe(graph, src_flow, 'Flow', risk_scores)
            dst_risk = self._get_node_risk_safe(graph, dst_flow, 'Flow', risk_scores)

            # 负样本约束：违反攻击传播逻辑
            is_valid_negative = (
                (src_risk > 0.8 and dst_risk < 0.3) or  # 高风险 -> 低风险 (攻击衰减)
                (src_risk < 0.2 and dst_risk > 0.7) or  # 低风险 -> 高风险 (不合理跳跃)
                (src_risk < 0.3 and dst_risk < 0.3)     # 双低风险传播
            )

            if is_valid_negative:
                negatives.append(edge)

        return negatives

    def _generate_generic_negatives(self, graph: dgl.DGLGraph,
                                  positive_edge_set: Set[Tuple[int, int]],
                                  risk_scores: torch.Tensor,
                                  count: int,
                                  edge_type: str) -> List[Tuple[int, int]]:
        """生成通用负样本"""
        negatives = []
        attempts = 0
        max_attempts = count * 10

        # 获取该边类型的源和目标节点类型
        src_type, dst_type = self._get_edge_type_nodes(graph, edge_type)
        src_nodes = list(range(graph.number_of_nodes(src_type))) if src_type in graph.ntypes else []
        dst_nodes = list(range(graph.number_of_nodes(dst_type))) if dst_type in graph.ntypes else []

        while len(negatives) < count and attempts < max_attempts:
            attempts += 1

            if not src_nodes or not dst_nodes:
                break

            src_node = random.choice(src_nodes)
            dst_node = random.choice(dst_nodes)

            edge = (src_node, dst_node)

            if edge not in positive_edge_set:
                negatives.append(edge)

        return negatives

    def _get_edge_type_nodes(self, graph: dgl.DGLGraph, edge_type: str) -> Tuple[str, str]:
        """获取边类型对应的源和目标节点类型"""
        # 简化映射，实际应该根据图schema确定
        type_mapping = {
            'initiates': ('UE', 'Flow'),
            'triggers': ('Flow', 'Alert'),
            'temporal_adj': ('Flow', 'Flow'),
        }

        return type_mapping.get(edge_type, ('Flow', 'Flow'))  # 默认Flow->Flow

    def _sample_homogeneous_positive(self, graph: dgl.DGLGraph,
                                   risk_scores: torch.Tensor) -> List[Tuple[int, int]]:
        """同质图正样本采样"""
        # 基于风险分数的简单采样
        edges = graph.edges()
        src, dst = edges[0], edges[1]

        edge_risks = []
        for i in range(len(src)):
            src_risk = risk_scores[src[i]].item() if src[i] < len(risk_scores) else 0.1
            dst_risk = risk_scores[dst[i]].item() if dst[i] < len(risk_scores) else 0.1
            edge_risks.append((src_risk + dst_risk) / 2)

        # 选择风险最高的30%作为正样本
        num_positive = max(1, int(len(src) * 0.3))
        top_indices = np.argsort(edge_risks)[-num_positive:]

        return [(src[i].item(), dst[i].item()) for i in top_indices]

    def _sample_homogeneous_negative(self, graph: dgl.DGLGraph,
                                   positive_edges: torch.Tensor,
                                   risk_scores: torch.Tensor,
                                   num_negative: int) -> List[Tuple[int, int]]:
        """同质图负样本采样"""
        negatives = []
        positive_edge_set = set((src.item(), dst.item()) for src, dst in positive_edges)

        num_nodes = graph.number_of_nodes()
        attempts = 0
        max_attempts = num_negative * 10

        while len(negatives) < num_negative and attempts < max_attempts:
            attempts += 1

            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)

            if src != dst and (src, dst) not in positive_edge_set:
                negatives.append((src, dst))

        return negatives

    def get_config(self) -> Dict:
        """返回采样器配置"""
        return {
            'type': 'attack_propagation',
            'positive_ratios': self.config.positive_ratios,
            'negative_ratio': self.config.negative_ratio,
            'temporal_window': self.config.temporal_window,
            'propagation_depth': self.config.propagation_depth,
            'seed': self.seed
        }
