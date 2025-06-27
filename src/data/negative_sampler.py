"""
Negative Sampling Strategies for RiskNICLP-5G

Implements different negative sampling strategies:
- BaselineNegativeSampler: Uniform negative sampling
- RiskAwareNegativeSampler: Risk-stratified negative sampling
"""

import torch
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
import dgl
from ..common.logger import get_logger


class BaselineNegativeSampler:
    """基线均匀负采样器"""
    
    def __init__(self, neg_ratio: int = 3, seed: int = 42):
        """
        Args:
            neg_ratio: 负样本与正样本的比例
            seed: 随机种子
        """
        self.neg_ratio = neg_ratio
        self.seed = seed
        self.logger = get_logger()
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def sample_negatives(
        self, 
        graph: dgl.DGLGraph, 
        positive_edges: torch.Tensor, 
        risk_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        均匀负采样
        
        Args:
            graph: DGL图
            positive_edges: 正样本边 [num_pos_edges, 2]
            risk_scores: 风险分数 (未使用，保持接口一致)
            
        Returns:
            negative_edges: 负样本边 [num_neg_edges, 2]
        """
        num_nodes = graph.number_of_nodes()
        num_pos_edges = positive_edges.shape[0]
        num_neg_edges = num_pos_edges * self.neg_ratio
        
        # 将正样本边转换为set以快速查找
        positive_edge_set = set()
        for i in range(num_pos_edges):
            src, dst = positive_edges[i].tolist()
            positive_edge_set.add((src, dst))
            positive_edge_set.add((dst, src))  # 无向图
            
        negative_edges = []
        max_attempts = num_neg_edges * 10  # 避免无限循环
        attempts = 0
        
        while len(negative_edges) < num_neg_edges and attempts < max_attempts:
            # 随机选择两个节点
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            
            # 确保不是自环且不在正样本中
            if src != dst and (src, dst) not in positive_edge_set:
                negative_edges.append([src, dst])
                
            attempts += 1
            
        if len(negative_edges) < num_neg_edges:
            self.logger.warning(
                f"只生成了 {len(negative_edges)} 个负样本，目标是 {num_neg_edges}"
            )
            
        return torch.tensor(negative_edges, dtype=torch.long)
    
    def get_config(self) -> Dict:
        """返回采样器配置"""
        return {
            'type': 'baseline',
            'neg_ratio': self.neg_ratio,
            'seed': self.seed
        }


class RiskAwareNegativeSampler:
    """风险感知负采样器 - 重构版本"""
    
    def __init__(
        self, 
        strategy: str = 'stratified',
        neg_ratio: int = 3, 
        risk_thresholds: List[float] = [0.8, 0.5, 0.2],
        hard_ratio: float = 0.3,
        seed: int = 42,
        attack_type_mapping: Dict[str, float] = None,
        max_attempts_multiplier: int = 10,
        quality_threshold: float = 0.9,
        semantic_ratios: Dict[str, float] = None
    ):
        """
        Args:
            strategy: 采样策略 ('uniform', 'stratified', 'hard', 'semantic')
            neg_ratio: 负样本与正样本的比例
            risk_thresholds: 风险分层阈值 [高, 中, 低]
            hard_ratio: 困难负样本比例
            seed: 随机种子
            attack_type_mapping: 攻击类型到风险分数的映射
            max_attempts_multiplier: 最大尝试次数倍数
            quality_threshold: 采样质量阈值
            semantic_ratios: 异质图边类型采样比例
        """
        self.strategy = strategy
        self.neg_ratio = neg_ratio
        self.risk_thresholds = sorted(risk_thresholds, reverse=True)
        self.hard_ratio = hard_ratio
        self.seed = seed
        self.max_attempts_multiplier = max_attempts_multiplier
        self.quality_threshold = quality_threshold
        self.logger = get_logger()
        
        # 设置语义感知采样比例
        default_semantic_ratios = {
            'ue_flow_ratio': 0.4,
            'flow_alert_ratio': 0.3,
            'high_risk_ratio': 0.3
        }
        if semantic_ratios is None:
            self.semantic_ratios = default_semantic_ratios
        else:
            # 合并用户配置和默认配置，确保所有键都存在
            self.semantic_ratios = {**default_semantic_ratios, **semantic_ratios}
            
        # 类型验证：确保所有值都是float类型
        for key, value in self.semantic_ratios.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"semantic_ratios['{key}'] 必须是数值类型，得到: {type(value)}")
            self.semantic_ratios[key] = float(value)
        
        # 攻击类型到风险分数映射
        if attack_type_mapping is None:
            self.attack_type_mapping = {
                'Benign': 0.0,
                'ICMPFlood': 0.3,
                'UDPFlood': 0.4,
                'SYNFlood': 0.6,
                'HTTPFlood': 0.7,
                'SlowrateDoS': 0.8,
                'TCPConnectScan': 0.7,
                'SYNScan': 0.6,
                'UDPScan': 0.5
            }
        else:
            self.attack_type_mapping = attack_type_mapping
        
        # 🔧 动态阈值：初始化为None，在采样时基于实际数据计算
        self._dynamic_risk_thresholds = None
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 预计算常用值
        self._prepare_sampling_parameters()
    
    def _prepare_sampling_parameters(self):
        """预计算采样参数"""
        self._sampling_strategies = {
            'uniform': self._uniform_sampling,
            'stratified': self._stratified_sampling,
            'hard': self._hard_sampling,
            'semantic': self._semantic_aware_sampling
        }
        
    def _calculate_adaptive_neg_ratio(self, positive_edges: torch.Tensor, graph: dgl.DGLGraph) -> int:
        """🔧 自适应负采样比例计算，基于正样本数量和图规模"""
        num_pos_edges = positive_edges.shape[0]
        num_nodes = graph.number_of_nodes()
        
        if num_pos_edges == 0:
            return min(self.neg_ratio, 1)  # 无正样本时使用最小比例
        
        # 动态调整策略：正样本少时增加负样本，正样本多时减少负样本
        if num_pos_edges < 50:
            # 正样本极少：增加负样本比例，但限制总数
            adaptive_ratio = min(self.neg_ratio + 2, 5)
            max_neg_samples = min(adaptive_ratio * num_pos_edges, 3000)
            final_ratio = max_neg_samples // num_pos_edges if num_pos_edges > 0 else 3
        elif num_pos_edges < 200:
            # 正样本较少：标准比例
            final_ratio = self.neg_ratio
        else:
            # 正样本充足：可以减少负样本比例
            final_ratio = max(self.neg_ratio - 1, 2)
        
        return final_ratio
    
    def sample_negatives(
        self, 
        graph: dgl.DGLGraph, 
        positive_edges: torch.Tensor, 
        risk_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        统一负采样入口
        
        Args:
            graph: DGL图
            positive_edges: 正样本边 [num_pos_edges, 2]
            risk_scores: 风险分数 [num_nodes]
            
        Returns:
            negative_edges: 负样本边 [num_neg_edges, 2]
        """
        # 1. 类型检查 - 确保risk_scores是torch.Tensor
        if not torch.is_tensor(risk_scores):
            raise TypeError(
                f"risk_scores must be a torch.Tensor, but got {type(risk_scores)}. "
                f"If you have an attack_type_mapping dict, please convert it to a tensor first."
            )
        
        # 2. 维度检查
        if risk_scores.dim() != 1:
            raise ValueError(
                f"risk_scores must be a 1D tensor with shape [num_nodes], but got shape {risk_scores.shape}"
            )
        
        # 3. 数值检查
        if len(risk_scores) == 0:
            raise ValueError("risk_scores cannot be empty")
        
        # 🔧 动态阈值计算：基于实际风险分布
        self._update_dynamic_thresholds(risk_scores)
        
        # 🔧 自适应负采样比例
        original_neg_ratio = self.neg_ratio
        self.neg_ratio = self._calculate_adaptive_neg_ratio(positive_edges, graph)
        
        # 4. 策略存在性检查
        if self.strategy not in self._sampling_strategies:
            raise ValueError(f"未知的采样策略: {self.strategy}")
            
        try:
            sampling_func = self._sampling_strategies[self.strategy]
            result = sampling_func(graph, positive_edges, risk_scores)
        finally:
            # 恢复原始neg_ratio
            self.neg_ratio = original_neg_ratio
        
        return result
    
    def _update_dynamic_thresholds(self, risk_scores: torch.Tensor):
        """🔧 动态更新风险阈值，基于实际数据分布"""
        if len(risk_scores) > 0:
            # 基于分位数动态计算阈值
            q90 = torch.quantile(risk_scores, 0.9).item()
            q60 = torch.quantile(risk_scores, 0.6).item()
            q30 = torch.quantile(risk_scores, 0.3).item()
            
            self._dynamic_risk_thresholds = [q90, q60, q30]
            
            # 如果所有节点风险相同，使用微小偏移
            if q90 == q30:
                max_risk = risk_scores.max().item()
                self._dynamic_risk_thresholds = [max_risk * 0.9, max_risk * 0.6, max_risk * 0.3]
        else:
            self._dynamic_risk_thresholds = self.risk_thresholds
        
    def _uniform_sampling(
        self, 
        graph: dgl.DGLGraph, 
        positive_edges: torch.Tensor, 
        risk_scores: torch.Tensor
    ) -> torch.Tensor:
        """均匀负采样（与BaselineNegativeSampler相同）"""
        baseline_sampler = BaselineNegativeSampler(self.neg_ratio, self.seed)
        return baseline_sampler.sample_negatives(graph, positive_edges, risk_scores)
        
    def _stratified_sampling(
        self, 
        graph: dgl.DGLGraph, 
        positive_edges: torch.Tensor, 
        risk_scores: torch.Tensor
    ) -> torch.Tensor:
        """分层负采样：高风险困难负样本 + 随机简单负样本"""
        num_nodes = graph.number_of_nodes()
        num_pos_edges = positive_edges.shape[0]
        num_neg_edges = num_pos_edges * self.neg_ratio
        
        # 🔧 使用动态阈值，确保基于实际数据分布
        if self._dynamic_risk_thresholds is not None:
            thresholds = self._dynamic_risk_thresholds
        else:
            thresholds = self.risk_thresholds
        
        try:
            high_threshold = float(thresholds[0])
            low_threshold = float(thresholds[2])
        except (TypeError, ValueError, IndexError) as e:
            # 使用默认阈值
            high_threshold = 0.8
            low_threshold = 0.2
            if hasattr(self, 'logger'):
                self.logger.warning(f"风险阈值类型错误，使用默认值: {e}")
        
        # 节点风险分层 - 安全比较
        try:
            # 确保类型安全的比较
            risk_scores = risk_scores.float()
            high_threshold = float(high_threshold)
            low_threshold = float(low_threshold)
            
            high_risk_nodes = torch.where(risk_scores > high_threshold)[0]
            medium_risk_nodes = torch.where(
                (risk_scores > low_threshold) & 
                (risk_scores <= high_threshold)
            )[0]
            low_risk_nodes = torch.where(risk_scores <= low_threshold)[0]
            
            # 🔧 调试信息：显示动态阈值效果
            if hasattr(self, 'logger') and len(high_risk_nodes) > 0:
                self.logger.debug(f"动态阈值: [高:{high_threshold:.3f}, 低:{low_threshold:.3f}] -> 高风险节点:{len(high_risk_nodes)}")
                
        except Exception as e:
            # 如果风险分数类型错误，使用随机节点分组
            if hasattr(self, 'logger'):
                self.logger.warning(f"风险分数比较失败，使用随机分组: {e}")
            all_nodes = torch.arange(num_nodes)
            # 简单的三等分
            third = num_nodes // 3
            high_risk_nodes = all_nodes[:third]
            medium_risk_nodes = all_nodes[third:2*third]
            low_risk_nodes = all_nodes[2*third:]
        
        # 将正样本边转换为set
        positive_edge_set = set()
        for i in range(num_pos_edges):
            src, dst = positive_edges[i].tolist()
            positive_edge_set.add((src, dst))
            positive_edge_set.add((dst, src))
            
        negative_edges = []
        
        # 1. 高风险困难负样本 (30%)
        num_hard_edges = int(num_neg_edges * self.hard_ratio)
        if len(high_risk_nodes) > 1:
            hard_edges = self._sample_from_node_set(
                high_risk_nodes, num_hard_edges, positive_edge_set
            )
            negative_edges.extend(hard_edges)
            
        # 2. 中等风险负样本 (40%)
        num_medium_edges = int(num_neg_edges * 0.4)
        if len(medium_risk_nodes) > 1:
            medium_edges = self._sample_from_node_set(
                medium_risk_nodes, num_medium_edges, positive_edge_set
            )
            negative_edges.extend(medium_edges)
            
        # 3. 随机负样本填充剩余 (30%)
        remaining = num_neg_edges - len(negative_edges)
        if remaining > 0:
            all_nodes = torch.arange(num_nodes)
            random_edges = self._sample_from_node_set(
                all_nodes, remaining, positive_edge_set
            )
            negative_edges.extend(random_edges)
            
        # 截断到目标数量
        negative_edges = negative_edges[:num_neg_edges]
        
        return torch.tensor(negative_edges, dtype=torch.long)
        
    def _hard_sampling(
        self, 
        graph: dgl.DGLGraph, 
        positive_edges: torch.Tensor, 
        risk_scores: torch.Tensor
    ) -> torch.Tensor:
        """困难负采样：主要从高风险节点间采样"""
        num_nodes = graph.number_of_nodes()
        num_pos_edges = positive_edges.shape[0]
        num_neg_edges = num_pos_edges * self.neg_ratio
        
        # 安全获取高风险阈值
        try:
            high_threshold = float(self.risk_thresholds[0])
        except (TypeError, ValueError, IndexError) as e:
            high_threshold = 0.8  # 默认高风险阈值
            if hasattr(self, 'logger'):
                self.logger.warning(f"风险阈值类型错误，使用默认值: {e}")
        
        # 高风险节点 - 安全比较
        try:
            # 确保类型安全的比较
            risk_scores = risk_scores.float()
            high_threshold = float(high_threshold)
            high_risk_nodes = torch.where(risk_scores > high_threshold)[0]
        except Exception as e:
            # 如果风险分数比较失败，使用前20%节点作为高风险
            if hasattr(self, 'logger'):
                self.logger.warning(f"风险分数比较失败，使用默认高风险节点: {e}")
            high_risk_count = max(1, num_nodes // 5)  # 前20%
            high_risk_nodes = torch.arange(high_risk_count)
        
        # 将正样本边转换为set
        positive_edge_set = set()
        for i in range(num_pos_edges):
            src, dst = positive_edges[i].tolist()
            positive_edge_set.add((src, dst))
            positive_edge_set.add((dst, src))
            
        negative_edges = []
        
        # 主要从高风险节点采样 (80%)
        num_hard_edges = int(num_neg_edges * 0.8)
        if len(high_risk_nodes) > 1:
            hard_edges = self._sample_from_node_set(
                high_risk_nodes, num_hard_edges, positive_edge_set
            )
            negative_edges.extend(hard_edges)
            
        # 随机采样填充剩余 (20%)
        remaining = num_neg_edges - len(negative_edges)
        if remaining > 0:
            all_nodes = torch.arange(num_nodes)
            random_edges = self._sample_from_node_set(
                all_nodes, remaining, positive_edge_set
            )
            negative_edges.extend(random_edges)
            
        negative_edges = negative_edges[:num_neg_edges]
        
        return torch.tensor(negative_edges, dtype=torch.long)
        
    def _sample_from_node_set(
        self, 
        node_set: torch.Tensor, 
        num_samples: int, 
        positive_edge_set: set
    ) -> List[List[int]]:
        """从指定节点集合中采样边 - 优化版本"""
        negative_edges = []
        max_attempts = min(num_samples * 5, 5000)  # 大幅降低最大尝试次数
        attempts = 0
        
        node_list = node_set.tolist()
        
        while len(negative_edges) < num_samples and attempts < max_attempts:
            if len(node_list) < 2:
                break
                
            src = random.choice(node_list)
            dst = random.choice(node_list)
            
            if src != dst and (src, dst) not in positive_edge_set:
                negative_edges.append([src, dst])
                
            attempts += 1
            
        return negative_edges
    
    def _semantic_aware_sampling(
        self, 
        graph: dgl.DGLGraph, 
        positive_edges: torch.Tensor, 
        risk_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        优化版本的语义感知负采样 - 大幅提升性能
        
        Args:
            graph: DGL异质图
            positive_edges: 正样本边
            risk_scores: 风险分数
            
        Returns:
            negative_edges: 语义感知的负样本边
        """
        num_pos_edges = positive_edges.shape[0]
        num_neg_edges = num_pos_edges * self.neg_ratio
        
        # 将正样本边转换为set
        positive_edge_set = set()
        for i in range(num_pos_edges):
            src, dst = positive_edges[i].tolist()
            positive_edge_set.add((src, dst))
            positive_edge_set.add((dst, src))
        
        negative_edges = []
        
        if hasattr(graph, 'etypes') and len(graph.etypes) > 0:
            # 异质图的快速语义感知采样
            node_type_ranges = self._get_node_type_ranges(graph)
            
            # 安全获取采样比例，确保类型正确
            try:
                ue_flow_ratio = float(self.semantic_ratios.get('ue_flow_ratio', 0.4))
                flow_alert_ratio = float(self.semantic_ratios.get('flow_alert_ratio', 0.3))
                high_risk_ratio = float(self.semantic_ratios.get('high_risk_ratio', 0.3))
            except (TypeError, ValueError) as e:
                self.logger.warning(f"语义采样比例类型错误，使用默认值: {e}")
                ue_flow_ratio, flow_alert_ratio, high_risk_ratio = 0.4, 0.3, 0.3
            
            # 1. UE-Flow虚假发起边 (40%) - 快速采样
            num_ue_flow_edges = int(num_neg_edges * ue_flow_ratio)
            ue_flow_negatives = self._fast_cross_type_sampling(
                node_type_ranges.get('UE', []), 
                node_type_ranges.get('Flow', []),
                num_ue_flow_edges, positive_edge_set, risk_scores
            )
            negative_edges.extend(ue_flow_negatives)
            
            # 2. Flow-Alert虚假触发边 (30%) - 快速采样
            num_flow_alert_edges = int(num_neg_edges * flow_alert_ratio)
            flow_alert_negatives = self._fast_cross_type_sampling(
                node_type_ranges.get('Flow', []), 
                node_type_ranges.get('Alert', []),
                num_flow_alert_edges, positive_edge_set, risk_scores
            )
            negative_edges.extend(flow_alert_negatives)
            
            # 3. 高风险节点内部虚假连接 (30%) - 快速采样
            remaining = num_neg_edges - len(negative_edges)
            if remaining > 0:
                # 安全获取风险阈值
                try:
                    risk_threshold = float(self.risk_thresholds[0])
                except (TypeError, ValueError, IndexError):
                    risk_threshold = 0.8  # 默认高风险阈值
                
                # 安全的风险分数比较
                try:
                    # 确保risk_scores是float tensor
                    risk_scores = risk_scores.float()
                    risk_threshold = float(risk_threshold)
                    high_risk_mask = risk_scores > risk_threshold
                    high_risk_nodes = torch.where(high_risk_mask)[0]
                except Exception as e:
                    # 如果比较失败，使用前20%节点作为高风险
                    num_high_risk = max(1, graph.number_of_nodes() // 5)
                    high_risk_nodes = torch.arange(num_high_risk)
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"风险分数比较失败，使用默认高风险节点: {e}")
                
                if len(high_risk_nodes) > 1:
                    high_risk_negatives = self._fast_same_type_sampling(
                        high_risk_nodes.tolist(), remaining, positive_edge_set
                    )
                    negative_edges.extend(high_risk_negatives)
        else:
            # 同质图的风险分层采样
            negative_edges = self._stratified_sampling(graph, positive_edges, risk_scores)
            return negative_edges
        
        # 确保数量正确
        negative_edges = negative_edges[:num_neg_edges]
        
        return torch.tensor(negative_edges, dtype=torch.long) if negative_edges else torch.empty((0, 2), dtype=torch.long)
    
    def _get_node_type_ranges(self, graph: dgl.DGLGraph) -> Dict[str, List[int]]:
        """
        获取异质图中各节点类型的索引范围
        
        注意：对于异质图，节点索引需要映射到统一的全局索引空间
        """
        node_type_ranges = {}
        
        if hasattr(graph, 'ntypes') and len(graph.ntypes) > 1:
            # 异质图：使用DGL的节点ID映射
            start_idx = 0
            for ntype in graph.ntypes:
                num_nodes = graph.number_of_nodes(ntype)
                if num_nodes > 0:
                    # 对于异质图，每种节点类型的ID范围是连续的
                    node_type_ranges[ntype] = list(range(start_idx, start_idx + num_nodes))
                    start_idx += num_nodes
                else:
                    node_type_ranges[ntype] = []
        else:
            # 同质图：所有节点都属于同一类型
            num_nodes = graph.number_of_nodes()
            node_type_ranges['default'] = list(range(num_nodes))
        
        return node_type_ranges
    
    def _fast_cross_type_sampling(
        self, 
        src_nodes: List[int], 
        dst_nodes: List[int],
        num_samples: int, 
        positive_edge_set: set,
        risk_scores: torch.Tensor
    ) -> List[List[int]]:
        """快速跨类型边采样 - 避免复杂的加权采样"""
        negative_edges = []
        
        if len(src_nodes) == 0 or len(dst_nodes) == 0 or num_samples == 0:
            return negative_edges
        
        # 限制尝试次数避免无限循环
        max_attempts = min(num_samples * 5, 10000)  # 大幅降低最大尝试次数
        attempts = 0
        
        # 安全索引访问
        max_idx = len(risk_scores) - 1
        valid_src_nodes = [n for n in src_nodes if 0 <= n <= max_idx]
        valid_dst_nodes = [n for n in dst_nodes if 0 <= n <= max_idx]
        
        if not valid_src_nodes or not valid_dst_nodes:
            return negative_edges
        
        # 简化采样策略：基于风险分数排序，取前N%高风险节点
        try:
            src_risks = risk_scores[torch.tensor(valid_src_nodes)]
            dst_risks = risk_scores[torch.tensor(valid_dst_nodes)]
            
            # 取前50%高风险节点进行采样
            src_top_k = max(1, len(valid_src_nodes) // 2)
            dst_top_k = max(1, len(valid_dst_nodes) // 2)
            
            src_sorted_indices = torch.argsort(src_risks, descending=True)[:src_top_k]
            dst_sorted_indices = torch.argsort(dst_risks, descending=True)[:dst_top_k]
            
            high_risk_src_nodes = [valid_src_nodes[i] for i in src_sorted_indices]
            high_risk_dst_nodes = [valid_dst_nodes[i] for i in dst_sorted_indices]
        except:
            # 回退到随机采样
            high_risk_src_nodes = valid_src_nodes
            high_risk_dst_nodes = valid_dst_nodes
        
        # 🔧 改进多样性采样：温度控制的概率采样
        temp = 0.7  # 温度参数
        while len(negative_edges) < num_samples and attempts < max_attempts:
            try:
                # 温度控制的概率选择，而非pure random
                if len(high_risk_src_nodes) > 1:
                    src_idx = int(torch.multinomial(
                        torch.softmax(torch.randn(len(high_risk_src_nodes)) / temp, dim=0), 
                        1
                    ).item())
                    src_node = high_risk_src_nodes[src_idx]
                else:
                    src_node = high_risk_src_nodes[0] if high_risk_src_nodes else valid_src_nodes[0]
                
                if len(high_risk_dst_nodes) > 1:
                    dst_idx = int(torch.multinomial(
                        torch.softmax(torch.randn(len(high_risk_dst_nodes)) / temp, dim=0), 
                        1
                    ).item())
                    dst_node = high_risk_dst_nodes[dst_idx]
                else:
                    dst_node = high_risk_dst_nodes[0] if high_risk_dst_nodes else valid_dst_nodes[0]
            except:
                # 回退到随机采样
                src_node = random.choice(high_risk_src_nodes) if high_risk_src_nodes else random.choice(valid_src_nodes)
                dst_node = random.choice(high_risk_dst_nodes) if high_risk_dst_nodes else random.choice(valid_dst_nodes)
            
            if src_node != dst_node and (src_node, dst_node) not in positive_edge_set:
                negative_edges.append([src_node, dst_node])
                
            attempts += 1
            
        return negative_edges
    
    def _fast_same_type_sampling(
        self, 
        nodes: List[int], 
        num_samples: int, 
        positive_edge_set: set
    ) -> List[List[int]]:
        """快速同类型节点内部边采样"""
        negative_edges = []
        
        if len(nodes) < 2 or num_samples == 0:
            return negative_edges
        
        # 限制尝试次数
        max_attempts = min(num_samples * 5, 5000)  # 进一步降低尝试次数
        attempts = 0
        
        while len(negative_edges) < num_samples and attempts < max_attempts:
            src = random.choice(nodes)
            dst = random.choice(nodes)
            
            if src != dst and (src, dst) not in positive_edge_set:
                negative_edges.append([src, dst])
                
            attempts += 1
            
        return negative_edges

    def get_config(self) -> Dict:
        """返回采样器配置"""
        return {
            'type': 'risk_aware',
            'strategy': self.strategy,
            'neg_ratio': self.neg_ratio,
            'risk_thresholds': self.risk_thresholds,
            'hard_ratio': self.hard_ratio,
            'seed': self.seed,
            'attack_type_mapping': self.attack_type_mapping,
            'max_attempts_multiplier': self.max_attempts_multiplier,
            'quality_threshold': self.quality_threshold,
            'semantic_ratios': self.semantic_ratios
        }


def create_negative_sampler(config: Dict) -> BaselineNegativeSampler:
    """
    根据配置创建负采样器
    
    Args:
        config: 采样器配置
        
    Returns:
        负采样器实例
    """
    sampler_type = config.get('type', 'baseline')
    
    if sampler_type == 'baseline':
        return BaselineNegativeSampler(
            neg_ratio=config.get('neg_ratio', 3),
            seed=config.get('seed', 42)
        )
    elif sampler_type == 'risk_aware':
        return RiskAwareNegativeSampler(
            strategy=config.get('strategy', 'stratified'),
            neg_ratio=config.get('neg_ratio', 3),
            risk_thresholds=config.get('risk_thresholds', [0.8, 0.5, 0.2]),
            hard_ratio=config.get('hard_ratio', 0.3),
            seed=config.get('seed', 42),
            attack_type_mapping=config.get('attack_type_mapping', None),
            max_attempts_multiplier=config.get('max_attempts_multiplier', 10),
            quality_threshold=config.get('quality_threshold', 0.9),
            semantic_ratios=config.get('semantic_ratios', None)
        )
    else:
        raise ValueError(f"未知的采样器类型: {sampler_type}") 