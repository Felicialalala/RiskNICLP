"""
Negative Sampling Strategies for RiskNICLP-5G

Implements different negative sampling strategies:
- BaselineNegativeSampler: Uniform negative sampling
- RiskAwareNegativeSampler: Risk-stratified negative sampling
"""

import torch
import dgl
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any, Set
from ..common.logger import get_logger
from pathlib import Path
import time


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
        # 处理异质图和同质图的节点数量
        if hasattr(graph, 'ntypes') and len(graph.ntypes) > 1:
            # 异质图：使用最大的节点类型作为采样空间
            num_nodes = max([graph.number_of_nodes(ntype) for ntype in graph.ntypes])
        else:
            # 同质图
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
        风险感知负采样器
        
        Args:
            strategy: 采样策略 ('uniform', 'stratified', 'hard', 'semantic')
            neg_ratio: 负采样比例
            risk_thresholds: 风险阈值列表 [high, mid, low]
            hard_ratio: 困难样本比例
            seed: 随机种子
            attack_type_mapping: 攻击类型映射权重
            max_attempts_multiplier: 最大尝试次数倍数
            quality_threshold: 质量阈值
            semantic_ratios: 语义采样比例 (已弃用，请使用semantic_pyramid策略)
        """
        # 向后兼容性处理
        if semantic_ratios is not None:
            import warnings
            warnings.warn(
                "参数'semantic_ratios'已弃用，请使用type='semantic_pyramid'的独立采样器。"
                "当前将继续工作但建议迁移到SemanticNegativeSampler。",
                DeprecationWarning,
                stacklevel=2
            )
        
        # 策略兼容性检查
        if strategy == 'semantic' and 'semantic' not in self._get_available_strategies():
            import warnings
            warnings.warn(
                "strategy='semantic'现已升级为三层金字塔架构。"
                "建议使用type='semantic_pyramid'创建专用采样器获得最佳效果。",
                FutureWarning,
                stacklevel=2
            )
            
        self.strategy = strategy
        self.neg_ratio = neg_ratio
        self.risk_thresholds = risk_thresholds
        self.hard_ratio = hard_ratio
        self.seed = seed
        self.attack_type_mapping = attack_type_mapping or {}
        self.max_attempts_multiplier = max_attempts_multiplier
        self.quality_threshold = quality_threshold
        self.semantic_ratios = semantic_ratios  # 保留以维持向后兼容性
        
        # 初始化采样统计
        self.sampling_stats = {
            'total_calls': 0,
            'successful_samples': 0,
            'fallback_count': 0,
            'avg_quality': 0.0
        }
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 预计算采样参数
        self._prepare_sampling_parameters()
    
    def _get_available_strategies(self) -> List[str]:
        """获取当前可用的采样策略列表"""
        return ['uniform', 'stratified', 'hard', 'semantic']
    
    def _prepare_sampling_parameters(self):
        """预计算采样参数"""
        self._sampling_strategies = {
            'uniform': self._uniform_sampling,
            'stratified': self._stratified_sampling,
            'hard': self._hard_sampling,
            'semantic': self._three_layer_pyramid_sampling  # 替换为新的三层金字塔采样
        }
        
    def _three_layer_pyramid_sampling(
        self, 
        graph: dgl.DGLGraph, 
        positive_edges: torch.Tensor, 
        risk_scores: torch.Tensor
    ) -> torch.Tensor:
        """三层金字塔语义采样：Easy + Mid + Hard负样本分层策略"""
        print("🔥 [三层金字塔采样] 启动新的语义负采样策略")
        
        if len(positive_edges) == 0:
            return torch.empty((0, 2), dtype=torch.long)
            
        # 🔥 新增：总体超时保护机制
        start_time = time.time()
        max_total_time = 300  # 5分钟总超时
        
        num_pos_edges = positive_edges.shape[0]
        num_neg_edges = num_pos_edges * self.neg_ratio
        
        print(f"📊 [金字塔采样] 目标: {num_pos_edges}正样本 -> {num_neg_edges}负样本 (比例1:{self.neg_ratio})")
        
        # 🔥 大图优化：如果负样本需求过大，启用分批采样
        if num_neg_edges > 500000:  # 超过50万条负样本
            print(f"⚡ [大图优化] 检测到大规模采样需求({num_neg_edges}边)，启用分批处理")
            return self._chunked_pyramid_sampling(graph, positive_edges, risk_scores, max_total_time)
        
        # 构建正样本边集合用于查重
        positive_edge_set = set(map(tuple, positive_edges.cpu().numpy()))
        
        # 计算自适应风险阈值
        thresholds = self._compute_pyramid_thresholds(risk_scores)
        print(f"🎯 [风险阈值] T_low={thresholds['T_low']:.3f}, T_high={thresholds['T_high']:.3f}")
        
        # 计算各层级需要的边数（Phase 1配置：0.6/0.3/0.1）
        num_easy = int(num_neg_edges * 0.6)  # Easy-Neg: 60%
        num_mid = int(num_neg_edges * 0.3)   # Mid-Neg: 30%
        num_hard = num_neg_edges - num_easy - num_mid  # Hard-Neg: 10%
        
        print(f"📐 [三层配置] Easy:{num_easy}(60%), Mid:{num_mid}(30%), Hard:{num_hard}(10%)")
        
        negative_edges = []
        
        # 初始化各层结果变量
        easy_negatives = []
        mid_negatives = []
        hard_negatives = []
        
        # Layer 1: Easy negatives (低风险节点连接)
        if num_easy > 0:
            print(f"🟢 [Layer 1] 开始Easy-Neg采样: 目标{num_easy}条边")
            layer_start_time = time.time()
            
            easy_negatives = self._sample_easy_negatives_pyramid(
                graph, num_easy, positive_edge_set, risk_scores, thresholds
            )
            negative_edges.extend(easy_negatives)
            
            layer_time = time.time() - layer_start_time
            print(f"✅ [Layer 1] Easy-Neg完成: 实际生成{len(easy_negatives)}条边 (耗时{layer_time:.1f}s)")
            
            # 检查超时
            if time.time() - start_time > max_total_time:
                print(f"⏰ [超时保护] Easy-Neg后总时间超限，启用紧急回退")
                return self._emergency_fallback_sampling(graph, positive_edges, negative_edges, num_neg_edges)
        
        # Layer 2: Mid negatives (异质风险连接)
        if num_mid > 0:
            print(f"🟡 [Layer 2] 开始Mid-Neg采样: 目标{num_mid}条边")
            layer_start_time = time.time()
            
            mid_negatives = self._sample_mid_negatives_pyramid(
                graph, num_mid, positive_edge_set, risk_scores, thresholds
            )
            negative_edges.extend(mid_negatives)
            
            layer_time = time.time() - layer_start_time
            print(f"✅ [Layer 2] Mid-Neg完成: 实际生成{len(mid_negatives)}条边 (耗时{layer_time:.1f}s)")
            
            # 检查超时
            if time.time() - start_time > max_total_time:
                print(f"⏰ [超时保护] Mid-Neg后总时间超限，启用紧急回退")
                return self._emergency_fallback_sampling(graph, positive_edges, negative_edges, num_neg_edges)
            
        # Layer 3: Hard negatives (高风险+互斥条件)
        if num_hard > 0:
            print(f"🔴 [Layer 3] 开始Hard-Neg采样: 目标{num_hard}条边")
            layer_start_time = time.time()
            
            hard_negatives = self._sample_hard_negatives_pyramid(
                graph, num_hard, positive_edge_set, risk_scores, thresholds
            )
            negative_edges.extend(hard_negatives)
            
            layer_time = time.time() - layer_start_time
            print(f"✅ [Layer 3] Hard-Neg完成: 实际生成{len(hard_negatives)}条边 (耗时{layer_time:.1f}s)")
        
        # 确保数量正确并返回
        negative_edges = negative_edges[:num_neg_edges]
        
        # 记录采样统计信息
        actual_generated = len(negative_edges)
        success_rate = actual_generated / num_neg_edges if num_neg_edges > 0 else 0
        total_time = time.time() - start_time
        
        self._log_pyramid_sampling_stats(actual_generated, len(easy_negatives), len(mid_negatives), len(hard_negatives))
        
        print(f"🎉 [三层金字塔] 采样完成: {actual_generated}/{num_neg_edges} (成功率{success_rate:.1%}, 总耗时{total_time:.1f}s)")
        
        return torch.tensor(negative_edges, dtype=torch.long) if negative_edges else torch.empty((0, 2), dtype=torch.long)
    
    def _compute_pyramid_thresholds(self, risk_scores: torch.Tensor) -> Dict[str, float]:
        """计算三层金字塔的自适应风险阈值"""
        return {
            'T_low': torch.quantile(risk_scores, 0.3).item(),   # 30%分位作为低风险阈值
            'T_high': torch.quantile(risk_scores, 0.7).item(),  # 70%分位作为高风险阈值
            'T_mid': torch.quantile(risk_scores, 0.5).item()    # 50%分位作为中等风险阈值
        }
    
    def _sample_easy_negatives_pyramid(self, graph, num_edges, positive_edge_set, risk_scores, thresholds):
        """Layer 1: Easy-Neg采样（低风险节点连接）"""
        if num_edges == 0:
            return []
            
        low_risk_nodes = torch.where(risk_scores < thresholds['T_low'])[0]
        
        if len(low_risk_nodes) < 2:
            # 回退到最低风险20%节点
            num_low_risk = max(2, len(risk_scores) // 5)
            _, indices = torch.topk(risk_scores, num_low_risk, largest=False)
            low_risk_nodes = indices
        
        easy_negatives = []
        max_attempts = num_edges * 20
        attempts = 0
        
        while len(easy_negatives) < num_edges and attempts < max_attempts:
            try:
                u, v = self._sample_node_pair_safe(low_risk_nodes)
                
                # Easy-Neg检查：避免正边和历史边
                if (u, v) not in positive_edge_set and \
                   not self._has_edge_in_graph(graph, u, v):
                    easy_negatives.append([u, v])
            except:
                pass  # 采样失败，继续尝试
            attempts += 1
        
        return easy_negatives
    
    def _sample_mid_negatives_pyramid(self, graph, num_edges, positive_edge_set, risk_scores, thresholds):
        """Layer 2: Mid-Neg采样（异质风险连接）- 修复大图性能"""
        if num_edges == 0:
            return []
            
        print(f"   🔍 [Mid-Neg] 检查节点分布...")
        
        high_risk_nodes = torch.where(risk_scores > thresholds['T_high'])[0]
        low_risk_nodes = torch.where(risk_scores < thresholds['T_low'])[0]
        
        print(f"   📊 [Mid-Neg] 高风险节点:{len(high_risk_nodes)}, 低风险节点:{len(low_risk_nodes)}")
        
        # 🔥 关键修复：检查节点数量是否足够高效采样
        min_nodes_for_efficient_sampling = max(100, num_edges // 1000)  # 自适应阈值
        
        if len(high_risk_nodes) < min_nodes_for_efficient_sampling or len(low_risk_nodes) < min_nodes_for_efficient_sampling:
            print(f"   ⚠️ [Mid-Neg] 高/低风险节点不足，触发自适应调整")
            
            # 自适应策略：扩大风险阈值范围
            adjusted_thresholds = self._adjust_thresholds_for_efficiency(risk_scores, num_edges)
            high_risk_nodes = torch.where(risk_scores > adjusted_thresholds['T_high'])[0]
            low_risk_nodes = torch.where(risk_scores < adjusted_thresholds['T_low'])[0]
            
            print(f"   🔧 [Mid-Neg] 调整后节点数: 高风险:{len(high_risk_nodes)}, 低风险:{len(low_risk_nodes)}")
            
            # 如果调整后仍然不足，使用回退策略
            if len(high_risk_nodes) == 0 or len(low_risk_nodes) == 0:
                print(f"   🔄 [Mid-Neg] 回退到中等风险节点内采样")
                return self._fallback_mid_range_sampling(graph, num_edges, positive_edge_set, risk_scores, thresholds)
        
        # 🔥 性能优化：预计算采样效率，设置合理的尝试次数
        sampling_efficiency = min(len(high_risk_nodes), len(low_risk_nodes)) / max(num_edges, 1)
        if sampling_efficiency < 0.1:  # 效率太低
            max_attempts = min(num_edges * 50, 100000)  # 限制最大尝试次数
            print(f"   ⚡ [Mid-Neg] 低效率模式: 采样效率{sampling_efficiency:.3f}, 限制尝试{max_attempts}次")
        else:
            max_attempts = num_edges * 10  # 高效率模式
            print(f"   🚀 [Mid-Neg] 高效率模式: 采样效率{sampling_efficiency:.3f}, 尝试{max_attempts}次")
        
        mid_negatives = []
        attempts = 0
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        # 🔥 新增：批量采样优化（每次生成多个候选）
        batch_size = min(1000, max(100, num_edges // 100))
        progress_checkpoint = max_attempts // 10  # 每10%进度报告一次
        
        while len(mid_negatives) < num_edges and attempts < max_attempts:
            batch_start = attempts
            
            # 批量生成候选边
            for _ in range(batch_size):
                if attempts >= max_attempts:
                    break
                    
                try:
                    u = self._sample_from_nodes_safe(high_risk_nodes)
                    v = self._sample_from_nodes_safe(low_risk_nodes)
                    
                    # 随机化高/低端顺序，防止方向性偏差
                    if torch.rand(1).item() < 0.5:
                        u, v = v, u
                        
                    if (u, v) not in positive_edge_set:
                        mid_negatives.append([u, v])
                        
                        # 🔥 早期成功退出：如果达到目标，立即退出
                        if len(mid_negatives) >= num_edges:
                            break
                            
                except Exception as e:
                    pass  # 静默处理异常，继续采样
                    
                attempts += 1
            
            # 进度报告
            if attempts % progress_checkpoint == 0:
                progress = (attempts / max_attempts) * 100
                current_count = len(mid_negatives)
                success_rate = current_count / attempts if attempts > 0 else 0
                print(f"   📈 [Mid-Neg] 进度:{progress:.1f}%, 已生成:{current_count}/{num_edges}, 成功率:{success_rate:.1%}")
        
        actual_generated = len(mid_negatives)
        success_rate = actual_generated / num_edges if num_edges > 0 else 0
        
        print(f"   ✅ [Mid-Neg] 完成: 生成{actual_generated}/{num_edges}边 (成功率{success_rate:.1%}, 尝试{attempts}次)")
        
        # 🔥 质量不足时的补救措施
        if actual_generated < num_edges * 0.8:  # 成功率低于80%
            print(f"   🔄 [Mid-Neg] 成功率低，启用补救采样")
            shortage = num_edges - actual_generated
            fallback_edges = self._fallback_mid_range_sampling(graph, shortage, positive_edge_set, risk_scores, thresholds)
            mid_negatives.extend(fallback_edges)
            print(f"   🔧 [Mid-Neg] 补救完成: 额外生成{len(fallback_edges)}边")
        
        return mid_negatives[:num_edges]  # 确保不超出目标数量
    
    def _sample_hard_negatives_pyramid(self, graph, num_edges, positive_edge_set, risk_scores, thresholds):
        """Layer 3: Hard-Neg采样（高风险+互斥条件）"""
        if num_edges == 0:
            return []
            
        high_risk_nodes = torch.where(risk_scores > thresholds['T_high'])[0]
        
        if len(high_risk_nodes) < 2:
            # 回退到Mid-Neg策略
            return self._sample_mid_negatives_pyramid(graph, num_edges, positive_edge_set, risk_scores, thresholds)
        
        hard_negatives = []
        fallback_count = 0
        max_attempts = num_edges * 30  # Phase 1减少尝试次数
        attempts = 0
        
        while attempts < max_attempts and len(hard_negatives) < num_edges:
            try:
                u, v = self._sample_node_pair_safe(high_risk_nodes)
                
                if (u, v) not in positive_edge_set and \
                   self._check_basic_mutual_exclusion(u, v, graph):
                    hard_negatives.append([u, v])
            except:
                pass
            attempts += 1
        
        # Hard-Neg失败回退策略
        actual_hard_count = len(hard_negatives)
        min_required = max(1, int(num_edges * 0.05))  # 最少5%
        
        if actual_hard_count < min_required:
            # 用Mid-Neg补齐不足部分
            fallback_needed = min_required - actual_hard_count
            fallback_edges = self._sample_mid_negatives_pyramid(
                graph, fallback_needed, positive_edge_set, risk_scores, thresholds
            )
            hard_negatives.extend(fallback_edges)
            fallback_count = len(fallback_edges)
        
        return hard_negatives[:num_edges]
    
    def _sample_node_pair_safe(self, node_set):
        """安全的节点对采样"""
        if len(node_set) < 2:
            raise ValueError("节点集合太小")
        
        indices = torch.randperm(len(node_set))[:2]
        u = node_set[indices[0]].item()
        v = node_set[indices[1]].item()
        return u, v
    
    def _sample_from_nodes_safe(self, node_set):
        """安全的单节点采样"""
        if len(node_set) == 0:
            raise ValueError("节点集合为空")
        
        idx = torch.randint(0, len(node_set), (1,)).item()
        return node_set[idx].item()
    
    def _sample_from_node_group(self, nodes, num_edges, positive_edge_set):
        """从同一节点组采样边"""
        sampled_edges = []
        max_attempts = num_edges * 15
        attempts = 0
        
        while len(sampled_edges) < num_edges and attempts < max_attempts:
            try:
                if len(nodes) >= 2:
                    u, v = self._sample_node_pair_safe(nodes)
                    if (u, v) not in positive_edge_set:
                        sampled_edges.append([u, v])
            except:
                pass
            attempts += 1
        
        return sampled_edges
    
    def _fallback_random_sampling_pyramid(self, graph, num_edges, positive_edge_set):
        """三层金字塔的兜底随机采样"""
        num_nodes = graph.number_of_nodes()
        sampled_edges = []
        max_attempts = num_edges * 10
        attempts = 0
        
        while len(sampled_edges) < num_edges and attempts < max_attempts:
            u = torch.randint(0, num_nodes, (1,)).item()
            v = torch.randint(0, num_nodes, (1,)).item()
            
            if u != v and (u, v) not in positive_edge_set:
                sampled_edges.append([u, v])
            attempts += 1
        
        return sampled_edges
    
    def _has_edge_in_graph(self, graph, u, v):
        """检查图中是否已存在边"""
        try:
            if graph.is_homogeneous:
                return graph.has_edge_between(u, v)
            else:
                # 异质图：检查所有边类型
                for etype in graph.etypes:
                    try:
                        if graph.has_edge_between(u, v, etype=etype):
                            return True
                    except:
                        continue
                return False
        except:
            return False
    
    def _check_basic_mutual_exclusion(self, u, v, graph):
        """基础互斥条件检查（Phase 1简化版）"""
        # 基础检查：节点不能相同
        if u == v:
            return False
        
        # Phase 1: 简化互斥逻辑，主要检查是否为不同节点即可
        # 后续Phase会逐步增强互斥条件
        return True
    
    def _log_pyramid_sampling_stats(self, total_generated, num_easy, num_mid, num_hard):
        """记录三层金字塔采样统计信息"""
        print(f"📈 [采样统计] Easy:{num_easy}, Mid:{num_mid}, Hard:{num_hard}, 总计:{total_generated}")
        
        if not hasattr(self, '_pyramid_stats'):
            self._pyramid_stats = {}
        
        self._pyramid_stats.update({
            'total_generated': total_generated,
            'easy_target': num_easy,
            'mid_target': num_mid,
            'hard_target': num_hard,
            'easy_ratio': num_easy / total_generated if total_generated > 0 else 0,
            'mid_ratio': num_mid / total_generated if total_generated > 0 else 0,
            'hard_ratio': num_hard / total_generated if total_generated > 0 else 0
        })
        
        if hasattr(self, 'logger'):
            self.logger.info(f"🔄 三层金字塔采样统计:")
            self.logger.info(f"   Easy-Neg: {num_easy} 条边")
            self.logger.info(f"   Mid-Neg: {num_mid} 条边") 
            self.logger.info(f"   Hard-Neg: {num_hard} 条边")
            self.logger.info(f"   总计: {total_generated} 条负样本边")
    
    def _adjust_thresholds_for_efficiency(self, risk_scores, num_edges):
        """自适应调整风险阈值以提高采样效率"""
        print(f"   🔧 [阈值调整] 开始自适应调整...")
        
        # 获取风险分数统计信息
        min_risk = torch.min(risk_scores).item()
        max_risk = torch.max(risk_scores).item()
        mean_risk = torch.mean(risk_scores).item()
        std_risk = torch.std(risk_scores).item()
        
        print(f"   📊 [阈值调整] 风险统计: min={min_risk:.3f}, max={max_risk:.3f}, mean={mean_risk:.3f}, std={std_risk:.3f}")
        
        # 确保有足够的节点进行高效采样
        target_nodes_per_group = max(1000, num_edges // 100)  # 至少1000个节点或num_edges/100
        total_nodes = len(risk_scores)
        
        # 如果标准差很小，说明风险分布很集中，需要微调阈值
        if std_risk < 0.1:
            # 风险分布极度集中，使用均匀分割
            print(f"   ⚠️ [阈值调整] 风险分布集中(std={std_risk:.3f})，使用均匀分割")
            
            # 按节点数量均匀分割
            sorted_risks, _ = torch.sort(risk_scores)
            low_idx = total_nodes // 3
            high_idx = total_nodes * 2 // 3
            
            adjusted_T_low = sorted_risks[low_idx].item()
            adjusted_T_high = sorted_risks[high_idx].item()
            
        else:
            # 使用动态分位数调整
            desired_high_ratio = min(0.4, target_nodes_per_group / total_nodes)  # 最多40%
            desired_low_ratio = min(0.4, target_nodes_per_group / total_nodes)   # 最多40%
            
            # 确保比例合理
            high_quantile = max(0.6, 1.0 - desired_high_ratio)  # 至少60%分位以上
            low_quantile = min(0.4, desired_low_ratio)           # 最多40%分位以下
            
            adjusted_T_high = torch.quantile(risk_scores, high_quantile).item()
            adjusted_T_low = torch.quantile(risk_scores, low_quantile).item()
            
        # 防止阈值重叠
        if adjusted_T_high <= adjusted_T_low:
            mid_point = (min_risk + max_risk) / 2
            adjusted_T_low = min_risk + (mid_point - min_risk) * 0.3
            adjusted_T_high = min_risk + (mid_point - min_risk) * 0.7
            
        # 验证调整后的节点数量
        adjusted_high_nodes = torch.sum(risk_scores > adjusted_T_high).item()
        adjusted_low_nodes = torch.sum(risk_scores < adjusted_T_low).item()
        
        print(f"   ✅ [阈值调整] 新阈值: T_low={adjusted_T_low:.3f}→{adjusted_low_nodes}节点, T_high={adjusted_T_high:.3f}→{adjusted_high_nodes}节点")
        
        return {
            'T_low': adjusted_T_low,
            'T_high': adjusted_T_high,
            'T_mid': (adjusted_T_low + adjusted_T_high) / 2
        }
    
    def _fallback_mid_range_sampling(self, graph, num_edges, positive_edge_set, risk_scores, thresholds):
        """回退策略：从中等风险节点内采样"""
        print(f"   🔄 [回退采样] 启动中等风险节点内采样")
        
        # 尝试使用中等风险节点
        mid_risk_nodes = torch.where(
            (risk_scores >= thresholds['T_low']) & 
            (risk_scores <= thresholds['T_high'])
        )[0]
        
        if len(mid_risk_nodes) >= 2:
            print(f"   📊 [回退采样] 中等风险节点:{len(mid_risk_nodes)}个")
            return self._sample_from_node_group(mid_risk_nodes, num_edges, positive_edge_set)
        
        # 如果中等风险节点也不足，使用风险分数排序的采样
        print(f"   ⚠️ [回退采样] 中等风险节点不足，使用分层随机采样")
        
        total_nodes = len(risk_scores)
        if total_nodes < 100:
            # 小图：直接随机采样
            return self._fallback_random_sampling_pyramid(graph, num_edges, positive_edge_set)
        
        # 大图：分层采样策略
        sampled_edges = []
        max_attempts = min(num_edges * 20, 50000)  # 限制最大尝试次数
        attempts = 0
        
        # 按风险分数排序，取中间部分节点
        sorted_indices = torch.argsort(risk_scores)
        quarter = total_nodes // 4
        mid_range_nodes = sorted_indices[quarter:3*quarter]  # 中间50%的节点
        
        print(f"   🎯 [回退采样] 使用中间{len(mid_range_nodes)}个节点进行采样")
        
        while len(sampled_edges) < num_edges and attempts < max_attempts:
            try:
                if len(mid_range_nodes) >= 2:
                    idx1, idx2 = torch.randperm(len(mid_range_nodes))[:2]
                    u = mid_range_nodes[idx1].item()
                    v = mid_range_nodes[idx2].item()
                    
                    if u != v and (u, v) not in positive_edge_set:
                        sampled_edges.append([u, v])
            except:
                pass
            attempts += 1
        
        print(f"   ✅ [回退采样] 完成: 生成{len(sampled_edges)}/{num_edges}边")
        return sampled_edges
    
    def _calculate_adaptive_neg_ratio(self, positive_edges: torch.Tensor, graph: dgl.DGLGraph) -> int:
        """🔧 修复自适应负采样比例计算，严格遵循配置比例"""
        num_pos_edges = positive_edges.shape[0]
        num_nodes = graph.number_of_nodes()
        
        if num_pos_edges == 0:
            return self.neg_ratio  # 修复：即使无正样本也使用配置比例
        
        # 🔧 修复：移除错误的比例减少逻辑，严格遵循配置
        # 只在极端情况下微调，保持接近配置值
        if num_pos_edges < 10:
            # 正样本极少：适度增加负样本比例避免过拟合
            final_ratio = min(self.neg_ratio + 1, self.neg_ratio + 2)
        elif num_pos_edges > 500000:
            # 正样本过多：计算资源限制，适度减少但不低于配置值
            final_ratio = max(self.neg_ratio - 1, self.neg_ratio)
        else:
            # 🔥 关键修复：正常情况严格使用配置比例
            final_ratio = self.neg_ratio
        
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
        # 🔥 新增：负采样开始日志
        num_pos_edges = len(positive_edges)
        target_neg_edges = num_pos_edges * self.neg_ratio
        print(f"🔄 负采样开始: 策略={self.strategy}, 正样本={num_pos_edges}, 目标负样本={target_neg_edges}")
        
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
        
        # 🔥 新增：显示自适应调整结果
        if self.neg_ratio != original_neg_ratio:
            print(f"   📊 自适应调整: 负样本比例 {original_neg_ratio}:1 → {self.neg_ratio}:1")
        
        # 4. 策略存在性检查
        if self.strategy not in self._sampling_strategies:
            raise ValueError(f"未知的采样策略: {self.strategy}")
            
        try:
            sampling_func = self._sampling_strategies[self.strategy]
            result = sampling_func(graph, positive_edges, risk_scores)
            
            # 🔥 新增：负采样成功日志
            actual_neg_edges = len(result)
            success_ratio = actual_neg_edges / target_neg_edges if target_neg_edges > 0 else 0
            print(f"✅ 负采样完成: 实际生成={actual_neg_edges}, 成功率={success_ratio:.1%}, 最终比例=1:{actual_neg_edges/num_pos_edges:.1f}")
            
            # 🔥 风险分布统计
            if len(result) > 0:
                try:
                    src_risks = risk_scores[result[:, 0]]
                    dst_risks = risk_scores[result[:, 1]]
                    avg_neg_risk = torch.mean(src_risks + dst_risks).item()
                    high_risk_neg_ratio = ((src_risks > 0.7) | (dst_risks > 0.7)).float().mean().item()
                    print(f"   📈 负样本风险统计: 平均风险={avg_neg_risk:.3f}, 高风险边比例={high_risk_neg_ratio:.1%}")
                except:
                    pass  # 风险统计失败不影响主流程
            
        except Exception as e:
            print(f"❌ 负采样失败: {e}")
            raise
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
        """🔥 强化困难负采样：主要从高风险节点间采样，提升训练挑战性"""
        num_nodes = graph.number_of_nodes()
        num_pos_edges = positive_edges.shape[0]
        num_neg_edges = num_pos_edges * self.neg_ratio
        
        # 🔥 关键改进：更智能的困难负样本策略
        # 1. 多层风险分级，而不是单一0.8阈值
        try:
            high_threshold = float(self.risk_thresholds[0])    # 0.8 超高风险
            mid_threshold = float(self.risk_thresholds[1])     # 0.5 中等风险
            low_threshold = float(self.risk_thresholds[2])     # 0.2 低风险
        except (TypeError, ValueError, IndexError) as e:
            high_threshold, mid_threshold, low_threshold = 0.8, 0.5, 0.2
            if hasattr(self, 'logger'):
                self.logger.warning(f"风险阈值解析失败: {e}，使用默认值")
        
        # 2. 🔥 修复：异质图节点类型分离
        node_types = {}
        edge_types = {}
        
        if not graph.is_homogeneous:
            # 异质图：按节点类型分组
            for ntype in graph.ntypes:
                node_indices = graph.nodes(ntype)
                if len(node_indices) > 0:
                    node_types[ntype] = node_indices
            
            # 按边类型分组
            for etype in graph.etypes:
                try:
                    src_nodes, dst_nodes = graph.edges(etype=etype)
                    if len(src_nodes) > 0:
                        edge_types[etype] = (src_nodes, dst_nodes)
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"边类型 {etype} 处理失败: {e}")
                    continue
        else:
            # 同质图：所有节点同一类型
            all_nodes = torch.arange(num_nodes)
            node_types['default'] = all_nodes
            src_nodes, dst_nodes = graph.edges()
            edge_types['default'] = (src_nodes, dst_nodes)
        
        # 3. 按风险等级分组节点
        high_risk_nodes = torch.where(risk_scores > high_threshold)[0]
        mid_risk_nodes = torch.where((risk_scores > mid_threshold) & (risk_scores <= high_threshold))[0]
        low_risk_nodes = torch.where((risk_scores > low_threshold) & (risk_scores <= mid_threshold))[0]
        all_nodes = torch.arange(num_nodes)
        
        if hasattr(self, 'logger'):
            self.logger.info(f"🎯 风险分层节点: 高风险={len(high_risk_nodes)}, 中风险={len(mid_risk_nodes)}, 低风险={len(low_risk_nodes)}")
        
        # 4. 构建正边集合 (用于去重)
        positive_edge_set = set()
        for i in range(len(positive_edges)):
            u, v = positive_edges[i].tolist()
            positive_edge_set.add((u, v))
            positive_edge_set.add((v, u))  # 考虑双向
        
        # 5. ‼️ 按边类型平衡的强化采样策略
        negative_edges = []
        edge_type_neg_counts = {}  # 记录每种边类型的负样本数量
        
        # ‼️ 根据正边的边类型分布来分配负样本配额
        if not graph.is_homogeneous and edge_types:
            # 计算各边类型的正边数量
            pos_etype_counts = {}
            for etype, (src_nodes, dst_nodes) in edge_types.items():
                count = 0
                for u, v in positive_edges:
                    if (u.item() in src_nodes and v.item() in dst_nodes) or \
                       (v.item() in src_nodes and u.item() in dst_nodes):
                        count += 1
                pos_etype_counts[etype] = count
            
            # 按比例分配负样本
            total_pos = sum(pos_etype_counts.values()) + 1e-8
            for etype, pos_count in pos_etype_counts.items():
                ratio = pos_count / total_pos
                edge_type_neg_counts[etype] = int(num_neg_edges * ratio)
        else:
            # 同质图：单一边类型
            edge_type_neg_counts['default'] = num_neg_edges
        
        # 按边类型分别采样
        for etype, target_count in edge_type_neg_counts.items():
            if target_count <= 0:
                continue
                
            # 策略1: 高风险节点内部连接 (50%)
            num_high_high = int(target_count * 0.5)
            if len(high_risk_nodes) > 1:
                high_high_edges = self._sample_from_node_set_safe(
                    high_risk_nodes, num_high_high, positive_edge_set,
                    node_types, edge_types, strategy="intensive", target_etype=etype
                )
                negative_edges.extend(high_high_edges)
            
            # 策略2: 高-中风险跨级连接 (30%)
            num_high_mid = int(target_count * 0.3)
            if len(high_risk_nodes) > 0 and len(mid_risk_nodes) > 0:
                high_mid_edges = self._cross_risk_sampling_safe(
                    high_risk_nodes, mid_risk_nodes, num_high_mid, positive_edge_set,
                    node_types, edge_types, target_etype=etype
                )
                negative_edges.extend(high_mid_edges)
            
            # 策略3: 随机填充 (20%)
            current_count = len([e for e in negative_edges if self._edge_matches_type(e, etype, edge_types)])
            remaining = target_count - current_count
            if remaining > 0:
                random_edges = self._sample_from_node_set_safe(
                    all_nodes, remaining, positive_edge_set,
                    node_types, edge_types, strategy="random", target_etype=etype
                )
                negative_edges.extend(random_edges)
        
        if hasattr(self, 'logger'):
            for etype, target in edge_type_neg_counts.items():
                actual = len([e for e in negative_edges if self._edge_matches_type(e, etype, edge_types)])
                self.logger.info(f"   ‼️ {etype}: 目标={target}, 实际={actual}")
        
        # 6. 🔥 去重和质量检查
        if negative_edges:
            # 转换为张量并去重
            neg_tensor = torch.tensor(negative_edges, dtype=torch.long)
            if len(neg_tensor.shape) == 2 and neg_tensor.shape[1] == 2:
                # 使用torch.unique去重 (按行)
                unique_edges = torch.unique(neg_tensor, dim=0)
                
                # 自环过滤 (u != v)
                valid_mask = unique_edges[:, 0] != unique_edges[:, 1]
                final_edges = unique_edges[valid_mask]
                
                # 困难程度验证
                if len(final_edges) > 0:
                    src_risks = risk_scores[final_edges[:, 0]]
                    dst_risks = risk_scores[final_edges[:, 1]]
                    high_risk_edge_ratio = ((src_risks > 0.8) | (dst_risks > 0.8)).float().mean().item()
                    
                    if hasattr(self, 'logger'):
                        self.logger.info(f"🔥 困难负样本质量: 高风险边比例={high_risk_edge_ratio:.1%}, 总数={len(final_edges)}")
                
                return final_edges
        
        # 兜底：返回基础随机采样
        if hasattr(self, 'logger'):
            self.logger.warning("困难负采样失败，回退到基础随机采样")
        return self._fallback_random_sampling(graph, num_neg_edges, positive_edge_set)
    
    def _sample_from_node_set_safe(
        self, 
        node_set: torch.Tensor, 
        num_edges: int, 
        positive_edge_set: set,
        node_types: dict,
        edge_types: dict,
        strategy: str = "random",
        target_etype: str = None
    ) -> list:
        """🔥 安全的节点集采样：处理异质图兼容性"""
        
        edges = []
        max_attempts = 8000 if strategy == "intensive" else 1000
        
        for _ in range(max_attempts):
            if len(edges) >= num_edges:
                break
                
            # 随机选择两个节点
            if len(node_set) < 2:
                break
                
            indices = torch.randperm(len(node_set))[:2]
            u, v = node_set[indices[0]].item(), node_set[indices[1]].item()
            
            # 检查有效性
            if u == v:  # 自环过滤
                continue
            if (u, v) in positive_edge_set or (v, u) in positive_edge_set:  # 正边去重
                continue
                
            # 🔥 异质图类型兼容性检查
            if self._is_valid_heterogeneous_edge(u, v, node_types, edge_types):
                edges.append([u, v])
        
        return edges
    
    def _cross_risk_sampling_safe(
        self,
        high_nodes: torch.Tensor,
        mid_nodes: torch.Tensor, 
        num_edges: int,
        positive_edge_set: set,
        node_types: dict,
        edge_types: dict,
        target_etype: str = None
    ) -> list:
        """🔥 跨风险级别安全采样"""
        
        edges = []
        max_attempts = 5000
        
        for _ in range(max_attempts):
            if len(edges) >= num_edges:
                break
                
            if len(high_nodes) == 0 or len(mid_nodes) == 0:
                break
                
            # 随机选择高风险和中风险节点
            u = high_nodes[torch.randint(len(high_nodes), (1,))].item()
            v = mid_nodes[torch.randint(len(mid_nodes), (1,))].item()
            
            # 有效性检查
            if u == v:
                continue
            if (u, v) in positive_edge_set or (v, u) in positive_edge_set:
                continue
                
            # 异质图类型检查
            if self._is_valid_heterogeneous_edge(u, v, node_types, edge_types):
                edges.append([u, v])
        
        return edges
    
    def _edge_matches_type(self, edge: list, target_etype: str, edge_types: dict) -> bool:
        """‼️ 检查边是否匹配指定的边类型"""
        if target_etype is None or target_etype == 'default':
            return True
            
        if target_etype not in edge_types:
            return True
            
        u, v = edge[0], edge[1]
        src_nodes, dst_nodes = edge_types[target_etype]
        
        # 检查边是否符合该类型的节点约束
        return ((u in src_nodes and v in dst_nodes) or 
                (v in src_nodes and u in dst_nodes))
    
    def _is_valid_heterogeneous_edge(
        self, 
        u: int, 
        v: int, 
        node_types: dict, 
        edge_types: dict
    ) -> bool:
        """🔥 检查异质图边的类型有效性"""
        
        # 同质图：所有边都有效
        if len(node_types) <= 1:
            return True
            
        # 异质图：检查边类型是否合法
        u_type = None
        v_type = None
        
        # 确定节点类型
        for ntype, nodes in node_types.items():
            if u in nodes:
                u_type = ntype
            if v in nodes:
                v_type = ntype
                
        if u_type is None or v_type is None:
            return False
            
        # 检查是否存在合法的边类型
        for etype, (src_nodes, dst_nodes) in edge_types.items():
            # 简化检查：如果边类型中包含这两种节点类型，认为是合法的
            if len(src_nodes) > 0 and len(dst_nodes) > 0:
                return True  # 简化为允许所有类型组合
                
        return True  # 保守策略：允许所有边
    
    def _fallback_random_sampling(
        self, 
        graph: dgl.DGLGraph, 
        num_edges: int, 
        positive_edge_set: set
    ) -> torch.Tensor:
        """兜底随机采样策略"""
        
        num_nodes = graph.number_of_nodes()
        edges = []
        max_attempts = num_edges * 10
        
        for _ in range(max_attempts):
            if len(edges) >= num_edges:
                break
                
            u = torch.randint(0, num_nodes, (1,)).item()
            v = torch.randint(0, num_nodes, (1,)).item()
            
            if u != v and (u, v) not in positive_edge_set and (v, u) not in positive_edge_set:
                edges.append([u, v])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long)
        else:
            # 最终兜底：生成简单的连续边对
            u_indices = torch.arange(min(num_edges, num_nodes // 2))
            v_indices = torch.arange(min(num_edges, num_nodes // 2)) + 1
            return torch.stack([u_indices, v_indices], dim=1)
    
    def _semantic_aware_sampling(
        self, 
        graph: dgl.DGLGraph, 
        positive_edges: torch.Tensor, 
        risk_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        DEPRECATED: 使用三层金字塔采样替代
        
        为保持向后兼容性，直接调用新的三层金字塔采样方法
        """
        import warnings
        warnings.warn(
            "⚠️ _semantic_aware_sampling已被三层金字塔采样替代，"
            "请使用strategy='semantic'配合RiskAwareNegativeSampler", 
            DeprecationWarning, 
            stacklevel=2
        )
        
        # 直接调用新的三层金字塔采样
        print("🔄 [DEPRECATED] semantic_aware_sampling -> 路由到三层金字塔采样")
        return self._three_layer_pyramid_sampling(graph, positive_edges, risk_scores)
    
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
        """快速跨类型边采样 - 大图分块处理"""
        negative_edges = []
        
        if len(src_nodes) == 0 or len(dst_nodes) == 0 or num_samples == 0:
            return negative_edges
        
        # 🔥 修复1: 大图分块采样策略
        if num_samples > 50000:  # 大图检测阈值
            print(f"🔧 检测到大图采样需求: {num_samples} 负样本，启用分块采样")
            return self._chunked_large_graph_sampling(src_nodes, dst_nodes, num_samples, positive_edge_set, risk_scores)
        
        # 🔥 修复2: 动态最大尝试次数
        max_attempts = min(num_samples * 20, 200000)  # 从10000提升至20万
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
    
    def _chunked_large_graph_sampling(
        self, 
        src_nodes: List[int], 
        dst_nodes: List[int],
        num_samples: int, 
        positive_edge_set: set,
        risk_scores: torch.Tensor
    ) -> List[List[int]]:
        """🔥 大图分块采样 - 解决大图负采样失败问题"""
        negative_edges = []
        chunk_size = 10000  # 每块最多1万负样本
        
        print(f"🔄 大图分块采样开始: 目标={num_samples}, 块大小={chunk_size}")
        
        # 安全索引访问
        max_idx = len(risk_scores) - 1
        valid_src_nodes = [n for n in src_nodes if 0 <= n <= max_idx]
        valid_dst_nodes = [n for n in dst_nodes if 0 <= n <= max_idx]
        
        if not valid_src_nodes or not valid_dst_nodes:
            print(f"❌ 无有效节点: src={len(valid_src_nodes)}, dst={len(valid_dst_nodes)}")
            return negative_edges
        
        # 预先计算高风险节点（减少重复计算）
        try:
            src_risks = risk_scores[torch.tensor(valid_src_nodes)]
            dst_risks = risk_scores[torch.tensor(valid_dst_nodes)]
            
            # 取前30%高风险节点（提高成功率）
            src_top_k = max(10, len(valid_src_nodes) // 3)
            dst_top_k = max(10, len(valid_dst_nodes) // 3)
            
            src_sorted_indices = torch.argsort(src_risks, descending=True)[:src_top_k]
            dst_sorted_indices = torch.argsort(dst_risks, descending=True)[:dst_top_k]
            
            high_risk_src_nodes = [valid_src_nodes[i] for i in src_sorted_indices]
            high_risk_dst_nodes = [valid_dst_nodes[i] for i in dst_sorted_indices]
            
            print(f"📊 高风险节点池: src={len(high_risk_src_nodes)}, dst={len(high_risk_dst_nodes)}")
        except:
            high_risk_src_nodes = valid_src_nodes
            high_risk_dst_nodes = valid_dst_nodes
        
        # 分块生成
        remaining_samples = num_samples
        chunk_count = 0
        
        while remaining_samples > 0 and chunk_count < 100:  # 最多100块
            current_chunk_size = min(chunk_size, remaining_samples)
            chunk_edges = []
            
            # 高效随机采样：预生成索引
            max_chunk_attempts = current_chunk_size * 3  # 降低尝试比例
            
            for _ in range(max_chunk_attempts):
                if len(chunk_edges) >= current_chunk_size:
                    break
                
                try:
                    src_node = random.choice(high_risk_src_nodes)
                    dst_node = random.choice(high_risk_dst_nodes)
                    
                    if (src_node != dst_node and 
                        (src_node, dst_node) not in positive_edge_set and
                        [src_node, dst_node] not in chunk_edges):
                        chunk_edges.append([src_node, dst_node])
                except:
                    continue
            
            negative_edges.extend(chunk_edges)
            remaining_samples -= len(chunk_edges)
            chunk_count += 1
            
            if chunk_count % 10 == 0:  # 每10块输出进度
                print(f"   📊 进度: 已生成 {len(negative_edges)}/{num_samples} ({len(negative_edges)/num_samples*100:.1f}%)")
        
        success_rate = len(negative_edges) / num_samples if num_samples > 0 else 0
        print(f"✅ 大图分块采样完成: 生成 {len(negative_edges)}/{num_samples} ({success_rate:.1%})")
        
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
        
        # 🔥 修复3: 大图检测与分块处理
        if num_samples > 20000:  # 大图内部采样检测
            print(f"🔧 检测到大图内部采样: {num_samples} 负样本，启用分块处理")
            return self._chunked_same_type_sampling(nodes, num_samples, positive_edge_set)
        
        # 动态尝试次数
        max_attempts = min(num_samples * 15, 100000)  # 从5000提升至10万
        attempts = 0
        
        while len(negative_edges) < num_samples and attempts < max_attempts:
            src = random.choice(nodes)
            dst = random.choice(nodes)
            
            if src != dst and (src, dst) not in positive_edge_set:
                negative_edges.append([src, dst])
                
            attempts += 1
            
        return negative_edges
    
    def _chunked_same_type_sampling(
        self, 
        nodes: List[int], 
        num_samples: int, 
        positive_edge_set: set
    ) -> List[List[int]]:
        """🔥 大图同类型节点分块采样"""
        negative_edges = []
        chunk_size = 5000  # 每块5000负样本
        
        print(f"🔄 同类型分块采样开始: 目标={num_samples}, 节点数={len(nodes)}")
        
        remaining_samples = num_samples
        chunk_count = 0
        
        while remaining_samples > 0 and chunk_count < 200:  # 最多200块
            current_chunk_size = min(chunk_size, remaining_samples)
            chunk_edges = []
            
            # 高效采样：限制尝试次数
            max_chunk_attempts = current_chunk_size * 2  # 降低尝试比例
            
            for _ in range(max_chunk_attempts):
                if len(chunk_edges) >= current_chunk_size:
                    break
                
                try:
                    src = random.choice(nodes)
                    dst = random.choice(nodes)
                    
                    if (src != dst and 
                        (src, dst) not in positive_edge_set and
                        [src, dst] not in chunk_edges):
                        chunk_edges.append([src, dst])
                except:
                    continue
            
            negative_edges.extend(chunk_edges)
            remaining_samples -= len(chunk_edges)
            chunk_count += 1
            
            if chunk_count % 20 == 0:  # 每20块输出进度
                print(f"   📊 进度: 已生成 {len(negative_edges)}/{num_samples} ({len(negative_edges)/num_samples*100:.1f}%)")
        
        success_rate = len(negative_edges) / num_samples if num_samples > 0 else 0
        print(f"✅ 同类型分块采样完成: 生成 {len(negative_edges)}/{num_samples} ({success_rate:.1%})")
        
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

    def _chunked_pyramid_sampling(self, graph, positive_edges, risk_scores, max_total_time):
        """分批金字塔采样：处理超大规模图"""
        print(f"⚡ [分批采样] 启动超大图分批处理模式")
        
        num_pos_edges = positive_edges.shape[0]
        num_neg_edges = num_pos_edges * self.neg_ratio
        
        # 分批策略：将大任务分解为多个小任务
        chunk_size = 100000  # 每批最多10万条边
        num_chunks = (num_neg_edges + chunk_size - 1) // chunk_size
        
        print(f"📊 [分批采样] 总需求{num_neg_edges}边，分为{num_chunks}批，每批{chunk_size}边")
        
        all_negative_edges = []
        positive_edge_set = set(map(tuple, positive_edges.cpu().numpy()))
        thresholds = self._compute_pyramid_thresholds(risk_scores)
        
        start_time = time.time()
        
        for chunk_idx in range(num_chunks):
            chunk_start_time = time.time()
            
            # 计算当前批次的目标边数
            remaining_edges = num_neg_edges - len(all_negative_edges)
            current_chunk_size = min(chunk_size, remaining_edges)
            
            if current_chunk_size <= 0:
                break
            
            print(f"🔄 [分批采样] 处理第{chunk_idx+1}/{num_chunks}批 (目标:{current_chunk_size}边)")
            
            # 为当前批次分配三层比例
            chunk_easy = int(current_chunk_size * 0.6)
            chunk_mid = int(current_chunk_size * 0.3)
            chunk_hard = current_chunk_size - chunk_easy - chunk_mid
            
            chunk_edges = []
            
            # 快速采样策略：针对大图优化
            if chunk_easy > 0:
                easy_edges = self._fast_easy_sampling(graph, chunk_easy, positive_edge_set, risk_scores, thresholds)
                chunk_edges.extend(easy_edges)
            
            if chunk_mid > 0:
                mid_edges = self._fast_mid_sampling(graph, chunk_mid, positive_edge_set, risk_scores, thresholds)
                chunk_edges.extend(mid_edges)
            
            if chunk_hard > 0:
                hard_edges = self._fast_hard_sampling(graph, chunk_hard, positive_edge_set, risk_scores, thresholds)
                chunk_edges.extend(hard_edges)
            
            all_negative_edges.extend(chunk_edges)
            
            chunk_time = time.time() - chunk_start_time
            total_time = time.time() - start_time
            
            print(f"✅ [分批采样] 第{chunk_idx+1}批完成: 生成{len(chunk_edges)}边 (批耗时{chunk_time:.1f}s, 总耗时{total_time:.1f}s)")
            
            # 检查超时
            if total_time > max_total_time:
                print(f"⏰ [分批采样] 超时退出，已完成{chunk_idx+1}/{num_chunks}批")
                break
        
        final_count = len(all_negative_edges)
        success_rate = final_count / num_neg_edges if num_neg_edges > 0 else 0
        total_time = time.time() - start_time
        
        print(f"🎉 [分批采样] 完成: {final_count}/{num_neg_edges}边 (成功率{success_rate:.1%}, 总耗时{total_time:.1f}s)")
        
        return torch.tensor(all_negative_edges[:num_neg_edges], dtype=torch.long) if all_negative_edges else torch.empty((0, 2), dtype=torch.long)
    
    def _emergency_fallback_sampling(self, graph, positive_edges, existing_edges, target_total):
        """紧急回退采样：超时时的快速补齐策略"""
        print(f"🚨 [紧急回退] 启动快速补齐采样")
        
        remaining_needed = target_total - len(existing_edges)
        if remaining_needed <= 0:
            return torch.tensor(existing_edges, dtype=torch.long)
        
        print(f"📊 [紧急回退] 需要补齐{remaining_needed}条边")
        
        # 快速随机采样策略
        num_nodes = graph.number_of_nodes()
        positive_edge_set = set(map(tuple, positive_edges.cpu().numpy()))
        emergency_edges = []
        
        # 限制尝试次数，避免无限循环
        max_attempts = min(remaining_needed * 5, 50000)
        attempts = 0
        
        start_time = time.time()
        emergency_timeout = 30  # 30秒紧急超时
        
        while len(emergency_edges) < remaining_needed and attempts < max_attempts:
            # 检查紧急超时
            if time.time() - start_time > emergency_timeout:
                print(f"⏰ [紧急回退] 达到紧急超时({emergency_timeout}s)，停止采样")
                break
            
            u = torch.randint(0, num_nodes, (1,)).item()
            v = torch.randint(0, num_nodes, (1,)).item()
            
            if u != v and (u, v) not in positive_edge_set:
                emergency_edges.append([u, v])
            
            attempts += 1
        
        # 合并所有边
        all_edges = existing_edges + emergency_edges
        final_edges = all_edges[:target_total]  # 确保不超出目标
        
        emergency_time = time.time() - start_time
        print(f"🚨 [紧急回退] 完成: 补齐{len(emergency_edges)}边 (总计{len(final_edges)}/{target_total}, 耗时{emergency_time:.1f}s)")
        
        return torch.tensor(final_edges, dtype=torch.long)
    
    def _fast_easy_sampling(self, graph, num_edges, positive_edge_set, risk_scores, thresholds):
        """快速Easy-Neg采样：大图优化版"""
        low_risk_nodes = torch.where(risk_scores < thresholds['T_low'])[0]
        
        if len(low_risk_nodes) < 10:
            # 节点太少，扩大范围
            num_low_risk = max(100, len(risk_scores) // 10)
            _, indices = torch.topk(risk_scores, num_low_risk, largest=False)
            low_risk_nodes = indices
        
        edges = []
        max_attempts = min(num_edges * 10, 20000)  # 减少尝试次数
        
        for _ in range(max_attempts):
            if len(edges) >= num_edges:
                break
            try:
                u, v = self._sample_node_pair_safe(low_risk_nodes)
                if (u, v) not in positive_edge_set:
                    edges.append([u, v])
            except:
                continue
        
        return edges
    
    def _fast_mid_sampling(self, graph, num_edges, positive_edge_set, risk_scores, thresholds):
        """快速Mid-Neg采样：大图优化版"""
        # 使用调整后的阈值
        adjusted_thresholds = self._adjust_thresholds_for_efficiency(risk_scores, num_edges)
        high_risk_nodes = torch.where(risk_scores > adjusted_thresholds['T_high'])[0]
        low_risk_nodes = torch.where(risk_scores < adjusted_thresholds['T_low'])[0]
        
        if len(high_risk_nodes) == 0 or len(low_risk_nodes) == 0:
            return self._fast_random_sampling(graph, num_edges, positive_edge_set)
        
        edges = []
        max_attempts = min(num_edges * 5, 10000)  # 大幅减少尝试次数
        
        for _ in range(max_attempts):
            if len(edges) >= num_edges:
                break
            try:
                u = self._sample_from_nodes_safe(high_risk_nodes)
                v = self._sample_from_nodes_safe(low_risk_nodes)
                if torch.rand(1).item() < 0.5:
                    u, v = v, u
                if (u, v) not in positive_edge_set:
                    edges.append([u, v])
            except:
                continue
        
        return edges
    
    def _fast_hard_sampling(self, graph, num_edges, positive_edge_set, risk_scores, thresholds):
        """快速Hard-Neg采样：大图优化版"""
        # 简化Hard-Neg逻辑，减少计算复杂度
        return self._fast_mid_sampling(graph, num_edges, positive_edge_set, risk_scores, thresholds)
    
    def _fast_random_sampling(self, graph, num_edges, positive_edge_set):
        """快速随机采样：兜底策略"""
        num_nodes = graph.number_of_nodes()
        edges = []
        max_attempts = min(num_edges * 5, 10000)
        
        for _ in range(max_attempts):
            if len(edges) >= num_edges:
                break
            u = torch.randint(0, num_nodes, (1,)).item()
            v = torch.randint(0, num_nodes, (1,)).item()
            if u != v and (u, v) not in positive_edge_set:
                edges.append([u, v])
        
        return edges


def create_negative_sampler(config: Dict):
    """
    根据配置创建负采样器 (重构后)

    Args:
        config: 采样器配置

    Returns:
        负采样器实例
    """
    sampler_type = config.get('type', 'baseline')

    if sampler_type == 'attack_propagation':
        # 🚀 新增：攻击传播采样器（专门针对攻击链路预测）
        from .attack_propagation_sampler import AttackPropagationSampler, AttackPropagationConfig

        # 创建攻击传播配置
        attack_config = AttackPropagationConfig(
            negative_ratio=config.get('neg_ratio', 2.0),
            temporal_window=config.get('temporal_window', 60.0),
            propagation_depth=config.get('propagation_depth', 3),
            min_attack_confidence=config.get('min_attack_confidence', 0.6)
        )

        return AttackPropagationSampler(attack_config, config.get('seed', 42))
    elif sampler_type == 'semantic_aware':
        # 🚀 使用新的语义感知采样器（重构后的最优方案）
        from .sampling_integration import create_negative_sampler_from_config
        return create_negative_sampler_from_config(config)
    elif sampler_type in ['unified', 'emergency', 'enhanced_emergency']:
        # 🔥 使用重构后的统一采样框架
        from .sampling_framework import NegativeSampler, SamplingConfig

        sampling_config = SamplingConfig(
            negative_ratio=config.get('neg_ratio', 1.0),
            hard_negative_ratio=config.get('hard_ratio', 0.3),
            temporal_negative_ratio=config.get('temporal_negative_ratio', 0.2),
            quality_threshold=config.get('quality_threshold', 0.7),
            max_attempts_multiplier=config.get('max_attempts_multiplier', 10),
            seed=config.get('seed', 42)
        )

        return NegativeSampler(sampling_config)
    elif sampler_type == 'baseline':
        return BaselineNegativeSampler(
            neg_ratio=config.get('neg_ratio', 3),
            seed=config.get('seed', 42)
        )
    elif sampler_type in ['risk_aware', 'semantic_pyramid']:
        # 🔥 风险感知和语义采样也使用统一框架
        from .sampling_framework import NegativeSampler, SamplingConfig

        sampling_config = SamplingConfig(
            negative_ratio=config.get('neg_ratio', 1.0),
            hard_negative_ratio=config.get('hard_ratio', 0.3),
            temporal_negative_ratio=config.get('temporal_negative_ratio', 0.2),
            quality_threshold=config.get('quality_threshold', 0.7),
            max_attempts_multiplier=config.get('max_attempts_multiplier', 10),
            seed=config.get('seed', 42)
        )

        return NegativeSampler(sampling_config)
    else:
        # 默认使用基线采样器
        return BaselineNegativeSampler(
            neg_ratio=config.get('neg_ratio', 3),
            seed=config.get('seed', 42)
        )


class LayerRatioScheduler:
    """动态调整层级采样比例"""
    
    def __init__(self, 
                 warmup_epochs: int = 5,
                 target_ratios: Dict = {'easy': 0.4, 'mid': 0.4, 'hard': 0.2},
                 init_ratios: Dict = {'easy': 0.6, 'mid': 0.3, 'hard': 0.1}):
        self.warmup_epochs = warmup_epochs
        self.target_ratios = target_ratios
        self.init_ratios = init_ratios
    
    def get_ratios(self, epoch: int) -> Dict[str, float]:
        """获取当前epoch的层级比例"""
        if epoch < self.warmup_epochs:
            # 训练前期: 降低Hard-Neg比例，稳定收敛
            return self.init_ratios
        else:
            # 训练后期: 提升到目标比例
            progress = min(1.0, (epoch - self.warmup_epochs) / self.warmup_epochs)
            ratios = {}
            for layer in ['easy', 'mid', 'hard']:
                init_val = self.init_ratios[layer]
                target_val = self.target_ratios[layer]
                ratios[layer] = init_val + progress * (target_val - init_val)
            return ratios

class SemanticNegativeSampler:
    """三层金字塔语义负采样器 - 工程优化版"""
    
    def __init__(self, 
                 easy_ratio: float = 0.6,  # 调整: 训练前期降低难度
                 mid_ratio: float = 0.3,   # 调整: 对应降低
                 hard_ratio: float = 0.1,  # 调整: 先用低比例验证
                 min_hard_ratio: float = 0.05,  # 新增: 失败回退最小比例
                 neg_ratio: int = 3,
                 risk_percentiles: Tuple[float, float] = (0.3, 0.7),  # 调整: smoke test用
                 mutual_exclusion_config: Dict = None,
                 layer_ratio_scheduler: Dict = None,  # 新增: 动态比例调整
                 sampling_cache_config: Dict = None,  # 新增: 采样缓存配置
                 seed: int = 42):
        
        self.neg_ratio = neg_ratio
        self.layer_ratios = {
            'easy': easy_ratio,
            'mid': mid_ratio, 
            'hard': hard_ratio
        }
        self.min_hard_ratio = min_hard_ratio
        self.risk_thresholds = risk_percentiles
        self.exclusion_config = mutual_exclusion_config or {}
        self.scheduler_config = layer_ratio_scheduler or {}
        self.cache_config = sampling_cache_config or {}
        self.seed = seed
        
        # 性能优化: 预分配采样缓存
        self._positive_edge_bitmap = None
        self._attack_metadata_cache = {}
        
        # 初始化层级比例调度器
        if self.scheduler_config.get('enable_ratio_scheduling', False):
            self.ratio_scheduler = LayerRatioScheduler(
                warmup_epochs=self.scheduler_config.get('warmup_epochs', 5),
                target_ratios=self.scheduler_config.get('target_ratios', {'easy': 0.4, 'mid': 0.4, 'hard': 0.2}),
                init_ratios=self.layer_ratios
            )
        else:
            self.ratio_scheduler = None
            
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def sample_negatives(self, graph: dgl.DGLGraph, positive_edges: torch.Tensor, risk_scores: torch.Tensor, epoch: int = None) -> torch.Tensor:
        """三层负采样主函数 - 支持动态比例调整"""
        if len(positive_edges) == 0:
            return torch.empty((0, 2), dtype=torch.long)
            
        # 更新层级比例（如果启用调度器）
        current_ratios = self._update_layer_ratios(epoch) if epoch is not None else self.layer_ratios
        
        num_pos_edges = positive_edges.shape[0]
        num_neg_edges = num_pos_edges * self.neg_ratio
        
        # 构建正样本边集合用于查重
        positive_edge_set = self._build_positive_edge_bitmap(positive_edges)
        
        # 预计算攻击元数据缓存
        self._precompute_attack_metadata(graph)
        
        # 计算各层级需要的边数
        num_easy = int(num_neg_edges * current_ratios['easy'])
        num_mid = int(num_neg_edges * current_ratios['mid'])
        num_hard = num_neg_edges - num_easy - num_mid  # 剩余的给hard
        
        negative_edges = []
        
        # Layer 1: Easy negatives (低风险节点连接)
        if num_easy > 0:
            easy_negatives = self._sample_easy_negatives_enhanced(
                graph, num_easy, positive_edge_set, risk_scores
            )
            negative_edges.extend(easy_negatives)
        
        # Layer 2: Mid negatives (异质风险连接)
        if num_mid > 0:
            mid_negatives = self._sample_mid_negatives_enhanced(
                graph, num_mid, positive_edge_set, risk_scores
            )
            negative_edges.extend(mid_negatives)
            
        # Layer 3: Hard negatives (高风险+互斥条件)
        if num_hard > 0:
            hard_negatives = self._sample_hard_negatives_enhanced(
                graph, num_hard, positive_edge_set, risk_scores
            )
            negative_edges.extend(hard_negatives)
        
        # 确保数量正确并返回
        negative_edges = negative_edges[:num_neg_edges]
        
        return torch.tensor(negative_edges, dtype=torch.long) if negative_edges else torch.empty((0, 2), dtype=torch.long)
    
    def _update_layer_ratios(self, epoch: int) -> Dict[str, float]:
        """动态调整层级比例"""
        if self.ratio_scheduler is not None:
            return self.ratio_scheduler.get_ratios(epoch)
        else:
            return self.layer_ratios
    
    def _build_positive_edge_bitmap(self, positive_edges) -> set:
        """优化: 使用set快速查重，避免O(n)"""
        return set(map(tuple, positive_edges.cpu().numpy()))
    
    def _compute_adaptive_thresholds(self, risk_scores: torch.Tensor) -> Dict[str, float]:
        """基于数据分布的自适应风险阈值"""
        return {
            'T_low': torch.quantile(risk_scores, self.risk_thresholds[0]).item(),
            'T_high': torch.quantile(risk_scores, self.risk_thresholds[1]).item(),
            'T_mid': torch.quantile(risk_scores, 0.5).item()
        }
    
    def _precompute_attack_metadata(self, graph):
        """预序列化攻击元数据到缓存"""
        if hasattr(graph, 'ndata') and not self._attack_metadata_cache:
            node_types = graph.ntypes if hasattr(graph, 'ntypes') else ['default']
            for ntype in node_types:
                try:
                    nodes = graph.nodes(ntype) if hasattr(graph, 'ntypes') else range(graph.number_of_nodes())
                    for node_id in nodes:
                        self._attack_metadata_cache[int(node_id)] = {
                            'attack_type': self._get_node_attack_type(graph, node_id, ntype),
                            'timestamp': self._get_node_timestamp(graph, node_id, ntype),
                            'port_range': self._get_node_port_range(graph, node_id, ntype),
                            'geo_location': self._get_node_location(graph, node_id, ntype)
                        }
                except Exception as e:
                    continue
    
    def _get_node_attack_type(self, graph, node_id, ntype='default'):
        """获取节点攻击类型"""
        try:
            if hasattr(graph, 'ntypes') and ntype in graph.ntypes:
                node_data = graph.nodes[ntype].data
                if 'attack_type' in node_data:
                    return node_data['attack_type'][node_id].item()
            elif hasattr(graph, 'ndata') and 'attack_type' in graph.ndata:
                return graph.ndata['attack_type'][node_id].item()
            
            # 默认值基于节点类型
            type_mapping = {'Alert': 'alert', 'Flow': 'flow', 'UE': 'ue'}
            return type_mapping.get(ntype, 'unknown')
        except:
            return 'unknown'
    
    def _get_node_timestamp(self, graph, node_id, ntype='default'):
        """获取节点时间戳"""
        try:
            if hasattr(graph, 'ntypes') and ntype in graph.ntypes:
                node_data = graph.nodes[ntype].data
                if 'timestamp' in node_data:
                    return node_data['timestamp'][node_id].item()
            elif hasattr(graph, 'ndata') and 'timestamp' in graph.ndata:
                return graph.ndata['timestamp'][node_id].item()
            return 0
        except:
            return 0
    
    def _get_node_port_range(self, graph, node_id, ntype='default'):
        """获取节点端口范围"""
        try:
            if hasattr(graph, 'ntypes') and ntype in graph.ntypes:
                node_data = graph.nodes[ntype].data
                if 'port' in node_data:
                    return node_data['port'][node_id].item()
            elif hasattr(graph, 'ndata') and 'port' in graph.ndata:
                return graph.ndata['port'][node_id].item()
            return 0
        except:
            return 0
    
    def _get_node_location(self, graph, node_id, ntype='default'):
        """获取节点地理位置"""
        try:
            if hasattr(graph, 'ntypes') and ntype in graph.ntypes:
                node_data = graph.nodes[ntype].data
                if 'location' in node_data:
                    return node_data['location'][node_id].item()
            elif hasattr(graph, 'ndata') and 'location' in graph.ndata:
                return graph.ndata['location'][node_id].item()
            return 0
        except:
            return 0
    
    def get_config(self) -> Dict:
        """获取采样器配置"""
        return {
            'type': 'semantic_pyramid',
            'neg_ratio': self.neg_ratio,
            'layer_ratios': self.layer_ratios,
            'min_hard_ratio': self.min_hard_ratio,
            'risk_percentiles': self.risk_thresholds,
            'mutual_exclusion_config': self.exclusion_config,
            'scheduler_config': self.scheduler_config,
            'cache_config': self.cache_config,
            'seed': self.seed
        }
    
    def _sample_easy_negatives_enhanced(self, graph, num_edges, positive_edge_set, risk_scores):
        """Enhanced Easy-Neg: 低风险 + 历史查重 (60%->40%)"""
        if num_edges == 0:
            return []
            
        thresholds = self._compute_adaptive_thresholds(risk_scores)
        low_risk_nodes = torch.where(risk_scores < thresholds['T_low'])[0]
        
        if len(low_risk_nodes) < 2:
            # 回退到所有节点的最低风险20%
            num_low_risk = max(2, len(risk_scores) // 5)
            _, indices = torch.topk(risk_scores, num_low_risk, largest=False)
            low_risk_nodes = indices
        
        easy_negatives = []
        max_attempts = num_edges * 20
        attempts = 0
        
        while len(easy_negatives) < num_edges and attempts < max_attempts:
            u, v = self._sample_node_pair(low_risk_nodes)
            
            # 补强1: Easy-Neg也做互斥检查，避免伪正例
            if (u, v) not in positive_edge_set and \
               not self._has_historical_edge(graph, u, v) and \
               self._check_mutual_exclusion(u, v, graph):
                easy_negatives.append([u, v])
            attempts += 1
        
        return easy_negatives

    def _sample_mid_negatives_enhanced(self, graph, num_edges, positive_edge_set, risk_scores):
        """Enhanced Mid-Neg: 异质风险 + 随机化方向 (30%->40%)"""
        if num_edges == 0:
            return []
            
        thresholds = self._compute_adaptive_thresholds(risk_scores)
        high_risk_nodes = torch.where(risk_scores > thresholds['T_high'])[0]
        low_risk_nodes = torch.where(risk_scores < thresholds['T_low'])[0]
        
        if len(high_risk_nodes) == 0 or len(low_risk_nodes) == 0:
            # 回退到中等风险节点
            mid_risk_nodes = torch.where(
                (risk_scores >= thresholds['T_low']) & 
                (risk_scores <= thresholds['T_high'])
            )[0]
            if len(mid_risk_nodes) >= 2:
                return self._sample_from_same_risk_group(mid_risk_nodes, num_edges, positive_edge_set)
            else:
                return self._fallback_random_sampling_simple(graph, num_edges, positive_edge_set)
        
        mid_negatives = []
        max_attempts = num_edges * 20
        attempts = 0
        
        while len(mid_negatives) < num_edges and attempts < max_attempts:
            u = self._sample_from_nodes(high_risk_nodes)
            v = self._sample_from_nodes(low_risk_nodes)
            
            # 补强2: 随机化高/低端顺序，防止方向性偏差
            if torch.rand(1).item() < 0.5:
                u, v = v, u
                
            if (u, v) not in positive_edge_set:
                mid_negatives.append([u, v])
            attempts += 1
        
        return mid_negatives

    def _sample_hard_negatives_enhanced(self, graph, num_edges, positive_edge_set, risk_scores):
        """Enhanced Hard-Neg: 高风险+互斥 + 失败回退 (10%->20%)"""
        if num_edges == 0:
            return []
            
        thresholds = self._compute_adaptive_thresholds(risk_scores)
        high_risk_nodes = torch.where(risk_scores > thresholds['T_high'])[0]
        
        if len(high_risk_nodes) < 2:
            # 回退到Mid-Neg策略
            return self._sample_mid_negatives_enhanced(graph, num_edges, positive_edge_set, risk_scores)
        
        hard_negatives = []
        fallback_count = 0
        max_attempts = num_edges * 50  # 增加尝试次数
        attempts = 0
        
        while attempts < max_attempts and len(hard_negatives) < num_edges:
            u, v = self._sample_node_pair(high_risk_nodes)
            
            if (u, v) not in positive_edge_set and \
               self._check_mutual_exclusion(u, v, graph):
                hard_negatives.append([u, v])
            attempts += 1
        
        # 补强3: Hard-Neg失败回退策略
        actual_hard_count = len(hard_negatives)
        min_required = max(1, int(num_edges * self.min_hard_ratio))
        
        if actual_hard_count < min_required:
            # 用Mid-Neg补齐不足部分
            fallback_needed = min_required - actual_hard_count
            fallback_edges = self._sample_mid_negatives_enhanced(
                graph, fallback_needed, positive_edge_set, risk_scores
            )
            hard_negatives.extend(fallback_edges)
            fallback_count = len(fallback_edges)
        
        # 记录退回率用于监控
        self._log_fallback_stats(actual_hard_count, fallback_count, num_edges)
        
        return hard_negatives[:num_edges]
    
    def _sample_node_pair(self, node_set):
        """从节点集合中采样一对节点"""
        if len(node_set) < 2:
            raise ValueError("节点集合太小，无法采样节点对")
        
        indices = torch.randperm(len(node_set))[:2]
        u = node_set[indices[0]].item()
        v = node_set[indices[1]].item()
        return u, v
    
    def _sample_from_nodes(self, node_set):
        """从节点集合中采样单个节点"""
        if len(node_set) == 0:
            raise ValueError("节点集合为空")
        
        idx = torch.randint(0, len(node_set), (1,)).item()
        return node_set[idx].item()
    
    def _sample_from_same_risk_group(self, nodes, num_edges, positive_edge_set):
        """从同一风险组中采样边"""
        sampled_edges = []
        max_attempts = num_edges * 20
        attempts = 0
        
        while len(sampled_edges) < num_edges and attempts < max_attempts:
            if len(nodes) >= 2:
                u, v = self._sample_node_pair(nodes)
                if (u, v) not in positive_edge_set:
                    sampled_edges.append([u, v])
            attempts += 1
        
        return sampled_edges
    
    def _fallback_random_sampling_simple(self, graph, num_edges, positive_edge_set):
        """简单随机采样作为兜底"""
        num_nodes = graph.number_of_nodes()
        sampled_edges = []
        max_attempts = num_edges * 10
        attempts = 0
        
        while len(sampled_edges) < num_edges and attempts < max_attempts:
            u = torch.randint(0, num_nodes, (1,)).item()
            v = torch.randint(0, num_nodes, (1,)).item()
            
            if u != v and (u, v) not in positive_edge_set:
                sampled_edges.append([u, v])
            attempts += 1
        
        return sampled_edges
    
    def _has_historical_edge(self, graph, u, v):
        """检查是否存在历史边"""
        # 简化实现：检查图中是否已存在该边
        try:
            if graph.is_homogeneous:
                return graph.has_edge_between(u, v)
            else:
                # 异质图：检查所有边类型
                for etype in graph.etypes:
                    try:
                        if graph.has_edge_between(u, v, etype=etype):
                            return True
                    except:
                        continue
                return False
        except:
            return False
    
    def _check_mutual_exclusion(self, u, v, graph):
        """互斥条件判断 - 性能优化版"""
        # 基础检查：节点不能相同
        if u == v:
            return False
        
        # 如果没有启用互斥检查，直接通过
        if not self.exclusion_config:
            return True
        
        # 攻击类型不兼容检查
        if self.exclusion_config.get('attack_type_check', False):
            if not self._check_attack_type_mismatch(u, v):
                return False
        
        # UE-Flow地域/端口不匹配检查
        if self.exclusion_config.get('ue_flow_mismatch', False):
            if not self._check_ue_flow_mismatch(u, v):
                return False
        
        # 时间窗口检查
        temporal_window = self.exclusion_config.get('temporal_window_hours', 0)
        if temporal_window > 0:
            if not self._check_temporal_window_violation_optimized(u, v, temporal_window):
                return False
        
        return True
    
    def _check_attack_type_mismatch(self, u, v):
        """攻击类型不兼容检查"""
        u_metadata = self._attack_metadata_cache.get(u, {})
        v_metadata = self._attack_metadata_cache.get(v, {})
        
        u_type = u_metadata.get('attack_type', 'unknown')
        v_type = v_metadata.get('attack_type', 'unknown')
        
        # 如果是相同攻击类型，则不互斥（返回False表示不满足互斥条件）
        return u_type != v_type
    
    def _check_ue_flow_mismatch(self, u, v):
        """UE-Flow地域/端口不匹配检查"""
        u_metadata = self._attack_metadata_cache.get(u, {})
        v_metadata = self._attack_metadata_cache.get(v, {})
        
        u_port = u_metadata.get('port_range', 0)
        v_port = v_metadata.get('port_range', 0)
        u_location = u_metadata.get('geo_location', 0)
        v_location = v_metadata.get('geo_location', 0)
        
        # 端口范围或地理位置不匹配才满足互斥条件
        port_mismatch = abs(u_port - v_port) > 1000  # 端口差异阈值
        location_mismatch = u_location != v_location
        
        return port_mismatch or location_mismatch
    
    def _check_temporal_window_violation_optimized(self, u, v, max_time_diff_hours=2):
        """O(log n)时间窗口检查"""
        u_metadata = self._attack_metadata_cache.get(u, {})
        v_metadata = self._attack_metadata_cache.get(v, {})
        
        u_time = u_metadata.get('timestamp', 0)
        v_time = v_metadata.get('timestamp', 0)
        
        time_diff_hours = abs(u_time - v_time) / 3600  # 假设timestamp是秒
        return time_diff_hours > max_time_diff_hours
    
    def _log_fallback_stats(self, actual_hard_count, fallback_count, target_count):
        """记录退回统计信息"""
        fallback_rate = fallback_count / target_count if target_count > 0 else 0
        # 简化版本：存储到实例变量中供外部访问
        if not hasattr(self, '_sampling_stats'):
            self._sampling_stats = {}
        
        self._sampling_stats.update({
            'hard_sampling_success_count': actual_hard_count,
            'fallback_count': fallback_count,
            'target_hard_count': target_count,
            'fallback_rate': fallback_rate
        }) 