"""
Edge Label Cache System for RiskNICLP-5G

提前计算并缓存所有图的边标签，避免每次训练重复计算
基于边类型语义定义生成确定性标签，支持快速加载
"""

import os
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import dgl
import numpy as np
from tqdm import tqdm

from ..common.logger import get_logger


class EdgeLabelCache:
    """边标签缓存管理器"""
    
    def __init__(
        self,
        cache_dir: str = "data/edge_label_cache",
        semantic_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化边标签缓存
        
        Args:
            cache_dir: 缓存目录
            semantic_config: 边类型语义配置
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认语义配置
        self.semantic_config = semantic_config or {
            'triggers': {'pos_ratio': 0.8, 'desc': 'Flow->Alert高风险传播'},
            'initiates': {'pos_ratio': 0.4, 'desc': 'UE->Flow中风险发起'},
            'temporal_adj': {'pos_ratio': 0.15, 'desc': 'Flow->Flow时序关联'},
            'default': {'pos_ratio': 0.1, 'desc': '其他边类型'}
        }
        
        self.logger = get_logger("EdgeLabelCache").logger
        self.cache_index = {}  # 缓存索引：file_path -> cache_file_path
        
        # 加载现有缓存索引
        self._load_cache_index()
    
    def _load_cache_index(self):
        """加载缓存索引"""
        index_file = self.cache_dir / "cache_index.pkl"
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    self.cache_index = pickle.load(f)
                self.logger.info(f"📂 加载缓存索引: {len(self.cache_index)} 个文件")
            except Exception as e:
                self.logger.warning(f"⚠️ 缓存索引加载失败: {e}")
                self.cache_index = {}
    
    def _save_cache_index(self):
        """保存缓存索引"""
        index_file = self.cache_dir / "cache_index.pkl"
        try:
            with open(index_file, 'wb') as f:
                pickle.dump(self.cache_index, f)
        except Exception as e:
            self.logger.error(f"❌ 缓存索引保存失败: {e}")
    
    def _get_cache_key(self, graph_file: str, risk_scores: torch.Tensor) -> str:
        """生成缓存键"""
        # 基于文件路径和风险分数生成唯一键
        file_hash = hashlib.md5(str(graph_file).encode()).hexdigest()[:8]
        risk_hash = hashlib.md5(risk_scores.numpy().tobytes()).hexdigest()[:8]
        semantic_hash = hashlib.md5(str(self.semantic_config).encode()).hexdigest()[:8]
        
        return f"{file_hash}_{risk_hash}_{semantic_hash}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"edge_labels_{cache_key}.pkl"
    
    def generate_edge_labels(
        self, 
        graph: dgl.DGLGraph, 
        risk_scores: torch.Tensor,
        graph_file: str
    ) -> torch.Tensor:
        """
        生成边标签（基于语义配置）
        
        Args:
            graph: DGL图
            risk_scores: 节点风险分数
            graph_file: 图文件路径
            
        Returns:
            edge_labels: 边标签张量
        """
        num_edges = graph.number_of_edges()
        if num_edges == 0:
            return torch.empty(0, dtype=torch.float32)
        
        # 异质图：基于边类型语义
        if not graph.is_homogeneous and hasattr(graph, 'etypes'):
            edge_labels = torch.zeros(num_edges, dtype=torch.float32)
            edge_offset = 0
            
            # 计算边的风险分数
            all_src_nodes = []
            all_dst_nodes = []
            
            for etype in graph.etypes:
                src, dst = graph.edges(etype=etype)
                all_src_nodes.append(src)
                all_dst_nodes.append(dst)
            
            if all_src_nodes and all_dst_nodes:
                src_nodes = torch.cat(all_src_nodes)
                dst_nodes = torch.cat(all_dst_nodes)
                
                # 确保风险分数覆盖所有节点
                max_node_id = max(src_nodes.max().item(), dst_nodes.max().item())
                if len(risk_scores) <= max_node_id:
                    # 扩展风险分数
                    extended_risk = torch.zeros(max_node_id + 1)
                    extended_risk[:len(risk_scores)] = risk_scores
                    risk_scores = extended_risk
                
                src_risks = risk_scores[src_nodes]
                dst_risks = risk_scores[dst_nodes]
                edge_risk_scores = (src_risks + dst_risks) / 2.0
            else:
                edge_risk_scores = torch.zeros(num_edges)
            
            # 按边类型分配标签
            edge_offset = 0
            for etype in graph.etypes:
                src, dst = graph.edges(etype=etype)
                num_etype_edges = len(src)
                
                if num_etype_edges > 0:
                    # 获取该边类型的正样本比例
                    config = self.semantic_config.get(etype, self.semantic_config['default'])
                    pos_ratio = config['pos_ratio']
                    
                    if pos_ratio > 0:
                        # 基于风险分数选择正样本
                        etype_edge_risks = edge_risk_scores[edge_offset:edge_offset + num_etype_edges]
                        num_positive = max(1, int(num_etype_edges * pos_ratio))
                        
                        # 选择风险最高的边作为正样本
                        if len(etype_edge_risks) > 0:
                            top_indices = torch.topk(
                                etype_edge_risks, 
                                min(num_positive, len(etype_edge_risks))
                            ).indices
                            edge_labels[edge_offset + top_indices] = 1.0
                    
                    edge_offset += num_etype_edges
        
        else:
            # 同质图：基于风险分数分层
            if graph.number_of_edges() > 0:
                src, dst = graph.edges()
                if len(src) > 0 and len(dst) > 0:
                    max_node_id = max(src.max().item(), dst.max().item())
                    if len(risk_scores) <= max_node_id:
                        extended_risk = torch.zeros(max_node_id + 1)
                        extended_risk[:len(risk_scores)] = risk_scores
                        risk_scores = extended_risk
                    
                    src_risks = risk_scores[src]
                    dst_risks = risk_scores[dst]
                    edge_risk_scores = (src_risks + dst_risks) / 2.0
                    
                    # 70%分位数作为阈值
                    risk_threshold = torch.quantile(edge_risk_scores, 0.7)
                    edge_labels = (edge_risk_scores > risk_threshold).float()
                    
                    # 确保至少20%正样本
                    min_positive = max(1, int(len(edge_labels) * 0.2))
                    if edge_labels.sum() < min_positive:
                        top_indices = torch.topk(edge_risk_scores, min_positive).indices
                        edge_labels[top_indices] = 1.0
                else:
                    edge_labels = torch.empty(0, dtype=torch.float32)
            else:
                edge_labels = torch.empty(0, dtype=torch.float32)
        
        return edge_labels
    
    def get_or_generate_labels(
        self,
        graph: dgl.DGLGraph,
        risk_scores: torch.Tensor,
        graph_file: str,
        force_regenerate: bool = False
    ) -> torch.Tensor:
        """
        获取或生成边标签（带缓存）
        
        Args:
            graph: DGL图
            risk_scores: 节点风险分数
            graph_file: 图文件路径
            force_regenerate: 强制重新生成
            
        Returns:
            edge_labels: 边标签张量
        """
        # 生成缓存键
        cache_key = self._get_cache_key(graph_file, risk_scores)
        cache_path = self._get_cache_path(cache_key)
        
        # 检查缓存
        if not force_regenerate and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # 验证缓存数据
                if (cached_data['num_edges'] == graph.number_of_edges() and
                    cached_data['semantic_config'] == self.semantic_config):
                    
                    self.logger.debug(f"📂 从缓存加载: {Path(graph_file).name}")
                    return cached_data['edge_labels']
                else:
                    self.logger.debug(f"⚠️ 缓存失效: {Path(graph_file).name} (边数或配置变化)")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ 缓存读取失败: {e}")
        
        # 生成新的边标签
        edge_labels = self.generate_edge_labels(graph, risk_scores, graph_file)
        
        # 保存到缓存
        try:
            cache_data = {
                'edge_labels': edge_labels,
                'num_edges': graph.number_of_edges(),
                'semantic_config': self.semantic_config,
                'graph_file': graph_file,
                'cache_key': cache_key
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # 更新索引
            self.cache_index[graph_file] = str(cache_path)
            
            self.logger.debug(f"💾 缓存保存: {Path(graph_file).name}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 缓存保存失败: {e}")
        
        return edge_labels
    
    def precompute_all_labels(
        self,
        graph_files: List[str],
        get_graph_and_risk_fn,
        force_regenerate: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        预计算所有图的边标签
        
        Args:
            graph_files: 图文件路径列表
            get_graph_and_risk_fn: 获取图和风险分数的函数
            force_regenerate: 强制重新生成所有缓存
            
        Returns:
            statistics: 预计算统计信息
        """
        self.logger.info(f"🚀 开始预计算边标签: {len(graph_files)} 个图文件")
        
        stats = {
            'total_files': len(graph_files),
            'cached_files': 0,
            'generated_files': 0,
            'failed_files': 0,
            'total_edges': 0,
            'total_positive': 0,
            'error_files': []
        }
        
        for graph_file in tqdm(graph_files, desc="预计算边标签"):
            try:
                # 获取图和风险分数
                graph, risk_scores = get_graph_and_risk_fn(graph_file)
                
                # 检查是否需要生成
                cache_key = self._get_cache_key(graph_file, risk_scores)
                cache_path = self._get_cache_path(cache_key)
                
                if not force_regenerate and cache_path.exists():
                    stats['cached_files'] += 1
                    # 读取统计信息
                    try:
                        with open(cache_path, 'rb') as f:
                            cached_data = pickle.load(f)
                        stats['total_edges'] += cached_data['num_edges']
                        stats['total_positive'] += cached_data['edge_labels'].sum().item()
                    except:
                        pass
                else:
                    # 生成边标签
                    edge_labels = self.get_or_generate_labels(
                        graph, risk_scores, graph_file, force_regenerate
                    )
                    
                    stats['generated_files'] += 1
                    stats['total_edges'] += len(edge_labels)
                    stats['total_positive'] += edge_labels.sum().item()
                    
            except Exception as e:
                self.logger.error(f"❌ 处理文件失败 {graph_file}: {e}")
                stats['failed_files'] += 1
                stats['error_files'].append(graph_file)
        
        # 保存缓存索引
        self._save_cache_index()
        
        # 输出统计信息
        if stats['total_edges'] > 0:
            pos_ratio = stats['total_positive'] / stats['total_edges']
            self.logger.info(f"✅ 预计算完成:")
            self.logger.info(f"   📊 处理文件: {stats['generated_files']} 新生成 + {stats['cached_files']} 已缓存")
            self.logger.info(f"   📊 总边数: {stats['total_edges']:,}")
            self.logger.info(f"   📊 正样本: {stats['total_positive']:,} ({pos_ratio:.1%})")
            if stats['failed_files'] > 0:
                self.logger.warning(f"   ⚠️ 失败文件: {stats['failed_files']}")
        
        return stats
    
    def clear_cache(self, pattern: Optional[str] = None):
        """清理缓存"""
        if pattern:
            # 清理匹配模式的缓存
            count = 0
            for cache_file in self.cache_dir.glob(f"*{pattern}*"):
                cache_file.unlink()
                count += 1
            self.logger.info(f"🗑️ 清理缓存: {count} 个文件 (模式: {pattern})")
        else:
            # 清理所有缓存
            count = 0
            for cache_file in self.cache_dir.glob("edge_labels_*.pkl"):
                cache_file.unlink()
                count += 1
            
            # 清理索引
            index_file = self.cache_dir / "cache_index.pkl"
            if index_file.exists():
                index_file.unlink()
            
            self.cache_index = {}
            self.logger.info(f"🗑️ 清理所有缓存: {count} 个文件")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        cache_files = list(self.cache_dir.glob("edge_labels_*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cached_files': len(cache_files),
            'cache_dir': str(self.cache_dir),
            'total_size_mb': total_size / (1024 * 1024),
            'index_entries': len(self.cache_index)
        } 