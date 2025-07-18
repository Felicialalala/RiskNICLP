"""
Edge Label Cache System for RiskNICLP-5G

提前计算并缓存所有图的边标签，避免每次训练重复计算
基于边类型语义定义生成确定性标签，支持快速加载
"""

import os
import pickle
import gzip
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import dgl
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json

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
        
        # 🚀 攻击链路预测：重新设计的正样本比例配置
        self.semantic_config = semantic_config or {
            'triggers': {'pos_ratio': 0.35, 'desc': 'Flow->Alert 35%攻击告警触发预测'},
            'initiates': {'pos_ratio': 0.30, 'desc': 'UE->Flow 30%攻击发起预测'},
            'temporal_adj': {'pos_ratio': 0.25, 'desc': 'Flow->Flow 25%攻击传播预测'},
            'default': {'pos_ratio': 0.20, 'desc': '其他边类型 20%基础攻击关联'}
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

class PositiveNegativeSampleCache(EdgeLabelCache):
    """正负采样结果缓存管理器 - 扩展EdgeLabelCache"""
    
    def __init__(
        self,
        cache_dir: str = "data/edge_label_cache",
        semantic_config: Optional[Dict[str, Any]] = None,
        sample_cache_dir: Optional[str] = None,
        max_cache_size_gb: float = 10.0,
        compression_enabled: bool = True
    ):
        """
        初始化正负采样缓存
        
        Args:
            cache_dir: 边标签缓存目录
            semantic_config: 边类型语义配置
            sample_cache_dir: 采样结果缓存目录
            max_cache_size_gb: 最大缓存大小(GB)
            compression_enabled: 是否启用压缩存储
        """
        # 初始化父类
        super().__init__(cache_dir, semantic_config)
        
        # 采样缓存配置
        self.sample_cache_dir = Path(sample_cache_dir or cache_dir + "_samples")
        self.sample_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size_gb = max_cache_size_gb
        self.compression_enabled = compression_enabled
        self.cache_version = "2.0_pyramid"  # 🔥 强制更新：三层金字塔采样版本
        
        # 采样缓存索引
        self.sample_cache_index = {}
        
        # 大图检测阈值
        self.large_graph_threshold = 100000  # 10万边以上认为是大图
        
        # 性能统计
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'generation_time': 0.0,
            'load_time': 0.0
        }
        
        # 加载采样缓存索引
        self._load_sample_cache_index()
        
        self.logger.info(f"🚀 PositiveNegativeSampleCache初始化完成")
        self.logger.info(f"   📂 采样缓存目录: {self.sample_cache_dir}")
        self.logger.info(f"   💾 压缩存储: {'启用' if compression_enabled else '禁用'}")
        self.logger.info(f"   📊 缓存大小限制: {max_cache_size_gb}GB")
    
    def _load_sample_cache_index(self):
        """加载采样缓存索引"""
        index_file = self.sample_cache_dir / "sample_cache_index.pkl"
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    self.sample_cache_index = pickle.load(f)
                self.logger.info(f"📂 加载采样缓存索引: {len(self.sample_cache_index)} 个文件")
            except Exception as e:
                self.logger.warning(f"⚠️ 采样缓存索引加载失败: {e}")
                self.sample_cache_index = {}
    
    def _save_sample_cache_index(self):
        """保存采样缓存索引"""
        index_file = self.sample_cache_dir / "sample_cache_index.pkl"
        try:
            with open(index_file, 'wb') as f:
                pickle.dump(self.sample_cache_index, f)
        except Exception as e:
            self.logger.error(f"❌ 采样缓存索引保存失败: {e}")
    
    def _get_sample_cache_key(
        self, 
        graph_file: str, 
        edge_labels: torch.Tensor,
        risk_scores: torch.Tensor,
        sampler_config: Dict[str, Any]
    ) -> str:
        """
        生成采样缓存键
        
        Args:
            graph_file: 图文件路径
            edge_labels: 边标签张量
            risk_scores: 风险分数张量
            sampler_config: 采样器配置
            
        Returns:
            cache_key: 缓存键字符串
        """
        # 图文件哈希
        file_hash = hashlib.md5(str(graph_file).encode()).hexdigest()[:8]
        
        # 边标签哈希
        label_hash = hashlib.md5(edge_labels.numpy().tobytes()).hexdigest()[:8]
        
        # 风险分数哈希
        risk_hash = hashlib.md5(risk_scores.numpy().tobytes()).hexdigest()[:8]
        
        # 采样配置哈希
        config_str = json.dumps(sampler_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # 版本号
        version_hash = hashlib.md5(self.cache_version.encode()).hexdigest()[:4]
        
        return f"{file_hash}_{label_hash}_{risk_hash}_{config_hash}_{version_hash}"
    
    def _get_sample_cache_path(self, cache_key: str) -> Path:
        """获取采样缓存文件路径"""
        if self.compression_enabled:
            return self.sample_cache_dir / f"samples_{cache_key}.pkl.gz"
        else:
            return self.sample_cache_dir / f"samples_{cache_key}.pkl"
    
    def _detect_large_graph(self, graph: dgl.DGLGraph) -> bool:
        """检测是否为大图"""
        return graph.number_of_edges() > self.large_graph_threshold
    
    def _sample_cache_exists(self, cache_key: str) -> bool:
        """检查采样缓存是否存在"""
        cache_path = self._get_sample_cache_path(cache_key)
        return cache_path.exists()
    
    def _load_sample_cache(self, cache_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加载采样缓存
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Tuple[positive_edges, negative_edges]: 正样本边和负样本边
        """
        cache_path = self._get_sample_cache_path(cache_key)
        
        start_time = datetime.now()
        
        try:
            if self.compression_enabled:
                with gzip.open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
            else:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
            
            # 验证缓存数据完整性
            if ('positive_edges' not in cached_data or 
                'negative_edges' not in cached_data or
                'metadata' not in cached_data):
                raise ValueError("缓存数据格式不完整")
            
            # 验证版本兼容性
            metadata = cached_data['metadata']
            if metadata.get('cache_version') != self.cache_version:
                raise ValueError(f"缓存版本不兼容: {metadata.get('cache_version')} != {self.cache_version}")
            
            positive_edges = cached_data['positive_edges']
            negative_edges = cached_data['negative_edges']
            
            # 类型检查
            if not torch.is_tensor(positive_edges) or not torch.is_tensor(negative_edges):
                raise ValueError("缓存的采样结果类型错误")
            
            load_time = (datetime.now() - start_time).total_seconds()
            self.cache_stats['load_time'] += load_time
            self.cache_stats['hits'] += 1
            
            self.logger.debug(f"📂 采样缓存加载成功: {cache_key} ({load_time:.3f}s)")
            
            return positive_edges, negative_edges
            
        except Exception as e:
            self.logger.warning(f"⚠️ 采样缓存加载失败 {cache_key}: {e}")
            # 删除损坏的缓存文件
            if cache_path.exists():
                cache_path.unlink()
            raise e
    
    def _save_sample_cache(
        self,
        cache_key: str,
        positive_edges: torch.Tensor,
        negative_edges: torch.Tensor,
        metadata: Dict[str, Any]
    ):
        """
        保存采样缓存
        
        Args:
            cache_key: 缓存键
            positive_edges: 正样本边
            negative_edges: 负样本边
            metadata: 元数据
        """
        cache_path = self._get_sample_cache_path(cache_key)
        
        try:
            # 准备缓存数据
            cache_data = {
                'positive_edges': positive_edges,
                'negative_edges': negative_edges,
                'metadata': {
                    **metadata,
                    'cache_version': self.cache_version,
                    'generation_time': datetime.now().isoformat(),
                    'compression_enabled': self.compression_enabled
                }
            }
            
            # 保存到文件
            if self.compression_enabled:
                with gzip.open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
            
            # 更新索引
            self.sample_cache_index[cache_key] = str(cache_path)
            
            # 检查缓存大小
            self._check_cache_size()
            
            self.logger.debug(f"💾 采样缓存保存成功: {cache_key}")
            
        except Exception as e:
            self.logger.error(f"❌ 采样缓存保存失败 {cache_key}: {e}")
            # 清理失败的缓存文件
            if cache_path.exists():
                cache_path.unlink()
    
    def _check_cache_size(self):
        """检查并管理缓存大小"""
        try:
            # 计算当前缓存大小
            total_size = 0
            cache_files = []
            
            for cache_file in self.sample_cache_dir.glob("samples_*.pkl*"):
                if cache_file.exists():
                    file_size = cache_file.stat().st_size
                    total_size += file_size
                    cache_files.append((cache_file, file_size, cache_file.stat().st_mtime))
            
            current_size_gb = total_size / (1024 ** 3)
            
            # 如果超过限制，删除最旧的缓存
            if current_size_gb > self.max_cache_size_gb:
                self.logger.warning(f"⚠️ 缓存大小超限: {current_size_gb:.2f}GB > {self.max_cache_size_gb}GB")
                
                # 按修改时间排序，删除最旧的文件
                cache_files.sort(key=lambda x: x[2])  # 按修改时间排序
                
                removed_size = 0
                removed_count = 0
                target_size_gb = self.max_cache_size_gb * 0.8  # 清理到80%
                
                for cache_file, file_size, _ in cache_files:
                    if current_size_gb - removed_size / (1024 ** 3) <= target_size_gb:
                        break
                    
                    cache_file.unlink()
                    removed_size += file_size
                    removed_count += 1
                
                self.logger.info(f"🗑️ 清理旧缓存: {removed_count} 个文件 ({removed_size / (1024**2):.1f}MB)")
        
        except Exception as e:
            self.logger.error(f"❌ 缓存大小检查失败: {e}")
    
    def _generate_and_cache_samples(
        self,
        graph: dgl.DGLGraph,
        edge_labels: torch.Tensor,
        risk_scores: torch.Tensor,
        negative_sampler,
        graph_file: str,
        cache_key: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成并缓存正负采样结果
        
        Args:
            graph: DGL图
            edge_labels: 边标签
            risk_scores: 风险分数
            negative_sampler: 负采样器实例
            graph_file: 图文件路径
            cache_key: 缓存键
            
        Returns:
            Tuple[positive_edges, negative_edges]: 正负样本边
        """
        start_time = datetime.now()
        
        try:
            # 提取正样本边
            all_edges = self._extract_all_edges(graph)
            positive_mask = edge_labels == 1
            positive_edges = all_edges[positive_mask]
            
            # 生成负样本边
            negative_edges = negative_sampler.sample_negatives(graph, positive_edges, risk_scores)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            self.cache_stats['generation_time'] += generation_time
            self.cache_stats['misses'] += 1
            
            # 准备元数据
            metadata = {
                'graph_file': graph_file,
                'num_positive': len(positive_edges),
                'num_negative': len(negative_edges),
                'sampling_ratio': len(negative_edges) / max(len(positive_edges), 1),
                'sampler_type': type(negative_sampler).__name__,
                'graph_stats': {
                    'num_nodes': graph.number_of_nodes(),
                    'num_edges': graph.number_of_edges(),
                    'is_large_graph': self._detect_large_graph(graph)
                },
                'generation_time_seconds': generation_time
            }
            
            # 保存到缓存
            self._save_sample_cache(cache_key, positive_edges, negative_edges, metadata)
            
            self.logger.debug(f"🔄 生成采样结果: 正边={len(positive_edges)}, 负边={len(negative_edges)} ({generation_time:.3f}s)")
            
            return positive_edges, negative_edges
            
        except Exception as e:
            self.logger.error(f"❌ 生成采样结果失败: {e}")
            raise e
    
    def _extract_all_edges(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """提取图中的所有边"""
        if graph.is_homogeneous:
            src, dst = graph.edges()
            return torch.stack([src, dst], dim=1)
        else:
            # 异质图：收集所有边类型的边
            all_edges_list = []
            for etype in graph.etypes:
                src, dst = graph.edges(etype=etype)
                if len(src) > 0:
                    edges = torch.stack([src, dst], dim=1)
                    all_edges_list.append(edges)
            
            if all_edges_list:
                return torch.cat(all_edges_list, dim=0)
            else:
                return torch.empty((0, 2), dtype=torch.long)
    
    def get_or_generate_samples(
        self,
        graph: dgl.DGLGraph,
        edge_labels: torch.Tensor,
        risk_scores: torch.Tensor,
        negative_sampler,
        graph_file: str,
        force_regenerate: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取或生成正负采样结果（主接口）
        
        Args:
            graph: DGL图
            edge_labels: 边标签张量
            risk_scores: 风险分数张量
            negative_sampler: 负采样器实例
            graph_file: 图文件路径
            force_regenerate: 强制重新生成
            
        Returns:
            Tuple[positive_edges, negative_edges]: 正负样本边
        """
        # 获取采样器配置
        sampler_config = negative_sampler.get_config()
        
        # 生成缓存键
        cache_key = self._get_sample_cache_key(graph_file, edge_labels, risk_scores, sampler_config)
        
        # 检查缓存
        if not force_regenerate and self._sample_cache_exists(cache_key):
            try:
                positive_edges, negative_edges = self._load_sample_cache(cache_key)
                self.logger.debug(f"🚀 采样缓存命中: {Path(graph_file).name}")
                return positive_edges, negative_edges
            except Exception as e:
                self.logger.warning(f"⚠️ 缓存加载失败，重新生成: {e}")
        
        # 生成新的采样结果
        self.logger.debug(f"🔄 生成新采样结果: {Path(graph_file).name}")
        return self._generate_and_cache_samples(
            graph, edge_labels, risk_scores, negative_sampler, graph_file, cache_key
        )
    
    def precompute_all_samples(
        self,
        graph_files: List[str],
        get_graph_data_fn,
        negative_sampler,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        预计算所有图的正负采样结果
        
        Args:
            graph_files: 图文件路径列表
            get_graph_data_fn: 获取图数据的函数 (返回graph, edge_labels, risk_scores)
            negative_sampler: 负采样器实例
            force_regenerate: 强制重新生成
            
        Returns:
            statistics: 预计算统计信息
        """
        self.logger.info(f"🚀 开始预计算采样结果: {len(graph_files)} 个图文件")
        
        stats = {
            'total_files': len(graph_files),
            'cached_files': 0,
            'generated_files': 0,
            'failed_files': 0,
            'total_positive_edges': 0,
            'total_negative_edges': 0,
            'total_generation_time': 0.0,
            'error_files': []
        }
        
        for graph_file in tqdm(graph_files, desc="预计算采样结果"):
            try:
                # 获取图数据
                graph, edge_labels, risk_scores = get_graph_data_fn(graph_file)
                
                # 获取采样器配置和缓存键
                sampler_config = negative_sampler.get_config()
                cache_key = self._get_sample_cache_key(graph_file, edge_labels, risk_scores, sampler_config)
                
                if not force_regenerate and self._sample_cache_exists(cache_key):
                    stats['cached_files'] += 1
                    try:
                        positive_edges, negative_edges = self._load_sample_cache(cache_key)
                        stats['total_positive_edges'] += len(positive_edges)
                        stats['total_negative_edges'] += len(negative_edges)
                    except:
                        pass
                else:
                    # 生成采样结果
                    start_time = datetime.now()
                    positive_edges, negative_edges = self.get_or_generate_samples(
                        graph, edge_labels, risk_scores, negative_sampler, graph_file, force_regenerate
                    )
                    generation_time = (datetime.now() - start_time).total_seconds()
                    
                    stats['generated_files'] += 1
                    stats['total_positive_edges'] += len(positive_edges)
                    stats['total_negative_edges'] += len(negative_edges)
                    stats['total_generation_time'] += generation_time
                    
            except Exception as e:
                self.logger.error(f"❌ 处理文件失败 {graph_file}: {e}")
                stats['failed_files'] += 1
                stats['error_files'].append(graph_file)
        
        # 保存采样缓存索引
        self._save_sample_cache_index()
        
        # 输出统计信息
        self.logger.info(f"✅ 采样预计算完成:")
        self.logger.info(f"   📊 处理文件: {stats['generated_files']} 新生成 + {stats['cached_files']} 已缓存")
        self.logger.info(f"   📊 正样本: {stats['total_positive_edges']:,}")
        self.logger.info(f"   📊 负样本: {stats['total_negative_edges']:,}")
        if stats['total_positive_edges'] > 0:
            ratio = stats['total_negative_edges'] / stats['total_positive_edges']
            self.logger.info(f"   📊 采样比例: 1:{ratio:.1f}")
        if stats['total_generation_time'] > 0:
            self.logger.info(f"   ⏱️ 总生成时间: {stats['total_generation_time']:.1f}s")
        if stats['failed_files'] > 0:
            self.logger.warning(f"   ⚠️ 失败文件: {stats['failed_files']}")
        
        return stats
    
    def clear_sample_cache(self, pattern: Optional[str] = None):
        """清理采样缓存"""
        if pattern:
            # 清理匹配模式的缓存
            count = 0
            for cache_file in self.sample_cache_dir.glob(f"*{pattern}*"):
                cache_file.unlink()
                count += 1
            self.logger.info(f"🗑️ 清理采样缓存: {count} 个文件 (模式: {pattern})")
        else:
            # 清理所有采样缓存
            count = 0
            for cache_file in self.sample_cache_dir.glob("samples_*.pkl*"):
                cache_file.unlink()
                count += 1
            
            # 清理索引
            index_file = self.sample_cache_dir / "sample_cache_index.pkl"
            if index_file.exists():
                index_file.unlink()
            
            self.sample_cache_index = {}
            self.logger.info(f"🗑️ 清理所有采样缓存: {count} 个文件")
    
    def get_sample_cache_stats(self) -> Dict[str, Any]:
        """获取采样缓存统计信息"""
        cache_files = list(self.sample_cache_dir.glob("samples_*.pkl*"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        hit_rate = 0.0
        if self.cache_stats['hits'] + self.cache_stats['misses'] > 0:
            hit_rate = self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses'])
        
        return {
            'cached_files': len(cache_files),
            'sample_cache_dir': str(self.sample_cache_dir),
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_gb': self.max_cache_size_gb,
            'compression_enabled': self.compression_enabled,
            'index_entries': len(self.sample_cache_index),
            'cache_version': self.cache_version,
            'performance': {
                'hit_rate': hit_rate,
                'cache_hits': self.cache_stats['hits'],
                'cache_misses': self.cache_stats['misses'],
                'avg_generation_time': self.cache_stats['generation_time'] / max(self.cache_stats['misses'], 1),
                'avg_load_time': self.cache_stats['load_time'] / max(self.cache_stats['hits'], 1)
            }
        } 