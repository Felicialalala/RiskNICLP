"""
Test suite for PositiveNegativeSampleCache

测试正负采样结果缓存系统的各项功能
"""

import os
import tempfile
import shutil
import torch
import dgl
import pytest
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.edge_label_cache import PositiveNegativeSampleCache
from src.data.negative_sampler import create_negative_sampler


class TestPositiveNegativeSampleCache:
    """PositiveNegativeSampleCache测试类"""
    
    def setup_method(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "edge_cache"
        self.sample_cache_dir = Path(self.temp_dir) / "sample_cache"
        
        # 创建测试缓存实例
        self.cache = PositiveNegativeSampleCache(
            cache_dir=str(self.cache_dir),
            sample_cache_dir=str(self.sample_cache_dir),
            max_cache_size_gb=1.0,  # 1GB限制用于测试
            compression_enabled=True
        )
        
        # 创建测试负采样器
        self.negative_sampler = create_negative_sampler({
            'type': 'risk_aware',
            'strategy': 'semantic',
            'neg_ratio': 3,
            'seed': 42
        })
        
    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_graph(self, num_nodes: int = 100, num_edges: int = 300) -> dgl.DGLGraph:
        """创建测试图"""
        # 创建随机异质图
        graph_data = {
            ('UE', 'initiates', 'Flow'): (torch.randint(0, num_nodes//3, (num_edges//3,)), 
                                          torch.randint(num_nodes//3, 2*num_nodes//3, (num_edges//3,))),
            ('Flow', 'triggers', 'Alert'): (torch.randint(num_nodes//3, 2*num_nodes//3, (num_edges//3,)), 
                                           torch.randint(2*num_nodes//3, num_nodes, (num_edges//3,))),
            ('Flow', 'temporal_adj', 'Flow'): (torch.randint(num_nodes//3, 2*num_nodes//3, (num_edges//3,)), 
                                              torch.randint(num_nodes//3, 2*num_nodes//3, (num_edges//3,)))
        }
        
        graph = dgl.heterograph(graph_data)
        
        # 添加节点特征
        for ntype in graph.ntypes:
            num_nodes_type = graph.number_of_nodes(ntype)
            graph.nodes[ntype].data['feat'] = torch.randn(num_nodes_type, 16)
            
        return graph
    
    def create_test_data(self) -> tuple:
        """创建测试数据"""
        graph = self.create_test_graph()
        
        # 创建边标签
        num_edges = graph.number_of_edges()
        edge_labels = torch.zeros(num_edges, dtype=torch.float32)
        # 随机设置20%的边为正样本
        num_positive = num_edges // 5
        positive_indices = torch.randperm(num_edges)[:num_positive]
        edge_labels[positive_indices] = 1.0
        
        # 创建风险分数
        num_nodes = graph.number_of_nodes()
        risk_scores = torch.rand(num_nodes)
        
        graph_file = str(self.temp_dir / "test_graph.dgl")
        
        return graph, edge_labels, risk_scores, graph_file
    
    def test_cache_key_generation(self):
        """测试缓存键生成"""
        graph, edge_labels, risk_scores, graph_file = self.create_test_data()
        sampler_config = self.negative_sampler.get_config()
        
        # 生成缓存键
        cache_key1 = self.cache._get_sample_cache_key(
            graph_file, edge_labels, risk_scores, sampler_config
        )
        
        # 相同参数应生成相同的键
        cache_key2 = self.cache._get_sample_cache_key(
            graph_file, edge_labels, risk_scores, sampler_config
        )
        
        assert cache_key1 == cache_key2
        assert len(cache_key1) > 0
        
        # 不同参数应生成不同的键
        different_labels = edge_labels.clone()
        different_labels[0] = 1 - different_labels[0]  # 修改一个标签
        
        cache_key3 = self.cache._get_sample_cache_key(
            graph_file, different_labels, risk_scores, sampler_config
        )
        
        assert cache_key1 != cache_key3
    
    def test_cache_exists_and_path(self):
        """测试缓存存在检查和路径生成"""
        cache_key = "test_key_12345"
        
        # 初始状态不存在
        assert not self.cache._sample_cache_exists(cache_key)
        
        # 获取缓存路径
        cache_path = self.cache._get_sample_cache_path(cache_key)
        assert cache_path.suffix == '.gz'  # 压缩启用
        assert 'samples_' in cache_path.name
    
    def test_sample_generation_and_caching(self):
        """测试采样生成和缓存"""
        graph, edge_labels, risk_scores, graph_file = self.create_test_data()
        
        # 第一次调用应生成新结果
        positive_edges1, negative_edges1 = self.cache.get_or_generate_samples(
            graph, edge_labels, risk_scores, self.negative_sampler, graph_file
        )
        
        # 验证结果格式
        assert torch.is_tensor(positive_edges1)
        assert torch.is_tensor(negative_edges1)
        assert positive_edges1.shape[1] == 2  # [num_edges, 2]
        assert negative_edges1.shape[1] == 2
        
        # 验证采样比例
        if len(positive_edges1) > 0:
            ratio = len(negative_edges1) / len(positive_edges1)
            assert 2.5 <= ratio <= 3.5  # 允许一定误差
        
        # 第二次调用应从缓存读取
        positive_edges2, negative_edges2 = self.cache.get_or_generate_samples(
            graph, edge_labels, risk_scores, self.negative_sampler, graph_file
        )
        
        # 验证缓存一致性
        assert torch.equal(positive_edges1, positive_edges2)
        assert torch.equal(negative_edges1, negative_edges2)
        
        # 验证缓存命中率
        assert self.cache.cache_stats['hits'] >= 1
    
    def test_force_regenerate(self):
        """测试强制重新生成"""
        graph, edge_labels, risk_scores, graph_file = self.create_test_data()
        
        # 第一次生成
        positive_edges1, negative_edges1 = self.cache.get_or_generate_samples(
            graph, edge_labels, risk_scores, self.negative_sampler, graph_file
        )
        
        # 强制重新生成
        positive_edges2, negative_edges2 = self.cache.get_or_generate_samples(
            graph, edge_labels, risk_scores, self.negative_sampler, graph_file,
            force_regenerate=True
        )
        
        # 正样本应该相同（基于相同的edge_labels）
        assert torch.equal(positive_edges1, positive_edges2)
        
        # 负样本可能不同（随机采样）
        # 但数量应该相近
        ratio1 = len(negative_edges1) / max(len(positive_edges1), 1)
        ratio2 = len(negative_edges2) / max(len(positive_edges2), 1)
        assert abs(ratio1 - ratio2) < 0.5
    
    def test_large_graph_detection(self):
        """测试大图检测"""
        # 小图
        small_graph = self.create_test_graph(num_nodes=50, num_edges=100)
        assert not self.cache._detect_large_graph(small_graph)
        
        # 大图（边数超过阈值）
        large_graph = self.create_test_graph(num_nodes=1000, num_edges=150000)
        assert self.cache._detect_large_graph(large_graph)
    
    def test_cache_size_management(self):
        """测试缓存大小管理"""
        # 创建多个不同的测试数据
        for i in range(5):
            graph, edge_labels, risk_scores, graph_file = self.create_test_data()
            graph_file = str(self.temp_dir / f"test_graph_{i}.dgl")
            
            # 修改边标签使其不同
            edge_labels[i % len(edge_labels)] = 1.0
            
            self.cache.get_or_generate_samples(
                graph, edge_labels, risk_scores, self.negative_sampler, graph_file
            )
        
        # 检查缓存统计
        stats = self.cache.get_sample_cache_stats()
        assert stats['cached_files'] >= 5
        assert stats['total_size_mb'] > 0
    
    def test_cache_cleanup(self):
        """测试缓存清理"""
        graph, edge_labels, risk_scores, graph_file = self.create_test_data()
        
        # 生成一些缓存
        self.cache.get_or_generate_samples(
            graph, edge_labels, risk_scores, self.negative_sampler, graph_file
        )
        
        # 验证缓存存在
        stats_before = self.cache.get_sample_cache_stats()
        assert stats_before['cached_files'] > 0
        
        # 清理缓存
        self.cache.clear_sample_cache()
        
        # 验证缓存已清理
        stats_after = self.cache.get_sample_cache_stats()
        assert stats_after['cached_files'] == 0
    
    def test_error_handling(self):
        """测试错误处理"""
        graph, edge_labels, risk_scores, graph_file = self.create_test_data()
        
        # 测试无效边标签长度
        invalid_labels = torch.tensor([1.0, 0.0])  # 长度不匹配
        
        with pytest.raises(Exception):  # 应该抛出错误
            self.cache._generate_and_cache_samples(
                graph, invalid_labels, risk_scores, 
                self.negative_sampler, graph_file, "test_key"
            )
    
    def test_backward_compatibility(self):
        """测试向后兼容性"""
        # 确保可以作为EdgeLabelCache使用
        graph, edge_labels, risk_scores, graph_file = self.create_test_data()
        
        # 使用父类方法
        generated_labels = self.cache.get_or_generate_labels(
            graph, risk_scores, graph_file
        )
        
        assert torch.is_tensor(generated_labels)
        assert len(generated_labels) == graph.number_of_edges()
    
    def test_cache_stats(self):
        """测试缓存统计"""
        graph, edge_labels, risk_scores, graph_file = self.create_test_data()
        
        # 初始统计
        initial_stats = self.cache.get_sample_cache_stats()
        assert initial_stats['cache_hits'] == 0
        assert initial_stats['cache_misses'] == 0
        
        # 生成一次（缓存未命中）
        self.cache.get_or_generate_samples(
            graph, edge_labels, risk_scores, self.negative_sampler, graph_file
        )
        
        # 再次调用（缓存命中）
        self.cache.get_or_generate_samples(
            graph, edge_labels, risk_scores, self.negative_sampler, graph_file
        )
        
        # 检查统计
        final_stats = self.cache.get_sample_cache_stats()
        assert final_stats['cache_hits'] >= 1
        assert final_stats['cache_misses'] >= 1
        assert final_stats['hit_rate'] > 0


def test_integration_with_dataset():
    """集成测试：与RiskNICLPDataset的集成"""
    from src.data.graph_dataset import RiskNICLPDataset
    from src.data import FeatureProcessor, RiskVectorizer
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建虚拟处理器
        feature_processor = FeatureProcessor(
            feature_selection_method='hybrid',
            variance_threshold=0.01,
            correlation_threshold=0.9
        )
        
        risk_vectorizer = RiskVectorizer(
            method='zscore_sigmoid',
            num_bins=8
        )
        
        # 创建启用采样缓存的数据集
        dataset = RiskNICLPDataset(
            graph_dir="data/processed/graphs",  # 使用实际路径
            metadata_dir="data/processed/metadata",
            feature_processor=feature_processor,
            risk_vectorizer=risk_vectorizer,
            enable_edge_label_cache=True,
            enable_sample_cache=True,
            sample_cache_config={
                'max_cache_size_gb': 1.0,
                'compression_enabled': True
            }
        )
        
        # 验证缓存实例类型
        assert hasattr(dataset, 'sample_cache')
        assert isinstance(dataset.sample_cache, PositiveNegativeSampleCache)
        assert dataset.enable_sample_cache == True
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    """运行测试"""
    pytest.main([__file__, "-v"]) 