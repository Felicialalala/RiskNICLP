#!/usr/bin/env python3
"""
RiskNICLP-5G 缓存管理工具

用于管理边标签缓存和正负采样结果缓存
支持清理、统计、预计算等功能
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.edge_label_cache import PositiveNegativeSampleCache
from src.data.negative_sampler import create_negative_sampler
from src.config.config_loader import load_config


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RiskNICLP-5G 缓存管理工具")
    
    # 基本参数
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/baseline_config.yaml',
        help='配置文件路径'
    )
    
    # 操作类型
    parser.add_argument(
        '--action', 
        type=str, 
        choices=['stats', 'clear', 'precompute', 'cleanup', 'validate'],
        required=True,
        help='操作类型'
    )
    
    # 缓存类型
    parser.add_argument(
        '--cache-type',
        type=str,
        choices=['edge_labels', 'samples', 'all'],
        default='all',
        help='缓存类型'
    )
    
    # 清理选项
    parser.add_argument(
        '--pattern',
        type=str,
        help='清理缓存时的模式匹配'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制执行操作'
    )
    
    # 预计算选项
    parser.add_argument(
        '--precompute-workers',
        type=int,
        default=4,
        help='预计算时的并行工作进程数'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='预计算时处理的最大文件数'
    )
    
    # 输出选项
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出结果到文件'
    )
    
    return parser.parse_args()


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, config_path: str, verbose: bool = False):
        """
        初始化缓存管理器
        
        Args:
            config_path: 配置文件路径
            verbose: 是否输出详细信息
        """
        self.config = load_config(config_path)
        self.verbose = verbose
        
        # 初始化缓存系统
        self._init_cache_system()
        
    def _init_cache_system(self):
        """初始化缓存系统"""
        # 获取缓存配置
        cache_dir = Path(self.config.data.graph_dir).parent / "edge_label_cache"
        sample_cache_dir = Path(self.config.data.graph_dir).parent / "sample_cache"
        
        # 从配置获取采样缓存设置
        sample_cache_config = getattr(self.config.data, 'sample_cache_config', {})
        max_cache_size_gb = sample_cache_config.get('max_cache_size_gb', 10.0)
        compression_enabled = sample_cache_config.get('compression_enabled', True)
        
        # 创建增强缓存实例
        self.cache = PositiveNegativeSampleCache(
            cache_dir=str(cache_dir),
            sample_cache_dir=str(sample_cache_dir),
            max_cache_size_gb=max_cache_size_gb,
            compression_enabled=compression_enabled
        )
        
        # 创建负采样器（用于预计算）
        if hasattr(self.config, 'phase2') and 'negative_sampling' in self.config.phase2:
            self.negative_sampler = create_negative_sampler(self.config.phase2['negative_sampling'])
        else:
            # 默认负采样器
            self.negative_sampler = create_negative_sampler({'type': 'baseline', 'neg_ratio': 3})
        
        if self.verbose:
            print(f"📂 边标签缓存目录: {cache_dir}")
            print(f"📂 采样结果缓存目录: {sample_cache_dir}")
            print(f"💾 缓存大小限制: {max_cache_size_gb}GB")
            print(f"🗜️ 压缩存储: {'启用' if compression_enabled else '禁用'}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        edge_stats = self.cache.get_cache_stats()
        sample_stats = self.cache.get_sample_cache_stats()
        
        return {
            'edge_label_cache': edge_stats,
            'sample_cache': sample_stats,
            'total_size_mb': edge_stats['total_size_mb'] + sample_stats['total_size_mb'],
            'total_files': edge_stats['cached_files'] + sample_stats['cached_files']
        }
    
    def print_stats(self, stats: Dict[str, Any]):
        """打印缓存统计信息"""
        print("🔍 缓存统计信息")
        print("=" * 50)
        
        # 边标签缓存统计
        edge_stats = stats['edge_label_cache']
        print(f"📊 边标签缓存:")
        print(f"   文件数量: {edge_stats['cached_files']}")
        print(f"   缓存大小: {edge_stats['total_size_mb']:.2f}MB")
        print(f"   索引条目: {edge_stats['index_entries']}")
        
        # 采样结果缓存统计
        sample_stats = stats['sample_cache']
        print(f"📊 采样结果缓存:")
        print(f"   文件数量: {sample_stats['cached_files']}")
        print(f"   缓存大小: {sample_stats['total_size_mb']:.2f}MB")
        print(f"   最大限制: {sample_stats['max_size_gb']:.1f}GB")
        print(f"   压缩存储: {'启用' if sample_stats['compression_enabled'] else '禁用'}")
        print(f"   索引条目: {sample_stats['index_entries']}")
        
        # 性能统计
        if 'performance' in sample_stats:
            perf = sample_stats['performance']
            print(f"📈 性能统计:")
            print(f"   缓存命中率: {perf['hit_rate']:.1%}")
            print(f"   缓存命中: {perf['cache_hits']}")
            print(f"   缓存未命中: {perf['cache_misses']}")
            if perf['cache_hits'] > 0:
                print(f"   平均加载时间: {perf['avg_load_time']:.3f}s")
            if perf['cache_misses'] > 0:
                print(f"   平均生成时间: {perf['avg_generation_time']:.3f}s")
        
        # 总计
        print(f"📊 总计:")
        print(f"   总文件数: {stats['total_files']}")
        print(f"   总大小: {stats['total_size_mb']:.2f}MB")
    
    def clear_cache(self, cache_type: str = 'all', pattern: Optional[str] = None, force: bool = False):
        """清理缓存"""
        if not force:
            response = input(f"确认清理 {cache_type} 缓存? (y/N): ")
            if response.lower() != 'y':
                print("操作已取消")
                return
        
        print(f"🗑️ 开始清理 {cache_type} 缓存...")
        
        if cache_type in ['edge_labels', 'all']:
            self.cache.clear_cache(pattern)
            print("✅ 边标签缓存已清理")
        
        if cache_type in ['samples', 'all']:
            self.cache.clear_sample_cache(pattern)
            print("✅ 采样结果缓存已清理")
        
        print("🎉 缓存清理完成")
    
    def precompute_cache(self, max_files: Optional[int] = None, workers: int = 4):
        """预计算缓存"""
        print("🚀 开始预计算缓存...")
        
        # 获取图文件列表
        graph_dir = Path(self.config.data.graph_dir)
        graph_files = list(graph_dir.glob("*.dgl"))
        
        if max_files is not None:
            graph_files = graph_files[:max_files]
        
        print(f"📂 找到 {len(graph_files)} 个图文件")
        
        if len(graph_files) == 0:
            print("⚠️ 未找到图文件")
            return
        
        # 定义数据获取函数
        def get_graph_data(graph_file):
            """获取图数据的辅助函数"""
            import dgl
            import torch
            
            # 加载图
            graphs, _ = dgl.load_graphs(str(graph_file))
            graph = graphs[0]
            
            # 生成边标签
            edge_labels = self.cache.get_or_generate_labels(
                graph=graph,
                risk_scores=torch.rand(graph.number_of_nodes()),  # 临时风险分数
                graph_file=str(graph_file)
            )
            
            # 生成风险分数
            risk_scores = torch.rand(graph.number_of_nodes())
            
            return graph, edge_labels, risk_scores
        
        # 预计算采样结果
        start_time = time.time()
        
        try:
            stats = self.cache.precompute_all_samples(
                graph_files=[str(f) for f in graph_files],
                get_graph_data_fn=get_graph_data,
                negative_sampler=self.negative_sampler,
                force_regenerate=False
            )
            
            elapsed_time = time.time() - start_time
            
            print("✅ 预计算完成")
            print(f"⏱️ 耗时: {elapsed_time:.1f}s")
            print(f"📊 处理文件: {stats['generated_files']} 新生成 + {stats['cached_files']} 已缓存")
            print(f"📊 正样本: {stats['total_positive_edges']:,}")
            print(f"📊 负样本: {stats['total_negative_edges']:,}")
            
            if stats['failed_files'] > 0:
                print(f"⚠️ 失败文件: {stats['failed_files']}")
                if self.verbose:
                    for error_file in stats['error_files']:
                        print(f"   ❌ {error_file}")
        
        except Exception as e:
            print(f"❌ 预计算失败: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
    
    def cleanup_old_cache(self, days: int = 30):
        """清理旧缓存文件"""
        print(f"🧹 清理 {days} 天前的缓存文件...")
        
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 3600)
        
        cleaned_count = 0
        cleaned_size = 0
        
        # 清理边标签缓存
        edge_cache_dir = Path(self.cache.cache_dir)
        for cache_file in edge_cache_dir.glob("edge_labels_*.pkl"):
            if cache_file.stat().st_mtime < cutoff_time:
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                cleaned_count += 1
                cleaned_size += file_size
        
        # 清理采样结果缓存
        sample_cache_dir = Path(self.cache.sample_cache_dir)
        for cache_file in sample_cache_dir.glob("samples_*.pkl*"):
            if cache_file.stat().st_mtime < cutoff_time:
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                cleaned_count += 1
                cleaned_size += file_size
        
        print(f"✅ 清理完成: {cleaned_count} 个文件 ({cleaned_size / (1024*1024):.1f}MB)")
    
    def validate_cache(self):
        """验证缓存完整性"""
        print("🔍 验证缓存完整性...")
        
        # 获取统计信息
        stats = self.get_stats()
        
        # 检查磁盘空间
        edge_cache_dir = Path(self.cache.cache_dir)
        sample_cache_dir = Path(self.cache.sample_cache_dir)
        
        issues = []
        
        # 检查目录存在性
        if not edge_cache_dir.exists():
            issues.append(f"边标签缓存目录不存在: {edge_cache_dir}")
        
        if not sample_cache_dir.exists():
            issues.append(f"采样结果缓存目录不存在: {sample_cache_dir}")
        
        # 检查缓存大小
        total_size_gb = stats['total_size_mb'] / 1024
        max_size_gb = stats['sample_cache']['max_size_gb']
        
        if total_size_gb > max_size_gb:
            issues.append(f"缓存大小超限: {total_size_gb:.2f}GB > {max_size_gb:.2f}GB")
        
        # 检查索引一致性
        edge_index_count = stats['edge_label_cache']['index_entries']
        edge_file_count = stats['edge_label_cache']['cached_files']
        
        if edge_index_count != edge_file_count:
            issues.append(f"边标签缓存索引不一致: 索引{edge_index_count} vs 文件{edge_file_count}")
        
        sample_index_count = stats['sample_cache']['index_entries']
        sample_file_count = stats['sample_cache']['cached_files']
        
        if sample_index_count != sample_file_count:
            issues.append(f"采样缓存索引不一致: 索引{sample_index_count} vs 文件{sample_file_count}")
        
        # 输出结果
        if issues:
            print("❌ 发现以下问题:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ 缓存验证通过，未发现问题")
        
        return len(issues) == 0


def main():
    """主函数"""
    args = parse_arguments()
    
    try:
        # 创建缓存管理器
        manager = CacheManager(args.config, args.verbose)
        
        # 执行操作
        if args.action == 'stats':
            stats = manager.get_stats()
            manager.print_stats(stats)
            
            # 输出到文件
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"📄 统计信息已保存到: {args.output}")
        
        elif args.action == 'clear':
            manager.clear_cache(args.cache_type, args.pattern, args.force)
        
        elif args.action == 'precompute':
            manager.precompute_cache(args.max_files, args.precompute_workers)
        
        elif args.action == 'cleanup':
            manager.cleanup_old_cache(30)  # 清理30天前的缓存
        
        elif args.action == 'validate':
            manager.validate_cache()
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 