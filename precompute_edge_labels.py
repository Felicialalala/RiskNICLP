#!/usr/bin/env python3
"""
边标签预计算脚本

提前计算并缓存所有图的边标签，避免训练时重复计算
运行方式：python precompute_edge_labels.py [--config config/baseline_config.yaml] [--force]
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import dgl
from tqdm import tqdm

from src.common.config import load_config
from src.data.feature_processor import FeatureProcessor
from src.data.risk_vectorizer import RiskVectorizer
from src.data.edge_label_cache import EdgeLabelCache
from src.common.logger import get_logger


def load_graph_and_risk_scores(graph_file: str) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """
    加载图和风险分数的辅助函数
    
    Args:
        graph_file: 图文件路径
        
    Returns:
        graph: DGL图
        risk_scores: 节点风险分数
    """
    # 加载图
    graphs, labels = dgl.load_graphs(str(graph_file))
    graph = graphs[0]
    
    # 生成风险分数（与数据集中的逻辑保持一致）
    if graph.is_homogeneous:
        if 'risk_score' in graph.ndata and torch.is_tensor(graph.ndata['risk_score']):
            risk_scores = graph.ndata['risk_score']
        else:
            N = graph.number_of_nodes()
            risk_scores = torch.full((N,), 0.3, dtype=torch.float32)
    else:
        # 异质图：按节点类型合并风险分数
        parts = []
        for ntype in graph.ntypes:
            try:
                node_data = graph.nodes[ntype].data
                if 'risk_score' in node_data:
                    risk_tensor = node_data['risk_score']
                    if torch.is_tensor(risk_tensor):
                        parts.append(risk_tensor)
                    else:
                        default_risk = {'Alert': 0.8, 'Flow': 0.3, 'UE': 0.1}.get(ntype, 0.2)
                        num_nodes = graph.number_of_nodes(ntype)
                        parts.append(torch.full((num_nodes,), default_risk, dtype=torch.float32))
                else:
                    default_risk = {'Alert': 0.8, 'Flow': 0.3, 'UE': 0.1}.get(ntype, 0.2)
                    num_nodes = graph.number_of_nodes(ntype)
                    parts.append(torch.full((num_nodes,), default_risk, dtype=torch.float32))
            except Exception as e:
                print(f"⚠️ 处理 {ntype} 节点数据时出错: {e}")
                default_risk = {'Alert': 0.8, 'Flow': 0.3, 'UE': 0.1}.get(ntype, 0.2)
                num_nodes = graph.number_of_nodes(ntype)
                parts.append(torch.full((num_nodes,), default_risk, dtype=torch.float32))
        
        if parts:
            risk_scores = torch.cat(parts, dim=0)
        else:
            risk_scores = torch.tensor([0.3], dtype=torch.float32)
    
    # 确保数据类型正确
    if not torch.is_tensor(risk_scores):
        raise RuntimeError(f"risk_scores必须是tensor，但得到: {type(risk_scores)}")
    
    if risk_scores.dim() != 1:
        raise RuntimeError(f"risk_scores必须是1维tensor，但形状为: {risk_scores.shape}")
    
    if risk_scores.dtype != torch.float32:
        risk_scores = risk_scores.float()
    
    return graph, risk_scores


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="预计算所有图的边标签，加速后续训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/baseline_config.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制重新生成所有缓存（覆盖现有缓存）"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="自定义缓存目录（可选）"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行处理的工作进程数"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    logger = get_logger("EdgeLabelPrecompute").logger
    logger.info("🚀 开始边标签预计算...")
    
    try:
        config = load_config(args.config)
        logger.info(f"📝 加载配置文件: {args.config}")
    except Exception as e:
        logger.error(f"❌ 配置文件加载失败: {e}")
        return 1
    
    # 获取图文件列表
    graph_dir = Path(config.data.graph_dir)
    if not graph_dir.exists():
        logger.error(f"❌ 图文件目录不存在: {graph_dir}")
        return 1
    
    graph_files = list(graph_dir.glob("*.dgl"))
    if not graph_files:
        logger.error(f"❌ 在 {graph_dir} 中未找到 .dgl 文件")
        return 1
    
    graph_files = sorted(graph_files)
    logger.info(f"📂 找到 {len(graph_files)} 个图文件")
    
    # 设置缓存目录
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = graph_dir.parent / "edge_label_cache"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"💾 缓存目录: {cache_dir}")
    
    # 创建边标签缓存管理器
    edge_cache = EdgeLabelCache(
        cache_dir=str(cache_dir),
        semantic_config={
            'triggers': {'pos_ratio': 0.8, 'desc': 'Flow->Alert高风险传播'},
            'initiates': {'pos_ratio': 0.4, 'desc': 'UE->Flow中风险发起'},
            'temporal_adj': {'pos_ratio': 0.15, 'desc': 'Flow->Flow时序关联'},
            'default': {'pos_ratio': 0.1, 'desc': '其他边类型'}
        }
    )
    
    # 显示缓存统计信息
    cache_stats = edge_cache.get_cache_stats()
    logger.info(f"📊 当前缓存状态: {cache_stats['cached_files']} 个文件, "
                f"{cache_stats['total_size_mb']:.1f} MB")
    
    if args.force:
        logger.info("🗑️ 强制模式：清理现有缓存...")
        edge_cache.clear_cache()
    
    # 预计算所有边标签
    start_time = time.time()
    
    try:
        stats = edge_cache.precompute_all_labels(
            graph_files=[str(f) for f in graph_files],
            get_graph_and_risk_fn=load_graph_and_risk_scores,
            force_regenerate=args.force
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 输出详细统计信息
        logger.info("🎉 预计算完成！")
        logger.info(f"⏱️ 总耗时: {duration:.1f} 秒")
        logger.info(f"📁 处理文件: {stats['total_files']} 总计")
        logger.info(f"   🆕 新生成: {stats['generated_files']} 个")
        logger.info(f"   📂 已缓存: {stats['cached_files']} 个")
        
        if stats['failed_files'] > 0:
            logger.warning(f"   ❌ 失败: {stats['failed_files']} 个")
            for error_file in stats['error_files'][:5]:  # 显示前5个错误文件
                logger.warning(f"      {error_file}")
            if len(stats['error_files']) > 5:
                logger.warning(f"      ... 还有 {len(stats['error_files']) - 5} 个文件")
        
        if stats['total_edges'] > 0:
            pos_ratio = stats['total_positive'] / stats['total_edges']
            logger.info(f"📊 边统计:")
            logger.info(f"   📐 总边数: {stats['total_edges']:,}")
            logger.info(f"   ✅ 正样本: {stats['total_positive']:,} ({pos_ratio:.1%})")
            logger.info(f"   ❌ 负样本: {stats['total_edges'] - stats['total_positive']:,} ({1-pos_ratio:.1%})")
        
        # 显示性能提升估计
        if stats['generated_files'] > 0:
            avg_time_per_file = duration / stats['generated_files']
            estimated_savings = avg_time_per_file * stats['total_files']
            logger.info(f"🚀 性能提升:")
            logger.info(f"   每个文件平均耗时: {avg_time_per_file:.3f} 秒")
            logger.info(f"   预计为每次训练节省时间: {estimated_savings:.1f} 秒")
        
        # 最终缓存统计
        final_stats = edge_cache.get_cache_stats()
        logger.info(f"💾 最终缓存: {final_stats['cached_files']} 个文件, "
                    f"{final_stats['total_size_mb']:.1f} MB")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("⚠️ 用户中断，预计算未完成")
        return 1
    except Exception as e:
        logger.error(f"❌ 预计算过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 