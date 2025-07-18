#!/usr/bin/env python3
"""
RiskNICLP-5G 采样质量测试
测试风险权重重构和语义感知负采样的效果

用法:
    python -m tests.test_sampling_quality --graph_path data/processed/graphs/hetero_graph_fine_t1663200001.dgl
"""

import sys
import os
import argparse
import torch
import dgl
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入重构模块
from src.data.risk_weight_recalibration import RiskWeightRecalibrator, RiskRecalibrationConfig
from src.data.semantic_negative_sampler import SemanticNegativeSampler, SemanticSamplingConfig
from src.data.sampling_integration import SamplingIntegrator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SamplingQualityTester:
    """采样质量测试器"""
    
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(__name__)
        
        # 测试配置
        self.test_config = {
            'graph_path': args.graph_path,
            'output_dir': args.output_dir or 'results/sampling_quality'
        }
        
        # 确保输出目录存在
        Path(self.test_config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    def run_quality_test(self):
        """运行质量测试"""
        self.logger.info("🚀 开始采样质量测试")
        self.logger.info("=" * 60)
        
        # 加载测试图
        graph = self._load_test_graph()
        
        if graph is None:
            self.logger.error("❌ 未能加载测试图")
            return
        
        # 测试结果存储
        test_results = {
            'original_quality': {},
            'reconstructed_quality': {},
            'improvement_summary': {}
        }
        
        # 原始质量评估
        original_quality = self._evaluate_original_sampling(graph)
        test_results['original_quality'] = original_quality
        
        # 重构测试
        reconstructed_quality = self._test_reconstruction_pipeline(graph)
        test_results['reconstructed_quality'] = reconstructed_quality
        
        # 计算改善程度
        improvement = self._calculate_improvement(original_quality, reconstructed_quality)
        test_results['improvement_summary'] = improvement
        
        # 生成总体报告
        self._save_test_results(test_results)
        
        self.logger.info("🎉 采样质量测试完成!")
        return test_results
    
    def _load_test_graph(self) -> dgl.DGLGraph:
        """加载测试图"""
        graph_path = self.test_config['graph_path']
        
        if not Path(graph_path).exists():
            self.logger.error(f"图文件不存在: {graph_path}")
            return None
        
        try:
            graph = dgl.load_graphs(graph_path)[0][0]
            self.logger.info(f"✅ 加载图: {Path(graph_path).name}")
            return graph
        except Exception as e:
            self.logger.error(f"❌ 加载图失败: {e}")
            return None
    
    def _evaluate_original_sampling(self, graph: dgl.DGLGraph) -> Dict:
        """评估原始采样质量"""
        self.logger.info("🔍 评估原始采样质量...")
        
        # 提取正样本
        positive_edges = self._extract_positive_edges(graph)
        
        # 使用原始方法生成负样本
        original_negative_edges = self._generate_original_negatives(graph, positive_edges)
        
        # 计算质量指标
        quality_metrics = self._calculate_quality_metrics(
            graph, positive_edges, original_negative_edges, "original"
        )
        
        return quality_metrics
    
    def _test_reconstruction_pipeline(self, graph: dgl.DGLGraph) -> Dict:
        """测试重构流水线"""
        self.logger.info("🔧 测试重构流水线...")
        
        # 创建采样集成器
        config = {
            'type': 'semantic_aware',
            'neg_ratio': 1.2,
            'difficulty_distribution': {
                'easy': 0.7,
                'medium': 0.25,
                'hard': 0.05
            },
            'risk_recalibration': {
                'enabled': True,
                'flow_decay_factor': 0.5,
                'alert_boost_factor': 2.0,
                'ue_baseline': 0.2,
                'enable_type_wise_normalization': True
            },
            'semantic_constraints': {
                'temporal_constraint_enabled': True,
                'spatial_constraint_enabled': True,
                'min_temporal_gap': 5.0
            },
            'max_attempts_multiplier': 15,
            'quality_threshold': 0.8,
            'seed': 42
        }
        
        integrator = SamplingIntegrator(config)
        
        # 提取正样本
        positive_edges = self._extract_positive_edges(graph)
        
        # 应用重构流水线
        reconstructed_negative_edges = integrator.sample_negatives(graph, positive_edges)
        
        # 获取处理后的图（用于质量评估）
        processed_graph = integrator.process_graph(graph)
        
        # 计算重构后质量指标
        quality_metrics = self._calculate_quality_metrics(
            processed_graph, positive_edges, reconstructed_negative_edges, "reconstructed"
        )
        
        # 添加重构统计
        quality_metrics['sampling_stats'] = integrator.get_sampling_stats()
        
        return quality_metrics
    
    def _extract_positive_edges(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """提取正样本边"""
        all_edges = []
        
        for etype in graph.etypes:
            src, dst = graph.edges(etype=etype)
            if len(src) > 0:
                edges = torch.stack([src, dst], dim=1)
                all_edges.append(edges)
        
        if all_edges:
            return torch.cat(all_edges, dim=0)
        else:
            return torch.empty((0, 2), dtype=torch.long)
    
    def _generate_original_negatives(self, graph: dgl.DGLGraph, positive_edges: torch.Tensor) -> torch.Tensor:
        """使用原始方法生成负样本（模拟）"""
        num_pos = len(positive_edges)
        num_neg = num_pos * 4  # 原始4:1比例
        
        # 简化的原始负采样：60%困难样本
        negative_edges = []
        max_attempts = num_neg * 10
        attempts = 0
        
        # 构建正样本集合
        pos_set = set()
        for edge in positive_edges:
            pos_set.add((edge[0].item(), edge[1].item()))
        
        # 获取高风险节点（模拟原始的困难采样）
        high_risk_nodes = self._get_high_risk_nodes_simple(graph)
        all_nodes = list(range(graph.number_of_nodes()))
        
        while len(negative_edges) < num_neg and attempts < max_attempts:
            if len(negative_edges) < num_neg * 0.6:  # 60%困难样本
                if len(high_risk_nodes) >= 2:
                    src = np.random.choice(high_risk_nodes)
                    dst = np.random.choice(high_risk_nodes)
                else:
                    src = np.random.choice(all_nodes)
                    dst = np.random.choice(all_nodes)
            else:  # 40%简单样本
                src = np.random.choice(all_nodes)
                dst = np.random.choice(all_nodes)
            
            if src != dst and (src, dst) not in pos_set:
                negative_edges.append([src, dst])
            
            attempts += 1
        
        return torch.tensor(negative_edges, dtype=torch.long)
    
    def _get_high_risk_nodes_simple(self, graph: dgl.DGLGraph) -> List[int]:
        """简单获取高风险节点"""
        high_risk_nodes = []
        node_offset = 0
        
        for ntype in graph.ntypes:
            if 'risk_score' in graph.nodes[ntype].data:
                risks = graph.nodes[ntype].data['risk_score']
                high_risk_mask = risks > 0.7  # 原始高阈值
                high_risk_indices = torch.where(high_risk_mask)[0]
                
                global_indices = [idx.item() + node_offset for idx in high_risk_indices]
                high_risk_nodes.extend(global_indices)
            
            node_offset += graph.number_of_nodes(ntype)
        
        return high_risk_nodes
    
    def _calculate_quality_metrics(self, graph: dgl.DGLGraph, positive_edges: torch.Tensor,
                                  negative_edges: torch.Tensor, method_name: str) -> Dict:
        """计算质量指标"""
        metrics = {
            'method': method_name,
            'num_positive': len(positive_edges),
            'num_negative': len(negative_edges),
            'negative_ratio': len(negative_edges) / max(len(positive_edges), 1)
        }
        
        if len(positive_edges) > 0 and len(negative_edges) > 0:
            # 计算风险分数统计
            pos_risks = self._calculate_edge_risks(graph, positive_edges)
            neg_risks = self._calculate_edge_risks(graph, negative_edges)
            
            metrics.update({
                'positive_risk_mean': float(pos_risks.mean()),
                'negative_risk_mean': float(neg_risks.mean()),
                'positive_risk_std': float(pos_risks.std()),
                'negative_risk_std': float(neg_risks.std()),
                'risk_separation': float(pos_risks.mean() - neg_risks.mean()),
                'effect_size': float((pos_risks.mean() - neg_risks.mean()) / 
                                   torch.sqrt((pos_risks.var() + neg_risks.var()) / 2))
            })
            
            # 计算质量评分
            quality_score = self._calculate_quality_score(metrics)
            metrics['quality_score'] = quality_score
        
        return metrics
    
    def _calculate_edge_risks(self, graph: dgl.DGLGraph, edges: torch.Tensor) -> torch.Tensor:
        """计算边的风险分数"""
        edge_risks = []
        node_offset = 0
        
        # 构建全局风险分数映射
        global_risk_scores = {}
        for ntype in graph.ntypes:
            if 'risk_score' in graph.nodes[ntype].data:
                risks = graph.nodes[ntype].data['risk_score']
                for i, risk in enumerate(risks):
                    global_risk_scores[node_offset + i] = risk.item()
            node_offset += graph.number_of_nodes(ntype)
        
        # 计算每条边的平均风险
        for edge in edges:
            src, dst = edge[0].item(), edge[1].item()
            src_risk = global_risk_scores.get(src, 0.0)
            dst_risk = global_risk_scores.get(dst, 0.0)
            edge_risk = (src_risk + dst_risk) / 2
            edge_risks.append(edge_risk)
        
        return torch.tensor(edge_risks)
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """计算质量评分"""
        score = 0
        
        # 风险分离度评分 (40分)
        risk_separation = metrics.get('risk_separation', 0)
        if risk_separation > 0.2:
            score += 40
        elif risk_separation > 0.1:
            score += 30
        elif risk_separation > 0:
            score += 20
        else:
            score += 10
        
        # 效应量评分 (30分)
        effect_size = metrics.get('effect_size', 0)
        if effect_size > 0.8:
            score += 30
        elif effect_size > 0.5:
            score += 25
        elif effect_size > 0.2:
            score += 15
        else:
            score += 5
        
        # 比例合理性评分 (30分)
        neg_ratio = metrics.get('negative_ratio', 0)
        if 1.5 <= neg_ratio <= 3.0:
            score += 30
        elif 1.0 <= neg_ratio <= 4.0:
            score += 20
        else:
            score += 10
        
        return score
    
    def _calculate_improvement(self, original: Dict, reconstructed: Dict) -> Dict:
        """计算改善程度"""
        improvement = {
            'quality_score_improvement': (reconstructed.get('quality_score', 0) - 
                                        original.get('quality_score', 0)),
            'risk_separation_improvement': (reconstructed.get('risk_separation', 0) - 
                                          original.get('risk_separation', 0)),
            'effect_size_improvement': (reconstructed.get('effect_size', 0) - 
                                      original.get('effect_size', 0))
        }
        
        # 计算相对改善百分比
        if original.get('quality_score', 0) > 0:
            improvement['relative_improvement'] = (
                improvement['quality_score_improvement'] / original['quality_score'] * 100
            )
        else:
            improvement['relative_improvement'] = 0
        
        return improvement
    
    def _save_test_results(self, test_results: Dict):
        """保存测试结果"""
        output_dir = Path(self.test_config['output_dir'])
        
        # 保存详细结果
        with open(output_dir / 'sampling_quality_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        # 生成可读报告
        self._generate_readable_report(test_results, output_dir)
        
        self.logger.info(f"📄 测试结果已保存到: {output_dir}")
    
    def _generate_readable_report(self, test_results: Dict, output_dir: Path):
        """生成可读的报告"""
        report_content = []
        report_content.append("# RiskNICLP-5G 采样质量测试报告\n")
        report_content.append(f"测试时间: {np.datetime64('now')}\n")
        report_content.append("=" * 60 + "\n")
        
        # 测试图信息
        report_content.append(f"## 📊 测试图信息\n")
        report_content.append(f"- 图文件: {self.test_config['graph_path']}\n\n")
        
        # 质量对比
        original = test_results['original_quality']
        reconstructed = test_results['reconstructed_quality']
        improvement = test_results['improvement_summary']
        
        report_content.append("## 📈 质量对比\n")
        report_content.append("| 指标 | 原始方法 | 重构方法 | 改善程度 |\n")
        report_content.append("|------|----------|----------|----------|\n")
        report_content.append(f"| **质量评分** | {original.get('quality_score', 0)}/100 | **{reconstructed.get('quality_score', 0)}/100** | **+{improvement.get('quality_score_improvement', 0)}分** |\n")
        report_content.append(f"| **正样本风险** | {original.get('positive_risk_mean', 0):.3f} | {reconstructed.get('positive_risk_mean', 0):.3f} | {reconstructed.get('positive_risk_mean', 0) - original.get('positive_risk_mean', 0):.3f} |\n")
        report_content.append(f"| **负样本风险** | {original.get('negative_risk_mean', 0):.3f} | {reconstructed.get('negative_risk_mean', 0):.3f} | {reconstructed.get('negative_risk_mean', 0) - original.get('negative_risk_mean', 0):.3f} |\n")
        report_content.append(f"| **风险分离度** | {original.get('risk_separation', 0):.3f} | **{reconstructed.get('risk_separation', 0):.3f}** | **+{improvement.get('risk_separation_improvement', 0):.3f}** |\n")
        report_content.append(f"| **效应量** | {original.get('effect_size', 0):.3f} | **{reconstructed.get('effect_size', 0):.3f}** | **+{improvement.get('effect_size_improvement', 0):.3f}** |\n")
        report_content.append(f"| **负采样比例** | {original.get('negative_ratio', 0):.1f}:1 | **{reconstructed.get('negative_ratio', 0):.1f}:1** | **-{(original.get('negative_ratio', 0) - reconstructed.get('negative_ratio', 0)) / original.get('negative_ratio', 1) * 100:.1f}%** |\n\n")
        
        # 结论
        report_content.append("## 🎯 结论\n")
        if improvement.get('quality_score_improvement', 0) > 0:
            if reconstructed.get('quality_score', 0) >= 80:
                report_content.append("🎉 **重构成功！** 达到80分以上目标\n")
            else:
                report_content.append("✅ **重构有效！** 质量有明显改善\n")
        else:
            report_content.append("⚠️ **重构效果不明显**，需要进一步调优\n")
        
        # 保存报告
        with open(output_dir / 'sampling_quality_test_report.md', 'w', encoding='utf-8') as f:
            f.writelines(report_content)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RiskNICLP-5G 采样质量测试')
    parser.add_argument('--graph_path', type=str, required=True, help='测试图文件路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    tester = SamplingQualityTester(args)
    test_results = tester.run_quality_test()
    
    # 打印关键结果
    if test_results:
        original = test_results['original_quality']
        reconstructed = test_results['reconstructed_quality']
        improvement = test_results['improvement_summary']
        
        print("\n🎯 关键结果:")
        print(f"质量评分: {original.get('quality_score', 0)} → {reconstructed.get('quality_score', 0)} (+{improvement.get('quality_score_improvement', 0)}分)")
        print(f"风险分离度: {original.get('risk_separation', 0):.3f} → {reconstructed.get('risk_separation', 0):.3f} (+{improvement.get('risk_separation_improvement', 0):.3f})")
        print(f"效应量: {original.get('effect_size', 0):.3f} → {reconstructed.get('effect_size', 0):.3f} (+{improvement.get('effect_size_improvement', 0):.3f})")


if __name__ == "__main__":
    main()
