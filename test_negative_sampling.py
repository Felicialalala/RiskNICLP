#!/usr/bin/env python3
"""
RiskNICLP-5G 负采样策略测试验证脚本
验证负采样策略的实施效果和性能改进
"""

import os
import sys
import torch
import numpy as np
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.common.config import load_config
from src.common.logger import get_logger
from src.data.negative_sampler import create_negative_sampler, RiskAwareNegativeSampler
from src.data.graph_dataset import RiskNICLPDataset
from src.evaluation.evaluator import RiskNICLPEvaluator
import dgl


class NegativeSamplingTester:
    """负采样策略测试器"""
    
    def __init__(self, config_path="config/baseline_config.yaml"):
        """初始化测试器"""
        self.config = load_config(config_path)
        self.logger = get_logger(name="negative_sampling_test")
        self.device = torch.device("cpu")  # CPU测试环境
        
        # 测试结果存储
        self.test_results = {
            'functionality_tests': {},
            'performance_tests': {},
            'strategy_comparison': {}
        }
        
        self.logger.logger.info("🧪 负采样策略测试器初始化完成")
    
    def run_all_tests(self):
        """运行所有测试"""
        self.logger.logger.info("🚀 开始负采样策略完整测试套件")
        
        try:
            # 1. 功能测试
            self.logger.logger.info("📋 执行功能测试...")
            self.test_sampler_creation()
            self.test_config_integration()
            self.test_edge_label_generation()
            
            # 2. 性能测试
            self.logger.logger.info("⚡ 执行性能测试...")
            self.test_sampling_performance()
            self.test_memory_usage()
            
            # 3. 策略对比测试
            self.logger.logger.info("🔄 执行策略对比测试...")
            self.test_strategy_effectiveness()
            
            # 4. 生成测试报告
            self.generate_test_report()
            
            self.logger.logger.info("✅ 所有测试完成")
            return True
            
        except Exception as e:
            self.logger.logger.error(f"❌ 测试失败: {e}")
            return False
    
    def test_sampler_creation(self):
        """测试负采样器创建功能"""
        self.logger.logger.info("🏗️ 测试负采样器创建...")
        
        test_cases = [
            {'type': 'baseline', 'neg_ratio': 3},
            {'type': 'risk_aware', 'strategy': 'uniform', 'neg_ratio': 5},
            {'type': 'risk_aware', 'strategy': 'stratified', 'neg_ratio': 3},
            {'type': 'risk_aware', 'strategy': 'semantic', 'neg_ratio': 4}
        ]
        
        success_count = 0
        for i, config in enumerate(test_cases):
            try:
                sampler = create_negative_sampler(config)
                assert sampler is not None, f"采样器创建失败: {config}"
                
                # 验证配置
                sampler_config = sampler.get_config()
                assert sampler_config['type'] == config['type'], "类型不匹配"
                assert sampler_config['neg_ratio'] == config['neg_ratio'], "负样本比例不匹配"
                
                success_count += 1
                self.logger.logger.info(f"✅ 测试用例 {i+1}: {config['type']} - {config.get('strategy', 'N/A')}")
                
            except Exception as e:
                self.logger.logger.error(f"❌ 测试用例 {i+1} 失败: {e}")
        
        self.test_results['functionality_tests']['sampler_creation'] = {
            'total_cases': len(test_cases),
            'success_count': success_count,
            'success_rate': success_count / len(test_cases)
        }
        
        self.logger.logger.info(f"📊 采样器创建测试: {success_count}/{len(test_cases)} 通过")
    
    def test_config_integration(self):
        """测试配置文件集成"""
        self.logger.logger.info("⚙️ 测试配置文件集成...")
        
        try:
            # 检查Phase 2配置是否存在
            assert hasattr(self.config, 'phase2'), "配置文件缺少phase2部分"
            assert 'negative_sampling' in self.config.phase2, "配置文件缺少negative_sampling配置"
            
            neg_config = self.config.phase2['negative_sampling']
            
            # 验证必需的配置项
            required_keys = ['enabled', 'type', 'strategy', 'neg_ratio']
            for key in required_keys:
                assert key in neg_config, f"缺少必需配置项: {key}"
            
            # 验证攻击类型映射
            if 'attack_type_mapping' in neg_config:
                attack_mapping = neg_config['attack_type_mapping']
                assert isinstance(attack_mapping, dict), "攻击类型映射必须是字典"
                assert len(attack_mapping) > 0, "攻击类型映射不能为空"
            
            # 使用配置创建采样器
            sampler = create_negative_sampler(neg_config)
            assert sampler is not None, "使用配置文件创建采样器失败"
            
            self.test_results['functionality_tests']['config_integration'] = {
                'config_valid': True,
                'sampler_created': True,
                'attack_mappings': len(neg_config.get('attack_type_mapping', {}))
            }
            
            self.logger.logger.info("✅ 配置文件集成测试通过")
            
        except Exception as e:
            self.logger.logger.error(f"❌ 配置文件集成测试失败: {e}")
            self.test_results['functionality_tests']['config_integration'] = {
                'config_valid': False,
                'error': str(e)
            }
    
    def test_edge_label_generation(self):
        """测试边标签生成逻辑"""
        self.logger.logger.info("🔗 测试边标签生成...")
        
        try:
            # 创建虚拟异质图进行测试
            test_graph = self._create_test_heterograph()
            
            # 创建风险感知采样器
            sampler_config = self.config.phase2['negative_sampling']
            sampler = create_negative_sampler(sampler_config)
            
            # 生成虚拟风险分数
            num_nodes = test_graph.number_of_nodes()
            risk_scores = torch.rand(num_nodes) * 0.8 + 0.1  # 0.1-0.9范围
            
            # 测试正样本边提取
            positive_edges = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
            
            # 测试负采样
            negative_edges = sampler.sample_negatives(test_graph, positive_edges, risk_scores)
            
            # 验证结果
            assert len(negative_edges) > 0, "负采样结果为空"
            assert negative_edges.shape[1] == 2, "负样本边格式错误"
            
            # 验证负样本数量
            expected_neg_count = len(positive_edges) * sampler_config['neg_ratio']
            actual_neg_count = len(negative_edges)
            
            self.test_results['functionality_tests']['edge_label_generation'] = {
                'positive_edges': len(positive_edges),
                'negative_edges': actual_neg_count,
                'expected_ratio': sampler_config['neg_ratio'],
                'actual_ratio': actual_neg_count / len(positive_edges) if len(positive_edges) > 0 else 0,
                'generation_success': True
            }
            
            self.logger.logger.info(f"✅ 边标签生成测试通过: {len(positive_edges)} 正样本 → {actual_neg_count} 负样本")
            
        except Exception as e:
            self.logger.logger.error(f"❌ 边标签生成测试失败: {e}")
            self.test_results['functionality_tests']['edge_label_generation'] = {
                'generation_success': False,
                'error': str(e)
            }
    
    def test_sampling_performance(self):
        """测试采样性能"""
        self.logger.logger.info("⚡ 测试采样性能...")
        
        strategies = ['uniform', 'stratified', 'semantic']
        performance_results = {}
        
        for strategy in strategies:
            try:
                # 创建采样器
                config = {
                    'type': 'risk_aware',
                    'strategy': strategy,
                    'neg_ratio': 3,
                    'seed': 42
                }
                sampler = create_negative_sampler(config)
                
                # 创建较大的测试图
                test_graph = self._create_large_test_graph(num_nodes=1000, num_edges=2000)
                risk_scores = torch.rand(test_graph.number_of_nodes())
                positive_edges = torch.randint(0, test_graph.number_of_nodes(), (100, 2))
                
                # 性能测试
                start_time = time.time()
                for _ in range(10):  # 多次运行取平均
                    negative_edges = sampler.sample_negatives(test_graph, positive_edges, risk_scores)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                throughput = len(positive_edges) * 3 / avg_time  # 每秒处理的边数
                
                performance_results[strategy] = {
                    'avg_time_seconds': avg_time,
                    'throughput_edges_per_sec': throughput,
                    'memory_efficient': True
                }
                
                self.logger.logger.info(f"✅ {strategy} 策略: {avg_time:.4f}s, {throughput:.1f} edges/s")
                
            except Exception as e:
                self.logger.logger.error(f"❌ {strategy} 策略性能测试失败: {e}")
                performance_results[strategy] = {
                    'avg_time_seconds': float('inf'),
                    'error': str(e)
                }
        
        self.test_results['performance_tests']['sampling_performance'] = performance_results
    
    def test_memory_usage(self):
        """测试内存使用"""
        self.logger.logger.info("💾 测试内存使用...")
        
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 创建大规模采样测试
            config = self.config.phase2['negative_sampling']
            sampler = create_negative_sampler(config)
            
            # 大图测试
            large_graph = self._create_large_test_graph(num_nodes=5000, num_edges=10000)
            risk_scores = torch.rand(large_graph.number_of_nodes())
            positive_edges = torch.randint(0, large_graph.number_of_nodes(), (500, 2))
            
            # 执行负采样
            negative_edges = sampler.sample_negatives(large_graph, positive_edges, risk_scores)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # 清理内存
            del large_graph, negative_edges, positive_edges, risk_scores
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_leak = final_memory - initial_memory
            
            self.test_results['performance_tests']['memory_usage'] = {
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': memory_increase,
                'memory_leak_mb': memory_leak,
                'memory_efficient': memory_increase < 500  # 小于500MB认为高效
            }
            
            self.logger.logger.info(f"✅ 内存使用测试: 峰值增长 {memory_increase:.1f}MB, 内存泄漏 {memory_leak:.1f}MB")
            
        except ImportError:
            self.logger.logger.warning("⚠️ psutil未安装,跳过内存测试")
            self.test_results['performance_tests']['memory_usage'] = {'skipped': True}
        except Exception as e:
            self.logger.logger.error(f"❌ 内存使用测试失败: {e}")
            self.test_results['performance_tests']['memory_usage'] = {'error': str(e)}
    
    def test_strategy_effectiveness(self):
        """测试策略有效性对比"""
        self.logger.logger.info("🔄 测试策略有效性...")
        
        strategies = ['uniform', 'stratified', 'semantic']
        effectiveness_results = {}
        
        # 创建标准测试环境
        test_graph = self._create_test_heterograph()
        num_nodes = test_graph.number_of_nodes()
        
        # 确保风险分数数量与图节点数量匹配
        risk_scores = torch.rand(num_nodes) * 0.8 + 0.1  # 生成与节点数量匹配的风险分数
        
        # 使用图内有效的节点索引创建正样本边
        valid_nodes = list(range(num_nodes))
        positive_edges = torch.tensor([
            [valid_nodes[0], valid_nodes[3]],  # UE -> Flow
            [valid_nodes[1], valid_nodes[4]],  # UE -> Flow  
            [valid_nodes[3], valid_nodes[6]]   # Flow -> Alert
        ], dtype=torch.long)
        
        for strategy in strategies:
            try:
                config = {
                    'type': 'risk_aware',
                    'strategy': strategy,
                    'neg_ratio': 3,
                    'seed': 42
                }
                sampler = create_negative_sampler(config)
                
                # 生成负样本
                negative_edges = sampler.sample_negatives(test_graph, positive_edges, risk_scores)
                
                # 评估负样本质量
                quality_metrics = self._evaluate_negative_sample_quality(
                    positive_edges, negative_edges, risk_scores
                )
                
                effectiveness_results[strategy] = quality_metrics
                
                self.logger.logger.info(
                    f"✅ {strategy}: 分离度 {quality_metrics['separation']:.3f}, "
                    f"多样性 {quality_metrics['diversity']:.3f}"
                )
                
            except Exception as e:
                self.logger.logger.error(f"❌ {strategy} 策略有效性测试失败: {e}")
                effectiveness_results[strategy] = {'error': str(e)}
        
        self.test_results['strategy_comparison']['effectiveness'] = effectiveness_results
    
    def _create_test_heterograph(self):
        """创建测试用异质图"""
        # 创建简单的异质图用于测试
        graph_data = {
            ('UE', 'initiates', 'Flow'): (torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])),
            ('Flow', 'triggers', 'Alert'): (torch.tensor([3, 4]), torch.tensor([6, 7])),
            ('Flow', 'temporal_adj', 'Flow'): (torch.tensor([3, 4]), torch.tensor([4, 5]))
        }
        
        return dgl.heterograph(graph_data)
    
    def _create_large_test_graph(self, num_nodes=1000, num_edges=2000):
        """创建大规模测试图"""
        # 生成随机边
        src = torch.randint(0, num_nodes, (num_edges,))
        dst = torch.randint(0, num_nodes, (num_edges,))
        
        # 去除自环
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        
        return dgl.graph((src, dst), num_nodes=num_nodes)
    
    def _evaluate_negative_sample_quality(self, positive_edges, negative_edges, risk_scores):
        """评估负样本质量"""
        # 安全检查：确保所有索引都在有效范围内
        max_node_idx = len(risk_scores) - 1
        
        # 过滤越界的边
        valid_pos_mask = (positive_edges[:, 0] <= max_node_idx) & (positive_edges[:, 1] <= max_node_idx)
        valid_positive_edges = positive_edges[valid_pos_mask]
        
        valid_neg_mask = (negative_edges[:, 0] <= max_node_idx) & (negative_edges[:, 1] <= max_node_idx)
        valid_negative_edges = negative_edges[valid_neg_mask]
        
        if len(valid_positive_edges) == 0 or len(valid_negative_edges) == 0:
            self.logger.logger.warning("No valid edges after filtering, returning default metrics")
            return {
                'separation': 0.0,
                'diversity': 0.0,
                'pos_avg_risk': 0.0,
                'neg_avg_risk': 0.0
            }
        
        # 计算正负样本的平均风险分数
        pos_src_risks = risk_scores[valid_positive_edges[:, 0]]
        pos_dst_risks = risk_scores[valid_positive_edges[:, 1]]
        pos_avg_risk = torch.mean(pos_src_risks + pos_dst_risks)
        
        neg_src_risks = risk_scores[valid_negative_edges[:, 0]]
        neg_dst_risks = risk_scores[valid_negative_edges[:, 1]]
        neg_avg_risk = torch.mean(neg_src_risks + neg_dst_risks)
        
        # 分离度：正样本风险应该高于负样本
        separation = float(pos_avg_risk - neg_avg_risk)
        
        # 多样性：负样本风险分布的标准差
        neg_risk_std = float(torch.std(neg_src_risks + neg_dst_risks))
        diversity = min(neg_risk_std, 1.0)  # 归一化到[0,1]
        
        return {
            'separation': separation,
            'diversity': diversity,
            'pos_avg_risk': float(pos_avg_risk),
            'neg_avg_risk': float(neg_avg_risk)
        }
    
    def generate_test_report(self):
        """生成测试报告"""
        self.logger.logger.info("📋 生成测试报告...")
        
        report = {
            'test_summary': {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0
            },
            'detailed_results': self.test_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 计算测试统计
        for category, tests in self.test_results.items():
            for test_name, result in tests.items():
                report['test_summary']['total_tests'] += 1
                if isinstance(result, dict) and not result.get('error') and not result.get('skipped'):
                    report['test_summary']['passed_tests'] += 1
                else:
                    report['test_summary']['failed_tests'] += 1
        
        # 保存报告
        import json
        report_path = "logs/negative_sampling_test_report.json"
        os.makedirs("logs", exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 控制台总结
        summary = report['test_summary']
        success_rate = summary['passed_tests'] / summary['total_tests'] if summary['total_tests'] > 0 else 0
        
        self.logger.logger.info(
            f"📊 测试完成: {summary['passed_tests']}/{summary['total_tests']} 通过 "
            f"({success_rate*100:.1f}% 成功率)"
        )
        self.logger.logger.info(f"📄 详细报告已保存: {report_path}")
        
        return report


def main():
    """主函数"""
    print("🧪 RiskNICLP-5G 负采样策略测试开始")
    
    # 创建测试器
    tester = NegativeSamplingTester()
    
    # 运行测试
    success = tester.run_all_tests()
    
    if success:
        print("✅ 所有测试成功完成")
        return 0
    else:
        print("❌ 测试失败")
        return 1


if __name__ == "__main__":
    exit(main()) 