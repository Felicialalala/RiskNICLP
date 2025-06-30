#!/usr/bin/env python3
"""
张量连续性单元测试

验证RiskNICLP-5G模型中所有张量操作的连续性，确保DGL图操作兼容性
"""

import torch
import dgl
import numpy as np
import unittest
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.risk_bias_attention import TrueRiskBiasGATConv


class TestTensorContinuity(unittest.TestCase):
    """张量连续性测试类"""
    
    def setUp(self):
        """测试准备"""
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        
        # 创建测试图
        self.num_nodes = 100
        self.num_edges = 200
        self.in_feats = 32
        self.out_feats = 16
        self.num_heads = 4
        
        # 创建DGL图
        src = torch.randint(0, self.num_nodes, (self.num_edges,))
        dst = torch.randint(0, self.num_nodes, (self.num_edges,))
        self.graph = dgl.graph((src, dst), num_nodes=self.num_nodes)
        
        # 创建测试特征和风险分数
        self.features = torch.randn(self.num_nodes, self.in_feats)
        self.risk_scores = torch.rand(self.num_nodes)
        
        # 创建测试模型
        self.model = TrueRiskBiasGATConv(
            self.in_feats, 
            self.out_feats, 
            self.num_heads
        )
        
    def test_model_initialization(self):
        """测试模型初始化"""
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, '_ensure_contiguous'))
        
    def test_ensure_contiguous_method(self):
        """测试_ensure_contiguous方法"""
        # 测试连续张量
        continuous_tensor = torch.randn(10, 5)
        result = self.model._ensure_contiguous(continuous_tensor, "test")
        self.assertTrue(result.is_contiguous())
        self.assertTrue(torch.equal(result, continuous_tensor))
        
        # 测试非连续张量
        non_continuous_tensor = torch.randn(10, 5).transpose(0, 1)
        self.assertFalse(non_continuous_tensor.is_contiguous())
        
        result = self.model._ensure_contiguous(non_continuous_tensor, "test")
        self.assertTrue(result.is_contiguous())
        self.assertTrue(torch.equal(result, non_continuous_tensor.contiguous()))
        
    def test_forward_pass_continuity(self):
        """测试前向传播中的张量连续性"""
        # 执行前向传播
        output = self.model(self.graph, self.features, self.risk_scores)
        
        # 验证输出张量连续性
        self.assertTrue(output.is_contiguous(), "输出张量必须是连续的")
        self.assertEqual(output.shape, (self.num_nodes, self.out_feats))
        
    def test_no_dgl_errors(self):
        """测试是否产生DGL连续性错误"""
        try:
            # 多次前向传播，确保稳定性
            for _ in range(5):
                output = self.model(self.graph, self.features, self.risk_scores)
                self.assertTrue(output.is_contiguous())
        except Exception as e:
            if "Cannot convert non-contiguous tensors to dgl ndarray" in str(e):
                self.fail(f"DGL连续性错误未修复: {e}")
            else:
                # 其他错误重新抛出
                raise
                
    def test_empty_graph_handling(self):
        """测试空图处理"""
        empty_graph = dgl.graph(([], []), num_nodes=10)
        empty_features = torch.randn(10, self.in_feats)
        empty_risk_scores = torch.rand(10)
        
        output = self.model(empty_graph, empty_features, empty_risk_scores)
        self.assertTrue(output.is_contiguous())
        self.assertEqual(output.shape, (10, self.out_feats))
        
    def test_large_graph_memory_safety(self):
        """测试大图内存安全性"""
        # 创建较大的测试图
        large_num_nodes = 1000
        large_num_edges = 2000
        
        src = torch.randint(0, large_num_nodes, (large_num_edges,))
        dst = torch.randint(0, large_num_nodes, (large_num_edges,))
        large_graph = dgl.graph((src, dst), num_nodes=large_num_nodes)
        
        large_features = torch.randn(large_num_nodes, self.in_feats)
        large_risk_scores = torch.rand(large_num_nodes)
        
        # 验证不会产生内存错误
        output = self.model(large_graph, large_features, large_risk_scores)
        self.assertTrue(output.is_contiguous())
        self.assertEqual(output.shape, (large_num_nodes, self.out_feats))
        
    def test_gradient_flow_continuity(self):
        """测试梯度流的连续性"""
        # 启用梯度计算
        self.features.requires_grad_(True)
        self.risk_scores.requires_grad_(True)
        
        output = self.model(self.graph, self.features, self.risk_scores)
        loss = output.sum()
        
        # 反向传播
        loss.backward()
        
        # 验证梯度张量连续性
        if self.features.grad is not None:
            self.assertTrue(self.features.grad.is_contiguous())
        if self.risk_scores.grad is not None:
            self.assertTrue(self.risk_scores.grad.is_contiguous())


def run_continuity_tests():
    """运行张量连续性测试"""
    print("🧪 运行张量连续性测试...")
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTensorContinuity)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 报告结果
    if result.wasSuccessful():
        print("✅ 所有张量连续性测试通过！")
        return True
    else:
        print("❌ 张量连续性测试失败！")
        for failure in result.failures:
            print(f"失败: {failure[0]} - {failure[1]}")
        for error in result.errors:
            print(f"错误: {error[0]} - {error[1]}")
        return False


if __name__ == "__main__":
    success = run_continuity_tests()
    sys.exit(0 if success else 1) 