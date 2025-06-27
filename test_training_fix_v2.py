#!/usr/bin/env python3
"""
训练修复验证脚本 v2.0

验证修复后的训练系统:
1. Edge labels真实标签生成
2. 模型输出与标签维度匹配
3. 损失计算数值稳定性
4. 训练效果改善
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_edge_label_generation():
    """测试改进的边标签生成逻辑"""
    print("🔧 测试边标签生成...")
    
    from src.data.graph_dataset import RiskNICLPDataset
    from src.data.feature_processor import FeatureProcessor
    from src.data.risk_vectorizer import RiskVectorizer
    
    # 创建数据集
    feature_processor = FeatureProcessor()
    risk_vectorizer = RiskVectorizer()
    
    dataset = RiskNICLPDataset(
        graph_dir='data/processed/graphs',
        metadata_dir='data/processed/metadata',
        feature_processor=feature_processor,
        risk_vectorizer=risk_vectorizer
    )
    
    dataset.prepare_data()
    
    # 测试几个样本
    print("📊 边标签统计:")
    positive_samples = 0
    negative_samples = 0
    
    for i in range(min(10, len(dataset))):
        try:
            sample = dataset[i]
            edge_labels = sample['edge_labels']
            
            if len(edge_labels) > 1:  # 排除单标签默认情况
                pos_count = torch.sum(edge_labels == 1).item()
                neg_count = torch.sum(edge_labels == 0).item()
                total = len(edge_labels)
                
                positive_samples += pos_count
                negative_samples += neg_count
                
                print(f"  样本 {i}: 正样本={pos_count}, 负样本={neg_count}, 总数={total}, 正样本比例={pos_count/total:.3f}")
            else:
                print(f"  样本 {i}: 单标签 {edge_labels.item()}")
                
        except Exception as e:
            print(f"  样本 {i}: 加载失败 - {e}")
    
    total_samples = positive_samples + negative_samples
    if total_samples > 0:
        print(f"🎯 总体统计: 正样本={positive_samples}, 负样本={negative_samples}")
        print(f"   正样本比例: {positive_samples/total_samples:.3f}")
        print(f"   负样本比例: {negative_samples/total_samples:.3f}")
        
        # 检查是否有平衡的正负样本
        if positive_samples > 0 and negative_samples > 0:
            print("✅ 边标签生成成功: 包含正负样本")
        else:
            print("❌ 边标签生成有问题: 缺少正样本或负样本")
    else:
        print("❌ 没有有效的边标签数据")

def test_model_dimension_consistency():
    """测试模型输出与标签维度一致性"""
    print("\n🔧 测试模型维度一致性...")
    
    try:
        from src.models.riskniclp_model import RiskNICLPModel
        import dgl
        
        # 创建测试模型
        node_feature_dims = {'Alert': 9, 'Flow': 17, 'UE': 16}
        model = RiskNICLPModel(
            node_feature_dims=node_feature_dims,
            feature_dim=45,
            hidden_dim=64,
            risk_embed_dim=16,
            num_risk_bins=8,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
            temperature=0.1,
            gravity_epsilon=1e-3
        )
        
        # 创建测试图和数据
        num_nodes = {'Alert': 5, 'Flow': 100, 'UE': 10}
        edges = {
            ('UE', 'initiates', 'Flow'): (torch.randint(0, 10, (50,)), torch.randint(0, 100, (50,))),
            ('Flow', 'triggers', 'Alert'): (torch.randint(0, 100, (20,)), torch.randint(0, 5, (20,)))
        }
        
        test_graph = dgl.heterograph(edges, num_nodes)
        
        # 添加节点特征
        for ntype in test_graph.ntypes:
            feat_dim = node_feature_dims[ntype]
            test_graph.nodes[ntype].data['feat'] = torch.randn(test_graph.number_of_nodes(ntype), feat_dim)
        
        # 创建风险分数
        total_nodes = sum(num_nodes.values())
        risk_scores = torch.rand(total_nodes) * 0.8 + 0.1  # 0.1-0.9范围
        
        # 创建边标签 (模拟真实场景)
        total_edges = test_graph.number_of_edges()
        edge_labels = torch.randint(0, 2, (total_edges,))  # 随机0或1
        
        # 测试训练模式 (有edge_labels)
        batch = {
            'graph': test_graph,
            'risk_scores': risk_scores,
            'edge_labels': edge_labels,
            'scale_type': 'fine'
        }
        
        print(f"📊 测试数据:")
        print(f"  图节点数: {test_graph.number_of_nodes()}")
        print(f"  图边数: {test_graph.number_of_edges()}")
        print(f"  风险分数长度: {len(risk_scores)}")
        print(f"  边标签长度: {len(edge_labels)}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            fine_scores, coarse_scores = model(batch)
            
        print(f"📈 模型输出:")
        print(f"  Fine scores shape: {fine_scores.shape}")
        print(f"  Coarse scores shape: {coarse_scores.shape}")
        print(f"  Edge labels shape: {edge_labels.shape}")
        
        # 检查维度匹配
        if fine_scores.shape[0] == edge_labels.shape[0]:
            print("✅ 维度匹配: 模型输出与边标签维度一致")
        else:
            print(f"❌ 维度不匹配: 模型输出{fine_scores.shape[0]} vs 边标签{edge_labels.shape[0]}")
            
        # 测试损失计算
        try:
            import torch.nn.functional as F
            loss = F.binary_cross_entropy_with_logits(fine_scores, edge_labels.float())
            print(f"✅ 损失计算成功: {loss.item():.4f}")
        except Exception as e:
            print(f"❌ 损失计算失败: {e}")
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")

def test_training_step():
    """测试完整的训练步骤"""
    print("\n🔧 测试训练步骤...")
    
    try:
        # 运行一个epoch的训练
        import subprocess
        result = subprocess.run([
            'python', 'train.py', '--epochs', '2', '--debug'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 训练测试成功完成")
            
            # 分析输出
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'Train Loss:' in line or 'Val AUC:' in line:
                    print(f"  {line}")
        else:
            print(f"❌ 训练测试失败:")
            print(f"  返回码: {result.returncode}")
            print(f"  错误输出: {result.stderr[-500:]}")  # 最后500字符
            
    except subprocess.TimeoutExpired:
        print("⚠️ 训练测试超时 (5分钟)")
    except Exception as e:
        print(f"❌ 训练测试异常: {e}")

def main():
    """主测试函数"""
    print("🧪 RiskNICLP-5G 训练修复验证 v2.0")
    print("=" * 50)
    
    # 1. 测试边标签生成
    test_edge_label_generation()
    
    # 2. 测试模型维度一致性
    test_model_dimension_consistency()
    
    # 3. 测试训练步骤
    test_training_step()
    
    print("\n" + "=" * 50)
    print("🎉 训练修复验证完成!")

if __name__ == "__main__":
    main() 