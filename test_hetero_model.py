#!/usr/bin/env python3
"""
测试异质图模型架构 - 验证不同节点类型特征维度处理
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import dgl
from typing import Dict, Any


class HeteroNodeProjections(nn.Module):
    """异质图节点类型专属投影层"""
    
    def __init__(self, node_feature_dims: Dict[str, int], target_dim: int = 45):
        super().__init__()
        self.target_dim = target_dim
        
        # 为每种节点类型创建专属投影层
        self.projections = nn.ModuleDict({
            node_type: nn.Linear(input_dim, target_dim)
            for node_type, input_dim in node_feature_dims.items()
        })
        
        print(f"📊 初始化节点投影层:")
        for node_type, input_dim in node_feature_dims.items():
            print(f"  {node_type}: {input_dim} → {target_dim}")
    
    def forward(self, graph: dgl.DGLGraph) -> Dict[str, torch.Tensor]:
        """按节点类型分别投影特征"""
        projected_features = {}
        
        for ntype in graph.ntypes:
            if 'feat' in graph.nodes[ntype].data:
                features = graph.nodes[ntype].data['feat']
                print(f"🔍 处理节点类型 {ntype}: {features.shape}")
                
                if ntype in self.projections:
                    projected = self.projections[ntype](features)
                    projected_features[ntype] = projected
                    print(f"✅ 投影后 {ntype}: {projected.shape}")
                else:
                    print(f"⚠️ 未找到节点类型 {ntype} 的投影层")
        
        return projected_features


def test_single_graph():
    """测试单个图的特征投影"""
    print("🧪 测试1: 单个异质图特征投影")
    
    # 模拟一个异质图
    ue_nodes = 5
    flow_nodes = 10  
    alert_nodes = 3
    
    # 创建异质图结构 - 修复边连接
    graph_data = {
        ('UE', 'connects', 'Flow'): (
            torch.randint(0, ue_nodes, (8,)),      # UE节点随机连接到Flow
            torch.randint(0, flow_nodes, (8,))     # 8条UE-Flow边
        ),
        ('Flow', 'triggers', 'Alert'): (
            torch.randint(0, flow_nodes, (5,)),    # Flow节点随机连接到Alert
            torch.randint(0, alert_nodes, (5,))    # 5条Flow-Alert边
        )
    }
    
    graph = dgl.heterograph(graph_data)
    
    # 设置不同维度的节点特征
    graph.nodes['UE'].data['feat'] = torch.randn(ue_nodes, 16)      # UE: 16维
    graph.nodes['Flow'].data['feat'] = torch.randn(flow_nodes, 17)  # Flow: 17维  
    graph.nodes['Alert'].data['feat'] = torch.randn(alert_nodes, 9) # Alert: 9维
    
    print(f"📊 图结构:")
    print(f"  节点数量: UE={ue_nodes}, Flow={flow_nodes}, Alert={alert_nodes}")
    print(f"  边类型: {graph.etypes}")
    print(f"  边数量: {[graph.number_of_edges(etype) for etype in graph.etypes]}")
    
    # 创建投影层
    node_dims = {'UE': 16, 'Flow': 17, 'Alert': 9}
    projector = HeteroNodeProjections(node_dims, target_dim=45)
    
    # 测试投影
    projected = projector(graph)
    
    print(f"📊 投影结果:")
    for ntype, features in projected.items():
        print(f"  {ntype}: {features.shape}")
    
    # 验证所有特征都投影到目标维度
    target_dim = 45
    assert all(feat.shape[1] == target_dim for feat in projected.values()), \
           f"所有特征应投影到{target_dim}维"
    
    print("✅ 单图测试通过!")
    return projected


def test_batch_processing():
    """测试批处理场景"""
    print("\n🧪 测试2: 批处理异质图")
    
    graphs = []
    
    # 创建多个图
    for i in range(3):
        ue_nodes = 3 + i
        flow_nodes = 8 + i * 2
        alert_nodes = 2 + i
        
        # 修复边连接
        graph_data = {
            ('UE', 'connects', 'Flow'): (
                torch.randint(0, ue_nodes, (5,)),
                torch.randint(0, flow_nodes, (5,))
            ),
            ('Flow', 'triggers', 'Alert'): (
                torch.randint(0, flow_nodes, (3,)),
                torch.randint(0, alert_nodes, (3,))
            )
        }
        
        graph = dgl.heterograph(graph_data)
        graph.nodes['UE'].data['feat'] = torch.randn(ue_nodes, 16)
        graph.nodes['Flow'].data['feat'] = torch.randn(flow_nodes, 17)
        graph.nodes['Alert'].data['feat'] = torch.randn(alert_nodes, 9)
        
        graphs.append(graph)
        print(f"  图{i}: UE={ue_nodes}, Flow={flow_nodes}, Alert={alert_nodes}")
    
    # 批处理
    batched_graph = dgl.batch(graphs)
    
    print(f"📊 批处理后:")
    for ntype in batched_graph.ntypes:
        feat_shape = batched_graph.nodes[ntype].data['feat'].shape
        print(f"  {ntype}: {feat_shape}")
    
    # 测试投影
    node_dims = {'UE': 16, 'Flow': 17, 'Alert': 9}
    projector = HeteroNodeProjections(node_dims, target_dim=45)
    
    projected = projector(batched_graph)
    
    print(f"📊 批处理投影结果:")
    for ntype, features in projected.items():
        print(f"  {ntype}: {features.shape}")
    
    print("✅ 批处理测试通过!")
    return projected


def test_model_integration():
    """测试与模型集成"""
    print("\n🧪 测试3: 模型集成测试")
    
    # 模拟RiskNICLP模型的核心部分
    class TestRiskNICLPModel(nn.Module):
        def __init__(self, node_feature_dims: Dict[str, int], hidden_dim: int = 64):
            super().__init__()
            
            # 节点投影层
            self.node_projections = HeteroNodeProjections(
                node_feature_dims, target_dim=45
            )
            
            # 风险嵌入
            self.risk_embedding = nn.Embedding(8, 16)  # 8-bin风险嵌入
            
            # 简化的共享编码器
            self.feature_encoder = nn.Linear(45 + 16, hidden_dim)  # 45 + 16 (risk)
            
        def forward(self, graph: dgl.DGLGraph, risk_scores: torch.Tensor):
            # 1. 节点特征投影
            node_features = self.node_projections(graph)
            
            # 2. 风险嵌入
            risk_bins = torch.clamp((risk_scores * 8).long(), 0, 7)
            risk_embeds = self.risk_embedding(risk_bins)
            
            # 3. 特征融合（简化版）
            all_embeddings = {}
            total_nodes = sum(graph.number_of_nodes(ntype) for ntype in graph.ntypes)
            
            # 组合节点特征和风险嵌入
            start_idx = 0
            for ntype in graph.ntypes:
                num_nodes = graph.number_of_nodes(ntype)
                
                node_feat = node_features[ntype]  # [num_nodes, 45]
                node_risk = risk_embeds[start_idx:start_idx + num_nodes]  # [num_nodes, 16]
                
                # 拼接特征
                combined = torch.cat([node_feat, node_risk], dim=-1)  # [num_nodes, 61]
                encoded = self.feature_encoder(combined)  # [num_nodes, 64]
                
                all_embeddings[ntype] = encoded
                start_idx += num_nodes
            
            return all_embeddings
    
    # 创建测试图 - 修复边连接
    graph_data = {
        ('UE', 'connects', 'Flow'): (
            torch.randint(0, 5, (6,)),
            torch.randint(0, 8, (6,))
        ),
        ('Flow', 'triggers', 'Alert'): (
            torch.randint(0, 8, (4,)),
            torch.randint(0, 3, (4,))
        )
    }
    graph = dgl.heterograph(graph_data)
    graph.nodes['UE'].data['feat'] = torch.randn(5, 16)
    graph.nodes['Flow'].data['feat'] = torch.randn(8, 17)
    graph.nodes['Alert'].data['feat'] = torch.randn(3, 9)
    
    # 模拟风险分数
    total_nodes = 5 + 8 + 3  # 16个节点
    risk_scores = torch.rand(total_nodes)
    
    # 创建并测试模型
    node_dims = {'UE': 16, 'Flow': 17, 'Alert': 9}
    model = TestRiskNICLPModel(node_dims, hidden_dim=64)
    
    print(f"🔍 输入:")
    print(f"  图节点数: UE=5, Flow=8, Alert=3")
    print(f"  风险分数: {risk_scores.shape}")
    
    # 前向传播
    embeddings = model(graph, risk_scores)
    
    print(f"📊 输出嵌入:")
    for ntype, emb in embeddings.items():
        print(f"  {ntype}: {emb.shape}")
    
    # 验证输出维度
    assert embeddings['UE'].shape == (5, 64), f"UE嵌入维度错误: {embeddings['UE'].shape}"
    assert embeddings['Flow'].shape == (8, 64), f"Flow嵌入维度错误: {embeddings['Flow'].shape}"
    assert embeddings['Alert'].shape == (3, 64), f"Alert嵌入维度错误: {embeddings['Alert'].shape}"
    
    print("✅ 模型集成测试通过!")


def main():
    """运行所有测试"""
    print("🚀 开始异质图模型架构测试")
    print("=" * 50)
    
    try:
        # 测试1: 单图投影
        test_single_graph()
        
        # 测试2: 批处理
        test_batch_processing()
        
        # 测试3: 模型集成
        test_model_integration()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过! 异质图架构ready!")
        print("\n📋 下一步:")
        print("1. 修改 src/models/riskniclp_model.py")
        print("2. 更新配置文件 config/baseline_config.yaml")
        print("3. 测试完整训练流程")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    main() 