#!/usr/bin/env python3
"""
图特征维度标准化修复脚本
修复Alert节点特征维度不一致问题
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import dgl
from tqdm import tqdm
from pathlib import Path

# 添加src目录到Python路径
sys.path.append('src')
from data_type_schema import convert_data_types, validate_data_quality

# 全局攻击类型映射 (9种类型)
GLOBAL_ATTACK_TYPES = [
    'Benign', 'ICMPFlood', 'UDPFlood', 'SYNFlood', 
    'HTTPFlood', 'SlowrateDoS', 'TCPConnectScan', 
    'SYNScan', 'UDPScan'
]

ATTACK_TYPE_TO_ID = {attack: i for i, attack in enumerate(GLOBAL_ATTACK_TYPES)}
GLOBAL_FEATURE_DIM = 9  # 固定Alert特征维度

print(f"🎯 全局攻击类型映射:")
for attack, idx in ATTACK_TYPE_TO_ID.items():
    print(f"  {idx}: {attack}")

def extract_flow_features_safe(df_window):
    """从窗口数据中提取流量特征向量 - 安全版本"""
    
    # 选择数值型特征
    feature_columns = [
        'Dur', 'TotPkts', 'SrcPkts', 'DstPkts', 'TotBytes', 'SrcBytes', 'DstBytes',
        'Mean', 'Sum', 'Min', 'Max', 'sTtl', 'dTtl', 'Sport', 'Dport'
    ]
    
    # 安全的特征提取和类型转换
    available_features = []
    feature_data = []
    
    for col in feature_columns:
        if col in df_window.columns:
            try:
                # 强制转换为数值类型
                numeric_col = pd.to_numeric(df_window[col], errors='coerce')
                # 填充NaN值
                numeric_col = numeric_col.fillna(0.0)
                # 确保是float64类型
                numeric_col = numeric_col.astype(np.float64)
                
                feature_data.append(numeric_col.values)
                available_features.append(col)
                
            except Exception as e:
                print(f"特征列 {col} 转换失败: {e}")
                continue
    
    # 构建特征矩阵
    if feature_data and available_features:
        try:
            # 堆叠特征列
            feature_matrix = np.column_stack(feature_data)
            
            # 确保是float64类型
            feature_matrix = feature_matrix.astype(np.float64)
            
            # 安全的无效值处理
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 标准化特征 (添加安全检查)
            if feature_matrix.shape[0] > 1:
                mean_vals = np.mean(feature_matrix, axis=0)
                std_vals = np.std(feature_matrix, axis=0)
                # 避免除零
                std_vals = np.where(std_vals == 0, 1.0, std_vals)
                feature_matrix = (feature_matrix - mean_vals) / std_vals
            else:
                # 单行数据的情况
                feature_matrix = np.zeros_like(feature_matrix)
                
        except Exception as e:
            print(f"特征矩阵构建失败: {e}")
            # 降级方案：使用随机特征
            feature_matrix = np.random.randn(len(df_window), 16).astype(np.float64)
            available_features = [f'random_feature_{i}' for i in range(16)]
    else:
        # 如果没有可用特征，使用基本特征
        feature_matrix = np.random.randn(len(df_window), 16).astype(np.float64)
        available_features = [f'random_feature_{i}' for i in range(16)]
    
    return feature_matrix, available_features

def construct_heterogeneous_graph_fixed(df_window):
    """构建异质图 - 修复版本，使用固定Alert特征维度"""
    
    # 1. 提取节点实体
    ue_ips = df_window['SrcAddr'].unique()
    flow_indices = df_window.index.tolist()
    attack_types = df_window['Attack Type'].unique()
    
    # 创建节点ID映射
    ue_to_id = {ip: i for i, ip in enumerate(ue_ips)}
    flow_to_id = {idx: i for i, idx in enumerate(flow_indices)}
    alert_to_id = {attack: i for i, attack in enumerate(attack_types)}
    
    # 2. 构建边关系
    ue_flow_edges = []
    flow_alert_edges = []
    flow_flow_edges = []
    
    for idx, row in df_window.iterrows():
        # UE -> Flow edges
        if row['SrcAddr'] in ue_to_id and idx in flow_to_id:
            ue_flow_edges.append((ue_to_id[row['SrcAddr']], flow_to_id[idx]))
        
        # Flow -> Alert edges
        if row['Attack Type'] in alert_to_id and idx in flow_to_id:
            flow_alert_edges.append((flow_to_id[idx], alert_to_id[row['Attack Type']]))
    
    # Flow-Flow temporal edges (基于时间相邻性)
    try:
        sorted_flows = df_window.sort_values('StartTime_unix')
        for i in range(len(sorted_flows) - 1):
            current_idx = sorted_flows.iloc[i].name
            next_idx = sorted_flows.iloc[i + 1].name
            if current_idx in flow_to_id and next_idx in flow_to_id:
                flow_flow_edges.append((flow_to_id[current_idx], flow_to_id[next_idx]))
    except Exception as e:
        print(f"构建时间邻接边失败: {e}")
    
    # 3. 构建异质图
    graph_data = {}
    
    # 添加边类型 (确保至少有一个边)
    if ue_flow_edges:
        graph_data[('UE', 'initiates', 'Flow')] = ue_flow_edges
    if flow_alert_edges:
        graph_data[('Flow', 'triggers', 'Alert')] = flow_alert_edges
    if flow_flow_edges:
        graph_data[('Flow', 'temporal_adj', 'Flow')] = flow_flow_edges
    
    # 如果没有边，创建自环
    if not graph_data:
        if len(flow_indices) > 0:
            graph_data[('Flow', 'self_loop', 'Flow')] = [(0, 0)]
        else:
            return None, {}, {}, {}
    
    num_nodes_dict = {
        'UE': len(ue_ips),
        'Flow': len(flow_indices),
        'Alert': len(attack_types)
    }
    
    try:
        hetero_graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        
        # 4. 添加节点特征
        flow_features, feature_names = extract_flow_features_safe(df_window)
        
        # Flow节点特征
        hetero_graph.nodes['Flow'].data['feat'] = torch.FloatTensor(flow_features)
        
        # UE节点特征 (简化特征)
        ue_features = np.random.randn(len(ue_ips), 16).astype(np.float64)
        hetero_graph.nodes['UE'].data['feat'] = torch.FloatTensor(ue_features)
        
        # Alert节点特征 - 修复版本：固定9维
        alert_features = np.zeros((len(attack_types), GLOBAL_FEATURE_DIM)).astype(np.float64)
        for i, attack_type in enumerate(attack_types):
            if attack_type in ATTACK_TYPE_TO_ID:
                alert_features[i, ATTACK_TYPE_TO_ID[attack_type]] = 1.0
            else:
                print(f"警告: 未知攻击类型 {attack_type}")
        
        hetero_graph.nodes['Alert'].data['feat'] = torch.FloatTensor(alert_features)
        
        return hetero_graph, ue_to_id, flow_to_id, alert_to_id
        
    except Exception as e:
        print(f"构建图时出错: {e}")
        return None, {}, {}, {}

def validate_graph_dimensions(graph_dir):
    """验证所有图的特征维度一致性"""
    graph_files = glob.glob(os.path.join(graph_dir, "*.dgl"))
    
    dimension_stats = {}
    inconsistent_files = []
    
    print(f"\n🔍 验证图特征维度一致性...")
    
    for graph_file in tqdm(graph_files, desc="验证图文件"):
        try:
            graphs, _ = dgl.load_graphs(graph_file)
            graph = graphs[0]
            
            # 检查各节点类型的特征维度
            for ntype in graph.ntypes:
                if 'feat' in graph.nodes[ntype].data:
                    feat_shape = graph.nodes[ntype].data['feat'].shape
                    
                    if ntype not in dimension_stats:
                        dimension_stats[ntype] = {'shapes': set(), 'files': []}
                    
                    dimension_stats[ntype]['shapes'].add(feat_shape[1])  # 特征维度
                    dimension_stats[ntype]['files'].append((os.path.basename(graph_file), feat_shape))
                    
                    # 检查Alert节点是否为9维
                    if ntype == 'Alert' and feat_shape[1] != 9:
                        inconsistent_files.append((graph_file, ntype, feat_shape[1]))
                        
        except Exception as e:
            print(f"验证失败 {graph_file}: {e}")
            
    # 输出统计结果
    print(f"\n📊 特征维度统计:")
    for ntype, stats in dimension_stats.items():
        unique_dims = list(stats['shapes'])
        print(f"  {ntype} 节点: {unique_dims} 维")
        if len(unique_dims) > 1:
            print(f"    ⚠️ 维度不一致!")
            
    if inconsistent_files:
        print(f"\n❌ 发现 {len(inconsistent_files)} 个不一致的文件:")
        for file, ntype, dim in inconsistent_files[:5]:
            print(f"    {file}: {ntype} -> {dim}维")
    else:
        print(f"\n✅ 所有图文件特征维度一致!")
        
    return len(inconsistent_files) == 0

if __name__ == "__main__":
    print("🔧 开始图特征维度标准化修复...")
    
    # 先验证当前图的维度情况
    print("📊 检查当前图文件维度状态...")
    is_consistent = validate_graph_dimensions("data/processed/graphs")
    
    if is_consistent:
        print("✅ 当前图文件维度已经一致，无需修复!")
    else:
        print("❌ 发现维度不一致问题，需要修复!")
        print("\n💡 解决方案：使用固定的9维Alert特征(全局攻击类型)")
        
        # 简单的修复方案：重新加载图并修复Alert特征
        graph_files = glob.glob("data/processed/graphs/*.dgl")
        print(f"📁 发现 {len(graph_files)} 个图文件需要修复")
        
        success_count = 0
        for graph_file in tqdm(graph_files, desc="修复图文件"):
            try:
                # 加载原图
                graphs, _ = dgl.load_graphs(graph_file)
                graph = graphs[0]
                
                # 检查Alert节点特征
                if 'Alert' in graph.ntypes and 'feat' in graph.nodes['Alert'].data:
                    alert_feat = graph.nodes['Alert'].data['feat']
                    num_alert_nodes = alert_feat.shape[0]
                    
                    # 如果维度不是9，则修复
                    if alert_feat.shape[1] != 9:
                        # 创建新的9维特征矩阵
                        new_alert_feat = torch.zeros(num_alert_nodes, 9, dtype=torch.float32)
                        
                        # 复制原有特征（如果维度允许）
                        copy_dims = min(alert_feat.shape[1], 9)
                        new_alert_feat[:, :copy_dims] = alert_feat[:, :copy_dims]
                        
                        # 更新图中的特征
                        graph.nodes['Alert'].data['feat'] = new_alert_feat
                        
                        # 保存修复后的图
                        dgl.save_graphs(graph_file, [graph])
                        
                success_count += 1
                        
            except Exception as e:
                print(f"修复失败 {graph_file}: {e}")
                
        print(f"\n📊 修复完成: {success_count}/{len(graph_files)} 成功")
        
        # 再次验证
        print("\n🔍 验证修复结果...")
        is_consistent = validate_graph_dimensions("data/processed/graphs")
        
        if is_consistent:
            print("\n🎉 图特征维度标准化修复成功!")
        else:
            print("\n❌ 修复后仍存在维度不一致问题") 