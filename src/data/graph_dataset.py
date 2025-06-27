"""
RiskNICLP-5G Graph Dataset Module

Provides dataset classes for graph data loading and processing
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import dgl
import numpy as np
from .negative_sampler import BaselineNegativeSampler, RiskAwareNegativeSampler, create_negative_sampler


class RiskNICLPDataset(Dataset):
    """RiskNICLP-5G Dataset class"""
    
    def __init__(self, graph_dir: str, metadata_dir: str, feature_processor, risk_vectorizer, cache_manager=None, negative_sampler_config=None):
        """
        Initialize dataset
        
        Args:
            graph_dir: Directory containing graph files
            metadata_dir: Directory containing metadata
            feature_processor: Feature processor instance
            risk_vectorizer: Risk vectorizer instance
            cache_manager: Cache manager (optional)
            negative_sampler_config: Configuration for negative sampling
        """
        self.graph_dir = graph_dir
        self.metadata_dir = metadata_dir
        self.feature_processor = feature_processor
        self.risk_vectorizer = risk_vectorizer
        self.cache_manager = cache_manager
        
        # Initialize negative sampler
        if negative_sampler_config is None:
            # Default baseline negative sampler
            negative_sampler_config = {'type': 'baseline', 'neg_ratio': 3}
        self.negative_sampler = create_negative_sampler(negative_sampler_config)
        self.negative_sampler_config = negative_sampler_config
        
        # Initialize data storage
        self.graph_files = None
        self.data_index = None
        
    def prepare_data(self):
        """Prepare dataset by loading graph files and fitting processors"""
        # Load graph files
        self._load_graph_files()
        
        # Build data index mapping
        self._build_data_index()
        
        # Validate negative sampler configuration
        self._validate_negative_sampler()
        
    def _load_graph_files(self):
        """Load 75 DGL graph files from graph directory"""
        graph_dir = Path(self.graph_dir)
        if not graph_dir.exists():
            raise ValueError(f"Graph directory does not exist: {graph_dir}")
            
        # Find all .dgl files in directory
        graph_files = list(graph_dir.glob("*.dgl"))
        
        if len(graph_files) == 0:
            raise ValueError(f"No .dgl files found in {graph_dir}")
            
        # Sort files for consistent ordering
        self.graph_files = sorted(graph_files)
        
        print(f"Loaded {len(self.graph_files)} graph files from {graph_dir}")
        
    def _build_data_index(self):
        """Build data index mapping for efficient access"""
        if self.graph_files is None:
            raise ValueError("Graph files not loaded. Call _load_graph_files() first")
            
        self.data_index = {}
        for idx, graph_file in enumerate(self.graph_files):
            # Parse file name to get metadata
            file_name = graph_file.stem
            
            # Determine scale type from filename
            if 'fine' in file_name.lower():
                scale_type = 'fine'
            elif 'coarse' in file_name.lower():
                scale_type = 'coarse'
            else:
                scale_type = 'unknown'
                
            self.data_index[idx] = {
                'file_path': graph_file,
                'file_name': file_name,
                'scale_type': scale_type
            }
            
        print(f"Built data index for {len(self.data_index)} graph files")
    
    def _validate_negative_sampler(self):
        """验证负采样器与图数据的兼容性"""
        if self.negative_sampler is None:
            print("Warning: No negative sampler configured. Using default baseline sampler.")
            self.negative_sampler = create_negative_sampler({'type': 'baseline', 'neg_ratio': 3})
            
        # 记录采样器配置
        config = self.negative_sampler.get_config()
        print(f"✅ 负采样器已配置: {config['type']} - {config.get('strategy', 'N/A')}")
        print(f"   负样本比例: {config['neg_ratio']}:1")
        
        if config['type'] == 'risk_aware':
            print(f"   风险阈值: {config['risk_thresholds']}")
            print(f"   困难样本比例: {config['hard_ratio']}")
            if hasattr(self.negative_sampler, 'attack_type_mapping'):
                print(f"   攻击类型映射: {len(self.negative_sampler.attack_type_mapping)} 种攻击类型")
    
    def _generate_positive_negative_edges(self, graph, edge_labels, risk_scores, graph_file=None):
        """
        Extract positive and negative edge samples based on authoritative edge labels
        
        🔧 修复：完全使用传入的edge_labels，不再重新生成标签
        
        Args:
            graph: DGL graph
            edge_labels: Authoritative edge labels from __getitem__ (权威边标签)
            risk_scores: Node risk scores (for negative sampling only)
            
        Returns:
            Tuple of (positive_edges, negative_edges)
        """
        # 🛡️ 严格验证：确保edge_labels长度与边数匹配
        num_edges = graph.number_of_edges()
        if len(edge_labels) != num_edges:
            raise ValueError(f"❌ edge_labels长度{len(edge_labels)}与图边数{num_edges}不匹配")
        
        # Extract all edges from graph
        if graph.is_homogeneous:
            # Homogeneous graph
            src, dst = graph.edges()
            all_edges = torch.stack([src, dst], dim=1)
        else:
            # Heterogeneous graph - collect all edges
            all_edges_list = []
            for etype in graph.etypes:
                src, dst = graph.edges(etype=etype)
                if len(src) > 0:
                    edges = torch.stack([src, dst], dim=1)
                    all_edges_list.append(edges)
            
            if all_edges_list:
                all_edges = torch.cat(all_edges_list, dim=0)
            else:
                all_edges = torch.empty((0, 2), dtype=torch.long)
        
        # 🔧 关键修复：严格使用传入的权威edge_labels
        if len(edge_labels) > 0 and len(all_edges) > 0:
            # 提取正样本边：基于权威标签edge_labels==1
            positive_mask = edge_labels == 1
            positive_edges = all_edges[positive_mask]
            
            from pathlib import Path
            print(f"✅ 正样本边提取: 文件={Path(graph_file).name if graph_file else 'Unknown'}, "
                  f"正边={len(positive_edges)}/{len(all_edges)} ({len(positive_edges)/len(all_edges)*100:.1f}%)")
        else:
            positive_edges = torch.empty((0, 2), dtype=torch.long)
        
        # Generate negative edges using sampler
        if len(positive_edges) > 0:
            try:
                negative_edges = self.negative_sampler.sample_negatives(
                    graph, positive_edges, risk_scores
                )
            except Exception as e:
                print(f"Warning: Negative sampling failed: {e}. Using fallback.")
                # Fallback to simple negative sampling
                num_neg = len(positive_edges) * self.negative_sampler_config.get('neg_ratio', 3)
                num_nodes = graph.number_of_nodes()
                
                # Generate random negative edges avoiding positive ones
                positive_edge_set = set()
                for i in range(len(positive_edges)):
                    src, dst = positive_edges[i].tolist()
                    positive_edge_set.add((src, dst))
                    positive_edge_set.add((dst, src))
                
                neg_edges = []
                max_attempts = num_neg * 10
                attempts = 0
                
                while len(neg_edges) < num_neg and attempts < max_attempts:
                    src = torch.randint(0, num_nodes, (1,)).item()
                    dst = torch.randint(0, num_nodes, (1,)).item()
                    
                    if src != dst and (src, dst) not in positive_edge_set:
                        neg_edges.append([src, dst])
                    attempts += 1
                
                negative_edges = torch.tensor(neg_edges, dtype=torch.long) if neg_edges else torch.empty((0, 2), dtype=torch.long)
        else:
            negative_edges = torch.empty((0, 2), dtype=torch.long)
            
        return positive_edges, negative_edges
        
    def create_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> Tuple['RiskNICLPDataset', 'RiskNICLPDataset', 'RiskNICLPDataset']:
        """
        Create train/val/test splits with proper ratio implementation
        
        Args:
            train_ratio: Training data ratio (default 0.8)
            val_ratio: Validation data ratio (default 0.1)
            test_ratio: Test data ratio (default 0.1)
            seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        from .data_splits import create_data_splits
        
        # Ensure data is loaded
        if not hasattr(self, 'graph_files') or self.graph_files is None:
            self._load_graph_files()
            
        # Use the real data splitting function
        return create_data_splits(self, train_ratio, val_ratio, test_ratio, seed)
        
    def __len__(self) -> int:
        """Return dataset length based on actual loaded graph files"""
        if not hasattr(self, 'graph_files') or self.graph_files is None:
            self._load_graph_files()
        return len(self.graph_files)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index - load and process graph data"""
        if self.data_index is None:
            raise ValueError("Dataset not prepared. Call prepare_data() first")
            
        if idx >= len(self.data_index):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data_index)}")
            
        # Get file info
        file_info = self.data_index[idx]
        graph_file = file_info['file_path']
        
        try:
            # Load DGL graph
            graphs, labels = dgl.load_graphs(str(graph_file))
            graph = graphs[0]  # Assume single graph per file
            
            # Generate synthetic risk scores if not available - 修复异质图类型错误
            # 🛡️ 先按图类型区分：同构图直接取，异质图按节点类型拼接
            if graph.is_homogeneous:
                # 同构图：直接从graph.ndata获取
                if 'risk_score' in graph.ndata and torch.is_tensor(graph.ndata['risk_score']):
                    risk_scores = graph.ndata['risk_score']
                else:
                    # 同构图上也没有risk_score，使用默认值0.3
                    N = graph.number_of_nodes()
                    risk_scores = torch.full((N,), 0.3, dtype=torch.float32)
            else:
                # 异质图：优先使用图文件中保存的risk_score
                parts = []
                for ntype in graph.ntypes:
                    try:
                        node_data = graph.nodes[ntype].data
                        if 'risk_score' in node_data:
                            risk_tensor = node_data['risk_score']
                            if torch.is_tensor(risk_tensor):
                                # 使用图文件中保存的风险分数
                                parts.append(risk_tensor)
                            else:
                                print(f"⚠️ {ntype} 节点的risk_score不是tensor: {type(risk_tensor)}")
                                # 创建默认风险分数
                                default_risk = {'Alert': 0.8, 'Flow': 0.3, 'UE': 0.1}.get(ntype, 0.2)
                                num_nodes = graph.number_of_nodes(ntype)
                                parts.append(torch.full((num_nodes,), default_risk, dtype=torch.float32))
                        else:
                            # 降级方案：使用各节点类型的默认值
                            default_risk = {'Alert': 0.8, 'Flow': 0.3, 'UE': 0.1}.get(ntype, 0.2)
                            num_nodes = graph.number_of_nodes(ntype)
                            parts.append(torch.full((num_nodes,), default_risk, dtype=torch.float32))
                    except Exception as e:
                        print(f"⚠️ 处理 {ntype} 节点数据时出错: {e}")
                        # 兜底处理
                        default_risk = {'Alert': 0.8, 'Flow': 0.3, 'UE': 0.1}.get(ntype, 0.2)
                        num_nodes = graph.number_of_nodes(ntype)
                        parts.append(torch.full((num_nodes,), default_risk, dtype=torch.float32))
                
                if parts:
                    risk_scores = torch.cat(parts, dim=0)
                else:
                    # 兜底：如果没有任何节点类型，创建最小默认值
                    risk_scores = torch.tensor([0.3], dtype=torch.float32)
            
            # 🛡️ 强制类型检查和断言 - 确保risk_scores格式正确
            if not torch.is_tensor(risk_scores):
                raise RuntimeError(f"❌ risk_scores必须是tensor，但得到: {type(risk_scores)}")
            
            if risk_scores.dim() != 1:
                raise RuntimeError(f"❌ risk_scores必须是1维tensor，但形状为: {risk_scores.shape}")
            
            # 确保数据类型为float32
            if risk_scores.dtype != torch.float32:
                risk_scores = risk_scores.float()
            
            # Extract existing edge labels from graph data with improved logic
            edge_labels = torch.tensor([])
            
            # 🔧 修复判断逻辑：优先检查图类型而非edata中是否有label
            if graph.is_homogeneous and 'label' in graph.edata:
                # Extract from global edge data (homogeneous graph only)
                edge_labels = graph.edata['label']
                if torch.is_tensor(edge_labels):
                    print(f"🔍 从同质图提取边标签: {edge_labels.shape}")
                else:
                    print(f"⚠️ 同质图边标签非tensor: {type(edge_labels)}, 将跳过到后续处理")
                    edge_labels = torch.tensor([])  # 重置为空tensor，让后续逻辑处理
            else:
                # Check heterogeneous edge data
                all_edge_labels = []
                for etype in graph.etypes:
                    if 'label' in graph.edges[etype].data:
                        etype_labels = graph.edges[etype].data['label']
                        if torch.is_tensor(etype_labels):
                            all_edge_labels.append(etype_labels)
                            print(f"🔍 从异质图 {etype} 提取边标签: {etype_labels.shape}")
                        else:
                            print(f"⚠️ 异质图 {etype} 边标签非tensor: {type(etype_labels)}")
                        
                if all_edge_labels:
                    edge_labels = torch.cat(all_edge_labels, dim=0)
                    print(f"🔍 合并异质图边标签: {edge_labels.shape}")
                else:
                    # 🔧 权威边标签生成逻辑 - 基于Risk-PPR确保长度匹配
                    num_edges = graph.number_of_edges()
                    if num_edges > 0:
                        # 🛡️ 严格确保边标签长度=实际边数
                        all_src_nodes = []
                        all_dst_nodes = []
                        
                        if hasattr(graph, 'etypes') and len(graph.etypes) > 1:
                            # 异质图：遍历所有边类型
                            for etype in graph.etypes:
                                try:
                                    src, dst = graph.edges(etype=etype)
                                    all_src_nodes.append(src)
                                    all_dst_nodes.append(dst)
                                except Exception as e:
                                    print(f"⚠️ 处理边类型 {etype} 时出错: {e}")
                                    continue
                            
                            if all_src_nodes and all_dst_nodes:
                                src_nodes = torch.cat(all_src_nodes)
                                dst_nodes = torch.cat(all_dst_nodes)
                            else:
                                # 空图：创建长度为0的边标签
                                edge_labels = torch.empty(0, dtype=torch.float32)
                                src_nodes = None
                                dst_nodes = None
                        else:
                            # 同质图：直接获取边
                            try:
                                src_nodes, dst_nodes = graph.edges()
                            except Exception as e:
                                print(f"⚠️ 同质图边获取失败: {e}")
                                edge_labels = torch.empty(0, dtype=torch.float32)
                                src_nodes = None
                                dst_nodes = None
                        
                        # 基于Risk-PPR生成权威边标签
                        if src_nodes is not None and dst_nodes is not None and len(risk_scores) > max(src_nodes.max(), dst_nodes.max()):
                            print(f"🔍 权威边标签生成: 文件={graph_file}, 边数={len(src_nodes)}, 风险分数数={len(risk_scores)}")
                            
                            src_risks = risk_scores[src_nodes]
                            dst_risks = risk_scores[dst_nodes]
                            edge_risk_scores = (src_risks + dst_risks) / 2.0
                            
                            # 🔧 修复: 基于边类型的语义正样本定义，而非文件名
                            # 异质图的语义正样本策略
                            if hasattr(graph, 'etypes') and len(graph.etypes) > 1:
                                # 异质图：基于边类型语义定义正样本
                                edge_labels = torch.zeros(num_edges, dtype=torch.float32)
                                edge_offset = 0
                                
                                for etype in graph.etypes:
                                    src, dst = graph.edges(etype=etype)
                                    num_etype_edges = len(src)
                                    
                                    if num_etype_edges > 0:
                                        # 基于边类型的正样本比例
                                        if etype == 'triggers':
                                            # Flow -> Alert: 高风险传播，80%正样本
                                            pos_ratio = 0.8
                                        elif etype == 'initiates':
                                            # UE -> Flow: 中风险发起，40%正样本
                                            pos_ratio = 0.4
                                        elif etype == 'temporal_adj':
                                            # Flow -> Flow: 时序关联，15%正样本
                                            pos_ratio = 0.15
                                        else:
                                            # 其他边类型：10%正样本
                                            pos_ratio = 0.1
                                        
                                        # 基于风险分数优选正样本
                                        if len(src) > 0 and edge_offset + len(src) <= len(edge_risk_scores):
                                            etype_edge_risks = edge_risk_scores[edge_offset:edge_offset + num_etype_edges]
                                            num_positive = max(1, int(num_etype_edges * pos_ratio))
                                            
                                            # 选择风险最高的边作为正样本
                                            top_indices = torch.topk(etype_edge_risks, min(num_positive, len(etype_edge_risks))).indices
                                            edge_labels[edge_offset + top_indices] = 1.0
                                        
                                        edge_offset += num_etype_edges
                                
                                print(f"✅ 基于边类型语义定义: 正样本={edge_labels.sum().item():.0f}/{len(edge_labels)} ({edge_labels.sum().item()/len(edge_labels)*100:.1f}%)")
                            else:
                                # 同质图：基于风险分数分层
                                risk_threshold = torch.quantile(edge_risk_scores, 0.7)  # 降低到70%分位数
                                edge_labels = (edge_risk_scores > risk_threshold).float()
                                
                                # 确保至少20%正样本
                                min_positive = max(1, int(len(edge_labels) * 0.2))
                                if edge_labels.sum() < min_positive:
                                    top_indices = torch.topk(edge_risk_scores, min_positive).indices
                                    edge_labels[top_indices] = 1.0
                                
                                print(f"✅ 同质图风险分层: 正样本={edge_labels.sum().item():.0f}/{len(edge_labels)} ({edge_labels.sum().item()/len(edge_labels)*100:.1f}%)")
                                
                            # 🛡️ 严格验证：确保标签长度=边数
                            assert len(edge_labels) == num_edges, f"❌ 边标签长度{len(edge_labels)}不等于边数{num_edges}"
                            print(f"✅ 权威边标签生成完成: 正样本={edge_labels.sum().item():.0f}/{len(edge_labels)}")
                        else:
                            # 🛡️ 移除占位符：必须生成与边数匹配的标签
                            if src_nodes is None or dst_nodes is None:
                                print(f"⚠️ 无法获取边信息，生成空边标签: 文件={graph_file}")
                                edge_labels = torch.empty(0, dtype=torch.float32)
                            else:
                                print(f"⚠️ 风险分数不匹配，生成默认边标签: 风险分数数={len(risk_scores)}, 最大节点索引={max(src_nodes.max(), dst_nodes.max())}")
                                # 生成默认平衡标签，确保长度=边数
                                edge_labels = torch.zeros(num_edges, dtype=torch.float32)
                                # 20%作为正样本
                                num_positive = max(1, num_edges // 5)
                                positive_indices = torch.randperm(num_edges)[:num_positive]
                                edge_labels[positive_indices] = 1.0
                                
                            # 🛡️ 最终验证
                            assert len(edge_labels) == num_edges, f"❌ 默认边标签长度{len(edge_labels)}不等于边数{num_edges}"
                    else:
                        # 空图的情况 - 长度为0的边标签
                        edge_labels = torch.empty(0, dtype=torch.float32)
            
            # 🛡️ 强制edge_labels类型检查和修复
            if not torch.is_tensor(edge_labels):
                print(f"⚠️ edge_labels类型错误: {type(edge_labels)}, 转换为tensor")
                if isinstance(edge_labels, dict):
                    # 如果是字典，必须生成与边数匹配的边标签
                    num_edges = graph.number_of_edges()
                    edge_labels = torch.zeros(num_edges, dtype=torch.float32)
                    # 20%作为正样本
                    if num_edges > 0:
                        num_pos = max(1, num_edges // 5)
                        pos_indices = torch.randperm(num_edges)[:num_pos]
                        edge_labels[pos_indices] = 1.0
                elif hasattr(edge_labels, '__len__'):
                    edge_labels = torch.tensor(edge_labels, dtype=torch.float32)
                else:
                    # 🚨 移除占位符：如果无法处理，生成空标签并报错
                    raise ValueError(f"❌ 无法处理的edge_labels类型: {type(edge_labels)}")
                print(f"✅ edge_labels已修复为tensor: {edge_labels.shape}")
            
            # 🛡️ 最终权威验证：确保边标签长度与边数完全匹配
            num_edges = graph.number_of_edges()
            if len(edge_labels) != num_edges:
                raise ValueError(f"❌ 致命错误：边标签长度{len(edge_labels)}与图边数{num_edges}不匹配")
            
            # Generate negative samples (传递文件路径用于攻击类型判断)
            positive_edges, negative_edges = self._generate_positive_negative_edges(graph, edge_labels, risk_scores, graph_file)
            
            # 验证正负样本平衡 - 修复后的调试信息
            num_positive = len(positive_edges) if len(positive_edges.shape) > 1 else 0
            num_negative = len(negative_edges) if len(negative_edges.shape) > 1 else 0
            
            # 每隔50个样本打印一次验证信息（减少日志输出）
            if idx % 50 == 0:
                print(f"Sample {idx}: 修复后采样 - 正边:{num_positive}, 负边:{num_negative} | "
                      f"平衡度:{min(num_positive, num_negative)/max(num_positive, num_negative) if num_positive > 0 and num_negative > 0 else 0.0:.3f}")
            
            # 🛡️ 最终检查：确保所有返回的数据都是正确的类型
            assert torch.is_tensor(edge_labels), f"edge_labels必须是tensor，但得到: {type(edge_labels)}"
            assert torch.is_tensor(risk_scores), f"risk_scores必须是tensor，但得到: {type(risk_scores)}"
            
            # Prepare output dict
            sample = {
                'graph': graph,
                'risk_scores': risk_scores,
                'edge_labels': edge_labels,
                'positive_edges': positive_edges,
                'negative_edges': negative_edges,
                'scale_type': file_info['scale_type'],
                'file_name': file_info['file_name'],
                'idx': idx,
                # 添加统计信息
                'num_positive_edges': num_positive,
                'num_negative_edges': num_negative
            }
            
            return sample
            
        except Exception as e:
            # 不使用占位符数据，而是直接抛出错误以便调试
            print(f"❌ Failed to load {graph_file}: {e}")
            import traceback
            traceback.print_exc()
            raise e


class GraphDataLoader:
    """Graph data loader utility"""
    
    def __init__(self, dataset: RiskNICLPDataset, batch_size: int = 4):
        """
        Initialize graph data loader
        
        Args:
            dataset: RiskNICLP dataset
            batch_size: Batch size
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
    def get_dataloader(self) -> DataLoader:
        """Get PyTorch DataLoader"""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True
        ) 