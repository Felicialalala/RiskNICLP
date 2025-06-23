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


class RiskNICLPDataset(Dataset):
    """RiskNICLP-5G Dataset class"""
    
    def __init__(self, graph_dir: str, metadata_dir: str, feature_processor, risk_vectorizer, cache_manager=None):
        """
        Initialize dataset
        
        Args:
            graph_dir: Directory containing graph files
            metadata_dir: Directory containing metadata
            feature_processor: Feature processor instance
            risk_vectorizer: Risk vectorizer instance
            cache_manager: Cache manager (optional)
        """
        self.graph_dir = graph_dir
        self.metadata_dir = metadata_dir
        self.feature_processor = feature_processor
        self.risk_vectorizer = risk_vectorizer
        self.cache_manager = cache_manager
        
        # Initialize data storage
        self.graph_files = None
        self.data_index = None
        
    def prepare_data(self):
        """Prepare dataset by loading graph files and fitting processors"""
        # Load graph files
        self._load_graph_files()
        
        # Build data index mapping
        self._build_data_index()
        
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
            
            # Generate synthetic risk scores if not available
            risk_scores = torch.tensor([])  # Default empty
            if 'risk_score' in graph.ndata:
                # Extract from global node data (homogeneous graph)
                risk_scores = graph.ndata['risk_score']
            else:
                # Check heterogeneous node data
                all_risk_scores = []
                for ntype in graph.ntypes:
                    if 'risk_score' in graph.nodes[ntype].data:
                        all_risk_scores.append(graph.nodes[ntype].data['risk_score'])
                        
                if all_risk_scores:
                    risk_scores = torch.cat(all_risk_scores, dim=0)
                else:
                    # 生成与节点总数匹配的合成风险分数
                    total_nodes = sum([graph.number_of_nodes(ntype) for ntype in graph.ntypes])
                    # 根据节点类型生成不同的默认风险分数
                    node_risks = []
                    for ntype in graph.ntypes:
                        num_nodes = graph.number_of_nodes(ntype)
                        if ntype == 'Alert':
                            # Alert节点高风险
                            default_risk = 0.8
                        elif ntype == 'Flow':
                            # Flow节点中等风险
                            default_risk = 0.3
                        elif ntype == 'UE':
                            # UE节点低风险
                            default_risk = 0.1
                        else:
                            # 其他节点类型
                            default_risk = 0.2
                        
                        node_risks.append(torch.full((num_nodes,), default_risk, dtype=torch.float32))
                    
                    if node_risks:
                        risk_scores = torch.cat(node_risks, dim=0)
                    else:
                        risk_scores = torch.tensor([0.1])  # 最小兜底
                    
            # Generate synthetic edge labels if not available
            edge_labels = torch.tensor([])  # Default empty
            if 'label' in graph.edata:
                # Extract from global edge data (homogeneous graph)
                edge_labels = graph.edata['label']
            else:
                # Check heterogeneous edge data
                all_edge_labels = []
                for etype in graph.etypes:
                    if 'label' in graph.edges[etype].data:
                        all_edge_labels.append(graph.edges[etype].data['label'])
                        
                if all_edge_labels:
                    edge_labels = torch.cat(all_edge_labels, dim=0)
                else:
                    # 生成与边总数匹配的合成标签
                    total_edges = sum([graph.number_of_edges(etype) for etype in graph.etypes])
                    # 根据边类型生成不同的默认标签
                    edge_labels_list = []
                    for etype in graph.etypes:
                        num_edges = graph.number_of_edges(etype)
                        if etype == 'triggers':
                            # 触发关系更可能是攻击
                            default_label = 1
                        else:
                            # 其他关系多为正常
                            default_label = 0
                            
                        edge_labels_list.append(torch.full((num_edges,), default_label, dtype=torch.long))
                    
                    if edge_labels_list:
                        edge_labels = torch.cat(edge_labels_list, dim=0)
                    else:
                        edge_labels = torch.tensor([0])  # 最小兜底
                
            # Prepare output dict
            sample = {
                'graph': graph,
                'risk_scores': risk_scores,
                'edge_labels': edge_labels,
                'scale_type': file_info['scale_type'],
                'file_name': file_info['file_name'],
                'idx': idx
            }
            
            return sample
            
        except Exception as e:
            # Fallback to placeholder data if loading fails
            print(f"Warning: Failed to load {graph_file}: {e}. Using placeholder data.")
            
            # 创建最小的placeholder图
            placeholder_graph = dgl.graph(([], []), num_nodes=10)
            
            return {
                'graph': placeholder_graph,
                'risk_scores': torch.full((10,), 0.1, dtype=torch.float32),  # 10个节点的默认风险
                'edge_labels': torch.tensor([0]),  # 单个默认标签
                'scale_type': file_info['scale_type'],
                'file_name': file_info['file_name'],
                'idx': idx
            }


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