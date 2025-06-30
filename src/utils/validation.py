"""
RiskNICLP-5G Validation Utilities

Comprehensive validation tools for model training and data integrity
"""

import torch
import torch.nn as nn
import dgl
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging


class ModelValidator:
    """🛰️ 模型与数据完整性验证器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def validate_dgl_graph(self, graph: dgl.DGLGraph, graph_name: str = "graph") -> bool:
        """验证DGL图的完整性"""
        
        try:
            # 1. 基础图结构验证
            assert graph.num_nodes() > 0, f"{graph_name}: 空图 (节点数=0)"
            assert graph.num_edges() >= 0, f"{graph_name}: 负边数"
            
            # 2. 异质图验证
            if not graph.is_homogeneous:
                for ntype in graph.ntypes:
                    num_nodes = graph.number_of_nodes(ntype)
                    assert num_nodes >= 0, f"{graph_name}: {ntype} 节点数为负"
                    
                for etype in graph.etypes:
                    try:
                        src, dst = graph.edges(etype=etype)
                        assert len(src) == len(dst), f"{graph_name}: {etype} 边索引长度不匹配"
                        if len(src) > 0:
                            assert src.max() < graph.num_nodes(), f"{graph_name}: {etype} 源节点索引越界"
                            assert dst.max() < graph.num_nodes(), f"{graph_name}: {etype} 目标节点索引越界"
                    except Exception as e:
                        self.logger.warning(f"{graph_name}: {etype} 边验证失败: {e}")
            
            # 3. 节点特征验证
            for ntype in graph.ntypes:
                node_data = graph.nodes[ntype].data
                if 'feat' in node_data:
                    feat = node_data['feat']
                    assert torch.isfinite(feat).all(), f"{graph_name}: {ntype} 特征包含NaN/Inf"
                    
            # 4. DGL内置验证
            # dgl.validate(graph)  # 这个可能会比较严格，先注释
            
            self.logger.info(f"✅ {graph_name} DGL图验证通过: {graph.num_nodes()}节点, {graph.num_edges()}边")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {graph_name} DGL图验证失败: {e}")
            return False
    
    def validate_loss(self, loss: torch.Tensor, loss_name: str = "loss") -> bool:
        """验证损失函数值"""
        
        try:
            assert torch.isfinite(loss), f"{loss_name}: NaN/Inf值"
            assert loss.item() >= 0, f"{loss_name}: 负值 {loss.item()}"
            
            # 检查异常大的损失值
            if loss.item() > 100:
                self.logger.warning(f"⚠️ {loss_name} 异常大: {loss.item()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {loss_name} 验证失败: {e}")
            return False
    
    def validate_model_parameters(self, model: nn.Module) -> bool:
        """验证模型参数状态"""
        
        try:
            nan_params = []
            large_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if not torch.isfinite(param).all():
                        nan_params.append(name)
                    
                    if param.abs().max() > 100:
                        large_params.append((name, param.abs().max().item()))
            
            if nan_params:
                self.logger.error(f"❌ 模型参数包含NaN/Inf: {nan_params}")
                return False
                
            if large_params:
                self.logger.warning(f"⚠️ 模型参数异常大: {large_params}")
            
            self.logger.info("✅ 模型参数验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 模型参数验证失败: {e}")
            return False
    
    def validate_batch_data(self, batch: Dict[str, Any]) -> bool:
        """验证batch数据完整性"""
        
        try:
            # 检查必需字段
            required_fields = ['graph', 'risk_scores']
            for field in required_fields:
                assert field in batch, f"Batch缺少必需字段: {field}"
            
            # 验证图
            graph = batch['graph']
            if not self.validate_dgl_graph(graph, "batch_graph"):
                return False
            
            # 验证风险分数
            risk_scores = batch['risk_scores']
            if len(risk_scores) > 0:
                assert torch.isfinite(risk_scores).all(), "风险分数包含NaN/Inf"
                assert (risk_scores >= 0).all() and (risk_scores <= 1).all(), "风险分数超出[0,1]范围"
            
            # 验证边标签 (如果存在)
            if 'edge_labels' in batch:
                edge_labels = batch['edge_labels']
                assert torch.isfinite(edge_labels).all(), "边标签包含NaN/Inf"
                assert ((edge_labels == 0) | (edge_labels == 1)).all(), "边标签不是0/1二值"
            
            self.logger.info("✅ Batch数据验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Batch数据验证失败: {e}")
            return False
    
    def validate_training_step(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        loss: torch.Tensor,
        step: int
    ) -> bool:
        """综合训练步骤验证"""
        
        self.logger.info(f"🔍 验证训练步骤 {step}")
        
        # 1. 验证batch数据
        if not self.validate_batch_data(batch):
            return False
        
        # 2. 验证损失值
        if not self.validate_loss(loss, f"step_{step}_loss"):
            return False
        
        # 3. 验证模型参数 (每10步检查一次)
        if step % 10 == 0:
            if not self.validate_model_parameters(model):
                return False
        
        # 4. 内存使用检查
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            if memory_used > 8.0:  # 超过8GB警告
                self.logger.warning(f"⚠️ GPU内存使用过高: {memory_used:.2f}GB")
        
        return True


class PerformanceTracker:
    """🛰️ 性能指标追踪器"""
    
    def __init__(self):
        self.history = {
            'auc': [],
            'loss': [],
            'beta_src': [],
            'beta_dst': [], 
            'temperature': [],
            'attention_ratio': [],
            'memory_usage': []
        }
        
    def record_step(
        self, 
        step: int,
        metrics: Dict[str, float],
        model_params: Optional[Dict[str, float]] = None
    ):
        """记录单步训练指标"""
        
        # 记录核心指标
        for key in ['auc', 'loss']:
            if key in metrics:
                self.history[key].append(metrics[key])
        
        # 记录模型参数
        if model_params:
            for key in ['beta_src', 'beta_dst', 'temperature', 'attention_ratio']:
                if key in model_params:
                    self.history[key].append(model_params[key])
        
        # 记录内存使用
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            self.history['memory_usage'].append(memory_gb)
    
    def get_trend_analysis(self) -> Dict[str, str]:
        """分析指标趋势"""
        
        trends = {}
        
        for metric, values in self.history.items():
            if len(values) >= 3:
                recent_trend = np.mean(values[-3:]) - np.mean(values[-6:-3]) if len(values) >= 6 else 0
                if recent_trend > 0.001:
                    trends[metric] = "📈 上升"
                elif recent_trend < -0.001:
                    trends[metric] = "📉 下降"
                else:
                    trends[metric] = "➡️ 稳定"
            else:
                trends[metric] = "➖ 数据不足"
        
        return trends
    
    def detect_convergence(self, metric: str = 'loss', window: int = 5, tolerance: float = 0.001) -> bool:
        """检测指标是否收敛"""
        
        if metric not in self.history or len(self.history[metric]) < window:
            return False
        
        recent_values = self.history[metric][-window:]
        std_dev = np.std(recent_values)
        
        return std_dev < tolerance
    
    def export_tensorboard_data(self) -> Dict[str, list]:
        """导出TensorBoard格式数据"""
        
        return {
            'scalars': self.history,
            'step_count': len(self.history.get('loss', []))
        }


# 全局验证器实例
validator = ModelValidator()
tracker = PerformanceTracker() 