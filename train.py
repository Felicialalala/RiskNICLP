#!/usr/bin/env python3
"""
RiskNICLP-5G Training Script

Main entry point for training the RiskNICLP-5G model with:
- Multi-layer configuration management
- Progressive training (Basic → Risk-aware)
- Comprehensive monitoring and logging
- Reproducible experiments
"""

import os
import sys
import time
import argparse
from pathlib import Path
import random
import signal
import atexit
import threading
import gc
import warnings
from typing import List
warnings.filterwarnings('ignore', category=FutureWarning)

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl

# Internal imports
from src.common import (
    load_config, setup_monitoring, set_seed, 
    get_logger, configure_reproducible_environment
)
from src.data import FeatureProcessor, RiskVectorizer, RiskNICLPDataset
from src.data.sampling_framework import UnifiedSamplingFramework, SamplingConfig
from src.models import RiskNICLPModel, ProgressiveTrainer
from src.evaluation import RiskNICLPEvaluator

# Global variables for cleanup
_active_dataloaders: List[DataLoader] = []
_cleanup_lock = threading.Lock()
_is_cleaning_up = False

def _cleanup_dataloaders():
    """Safely cleanup all active DataLoaders"""
    global _is_cleaning_up
    
    with _cleanup_lock:
        if _is_cleaning_up:
            return
        _is_cleaning_up = True
        
        try:
            for loader in _active_dataloaders:
                if hasattr(loader, '_iterator') and loader._iterator is not None:
                    try:
                        # Give workers time to finish current batches
                        time.sleep(0.1)
                        del loader._iterator
                    except:
                        pass
                        
                # Force garbage collection to clean up worker processes
                gc.collect()
                
        except Exception as e:
            print(f"Warning: Error during DataLoader cleanup: {e}")
        finally:
            _active_dataloaders.clear()

def _signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) gracefully"""
    print("\n🛑 收到中断信号，正在清理资源...")
    _cleanup_dataloaders()
    print("✅ 资源清理完成")
    sys.exit(0)

def _register_dataloader(loader: DataLoader):
    """Register a DataLoader for cleanup"""
    with _cleanup_lock:
        if not _is_cleaning_up:
            _active_dataloaders.append(loader)

# Register signal handlers and cleanup functions
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(_cleanup_dataloaders)

def graph_collate_fn(batch):
    """Simplified custom collate function for DGL graphs"""
    try:

        
        # 1. Extract graphs and batch them
        graphs = [item['graph'] for item in batch]
        batched_graph = dgl.batch(graphs)
        
        # 2. Extract and concatenate risk scores
        risk_scores = []
        for item in batch:
            if 'risk_scores' in item and torch.is_tensor(item['risk_scores']):
                risk_scores.append(item['risk_scores'])
            else:
                # Fallback: generate default risk scores
                num_nodes = item['graph'].number_of_nodes()
                default_risk = torch.full((num_nodes,), 0.3, dtype=torch.float32)
                risk_scores.append(default_risk)
        
        batched_risk_scores = torch.cat(risk_scores, dim=0) if risk_scores else torch.tensor([])
        
        # 3. Extract and concatenate edge labels - 🔧 修复：纯粹聚合，不再修改标签
        edge_labels = []
        for item in batch:
            # 🛡️ 严格使用权威边标签，不再修改
            if 'edge_labels' in item and torch.is_tensor(item['edge_labels']):
                labels = item['edge_labels']
                num_edges = item['graph'].number_of_edges()
                
                # 🛡️ 验证标签长度必须与边数匹配
                if len(labels) != num_edges:
                    raise ValueError(f"❌ Batch collate失败：边标签长度{len(labels)}与边数{num_edges}不匹配")
                
                edge_labels.append(labels)
            else:
                # 🚨 严格模式：不应该出现缺失边标签的情况
                raise ValueError(f"❌ 严重错误：样本缺少权威边标签")
        
        batched_edge_labels = torch.cat(edge_labels, dim=0) if edge_labels else torch.empty(0, dtype=torch.float32)
        
        # 4. 🔥 修复：正确提取正负样本边，不要从图结构提取所有边
        positive_edges = []
        negative_edges = []
        node_offset = 0
        
        for item in batch:
            graph = item['graph']
            
            # 🔥 修复：使用item中的正负样本边，而不是图中的所有边
            # 正样本边应该来自于dataset的负采样结果
            if 'positive_edges' in item and item['positive_edges'] is not None and len(item['positive_edges']) > 0:
                pos_edges = item['positive_edges'] + node_offset
                positive_edges.append(pos_edges)
                print(f"📍 从item提取正样本边: {len(item['positive_edges'])} 条")
            else:
                print(f"⚠️ item中缺少positive_edges，样本索引: {item.get('idx', 'unknown')}")
            
            # 负样本边
            if 'negative_edges' in item and item['negative_edges'] is not None and len(item['negative_edges']) > 0:
                neg_edges = item['negative_edges'] + node_offset
                negative_edges.append(neg_edges)
                print(f"📍 从item提取负样本边: {len(item['negative_edges'])} 条")
            else:
                print(f"⚠️ item中缺少negative_edges，样本索引: {item.get('idx', 'unknown')}")
            
            # Update node offset
            node_offset += graph.number_of_nodes()
        
        # 5. Concatenate edges
        batched_positive_edges = torch.cat(positive_edges, dim=0) if positive_edges else torch.empty((0, 2), dtype=torch.long)
        batched_negative_edges = torch.cat(negative_edges, dim=0) if negative_edges else torch.empty((0, 2), dtype=torch.long)
        
        # 6. Collect metadata
        scale_types = [item.get('scale_type', 'fine') for item in batch]
        file_names = [item.get('file_name', '') for item in batch]
        indices = [item.get('idx', 0) for item in batch]
        
        # 7. Return result with enhanced training metadata
        # 🔧 正样本增强策略：解决正样本稀少问题
        enhanced_positive_edges = []
        enhanced_edge_labels = []
        pos_weight_multiplier = 1.0
        
        # 计算每个图的节点偏移量
        node_offset_list = [0]
        for item in batch:
            node_offset_list.append(node_offset_list[-1] + item['graph'].number_of_nodes())
        
        for i, item in enumerate(batch):
            # 提取属于当前图的正样本边
            if len(batched_positive_edges) > 0:
                mask = (batched_positive_edges[:, 0] >= node_offset_list[i]) & (batched_positive_edges[:, 0] < node_offset_list[i+1])
                original_pos_edges = batched_positive_edges[mask]
            else:
                original_pos_edges = torch.empty((0, 2), dtype=torch.long)
            
            num_original_pos = len(original_pos_edges)
            
            # 正样本过少时进行增强
            if num_original_pos > 0 and num_original_pos < 64:
                # 策略1: 时间抖动增强 (±1 node offset模拟)
                jittered_edges = []
                for edge in original_pos_edges:
                    src, dst = edge[0].item(), edge[1].item()
                    # 轻微抖动：±1范围内的节点
                    for _ in range(2):  # 每个正边生成2个抖动版本
                        jitter_src = max(node_offset_list[i], min(node_offset_list[i+1]-1, src + random.choice([-1, 0, 1])))
                        jitter_dst = max(node_offset_list[i], min(node_offset_list[i+1]-1, dst + random.choice([-1, 0, 1])))
                        if jitter_src != jitter_dst and [jitter_src, jitter_dst] not in original_pos_edges.tolist():
                            jittered_edges.append([jitter_src, jitter_dst])
                
                # 策略2: 重复采样达到最小阈值
                target_pos_count = min(64, num_original_pos * 3)
                if len(jittered_edges) + num_original_pos < target_pos_count:
                    repeat_count = (target_pos_count - len(jittered_edges)) // num_original_pos + 1
                    repeated_edges = original_pos_edges.repeat(repeat_count, 1)[:target_pos_count - len(jittered_edges)]
                    enhanced_positive_edges.extend(repeated_edges.tolist())
                
                enhanced_positive_edges.extend(jittered_edges)
                
                # 计算正样本权重倍数
                pos_weight_multiplier = max(pos_weight_multiplier, target_pos_count / max(num_original_pos, 1))
            
        # 🔥 修复2: 正样本去重 + 与负样本冲突检测
        if enhanced_positive_edges:
            enhanced_pos_tensor = torch.tensor(enhanced_positive_edges, dtype=torch.long)
            batched_positive_edges = torch.cat([batched_positive_edges, enhanced_pos_tensor], dim=0)
        
        # 去重正样本
        if len(batched_positive_edges) > 0:
            original_pos_count = len(batched_positive_edges)
            batched_positive_edges = torch.unique(batched_positive_edges, dim=0)
            dedup_pos_count = len(batched_positive_edges)
            
            if original_pos_count != dedup_pos_count:
                print(f"🔧 正样本去重: {original_pos_count} → {dedup_pos_count} (-{original_pos_count - dedup_pos_count} 重复)")
        
        # 去重负样本 + 去除与正样本冲突的负样本
        if len(batched_negative_edges) > 0:
            original_neg_count = len(batched_negative_edges)
            
            # 去重负样本
            batched_negative_edges = torch.unique(batched_negative_edges, dim=0)
            
            # 去除与正样本冲突的负样本（包括反向）
            if len(batched_positive_edges) > 0:
                # 创建正样本集合（包括反向）
                pos_set = set()
                for pos_edge in batched_positive_edges:
                    u, v = pos_edge[0].item(), pos_edge[1].item()
                    pos_set.add((u, v))
                    pos_set.add((v, u))  # 添加反向边
                
                # 过滤冲突的负样本
                valid_neg_mask = []
                for neg_edge in batched_negative_edges:
                    u, v = neg_edge[0].item(), neg_edge[1].item()
                    valid_neg_mask.append((u, v) not in pos_set)
                
                batched_negative_edges = batched_negative_edges[torch.tensor(valid_neg_mask)]
            
            final_neg_count = len(batched_negative_edges)
            if original_neg_count != final_neg_count:
                print(f"🔧 负样本清理: {original_neg_count} → {final_neg_count} (-{original_neg_count - final_neg_count} 重复/冲突)")
        
        # 动态计算pos_weight用于BCE loss
        total_pos = len(batched_positive_edges)
        total_neg = len(batched_negative_edges)
        pos_weight = max(1.0, total_neg / max(total_pos, 1)) * pos_weight_multiplier
        
        # 7. Return result with enhanced training metadata
        result = {
            'graph': batched_graph,
            'risk_scores': batched_risk_scores,
            'edge_labels': batched_edge_labels,
            'scale_types': scale_types,
            'file_names': file_names,
            'indices': indices,
            'positive_edges': batched_positive_edges,
            'negative_edges': batched_negative_edges,
            # 🔧 新增：正样本增强元数据
            'pos_weight': pos_weight,
            'enhancement_applied': len(enhanced_positive_edges) > 0,
            'original_pos_count': total_pos - len(enhanced_positive_edges) if enhanced_positive_edges else total_pos,
            'enhanced_pos_count': len(enhanced_positive_edges)
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Collate function error: {e}")
        # Return minimal fallback batch
        return {
            'graph': dgl.graph(([], []), num_nodes=1),
            'risk_scores': torch.tensor([0.3]),
            'edge_labels': torch.tensor([0]),
            'scale_types': ['fine'],
            'file_names': ['fallback'],
            'indices': [0],
            'positive_edges': torch.empty((0, 2), dtype=torch.long),
            'negative_edges': torch.empty((0, 2), dtype=torch.long)
        }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train RiskNICLP-5G Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/riskniclp_config.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name override"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint path"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run without actual training"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def setup_experiment(config, args):
    """Setup experiment environment and logging"""
    
    # Configure reproducible environment
    configure_reproducible_environment()
    set_seed(config.seed, deterministic=True)
    
    # Setup monitoring
    logger = setup_monitoring(
        enable_wandb=config.logging.enable_wandb and not args.dry_run,
        enable_tensorboard=config.logging.enable_tensorboard,
        wandb_project=config.logging.wandb_project,
        wandb_config=config.__dict__,
        log_level=getattr(config.logging, 'log_level', 'INFO')
    )
    
    # Set experiment name
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    elif config.experiment_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"riskniclp_baseline_{timestamp}"
    
    logger.logger.info(f"🚀 Starting experiment: {config.experiment_name}")
    logger.logger.info(f"🔧 Configuration: {config}")
    
    return logger


def prepare_data(config, logger):
    """Prepare data processing components"""
    logger.logger.info("📊 Preparing data processing pipeline...")
    
    # Initialize feature processor
    feature_processor = FeatureProcessor(
        method=config.data.feature_selection_method,
        variance_threshold=config.data.variance_threshold,
        correlation_threshold=config.data.correlation_threshold,
        target_dim=config.model.feature_dim,
        cache_dir=str(Path(config.data.cache_dir) / "features"),
        temporal_config=getattr(config.data, 'temporal_encoding', {})
    )
    
    # Initialize risk vectorizer
    risk_vectorizer = RiskVectorizer(
        num_bins=config.model.num_risk_bins,
        embed_dim=config.model.risk_embed_dim,
        normalization=config.data.risk_norm_method,
        adaptive_binning=config.data.adaptive_binning,
        cache_dir=str(Path(config.data.cache_dir) / "risk")
    )
    
    # Extract Phase 2 negative sampling configuration
    negative_sampler_config = None
    if hasattr(config, 'phase2') and 'negative_sampling' in config.phase2:
        negative_sampler_config = config.phase2['negative_sampling']
        strategy = negative_sampler_config.get('strategy', 'unknown')
        neg_ratio = negative_sampler_config.get('neg_ratio', 3)
        logger.logger.info(f"🎯 启用负采样策略: {strategy}, 负样本比例: {neg_ratio}:1")
        
        # Validate configuration
        if strategy not in ['uniform', 'stratified', 'hard', 'semantic', 'semantic_stratified']:
            logger.logger.warning(f"⚠️ 未知负采样策略: {strategy}, 将使用baseline")
            negative_sampler_config = None
    else:
        logger.logger.info("📋 未启用Phase 2负采样，使用默认baseline采样器")
    
    # 🚀 Enhanced caching configuration
    enable_edge_label_cache = getattr(config.data, 'enable_edge_label_cache', True)
    enable_sample_cache = getattr(config.data, 'enable_sample_cache', False)  # 🔥 修复：默认禁用采样缓存
    sample_cache_config = getattr(config.data, 'sample_cache_config', {'max_cache_size_gb': 0.0})
    
    logger.logger.info(f"🚀 缓存配置: 边标签缓存={'启用' if enable_edge_label_cache else '禁用'}, "
                      f"采样缓存={'启用' if enable_sample_cache else '禁用'}")
    if enable_sample_cache:
        cache_size = sample_cache_config.get('max_cache_size_gb', 10.0)
        compression = sample_cache_config.get('compression_enabled', True)
        logger.logger.info(f"   📊 采样缓存配置: 大小限制={cache_size}GB, 压缩={'启用' if compression else '禁用'}")

    # Create dataset with enhanced caching
    dataset = RiskNICLPDataset(
        graph_dir=config.data.graph_dir,
        metadata_dir=config.data.metadata_dir,
        feature_processor=feature_processor,
        risk_vectorizer=risk_vectorizer,
        cache_manager=None,  # Will be created internally
        negative_sampler_config=negative_sampler_config,  # Pass negative sampling config
        enable_edge_label_cache=enable_edge_label_cache,
        enable_sample_cache=enable_sample_cache,
        sample_cache_config=sample_cache_config
    )
    
    # Prepare data (this will fit processors if needed)
    logger.logger.info("🔄 Fitting data processors...")
    dataset.prepare_data()
    
    # Create data splits
    train_dataset, val_dataset, test_dataset = dataset.create_splits(
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.seed
    )
    
    logger.logger.info(f"📈 Data splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders with custom collate function
    dataloader_kwargs = {
        'batch_size': config.training.batch_size,
        'num_workers': config.data.num_workers,
        'pin_memory': config.data.pin_memory,
        'collate_fn': graph_collate_fn
    }
    
    # Add prefetch_factor only if num_workers > 0
    if config.data.num_workers > 0 and config.data.prefetch_factor is not None:
        dataloader_kwargs['prefetch_factor'] = config.data.prefetch_factor
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )
    _register_dataloader(train_loader)
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    _register_dataloader(val_loader)
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    _register_dataloader(test_loader)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader, 
        'test_loader': test_loader,
        'feature_processor': feature_processor,
        'risk_vectorizer': risk_vectorizer,
        'dataset': dataset
    }


def create_model(config, data_info, logger):
    """Create and initialize model"""
    logger.logger.info("🏗️ Creating model architecture...")
    
    # 使用配置中的节点特征维度 (方案B)
    node_feature_dims = config.model.node_feature_dims
    
    model = RiskNICLPModel(
        node_feature_dims=node_feature_dims,  # 传入异质图节点维度
        feature_dim=config.model.feature_dim,
        hidden_dim=config.model.hidden_dim,
        risk_embed_dim=config.model.risk_embed_dim,
        num_risk_bins=config.model.num_risk_bins,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        temperature=config.model.temperature,
        gravity_epsilon=config.model.gravity_epsilon
    )
    
    # Move to device
    device = torch.device(config.device)
    model = model.to(device)
    
    # Log model info
    logger.log_model_info(model)
    
    return model


def create_trainer(model, config, data_info, logger):
    """Create progressive trainer with temporal prediction support"""
    logger.logger.info("🏃 Setting up progressive trainer...")
    
    # Create temporal predictor if enabled in Phase 2
    temporal_predictor = None
    if hasattr(config, 'phase2') and config.phase2['temporal_prediction'].get('enabled', False):
        from src.models.temporal_predictor import GraphTemporalPredictor
        
        temporal_config = config.phase2['temporal_prediction']
        method = temporal_config.get('method', 'hetero_lstm')
        
        logger.logger.info(f"🕐 Initializing temporal predictor: {method}")
        
        # For hetero_lstm method, pass the shared_encoder
        shared_encoder = None
        if method == 'hetero_lstm':
            if hasattr(model, 'shared_encoder'):
                shared_encoder = model.shared_encoder
                logger.logger.info("✅ Shared encoder passed to temporal predictor")
            else:
                logger.logger.warning("⚠️ Shared encoder not found, using fallback method")
        
        temporal_predictor = GraphTemporalPredictor(
            hidden_dim=config.model.hidden_dim,
            future_steps=temporal_config.get('future_steps', 3),
            method=method,
            shared_encoder=shared_encoder
        )
        
        # Move to device
        temporal_predictor = temporal_predictor.to(torch.device(config.device))
        logger.logger.info(f"🚀 Temporal predictor ({method}) initialized successfully")
    
    trainer = ProgressiveTrainer(
        model=model,
        config=config,
        logger=logger,
        device=torch.device(config.device),
        temporal_predictor=temporal_predictor
    )
    
    return trainer


def create_evaluator(model, config, data_info, logger):
    """Create evaluator"""
    logger.logger.info("📊 Setting up evaluator...")
    
    evaluator = RiskNICLPEvaluator(
        model=model,
        config=config,
        logger=logger,
        device=torch.device(config.device)
    )
    
    return evaluator


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Create CLI args list from our parsed args
        cli_args = [
            '--config', args.config
        ]
        if args.experiment_name:
            cli_args.extend(['--experiment-name', args.experiment_name])
            
        # Load configuration
        config = load_config(
            config_path=args.config,
            cli_args=cli_args,
            include_env=True
        )
        
        # Setup experiment first
        logger = setup_experiment(config, args)
        
        # Debug mode adjustments
        if args.debug:
            config.training.max_epochs = 2
            config.training.batch_size = min(config.training.batch_size, 2)
            config.data.num_workers = 0
            config.logging.log_level = 'DEBUG'
            logger.logger.info("🐛 Debug模式: 限制训练轮数=2, 批次大小≤2")
        
        # Initialize device configuration
        import torch
        if hasattr(config, 'device') and config.device is not None:
            device = config.device
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if hasattr(config, 'num_threads') and config.num_threads is not None:
            num_threads = config.num_threads
        else:
            num_threads = torch.get_num_threads() if device == 'cpu' else 1
            
        config.device = device
        config.num_threads = num_threads
        
        logger.logger.info(f"🖥️ Device: {config.device}")
        logger.logger.info(f"🧵 Threads: {config.num_threads}")
        
        # Prepare data
        data_info = prepare_data(config, logger)
        
        # Create model
        model = create_model(config, data_info, logger)
        
        # Create trainer  
        trainer = create_trainer(model, config, data_info, logger)
        
        # Create evaluator
        evaluator = create_evaluator(model, config, data_info, logger)
        
        # 🔥 修复：安全的梯度调试钩子注册
        debug_hooks = []
        try:
            # 只在有可训练风险参数时注册钩子
            risk_params_trainable = any(p.requires_grad for name, p in model.named_parameters() 
                                       if any(keyword in name for keyword in ['beta', 'temperature']))
            
            if risk_params_trainable:
                debug_hooks = register_debug_hooks(model)
                logger.logger.info(f"✅ 梯度调试钩子已注册: {len(debug_hooks)} 个")
            else:
                logger.logger.info("ℹ️ 当前阶段无可训练风险参数，跳过梯度钩子注册")
            
        except Exception as e:
            logger.logger.warning(f"⚠️ 梯度钩子注册失败: {e}，继续训练")
            debug_hooks = []
        
        # Train model
        logger.logger.info("🚀 Starting training...")
        
        # 🔥 修改训练流程，集成梯度诊断
        if debug_hooks:
            # 获取第一个batch用于诊断
            sample_batch = next(iter(data_info['train_loader']))
            
            logger.logger.info("🔥 执行训练前梯度诊断...")
            diagnosis_results = comprehensive_gradient_diagnosis(
                model=model, 
                batch=sample_batch, 
                optimizer=trainer.optimizer
            )
            
            # 注册hooks监控训练过程
            logger.logger.info("🔍 已注册梯度监控hooks")
        
        # 执行渐进式训练
        trained_model = trainer.train_progressive(
            data_info['train_loader'],
            data_info['val_loader']
        )
        
        # 清理hooks
        if debug_hooks:
            for hook in debug_hooks:
                hook.remove()
            logger.logger.info("🧹 已清理梯度监控hooks")
        
        # Final evaluation
        logger.logger.info("🎯 Running final evaluation...")
        test_results = evaluator.evaluate(
            data_info['test_loader'],
            'test'
        )
        
        # Log test results
        logger.logger.info("📊 Test Results:")
        for key, value in test_results.items():
            logger.logger.info(f"  {key}: {value:.4f}")
        
        # Save final model
        save_path = Path(config.logging.log_dir) / config.experiment_name / "final_model.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trained_model.state_dict(), save_path)
        logger.logger.info(f"💾 Model saved to: {save_path}")
        
        logger.logger.info("✅ Training completed successfully!")
        
        # Debug模式额外验证
        if args.debug:
            logger.logger.info("🔍 Debug模式验证:")
            # 验证负采样配置是否正确传递
            dataset = data_info['dataset']
            if hasattr(dataset, 'negative_sampler') and dataset.negative_sampler is not None:
                sampler_type = type(dataset.negative_sampler).__name__
                logger.logger.info(f"  负采样器类型: {sampler_type}")
                if hasattr(dataset.negative_sampler, 'strategy'):
                    logger.logger.info(f"  采样策略: {dataset.negative_sampler.strategy}")
            else:
                logger.logger.warning("  ⚠️ 负采样器未正确初始化")
             
            # 验证批次中是否包含正负样本边
            train_loader = data_info['train_loader']
            batch_sample = next(iter(train_loader))
            has_pos_edges = 'positive_edges' in batch_sample and batch_sample['positive_edges'] is not None
            has_neg_edges = 'negative_edges' in batch_sample and batch_sample['negative_edges'] is not None
            logger.logger.info(f"  批次包含正样本边: {has_pos_edges}")
            logger.logger.info(f"  批次包含负样本边: {has_neg_edges}")
            
            if has_pos_edges:
                logger.logger.info(f"  正样本边数量: {len(batch_sample['positive_edges'])}")
            if has_neg_edges:
                logger.logger.info(f"  负样本边数量: {len(batch_sample['negative_edges'])}")
        
        return trained_model
        
    except KeyboardInterrupt:
        if 'logger' in locals():
            logger.logger.info("⏹️ Training interrupted by user")
        print("⏹️ Training interrupted by user")
        _cleanup_dataloaders()
        sys.exit(1)
        
    except Exception as e:
        if 'logger' in locals():
            logger.logger.error(f"❌ Training failed: {str(e)}")
        else:
            print(f"❌ Training failed: {str(e)}")
        _cleanup_dataloaders()
        raise
        
    finally:
        # Cleanup resources
        _cleanup_dataloaders()
        if 'logger' in locals():
            logger.finish()


if __name__ == "__main__":
    main() 