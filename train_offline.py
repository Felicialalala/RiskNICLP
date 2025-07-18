#!/usr/bin/env python3
"""
RiskNICLP-5G 离线训练脚本
使用预计算的离线样本进行训练，大幅提升训练效率
"""

import os
import sys
import time
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import yaml

# Internal imports
from src.common import (
    load_config, setup_monitoring, set_seed, 
    get_logger, configure_reproducible_environment
)
from src.data import FeatureProcessor, RiskVectorizer
from src.data.offline_dataset import create_offline_dataloaders
from src.models import RiskNICLPModel, ProgressiveTrainer
from src.evaluation import RiskNICLPEvaluator


def setup_experiment(config, args):
    """设置实验环境"""
    # Setup monitoring and logging
    logger = setup_monitoring(
        experiment_name=args.experiment_name or "offline_training",
        config=config
    )
    
    # Set random seeds for reproducibility
    set_seed(config.get('seed', 42))
    
    # Configure environment for reproducible results
    configure_reproducible_environment()
    
    logger.logger.info("🚀 RiskNICLP-5G 离线训练开始")
    logger.logger.info(f"📁 离线样本文件: {args.offline_samples}")
    logger.logger.info(f"🎯 实验名称: {args.experiment_name or 'offline_training'}")
    
    return logger


def prepare_offline_data(config, args, logger):
    """准备离线数据"""
    logger.logger.info("📊 准备离线数据处理管道...")
    
    # 检查离线样本文件
    offline_file = Path(args.offline_samples)
    if not offline_file.exists():
        raise FileNotFoundError(f"离线样本文件不存在: {offline_file}")
    
    file_size = offline_file.stat().st_size / 1024 / 1024
    logger.logger.info(f"📁 离线样本文件大小: {file_size:.1f}MB")
    
    # Create feature processor
    feature_processor = FeatureProcessor(config.data.feature_processing)
    logger.logger.info(f"🔧 特征处理器: {feature_processor.__class__.__name__}")
    
    # Create risk vectorizer
    risk_vectorizer = RiskVectorizer(config.data.risk_vectorization)
    logger.logger.info(f"🎯 风险向量化器: {risk_vectorizer.__class__.__name__}")
    
    # Create offline data loaders
    logger.logger.info("🔄 创建离线数据加载器...")
    train_loader, val_loader, test_loader = create_offline_dataloaders(
        offline_samples_file=str(offline_file),
        feature_processor=feature_processor,
        risk_vectorizer=risk_vectorizer,
        config=config
    )
    
    logger.logger.info(f"✅ 数据加载器创建完成")
    logger.logger.info(f"   训练集: {len(train_loader.dataset)}个样本")
    logger.logger.info(f"   验证集: {len(val_loader.dataset)}个样本") 
    logger.logger.info(f"   测试集: {len(test_loader.dataset)}个样本")
    
    # Get data info for model creation
    data_info = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'feature_processor': feature_processor,
        'risk_vectorizer': risk_vectorizer,
        'feature_dim': feature_processor.output_dim,
        'risk_embed_dim': risk_vectorizer.embed_dim,
        'num_node_types': 3,  # UE, Flow, Alert
        'num_edge_types': 3   # initiates, triggers, temporal_adj
    }
    
    return data_info


def create_model(config, data_info, logger):
    """创建模型"""
    logger.logger.info("🏗️ 创建RiskNICLP模型...")
    
    # Model configuration
    model_config = config.model
    model_config.feature_dim = data_info['feature_dim']
    model_config.risk_embed_dim = data_info['risk_embed_dim']
    model_config.num_node_types = data_info['num_node_types']
    model_config.num_edge_types = data_info['num_edge_types']
    
    # Create model
    model = RiskNICLPModel(model_config)
    
    # Move to device
    device = torch.device(config.get('device', 'cpu'))
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.logger.info(f"✅ 模型创建完成")
    logger.logger.info(f"   总参数: {total_params:,}")
    logger.logger.info(f"   可训练参数: {trainable_params:,}")
    logger.logger.info(f"   设备: {device}")
    
    return model


def create_trainer(model, config, data_info, logger):
    """创建训练器"""
    logger.logger.info("🎓 创建渐进式训练器...")
    
    trainer = ProgressiveTrainer(
        model=model,
        config=config.training,
        device=config.get('device', 'cpu'),
        logger=logger
    )
    
    logger.logger.info(f"✅ 训练器创建完成")
    logger.logger.info(f"   优化器: {trainer.optimizer.__class__.__name__}")
    logger.logger.info(f"   学习率: {config.training.learning_rate}")
    logger.logger.info(f"   训练轮数: {config.training.max_epochs}")
    
    return trainer


def create_evaluator(model, config, data_info, logger):
    """创建评估器"""
    logger.logger.info("📊 创建评估器...")
    
    evaluator = RiskNICLPEvaluator(
        model=model,
        config=config.evaluation,
        device=config.get('device', 'cpu'),
        logger=logger
    )
    
    logger.logger.info(f"✅ 评估器创建完成")
    
    return evaluator


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RiskNICLP-5G 离线训练')
    parser.add_argument('--config', type=str, default='config/riskniclp_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--offline-samples', type=str, required=True,
                       help='离线采样文件路径')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='实验名称')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cpu/cuda)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批次大小（覆盖配置文件）')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to object for easier access
        class Config:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, Config(v))
                    else:
                        setattr(self, k, v)
        
        config = Config(config_dict)
        
        # Override with command line arguments
        if args.device:
            config.device = args.device
        if args.epochs:
            config.training.max_epochs = args.epochs
        if args.batch_size:
            config.training.batch_size = args.batch_size
        
        # Debug mode adjustments
        if args.debug:
            config.training.max_epochs = 2
            config.training.batch_size = min(config.training.batch_size, 2)
            config.data.num_workers = 0
            print("🐛 调试模式: 限制训练轮数=2, 批次大小≤2")
        
        # Setup experiment
        logger = setup_experiment(config, args)
        
        # Setup device
        if hasattr(config, 'device'):
            device = config.device
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            config.device = device
        
        logger.logger.info(f"🖥️ 设备: {device}")
        
        # Prepare offline data
        data_info = prepare_offline_data(config, args, logger)
        
        # Create model
        model = create_model(config, data_info, logger)
        
        # Create trainer
        trainer = create_trainer(model, config, data_info, logger)
        
        # Create evaluator
        evaluator = create_evaluator(model, config, data_info, logger)
        
        # Train model
        logger.logger.info("🚀 开始离线训练...")
        start_time = time.time()
        
        trained_model = trainer.train_progressive(
            data_info['train_loader'],
            data_info['val_loader']
        )
        
        training_time = time.time() - start_time
        logger.logger.info(f"✅ 训练完成! 耗时: {training_time:.1f}秒")
        
        # Final evaluation
        logger.logger.info("🎯 执行最终评估...")
        test_results = evaluator.evaluate(
            data_info['test_loader'],
            'test'
        )
        
        logger.logger.info("🎉 离线训练完成!")
        logger.logger.info(f"📊 测试结果: {test_results}")
        
        return trained_model, test_results
        
    except Exception as e:
        print(f"❌ 离线训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main()
