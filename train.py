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

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Internal imports
from src.common import (
    load_config, setup_monitoring, set_seed, 
    get_logger, configure_reproducible_environment
)
from src.data import FeatureProcessor, RiskVectorizer, RiskNICLPDataset
from src.models import RiskNICLPModel, ProgressiveTrainer
from src.evaluation import RiskNICLPEvaluator

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train RiskNICLP-5G Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/baseline_config.yaml",
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
    
    # Create dataset
    dataset = RiskNICLPDataset(
        graph_dir=config.data.graph_dir,
        metadata_dir=config.data.metadata_dir,
        feature_processor=feature_processor,
        risk_vectorizer=risk_vectorizer,
        cache_manager=None  # Will be created internally
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
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
    
    # Get actual feature dimensions from processors
    feature_dim = len(data_info['feature_processor'].get_feature_names()) if data_info['feature_processor'].selected_features else config.model.feature_dim
    
    model = RiskNICLPModel(
        input_dim=feature_dim,
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
    """Create progressive trainer"""
    logger.logger.info("🏃 Setting up progressive trainer...")
    
    trainer = ProgressiveTrainer(
        model=model,
        config=config,
        logger=logger,
        device=torch.device(config.device)
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
        # Load configuration
        config = load_config(
            config_path=args.config,
            include_env=True
        )
        
        # Debug mode adjustments
        if args.debug:
            config.logging.log_level = "DEBUG"
            config.training.max_epochs = 2
            config.data.num_workers = 0
        
        # Setup experiment
        logger = setup_experiment(config, args)
        
        # Log system information
        logger.logger.info(f"🖥️ Device: {config.device}")
        logger.logger.info(f"🧵 Threads: {config.num_threads}")
        if config.device == "cpu":
            torch.set_num_threads(config.num_threads)
        
        # Prepare data
        data_info = prepare_data(config, logger)
        
        # Create model
        model = create_model(config, data_info, logger)
        
        # Create trainer and evaluator
        trainer = create_trainer(model, config, data_info, logger)
        evaluator = create_evaluator(model, config, data_info, logger)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.logger.info(f"📂 Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        if args.dry_run:
            logger.logger.info("🧪 Dry run completed successfully!")
            return
        
        # Training
        logger.logger.info("🚀 Starting training...")
        
        # Progressive training
        best_model = trainer.train_progressive(
            train_loader=data_info['train_loader'],
            val_loader=data_info['val_loader']
        )
        
        # Final evaluation
        logger.logger.info("🎯 Running final evaluation...")
        
        test_results = evaluator.evaluate(
            data_loader=data_info['test_loader'],
            split_name="test",
            save_results=True
        )
        
        logger.logger.info("📊 Test Results:")
        for metric, value in test_results.items():
            logger.logger.info(f"  {metric}: {value:.4f}")
        
        # Log final metrics
        logger.log_metrics({
            f"final_test/{k}": v for k, v in test_results.items()
        })
        
        # Save final model
        output_dir = Path(config.logging.log_dir) / config.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / "final_model.pt"
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'config': config,
            'test_results': test_results
        }, model_path)
        
        logger.logger.info(f"💾 Model saved to: {model_path}")
        logger.logger.info("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.logger.info("⏹️ Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.logger.error(f"❌ Training failed: {str(e)}")
        raise
        
    finally:
        # Cleanup
        if 'logger' in locals():
            logger.finish()


if __name__ == "__main__":
    main() 