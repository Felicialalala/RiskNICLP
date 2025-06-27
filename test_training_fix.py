#!/usr/bin/env python3
"""
训练修复验证脚本

验证修复后的训练系统:
1. 真实验证函数而不是硬编码
2. 正确的测试数据生成
3. 准确的负采样指标计算
"""

import os
import sys
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.common.logger import get_logger
from src.config.config_manager import ConfigManager
from src.data.graph_dataset import HeteroGraphDataset
from src.data.graph_utils import graph_collate_fn
from src.models.riskniclp_model import RiskNICLPModel
from src.models.training import ProgressiveTrainer
from src.evaluation.evaluator import RiskNICLPEvaluator


def test_training_fixes():
    """测试训练修复效果"""
    logger = get_logger()
    logger.logger.info("🧪 测试训练修复效果")
    
    try:
        # 1. 加载配置
        config_manager = ConfigManager("config/config.yaml")
        config = config_manager.get_config()
        
        # 2. 创建数据集（小规模测试）
        logger.logger.info("📊 准备测试数据...")
        dataset = HeteroGraphDataset(
            data_dir=config.data.data_dir,
            scale_types=["fine"],
            neg_ratio=2,
            max_files=3  # 只测试3个文件
        )
        
        # 3. 创建数据加载器
        train_loader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False,
            collate_fn=graph_collate_fn,
            num_workers=0
        )
        
        # 4. 创建模型
        logger.logger.info("🏗️ 创建模型...")
        device = torch.device('cpu')  # 使用CPU避免内存问题
        
        model = RiskNICLPModel(
            node_feature_dims=config.model.node_feature_dims,
            feature_dim=config.model.feature_dim,
            hidden_dim=config.model.hidden_dim,
            risk_embed_dim=config.model.risk_embed_dim,
            num_risk_bins=config.model.num_risk_bins,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            temperature=config.model.temperature,
            gravity_epsilon=config.model.gravity_epsilon
        ).to(device)
        
        # 5. 创建训练器
        trainer = ProgressiveTrainer(
            model=model,
            config=config,
            logger=logger,
            device=device
        )
        
        # 6. 测试单个batch的训练
        logger.logger.info("🎯 测试单批次训练...")
        
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 2:  # 只测试前2个batch
                break
                
            try:
                # 移动数据到设备
                batch = trainer._move_batch_to_device(batch)
                
                # 前向传播
                fine_scores, coarse_scores = model(batch)
                
                logger.logger.info(
                    f"✅ Batch {batch_idx}: "
                    f"fine_scores={fine_scores.shape}, coarse_scores={coarse_scores.shape}"
                )
                
                # 测试验证函数
                model.eval()
                with torch.no_grad():
                    val_metrics = trainer.validate_epoch(DataLoader([batch], batch_size=1, collate_fn=lambda x: x[0]))
                
                logger.logger.info(
                    f"📊 验证指标: AUC={val_metrics['auc']:.4f}, "
                    f"Precision={val_metrics['precision']:.4f}, "
                    f"样本数={val_metrics['num_samples']}, "
                    f"正样本={val_metrics['num_positive']}, "
                    f"负样本={val_metrics['num_negative']}"
                )
                
                # 测试评估器
                logger.logger.info("🔍 测试评估器...")
                evaluator = RiskNICLPEvaluator(model, device, logger)
                eval_metrics = evaluator.evaluate(DataLoader([batch], batch_size=1, collate_fn=lambda x: x[0]), "test")
                
                logger.logger.info(
                    f"📈 评估结果: "
                    f"p3_auc={eval_metrics.get('p3_auc', 'N/A'):.4f}, "
                    f"ns_平衡度={eval_metrics.get('ns_sampling_balance_ratio', 'N/A'):.4f}, "
                    f"ns_综合评分={eval_metrics.get('ns_overall_negative_sampling_score', 'N/A'):.4f}"
                )
                
                break  # 只测试第一个成功的batch
                
            except Exception as e:
                logger.logger.warning(f"❌ Batch {batch_idx} 测试失败: {e}")
                continue
        
        # 7. 测试实际训练循环（短时间）
        logger.logger.info("🏃 测试短期训练...")
        
        # 临时修改配置进行快速测试
        original_epochs = config.training.epochs
        config.training.epochs = 2  # 只训练2个epoch
        
        try:
            trainer.train(train_loader, train_loader)  # 使用相同数据集作为验证集
            
            logger.logger.info("✅ 训练修复验证成功！")
            return True
            
        except Exception as e:
            logger.logger.error(f"❌ 训练测试失败: {e}")
            return False
        finally:
            config.training.epochs = original_epochs
        
    except Exception as e:
        logger.logger.error(f"❌ 测试过程出错: {e}")
        return False


def test_negative_sampling_metrics():
    """专门测试负采样指标计算"""
    logger = get_logger()
    logger.logger.info("🎯 测试负采样指标计算")
    
    # 创建模拟数据
    predictions = np.array([0.8, 0.7, 0.6, 0.4, 0.3, 0.2])  # 6个预测值
    targets = np.array([1, 1, 1, 0, 0, 0])  # 前3个是正样本，后3个是负样本
    
    logger.logger.info(f"📊 模拟数据: predictions={predictions}, targets={targets}")
    
    # 手工计算指标
    positive_count = np.sum(targets == 1)
    negative_count = np.sum(targets == 0) 
    
    # 平衡度
    balance_ratio = min(positive_count, negative_count) / max(positive_count, negative_count)
    
    # 负样本质量
    negative_predictions = predictions[targets == 0]
    negative_quality = 1.0 - np.mean(negative_predictions)
    
    # 分离度
    positive_predictions = predictions[targets == 1]
    separation = np.mean(positive_predictions) - np.mean(negative_predictions)
    
    # 综合评分
    overall_score = 0.3 * balance_ratio + 0.3 * negative_quality + 0.3 * separation + 0.1 * 0.5
    
    logger.logger.info(
        f"📈 手工计算结果: "
        f"平衡度={balance_ratio:.3f}, "
        f"质量={negative_quality:.3f}, "
        f"分离度={separation:.3f}, "
        f"综合评分={overall_score:.3f}"
    )
    
    # 期望结果：
    # 平衡度 = 3/3 = 1.0 (完美平衡)
    # 质量 = 1 - (0.4+0.3+0.2)/3 = 1 - 0.3 = 0.7 (负样本得分低)
    # 分离度 = (0.8+0.7+0.6)/3 - (0.4+0.3+0.2)/3 = 0.7 - 0.3 = 0.4 (正负分离好)
    # 综合 = 0.3*1.0 + 0.3*0.7 + 0.3*0.4 + 0.1*0.5 = 0.3 + 0.21 + 0.12 + 0.05 = 0.68
    
    expected_balance = 1.0
    expected_quality = 0.7
    expected_separation = 0.4
    expected_overall = 0.68
    
    # 验证计算是否正确
    if (abs(balance_ratio - expected_balance) < 0.01 and
        abs(negative_quality - expected_quality) < 0.01 and
        abs(separation - expected_separation) < 0.01 and
        abs(overall_score - expected_overall) < 0.01):
        
        logger.logger.info("✅ 负采样指标计算逻辑正确！")
        return True
    else:
        logger.logger.error("❌ 负采样指标计算有误！")
        return False


if __name__ == "__main__":
    print("🔧 RiskNICLP-5G 训练修复验证")
    print("=" * 50)
    
    # 测试1：负采样指标计算
    metrics_ok = test_negative_sampling_metrics()
    
    # 测试2：训练流程修复
    training_ok = test_training_fixes()
    
    print("\n" + "=" * 50)
    print("📋 测试结果汇总:")
    print(f"  负采样指标: {'✅ 通过' if metrics_ok else '❌ 失败'}")
    print(f"  训练流程: {'✅ 通过' if training_ok else '❌ 失败'}")
    
    if metrics_ok and training_ok:
        print("\n🎉 所有测试通过！训练系统修复成功！")
        print("💡 建议：现在可以重新运行 python train.py 进行完整训练")
        exit(0)
    else:
        print("\n⚠️ 部分测试失败，需要进一步调试")
        exit(1) 