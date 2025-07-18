#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目清理脚本
清理过时的测试文件、临时文件和不再使用的脚本

Created: 2025-07-17
Author: RiskNICLP-5G Team
"""

import os
import shutil
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_project():
    """清理项目中的过时文件"""
    
    logger.info("🧹 开始清理项目过时文件...")
    
    # 定义要清理的文件和目录
    files_to_remove = [
        # 过时的测试文件
        "quick_baseline_test.py",  # 已被test_graphsaint_improved_metrics.py替代
        "test_graphsaint_real_data.py",  # 已被改进版本替代
        "test_baseline_performance.py",  # 过时的基线测试
        "test_config_only.py",  # 配置测试，不再需要
        "test_implementation.py",  # 通用实现测试，过时
        "test_memory_optimization.py",  # 内存优化测试，已完成
        "test_minimal.py",  # 最小测试，不再需要
        "test_phase2_implementation.py",  # Phase2测试，已完成
        
        # 过时的调试和修复脚本
        "debug_training_steps.py",  # 调试脚本，问题已解决
        "diagnose_memory_issue.py",  # 内存诊断，问题已解决
        "correct_attack_prediction_fix.py",  # 攻击预测修复，已完成
        "quick_fix_memory.py",  # 快速内存修复，已完成
        "fix_positive_sample_ratios.py",  # 正样本比例修复，已完成
        
        # 过时的分析和评估脚本
        "analyze_sampling_quality.py",  # 采样质量分析，已完成
        "evaluate_sample_quality.py",  # 样本质量评估，已完成
        "resample_improved.py",  # 改进重采样，已集成
        
        # 过时的实验脚本
        "run_attack_prediction_experiment.py",  # 攻击预测实验，已重构
        "run_baseline_comparison.py",  # 基线对比，已有新版本
        "run_offline_sampling.py",  # 离线采样，已集成
        "run_sampling_test.py",  # 采样测试，已完成
        "validate_attack_prediction_redesign.py",  # 攻击预测重设计验证，已完成
        
        # 过时的日志文件
        "attack_prediction_experiment_20250710_190510.log",
        "attack_prediction_experiment_20250711_085730.log",
        "attack_prediction_validation.log",
        "sample_quality_report.json",
        
        # 过时的配置文件
        "config/attack_prediction_metrics.yaml",  # 已合并到主配置
        "config/risk_bias_optimized.yaml",  # 已合并到主配置
        "config/riskniclp_config_backup.yaml",  # 备份配置，不再需要
        "attack_prediction_validation_report.yaml",
    ]
    
    # 定义要清理的目录
    dirs_to_remove = [
        "logs/riskniclp_baseline_20250710_154256",  # 过时的日志目录
        "results/sampling_quality",  # 采样质量结果，已完成
        "results/sampling_reconstruction",  # 采样重构结果，已完成
        "data/csr_cache",  # CSR缓存，可重新生成
        "data/feature_cache",  # 特征缓存，可重新生成
        "data/risk_cache",  # 风险缓存，可重新生成
        "src/__pycache__",  # Python缓存
        "tests/__pycache__",  # 测试缓存
    ]
    
    # 定义要保留的重要文件（确保不误删）
    important_files = [
        "test_graphsaint_improved_metrics.py",  # 最新的GraphSAINT测试
        "src/evaluation/risk_link_metrics.py",  # 风险链路评估指标
        "src/baselines/graphsaint_baseline.py",  # GraphSAINT基线实现
        "config/riskniclp_config.yaml",  # 主配置文件
        "train.py",  # 主训练脚本
        "train_memory_optimized.py",  # 内存优化训练
        "train_offline.py",  # 离线训练
    ]
    
    removed_files = []
    removed_dirs = []
    kept_files = []
    
    # 清理文件
    for file_path in files_to_remove:
        full_path = Path(file_path)
        if full_path.exists():
            if file_path in important_files:
                logger.warning(f"⚠️ 跳过重要文件: {file_path}")
                kept_files.append(file_path)
                continue
            
            try:
                full_path.unlink()
                removed_files.append(file_path)
                logger.info(f"🗑️ 删除文件: {file_path}")
            except Exception as e:
                logger.error(f"❌ 删除文件失败 {file_path}: {e}")
        else:
            logger.debug(f"📝 文件不存在: {file_path}")
    
    # 清理目录
    for dir_path in dirs_to_remove:
        full_path = Path(dir_path)
        if full_path.exists() and full_path.is_dir():
            try:
                shutil.rmtree(full_path)
                removed_dirs.append(dir_path)
                logger.info(f"🗑️ 删除目录: {dir_path}")
            except Exception as e:
                logger.error(f"❌ 删除目录失败 {dir_path}: {e}")
        else:
            logger.debug(f"📝 目录不存在: {dir_path}")
    
    # 清理空目录
    empty_dirs_removed = []
    for root, dirs, files in os.walk(".", topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            try:
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    empty_dirs_removed.append(str(dir_path))
                    logger.info(f"🗑️ 删除空目录: {dir_path}")
            except Exception as e:
                logger.debug(f"跳过目录 {dir_path}: {e}")
    
    # 生成清理报告
    logger.info("\n" + "="*60)
    logger.info("🎉 项目清理完成！")
    logger.info("="*60)
    logger.info(f"📊 清理统计:")
    logger.info(f"   删除文件: {len(removed_files)} 个")
    logger.info(f"   删除目录: {len(removed_dirs)} 个")
    logger.info(f"   删除空目录: {len(empty_dirs_removed)} 个")
    logger.info(f"   保留重要文件: {len(kept_files)} 个")
    
    if removed_files:
        logger.info(f"\n🗑️ 已删除的文件:")
        for file in removed_files:
            logger.info(f"   - {file}")
    
    if removed_dirs:
        logger.info(f"\n🗑️ 已删除的目录:")
        for dir in removed_dirs:
            logger.info(f"   - {dir}")
    
    if kept_files:
        logger.info(f"\n✅ 保留的重要文件:")
        for file in kept_files:
            logger.info(f"   - {file}")
    
    logger.info("="*60)
    
    # 创建清理后的项目结构说明
    create_project_structure_doc()

def create_project_structure_doc():
    """创建清理后的项目结构说明"""
    
    structure_doc = """# RiskNICLP-5G 项目结构 (清理后)

## 📁 核心目录结构

```
RiskNICLP-5G/
├── 📊 数据处理
│   ├── data/
│   │   ├── processed/          # 处理后的数据
│   │   │   ├── graphs/         # DGL图文件
│   │   │   ├── metadata/       # 元数据和统计
│   │   │   └── offline_samples/ # 离线采样数据
│   │   └── raw/               # 原始数据
│   └── notebooks/             # 数据探索笔记本
│
├── 🧠 模型实现
│   └── src/
│       ├── baselines/         # 基线算法
│       │   ├── graphsaint_baseline.py
│       │   └── tgat_baseline.py
│       ├── models/            # RiskNICLP-5G模型
│       ├── evaluation/        # 评估指标
│       │   └── risk_link_metrics.py
│       ├── data/              # 数据加载器
│       ├── training/          # 训练逻辑
│       └── utils/             # 工具函数
│
├── ⚙️ 配置和脚本
│   ├── config/
│   │   └── riskniclp_config.yaml  # 主配置文件
│   ├── scripts/               # 数据处理脚本
│   └── tools/                 # 工具脚本
│
├── 🧪 测试和实验
│   ├── test_graphsaint_improved_metrics.py  # GraphSAINT改进测试
│   ├── train.py               # 主训练脚本
│   ├── train_memory_optimized.py
│   └── train_offline.py
│
├── 📊 结果和日志
│   ├── results/               # 实验结果
│   └── logs/                  # 运行日志
│
└── 📚 文档
    └── docs/                  # 项目文档
```

## 🎯 主要功能模块

### 1. 基线算法测试
- `test_graphsaint_improved_metrics.py` - 使用改进指标测试GraphSAINT

### 2. 模型训练
- `train.py` - 标准训练流程
- `train_memory_optimized.py` - 内存优化训练
- `train_offline.py` - 离线训练

### 3. 评估指标
- `src/evaluation/risk_link_metrics.py` - 专业的风险链路预测指标

### 4. 基线实现
- `src/baselines/graphsaint_baseline.py` - GraphSAINT基线
- `src/baselines/tgat_baseline.py` - TGAT基线

## 🚀 快速开始

1. **测试GraphSAINT基线**:
   ```bash
   python test_graphsaint_improved_metrics.py
   ```

2. **训练RiskNICLP-5G模型**:
   ```bash
   python train.py
   ```

3. **内存优化训练**:
   ```bash
   python train_memory_optimized.py
   ```

## 📈 评估指标说明

项目使用专业的风险链路预测指标：
- **Hit@K**: Top-K命中率
- **NDCG@K**: 归一化折扣累积增益
- **风险加权准确率**: 考虑风险权重的准确率
- **早期发现率**: Top 10%预测中的攻击发现率

这些指标比传统的准确率更适合评估风险链路预测任务。
"""
    
    with open("PROJECT_STRUCTURE.md", "w", encoding="utf-8") as f:
        f.write(structure_doc)
    
    logger.info("📚 已创建项目结构文档: PROJECT_STRUCTURE.md")

if __name__ == "__main__":
    cleanup_project()
