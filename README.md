# RiskNICLP-5G: Risk-aware Node-Importance Causal Link Prediction for 5G Intrusion Detection

🚀 **Phase 2 基础版本已完成** - "能跑、能评、能扩展"的CPU优化模型实现

**最新更新 (2025-06-23)**: 图特征维度标准化修复完成，训练流程稳定运行

## 📋 项目概述

RiskNICLP-5G是一个风险感知的异质图神经网络模型，专门用于5G网络入侵检测中的因果链路预测。该模型将孤立的流量记录升级为具有5G语义和威胁传播的动态异质图，实现主动的5G网络入侵路径预测。

### 🎯 核心特性

- **异质图建模**: UE-Flow-Alert三层异质图结构
- **风险感知**: Risk-PPR算法结合威胁传播建模
- **物理启发解码**: RiskGravity解码器模拟引力模型
- **双尺度建模**: 细粒度(3s)和粗粒度(30s)时间窗口
- **CPU优化**: 针对CPU环境的高效实现
- **工程化**: 完整的MLOps流程和监控体系

## 🏗️ 项目结构

```
5g-nidd-project/
├── config/
│   └── baseline_config.yaml      # 基础配置文件
├── src/
│   ├── common/                   # 基础设施
│   │   ├── logger.py            # 统一日志监控
│   │   ├── config.py            # 多层配置管理
│   │   ├── reproducibility.py   # 可重现性保障
│   │   └── validators.py        # 数据验证框架
│   ├── data/                    # 数据处理
│   │   ├── feature_processor.py # 特征选择(112→45维)
│   │   ├── graph_dataset.py     # 图数据加载
│   │   └── cache_manager.py     # CSR缓存管理
│   ├── models/                  # 模型架构
│   │   ├── encoders.py          # 共享编码器(GAT+GraphSAGE+GRU)
│   │   ├── decoders.py          # RiskGravity解码器
│   │   ├── risk_integration.py  # 风险集成模块
│   │   ├── riskniclp_model.py   # 完整模型
│   │   └── training.py          # 渐进训练框架
│   └── evaluation/              # 评估模块
│       ├── metrics.py           # 安全导向指标
│       └── evaluator.py         # 评估器
├── data/processed/              # 预处理数据
│   ├── graphs/                  # 75个DGL异质图
│   └── metadata/               # 元数据和报告
├── train.py                     # 主训练脚本
├── requirements.txt             # 项目依赖
└── README.md                   # 本文件
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建Python环境
conda create -n riskniclp-env python=3.11
conda activate riskniclp-env

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch, dgl; print(f'PyTorch: {torch.__version__}, DGL: {dgl.__version__}')"
```

### 2. 数据准备

确认数据处理pipeline已完成，具备以下数据：

```bash
# 检查预处理数据
ls data/processed/graphs/        # 应包含75个.dgl文件
ls data/processed/metadata/      # 应包含各阶段报告
```

如果数据未准备好，请先运行数据处理pipeline (参考`docs/PIPELINE_EXECUTION_SUMMARY.md`)。

### 3. 基础训练

```bash
# 基础训练 - 使用默认配置
python train.py

# 使用自定义配置
python train.py --config config/baseline_config.yaml

# 调试模式 - 快速验证
python train.py --debug

# 指定实验名称
python train.py --experiment-name "my_baseline_experiment"

# 干运行 - 验证配置无误
python train.py --dry-run
```

### 4. 配置参数

核心参数可通过配置文件或命令行调整：

```bash
# 调整批量大小和学习率
python train.py --batch-size 8 --learning-rate 5e-4

# 使用不同设备
python train.py --device cpu

# 环境变量覆盖
export RISKNICLP_BATCH_SIZE=8
export RISKNICLP_MAX_EPOCHS=30
python train.py
```

## 📊 模型架构

### 核心组件

1. **特征处理**: 112维→45维特征选择 + 风险8-bin向量化
2. **共享编码器**: 
   - UE编码器: GAT + Risk-Bias注意力
   - Flow编码器: GraphSAGE + GRU时序建模
3. **RiskGravity解码器**: 物理启发的风险感知链路预测
4. **双尺度融合**: Late Fusion整合细粒度和粗粒度信息

### 训练策略

- **渐进训练**: 5epoch基础GNN → 10epoch风险感知
- **参数共享**: 细粗尺度编码器权重共享
- **梯度累积**: 4步累积实现有效batch=16
- **优化技术**: BF16精度 + 梯度检查点 + CSR缓存

## 📈 评估指标

### 优先级指标体系

**P0 实时安全**:
- Lead-Time: 提前预警时间
- MTTD (≤30s): 高风险链路平均检测时间

**P1 检测质量**:
- Risk-Weighted Precision/Recall
- AUPR: 精确率-召回率曲线下面积

**P2 方向性**:
- Directional Hit@K (≥0.90)
- Direction Error Rate (≤5%)

**P3 传统**:
- AUC, Hit@K

## 🔧 高级用法

### 自定义配置

```yaml
# config/my_config.yaml
model:
  hidden_dim: 128        # 增加模型容量
  num_heads: 8          # 更多注意力头
  
training:
  batch_size: 8         # 更大批量
  max_epochs: 50        # 更长训练
  
logging:
  enable_wandb: true    # 启用WandB追踪
```

### 监控和调试

```bash
# 启用TensorBoard
tensorboard --logdir logs/

# 查看训练日志
tail -f logs/RiskNICLP-5G.log

# 系统资源监控
htop  # 查看CPU/内存使用
```

### 断点续训

```bash
# 从检查点恢复
python train.py --resume logs/experiment_name/checkpoints/best_model.pt
```

## 📋 实验清单

### Phase 1: 基础验证
- [x] 配置管理系统
- [x] 数据处理pipeline
- [x] 模型架构实现
- [x] 训练框架搭建
- [x] 基础评估指标
- [x] 图特征维度标准化修复 (2025-06-23)

### Phase 2: 性能优化
- [ ] Risk-Bias注意力机制
- [ ] 渐进训练切换
- [ ] 风险分层负采样
- [ ] 系统性能监控

### Phase 3: 高级特性
- [ ] Cross-Scale Context
- [ ] 完整消融研究
- [ ] 可解释性分析
- [ ] 生产部署优化

## 🐛 故障排除

### 常见问题

**1. OOM错误**
```bash
# 减小批量大小
python train.py --batch-size 2

# 启用梯度检查点
# 在config中设置: training.enable_gradient_checkpointing: true
```

**2. 数据加载慢**
```bash
# 减少worker数量
export RISKNICLP_NUM_WORKERS=2
python train.py
```

**3. DGL图加载失败**
```bash
# 检查图文件完整性
python -c "import dgl; g = dgl.load_graphs('data/processed/graphs/hetero_graph_fine_t0.dgl'); print('OK')"
```

### 性能基准

在CPU环境下的预期性能：
- **训练速度**: ~2-3秒/epoch (batch_size=4)
- **内存使用**: ~2-4GB RAM
- **收敛时间**: ~10-15分钟 (20 epochs)

## 📚 文档

- [数据处理总结](docs/PIPELINE_EXECUTION_SUMMARY.md): 完整的6阶段pipeline执行报告
- [设计文档](docs/RiskNICLP-5G_Design.md): 详细的技术设计说明
- [基础实现](docs/RiskNICLP-5G_Baseline_Implementation.md): 基础版本实现框架

## 🤝 贡献

这是一个研究项目的基础实现。如需扩展或改进：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🎯 项目状态

**当前阶段**: Phase 1 - 基础版本实现完成 ✅

**下一步**:
- Phase 2: 性能增强和Risk-Bias机制完善
- Phase 3: 高级特性和Cross-Scale Context实现

---

💡 **提示**: 这是基础版本，专注于"能跑、能评、能扩展"。高级特性将在后续阶段逐步实现。 