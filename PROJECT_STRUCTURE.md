# RiskNICLP-5G 项目结构 (清理后)

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
