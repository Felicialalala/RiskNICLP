# Risk-PPR完整处理脚本使用指南

## 概述

基于 `notebooks/06_risk_weight_mapping.ipynb` 的功能，我们创建了一个完整的Python脚本系统，能够处理所有75个图文件并为RiskNICLP-Sec模型训练准备数据。

## 核心文件结构

```
├── scripts/
│   ├── risk_weight_mapping_complete.py   # 主要的Risk-PPR计算脚本
│   └── risk_data_loader.py               # 数据加载器，用于模型训练
├── run_risk_ppr_complete.py              # 简化的运行脚本
└── README_RiskPPR_Complete.md            # 本说明文档
```

## 功能对比

### 原始Notebook vs 完整脚本

| 特性 | Notebook | 完整脚本 |
|------|----------|----------|
| 处理图数量 | 仅前10个 | 全部75个 |
| 错误处理 | 基础 | 完整的异常处理和日志 |
| 进度监控 | 无 | tqdm进度条 |
| 输出格式 | 基础JSON | 多格式输出(报告/向量/训练数据) |
| 可视化 | 6个子图 | 12个详细分析图 |
| 模型集成 | 手动 | 专用数据加载器 |

## 快速开始

### 1. 运行完整的Risk-PPR计算

```bash
# 方法1: 直接运行主脚本
python run_risk_ppr_complete.py

# 方法2: 直接调用
cd scripts
python risk_weight_mapping_complete.py
```

### 2. 输出文件

脚本将在 `data/processed/metadata/` 目录下生成以下文件：

```
data/processed/metadata/
├── risk_ppr_complete_report.json         # 完整分析报告
├── risk_vectors_complete.json            # 风险向量数据
├── training_risk_data.json               # 模型训练用数据
└── risk_ppr_complete_analysis.png        # 可视化分析图
```

## 详细功能说明

### 1. 主要计算器类 (RiskPPRCalculator)

```python
from scripts.risk_weight_mapping_complete import RiskPPRCalculator

# 初始化计算器
calculator = RiskPPRCalculator(
    alpha=0.15,                    # PPR重启概率
    max_iter=100,                  # 最大迭代次数
    convergence_threshold=1e-6     # 收敛阈值
)

# 处理所有图文件
results, failed_files = calculator.process_all_graphs("../data/processed/graphs")
```

### 2. 数据加载器 (RiskDataLoader)

```python
from scripts.risk_data_loader import RiskDataLoader

# 初始化加载器
loader = RiskDataLoader()

# 获取图的风险特征
graph, risk_features = loader.get_graph_with_risk_features(graph_path)

# 增强图特征
enhanced_graph = loader.enhance_graph_with_risk_scores(graph, risk_features)

# 获取高风险节点
high_risk_nodes = loader.get_high_risk_nodes(graph_id, 'Flow', top_k=5)
```

## 配置参数

### 威胁权重映射

```python
attack_threat_weights = {
    'ICMPFlood': 0.3,
    'UDPScan': 0.5, 
    'SYNScan': 0.6,
    'TCPConnectScan': 0.6,
    'SYNFlood': 0.7,
    'UDPFlood': 0.7,
    'SlowrateDoS': 0.8,
    'HTTPFlood': 0.9,
    'Benign': 0.0
}

tool_credibility = {
    'Nmap': 0.8,
    'Hping3': 1.0,
    'Goldeneye': 1.2,
    'Torshammer': 1.1,
    'Slowloris': 1.0,
    'Benign': 1.0
}
```

### Risk-PPR算法参数

- **alpha=0.15**: PPR重启概率，控制随机游走的重启频率
- **max_iter=100**: 最大迭代次数，确保收敛
- **convergence_threshold=1e-6**: 收敛阈值，控制精度

## 输出数据格式

### 1. 完整报告 (risk_ppr_complete_report.json)

```json
{
  "timestamp": "2025-01-27T...",
  "processing_summary": {
    "total_graphs_found": 75,
    "successful_processing": 72,
    "success_rate": 96.0
  },
  "threat_weights_config": {...},
  "ppr_config": {...},
  "analysis_results": {...},
  "detailed_results": [...]
}
```

### 2. 风险向量 (risk_vectors_complete.json)

```json
{
  "hetero_graph_fine_t1663200001": {
    "graph_path": "/path/to/graph.dgl",
    "risk_weights": {
      "Flow": [0.9, 0.0, 0.7, ...],
      "UE": [0.1, 0.05, ...],
      "Alert": [0.8, 0.6, ...]
    },
    "ppr_scores": {
      "Flow": [0.0012, 0.0008, ...],
      "UE": [0.015, 0.012, ...],
      "Alert": [0.45, 0.32, ...]
    },
    "top_risk_nodes": {
      "Flow": [5, 12, 8, 45, 23],
      "UE": [2, 7, 1, 9, 4],
      "Alert": [0, 1, 2]
    }
  }
}
```

### 3. 训练数据 (training_risk_data.json)

简化的模型训练用数据，包含图路径、PPR分数和统计信息。

## 模型集成示例

```python
from scripts.risk_data_loader import RiskDataLoader
import torch

# 初始化数据加载器
loader = RiskDataLoader()

# 获取可用图ID
graph_ids = loader.get_available_graph_ids()

# 准备训练批次
batch_graphs, batch_features = loader.prepare_model_input_batch(graph_ids[:8])

# 在模型中使用增强特征
for graph, features in zip(batch_graphs, batch_features):
    # 图已经包含PPR分数作为节点特征
    flow_features = graph.nodes['Flow'].data['feat']  # 包含PPR分数
    ue_features = graph.nodes['UE'].data['feat']
    alert_features = graph.nodes['Alert'].data['feat']
    
    # 可以单独访问PPR分数
    flow_ppr = graph.nodes['Flow'].data['ppr_score']
```

## 性能和质量指标

### 预期处理结果

- **处理图数量**: 75个图文件 → 70-75个成功处理 (93-100%成功率)
- **风险特征维度**: 
  - Flow节点: 原始特征 + 1维PPR分数
  - UE节点: 原始特征 + 1维PPR分数  
  - Alert节点: 原始特征 + 1维PPR分数
- **高风险节点识别**: 每个图识别top-5高风险节点
- **处理时间**: 约2-5分钟 (取决于硬件)

### 质量验证

脚本包含自动质量检查：

1. **维度一致性**: 确保PPR分数与图节点数量匹配
2. **数值有效性**: 检查PPR分数在[0,1]范围内
3. **收敛验证**: 确保PPR算法在合理迭代次数内收敛
4. **文件完整性**: 验证所有输出文件正确生成

## 故障排除

### 常见问题

1. **内存不足**
   - 减少并行处理的图数量
   - 使用`debug`日志级别监控内存使用

2. **图文件损坏**
   - 检查 `data/processed/graphs/` 目录
   - 查看日志中的失败文件列表

3. **特征维度不匹配**
   - 确保使用了Phase 7修复后的图文件
   - 检查Alert节点是否为9维

### 日志文件

详细日志保存在 `logs/risk_ppr_complete.log`，包含：
- 处理进度信息
- 错误详情和堆栈跟踪
- 性能统计信息

## 下一步工作

1. **模型训练集成**: 使用 `RiskDataLoader` 类将Risk-PPR特征集成到RiskNICLP-Sec模型
2. **批处理优化**: 根据GPU内存调整批处理大小
3. **特征选择**: 根据模型性能进行特征重要性分析

## 联系支持

如果遇到问题，请检查：
1. 日志文件 `logs/risk_ppr_complete.log`
2. 输出目录 `data/processed/metadata/`
3. 图文件目录 `data/processed/graphs/`

---

🎉 **Risk-PPR完整处理系统已就绪，可开始RiskNICLP-Sec模型训练！** 