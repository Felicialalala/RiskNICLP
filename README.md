# RiskNICLP-5G 数据预处理方案

基于5G-NIDD数据集构建融合5G语义与威胁传播的动态图谱的完整数据预处理方案。

## 项目结构

```
5g-nidd-project/
├── notebooks/                    # Jupyter Notebook处理流程
│   ├── 01_data_exploration.ipynb          # Phase 1: 数据探索分析
│   ├── 02_time_feature_engineering.ipynb  # Phase 2: 时间标准化
│   ├── 03_5g_entity_extraction.ipynb      # Phase 3: 5G实体识别
│   ├── 04_temporal_windowing.ipynb        # Phase 4: 时间窗口分片
│   ├── 05_heterogeneous_graph_construction.ipynb  # Phase 5: 异质图构建
│   └── 06_risk_weight_mapping.ipynb       # Phase 6: 威胁权重映射
├── data/
│   ├── raw/                      # 原始数据
│   └── processed/                # 处理后数据
│       ├── graphs/              # 图数据存储
│       └── metadata/            # 元数据和质量报告
├── config.yaml                  # 配置文件
├── requirements.txt             # 依赖文件
└── README.md                    # 说明文档
```

## 环境配置

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动Jupyter：
```bash
jupyter notebook
```

## 处理流程

### Phase 1: 数据探索分析
- **目的**: 深入分析5G-NIDD数据集结构
- **输入**: `data/raw/BTS1_BTS2_fields_preserved.csv`（1,215,890条记录）
- **输出**: 数据质量报告

### Phase 2: 时间标准化与特征工程
- **目的**: 统一时间戳格式，提取5G语义特征
- **处理**: MM:SS.s格式 → Unix时间戳
- **输出**: 标准化时间戳和特征

### Phase 3: 5G网络实体识别与分类
- **目的**: 基于IP地址模式识别UE、gNB、Core等实体
- **规则**: 
  - UE: 10.155.15.x
  - Core: 10.41.150.x
  - External: 109.105.x, 142.250.x等
- **输出**: 实体分类标签

### Phase 4: 动态时间窗口分片
- **目的**: 实现双尺度时间窗设计
- **细粒度**: 3s窗口/1s步长（实时告警）
- **粗粒度**: 30s窗口/10s步长（模式分析）
- **输出**: 时间窗口索引

### Phase 5: UE-Flow-Alert多层异质图构建
- **目的**: 构建三层异质图结构
- **节点类型**: UE, Flow, Alert
- **边类型**: initiates, triggers, temporal_adj
- **输出**: DGL异质图对象

### Phase 6: 威胁权重映射与Risk-PPR计算
- **目的**: 为节点注入威胁权重，计算风险驱动的重要度
- **威胁等级**: ICMPFlood(0.3) → HTTPFlood(0.9)
- **输出**: 节点风险权重和PPR重要度

## 数据集特征

- **总记录数**: 1,215,890条
- **字段数**: 112个特征维度
- **攻击类型**: 9种（包含正常流量）
- **协议分布**: TCP(89.15%), UDP(6.14%), ICMP(4.16%), SCTP(0.53%)

## 配置说明

主要配置参数在 `config.yaml` 中：
- 时间窗口大小和步长
- 实体分类规则
- 威胁权重映射
- Risk-PPR参数

## 预期输出

1. **数据质量报告**: JSON格式的完整性检查结果
2. **实体分类结果**: 网络拓扑实体映射
3. **时间窗口索引**: 双尺度时间片段划分
4. **异质图对象**: DGL格式的多层图结构
5. **风险权重向量**: 节点威胁评估结果
6. **可视化图表**: 数据分布和质量分析图

## 后续工作支持

该预处理方案为后续RiskNICLP-Sec模型训练提供：
- 标准化的图数据格式
- 威胁语义增强的节点特征
- 时序动态的图快照序列
- 5G网络语义的异质图结构 