# RiskNICLP-5G 项目细化设计文档

## 一、数据理解：需深入分析的字段要点

| 目标 | 关键字段 | 需确认的细节 | 目的 |
|------|---------|---------------|------|
| **终端识别 (UE)** | SrcAddr, DstAddr, Sport, Dport, Proto | 是否为专网 IP、是否包含 GTP 协议外层 IP | 区分 UE / gNB / Core 节点类型 |
| **链路时间序列** | StartTime, LastTime, Dur, RunTime | 时间格式与时区 | 滑动窗口图快照划分依据 |
| **5G 语义提取** | Proto, sTtl, dTtl, sTos, dTos | 是否含 GTP-U、UDP2152、SIP 等 | 区分核心网、控制面、用户面信息 |
| **流量强度/方向性** | SrcPkts, DstPkts, TotBytes | 与攻击类型的耦合度 | 威胁传播路径度量基础 |
| **攻击标签分析** | Label, Attack Type, Attack Tool | 多标签分布/类别不均衡情况 | 后续风险建模与负采样规则 |

建议：使用 Pandas/Polars 构建特征分布、缺失率、攻击字段的热图对比。

## 二、图建模设计：UE-Flow-Alert 多层异质图

| 层级 | 节点 | 边类型 | 属性建议 |
|------|------|--------|---------|
| **UE 层** | UE IP 或 IMSI（需推断） | UE → Flow (`Initiates`) | 包数、字节、发起时间、风险历史等 |
| **Flow 层** | 五元组 + 时间窗 | Flow ↔ Flow (`Temporal-Adj`) | 协议字段、速率、时长、Alert关联 |
| **Alert 层** | Attack Type, Tool | Flow → Alert (`Triggers`) | CVSS 分值、攻击家族、工具名称 |
| **gNB/Core 层** | gNB ID, UPF, AMF（若可提取） | UE → gNB、gNB → Core | 可选添加地理/拓扑属性 |

- 滑动窗口：推荐窗口长度 5/10/60s，步长一半，生成动态图序列 `𝒢_t`。
- 异质图存储：DGL HeteroGraph 或 PyG HeteroData 建模，多关系类型与元路径设计需提前确定。

## 三、节点威胁建模：Risk-PPR 风险传播初始化

1. **攻击类型 → 威胁分映射建议**：

| 攻击类型 | 威胁分值 (0-1) |
|----------|----------------|
| ICMP / UDP Flood | 0.4 |
| SYN Flood / Scan | 0.6 |
| SSH / TCP Connect | 0.7 |
| Slowloris / Torshammer | 0.8 |
| GoldenEye | 0.9 |

2. **节点威胁基数计算**：
   $$ r_i = 1 - \prod_{j}(1 - \text{score}_j) $$

3. **Risk-PPR 执行参数**：
   - 初始向量 α 为节点风险分归一化；
   - 转移系数 β = 0.85；
   - 可分别对 UE 层与 Flow+Alert 层建图计算 risk-aware importance。

输出：每个节点的 `risk_ppr_i`，用于嵌入初始化、注意力偏置或融合特征。

## 四、模型结构：RiskNICLP-Sec 设计

```
UE Encoder (GAT)     Flow Encoder (GraphSAGE)
       │                   │
       └──────Concat/Hadamard──────┐
                                  ↓
                Risk-Aware Decoder (G-Physics Inspired)
                                  ↓
              Direction-aware Link Probability (u→v)
```

- **UE Encoder**：对 UE→Flow 的邻接图执行 GAT，使用 risk_ppr_i 作为 attention bias。
- **Flow Encoder**：对 Flow→Alert 及时序邻居执行 GraphSAGE 聚合。
- **解码器设计**：
   - 受万有引力启发：
     $$ p_{uv} = \sigma \left( G \cdot \frac{m_u \cdot m_v}{\|h_u - h_v\|^2} \right) $$  
     其中 \( m_u, m_v \) 为从嵌入中学习的质量值。

- **训练目标**：
  - 存在性预测：二分类损失；
  - 方向性预测：Margin Ranking Loss，保证 u→v 得分高于 v→u。

## 五、评估方案与下游任务指标

| 维度 | 指标 | 描述 |
|------|------|------|
| 静态链路预测 | AUC, AP, Hits@K | 快照图中边的召回与排序能力 |
| 方向预测准确率 | Accuracy, F1-dir | 预测边方向是否正确 |
| 动态攻击路径回溯 | Path Recall@L | 预测未来 L 跳攻击传播路径 |
| 高风险节点识别 | Top-N Hit Rate | 预测风险前 N UE 中真实受害比例 |

- **对比基线**：Node2Vec, GCN-LP, GraphSAGE-LP, DGI, RGAT-LP；
- **消融实验**：
  - 移除 risk-PPR；
  - 替换为单编码器；
  - 使用传统解码器如 DistMult 替代。

## 六、实现建议与资源配置

| 项目 | 建议 |
|------|------|
| 数据加载 | 使用 Polars / PyArrow 读取 CSV → Parquet 缓存 |
| 图结构存储 | DGL / PyG 配合 SparseTensor 构建多层图 |
| 训练设备 | RTX 4060 支持 GNN 小图训练，内存不够可尝试 GraphSAINT |
| 实验管理 | Hydra + Weights & Biases 实现超参调度与实验记录 |

## 七、开发 Checklist（建议顺序）

1. [ ] 特征字段分布可视化与缺失值检查；
2. [ ] UE / gNB 映射规则建立；
3. [ ] 攻击类别 → 风险分映射表设定；
4. [ ] 滑动窗口划分逻辑与数据拆分；
5. [ ] Risk-PPR 脚本离线实现与验证；
6. [ ] 模型代码骨架搭建（异质图 → 编码器 → 解码器）；
7. [ ] 评估指标与对比 baseline 实现；
8. [ ] 消融与 ablation 实验配置；
9. [ ] 实验日志、论文图表生成自动化脚本。


