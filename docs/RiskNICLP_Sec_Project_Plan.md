# RiskNICLP-Sec 项目计划文档
项目环境：conda activate riskniclp-env

## 1. 项目目标（Problem Statement）

**构建一个融合 5 G 语义与威胁传播的动态图谱，并在其上训练 RiskNICLP-Sec 模型，实现对 5 G 网络入侵因果链路的 _精确预测_（Link Prediction）。**  
目标是在攻击真正发生前 **10–60 s** 给出最可能的后续入侵边，辅助 SOC 人员提前阻断攻击或分配高优先级工单。
其中，NICLP的理念来自我之前做的工作，详见：docs/无线数据知识图谱的构建、解析与应用v4.md。

---

## 2. 技术路线（High-Level Technical Route）

```
                 ┌──────────────────┐
                 │   5G-NIDD CSV    │  (标签+112流量特征)
                 └────────┬─────────┘
                          ▼
  ① 预处理&窗口切片  (Δt=1 s)        ➜  data/processed/
                          ▼
  ② 语义补齐 & 异质图构建           ➜  graph_build.py
     • UE / gNB / Slice / Flow / Alert
                          ▼
  ③ 风险权重注入 & Risk-PPR        ➜  risk_weight.py
     • wᵢ  (CVSS/ATT&CK)  +  rᵢ  (PPR)
                          ▼
  ④ RiskNICLP-Sec 训练             ➜  model_train.py
     • 双编码器 + 风险解码器
                          ▼
  ⑤ 链路预测评估 & 可解释化         ➜  evaluate.py
```

---

## 3. 实施步骤（Step-by-Step Plan）

| # | 步骤名称 | 关键工作 & 说明 | 产出 / 代码位置 |
|---|---------|----------------|-----------------|
| S1 | 环境与依赖准备 | Python 3.11，安装 PyTorch、PyG、DGL 等依赖 | `venv/`, `logs/env_check.log` |
| S2 | 数据校验 & EDA | 分析九类攻击分布、节点数量等 | `notebooks/EDA.ipynb` |
| S3 | Session 聚合 | 聚合五元组为会话，规范时间戳 | `preprocess.py`, `session_df.parquet` |
| S4 | 时间滑窗分片 | Δt 滑窗划分窗口，生成索引 | `window_index.csv` |
| S5 | 5G 语义映射 | 提取 UE/gNB/Slice 等实体 | `graph_build.py`, `graph_schema.md` |
| S6 | 异质图构建 | 构建多层节点和边的动态图 | `data/processed/G_*.dgl` |
| S7 | 风险权重映射 | CVSS/ATT&CK → wᵢ, PPR → rᵢ，组合为 riskᵢ | `risk_weight.py`, `risk_dist.png` |
| S8 | 模型实现 | 双编码器 + 风险解码器 | `src/model/` |
| S9 | 模型训练 | 训练配置与超参搜索 | `model_train.py`, `outputs/` |
| S10 | 链路预测评估 | 计算 AUC、Path-Hit@k 等指标 | `evaluate.py`, `metrics.json` |
| S11 | 可解释化可视化 | PyVis 展示高风险链 | `outputs/graph_viz.html` |
| S12 | 消融实验 | No-risk/No-time 对比分析 | `ablation_table.csv` |
| S13 | 论文撰写 | 撰写 Methods/Results | `draft_paper.tex` |
| S14 | 项目验收与展望 | 答辩、图LLM预研 | `future_plan.md` |

---

## 4. 参数配置与默认值

| 参数 | 默认 | 配置路径 | 说明 |
|------|------|---------|------|
| 滑窗 Δt | 1 s | `config.yaml: window_size_s` | 影响动态图快照粒度 |
| α (risk-mix) | 0.7 | `config.yaml: risk_alpha` | 风险融合权重 |
| 隐藏维度 | 128 | `config.yaml: hidden_dim` | GNN 隐层大小 |
| 采样邻居数 | [15, 10] | `config.yaml: samples_per_layer` | 两层采样 |
| 负样本率 | 3 | `config.yaml: neg_ratio` | 1:3 (pos:neg) |

---

## 5. 创新亮点

- **三层异质图建模**：UE / Flow / Alert 融合 5G 网络结构。
- **Risk-aware PPR**：结合结构传播与威胁标签建模节点风险。
- **因果链预测结构**：主动/被动依赖分流，保留攻击路径因果。
- **Lead-Time 评估指标**：实际反映提前预警能力。
- **可复现开源标准化流程**：促进安全图建模研究复现。

---


6. 5g-nidd数据集分析

   对捕获的数据进行一系列后处理步骤，包括移除GTP层、生成网络流、数据聚合和特征选择。使用Argus工具将数据从包格式转换为流格式，并生成112个特征。然后使用Pearson相关系数和ANOVA F-score进行特征选择，最终选择最重要的特征。

   **攻击类型和工具**：评估了几种常见的DoS攻击（如ICMP洪水、UDP洪水、SYN洪水、HTTP洪水和慢速DoS攻击）和端口扫描（如SYN扫描、TCP连接扫描和UDP扫描）。使用Hping3工具和Python脚本执行这些攻击。

   数据集中一共有 **8种攻击类型** 加上 **良性流量**，总共是 **9种类型。**

   - 具体的攻击类型
     1. HTTP Flood：通常表现为大量的HTTP请求，可能会导致服务器资源耗尽。
     2. ICMP Flood：主要通过发送大量的ICMP请求来淹没目标，导致网络拥塞。
     3. SYN Flood：通过发送大量的SYN包来占用目标服务器的连接资源，导致拒绝服务。
     4. SYN Scan
     5. Slowrate DoS
     6. TCP Connect Scan
     7. UDP Flood
     8. UDP Scan
        - 数据结构和差异
          1. 数据包特征：
             - 不同类型的攻击可能会使用不同的协议（如TCP、UDP、ICMP），这会影响数据包的特征，如源端口、目的端口、协议类型等。
          2. 流量速率：
             - 各种攻击的流量速率可能会有所不同。例如，HTTP Flood可能会在短时间内产生大量的请求，而Slowrate DoS则可能会以较慢的速率发送请求，以规避检测。
          3. 攻击持续时间：
             - 不同类型的攻击可能会有不同的持续时间，某些攻击可能会持续较长时间，而其他攻击可能是短暂的脉冲式攻击。
          4. 目标端口和协议：
             - 不同的攻击类型可能会针对不同的端口和协议，这会影响数据集中记录的流量特征。

尽管数据结构在基本特征上相似，但由于攻击类型的不同，具体的流量特征和行为模式会有显著差异。这些差异对于入侵检测系统的训练和评估是非常重要的，因为它们可以帮助模型识别和区分不同类型的攻击。

# 详细数据分析
总量统计：
总流记录数：1,215,890条（去除表头）
BS1数据：728,316条记录
BS2数据：487,574条记录
字段总数：112个特征维度

**数据文件规模：**

- Combined.csv: 1,215,891行（263MB）
- Encoded.csv: 1,215,891行（467MB）
- BTS1_BTS2_fields_preserved.csv: 1,215,891行（451MB）
- BTS_1.csv: 728,316行（135MB）
- BTS_2.csv: 487,574行（93MB）

**攻击类型分布：**

从BS1和BS2各自的attack_csv目录中，我观察到包含九种攻击类型：

1. SSH攻击
2. SYN Flood
3. SYN Scan
4. Slowloris
5. TCP Connect
6. Torshammer
7. UDP Flood
8. UDP Scan
9. Golden Eye
10. ICMP Flood

**数据特征字段：**

观察到数据包含大约112个特征字段，包括：

- 基础网络字段：Seq, Dur, RunTime, Mean, Sum, Min, Max

- 协议相关：Proto, sTos, dTos, sTtl, dTtl

- 流量统计：TotPkts, SrcPkts, DstPkts, TotBytes, SrcBytes, DstBytes

- 地址信息：SrcAddr, DstAddr, Sport, Dport

- 标签信息：Label, Attack Type, Attack Tool

  

## **数据集分析情况**

从 data/raw/BTS1_BTS2_fields_preserved.csv 文件分析可以观察到：

### **1. 攻击类型分布**

- **良性流量**：477,737 条记录 (39.2%)
- **UDPFlood**：457,340 条记录 (37.5%)
- **HTTPFlood**：140,812 条记录 (11.5%)
- **SlowrateDoS**：73,124 条记录 (6.0%)
- **TCPConnectScan**：20,052 条记录 (1.6%)
- **SYNScan**：20,043 条记录 (1.6%)
- **UDPScan**：15,906 条记录 (1.3%)
- **SYNFlood**：9,721 条记录 (0.8%)
- **ICMPFlood**：1,155 条记录 (0.1%)

### **2. 攻击工具分布**

- **Benign**：477,737 条记录
- **Hping3**：468,216 条记录
- **Goldeneye**：140,812 条记录
- **Torshammer**：59,870 条记录
- **Nmap**：56,001 条记录
- **Slowloris**：13,254 条记录

### **3. 关键字段特征**

- **时间字段**：StartTime、LastTime 格式为 MM:SS.s（如 "30:02.6"）
- **网络五元组**：SrcAddr、DstAddr、Proto、Sport、Dport
- **5G相关字段**：包含协议类型（icmp、udp、tcp、sctp）和QoS标记（sTos、dTos）
- **流量统计**：TotPkts、SrcPkts、DstPkts、TotBytes等112个流量特征

2. 网络拓扑特征发现
IP地址分布模式：
核心内网IP：10.41.150.x系列（基站核心网）
UE网段：10.155.15.x系列（用户设备网段）
外部服务IP：109.105.109.x、142.250.74.x（Google服务）、193.166.161.x等
协议分布：
TCP：89.15%（主要应用层攻击）
UDP：6.14%（洪泛攻击为主）
ICMP：4.16%
SCTP：0.53%（5G信令协议）
ARP：0.01%
3. 5G语义特征识别
控制面vs用户面识别关键字段：
Proto字段中的SCTP协议指示5G信令流量
sTos/dTos字段包含QoS标记（cs0, ef, af11等）
Sport/Dport包含标准5G端口信息
IP地址段体现了UE-gNB-Core的层次结构