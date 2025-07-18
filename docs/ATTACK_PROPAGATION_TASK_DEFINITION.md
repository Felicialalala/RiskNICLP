# RiskNICLP-5G 攻击链路预测任务定义

## 📋 任务概述

### 核心任务转换
**原任务**: 异常检测 (Anomaly Detection)  
**新任务**: 攻击链路预测 (Attack Propagation Prediction)

### 任务背景
5G-NIDD数据集特性分析表明：
- **攻击流量占比**: 60.8% (738,153条)
- **良性流量占比**: 39.2% (477,737条)
- **环境特征**: 攻击密集的实验环境，而非正常网络

**结论**: 这是一个攻击预测任务，而非传统的异常检测任务。

## 🎯 任务定义

### 输入输出规范
```
输入: 观察窗口 W_t 内的网络图状态
- G_t = (V, E, X, R)
- V: 节点集合 {UE, Flow, Alert}
- E: 边集合 {initiates, triggers, temporal_adj}  
- X: 节点特征 (流量特征、设备属性)
- R: 风险分数 (基于历史攻击模式)

输出: 预警窗口 W_{t+Δ} 内的攻击边预测
- Δ ∈ {10s, 30s, 60s} (多时间尺度预测)
- 预测边类型:
  1. 攻击发起边 (UE → Flow): 新的恶意用户行为
  2. 告警触发边 (Flow → Alert): 攻击被检测系统发现
  3. 攻击传播边 (Flow → Flow): 攻击在时序上的演化
```

### 核心预测目标
1. **攻击发起预测**: 识别即将发起攻击的UE节点
2. **告警触发预测**: 预测哪些Flow会触发安全告警
3. **攻击传播预测**: 预测攻击在时间维度上的演化路径

## 📊 样本定义

### 正样本 (25-35%)
```yaml
attack_initiation:    # UE → Flow (30%)
  description: "恶意用户发起新的攻击流量"
  criteria:
    - 新出现的UE-Flow连接
    - Flow具有攻击特征 (UDPFlood, HTTPFlood等)
    - 时间窗口内首次出现
    
alert_triggering:     # Flow → Alert (35%)
  description: "攻击流量触发安全告警"
  criteria:
    - Flow-Alert连接在预测窗口内新出现
    - Alert类型与Flow攻击类型匹配
    - 风险分数超过告警阈值
    
attack_propagation:   # Flow → Flow (25%)
  description: "攻击在时间维度上的传播演化"
  criteria:
    - 时序上的Flow-Flow连接
    - 攻击类型一致或升级
    - 风险分数呈递增趋势
```

### 负样本 (65-75%)
```yaml
semantic_impossible:  # 语义约束违反
  description: "违反5G网络语义的连接"
  examples:
    - Alert → UE (告警不能反向影响用户)
    - 低风险UE → 高风险Alert (风险传播不合理)
    
temporal_violations:  # 时序逻辑违反
  description: "违反攻击时序演化规律"
  examples:
    - 未来Flow → 过去Flow (时序倒置)
    - 告警先于攻击 (因果关系错误)
    
risk_inconsistent:    # 风险模式不一致
  description: "风险传播模式不符合攻击特征"
  examples:
    - 低风险节点触发高风险告警
    - 攻击强度突然下降
```

## 🎯 评估指标

### 主要指标 (替代AUC)
```yaml
attack_prediction_accuracy:
  description: "攻击边预测准确率"
  formula: "TP / (TP + FP)"
  target: "> 0.80"

early_warning_effectiveness:
  description: "提前预警有效性"
  formula: "成功预警攻击数 / 总攻击数"
  target: "> 0.75"
  constraint: "预警时间 > 30s"

attack_chain_completeness:
  description: "攻击链完整性召回率"
  formula: "完整预测的攻击链 / 总攻击链"
  target: "> 0.70"

threat_escalation_precision:
  description: "威胁升级预测精度"
  formula: "正确预测的威胁升级 / 预测的威胁升级"
  target: "> 0.85"
```

### 运营指标
```yaml
false_alarm_rate:
  description: "误报率 (SOC关键指标)"
  formula: "FP / (FP + TN)"
  target: "< 0.15"

attack_type_discrimination:
  description: "攻击类型区分能力"
  breakdown: ["UDPFlood", "HTTPFlood", "SlowrateDoS"]
  measurement: "每种攻击类型的F1-Score"

temporal_prediction_stability:
  description: "时序预测稳定性"
  measurement: "连续时间窗口预测结果的相关性"
  target: "> 0.85"
```

## 🔄 与异常检测的关键差异

| 维度 | 异常检测 | 攻击链路预测 |
|------|----------|-------------|
| 数据假设 | 正常为主(95%+) | 攻击密集(60%+) |
| 正样本比例 | 1-5% | 25-35% |
| 预测目标 | 识别异常 | 预测攻击传播 |
| 时间维度 | 静态检测 | 动态预测 |
| 评估重点 | AUC/AUPR | 预警效果/链路完整性 |
| 业务价值 | 发现异常 | 提前防护 |

## 🚀 实施优势

1. **符合数据特性**: 正样本比例与实际攻击密度匹配
2. **业务导向**: 关注SOC实际需求的预警能力
3. **时序建模**: 充分利用攻击的时序传播特性
4. **可解释性**: 攻击链路比异常模式更易解释
5. **实用价值**: 提前预警比事后检测更有价值

## 📝 实施建议

### 立即行动
1. 修改配置文件中的正样本比例设置
2. 更新评估指标计算逻辑
3. 调整损失函数权重

### 短期优化  
1. 实现AttackPropagationSampler
2. 重新设计评估脚本
3. 验证新任务定义的有效性

### 中期完善
1. 对比新旧任务定义的性能
2. 攻击类型专门化建模
3. 可解释性分析和可视化
