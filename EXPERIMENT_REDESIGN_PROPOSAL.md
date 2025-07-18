# RiskNICLP-5G 攻击链路预测实验设计方案

## 🎯 任务重新定义

### 核心任务转换
**从**: 异常检测 (Anomaly Detection)
**到**: 攻击链路预测 (Attack Propagation Prediction)

### 当前设计的根本问题
1. **数据集误解**: 将5G-NIDD(60.8%攻击流量)当作正常网络(1-5%异常)处理
2. **任务定义错误**: 异常检测 vs 攻击传播预测的混淆
3. **评估指标不匹配**: 用异常检测指标评估攻击预测任务
4. **正样本稀缺**: 3-5%正样本比例无法反映攻击密集环境的真实特性

### 5G-NIDD数据集真实特性分析
- **良性流量**: 477,737条 (39.2%) - 基线对比流量
- **攻击流量**: 738,153条 (60.8%) - 主要研究对象
  - UDPFlood: 37.5% (主导攻击类型)
  - HTTPFlood: 11.5% (次要攻击类型)
  - SlowrateDoS: 6.0% (持续性攻击)
  - 其他攻击: 5.5% (多样化攻击向量)

**关键洞察**: 这是一个"攻击密集的实验环境"，而非"正常网络中的异常检测场景"

## 🔄 攻击链路预测任务设计

### 核心任务定义
```
任务目标: 基于当前网络状态，预测未来时间窗口内的攻击传播路径

输入: 观察窗口 W_t 内的网络图状态 G_t = (V, E, X, R)
- V: 节点集合 (UE, Flow, Alert)
- E: 边集合 (initiates, triggers, temporal_adj)
- X: 节点特征 (流量特征、设备属性等)
- R: 风险分数 (基于历史攻击模式)

输出: 预警窗口 W_{t+Δ} (Δ=10/30/60s) 内新出现的攻击边
1. 攻击发起边 (UE → Flow): 新的恶意用户行为
2. 告警触发边 (Flow → Alert): 攻击被检测系统发现
3. 攻击传播边 (Flow_t → Flow_{t+1}): 攻击在时序上的演化
```

### 正负样本重新定义
```yaml
positive_samples:
  definition: "未来窗口中确实发生的攻击相关边"
  categories:
    attack_initiation:    # UE → Flow
      description: "恶意用户发起新的攻击流量"
      expected_ratio: 30%  # 基于UDPFlood等攻击的高发频率

    alert_triggering:     # Flow → Alert
      description: "攻击流量触发安全告警"
      expected_ratio: 35%  # 基于检测系统的响应率

    attack_propagation:   # Flow → Flow
      description: "攻击在时间维度上的传播演化"
      expected_ratio: 25%  # 基于攻击持续性特征

  overall_ratio: 25-35%  # 反映攻击密集环境的真实特性

negative_samples:
  definition: "未来窗口中不会发生的边连接"
  categories:
    semantic_impossible:     # 语义上不可能的连接
      description: "违反5G网络语义约束的边"
      examples: ["Alert → UE", "低风险UE → 高风险Alert"]

    temporal_violations:     # 时序逻辑违反
      description: "违反攻击时序演化规律的边"
      examples: ["未来Flow → 过去Flow", "告警先于攻击"]

    risk_inconsistent:       # 风险模式不一致
      description: "风险传播模式不符合攻击特征的边"
      examples: ["低风险节点触发高风险告警"]

  overall_ratio: 65-75%     # 补充到100%

sampling_strategy:
  temporal_settings:
    prediction_window: 60s    # 攻击预测窗口
    observation_window: 300s  # 历史观察窗口
    sliding_step: 30s         # 滑动窗口步长

  attack_context:
    focus_attack_types: ["UDPFlood", "HTTPFlood", "SlowrateDoS"]
    risk_threshold: 0.6       # 高风险节点阈值
    propagation_depth: 3      # 攻击传播深度
```

### 评估指标体系重新设计
```yaml
# 攻击预测专用指标 (替代异常检测AUC)
primary_metrics:
  attack_prediction_accuracy:    # 攻击边预测准确率
    description: "正确预测攻击边的比例"
    target: "> 0.80"

  early_warning_effectiveness:   # 提前预警有效性
    description: "成功提前预警的攻击比例"
    target: "> 0.75"
    measurement: "预警时间 > 30s 的攻击覆盖率"

  attack_chain_completeness:     # 攻击链完整性
    description: "完整预测攻击传播路径的能力"
    target: "> 0.70"
    measurement: "端到端攻击链召回率"

  threat_escalation_precision:   # 威胁升级精度
    description: "准确预测攻击强度升级的能力"
    target: "> 0.85"

# 安全运营导向指标
operational_metrics:
  false_alarm_rate:             # 误报率 (SOC关键指标)
    description: "错误攻击预警的比例"
    target: "< 0.15"

  attack_type_discrimination:   # 攻击类型区分能力
    description: "不同攻击类型的预测准确率"
    breakdown: ["UDPFlood", "HTTPFlood", "SlowrateDoS"]

  temporal_prediction_stability: # 时序预测稳定性
    description: "连续时间窗口预测结果的一致性"
    measurement: "预测结果时序相关性"
```

### 方案B：威胁演化预测

#### 任务定义
```
预测攻击的演化模式：
1. 攻击类型转换 (UDPFlood → HTTPFlood)
2. 威胁等级升级 (0.3 → 0.9)
3. 攻击工具切换 (Hping3 → Goldeneye)
```

#### 正负样本定义
```yaml
positive_samples:
  - threat_escalation: "威胁等级提升的边"
  - attack_type_evolution: "攻击类型变化的边"
  - tool_switching: "攻击工具切换的边"
  expected_ratio: 15-25%

negative_samples:
  - stable_attacks: "攻击模式保持不变的边"
  - threat_degradation: "威胁等级下降的边"
  - benign_transitions: "良性状态转换的边"
  expected_ratio: 75-85%
```

## 🔧 配置修复方案

### 1. 重新设计正样本提取逻辑

```python
# 新的正样本定义
attack_propagation_config = {
    'attack_initiation': {
        'pos_ratio': 0.30,  # 30%的UE→Flow边是新攻击发起
        'desc': '新攻击发起边：UE开始新的攻击流量'
    },
    'alert_trigger': {
        'pos_ratio': 0.25,  # 25%的Flow→Alert边是新告警
        'desc': '新告警触发边：Flow触发新的安全告警'
    },
    'attack_propagation': {
        'pos_ratio': 0.20,  # 20%的Flow→Flow边是攻击传播
        'desc': '攻击传播边：攻击在时间上的延续'
    },
    'threat_escalation': {
        'pos_ratio': 0.15,  # 15%的边涉及威胁升级
        'desc': '威胁升级边：攻击强度或类型的升级'
    }
}
```

### 2. 重新设计评估指标

```python
evaluation_metrics = {
    'attack_prediction': {
        'early_warning_auc': 'AUC for 10-60s early warning',
        'attack_chain_recall': 'Complete attack chain recall rate',
        'threat_escalation_precision': 'Threat escalation prediction precision'
    },
    'temporal_analysis': {
        'lead_time_distribution': 'Distribution of prediction lead times',
        'attack_duration_accuracy': 'Attack duration prediction accuracy',
        'propagation_path_accuracy': 'Attack propagation path accuracy'
    },
    'business_impact': {
        'false_positive_cost': 'Cost of false positive alerts',
        'missed_attack_cost': 'Cost of missed critical attacks',
        'soc_workload_optimization': 'SOC analyst workload optimization'
    }
}
```

### 3. 重新设计风险偏置机制

```python
risk_bias_redesign = {
    'attack_intensity_bias': {
        'description': '基于攻击强度的注意力偏置',
        'formula': 'attention_weight *= (1 + α * attack_intensity)',
        'parameters': {'α': 0.5}
    },
    'threat_evolution_bias': {
        'description': '基于威胁演化的偏置',
        'formula': 'attention_weight *= (1 + β * threat_evolution_score)',
        'parameters': {'β': 0.3}
    },
    'temporal_urgency_bias': {
        'description': '基于时间紧迫性的偏置',
        'formula': 'attention_weight *= (1 + γ * temporal_urgency)',
        'parameters': {'γ': 0.4}
    }
}
```

## 📊 预期改进效果

### 基于重新设计的实验目标

1. **任务匹配度**: 从错配(异常检测)到匹配(攻击预测)
2. **正样本比例**: 从不合理(3%)到合理(25-35%)
3. **评估指标**: 从通用指标到5G安全专用指标
4. **业务价值**: 从学术练习到实际SOC应用

### 预期性能提升

```yaml
baseline_performance:
  current_auc: 0.54  # 当前基础模型AUC
  
expected_performance:
  attack_prediction_auc: 0.75-0.85  # 攻击预测AUC
  early_warning_accuracy: 0.80+     # 提前预警准确率
  attack_chain_recall: 0.70+        # 攻击链召回率
  threat_escalation_precision: 0.85+ # 威胁升级精度
```

## 🚀 实施建议

### 立即行动 (1-2天)
1. **停止当前训练**：当前3%正样本比例的实验
2. **重新标注数据**：基于攻击传播逻辑重新定义正负样本
3. **修改配置文件**：更新正样本比例和评估指标

### 短期优化 (1周)
1. **实现新的采样器**：AttackPropagationSampler
2. **重新设计评估脚本**：5G安全专用指标
3. **验证数据质量**：确保新标注的合理性

### 中期完善 (2-3周)
1. **对比实验**：新旧任务定义的性能对比
2. **消融研究**：不同攻击类型的预测难度分析
3. **可解释性**：攻击传播路径的可视化

## 💡 关键洞察

1. **数据集特性决定任务定义**：5G-NIDD的高攻击密度决定了这是攻击预测而非异常检测
2. **正样本比例应反映真实场景**：25-35%的攻击传播比例更符合实际网络安全场景
3. **评估指标应匹配业务目标**：SOC关心的是攻击预测准确性和提前预警时间
4. **风险偏置应针对攻击特性**：攻击强度、威胁演化、时间紧迫性比通用风险分数更有意义

这个重新设计将使RiskNICLP-5G从一个"表现不佳的异常检测器"转变为一个"高效的攻击传播预测器"。
