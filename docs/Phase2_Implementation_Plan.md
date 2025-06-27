# RiskNICLP-5G Phase 2 实施计划
*基于Phase 1稳定基础，实现真正的风险感知与提前预警能力*

**当前状态**: Phase 1 ✅ 完成，立即启动 Phase 2  
**实施目标**: 从"能跑的基础模型"升级为"实用的安全预警系统"

---

## 🎯 Phase 2 核心目标

### 关键突破点
- **从随机基线（AUC=0.5）→ 实用基线（AUC>0.7）**
- **从无预警（Lead-Time=0秒）→ 有效预警（Lead-Time=5-15秒）**  
- **从简单增强 → 真正的Risk-Bias注意力机制**

### 成功验收标准
```yaml
P0_必达目标:
  lead_time: ">5秒"          # 实现提前预警
  auc: ">0.7"               # 突破随机基线
  risk_bias_working: true   # Risk-Bias机制生效
  training_stable: "50+epochs"  # 长期训练稳定

P1_重要目标:
  hit_100: ">0.5"           # 链路预测准确率
  risk_weighted_recall: ">0.3"  # 风险加权召回率
  memory_usage: "<6GB"      # 内存使用控制

P2_可选目标:
  cross_scale_context: true     # 跨尺度信息交互
  attention_visualization: true # 注意力可视化
  mttd_improvement: "<10秒"    # 检测时间进一步优化
```

---

## 📋 主要实施任务

### 1. 增强Risk-Bias注意力机制

**当前问题**: `src/models/encoders.py`中的RiskBiasGATConv只是简单的乘法增强
**解决方案**: 实现真正的风险偏置注意力

**需要做的事情**:
- [ ] 改进`RiskBiasGATConv.forward()`方法，实现基于边的风险偏置计算
- [ ] 添加可学习的风险偏置参数β₁、β₂，支持渐进训练
- [ ] 集成到现有的UEEncoder中，替换简单风险增强
- [ ] 验证注意力权重随风险分数变化

**核心技术要点**:
```python
# 需要实现的风险偏置注意力核心逻辑
def compute_risk_bias_attention(self, graph, features, risk_scores):
    # 基础GAT注意力 + 风险偏置
    src_nodes, dst_nodes = graph.edges()
    src_risk = risk_scores[src_nodes]
    dst_risk = risk_scores[dst_nodes]
    
    # β₁·r_i + β₂·r_j 风险偏置
    risk_bias = self.beta_1 * src_risk + self.beta_2 * dst_risk
    enhanced_attention = base_attention + risk_bias / self.temperature
    
    return enhanced_attention
```

### 2. 数据增强与负采样策略

**当前问题**: 测试集只有单类别，导致AUC=0.5随机基线
**解决方案**: 实现风险感知的负采样器

**需要做的事情**:
- [ ] 创建`src/data/negative_sampler.py`模块
- [ ] 实现`RiskAwareNegativeSampler`类，支持分层负采样
- [ ] 修改`src/data/graph_dataset.py`，集成负采样到数据加载
- [ ] A/B测试不同负采样策略（均匀 vs 风险分层）
- [ ] 验证正负样本平衡，确保AUC>0.7

**核心技术要点**:
```python
# 分层负采样策略
class RiskAwareNegativeSampler:
    def sample_negatives(self, graph, positive_edges, risk_scores):
        # 高风险节点间的虚假连接作为困难负样本
        # 中低风险节点的随机连接作为简单负样本
        # 确保负样本不在真实边中
```

### 3. 时序预警机制实现 - ✅ 优化架构实施完成

**当前问题**: Lead-Time=0，无法提前预警
**解决方案**: 实现基于图时序建模的未来风险预测

**架构决策更新** (2025-06-23):
基于技术分析和约束条件，采用**渐进式三阶段实施策略**:
- **P0 (立即实施)**: HeteroGNN→LSTM混合架构 ✅
- **P1 (性能目标)**: TGAT+Risk-Bias注意力
- **P2 (对比研究)**: Graph-RNN消融实验

**已完成任务**:
- [x] 创建`src/models/temporal_predictor.py`模块，支持四种时序方法
- [x] 实现`GraphTemporalPredictor`类，默认`hetero_lstm`方法
- [x] 修改`train.py`，集成temporal_predictor创建逻辑
- [x] 更新`config/baseline_config.yaml`，配置默认为hetero_lstm
- [x] 创建技术决策文档`docs/Temporal_Modeling_Architecture_Decision.md`

**P0 HeteroLSTM核心实现**:
```python
# 空间-时序解耦架构
class GraphTemporalPredictor(nn.Module):
    def __init__(self, method='hetero_lstm', shared_encoder=None):
        # P0: HeteroLSTM (推荐) - 复用SharedEncoder + LSTM
        if method == 'hetero_lstm':
            self.spatial_encoder = shared_encoder  # 复用Phase 1编码器
            self.temporal_encoder = nn.LSTM(hidden_dim, hidden_dim)
            self.graph_aggregator = nn.Sequential(...)  # 图级聚合
            
    def _hetero_lstm_forward(self, historical_graphs, current_graph):
        # Step 1: 空间编码 - 复用SharedEncoder
        temporal_embeddings = []
        for graph in all_graphs:
            spatial_embedding = self.spatial_encoder(graph)
            graph_embedding = self.graph_aggregator(spatial_embedding.mean(0))
            temporal_embeddings.append(graph_embedding)
            
        # Step 2: 时序建模 - LSTM处理图序列
        sequence = torch.stack(temporal_embeddings, 0).unsqueeze(0)
        lstm_out, (hidden, cell) = self.temporal_encoder(sequence)
        
        # Step 3: 风险预测 - 多步预测(5s,10s,15s)
        future_risks = self.risk_predictor(hidden[-1])
        return future_risks
```

**优势**: 
- ✅ 快速实施(3-5天) - 复用成熟组件
- ✅ CPU友好 - O(T·N)复杂度，内存可控  
- ✅ 稳定收敛 - LSTM序列建模成熟
- ✅ 向后兼容 - 不影响Phase 1功能

### 4. Cross-Scale Context机制

**当前问题**: `LateFusion`只是虚拟实现，未真正利用双尺度信息
**解决方案**: 实现细粒度与粗粒度的信息交互

**需要做的事情**:
- [ ] 创建`src/models/cross_scale_context.py`模块
- [ ] 实现`CrossScaleContextModule`类，支持双向信息流
- [ ] 修改`src/models/riskniclp_model.py`，支持真正的双尺度输入
- [ ] 替换虚拟LateFusion为真实的双尺度融合
- [ ] 实现注意力权重可视化（SOC友好）

**核心技术要点**:
```python
# 跨尺度上下文信息交互
class CrossScaleContextModule(nn.Module):
    def forward(self, fine_embeddings, coarse_embeddings):
        # 细粒度 → 粗粒度信息注入
        # 粗粒度 → 细粒度信息注入
        # 门控融合机制
        return enhanced_fine, enhanced_coarse
```

### 5. 系统优化与稳定性提升

**当前问题**: 只训练了2个epoch，需要支持长期训练
**解决方案**: 系统性能优化和超参数调优

**需要做的事情**:
- [ ] 超参数网格搜索（学习率、batch_size、hidden_dim等）
- [ ] 学习率调度策略优化（Cosine vs StepLR vs ReduceLROnPlateau）
- [ ] 损失函数权重调优（BCE vs Focal Loss处理类别不平衡）
- [ ] 50-epoch长训练稳定性测试
- [ ] 内存使用监控和优化（确保<6GB）

**核心技术要点**:
```python
# 关键超参数优化空间
hyperparams = {
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'batch_size': [2, 4, 8],
    'hidden_dim': [64, 128, 256],
    'risk_alpha': [0.5, 0.7, 0.9],
    'neg_ratio': [1, 3, 5]
}
```

---

## 🔧 技术实现策略

### 基于现有代码的改进
1. **复用Phase 1架构**: 保持现有的内存优化策略（分块计算、节点采样）
2. **渐进式改进**: 先实现核心功能，再添加高级特性
3. **向后兼容**: 确保Phase 1的稳定训练流程不受影响
4. **模块化设计**: 新功能独立模块，支持开关控制

### 验证策略
- **单元测试**: 每个新模块独立测试
- **集成测试**: 与现有系统集成验证
- **性能基准**: 与Phase 1基线对比
- **消融实验**: 验证每个改进的独立贡献

### 风险控制
- **开关机制**: 高级特性支持动态启用/禁用
- **兜底策略**: 复杂模块失败时回退到简单版本
- **内存监控**: 严格控制内存使用，避免OOM
- **渐进训练**: 支持基础→风险感知的阶段切换

---

## 📊 预期成果

### 性能提升目标
```yaml
# Phase 1 基线 → Phase 2 目标
实时安全性:
  MTTD_30s: 16.73秒 → <10秒
  lead_time: 0秒 → 5-15秒

检测质量:  
  auc: 0.5 → >0.7
  aupr: 0.5 → >0.7
  risk_weighted_recall: 0.0 → >0.3

方向性准确度:
  directional_hit_100: 100% → >95% (保持)
  direction_error_rate: 0% → <10% (保持)

系统能力:
  训练稳定性: 2epochs → 50+epochs
  内存使用: ~4GB → <6GB
  收敛时间: ~7分钟 → <30分钟
```

### 技术能力提升
- ✅ **真正的风险感知**: Risk-Bias注意力机制生效
- ✅ **提前预警能力**: Lead-Time>5秒的实际预警
- ✅ **数据增强效果**: 解决单类别问题，AUC突破随机基线
- ✅ **跨尺度融合**: 双尺度信息有效利用
- ✅ **长期训练稳定**: 支持生产级训练流程

---

## 🚀 实施优先级 - 更新状态

### ✅ 已完成（P0完成）:
3. **时序预警机制** - HeteroLSTM架构实施完成 ✅
   - 创建`temporal_predictor.py`模块 ✅
   - 实现四种时序方法支持 ✅
   - 集成到训练流程 ✅
   - 配置文件更新 ✅

### 🔄 立即开始（P0剩余）:
1. **Risk-Bias注意力增强** - 直接影响核心算法效果
2. **负采样策略实现** - 解决AUC随机基线问题  
4. **时序预警机制验证** - P0 HeteroLSTM性能测试

### 📋 跟进实施（P1）:
5. **Cross-Scale Context** - 提升双尺度融合效果
6. **系统优化调优** - 确保长期训练稳定性
7. **TGAT+Risk实施** - P1性能目标版本

### 🔮 按需扩展（P2）:
8. **Graph-RNN对比** - 消融实验与论文材料
9. **注意力可视化** - SOC友好的解释性分析
10. **API接口设计** - 为Phase 3生产部署准备

**实施策略更新**: 
- **P0优先**: 先验证HeteroLSTM基线性能(Lead-Time>5s, AUC>0.7)
- **渐进优化**: P0稳定后启动P1 TGAT实施
- **对比研究**: P2作为学术完整性补充

**核心理念**: 渐进式架构演进，确保每阶段可用可测可比，避免一次性复杂实施的风险。 