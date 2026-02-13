# 从零训练的场景重采样策略

## 📊 当前配置分析

```yaml
resample_scenes: true
resample_dataset_size: 20_000  # 从2万个场景中采样
resample_interval: 5_000_000   # 每500万步重采样一次
num_worlds: 16                 # 16个并行环境
k_unique_scenes: 72           # 每次采样72个唯一场景
```

## 🎯 核心问题

**从零训练时，应该：**
1. ❓ **频繁重采样**（如每1-2M步）→ 快速暴露多样性，但可能不稳定
2. ❓ **延迟重采样**（如先训练10-20M步再开启）→ 先稳定学习，再增加多样性

## 💡 推荐策略：渐进式课程学习（Curriculum Learning）

### 策略概述

**采用分阶段的重采样策略**，从稳定学习逐步过渡到多样性探索：

```
阶段1（0-30M步）：稳定基础学习
  → 较长重采样间隔（5-8M步）
  → 让模型先学会基本驾驶技能

阶段2（30-100M步）：逐步增加多样性
  → 中等重采样间隔（3-5M步）
  → 在稳定基础上增加场景多样性

阶段3（100M+步）：充分探索
  → 标准重采样间隔（2-3M步）
  → 最大化泛化能力
```

## 📋 具体配置方案

### 方案A：三阶段渐进式（推荐）

```yaml
# 阶段1：稳定基础（0-30M步）
resample_scenes: true
resample_interval: 6_000_000  # 6M步重采样一次（约3-4次）

# 阶段2：增加多样性（30-100M步）
resample_interval: 3_500_000  # 3.5M步重采样一次（约20次）

# 阶段3：充分探索（100M+步）
resample_interval: 2_500_000  # 2.5M步重采样一次
```

**优点**：
- ✅ 训练初期稳定，避免过早过拟合
- ✅ 逐步增加难度，符合课程学习原理
- ✅ 平衡学习效率与泛化能力

### 方案B：两阶段策略（简化版）

```yaml
# 阶段1：基础学习（0-50M步）
resample_scenes: true
resample_interval: 5_000_000  # 5M步重采样（约10次）

# 阶段2：充分探索（50M+步）
resample_interval: 2_500_000  # 2.5M步重采样
```

**优点**：
- ✅ 配置简单，易于实现
- ✅ 仍然遵循课程学习原则

### 方案C：固定间隔（当前配置）

```yaml
resample_scenes: true
resample_interval: 5_000_000  # 固定5M步重采样
```

**适用场景**：
- ✅ 如果数据集场景质量均匀
- ✅ 如果训练时间充足（180M步）
- ⚠️ 可能训练初期不够稳定

## 🔬 理论依据

### 1. 课程学习（Curriculum Learning）

**原理**：从简单到复杂的学习顺序
- **初期**：固定场景 → 学习基本技能（避障、跟车、变道）
- **中期**：增加场景 → 学习泛化能力
- **后期**：频繁切换 → 最大化鲁棒性

### 2. 过拟合风险

**问题**：如果过早频繁重采样
- ❌ 模型可能无法充分学习当前场景
- ❌ 训练信号不稳定，收敛慢
- ❌ 可能学到"浅层"策略而非"深层"理解

**解决**：先让模型在固定场景上充分学习
- ✅ 建立稳定的价值函数估计
- ✅ 学习基本的驾驶策略
- ✅ 然后再引入多样性

### 3. 样本效率

**计算**：
- 16个并行环境 × 1024步/rollout ≈ 16,384步/rollout
- 假设每个rollout约1000步有效数据
- 5M步 ≈ 5000个rollout ≈ 500万有效样本

**分析**：
- 对于72个场景，每个场景约7万样本
- 足够学习基本策略，但可能不够学习复杂交互

## 🛠️ 实现方式

### 方式1：手动分阶段训练（推荐）

```bash
# 阶段1：0-30M步
python baselines/ppo/my_ppo_pufferlib.py \
    baselines/ppo/config/ppo_base_puffer.yaml \
    --resample_interval 6000000

# 阶段2：从30M步继续训练
python baselines/ppo/my_ppo_pufferlib.py \
    baselines/ppo/config/ppo_continue_training.yaml \
    --continue_training true \
    --model_cpt <checkpoint_at_30M> \
    --resample_interval 3500000

# 阶段3：从100M步继续训练
python baselines/ppo/my_ppo_pufferlib.py \
    baselines/ppo/config/ppo_continue_training.yaml \
    --continue_training true \
    --model_cpt <checkpoint_at_100M> \
    --resample_interval 2500000
```

### 方式2：动态调整（需要修改代码）

在 `gpudrive/integrations/puffer/ppo.py` 的 `evaluate()` 函数中：

```python
def evaluate(data):
    # 动态调整重采样间隔
    progress = data.global_step / data.config.total_timesteps
    
    if progress < 0.17:  # 前30M/180M ≈ 17%
        effective_interval = 6_000_000
    elif progress < 0.56:  # 30-100M/180M ≈ 56%
        effective_interval = 3_500_000
    else:  # 100M+步
        effective_interval = 2_500_000
    
    if (
        data.config.resample_scenes
        and data.resample_buffer >= effective_interval  # 使用动态间隔
        and data.config.resample_dataset_size > data.vecenv.num_worlds
    ):
        print(f"Resampling scenarios at global step {data.global_step}")
        data.vecenv.resample_scenario_batch()
        data.resample_buffer = 0
    # ... 其余代码
```

## 📈 预期效果

### 训练曲线对比

**频繁重采样（1-2M步）**：
```
Reward: 波动大，收敛慢
Collision Rate: 不稳定
```

**延迟重采样（5M+步）**：
```
Reward: 稳定上升，收敛快
Collision Rate: 平稳下降
```

**渐进式重采样（推荐）**：
```
Reward: 初期稳定上升，后期快速提升
Collision Rate: 持续下降，最终最低
```

## ⚠️ 注意事项

1. **监控训练指标**
   - 如果reward波动大 → 增加重采样间隔
   - 如果collision rate不降 → 可能需要更频繁重采样

2. **数据集质量**
   - 如果场景质量差异大 → 需要更频繁重采样
   - 如果场景质量均匀 → 可以延长间隔

3. **计算资源**
   - 重采样会短暂暂停训练（IO操作）
   - 如果IO是瓶颈 → 减少重采样频率

## 🎯 最终建议

**对于从零训练，推荐使用方案A（三阶段渐进式）**：

1. **0-30M步**：`resample_interval: 6_000_000`
   - 让模型先学会基本驾驶技能
   - 建立稳定的价值函数

2. **30-100M步**：`resample_interval: 3_500_000`
   - 逐步增加场景多样性
   - 学习泛化能力

3. **100M+步**：`resample_interval: 2_500_000`
   - 最大化场景覆盖
   - 提升最终性能

这样既能保证训练稳定性，又能最大化泛化能力！
