# 继续训练性能下降问题诊断分析

## 问题描述

**现象**：在预训练模型（3M steps, epoch 184）的基础上继续训练1亿步，效果反而变差

## 预训练模型信息

```python
原始checkpoint信息：
- Global Step: 3,021,617 (约3M步)
- Update (Epoch): 184
- 模型架构: 
  - fusion_type: "attention"
  - input_dim: 64
  - hidden_dim: 128
  - dropout: 0.01
  - num_attention_heads: 4
- 参数量: 67,996 (约68k)
```

## 可能导致性能下降的原因

### 🔴 原因1: 优化器状态重置 ⭐⭐⭐ (最可能)

```python
# ppo.py Line 80-82
optimizer = torch.optim.Adam(
    policy.parameters(), lr=float(config.learning_rate), eps=1e-5
)
# ⚠️  问题：创建了新的优化器，没有加载旧优化器状态！

# 您的代码 my_ppo_pufferlib.py Line 52-93
if config.continue_training:
    saved_cpt = torch.load(...)
    policy = NeuralNet(...)
    policy.load_state_dict(saved_cpt["parameters"], strict=False)
    # ❌ 没有加载 saved_cpt["optimizer_state_dict"]
    return policy

# 然后在 train() 中
data = ppo.create(config, vecenv, policy, wandb=wandb)
# → ppo.create() 创建新优化器，丢失了momentum等状态
```

**影响**：
```python
Adam优化器的状态（被丢失的）：
1. 一阶动量 (moving average of gradients)
   - 记录了梯度的"惯性方向"
   - 帮助加速收敛和跳出鞍点
   
2. 二阶动量 (moving average of squared gradients)  
   - 自适应调整每个参数的学习率
   - 让频繁更新的参数学得慢，罕见参数学得快

丢失后的影响：
- 优化器从头开始累积动量
- 前期梯度更新不稳定（抖动）
- 可能破坏已经学好的参数
- 学习曲线"震荡下降"或"剧烈波动"

类比：
就像一辆正在高速行驶的车（预训练模型）
突然重置了油门和方向盘的记忆（优化器状态）
需要重新"摸索"如何平稳驾驶
→ 性能短期下降
```

### 🔴 原因2: 学习率不匹配 ⭐⭐⭐

```python
# 您的配置
learning_rate: 3e-4  # 固定学习率
anneal_lr: false     # 没有学习率衰减

问题场景A：预训练模型使用了学习率衰减
---------------------------------------
预训练过程（假设）：
- 初始lr: 3e-4
- 3M步后lr已衰减到: 1e-4 或更低
- 模型参数已经在低lr下精细调优

继续训练：
- 重新使用lr: 3e-4 ← 太大了！
- 相当于用"粗调"学习率去"微调"
- 破坏已经收敛的参数

类比：
用铁锤（大lr）去雕刻已经精细打磨的玉器（收敛的模型）
→ 性能崩溃

问题场景B：batch_size不匹配导致有效学习率变化
--------------------------------------------
有效学习率 ∝ lr / sqrt(batch_size)

预训练（假设）: batch_size = 32,768
继续训练: batch_size = 16,384

有效lr变化:
old_eff_lr = 3e-4 / sqrt(32768) ≈ 1.66e-6
new_eff_lr = 3e-4 / sqrt(16384) ≈ 2.34e-6
变化: +41% ← 可能导致不稳定
```

### 🔴 原因3: 数据分布偏移 ⭐⭐

```python
# 预训练模型的训练数据（推测）
可能来源：
- 特定的场景子集
- 特定的数据处理版本
- 特定的配置参数

# 继续训练的数据
data_dir: data/processed/training
resample_dataset_size: 10,000
k_unique_scenes: 64

如果数据分布不同：
场景差异：
- 预训练：简单高速路场景为主
- 继续训练：复杂城市路口为主
→ 策略不适应，性能下降

观测差异：
- polyline_reduction_threshold: 0.1 (当前)
- 如果预训练用的是0.3（更稀疏）
→ 观测密度不匹配，模型困惑

奖励权重差异：
- collision_weight: -0.75 (当前)
- 如果预训练用的是-0.5
→ 优化目标变化，策略冲突
```

### 🔴 原因4: 过拟合陷阱 ⭐⭐

```python
# 预训练模型的状态（推测）
假设预训练在小数据集上训练：
- 场景数: 100-500
- 训练步数: 3M
- 已经过拟合到这些场景

继续训练：
- 新的1000个场景
- 模型需要"忘记"旧场景，学习新场景
- 但参数已经"固化"在旧模式

结果：
- 在旧场景上：性能保持或略降（忘记了）
- 在新场景上：性能很差（学不会）
- 整体评估：性能下降

类比：
老司机在熟悉的城市开得很好
换到完全陌生的城市
还试图用旧习惯（已固化的参数）
→ 适应困难
```

### 🔴 原因5: 全局步数计数器重置 ⭐

```python
# ppo.py Line 94-98
return pufferlib.namespace(
    ...
    global_step=0,      # ❌ 重置为0！
    epoch=0,            # ❌ 重置为0！
    ...
)

影响：
1. 学习率调度失效
   如果 anneal_lr=True:
   frac = 1.0 - global_step / total_timesteps
   # global_step从0开始 → frac=1.0
   # lr重新从最大值开始！

2. Checkpoint命名冲突
   model_name = f"model_{exp_id}_{epoch:06d}.pt"
   # epoch从0开始 → 覆盖旧checkpoint

3. 统计指标混乱
   # WandB的x轴从0开始
   # 无法连续显示训练曲线
```

### 🔴 原因6: 策略分布突变 ⭐

```python
# 迁移学习带来的架构变化

预训练模型：
- fusion_type: "attention"
- 但可能是用 mean pooling (旧版本)
- fusion_output_dim = 64

继续训练：
- fusion_type: "attention"  
- 现在用 flatten (新版本)
- fusion_output_dim = 192

问题：
shared_embed层的输入维度变了！
- 旧: Linear(64, 128)
- 新: Linear(192, 128)
→ 完全不兼容，参数无法加载
→ shared_embed被随机初始化
→ 破坏了特征→动作的映射

验证：
检查 missing_keys 中是否包含 shared_embed 相关参数
```

### 🔴 原因7: Entropy系数不当 ⭐

```python
# 您的配置
ent_coef: 0.0001

问题：
预训练模型可能已经收敛（低entropy）
entropy ≈ 0.5-0.7

继续训练仍用 ent_coef=0.0001：
- 探索不足
- 策略过早收敛到局部最优
- 遇到新场景无法适应

建议的调整：
在继续训练初期增加 ent_coef：
- 初始: 0.001 (增加10倍)
- 逐步衰减到: 0.0001
- 鼓励在新数据上探索
```

## 🔍 诊断步骤

### Step 1: 检查优化器状态是否加载

```python
# 修改 my_ppo_pufferlib.py
def make_agent(env, config):
    if config.continue_training:
        saved_cpt = torch.load(...)
        policy = NeuralNet(...)
        policy.load_state_dict(saved_cpt["parameters"], strict=False)
        
        # ✅ 添加：返回优化器状态
        optimizer_state = saved_cpt.get("optimizer_state_dict", None)
        return policy, optimizer_state  # 返回两个值
    else:
        return policy, None

def train(args, vecenv):
    policy, optimizer_state = make_agent(...)
    
    # 创建优化器
    data = ppo.create(args.train, vecenv, policy, wandb=args.wandb)
    
    # ✅ 加载优化器状态
    if optimizer_state is not None:
        data.optimizer.load_state_dict(optimizer_state)
        print("✅ Optimizer state loaded successfully")
```

### Step 2: 检查模型架构兼容性

```python
# 运行训练时查看警告信息
Missing keys: [...]
Unexpected keys: [...]

# 如果有 shared_embed 相关的 missing keys
→ 说明架构不兼容
→ 需要确认预训练模型的 fusion_type 和 flatten/mean 配置
```

### Step 3: 对比训练配置

```python
# 创建对比表
预训练配置 vs 继续训练配置:

参数                预训练(推测)   继续训练(当前)   是否匹配
====================================================================
num_worlds          ?              16              ?
batch_size          ?              16,384          ?
learning_rate       3e-4           3e-4            ✓
anneal_lr           ?              false           ?
collision_weight    ?              -0.75           ?
off_road_weight     ?              -0.75           ?
polyline_reduction  ?              0.1             ?
obs_radius          ?              50.0            ?
```

### Step 4: 监控训练指标

```python
关键指标观察（继续训练的前10M步）:

1. Learning rate
   - 如果突然变化 → 学习率问题
   
2. Entropy
   - 如果从低突然升高 → 策略崩溃
   - 如果持续很低 → 探索不足
   
3. Explained variance
   - 如果从高突然降低 → Value网络崩溃
   - 如果始终低 → 数据分布偏移
   
4. KL divergence (old_approx_kl)
   - 如果异常大 (>0.1) → 策略变化剧烈
   - 如果很小 (<0.001) → 策略冻结
   
5. Clipfrac
   - 如果很高 (>0.5) → 学习率太大或策略震荡
   - 如果很低 (<0.05) → 学习率太小
```

## 🎯 修复方案

### 方案1: 正确加载优化器状态（必须）

```python
# 修改 my_ppo_pufferlib.py

def make_agent(env, config):
    """Create a policy based on the environment."""

    if config.continue_training:
        print("Loading checkpoint...")
        saved_cpt = torch.load(
            f=config.model_cpt,
            map_location=config.train.device,
            weights_only=False,
        )
        
        old_fusion_type = saved_cpt.get("model_arch", {}).get("fusion_type", "simple")
        old_num_heads = saved_cpt.get("model_arch", {}).get("num_attention_heads", 4)
        
        new_fusion_type = getattr(config.train.network, 'fusion_type', old_fusion_type)
        new_num_heads = getattr(config.train.network, 'num_attention_heads', old_num_heads)
        
        policy = NeuralNet(
            input_dim=saved_cpt["model_arch"]["input_dim"],
            action_dim=saved_cpt["action_dim"],
            hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
            config=config.environment,
            fusion_type=new_fusion_type,
            num_attention_heads=new_num_heads,
        )

        missing_keys, unexpected_keys = policy.load_state_dict(
            saved_cpt["parameters"], strict=False
        )
        
        if missing_keys:
            print(f"⚠️  Warning: Missing keys in checkpoint (will be randomly initialized):")
            for key in missing_keys:
                print(f"   - {key}")
        
        if unexpected_keys:
            print(f"⚠️  Warning: Unexpected keys in checkpoint (will be ignored):")
            for key in unexpected_keys:
                print(f"   - {key}")

        # ✅ 新增：返回优化器状态和全局步数
        optimizer_state = saved_cpt.get("optimizer_state_dict", None)
        global_step_offset = saved_cpt.get("global_step", 0)
        epoch_offset = saved_cpt.get("update", 0)
        
        return policy, optimizer_state, global_step_offset, epoch_offset

    else:
        return NeuralNet(...), None, 0, 0


def train(args, vecenv):
    """Main training loop for the PPO agent."""
    policy, optimizer_state, global_step_offset, epoch_offset = make_agent(
        env=vecenv.driver_env, config=args
    )
    policy = policy.to(args.train.device)

    args.train.network.num_parameters = get_model_parameters(policy)
    args.train.env = args.environment.name

    args.wandb = init_wandb(args, args.train.exp_id, id=args.train.exp_id)
    args.train.__dict__.update(dict(args.wandb.config.train))

    data = ppo.create(args.train, vecenv, policy, wandb=args.wandb)
    
    # ✅ 新增：加载优化器状态和恢复全局步数
    if optimizer_state is not None:
        data.optimizer.load_state_dict(optimizer_state)
        print(f"✅ Optimizer state loaded successfully")
        print(f"✅ Resuming from global_step={global_step_offset}, epoch={epoch_offset}")
    
    # ✅ 恢复全局计数器
    data.global_step = global_step_offset
    data.epoch = epoch_offset
    
    while data.global_step < args.train.total_timesteps:
        try:
            ppo.evaluate(data)
            ppo.train(data)
        except KeyboardInterrupt:
            ppo.close(data)
            os._exit(0)
        except Exception as e:
            print(f"An error occurred: {e}")
            Console().print_exception()
            os._exit(1)

    ppo.evaluate(data)
    ppo.close(data)
```

### 方案2: 降低学习率（如果方案1不够）

```yaml
# ppo_base_puffer.yaml

train:
  learning_rate: 1e-4  # ✅ 从3e-4降低到1e-4
  anneal_lr: true      # ✅ 启用学习率衰减
  
  # 或者使用warmup + decay
  # 初始几M步用低lr适应，再逐步恢复
```

**原理**：
```python
# 继续训练的学习率策略

阶段1 (0-10M steps): lr = 1e-4 (低lr适应期)
- 让模型在新数据上温和适应
- 避免破坏已学到的知识

阶段2 (10M-50M steps): lr = 3e-4 (正常训练)
- 模型已适应新数据
- 可以正常优化

阶段3 (50M-100M steps): lr衰减到1e-5
- 精细调优
- 收敛到最优
```

### 方案3: 增加探索（Entropy）

```yaml
train:
  ent_coef: 0.001  # ✅ 从0.0001增加到0.001
```

**原理**：
```python
预训练模型可能过度收敛：
- Entropy很低
- 策略确定性强
- 难以适应新场景

增加ent_coef：
- 鼓励探索新动作
- 防止过早收敛
- 在新场景上更灵活

建议：
初期: ent_coef = 0.001 (探索)
后期: 逐步降到 0.0001 (利用)
```

### 方案4: 调整PPO Clip范围

```yaml
train:
  clip_coef: 0.1  # ✅ 从0.2降低到0.1（更保守）
```

**原理**：
```python
# PPO的clip机制

ratio = π_new(a|s) / π_old(a|s)
clipped = clip(ratio, 1-ε, 1+ε)

当前: ε = 0.2 (允许±20%的策略变化)
建议: ε = 0.1 (只允许±10%的变化)

效果：
- 更保守的更新
- 避免破坏已学到的策略
- 适合继续训练（微调）
```

### 方案5: 使用更小的batch_size（探索性）

```yaml
train:
  batch_size: 8192   # ✅ 从16384减半
  minibatch_size: 1024  # 相应调整
```

**原理**：
```python
小batch_size的优势：
1. 梯度估计更noisy
   - 有助于逃离局部最优
   - 增加随机性，类似于探索

2. 更频繁的参数更新
   - 更快适应新数据分布
   - 但可能不稳定

3. 正则化效果
   - 类似于dropout
   - 防止过拟合

权衡：
- 训练可能更慢
- 但更可能改善性能
```

## 📊 对比实验建议

### 实验设置

```python
# 为了找出确切原因，运行对比实验

Baseline (当前配置):
- 加载预训练模型
- 使用当前配置继续训练10M步
- 记录性能

Exp1 (加载优化器):
- ✅ 加载optimizer_state_dict
- 其他不变
- 对比性能变化

Exp2 (降低学习率):
- learning_rate: 3e-4 → 1e-4
- 其他不变
- 对比性能变化

Exp3 (增加探索):
- ent_coef: 0.0001 → 0.001
- 其他不变
- 对比性能变化

Exp4 (从头训练):
- continue_training: false
- 作为upper bound
- 看最终能达到什么性能
```

## 🔬 性能下降的典型表现

### 症状1: 立即崩溃型

```python
训练曲线：
Reward
  │ ●●●●●        ← 预训练性能
  │      ╲
  │       ╲      ← 继续训练开始
  │        ●●
  │          ●●  ← 性能快速下降
  │            ●●
  └─────────────
    3M   10M  20M

原因：
- 学习率太大
- 优化器状态丢失
- 参数破坏严重

解决：
- 降低lr到1e-5
- 加载optimizer_state
```

### 症状2: 震荡不稳定型

```python
训练曲线：
Reward
  │ ●●●●●        ← 预训练
  │    ╱ ╲╱ ╲
  │   ●   ●  ●   ← 剧烈震荡
  │  ╱     ╲  ╲
  │ ●       ●  ●
  └─────────────
    3M   20M  40M

原因：
- batch_size不匹配
- 数据分布差异大
- ent_coef不当

解决：
- 匹配batch_size
- 增加ent_coef
- 降低lr
```

### 症状3: 缓慢退化型

```python
训练曲线：
Reward
  │ ●●●●●        ← 预训练
  │      ●
  │       ●      ← 缓慢下降
  │        ●
  │         ●
  │          ●
  └─────────────
    3M   50M  100M

原因：
- 数据分布偏移
- 灾难性遗忘
- 过拟合新数据，忘记旧知识

解决：
- 混合新旧数据
- 使用EWC (Elastic Weight Consolidation)
- 降低lr，保守更新
```

## 🎯 立即行动建议

### 优先级1: 加载优化器状态（必须）

这是最可能的原因！丢失Adam动量会严重影响训练。

### 优先级2: 降低学习率（建议）

```yaml
learning_rate: 1e-4  # 降低3倍
```

### 优先级3: 增加探索（可选）

```yaml
ent_coef: 0.0005  # 增加5倍
```

### 优先级4: 检查架构兼容性（验证）

运行训练，查看是否有大量 `missing_keys`。

## 总结

**最可能的原因排序**：

1. ⭐⭐⭐ **优化器状态丢失** (80%概率)
   - Adam动量重置
   - 梯度更新不稳定
   - 破坏已学到的参数

2. ⭐⭐⭐ **学习率不匹配** (70%概率)
   - 预训练可能已衰减lr
   - 继续用初始lr破坏精细参数

3. ⭐⭐ **数据分布偏移** (50%概率)
   - 新场景与预训练差异大
   - 策略不适应

4. ⭐⭐ **全局步数重置** (40%概率)
   - 影响lr调度
   - 统计指标混乱

5. ⭐ **架构不兼容** (30%概率)
   - 如果有missing_keys
   - 部分层被随机初始化

**建议先实施方案1（加载优化器状态），这很可能就能解决80%的问题！**

