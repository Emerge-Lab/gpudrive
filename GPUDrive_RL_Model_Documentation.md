# GPUDrive 强化学习与模型架构详解

> 本文档专注于 GPUDrive 项目的强化学习训练流程、神经网络架构、奖励函数设计等核心内容。

---

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [神经网络模型](#神经网络模型)
- [强化学习算法](#强化学习算法)
- [奖励函数设计](#奖励函数设计)
- [训练流程](#训练流程)
- [关键配置参数](#关键配置参数)
- [性能优化](#性能优化)
- [常见问题与调优](#常见问题与调优)

---

## 项目概述

### 什么是 GPUDrive？

GPUDrive 是一个**极速的、数据驱动的**自动驾驶模拟器，具有以下特点：

- **⚡ 超高速度**: 基于 Madrona 引擎，可达到 **100万 FPS** 的仿真速度
- **🐍 Python 友好**: 提供 Gymnasium 兼容的 Torch/JAX 接口
- **🏃 大规模数据**: 兼容 Waymo Open Motion Dataset (100K+ 场景)
- **📜 即用 RL**: 内置 PPO 实现（Stable-Baselines3 和 PufferLib）
- **🎨 多样场景**: 支持车辆、骑行者、行人等多种智能体

### 核心优势

| 维度 | 传统模拟器 | GPUDrive |
|-----|----------|----------|
| **仿真速度** | 100-1000 FPS | **100万+ FPS** |
| **并行环境** | 10-100 | **数千个** |
| **硬件加速** | CPU/少量GPU | **全GPU加速** |
| **数据集成** | 手动设计 | **Waymo真实数据** |
| **训练吞吐** | 1K SPS | **100K-300K SPS** |

---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     Python 层 (用户接口)                      │
├─────────────────────────────────────────────────────────────┤
│  训练脚本             │  神经网络模型          │  环境封装      │
│  ppo_pufferlib.py    │  late_fusion.py       │  env_puffer.py │
│  ↓                   │  ↓                    │  ↓             │
│  PufferLib PPO       │  NeuralNet (PyTorch)  │  PufferGPUDrive│
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ↓                          ↓
┌─────────────────────────────────────────────────────────────┐
│              C++ 层 (Madrona 模拟器后端)                      │
├─────────────────────────────────────────────────────────────┤
│  sim.cpp          │  types.hpp         │  MapReader.cpp     │
│  (物理&碰撞)       │  (ECS组件)         │  (地图加载)         │
└──────────────┬──────────────────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────────────────────┐
│                    GPU CUDA 并行执行                          │
│  数千个场景 × 数十个智能体 = 数十万并行实体                    │
└─────────────────────────────────────────────────────────────┘
```

### 三层分工

1. **Python 训练层**
   - 定义训练超参数（学习率、批大小等）
   - 实现神经网络（观察编码、策略输出）
   - 执行 PPO 算法（优势估计、策略更新）

2. **C++ 模拟器层**
   - GPU 加速的 ECS（Entity-Component-System）
   - 物理动力学、碰撞检测
   - 观察生成（自车状态、道路图、其他车辆）

3. **CUDA 执行层**
   - 大规模并行计算
   - 内存高效管理
   - 多世界同步推进

---

## 神经网络模型

### 模型概览

GPUDrive 使用 **Late Fusion Network（晚期融合网络）**，这是一个**前馈神经网络（Feed-Forward Network）**，专门设计用于处理自动驾驶的多模态观察。

**网络特点**：
- ✅ 前馈架构（无循环连接）
- ✅ 多模态独立编码
- ✅ 晚期融合策略
- ❌ 无历史记忆（每个时间步独立）
- ❌ 无循环单元（无LSTM/GRU）

```python
class NeuralNet(nn.Module):
    def __init__(
        self,
        action_dim=91,        # 动作空间大小 (7加速 × 13转向)
        input_dim=64,         # 嵌入维度
        hidden_dim=128,       # 隐藏层维度
        fusion_type="attention",  # 融合方式
        num_attention_heads=4,    # 注意力头数
    )
```

### 网络架构详解

#### 1. 观察空间结构

GPUDrive 的观察空间包含三个模态：

| 模态 | 维度 | 内容 | 说明 |
|-----|------|------|------|
| **Ego State** | 6 | 自车状态 | 速度、车辆尺寸、相对目标位置、碰撞状态 |
| **Partner Obs** | 7 × 63 = 441 | 其他车辆 | 最多63个其他智能体的状态 |
| **Road Map** | 8 × 314 = 2512 | 道路图 | 最多314个道路点 (车道线、边界等) |

**总观察维度**: 6 + 441 + 2512 = **2959 维**

**注**：
- C++ 层实际提供 8 维自车状态（包含 vehicle_height 和 id）
- Python 层只使用其中 6 维（不使用 height 和 id）
- Partner Obs 每个智能体 7 维（包含了 height）

#### 2. 编码器设计

```
输入观察 (2959维)
    │
    ├──► Ego Encoder (MLP)
    │    [6] → [64] → [64]
    │    ↓
    │    Ego Embedding [64]
    │
    ├──► Partner Encoder (MLP)
    │    [7×63] → [64×63] → [64]  (MaxPool聚合)
    │    ↓
    │    Partner Embedding [64]
    │
    └──► Road Encoder (MLP)
         [8×314] → [64×314] → [64]  (MaxPool聚合)
         ↓
         Road Embedding [64]
```

**详细维度说明**：

```python
# Ego State (6维)
ego_state = [
    speed,              # 速度
    vehicle_length,     # 车辆长度
    vehicle_width,      # 车辆宽度
    rel_goal_x,         # 相对目标位置x
    rel_goal_y,         # 相对目标位置y
    is_collided,        # 碰撞状态 (0/1)
]

# Partner Obs (7维 × 63个智能体)
partner_obs = [
    [speed, length, width, height, rel_pos_x, rel_pos_y, rel_heading],  # Agent 1
    [speed, length, width, height, rel_pos_x, rel_pos_y, rel_heading],  # Agent 2
    ...  # 共63个
]

# Road Map (8维 × 314个道路点)
road_map = [
    [pos_x, pos_y, scale_x, scale_y, scale_z, heading, type, map_type],  # Point 1
    [pos_x, pos_y, scale_x, scale_y, scale_z, heading, type, map_type],  # Point 2
    ...  # 共314个
]
```

**关键技术**：
- 使用 **MaxPool** 聚合可变数量的智能体/道路点
- 保持排列不变性（Permutation Invariance）
- 每个模态独立编码，保留模态特异性

**"Late Fusion" vs "Early Fusion"**：

```python
# Early Fusion（早期融合）
input = concat([ego, partner, road])  # 先拼接
output = FFN(input)                   # 再编码
# 优点：简单，参数少
# 缺点：丢失模态特异性

# Late Fusion（晚期融合）⭐ GPUDrive使用
ego_embed = FFN_ego(ego)             # 各自独立编码
partner_embed = FFN_partner(partner)
road_embed = FFN_road(road)
fused = fusion(ego_embed, partner_embed, road_embed)  # 后融合
output = FFN_shared(fused)
# 优点：保留模态特征，更适合异构数据
# 缺点：参数稍多
```

#### 3. 融合机制

GPUDrive 支持三种融合方式：

##### 方式 1: Simple Fusion（简单拼接）

```python
fusion_type = "simple"

# 直接拼接三个嵌入
fused = torch.cat([ego_embed, partner_embed, road_embed], dim=-1)
# [64 + 64 + 64] = [192]

# 通过共享MLP
shared_output = shared_mlp(fused)  # [192] → [128]
```

**优点**: 简单高效  
**缺点**: 无法学习模态间的重要性权重

##### 方式 2: Attention Fusion（注意力融合）⭐

```python
fusion_type = "attention"
num_attention_heads = 4

# 将三个嵌入视为序列
modalities = torch.stack([ego_embed, partner_embed, road_embed], dim=1)
# Shape: [batch, 3, 64]

# 多头自注意力
attended, attention_weights = MultiheadAttention(
    modalities, modalities, modalities,
    num_heads=4
)
# attended: [batch, 3, 64]

# 残差连接 + Layer Norm
attended = LayerNorm(attended + modalities)

# 展平
fused = attended.flatten(start_dim=1)  # [batch, 192]

# 通过共享MLP
shared_output = shared_mlp(fused)  # [192] → [128]
```

**优点**: 
- 自动学习模态间的重要性
- 可以可视化注意力权重
- 性能通常更好

**注意力权重示例**：
```
     Ego   Partner  Road
Ego   0.5    0.3    0.2   ← 自车关注自己和其他车
Partner 0.2  0.6    0.2   ← 其他车主要自注意
Road  0.1    0.1    0.8   ← 道路图主要自注意
```

##### 方式 3: Adaptive Fusion（自适应融合）

```python
fusion_type = "adaptive"

# 学习每个模态的权重
alpha_ego = sigmoid(W_ego(ego_embed))
alpha_partner = sigmoid(W_partner(partner_embed))
alpha_road = sigmoid(W_road(road_embed))

# 加权融合
fused = alpha_ego * ego_embed + 
        alpha_partner * partner_embed + 
        alpha_road * road_embed
# [64]

# 通过共享MLP
shared_output = shared_mlp(fused)  # [64] → [128]
```

**优点**: 参数少，计算快  
**缺点**: 表达能力不如注意力机制

#### 4. 策略头和价值头

```
Shared Output [128]
    │
    ├──► Policy Head (Actor)
    │    Linear(128 → 91)
    │    ↓
    │    Action Logits [91]
    │    ↓
    │    Softmax → Action Distribution
    │
    └──► Value Head (Critic)
         Linear(128 → 1)
         ↓
         State Value V(s)
```

**动作空间**:
- 离散动作空间：7 × 13 = 91 种组合
- 加速度：[-4.0, 4.0] m/s² (7档)
- 转向角：[-π, π] rad (13档)

### 模型参数统计

| 组件 | 参数量 | 占比 |
|-----|--------|------|
| Ego Encoder | ~8K | 12% |
| Partner Encoder | ~16K | 24% |
| Road Encoder | ~20K | 29% |
| Attention Fusion | ~10K | 15% |
| Shared MLP | ~10K | 15% |
| Policy Head | ~3K | 4% |
| Value Head | ~0.1K | 0.1% |
| **总计** | **~68K** | **100%** |

**对比**：
- GPUDrive 模型：68K 参数
- 典型 CNN (ResNet-18)：11M 参数
- 大型 Transformer：100M+ 参数

→ GPUDrive 模型**极其轻量**，适合快速推理和大规模并行

---

## 强化学习算法

### PPO (Proximal Policy Optimization)

GPUDrive 使用 **IPPO (Independent PPO)** 的多智能体变体。

#### 为什么选择 PPO？

| 算法 | 样本效率 | 稳定性 | 并行性 | 适用性 |
|-----|---------|--------|--------|--------|
| **PPO** ✅ | 中等 | **高** | **优秀** | **连续+离散** |
| SAC | 高 | 中等 | 良好 | 仅连续动作 |
| DQN | 低 | 中等 | 良好 | 仅离散动作 |
| A3C | 低 | 低 | 优秀 | 连续+离散 |

**PPO 的关键优势**：
1. **稳定性高**：通过裁剪目标限制更新幅度
2. **并行友好**：天然支持多环境采样
3. **易于调试**：超参数鲁棒性强
4. **工业标准**：OpenAI、DeepMind 广泛使用

#### PPO 算法流程

```
1. 策略采样阶段
   ┌─────────────────────┐
   │ for step in batch:  │
   │   action = π(s)     │  ← 当前策略采样动作
   │   s', r = env.step  │  ← 环境执行
   │   save (s,a,r,s')   │  ← 存入 buffer
   └─────────────────────┘
   
2. 优势估计（GAE）
   ┌─────────────────────┐
   │ V(s) = Critic(s)    │  ← 价值函数估计
   │ δ_t = r + γV(s') - V(s) │  ← TD误差
   │ A_t = Σ(γλ)^k δ_{t+k}  │  ← GAE优势
   └─────────────────────┘
   
3. 策略更新（多轮）
   ┌─────────────────────┐
   │ for epoch in K:     │  ← 重复K次
   │   ratio = π_new/π_old│  ← 重要性比率
   │   L_clip = min(     │
   │     ratio * A,      │  ← 未裁剪目标
   │     clip(ratio)*A   │  ← 裁剪目标
   │   )                 │
   │   L_value = MSE     │  ← 价值损失
   │   L = L_clip + L_V  │  ← 总损失
   │   Backprop & Update │
   └─────────────────────┘
```

#### 关键超参数

```yaml
# PPO 核心参数
gamma: 0.99                  # 折扣因子
gae_lambda: 0.95            # GAE λ参数
clip_coef: 0.2              # PPO裁剪系数
update_epochs: 4            # 每批数据更新轮数

# 训练参数
batch_size: 16384           # 经验批大小
minibatch_size: 2048        # 小批次大小
learning_rate: 3e-4         # 学习率
ent_coef: 0.0001           # 熵正则化系数

# 价值函数
vf_coef: 0.3               # 价值损失系数
clip_vloss: false          # 是否裁剪价值损失
vf_clip_coef: 0.2          # 价值裁剪系数

# 优化器
max_grad_norm: 0.5         # 梯度裁剪
```

#### GAE (Generalized Advantage Estimation)

**优势函数**用于衡量某个动作比平均水平好多少：

```python
# 标准优势
A(s,a) = Q(s,a) - V(s) = r + γV(s') - V(s)

# GAE 优势（平滑版）
A^GAE(s,a) = Σ_{t=0}^∞ (γλ)^t δ_{t+l}

其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**λ 的作用**：
- λ = 0：只用1步TD误差（低方差，高偏差）
- λ = 1：用完整回报（高方差，低偏差）
- λ = 0.95：**平衡**方差和偏差 ✅

#### 重要性采样与裁剪

**问题**：策略更新后，旧数据不再适用

**解决**：重要性采样 + 裁剪

```python
# 重要性比率
ratio = π_θ_new(a|s) / π_θ_old(a|s)

# PPO 裁剪目标
L^CLIP = E[ min(
    ratio * A,                           # 未裁剪
    clip(ratio, 1-ε, 1+ε) * A           # 裁剪到 [0.8, 1.2]
)]
```

**效果**：
- ratio ≈ 1：新旧策略相似，正常更新
- ratio > 1+ε：新策略过于激进，裁剪限制
- ratio < 1-ε：新策略过于保守，裁剪限制

→ 保证每次更新不会偏离太远，训练稳定

---

## 奖励函数设计

### 奖励函数类型对比

GPUDrive 支持多种奖励函数，适应不同的训练目标。

| 类型 | 碰撞惩罚 | 越界惩罚 | 到达奖励 | 特点 | 适用场景 |
|-----|---------|---------|---------|------|---------|
| **sparse_on_goal_achieved** | ❌ | ❌ | ✅ 一次性 | 极度稀疏 | 基线测试 |
| **weighted_combination** | ✅ 每步 | ✅ 每步 | ✅ 每步 | 密集，易刷分 | 快速收敛 |
| **distance_to_logs** | ✅ 每步 | ✅ 每步 | ✅ 每步 + 轨迹 | 行为克隆引导 | 模仿学习 |
| **safe_arrival** (已删除) | ✅ 大 | ✅ 中 | ✅ 分级 | 严格安全 | 零容忍场景 |
| **safe_driving_score** (已删除) | ✅ 扣分 | ✅ 扣分 | ✅ 根据分数 | 积分制 | 渐进惩罚 |

### 1. sparse_on_goal_achieved（稀疏奖励）

```python
reward = 1.0  if  distance_to_goal < threshold  else  0.0
```

**特点**：
- ✅ 最简单，无需调参
- ❌ 极难探索（早期几乎全是0奖励）
- ❌ 无显式碰撞惩罚

**何时使用**：
- 基线对比
- 已有预训练模型，只需微调

**训练曲线**：
```
Reward
  │          ___
  │         /
  │        /
  │_______/  ← 长时间探索期
  └──────────
    0  20  40M steps
```

### 2. weighted_combination（加权组合）⭐

```python
reward = collision_weight * collided          # 例如 -0.75 * 1
       + off_road_weight * off_road           # 例如 -0.75 * 1
       + goal_achieved_weight * goal_achieved # 例如 +1.0 * 1
```

**特点**：
- ✅ 每步都有反馈，易于学习
- ✅ 可调权重平衡安全与效率
- ⚠️ 到达后会"驻留刷分"（每步+1.0）

**参数推荐**：

| 目标 | collision_weight | off_road_weight | goal_achieved_weight |
|-----|-----------------|-----------------|---------------------|
| 快速到达（不太安全） | -0.5 | -0.5 | 1.0 |
| 平衡 ✅ | -1.0 | -0.6 | 0.3 |
| 极度安全 | -2.0 | -2.0 | 0.1 |

**训练曲线**：
```
Reward
  │    ___---
  │  _/
  │ /
  │/   ← 快速收敛
  └──────────
    0  10  20M steps
```

**"驻留刷分"问题示例**：

```
智能体在第60步到达终点，之后停留31步

reward_60 = +1.0  (到达)
reward_61 = +1.0  (仍在终点区域)
reward_62 = +1.0
...
reward_90 = +1.0

总奖励 = 31 × 1.0 = 31.0 😱
```

**解决方案**：
1. 降低 `goal_achieved_weight` 到 0.1-0.3
2. 添加"到达后停留惩罚"（需修改代码）
3. 使用 sparse 或其他类型

### 3. distance_to_logs（轨迹引导）

```python
base_reward = weighted_combination  # 基础奖励

# 额外奖励：靠近人类轨迹
dist_to_human = ||agent_pos - human_traj[t]||
trajectory_bonus = log_distance_weight * exp(-dist_to_human)

reward = base_reward + trajectory_bonus
```

**特点**：
- ✅ 利用人类演示数据
- ✅ 加速早期探索
- ⚠️ 可能过拟合人类行为

**参数推荐**：
```yaml
log_distance_weight: 0.01  # 0.005-0.05 之间
collision_weight: -1.5
off_road_weight: -1.0
goal_achieved_weight: 0.2
```

**何时使用**：
- 早期训练阶段（前10-20M步）
- 复杂场景（多车交互、复杂路口）
- 需要人类风格的驾驶

**训练策略**：
1. 前20M步：distance_to_logs（学习基本行为）
2. 后80M步：weighted_combination（优化性能）

---

### 奖励函数选择指南

```
开始
  │
  ├─> 需要快速原型？
  │   └─> 是：weighted_combination (权重默认)
  │
  ├─> 已有预训练模型？
  │   └─> 是：sparse_on_goal_achieved
  │
  ├─> 想利用人类数据？
  │   └─> 是：distance_to_logs
  │
  └─> 追求极致安全？
      └─> 配置高惩罚的 weighted_combination
          collision_weight: -2.0
          off_road_weight: -2.0
          goal_achieved_weight: 0.1
```

---

## 训练流程

### 完整训练流程图

```
1. 环境初始化
   ┌──────────────────────┐
   │ 加载 Waymo 场景数据   │
   │ 创建并行环境          │  ← 16个世界 × 64智能体
   │ 初始化模拟器         │
   └──────────────────────┘
              ↓
2. 模型初始化
   ┌──────────────────────┐
   │ 构建 NeuralNet       │
   │ 初始化优化器 (Adam)   │  ← lr=3e-4
   │ 加载检查点 (可选)     │
   └──────────────────────┘
              ↓
3. 训练循环 (100M steps)
   ┌──────────────────────┐
   │ ┌──> 采样阶段        │
   │ │    - 策略推理       │  ← batch_size=16384
   │ │    - 环境step       │
   │ │    - 存储经验       │
   │ │                    │
   │ ├──> GAE计算         │  ← gamma=0.99, λ=0.95
   │ │    - 价值估计       │
   │ │    - 优势计算       │
   │ │                    │
   │ └──> PPO更新         │  ← 4 epochs
   │      - 策略损失       │
   │      - 价值损失       │
   │      - 梯度下降       │
   │                      │
   │ ┌──> 日志记录         │  ← 每1000步
   │ │    - WandB上传     │
   │ │    - 指标统计       │
   │ │                    │
   │ ├──> 场景重采样       │  ← 每2M步
   │ │    - 从64个场景重采  │
   │ │    - 增加多样性     │
   │ │                    │
   │ └──> 保存检查点       │  ← 每100次更新
   │      - 模型参数       │
   │      - 优化器状态     │
   │      - 训练进度       │
   └──────────────────────┘
              ↓
4. 训练完成
   ┌──────────────────────┐
   │ 保存最终模型         │
   │ 同步 WandB 数据      │
   │ 评估性能             │
   └──────────────────────┘
```

### 关键训练指标

#### 1. 性能指标

| 指标 | 说明 | 目标值 | 当前值示例 |
|-----|------|--------|----------|
| **SPS** | 每秒训练步数 | >1000 | 820-1500 |
| **mean_episode_reward** | 平均回合奖励 | >5.0 | 0.911 |
| **perc_goal_achieved** | 到达率 | >80% | 98.3% |
| **perc_veh_collisions** | 车辆碰撞率 | <5% | 1.1% |
| **perc_off_road** | 越界率 | <5% | 0.9% |

#### 2. 算法指标

| 指标 | 说明 | 健康范围 |
|-----|------|---------|
| **policy_loss** | 策略损失 | -0.01 ~ -0.001 |
| **value_loss** | 价值损失 | 0.001 ~ 0.1 |
| **entropy** | 策略熵 | 0.5 ~ 2.0 |
| **approx_kl** | KL散度 | <0.03 |
| **clipfrac** | 裁剪比例 | 0.05 ~ 0.15 |
| **explained_variance** | 解释方差 | >0.5 |

**指标解读**：

```python
# 好的训练状态
entropy: 1.131           # 足够探索
approx_kl: 0.012         # 更新适度
clipfrac: 0.099          # ~10% 被裁剪
explained_variance: 0.578 # 价值函数拟合良好

# 异常状态警告
entropy < 0.3            # ⚠️ 策略过于确定，可能陷入局部最优
approx_kl > 0.05         # ⚠️ 更新过激进，可能不稳定
explained_variance < 0   # ⚠️ 价值函数拟合失败
```

### 训练脚本使用

```bash
# 基础训练
python baselines/ppo/my_ppo_pufferlib.py

# 继续训练
python baselines/ppo/my_ppo_pufferlib.py \
    --continue-training \
    --model-cpt path/to/checkpoint.pt

# 自定义参数
python baselines/ppo/my_ppo_pufferlib.py \
    --num-worlds 16 \
    --k-unique-scenes 64 \
    --learning-rate 3e-4 \
    --batch-size 16384
```

### 场景重采样机制

```python
# 每隔 resample_interval 步重新采样场景
if global_step % resample_interval == 0:
    # 从 10,000 个场景中随机采样 64 个
    new_scenes = sample(all_scenes, k=64)
    # 加载新场景到 16 个并行世界
    env.reset(new_scenes)
```

**作用**：
- 增加数据多样性
- 防止过拟合特定场景
- 模拟课程学习

**参数推荐**：
```yaml
resample_interval: 2_000_000  # 2M steps
resample_dataset_size: 10_000  # 10K 场景池
k_unique_scenes: 64            # 每次采样 64 个
```

---

## 关键配置参数

### 环境配置 (environment)

```yaml
environment:
  # 基础设置
  num_worlds: 16              # 并行环境数量
  k_unique_scenes: 64         # 唯一场景数（建议 4×num_worlds）
  max_controlled_agents: 64   # 每个场景最大可控智能体
  
  # 观察空间
  ego_state: true             # 自车状态
  road_map_obs: true          # 道路图观察
  partner_obs: true           # 其他车辆观察
  norm_obs: true              # 归一化观察
  lidar_obs: false            # LiDAR观察（实验性）
  
  # 奖励函数
  reward_type: "weighted_combination"
  collision_weight: -0.75     # 碰撞惩罚
  off_road_weight: -0.75      # 越界惩罚
  goal_achieved_weight: 1.0   # 到达奖励
  
  # 物理与碰撞
  dynamics_model: "classic"   # 动力学模型
  collision_behavior: "ignore" # 碰撞行为: ignore/stop/remove
  dist_to_goal_threshold: 2.0 # 到达阈值 (米)
  
  # 观察细节
  obs_radius: 50.0            # 观察半径 (米)
  polyline_reduction_threshold: 0.1  # 道路点采样密度
  
  # 动作空间
  action_space_steer_disc: 13 # 转向离散化数量
  action_space_accel_disc: 7  # 加速度离散化数量
```

### 训练配置 (train)

```yaml
train:
  # 基础设置
  seed: 42
  device: "cuda"              # cuda 或 cpu
  total_timesteps: 100_000_000 # 总训练步数
  
  # 数据采样
  resample_scenes: true
  resample_dataset_size: 10_000
  resample_interval: 2_000_000 # 场景重采样间隔
  sample_with_replacement: true
  
  # PPO 超参数
  batch_size: 16_384          # 经验批大小
  minibatch_size: 2_048       # 小批次大小
  learning_rate: 3e-4         # 学习率
  anneal_lr: false            # 是否衰减学习率
  gamma: 0.99                 # 折扣因子
  gae_lambda: 0.95            # GAE λ
  update_epochs: 4            # 每批更新轮数
  norm_adv: true              # 优势归一化
  clip_coef: 0.2              # PPO 裁剪系数
  ent_coef: 0.0001            # 熵正则化系数
  vf_coef: 0.3                # 价值损失系数
  max_grad_norm: 0.5          # 梯度裁剪
  
  # 网络架构
  network:
    input_dim: 64
    hidden_dim: 128
    dropout: 0.01
    fusion_type: "attention"   # simple/attention/adaptive
    num_attention_heads: 4
  
  # 检查点
  checkpoint_interval: 100    # 保存间隔 (轮次)
  checkpoint_path: "./runs"
  
  # 可视化
  render: false
  render_k_scenarios: 0       # 渲染场景数 (0=禁用)
```

### WandB 配置

```yaml
wandb:
  entity: "your_username"
  project: "gpudrive"
  group: "experiment_group"
  mode: "online"              # online/offline/disabled
  tags: ["ppo", "attention"]
```

---

## 性能优化

### 1. 显存优化（8GB GPU）

```yaml
# 问题：OOM (Out of Memory)
# 解决方案：

environment:
  num_worlds: 16              # ↓ 从 25 降到 16
  k_unique_scenes: 64         # ↓ 从 100 降到 64
  render_k_scenarios: 0       # ↓ 禁用可视化

train:
  batch_size: 16_384          # ↓ 从 32768 降到 16384
  resample_interval: 2_000_000 # ↑ 增加间隔减少峰值
```

**显存使用分解**（16 worlds × 64 agents）：

| 组件 | 显存占用 | 说明 |
|-----|---------|------|
| 模拟器状态 | ~1.5 GB | 位置、速度、道路图 |
| 观察缓存 | ~0.8 GB | batch_size=16384 |
| 神经网络 | ~0.3 GB | 参数+激活 |
| 梯度+优化器 | ~0.6 GB | Adam 状态 |
| 可视化系统 | ~2.0 GB | (如启用) |
| 其他 | ~0.5 GB | CUDA上下文等 |
| **总计** | **~3.7 GB** | (不含可视化) |

### 2. 训练吞吐优化

| 优化项 | 方法 | SPS 提升 |
|-------|------|---------|
| **编译模型** | `torch.compile` | +20-30% |
| **批大小** | 增大到 32768 | +15% |
| **零拷贝** | `zero_copy: true` | +10% |
| **CPU卸载** | `cpu_offload: true` | -20% ❌ |

```yaml
# 高性能配置
train:
  compile: true
  compile_mode: "reduce-overhead"
  batch_size: 32_768          # 需要 >12GB VRAM
  
vec:
  zero_copy: true             # 减少 CPU<->GPU 拷贝
```

### 3. WandB 网络优化

```yaml
# 问题：训练卡住，GPU利用率降为0
# 原因：WandB上传阻塞主线程

# 解决方案1：禁用WandB
wandb:
  mode: "disabled"

# 解决方案2：离线模式
wandb:
  mode: "offline"             # 稍后手动同步

# 解决方案3：减少日志频率
train:
  log_window: 5000            # 从 1000 增加到 5000
  checkpoint_interval: 200    # 从 100 增加到 200
```

### 4. 数据加载优化

```python
# 场景重采样时的OOM
# 解决方案：分批加载

resample_interval: 3_000_000  # 增加间隔
k_unique_scenes: 64           # 减少场景数
```

---

## 常见问题与调优

### 问题 1: 训练不收敛（奖励不上升）

**症状**：
```
Epoch 1000: reward = 0.1
Epoch 2000: reward = 0.2
Epoch 5000: reward = 0.15  ← 波动，不增长
```

**可能原因与解决**：

| 原因 | 诊断 | 解决方案 |
|-----|------|---------|
| **奖励过于稀疏** | goal_achieved<5% | 切换到 weighted_combination |
| **学习率过高** | approx_kl>0.05, 损失震荡 | lr: 3e-4 → 1e-4 |
| **批大小过小** | 更新噪声大 | batch_size: 8192 → 16384 |
| **价值函数拟合差** | explained_variance<0 | vf_coef: 0.3 → 0.5 |
| **探索不足** | entropy<0.3 | ent_coef: 0.0001 → 0.001 |

**推荐调整**：
```yaml
train:
  learning_rate: 1e-4         # ↓ 降低学习率
  ent_coef: 0.0005           # ↑ 增加探索
  batch_size: 16384          # ↑ 增大批次
  
environment:
  reward_type: "weighted_combination"  # 使用密集奖励
```

### 问题 2: 智能体会撞墙/越界

**症状**：
```
perc_off_road: 30-40%        # 越界率高
perc_veh_collisions: 20%     # 碰撞率高
```

**解决方案（按优先级）**：

1. **提高惩罚权重**
```yaml
environment:
  collision_weight: -2.0      # ↑ 从 -0.75 提高
  off_road_weight: -2.0       # ↑ 从 -0.75 提高
  goal_achieved_weight: 0.3   # ↓ 从 1.0 降低
```

2. **改变碰撞行为**
```yaml
environment:
  collision_behavior: "stop"  # 从 "ignore" 改为 "stop"
  # 或 "remove"：更严格，碰撞直接出局
```

3. **增加折扣因子**
```yaml
train:
  gamma: 0.995                # ↑ 从 0.99 提高
  # 更重视长期安全
```

4. **使用轨迹引导**
```yaml
environment:
  reward_type: "distance_to_logs"
  log_distance_weight: 0.02   # 引导沿人类轨迹
```

### 问题 3: 到达后"驻留刷分"

**症状**：
```
智能体提前到达，然后停留刷分
episode_reward: 50-100  ← 异常高
episode_length: 90  ← 总是跑满
```

**解决方案**：

1. **降低到达奖励权重**
```yaml
goal_achieved_weight: 0.2    # ↓ 从 1.0 大幅降低
```

2. **切换到稀疏奖励**
```yaml
reward_type: "sparse_on_goal_achieved"  # 只在到达时给奖励
```

3. **修改代码加驻留惩罚**（需要改 Python）
```python
# 在 weighted_combination 分支中
if goal_achieved > 0:
    reward -= 0.05  # 到达后每步扣 0.05
```

### 问题 4: 继续训练性能下降

**症状**：
```
加载检查点继续训练
Before: reward = 5.0
After 10M: reward = 3.0  ← 反而下降
```

**原因**：
- 优化器状态丢失
- 学习率过高
- 探索不足

**解决方案**：
```yaml
# 使用专门的继续训练配置
train:
  learning_rate: 1e-4         # ↓ 降低3倍
  anneal_lr: true             # 启用学习率衰减
  clip_coef: 0.15             # ↓ 更保守的更新
  ent_coef: 0.0005            # ↑ 增加探索
  resample_interval: 3_000_000 # ↑ 稳定数据分布
```

### 问题 5: GPU 利用率突然降为 0%

**症状**：
```
训练正常
GPU: 95% → 0%  ← 突然降到0
程序未崩溃，但不再训练
```

**原因**：WandB 网络阻塞

**解决方案**：
```yaml
wandb:
  mode: "disabled"            # 禁用 WandB

# 或减少上传频率
train:
  checkpoint_interval: 200    # ↑ 从 100 增加
```

### 问题 6: 场景重采样时 OOM

**症状**：
```
Resampling scenarios at step 2M
CUDA_ERROR_OUT_OF_MEMORY  ← 崩溃
```

**原因**：重采样时显存峰值

**解决方案**：
```yaml
train:
  resample_interval: 3_000_000  # ↑ 增加间隔
  batch_size: 16_384            # ↓ 减小批次
  
environment:
  k_unique_scenes: 64           # ↓ 减少场景数
  render_k_scenarios: 0         # 禁用可视化
```

---

## 调参 Cheat Sheet

### 快速诊断表

| 症状 | 可能原因 | 快速fix |
|-----|---------|---------|
| 奖励不涨 | 学习率/批大小/奖励设计 | lr↓, batch↑, 换reward |
| 碰撞率高 | 惩罚太轻 | collision_weight: -2.0 |
| 越界率高 | 惩罚太轻 | off_road_weight: -2.0 |
| 不到达 | 奖励太稀疏 | weighted_combination |
| 刷分问题 | 到达奖励太高 | goal_weight: 0.2 |
| OOM | 显存不足 | worlds↓, batch↓, render=0 |
| SPS低 | 批大小/编译 | batch↑, compile: true |
| GPU闲置 | WandB阻塞 | wandb.mode: disabled |
| 继续训练差 | lr太高/优化器丢失 | lr: 1e-4, 加载optimizer |

### 推荐配置模板

#### 1. 快速原型（8GB GPU）

```yaml
environment:
  num_worlds: 16
  k_unique_scenes: 64
  reward_type: "weighted_combination"
  collision_weight: -1.0
  off_road_weight: -0.6
  goal_achieved_weight: 0.3
  collision_behavior: "stop"

train:
  batch_size: 16_384
  learning_rate: 3e-4
  gamma: 0.99
  ent_coef: 0.0001
  resample_interval: 2_000_000
  
  network:
    fusion_type: "attention"
    num_attention_heads: 4
```

#### 2. 高性能（24GB GPU）

```yaml
environment:
  num_worlds: 64
  k_unique_scenes: 256
  
train:
  batch_size: 65_536
  compile: true
  compile_mode: "reduce-overhead"
```

#### 3. 安全优先

```yaml
environment:
  reward_type: "weighted_combination"
  collision_weight: -2.0
  off_road_weight: -2.0
  goal_achieved_weight: 0.1
  collision_behavior: "remove"
  
train:
  gamma: 0.995
  ent_coef: 0.0005
```

#### 4. 继续训练

```yaml
continue_training: true
model_cpt: "path/to/checkpoint.pt"

train:
  learning_rate: 1e-4         # 降低3倍
  anneal_lr: true
  clip_coef: 0.15
  ent_coef: 0.0005
  resample_interval: 3_000_000
```

---

## 最佳实践总结

### ✅ DO (推荐做法)

1. **从小规模开始**
   - 16 worlds × 64 scenes
   - 验证配置后再扩展

2. **使用 attention fusion**
   - 性能通常优于 simple
   - 可解释性更好

3. **监控关键指标**
   - SPS、goal_achieved、collision
   - approx_kl、entropy、explained_variance

4. **定期保存检查点**
   - 每100轮保存一次
   - 保留多个版本

5. **场景多样性**
   - resample_interval: 2-3M
   - dataset_size: 10K

6. **继续训练降低学习率**
   - lr: 3e-4 → 1e-4
   - anneal_lr: true

### ❌ DON'T (避免做法)

1. **不要过早追求极致安全**
   - 会导致智能体不动
   - 先保证能到达，再加严惩罚

2. **不要忽略显存限制**
   - 8GB GPU 不要超过16 worlds
   - 超过会频繁 OOM

3. **不要频繁改变配置**
   - 每次改变至少训练 10M 步
   - 否则无法看出效果

4. **不要在网络不稳时用 WandB online**
   - 会阻塞训练
   - 用 offline 或 disabled

5. **不要丢失优化器状态**
   - 继续训练必须加载 optimizer_state_dict
   - 否则性能会下降

6. **不要用过大的 goal_achieved_weight**
   - 会导致驻留刷分
   - 建议 ≤ 0.3

---

## 参考资源

### 论文

- [GPUDrive: Data-driven, multi-agent driving simulation at 1 million FPS](https://arxiv.org/abs/2408.01584) - ICLR 2025
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) - PPO 原论文
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) - GAE
- [PufferLib: Making Reinforcement Learning Libraries and Environments Play Nice](https://arxiv.org/abs/2406.12905)

### 代码

- [GPUDrive GitHub](https://github.com/Emerge-Lab/gpudrive)
- [Madrona Engine](https://madrona-engine.github.io/)
- [PufferLib](https://github.com/PufferAI/PufferLib)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)

### 数据集

- [Waymo Open Motion Dataset](https://waymo.com/open/)
- [GPUDrive Dataset (Mini)](https://huggingface.co/datasets/EMERGE-lab/GPUDrive_mini)
- [GPUDrive Dataset (Full)](https://huggingface.co/datasets/EMERGE-lab/GPUDrive)

### 预训练模型

- [Best Policy (10K scenarios)](https://huggingface.co/daphne-cornelisse/policy_S10_000_02_27)
- [Alternative Policy (1K scenarios)](https://huggingface.co/daphne-cornelisse/policy_S1000_02_27)

---

## 附录：术语表

| 术语 | 英文 | 解释 |
|-----|------|------|
| **SPS** | Steps Per Second | 每秒训练步数，衡量训练吞吐 |
| **PPO** | Proximal Policy Optimization | 近端策略优化算法 |
| **GAE** | Generalized Advantage Estimation | 广义优势估计 |
| **IPPO** | Independent PPO | 独立PPO，多智能体变体 |
| **ECS** | Entity-Component-System | 实体组件系统架构 |
| **VRAM** | Video RAM | 显存，GPU内存 |
| **OOM** | Out Of Memory | 内存/显存不足 |
| **KL散度** | KL Divergence | 衡量两个分布差异 |
| **熵** | Entropy | 策略随机性，衡量探索程度 |
| **折扣因子** | Discount Factor (γ) | 未来奖励的折扣率 |
| **批大小** | Batch Size | 每次更新使用的经验数量 |
| **轮次** | Epoch | 对同一批数据的更新次数 |
| **Late Fusion** | 晚期融合 | 多模态特征在后期合并 |
| **Attention** | 注意力机制 | 自动学习重要性权重 |

---

## 更新日志

- **2025-11-03**: 初始版本，包含完整的模型、RL、奖励函数文档
- 重点整理了注意力机制融合、PPO算法细节、奖励函数对比
- 添加了常见问题诊断和调参指南

---

**文档作者**: AI Assistant  
**项目维护**: EMERGE Lab  
**许可证**: MIT License

如有疑问或建议，欢迎在 [GitHub Issues](https://github.com/Emerge-Lab/gpudrive/issues) 提出！


