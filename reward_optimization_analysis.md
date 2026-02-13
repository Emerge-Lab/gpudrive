# GPUDrive 奖励函数优化分析

## 当前奖励函数架构分析

### 1. 当前实现的奖励类型

根据代码分析，当前系统支持以下几种奖励类型：

```python
# gpudrive/env/env_torch.py
def get_rewards(
    self,
    collision_weight=-0.5,
    goal_achieved_weight=1.0,
    off_road_weight=-0.5,
    world_time_steps=None,
    log_distance_weight=0.01,
):
```

**支持的奖励模式**：
1. `sparse_on_goal_achieved`: 稀疏奖励（只在到达目标时给奖励）
2. `weighted_combination`: 加权组合（碰撞、到达目标、偏离道路）
3. `reward_conditioned`: 条件奖励（每个智能体有不同的权重）
4. `distance_to_vdb_trajs`: 基于VBD轨迹距离的奖励
5. `distance_to_logs`: 基于日志距离的奖励

---

## 🔴 发现的主要问题

### 问题1: **奖励信号过于稀疏**

```python
# 当前实现
weighted_rewards = (
    collision_weight * collided        # -0.5 (离散事件)
    + goal_achieved_weight * goal_achieved  # 1.0 (稀疏)
    + off_road_weight * off_road       # -0.5 (离散事件)
)
```

**问题分析**：
- 碰撞、到达目标、离开道路都是**离散事件**
- 在大部分时间步中，奖励 ≈ 0
- 智能体难以获得有效的学习信号
- 导致探索效率低下

**影响**：
- 训练初期：随机探索，学习缓慢
- 收敛速度：需要更多样本才能学到有效策略
- 泛化能力：可能学到次优策略

### 问题2: **缺乏渐进性奖励（Progress Reward）**

当前奖励函数没有考虑：
- 到目标的距离变化
- 速度和加速度的合理性
- 轨迹的平滑度
- 与其他车辆的安全距离

### 问题3: **奖励缩放不当**

```python
# 当前权重
collision_weight = -0.5
goal_achieved_weight = 1.0
off_road_weight = -0.5
```

**问题**：
- 碰撞惩罚(-0.5) vs 目标奖励(1.0)：比例为 1:2
- 可能导致智能体冒险行为（冒险碰撞换取更快到达目标）
- 缺乏动态调整机制

### 问题4: **没有奖励正则化/标准化**

在PPO训练代码中：

```python
# gpudrive/integrations/puffer/ppo.py (line 256-258)
advantages_np = compute_gae(
    dones_np, values_np, rewards_np, config.gamma, config.gae_lambda
)
```

- 原始奖励直接用于GAE计算
- 没有奖励裁剪（reward clipping）
- 没有奖励归一化（reward normalization）
- 可能导致价值函数估计不稳定

### 问题5: **缺少时间惩罚**

```python
# 当前没有考虑时间成本
# 智能体可能学会"等待"策略，消耗大量时间步
```

**问题**：
- 没有鼓励快速完成任务
- 可能导致低效行为
- 训练效率低下

### 问题6: **注释掉的重要功能**

```python
# gpudrive/integrations/puffer/ppo.py (line 177-187)
# 这段代码被注释掉了！
# done_but_truncated = truncated & terminal
# if done_but_truncated.any():
#     terminal_obs = data.vecenv.last_obs[done_but_truncated]
#     with torch.no_grad():
#         _, _, _, terminal_value = policy(terminal_obs)
#     # Add discounted value to reward
#     reward[done_but_truncated] += config.gamma * terminal_value.squeeze(-1)
```

**问题**：
- 截断状态的价值估计被忽略
- 导致价值函数估计偏差
- 影响GAE计算的准确性

---

## ✅ 优化建议

### 优化1: **引入密集奖励（Dense Reward）**

```python
def get_dense_rewards(
    self,
    collision_weight=-1.0,
    goal_achieved_weight=10.0,
    off_road_weight=-0.5,
    progress_weight=0.1,      # 新增：进度奖励
    speed_weight=0.01,         # 新增：速度奖励
    smoothness_weight=0.01,    # 新增：平滑度奖励
    safety_distance_weight=0.05, # 新增：安全距离奖励
    time_penalty=-0.01,        # 新增：时间惩罚
):
    """改进的密集奖励函数"""
    
    # 原有的离散奖励
    basic_rewards = (
        collision_weight * collided
        + goal_achieved_weight * goal_achieved
        + off_road_weight * off_road 
        + car_angle_weight * car_angle 
    )
    
    # 1. 进度奖励：鼓励向目标移动
    current_dist = torch.norm(agent_pos - goal_pos, dim=-1)
    prev_dist = self.prev_goal_distance  # 需要存储上一步距离
    progress_reward = progress_weight * (prev_dist - current_dist)
    self.prev_goal_distance = current_dist
    
    # 2. 速度奖励：鼓励合理速度
    target_speed = 5.0  # m/s
    speed_error = torch.abs(agent_speed - target_speed)
    speed_reward = speed_weight * torch.exp(-speed_error)
    
    # 3. 平滑度奖励：惩罚急刹车和急转弯
    accel_change = torch.abs(current_accel - prev_accel)
    steer_change = torch.abs(current_steer - prev_steer)
    smoothness_reward = -smoothness_weight * (accel_change + steer_change)
    
    # 4. 安全距离奖励
    min_distance_to_others = compute_min_distance(agent_pos, other_agents_pos)
    safe_threshold = 2.0  # meters
    safety_reward = safety_distance_weight * torch.clamp(
        min_distance_to_others - safe_threshold, min=0
    )
    
    # 5. 时间惩罚：鼓励快速完成
    time_reward = time_penalty * torch.ones_like(collided)
    
    # 组合所有奖励
    total_rewards = (
        basic_rewards
        + progress_reward
        + speed_reward
        + smoothness_reward
        + safety_reward
        + time_reward
    )
    
    return total_rewards
```

### 优化2: **奖励正则化与归一化**

```python
class RewardNormalizer:
    """运行时奖励归一化器"""
    
    def __init__(self, gamma=0.99, epsilon=1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0
        
    def update(self, rewards):
        """更新统计信息"""
        batch_mean = rewards.mean()
        batch_var = rewards.var()
        batch_count = rewards.numel()
        
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        
        self.running_mean += delta * batch_count / total_count
        self.running_var = (
            (self.count * self.running_var + batch_count * batch_var) / total_count
            + (delta ** 2) * self.count * batch_count / (total_count ** 2)
        )
        self.count = total_count
        
    def normalize(self, rewards):
        """归一化奖励"""
        return (rewards - self.running_mean) / (np.sqrt(self.running_var) + self.epsilon)
```

在PPO中应用：

```python
# 在train函数中添加
with profile.train_misc:
    idxs = experience.sort_training_data()
    dones_np = experience.dones_np[idxs]
    values_np = experience.values_np[idxs]
    rewards_np = experience.rewards_np[idxs]
    
    # ✅ 新增：奖励裁剪
    rewards_np = np.clip(rewards_np, -10.0, 10.0)
    
    # ✅ 新增：奖励归一化（可选）
    # rewards_np = (rewards_np - rewards_np.mean()) / (rewards_np.std() + 1e-8)
    
    # 数值稳定性检查
    if np.isnan(rewards_np).any():
        print("Warning: NaN detected in rewards")
        rewards_np = np.nan_to_num(rewards_np, nan=0.0)
```

### 优化3: **恢复截断状态价值估计**

```python
# 在evaluate函数中恢复这段代码
with profile.eval_misc:
    value = value.flatten()
    
    # ✅ 恢复截断状态的价值估计
    done_but_truncated = truncated & terminal
    if done_but_truncated.any():
        terminal_obs = data.vecenv.last_obs[done_but_truncated]
        with torch.no_grad():
            if lstm_h is not None:
                _, _, _, terminal_value, _ = policy(terminal_obs, (h, c))
            else:
                _, _, _, terminal_value = policy(terminal_obs)
        # 添加折扣价值到奖励
        reward[done_but_truncated] += config.gamma * terminal_value.squeeze(-1)
```

### 优化4: **自适应奖励权重**

```python
class AdaptiveRewardWeights:
    """根据训练进度自适应调整奖励权重"""
    
    def __init__(self, initial_weights, total_steps):
        self.initial_weights = initial_weights
        self.total_steps = total_steps
        self.current_step = 0
        
    def get_weights(self):
        """获取当前权重"""
        progress = self.current_step / self.total_steps
        
        # 策略：训练初期更多探索奖励，后期更多目标奖励
        collision_weight = -0.5 - 0.5 * progress  # -0.5 → -1.0
        goal_weight = 1.0 + 9.0 * progress       # 1.0 → 10.0
        progress_weight = 0.2 * (1 - progress)   # 0.2 → 0.0
        
        return {
            'collision': collision_weight,
            'goal': goal_weight,
            'progress': progress_weight
        }
    
    def step(self):
        self.current_step += 1
```

### 优化5: **奖励整形（Reward Shaping）**

使用势函数（Potential-based Reward Shaping）保持最优策略不变：

```python
def potential_based_shaping(
    current_state,
    next_state,
    gamma=0.99
):
    """基于势函数的奖励整形"""
    
    def potential(state):
        """定义势函数：到目标的负距离"""
        dist_to_goal = torch.norm(state.pos - state.goal, dim=-1)
        return -dist_to_goal
    
    # F(s, s') = γ * Φ(s') - Φ(s)
    shaping_reward = gamma * potential(next_state) - potential(current_state)
    
    return shaping_reward
```

### 优化6: **课程学习（Curriculum Learning）**

```python
class RewardCurriculum:
    """渐进式难度调整"""
    
    def __init__(self, stages):
        self.stages = stages
        self.current_stage = 0
        
    def get_stage_config(self, success_rate):
        """根据成功率调整阶段"""
        if success_rate > 0.8 and self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            
        return self.stages[self.current_stage]

# 使用示例
curriculum = RewardCurriculum([
    # 阶段1：简单场景，密集奖励
    {
        'collision_weight': -0.3,
        'progress_weight': 0.5,  # 高进度奖励
        'scenario_complexity': 'easy'
    },
    # 阶段2：中等场景
    {
        'collision_weight': -0.7,
        'progress_weight': 0.2,
        'scenario_complexity': 'medium'
    },
    # 阶段3：困难场景，稀疏奖励
    {
        'collision_weight': -1.0,
        'progress_weight': 0.05,
        'scenario_complexity': 'hard'
    }
])
```

---

## 🎯 实施优先级

### 高优先级（立即实施）

1. **恢复截断状态价值估计**（简单但重要）
2. **添加奖励裁剪**（防止极端值）
3. **引入进度奖励**（显著提升样本效率）

### 中优先级（短期实施）

4. **奖励归一化**（提高训练稳定性）
5. **调整奖励权重比例**（避免冒险行为）
6. **添加时间惩罚**（提高效率）

### 低优先级（长期优化）

7. **自适应权重调整**（根据训练进度）
8. **课程学习**（复杂项目）
9. **高级奖励整形**（研究性工作）

---

## 📊 预期改进效果

| 优化项 | 预期改进 | 实施难度 |
|--------|---------|---------|
| 截断状态价值估计 | +5-10% 性能 | 低 |
| 奖励裁剪 | 提高稳定性 | 低 |
| 进度奖励 | +20-30% 样本效率 | 中 |
| 奖励归一化 | +10-15% 收敛速度 | 中 |
| 自适应权重 | +15-25% 最终性能 | 高 |
| 课程学习 | +30-50% 整体效果 | 高 |

---

## 🔧 代码修改位置

1. **奖励函数修改**：`gpudrive/env/env_torch.py` (line 458-553)
2. **PPO训练修改**：`gpudrive/integrations/puffer/ppo.py` (line 232-260, 177-187)
3. **配置文件**：`gpudrive/env/config.py` (添加新的奖励参数)
4. **新增工具**：创建 `gpudrive/env/reward_utils.py`（奖励归一化、整形等工具）

---

## 🧪 建议的实验流程

1. **基线测试**：使用当前奖励函数训练，记录性能指标
2. **逐步添加优化**：每次添加一项优化，对比效果
3. **消融实验**：测试每个组件的独立贡献
4. **超参数搜索**：优化奖励权重和系数
5. **最终验证**：在不同场景下测试泛化性能

---

## 总结

当前奖励函数的主要问题是**过于稀疏**和**缺乏渐进性引导**。通过引入密集奖励信号、奖励正则化、以及恢复被注释掉的重要功能，可以显著提升PPO的训练效率和最终性能。

建议优先实施**高优先级优化**，这些修改相对简单但能带来明显改进。

