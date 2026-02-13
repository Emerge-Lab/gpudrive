# 安全到达优先的奖励函数设计

## 目标

**优先级**：安全（无碰撞）> 到达终点 > 速度

**不关注**：尽快到达终点

## 当前问题分析

### 现有 `weighted_combination` 的问题

```python
# 当前设计
reward = (
    collision_weight * collided +      # -0.75 × 碰撞
    goal_achieved_weight * goal_achieved +  # 1.0 × 到达后每步
    off_road_weight * off_road         # -0.75 × 越界
)
```

**问题**：
1. **鼓励快速到达**：到达越早，停留时间越长，累积奖励越高（最多+91）
2. **碰撞惩罚太轻**：碰撞只扣0.75，但每步停留奖励+1.0
   - 如果碰撞1次但提前10步到达 → 奖励 = -0.75 + 10×1.0 = 9.25（仍然很高）
3. **不区分安全到达和冒险到达**：只要到达就持续奖励

## 推荐方案

### 方案1: 一次性安全到达奖励（最简单）⭐⭐⭐

**设计理念**：只在无碰撞地到达终点时给一次性大奖励

```python
# 修改 env_torch.py 的 get_rewards 方法

def get_rewards(self, ...):
    info_tensor = self.sim.info_tensor().to_torch().clone()
    off_road = info_tensor[:, :, 0].to(torch.float)
    collided = info_tensor[:, :, 1:3].to(torch.float).sum(axis=2)
    goal_achieved = info_tensor[:, :, 3].to(torch.float)
    
    if self.config.reward_type == "safe_arrival":
        # 跟踪每个智能体是否曾经碰撞
        if not hasattr(self, 'ever_collided'):
            self.ever_collided = torch.zeros_like(collided, dtype=torch.bool)
        
        # 更新碰撞记录
        self.ever_collided |= (collided > 0)
        
        # 跟踪是否已经给过到达奖励
        if not hasattr(self, 'arrival_rewarded'):
            self.arrival_rewarded = torch.zeros_like(goal_achieved, dtype=torch.bool)
        
        # 计算奖励
        reward = torch.zeros_like(goal_achieved)
        
        # 只在首次到达且从未碰撞时给奖励
        safe_arrival = (goal_achieved > 0) & (~self.ever_collided) & (~self.arrival_rewarded)
        reward[safe_arrival] = 10.0  # 大奖励！
        
        # 记录已奖励的智能体
        self.arrival_rewarded |= safe_arrival
        
        # 碰撞惩罚（每次）
        reward[collided > 0] = -2.0  # 重惩罚
        
        # 越界惩罚（较轻）
        reward[off_road > 0] -= 0.5
        
        return reward
```

**奖励特点**：
- 安全到达（无碰撞）：+10.0（一次性）
- 碰撞到达或未到达：0
- 每次碰撞：-2.0（累积惩罚）
- 每次越界：-0.5

**示例**：
```python
智能体A（完美）：
  无碰撞，第50步到达
  → 奖励 = 10.0（安全到达）
  
智能体B（有碰撞但到达）：
  第30步碰撞1次，第60步到达
  → 奖励 = -2.0（碰撞惩罚），0（到达但不奖励）
  → 总奖励 = -2.0
  
智能体C（快速但危险）：
  第10步碰撞2次，第20步到达
  → 奖励 = -2.0×2 = -4.0
  → 总奖励 = -4.0（尽管到达了）
```

### 方案2: 分级安全奖励（更细粒度）⭐⭐

```python
def get_rewards(self, ...):
    # ... (获取info_tensor)
    
    if self.config.reward_type == "graded_safe_arrival":
        if not hasattr(self, 'collision_count_tracker'):
            self.collision_count_tracker = torch.zeros_like(collided)
        
        if not hasattr(self, 'arrival_rewarded'):
            self.arrival_rewarded = torch.zeros_like(goal_achieved, dtype=torch.bool)
        
        # 累积碰撞次数
        self.collision_count_tracker += collided
        
        reward = torch.zeros_like(goal_achieved)
        
        # 首次到达时根据碰撞次数给分级奖励
        first_arrival = (goal_achieved > 0) & (~self.arrival_rewarded)
        
        if first_arrival.any():
            # 根据碰撞次数分级
            collision_counts = self.collision_count_tracker[first_arrival]
            
            # 完美（0次碰撞）
            reward[first_arrival & (collision_counts == 0)] = 10.0
            
            # 良好（1-2次碰撞）
            reward[first_arrival & (collision_counts >= 1) & (collision_counts <= 2)] = 5.0
            
            # 及格（3-5次碰撞）
            reward[first_arrival & (collision_counts >= 3) & (collision_counts <= 5)] = 2.0
            
            # 不及格（>5次碰撞）
            reward[first_arrival & (collision_counts > 5)] = 0.0
            
            self.arrival_rewarded |= first_arrival
        
        # 实时碰撞惩罚
        reward[collided > 0] -= 1.5
        reward[off_road > 0] -= 0.3
        
        return reward
```

**奖励特点**：
- 0次碰撞到达：+10.0
- 1-2次碰撞到达：+5.0
- 3-5次碰撞到达：+2.0
- >5次碰撞到达：0
- 每次碰撞：-1.5
- 每次越界：-0.3

### 方案3: 安全驾驶积分制（最复杂但最灵活）⭐⭐⭐

```python
def get_rewards(self, ...):
    # ... (获取info_tensor)
    
    if self.config.reward_type == "safe_driving_score":
        # 初始化安全分数（满分100）
        if not hasattr(self, 'safety_score'):
            self.safety_score = torch.ones_like(goal_achieved) * 100.0
        
        if not hasattr(self, 'arrival_rewarded'):
            self.arrival_rewarded = torch.zeros_like(goal_achieved, dtype=torch.bool)
        
        reward = torch.zeros_like(goal_achieved)
        
        # 扣安全分
        self.safety_score[collided > 0] -= 10.0   # 碰撞扣10分
        self.safety_score[off_road > 0] -= 5.0    # 越界扣5分
        self.safety_score = torch.clamp(self.safety_score, 0, 100)  # 限制在0-100
        
        # 到达时根据剩余安全分给奖励
        first_arrival = (goal_achieved > 0) & (~self.arrival_rewarded)
        
        if first_arrival.any():
            # 奖励 = 安全分 / 10
            # 100分 → 10.0奖励
            # 50分 → 5.0奖励
            # 0分 → 0奖励
            reward[first_arrival] = self.safety_score[first_arrival] / 10.0
            self.arrival_rewarded |= first_arrival
        
        # 小的即时惩罚（避免扣完分后乱开）
        reward[collided > 0] -= 0.1
        reward[off_road > 0] -= 0.05
        
        return reward
```

**奖励特点**：
- 初始安全分：100
- 每次碰撞：-10分
- 每次越界：-5分
- 到达时奖励 = 安全分/10（0-10分）
- 如果碰撞太多导致安全分为0，到达也没奖励

**示例**：
```python
智能体A：
  碰撞1次(-10)，越界2次(-10) → 安全分80
  到达 → 奖励 = 80/10 = 8.0
  
智能体B：
  碰撞5次(-50)，越界4次(-20) → 安全分30
  到达 → 奖励 = 30/10 = 3.0
  
智能体C：
  碰撞8次(-80)，越界6次(-30) → 安全分0
  到达 → 奖励 = 0/10 = 0.0（白跑了！）
```

## 实现步骤

### Step 1: 修改 `env_torch.py`

<function_calls>
<invoke name="read_file">
<parameter name="target_file">/home/wbk/gpudrive/gpudrive/env/env_torch.py
