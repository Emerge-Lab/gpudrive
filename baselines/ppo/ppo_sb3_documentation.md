# GPUDrive PPO 训练脚本详细文档

## 文件概述

`baselines/ppo/ppo_sb3.py` 是 GPUDrive 项目中基于 Stable-Baselines3 框架的 PPO（Proximal Policy Optimization）训练脚本，专门用于多智能体自动驾驶策略的训练。

## 核心功能

### 1. 主要组件

#### 1.1 配置管理
- **`load_config(config_path)`**: 加载 YAML 配置文件
- **`linear_schedule(initial_value)`**: 实现线性学习率调度

#### 1.2 模型选择
支持两种神经网络架构：
- **`LateFusionNet`**: 晚期融合网络（默认）
- **`FFN`**: 基础前馈网络

#### 1.3 训练环境
- **`SB3MultiAgentEnv`**: 多智能体环境封装
- **`IPPO`**: 改进的 PPO 算法实现

### 2. 训练流程详解

#### 2.1 环境配置
```python
env_config = dataclasses.replace(
    EnvConfig(),
    reward_type=exp_config.reward_type,
    episode_len=exp_config.episode_len,
    remove_non_vehicles=exp_config.remove_non_vehicles,
    polyline_reduction_threshold=exp_config.polyline_reduction_threshold,
    obs_radius=exp_config.observation_radius,
    collision_behavior=exp_config.collision_behavior,
)
```

#### 2.2 模型初始化
```python
model = IPPO(
    n_steps=exp_config.n_steps,           # 每次rollout的步数
    batch_size=exp_config.batch_size,     # 批次大小
    env=env,                              # 训练环境
    seed=exp_config.seed,                 # 随机种子
    device=exp_config.device,             # 计算设备
    mlp_class=exp_config.mlp_class,       # 神经网络类型
    policy=exp_config.policy,             # 策略类型
    gamma=exp_config.gamma,               # 折扣因子
    gae_lambda=exp_config.gae_lambda,     # GAE参数
    clip_range=exp_config.clip_range,     # PPO裁剪范围
    learning_rate=linear_schedule(exp_config.lr),  # 学习率调度
    ent_coef=exp_config.ent_coef,        # 熵系数
    n_epochs=exp_config.n_epochs,        # 每批次的训练轮数
)
```

### 3. 关键配置参数

#### 3.1 环境参数
- **`num_worlds`**: 并行环境数量（默认100）
- **`episode_len`**: 每个episode的长度（默认91步）
- **`observation_radius`**: 观测半径（默认50.0）
- **`collision_behavior`**: 碰撞处理方式（"ignore"/"remove"/"stop"）

#### 3.2 训练参数
- **`total_timesteps`**: 总训练步数（默认100M）
- **`n_steps`**: 每次rollout的步数（默认91）
- **`n_epochs`**: 每批次的训练轮数（默认5）
- **`lr`**: 学习率（默认0.0003）
- **`gamma`**: 折扣因子（默认0.99）

#### 3.3 网络参数
- **`mlp_class`**: 神经网络类型（"late_fusion"/"feed_forward"）
- **`ego_state_layers`**: 自车状态编码层
- **`road_object_layers`**: 道路对象编码层
- **`road_graph_layers`**: 路网编码层
- **`shared_layers`**: 共享编码层

### 4. 监控与日志

#### 4.1 Weights & Biases 集成
- **项目名称**: "gpudrive"
- **实验分组**: "my_experiment"
- **同步TensorBoard**: 启用
- **日志频率**: 每100步

#### 4.2 关键指标
- **目标达成率**: `perc_goal_achieved`
- **偏离道路率**: `perc_off_road`
- **车辆碰撞率**: `perc_veh_collisions`
- **非车辆碰撞率**: `perc_non_veh_collision`
- **平均奖励**: `mean_episode_reward_per_agent`

### 5. 回调机制

#### 5.1 `MultiAgentCallback`
- **策略保存**: 定期保存训练好的策略
- **指标记录**: 记录训练过程中的关键指标
- **可视化**: 支持观测统计和性能图表

#### 5.2 场景重采样
- **`resample_scenes`**: 是否启用场景重采样
- **`resample_interval`**: 重采样间隔（默认2M步）
- **`resample_dataset_size`**: 重采样数据集大小

### 6. 使用方式

#### 6.1 直接运行
```bash
python baselines/ppo/ppo_sb3.py
```

#### 6.2 自定义配置
修改 `baselines/ppo/config/ppo_base_sb3.yaml` 文件中的参数

#### 6.3 训练监控
- 通过 Weights & Biases 实时监控训练进度
- 查看 TensorBoard 日志：`tensorboard --logdir runs/`

### 7. 输出文件

#### 7.1 模型保存
- **路径**: `runs/{run_id}/policies/`
- **格式**: `policy_{timesteps}.zip`
- **频率**: 每200个rollout保存一次

#### 7.2 日志文件
- **TensorBoard**: `runs/{run_id}/`
- **W&B**: 自动同步到云端

### 8. 性能优化

#### 8.1 计算优化
- **GPU加速**: 支持CUDA设备
- **并行环境**: 多环境并行训练
- **批次处理**: 优化的批次大小计算

#### 8.2 内存优化
- **观测缓存**: 高效的观测数据管理
- **梯度累积**: 支持大批次训练

### 9. 扩展性

#### 9.1 网络架构
- 支持自定义神经网络架构
- 可扩展新的观测编码方式

#### 9.2 奖励函数
- 支持多种奖励类型
- 可自定义奖励权重

#### 9.3 环境配置
- 灵活的环境参数配置
- 支持不同的动力学模型

### 10. 配置文件详解

#### 10.1 数据配置
```yaml
data_dir: "data/processed/examples"  # 数据目录
num_worlds: 100                      # 并行环境数
k_unique_scenes: 4                   # 唯一场景数
```

#### 10.2 训练配置
```yaml
total_timesteps: 100_000_000         # 总训练步数
n_steps: 91                          # 每次rollout步数
n_epochs: 5                          # 每批次训练轮数
lr: 0.0003                           # 学习率
gamma: 0.99                          # 折扣因子
```

#### 10.3 网络配置
```yaml
mlp_class: "late_fusion"             # 网络类型
ego_state_layers: [64, 32]           # 自车状态层
road_object_layers: [64, 64]         # 道路对象层
road_graph_layers: [64, 64]          # 路网层
shared_layers: [64, 64]              # 共享层
```

### 11. 常见问题与解决方案

#### 11.1 内存不足
- 减少 `num_worlds` 或 `batch_size`
- 使用 CPU 训练：`device: "cpu"`

#### 11.2 训练不稳定
- 调整学习率：降低 `lr`
- 增加熵系数：提高 `ent_coef`
- 调整裁剪范围：修改 `clip_range`

#### 11.3 收敛慢
- 增加 `n_epochs`
- 调整网络架构
- 优化奖励函数

### 12. 最佳实践

#### 12.1 超参数调优
- 使用网格搜索或贝叶斯优化
- 监控验证集性能
- 早停机制防止过拟合

#### 12.2 实验管理
- 使用有意义的实验名称
- 记录所有超参数
- 定期保存检查点

#### 12.3 性能监控
- 实时监控训练指标
- 分析奖励分布
- 可视化策略行为

---

这个训练脚本为 GPUDrive 项目提供了完整的 PPO 训练流程，支持多智能体自动驾驶策略的训练、监控和评估。通过合理的配置和监控，可以训练出高质量的自动驾驶策略。 