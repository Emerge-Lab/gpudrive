
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPUDrive 预训练模型使用脚本
基于 04_use_pretrained_sim_agent.ipynb 复现
独立运行，不依赖 Docker
"""

# 抑制TensorFlow和CUDA警告消息
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制TensorFlow INFO和WARNING消息

import torch
import dataclasses
import sys
import math
from pathlib import Path
from typing import Callable
from datetime import datetime
import numpy as np
from gpudrive.env.config import EnvConfig
from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.datatypes.observation import GlobalEgoState

# 设置matplotlib支持中文
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

# 尝试设置中文字体，如果失败则使用默认字体（避免警告）
try:
    # 尝试使用系统中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'STHeiti', 'Arial Unicode MS']
    font_found = False
    for font_name in chinese_fonts:
        try:
            # 检查字体是否存在
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            if font_name in available_fonts:
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                font_found = True
                break
        except:
            continue
    
    # 如果没有找到中文字体，使用DejaVu Sans（支持基本字符）
    if not font_found:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    # 如果无法设置字体，使用默认设置
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 抑制字体相关的警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')

def steering_to_front_wheel(steering):
    """
    将动作空间的转向角(steering)转换为实际的前轮转角(δ)
    
    根据classic动力学模型：
    - tan_delta = tan(steering)
    - beta = atan(0.5 * tan_delta)
    - tan_front_wheel = cos(beta) * tan_delta
    - front_wheel_angle = atan(tan_front_wheel)
    
    Args:
        steering: 动作空间的转向角（弧度）
        
    Returns:
        front_wheel_angle: 实际的前轮转角（弧度）
    """
    tan_delta = math.tan(steering)
    beta = math.atan(0.5 * tan_delta)
    tan_front_wheel = math.cos(beta) * tan_delta
    front_wheel_angle = math.atan(tan_front_wheel)
    return front_wheel_angle

def add_trajectory_to_frame(fig, env, env_idx, control_mask, trajectories, current_step):
    """
    在当前帧上叠加智能体的轨迹（只显示到当前时间步）
    
    Args:
        fig: matplotlib figure对象
        env: 环境对象
        env_idx: 环境索引
        control_mask: 控制掩码 [max_agents]
        trajectories: 轨迹字典 {env_idx: {agent_idx: [(x, y, step), ...]}}
        current_step: 当前时间步
    """
    from gpudrive.datatypes.observation import GlobalEgoState
    
    # 获取图像的axes
    ax = fig.axes[0] if len(fig.axes) > 0 else None
    if ax is None:
        return
    
    # 获取当前环境的控制掩码
    env_control_mask = control_mask[env_idx]
    controlled_agents = torch.where(env_control_mask)[0]
    
    if len(controlled_agents) == 0:
        return
    
    # 获取当前状态信息（用于颜色编码）
    try:
        agent_states = GlobalEgoState.from_tensor(
            env.sim.absolute_self_observation_tensor(),
            backend="torch",
            device=env.device,
        )
        info = env.get_infos()
    except:
        return
    
    # 为每个智能体绘制轨迹（只显示到当前时间步）
    colors = plt.cm.tab10(np.linspace(0, 1, len(controlled_agents)))
    
    for i, agent_idx in enumerate(controlled_agents):
        agent_idx_item = agent_idx.item()
        
        if agent_idx_item not in trajectories[env_idx]:
            continue
        
        # 只获取到当前时间步的轨迹
        traj = [(x, y, step) for x, y, step in trajectories[env_idx][agent_idx_item] if step <= current_step]
        
        if len(traj) < 2:  # 至少需要2个点才能画线
            continue
        
        # 提取x, y坐标，并过滤异常值
        traj_x = []
        traj_y = []
        for point in traj:
            x, y = point[0], point[1]
            # 过滤异常位置值
            if abs(x) < 10000 and abs(y) < 10000:
                traj_x.append(x)
                traj_y.append(y)
        
        if len(traj_x) < 2:  # 过滤后至少需要2个点才能画线
            continue
        
        # 确定轨迹颜色（根据当前状态）
        try:
            is_collided = info.collided[env_idx, agent_idx].item() > 0
            is_offroad = info.off_road[env_idx, agent_idx].item() > 0
            is_goal = info.goal_achieved[env_idx, agent_idx].item() > 0
            
            if is_collided:
                color = 'red'
            elif is_offroad:
                color = 'orange'
            elif is_goal:
                color = 'green'
            else:
                color = colors[i % len(colors)]
        except:
            color = colors[i % len(colors)]
        
        # 绘制轨迹线（半透明，较细）
        ax.plot(traj_x, traj_y, color=color, linewidth=1.0, alpha=0.5, zorder=3)

def add_front_wheel_visualization(fig, env, env_idx, control_mask, action_values_dict):
    """
    在matplotlib图像上添加前轮转角可视化
    
    Args:
        fig: matplotlib figure对象
        env: 环境对象
        env_idx: 环境索引
        control_mask: 控制掩码 [max_agents]
        action_values_dict: 字典，键为(env_idx, agent_idx)，值为(action_idx, steering, front_wheel_angle)
    """
    import matplotlib.pyplot as plt
    from gpudrive.datatypes.observation import GlobalEgoState
    
    # 获取当前环境的智能体状态
    agent_states = GlobalEgoState.from_tensor(
        env.sim.absolute_self_observation_tensor(),
        backend="torch",
        device=env.device,
    )
    
    # 获取当前环境的控制掩码
    env_control_mask = control_mask[env_idx]
    controlled_agents = torch.where(env_control_mask)[0]
    
    if len(controlled_agents) == 0:
        return
    
    # 获取图像的axes
    ax = fig.axes[0] if len(fig.axes) > 0 else None
    if ax is None:
        return
    
    # 前轮转角箭头长度（根据车辆长度调整）
    arrow_length = 3.0  # 米
    
    for agent_idx in controlled_agents:
        key = (env_idx, agent_idx.item())
        if key not in action_values_dict:
            continue
            
        _, steering, front_wheel_angle = action_values_dict[key]
        
        # 获取车辆位置和朝向
        pos_x = agent_states.pos_x[env_idx, agent_idx].item()
        pos_y = agent_states.pos_y[env_idx, agent_idx].item()
        vehicle_yaw = agent_states.rotation_angle[env_idx, agent_idx].item()
        vehicle_length = agent_states.vehicle_length[env_idx, agent_idx].item()
        
        # 计算前轮中心位置（车辆前部）
        front_center_x = pos_x + (vehicle_length / 2) * math.cos(vehicle_yaw)
        front_center_y = pos_y + (vehicle_length / 2) * math.sin(vehicle_yaw)
        
        # 计算前轮转角的箭头方向（相对于车辆朝向）
        # 前轮转角是相对于车辆纵轴的角度
        front_wheel_direction = vehicle_yaw + front_wheel_angle
        
        # 绘制前轮转角箭头（红色，较粗）
        arrow_dx = arrow_length * math.cos(front_wheel_direction)
        arrow_dy = arrow_length * math.sin(front_wheel_direction)
        
        ax.arrow(
            front_center_x, front_center_y,
            arrow_dx, arrow_dy,
            head_width=0.8, head_length=0.6,
            fc='red', ec='red', linewidth=2.5,
            alpha=0.8, zorder=10,
            length_includes_head=True
        )
        
        # 添加文本标签显示角度（可选）
        label_x = front_center_x + arrow_dx * 1.3
        label_y = front_center_y + arrow_dy * 1.3
        front_wheel_deg = front_wheel_angle * 180 / math.pi
        ax.text(
            label_x, label_y,
            f'δ={front_wheel_deg:.1f}°',
            fontsize=8, color='red', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='red'),
            zorder=11
        )

# ==================== 配置参数 ====================
# 模型路径
model_path = "/home/wbk/gpudrive/runs/PPO__C__S_72__01_30_19_06_17_932/model_PPO__C__S_72__01_30_19_06_17_932_036620.pt"

# 预测轨迹开关
ENABLE_TRAJECTORY_PREDICTION = True  # True: 绘制预测轨迹, False: 不绘制
TRAJECTORY_HORIZON = 20  # 预测步数（仅在 ENABLE_TRAJECTORY_PREDICTION=True 时有效）

# 预测轨迹平滑开关（对 x,y,yaw,speed 做一致性平滑）
ENABLE_TRAJECTORY_SMOOTHING = True
SMOOTH_WINDOW = 7  # 奇数，越大越平滑（建议 5~11）

# 动作打印开关
ENABLE_ACTION_PRINT = False  # True: 打印动作信息, False: 不打印
ACTION_PRINT_INTERVAL = 5  # 每N步打印一次（仅在 ENABLE_ACTION_PRINT=True 时有效）
# =================================================
def setup_environment():
    """自动查找并切换到gpudrive项目根目录"""
    script_dir = Path(__file__).resolve().parent
    current = script_dir
    while current.name != "gpudrive":
        if current.parent == current:
            raise RuntimeError("未找到gpudrive项目根目录！")
        current = current.parent
    os.chdir(current)
    print(f"项目根目录: {current}")
    sys.path.insert(0, str(current))
    return current

def main():
    print("=== GPUDrive 预训练模型使用脚本 ===")
    print(f"预测轨迹绘制: {'✅ 开启' if ENABLE_TRAJECTORY_PREDICTION else '❌ 关闭'}")
    if ENABLE_TRAJECTORY_PREDICTION:
        print(f"预测步数: {TRAJECTORY_HORIZON} 步 ({TRAJECTORY_HORIZON * 0.1:.1f} 秒)")
    print(f"动作打印: {'✅ 开启' if ENABLE_ACTION_PRINT else '❌ 关闭'}")
    if ENABLE_ACTION_PRINT:
        print(f"打印间隔: 每 {ACTION_PRINT_INTERVAL} 步打印一次")
    
    # 设置环境
    project_root = setup_environment()
    
    try:
        # 导入必要的模块
        from huggingface_hub import PyTorchModelHubMixin, ModelCard
        from gpudrive.env.env_torch import GPUDriveTorchEnv
        from gpudrive.visualize.utils import img_from_fig
        from gpudrive.env.dataset import SceneDataLoader
        from gpudrive.utils.config import load_config
        
        print("所有模块导入成功")
        
    except ImportError as e:
        print(f"模块导入失败: {e}")
        print("请确保已安装所有依赖包")
        return
    
    # 1. 加载配置
    print("\n1. 加载配置...")
    try:
        config_path = project_root / "examples/experimental/config/reliable_agents_params"
        config = load_config(str(config_path))
        print("配置加载成功")
        print(f"最大控制智能体数: {config.max_controlled_agents}")
    except Exception as e:
        print(f"配置加载失败: {e}")
        return
    
    # 2. 设置参数
    max_agents = config.max_controlled_agents
    num_envs = 4
    device = "cuda"  # 使用 CPU 避免 GPU 问题
    
    print(f"使用设备: {device}")
    print(f"环境数量: {num_envs}")
    

    
    # 4. 创建数据加载器
    print("\n3. 创建数据加载器...")
    try:
        data_path = project_root / "data/processed/examples"
        train_loader = SceneDataLoader(
            root=str(data_path),
            batch_size=num_envs,
            dataset_size=100,
            sample_with_replacement=False,
        )
        print("数据加载器创建成功")
        print(f"数据路径: {data_path}")
    except Exception as e:
        print(f"数据加载器创建失败: {e}")
        return
    
    # 5. 创建环境配置
    print("\n4. 创建环境配置...")
    try:
        env_config = dataclasses.replace(
            EnvConfig(),
            ego_state=config.ego_state,
            road_map_obs=config.road_map_obs,
            partner_obs=config.partner_obs,
            reward_type=config.reward_type,
            norm_obs=config.norm_obs,
            dynamics_model=config.dynamics_model,
            collision_behavior=config.collision_behavior,
            dist_to_goal_threshold=config.dist_to_goal_threshold,
            polyline_reduction_threshold=config.polyline_reduction_threshold,
            remove_non_vehicles=config.remove_non_vehicles,
            lidar_obs=config.lidar_obs,
            disable_classic_obs=config.lidar_obs,
            obs_radius=config.obs_radius,
            steer_actions=torch.round(
                torch.linspace(-torch.pi, torch.pi, config.action_space_steer_disc), decimals=3
            ),
            accel_actions=torch.round(
                torch.linspace(-4.0, 4.0, config.action_space_accel_disc), decimals=3
            ),
        )
        print("环境配置创建成功")
    except Exception as e:
        print(f"环境配置创建失败: {e}")
        return
    # 6. 创建环境
    print("\n5. 创建仿真环境...")
    try:
        env = GPUDriveTorchEnv(
            config=env_config,
            data_loader=train_loader,
            max_cont_agents=config.max_controlled_agents,
            device=device,
        )
        print("仿真环境创建成功")
    except Exception as e:
        print(f"仿真环境创建失败: {e}")
        return

    from gymnasium.spaces import Box
    import numpy as np

    print("当前环境观测空间：", env.observation_space)
    
    # 辅助函数：预测未来轨迹（改进版：每一步都使用策略预测动作）
    def predict_trajectory(env, policy, filtered_obs, control_mask, horizon=20, device="cuda", include_yaw_speed=True):
        """
        预测受控智能体的未来轨迹（改进版：考虑环境动态变化）
        
        在每一步都使用策略模型预测动作，而不是假设动作保持不变。
        会更新观察中的ego state部分（速度、相对目标位置等），但保持其他观察不变。
        
        Args:
            env: 环境对象
            policy: 策略模型
            filtered_obs: 已过滤的观察 [total_controlled_agents, obs_dim]
            control_mask: 控制掩码 [num_worlds, max_agents]
            horizon: 预测步数
            device: 设备
            
        Returns:
            predicted_trajectories:
                - include_yaw_speed=False: [num_worlds, max_agents, horizon, 2] (x, y)
                - include_yaw_speed=True:  [num_worlds, max_agents, horizon, 6] (x, y, yaw, vx, vy, speed)
            
        Note:
            - 这是一个闭环预测：每一步都根据当前状态预测下一步动作
            - 观察中的ego state部分会更新（速度、相对目标位置）
            - partner_obs和road_map_obs保持不变（简化假设）
            - 预测时假设智能体不会碰撞（is_collided=0）
        """
        num_worlds = control_mask.shape[0]
        max_agents = control_mask.shape[1]
        num_controlled = control_mask.sum().item()
        
        if num_controlled == 0:
            out_dim = 6 if include_yaw_speed else 2
            return torch.zeros((num_worlds, max_agents, horizon, out_dim), device=device)
        
        out_dim = 6 if include_yaw_speed else 2
        predicted_trajectories = torch.zeros((num_worlds, max_agents, horizon, out_dim), device=device)
        
        # 获取当前智能体状态（全局状态）
        agent_states = GlobalEgoState.from_tensor(
            env.sim.absolute_self_observation_tensor(),
            backend="torch",
            device=device,
        )
        
        # 获取速度信息（从self_observation_tensor）
        self_obs = env.sim.self_observation_tensor().to_torch().to(device)
        all_speeds = self_obs[:, :, 0]  # [num_worlds, max_agents]
        
        # 获取动作空间映射
        action_keys = env.action_keys_tensor  # [action_dim, 3] (accel, steer, head)
        
        # 提取受控智能体的初始状态
        x = agent_states.pos_x[control_mask].clone()  # [num_controlled]
        y = agent_states.pos_y[control_mask].clone()
        yaw = agent_states.rotation_angle[control_mask].clone()
        speed = all_speeds[control_mask].clone()
        vehicle_lengths = agent_states.vehicle_length[control_mask].clone()
        
        # 避免除以零
        vehicle_lengths = torch.clamp(vehicle_lengths, min=0.1)
        
        # 存储初始状态（step=0）
        # 需要将 [num_controlled] 的数据映射回 [num_worlds, max_agents] 格式
        flat_idx = 0
        for world_idx in range(num_worlds):
            world_mask = control_mask[world_idx]
            num_in_world = world_mask.sum().item()
            if num_in_world > 0:
                agent_indices = torch.where(world_mask)[0]
                for i, agent_idx in enumerate(agent_indices):
                    xi = x[flat_idx + i]
                    yi = y[flat_idx + i]
                    yawi = yaw[flat_idx + i]
                    si = speed[flat_idx + i]
                    predicted_trajectories[world_idx, agent_idx, 0, 0] = xi
                    predicted_trajectories[world_idx, agent_idx, 0, 1] = yi
                    if include_yaw_speed:
                        vxi = si * torch.cos(yawi)
                        vyi = si * torch.sin(yawi)
                        predicted_trajectories[world_idx, agent_idx, 0, 2] = yawi
                        predicted_trajectories[world_idx, agent_idx, 0, 3] = vxi
                        predicted_trajectories[world_idx, agent_idx, 0, 4] = vyi
                        predicted_trajectories[world_idx, agent_idx, 0, 5] = si
                flat_idx += num_in_world
        
        # 获取目标位置（用于更新观察中的相对目标位置）
        goal_x = agent_states.goal_x[control_mask].clone()
        goal_y = agent_states.goal_y[control_mask].clone()
        
        # 获取目标距离阈值
        dist_to_goal_threshold = env.config.dist_to_goal_threshold
        
        # 跟踪每个智能体是否已到达目标
        reached_goal = torch.zeros(num_controlled, dtype=torch.bool, device=device)
        
        # 获取ego state在观察中的维度信息
        # 根据env_torch.py的_get_ego_state()，ego state通常包含：
        # [speed, vehicle_length, vehicle_width, rel_goal_x, rel_goal_y, is_collided]
        # 如果是reward_conditioned，还会包含reward_weights
        
        # 初始化当前观察（用于预测）
        current_obs = filtered_obs.clone()  # [num_controlled, obs_dim]
        
        # 预测未来轨迹（每一步都重新预测动作）
        dt = 0.1
        for step in range(1, horizon):
            # 检查是否所有智能体都已到达目标
            if reached_goal.all():
                break
            
            # 只对未到达目标的智能体进行预测
            active_mask = ~reached_goal
            if not active_mask.any():
                break
            # 使用策略预测当前步骤的动作（考虑当前状态）
            # 只对未到达目标的智能体预测动作
            with torch.no_grad():
                if active_mask.all():
                    # 所有智能体都未到达，正常预测
                    action_indices, _, _, _ = policy(current_obs, deterministic=True)
                else:
                    # 部分智能体已到达，只对未到达的预测
                    # 创建临时观察，只包含未到达目标的智能体
                    active_obs = current_obs[active_mask]
                    active_action_indices, _, _, _ = policy(active_obs, deterministic=True)
                    # 创建完整的action_indices，已到达的设为0（停止动作）
                    action_indices = torch.zeros(num_controlled, dtype=torch.int64, device=device)
                    action_indices[active_mask] = active_action_indices
            
            # 将动作索引转换为实际值
            action_values = action_keys[action_indices]  # [num_controlled, 3]
            accel = action_values[:, 0]  # 加速度
            steer = action_values[:, 1]  # 转向角
            
            # 对已到达目标的智能体，停止运动
            accel = torch.where(reached_goal, torch.zeros_like(accel), accel)
            steer = torch.where(reached_goal, torch.zeros_like(steer), steer)
            speed = torch.where(reached_goal, torch.zeros_like(speed), speed)
            
            # 使用经典动力学模型rollout
            v = torch.clamp(speed + 0.5 * accel * dt, min=0.0)
            tan_delta = torch.tan(steer)
            beta = torch.atan(0.5 * tan_delta)
            
            # 更新位置
            dx = v * torch.cos(yaw + beta) * dt
            dy = v * torch.sin(yaw + beta) * dt
            x = x + dx
            y = y + dy
            
            # 更新朝向
            w = v * torch.cos(beta) * tan_delta / vehicle_lengths
            yaw = yaw + w * dt
            
            # 更新速度
            speed = torch.clamp(speed + accel * dt, min=0.0)

            # 估计全局速度分量（沿运动方向 yaw+beta）
            vel_dir = yaw + beta
            vx = speed * torch.cos(vel_dir)
            vy = speed * torch.sin(vel_dir)
            
            # 检查是否到达目标
            dist_to_goal = torch.sqrt((goal_x - x)**2 + (goal_y - y)**2)
            newly_reached = (dist_to_goal < dist_to_goal_threshold) & (~reached_goal)
            reached_goal = reached_goal | newly_reached
            
            # 对于刚到达目标的智能体，将位置设置为目标位置（避免继续移动）
            x = torch.where(newly_reached, goal_x, x)
            y = torch.where(newly_reached, goal_y, y)
            
            # 更新观察中的ego state部分
            # 计算相对于车辆的目标位置（在车辆坐标系中）
            dx_to_goal = goal_x - x
            dy_to_goal = goal_y - y
            
            # 转换到车辆坐标系（相对于车辆朝向）
            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)
            rel_goal_x = dx_to_goal * cos_yaw + dy_to_goal * sin_yaw
            rel_goal_y = -dx_to_goal * sin_yaw + dy_to_goal * cos_yaw
            
            # 更新观察中的ego state特征
            # 假设ego state是观察的前几个特征，根据env._get_ego_state()的结构：
            # 标准情况：[speed, vehicle_length, vehicle_width, rel_goal_x, rel_goal_y, is_collided] (6个特征)
            # reward_conditioned情况：还会加上reward_weights (3个)，共9个特征
            
            # 尝试更新ego state相关特征
            # 注意：这里假设ego state是观察的前几个特征，实际可能需要根据配置调整
            ego_state_dim = 6  # 标准ego state维度
            if env.config.reward_type == "reward_conditioned":
                ego_state_dim = 9  # 包含reward_weights
            
            if current_obs.shape[1] >= ego_state_dim:
                # 更新速度（索引0）
                if env.config.norm_obs:
                    # 如果观察被归一化，需要知道归一化参数
                    # 简化处理：假设速度范围是[0, 20]，归一化到[-1, 1]
                    normalized_speed = (speed / 10.0) - 1.0  # 简化归一化
                    current_obs[:, 0] = torch.clamp(normalized_speed, -1.0, 1.0)
                else:
                    current_obs[:, 0] = speed
                
                # 更新相对目标位置（索引3和4）
                if env.config.norm_obs:
                    # 假设rel_goal范围是[-100, 100]，归一化到[-1, 1]
                    normalized_rel_goal_x = torch.clamp(rel_goal_x / 100.0, -1.0, 1.0)
                    normalized_rel_goal_y = torch.clamp(rel_goal_y / 100.0, -1.0, 1.0)
                    current_obs[:, 3] = normalized_rel_goal_x
                    current_obs[:, 4] = normalized_rel_goal_y
                else:
                    current_obs[:, 3] = rel_goal_x
                    current_obs[:, 4] = rel_goal_y
                
                # is_collided保持为0（预测时假设不碰撞）
                current_obs[:, 5] = 0.0
            
            # 注意：partner_obs和road_map_obs保持不变（因为我们无法预测其他智能体和路网的变化）
            # 这是一个简化假设，实际环境中这些也会变化
            
            # 将结果映射回 [num_worlds, max_agents] 格式
            flat_idx = 0
            for world_idx in range(num_worlds):
                world_mask = control_mask[world_idx]
                num_in_world = world_mask.sum().item()
                if num_in_world > 0:
                    agent_indices = torch.where(world_mask)[0]
                    for i, agent_idx in enumerate(agent_indices):
                        xi = x[flat_idx + i]
                        yi = y[flat_idx + i]
                        predicted_trajectories[world_idx, agent_idx, step, 0] = xi
                        predicted_trajectories[world_idx, agent_idx, step, 1] = yi
                        if include_yaw_speed:
                            predicted_trajectories[world_idx, agent_idx, step, 2] = yaw[flat_idx + i]
                            predicted_trajectories[world_idx, agent_idx, step, 3] = vx[flat_idx + i]
                            predicted_trajectories[world_idx, agent_idx, step, 4] = vy[flat_idx + i]
                            predicted_trajectories[world_idx, agent_idx, step, 5] = speed[flat_idx + i]
                    flat_idx += num_in_world
        
        return predicted_trajectories
    
    # 3. 加载预训练模型
    print("\n2. 加载预训练模型...")
    try:
        # 加载.pt模型文件
        print(f"正在加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 从checkpoint中提取模型架构信息
        model_arch = checkpoint["model_arch"]
        action_dim = checkpoint["action_dim"]
        
        print(f"模型架构: input_dim={model_arch['input_dim']} (每个模态), hidden_dim={model_arch['hidden_dim']}")
        print(f"动作维度: {action_dim}")
        print(f"最大控制智能体数: {config.max_controlled_agents}")
        print(f"观察半径: {config.obs_radius}")
        print(f"奖励类型: {config.reward_type}")
        print(f"VBD功能: {'启用' if hasattr(config, 'vbd_in_obs') and config.vbd_in_obs else '禁用'}")
        
        # 创建NeuralNet模型，使用训练时的完整配置
        # 注意：使用原始融合方式以兼容预训练模型
        sim_agent = NeuralNet(
            input_dim=model_arch["input_dim"],
            action_dim=action_dim,
            hidden_dim=model_arch["hidden_dim"],
            dropout=model_arch["dropout"],
            max_controlled_agents=config.max_controlled_agents,  # 使用训练配置
            obs_dim=2984,  # 观察维度
            config=config,  # 传递完整的环境配置
            # fusion_type="attention",  # 使用注意力融合方式（与预训练模型匹配）
            # num_attention_heads=4,  # 注意力头数
        ).to(device)
        
        # 加载模型参数
        sim_agent.load_state_dict(checkpoint["parameters"])
        sim_agent.eval()
        
        print("预训练模型加载成功")

    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
        
    # 7. 运行仿真
    print("\n6. 开始仿真运行...")
    try:
        next_obs = env.reset()
        control_mask = env.cont_agent_mask
        
        print(f"观察形状: {next_obs.shape}")
        print(f"控制掩码形状: {control_mask.shape}")
        print(f"初始控制智能体数: {control_mask.sum().item()}")
        
        # 初始化统计变量
        frames = {f"env_{i}": [] for i in range(num_envs)}
        total_rewards = torch.zeros((num_envs, max_agents), device=device)
        collision_count = torch.zeros((num_envs, max_agents), dtype=torch.int32, device=device)
        off_road_count = torch.zeros((num_envs, max_agents), dtype=torch.int32, device=device)
        goal_achieved = torch.zeros((num_envs, max_agents), dtype=torch.int32, device=device)  # 改为int32
        agent_alive = torch.ones((num_envs, max_agents), dtype=torch.bool, device=device)
        
        # 初始化轨迹记录：{env_idx: {agent_idx: [(x, y, step), ...]}}
        trajectories = {i: {} for i in range(num_envs)}
        
        for time_step in range(env.episode_len):
            print(f"\r步骤: {time_step} / {env.episode_len-1}", end="", flush=True)
            # 预测动作
            with torch.no_grad():
                action, _, _, _ = sim_agent(
                    next_obs[control_mask], deterministic=False
                )
            action_template = torch.zeros(
                (num_envs, max_agents), dtype=torch.int64, device=device
            )
            action_template[control_mask] = action.to(device)
            
            # 打印动作信息
            if ENABLE_ACTION_PRINT and (time_step % ACTION_PRINT_INTERVAL == 0 or time_step < 5):
                print(f"\n\n[步骤 {time_step}] 动作信息:")
                print("-" * 80)
                # 获取动作空间映射
                action_keys = env.action_keys_tensor  # [action_dim, 3] (accel, steer, head)
                
                # 按环境分组打印
                flat_idx = 0
                for env_idx in range(num_envs):
                    env_control_mask = control_mask[env_idx]
                    num_controlled_in_env = env_control_mask.sum().item()
                    
                    if num_controlled_in_env > 0:
                        env_actions = action[flat_idx:flat_idx + num_controlled_in_env]
                        env_action_values = action_keys[env_actions]  # [num_controlled, 3]
                        
                        print(f"  环境 {env_idx} (受控智能体: {num_controlled_in_env}):")
                        for i, agent_idx in enumerate(torch.where(env_control_mask)[0]):
                            action_idx = env_actions[i].item()
                            accel = env_action_values[i, 0].item()
                            steer = env_action_values[i, 1].item()
                            steer_deg = steer * 180 / math.pi
                            
                            # 计算实际前轮转角
                            front_wheel_angle = steering_to_front_wheel(steer)
                            front_wheel_deg = front_wheel_angle * 180 / math.pi
                            
                            # 判断动作类型（基于实际前轮转角）
                            if abs(front_wheel_deg) < 5:
                                action_type = "直行"
                            elif abs(front_wheel_deg) < 30:
                                action_type = "小角度转向"
                            elif abs(front_wheel_deg) < 60:
                                action_type = "中等转向"
                            else:
                                action_type = "大角度转向"
                            
                            print(f"    智能体 {agent_idx.item()}:")
                            print(f"      动作索引: {action_idx:3d}, 加速度: {accel:5.2f}")
                            print(f"      动作空间转向角: {steer:6.3f}弧度({steer_deg:6.1f}度)")
                            print(f"      实际前轮转角δ: {front_wheel_angle:6.3f}弧度({front_wheel_deg:6.1f}度) [{action_type}]")
                        
                        flat_idx += num_controlled_in_env
                
                # 统计信息
                all_action_values = action_keys[action]  # [num_controlled, 3]
                all_accels = all_action_values[:, 0].cpu().numpy()
                all_steers = all_action_values[:, 1].cpu().numpy()
                all_steers_deg = all_steers * 180 / math.pi
                
                # 计算所有智能体的实际前轮转角
                all_front_wheel_angles = np.array([steering_to_front_wheel(s) for s in all_steers])
                all_front_wheel_deg = all_front_wheel_angles * 180 / math.pi
                
                print(f"\n  统计信息 (所有{action.shape[0]}个受控智能体):")
                print(f"    加速度: 均值={all_accels.mean():.2f}, "
                      f"范围=[{all_accels.min():.2f}, {all_accels.max():.2f}]")
                
                print(f"\n    动作空间转向角:")
                print(f"      均值={all_steers_deg.mean():.1f}度, "
                      f"范围=[{all_steers_deg.min():.1f}, {all_steers_deg.max():.1f}]度, "
                      f"绝对值均值={np.abs(all_steers_deg).mean():.1f}度")
                
                print(f"\n    实际前轮转角δ:")
                print(f"      均值={all_front_wheel_deg.mean():.1f}度, "
                      f"范围=[{all_front_wheel_deg.min():.1f}, {all_front_wheel_deg.max():.1f}]度, "
                      f"绝对值均值={np.abs(all_front_wheel_deg).mean():.1f}度")
                
                # 统计直行/转向比例（基于实际前轮转角）
                straight_count = (np.abs(all_front_wheel_deg) < 5).sum()
                small_turn_count = ((np.abs(all_front_wheel_deg) >= 5) & (np.abs(all_front_wheel_deg) < 30)).sum()
                medium_turn_count = ((np.abs(all_front_wheel_deg) >= 30) & (np.abs(all_front_wheel_deg) < 60)).sum()
                large_turn_count = (np.abs(all_front_wheel_deg) >= 60).sum()
                
                print(f"\n    实际转向分布: 直行(<5度)={straight_count}, "
                      f"小角度(5-30度)={small_turn_count}, "
                      f"中等(30-60度)={medium_turn_count}, "
                      f"大角度(≥60度)={large_turn_count}")
                print("-" * 80)
            
            # 环境步进
            env.step_dynamics(action_template)
            
            # 预测未来轨迹（用于可视化，根据开关决定是否执行）
            predicted_trajectories = None
            if ENABLE_TRAJECTORY_PREDICTION:
                try:
                    # 传递已过滤的观察（与策略输入一致）
                    filtered_obs = next_obs[control_mask]
                    predicted_trajectories = predict_trajectory(
                        env, sim_agent, filtered_obs, control_mask, 
                        horizon=TRAJECTORY_HORIZON, device=device
                    )

                    # 平滑处理：对 x,y,yaw,speed 做一致性平滑（不影响可视化接口，仍然用前两维画线）
                    if ENABLE_TRAJECTORY_SMOOTHING and predicted_trajectories is not None:
                        from gpudrive.utils.trajectory_smoothing import (
                            smooth_predicted_trajectories_xy_yaw_speed,
                        )
                        predicted_trajectories = smooth_predicted_trajectories_xy_yaw_speed(
                            predicted_trajectories,
                            dt=0.1,
                            window=SMOOTH_WINDOW,
                            yaw_index=2,
                            speed_index=5,
                            yaw_blend_from_xy=0.7,
                            speed_eps=0.2,
                        )
                except Exception as e:
                    print(f"\n警告：轨迹预测失败: {e}")
                    predicted_trajectories = None
            
            # 存储当前时间步的前轮转角信息（用于可视化）
            action_keys = env.action_keys_tensor
            front_wheel_data = {}  # {(env_idx, agent_idx): (action_idx, steering, front_wheel_angle)}
            flat_idx = 0
            for env_idx in range(num_envs):
                env_control_mask = control_mask[env_idx]
                num_controlled_in_env = env_control_mask.sum().item()
                
                if num_controlled_in_env > 0:
                    env_actions = action[flat_idx:flat_idx + num_controlled_in_env]
                    env_action_values = action_keys[env_actions]
                    
                    for i, agent_idx in enumerate(torch.where(env_control_mask)[0]):
                        steering = env_action_values[i, 1].item()
                        front_wheel_angle = steering_to_front_wheel(steering)
                        front_wheel_data[(env_idx, agent_idx.item())] = (
                            env_actions[i].item(), steering, front_wheel_angle
                        )
                    
                    flat_idx += num_controlled_in_env
            
            # 渲染（根据开关决定是否包含预测轨迹）
            sim_states = env.vis.plot_simulator_state(
                env_indices=list(range(num_envs)),
                time_steps=[time_step]*num_envs,
                zoom_radius=70,
                predicted_trajectories=predicted_trajectories if ENABLE_TRAJECTORY_PREDICTION else None,
            )
            
            # 在每个环境的图像上添加前轮转角可视化和轨迹
            for i in range(num_envs):
                add_front_wheel_visualization(
                    sim_states[i], env, i, control_mask, front_wheel_data
                )
                
                # 在当前帧上叠加轨迹（只显示到当前时间步的轨迹）
                add_trajectory_to_frame(sim_states[i], env, i, control_mask, trajectories, time_step)
                
                frames[f"env_{i}"].append(img_from_fig(sim_states[i]))
            
            # 获取新的观察和奖励
            next_obs = env.get_obs()
            reward = env.get_rewards()
            done = env.get_dones()
            info = env.get_infos()
            
            # 记录智能体轨迹（只记录未完成的智能体）
            agent_states = GlobalEgoState.from_tensor(
                env.sim.absolute_self_observation_tensor(),
                backend="torch",
                device=device,
            )
            for env_idx in range(num_envs):
                env_control_mask = control_mask[env_idx]
                for agent_idx in torch.where(env_control_mask)[0]:
                    agent_idx_item = agent_idx.item()
                    
                    # 检查智能体是否已完成（done == 1）或已死亡
                    is_done = done[env_idx, agent_idx].item() > 0
                    if is_done:
                        # 如果智能体已完成，停止记录轨迹
                        continue
                    
                    pos_x = agent_states.pos_x[env_idx, agent_idx].item()
                    pos_y = agent_states.pos_y[env_idx, agent_idx].item()
                    
                    # 过滤异常位置值（kPaddingPosition通常是很大的值，如10000）
                    # 如果位置突然变化很大，可能是被重置到了padding位置
                    if agent_idx_item in trajectories[env_idx] and len(trajectories[env_idx][agent_idx_item]) > 0:
                        last_x, last_y, _ = trajectories[env_idx][agent_idx_item][-1]
                        # 如果位置变化超过1000米，可能是异常值，跳过
                        if abs(pos_x - last_x) > 1000 or abs(pos_y - last_y) > 1000:
                            continue
                    
                    # 过滤明显异常的位置值（绝对值过大）
                    if abs(pos_x) > 10000 or abs(pos_y) > 10000:
                        continue
                    
                    if agent_idx_item not in trajectories[env_idx]:
                        trajectories[env_idx][agent_idx_item] = []
                    trajectories[env_idx][agent_idx_item].append((pos_x, pos_y, time_step))
            
            # 累积统计信息
            total_rewards += reward
            collision_count += (info.collided.int())
            off_road_count += (info.off_road.int())
            goal_achieved = torch.maximum(goal_achieved, info.goal_achieved.int())  # 使用maximum而不是|=
            
            # 更新存活状态（未done的智能体）
            agent_alive &= (~done.bool())
            
            if done.all():
                print(f"\n仿真在第 {time_step} 步结束")
                break
        
        
        print("\n\n" + "="*80)
        print("仿真运行完成 - 统计结果")
        print("="*80)
        
        print(f"\n⚠️  重要提示：当前使用的奖励类型为 'weighted_combination'")
        print(f"   - 达成目标后的每个时间步都会获得 +1.0 奖励")
        print(f"   - 越界每次 -0.75，碰撞每次 -0.75")
        print(f"   - 到达越早，停留时间越长，累积奖励越高")
        
        # 计算整体统计
        controlled_agents = control_mask.sum().item()
        
        # 只统计受控智能体的数据
        controlled_total_rewards = total_rewards[control_mask]
        controlled_collision_count = collision_count[control_mask]
        controlled_off_road_count = off_road_count[control_mask]
        controlled_goal_achieved = goal_achieved[control_mask]
        controlled_agent_alive = agent_alive[control_mask]
        
        # 存活智能体统计（未碰撞的智能体）
        alive_mask = controlled_agent_alive
        num_alive = alive_mask.sum().item()
        num_dead = controlled_agents - num_alive
        
        # 任务完成统计
        num_goal_achieved = (controlled_goal_achieved > 0).sum().item()  # 转换为bool再统计
        goal_rate = (num_goal_achieved / controlled_agents * 100) if controlled_agents > 0 else 0
        
        # 碰撞和越界统计
        total_collisions = controlled_collision_count.sum().item()
        total_off_road = controlled_off_road_count.sum().item()
        
        # 奖励统计
        mean_reward = controlled_total_rewards.mean().item()
        std_reward = controlled_total_rewards.std().item()
        min_reward = controlled_total_rewards.min().item()
        max_reward = controlled_total_rewards.max().item()
        
        # 存活智能体的奖励统计
        if num_alive > 0:
            alive_rewards = controlled_total_rewards[alive_mask]
            alive_mean_reward = alive_rewards.mean().item()
            alive_std_reward = alive_rewards.std().item()
            alive_min_reward = alive_rewards.min().item()
            alive_max_reward = alive_rewards.max().item()
            
            # 存活智能体中完成目标的数量
            alive_goal_achieved = (controlled_goal_achieved[alive_mask] > 0).sum().item()
            alive_goal_rate = (alive_goal_achieved / num_alive * 100) if num_alive > 0 else 0
        else:
            alive_mean_reward = 0.0
            alive_std_reward = 0.0
            alive_min_reward = 0.0
            alive_max_reward = 0.0
            alive_goal_achieved = 0
            alive_goal_rate = 0.0
        
        print(f"\n📊 总体统计 (所有{controlled_agents}个受控智能体)")
        print("-" * 80)
        print(f"  总奖励均值: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  奖励范围:   [{min_reward:.2f}, {max_reward:.2f}]")
        print(f"  达成目标:   {num_goal_achieved}/{controlled_agents} ({goal_rate:.1f}%)")
        print(f"  总碰撞次数: {total_collisions}")
        print(f"  总越界次数: {total_off_road}")
        
        # 奖励分解
        collision_penalty_total = total_collisions * (-0.75)
        off_road_penalty_total = total_off_road * (-0.75)
        implied_goal_reward = controlled_total_rewards.sum().item() - collision_penalty_total - off_road_penalty_total
        
        print(f"\n  📈 奖励分解 (所有受控智能体累计):")
        print(f"     碰撞惩罚:  {collision_penalty_total:.2f} ({total_collisions}次 × -0.75)")
        print(f"     越界惩罚:  {off_road_penalty_total:.2f} ({total_off_road}次 × -0.75)")
        print(f"     目标奖励:  {implied_goal_reward:.2f} (达成后累积)")
        print(f"     总奖励:    {controlled_total_rewards.sum().item():.2f}")
        
        # 估算平均停留时长（假设到达后每步+1.0）
        if num_goal_achieved > 0:
            avg_dwelling_steps = implied_goal_reward / num_goal_achieved
            print(f"     → 达成目标的智能体平均停留: {avg_dwelling_steps:.1f} 步")
        
        print(f"\n✅ 存活智能体统计 (存活{num_alive}个, 死亡{num_dead}个)")
        print("-" * 80)
        print(f"  存活率:     {num_alive}/{controlled_agents} ({num_alive/controlled_agents*100:.1f}%)")
        print(f"  存活奖励均值: {alive_mean_reward:.2f} ± {alive_std_reward:.2f}")
        print(f"  存活奖励范围: [{alive_min_reward:.2f}, {alive_max_reward:.2f}]")
        print(f"  存活智能体目标达成: {alive_goal_achieved}/{num_alive} ({alive_goal_rate:.1f}%)")
        
        # 分环境统计
        print(f"\n🌍 分环境详细统计")
        print("-" * 80)
        for env_idx in range(num_envs):
            env_control_mask = control_mask[env_idx]
            num_controlled_in_env = env_control_mask.sum().item()
            
            if num_controlled_in_env > 0:
                env_rewards = total_rewards[env_idx][env_control_mask]
                env_collisions = collision_count[env_idx][env_control_mask]
                env_off_road = off_road_count[env_idx][env_control_mask]
                env_goals = goal_achieved[env_idx][env_control_mask]
                env_alive = agent_alive[env_idx][env_control_mask]
                
                env_alive_count = env_alive.sum().item()
                env_goal_count = (env_goals > 0).sum().item()  # 转换为bool再统计
                env_mean_reward = env_rewards.mean().item()
                
                print(f"  环境 {env_idx}:")
                print(f"    受控智能体: {num_controlled_in_env}")
                print(f"    存活: {env_alive_count} ({env_alive_count/num_controlled_in_env*100:.1f}%)")
                print(f"    达成目标: {env_goal_count} ({env_goal_count/num_controlled_in_env*100:.1f}%)")
                print(f"    平均奖励: {env_mean_reward:.2f}")
                print(f"    碰撞: {env_collisions.sum().item()}, 越界: {env_off_road.sum().item()}")
        
        # 综合评分（存活智能体）
        print(f"\n🎯 综合评分 (仅存活智能体)")
        print("-" * 80)
        
        # 评分公式：平均奖励 + 目标达成率加权
        if num_alive > 0:
            survival_score = (num_alive / controlled_agents) * 100  # 存活率得分
            completion_score = alive_goal_rate  # 目标完成率得分
            reward_score = (alive_mean_reward + 100) / 2  # 奖励归一化到0-100
            
            # 综合得分 = 存活率30% + 目标完成率50% + 奖励20%
            overall_score = survival_score * 0.3 + completion_score * 0.5 + reward_score * 0.2
            
            print(f"  存活率得分:   {survival_score:.1f}/100 (权重30%)")
            print(f"  目标完成得分: {completion_score:.1f}/100 (权重50%)")
            print(f"  奖励得分:     {reward_score:.1f}/100 (权重20%)")
            print(f"  " + "━"*76)
            print(f"  综合得分:     {overall_score:.1f}/100")
            
            # 评级
            if overall_score >= 90:
                grade = "S (优秀)"
            elif overall_score >= 80:
                grade = "A (良好)"
            elif overall_score >= 70:
                grade = "B (中等)"
            elif overall_score >= 60:
                grade = "C (及格)"
            else:
                grade = "D (不及格)"
            
            print(f"  评级:         {grade}")
        else:
            print(f"  ⚠️  所有智能体都已死亡，无法计算存活智能体得分")
            overall_score = 0.0
        
        print("="*80 + "\n")
        
        # 创建输出目录（用于保存GIF）
        model_name = Path(model_path).stem
        output_dir = Path(f"output/{model_name}_gif")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 8. 保存结果
        print("\n8. 保存可视化结果...")
        try:
            from PIL import Image
            import numpy as np

            for i in range(num_envs):
                # 处理每个环境的帧
                images = []
                for frame in frames[f"env_{i}"]:
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        img = Image.fromarray(frame.astype(np.uint8))
                    else:
                        img = Image.fromarray(frame.astype(np.uint8)).convert('RGB')
                    images.append(img)

                # 保存 GIF
                output_file = output_dir / f"simulation_env_{i}.gif"
                images[0].save(
                    str(output_file),
                    save_all=True,
                    append_images=images[1:],
                    duration=67,  # 约15fps
                    loop=0
                )
                print(f"环境{i}的GIF已保存到: {output_file}")


        except Exception as e:
            print(f"保存结果失败: {e}")
            import traceback
            traceback.print_exc()
        
        env.close()
        print("\n=== 脚本执行完成 ===")
        
    except Exception as e:
        print(f"\n仿真运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()