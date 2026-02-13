
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPUDrive 预训练模型使用脚本
基于 04_use_pretrained_sim_agent.ipynb 复现
独立运行，不依赖 Docker
"""

import torch
import dataclasses
import mediapy
import os
import sys
from pathlib import Path
import wandb
import yaml
from box import Box
from typing import Callable
from datetime import datetime
import dataclasses
from gpudrive.integrations.sb3.ppo import IPPO
from stable_baselines3 import PPO
from gpudrive.integrations.sb3.callbacks import MultiAgentCallback
from gpudrive.env.config import EnvConfig
from gpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv

from gpudrive.networks.perm_eq_late_fusion import (
    LateFusionNet,
    LateFusionPolicy,
)
from gpudrive.networks.basic_ffn import FFN, FeedForwardPolicy


path = "/home/wbk/gpudrive/policies/policy_0818_100017536"
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
    
    # 设置环境
    project_root = setup_environment()
    
    try:
        # 导入必要的模块
        from huggingface_hub import PyTorchModelHubMixin, ModelCard
        from gpudrive.networks.late_fusion import NeuralNet
        from gpudrive.env.config import EnvConfig
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
        config_path = project_root / "examples/experimental/config/reliable_agents_params_sb3"
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
            norm_obs=False,
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
        # 3. 加载预训练模型
    print("\n2. 加载预训练模型...")
    try:
        sim_agent = IPPO.load(path, env=env,
        custom_objects={
          "observation_space": env.observation_space,
          "action_space": env.action_space,
      })
        sim_agent.policy.to(device)
        print("预训练模型加载成功")

    except Exception as e:
        print(
            f"模型加载失败: {e}")
        return
        
    # 7. 运行仿真
    print("\n6. 开始仿真运行...")
    try:
        next_obs = env.reset()
        control_mask = env.cont_agent_mask
        
        print(f"观察形状: {next_obs.shape}")
        print(f"控制掩码形状: {control_mask.shape}")
        
        frames = {f"env_{i}": [] for i in range(num_envs)}
        
        for time_step in range(env.episode_len):
            print(f"\r步骤: {time_step}", end="", flush=True)
            obs_for_predict = next_obs[control_mask].cpu().numpy()
            # 预测动作
            action, _ = sim_agent.predict(
                obs_for_predict, deterministic=False
            )
            action_tensor = torch.from_numpy(action).to(device)
            action_template = torch.zeros(
                (num_envs, max_agents), dtype=torch.int64, device=device
            )
            action_template[control_mask] = action_tensor
            
            # 环境步进
            env.step_dynamics(action_template)
            
            # 渲染
            sim_states = env.vis.plot_simulator_state(
                env_indices=list(range(num_envs)),
                time_steps=[time_step]*num_envs,
                zoom_radius=70,
            )
            
            for i in range(num_envs):
                frames[f"env_{i}"].append(img_from_fig(sim_states[i]))
            
            # 获取新的观察和奖励
            next_obs = env.get_obs()
            reward = env.get_rewards()
            done = env.get_dones()
            info = env.get_infos()
            
            # print(reward);
            
            if done.all():
                print(f"\n仿真在第 {time_step} 步结束")
                break
        
        
        print("\n仿真运行完成")
        
        # 8. 保存结果
        print("\n7. 保存可视化结果...")
        try:
            output_dir = Path(path + "_gif")
            output_dir.mkdir(parents=True, exist_ok=True)

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