#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPUDrive Hugging Face 预训练模型使用脚本
基于 04_use_pretrained_sim_agent.ipynb 创建
使用 Hugging Face Hub 上的预训练模型
"""

# 抑制TensorFlow警告消息（可选）
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import dataclasses
import sys
from pathlib import Path
import numpy as np
from typing import Optional

# GPUDrive相关导入
from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config

# Hugging Face相关导入
try:
    from huggingface_hub import PyTorchModelHubMixin, ModelCard
    HF_AVAILABLE = True
except ImportError:
    print("警告: 未安装 huggingface_hub，将无法从 Hugging Face Hub 加载模型")
    HF_AVAILABLE = False


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


def load_huggingface_model(model_name: str = "daphne-cornelisse/policy_S10_000_02_27", device: str = "cuda"):
    """从 Hugging Face Hub 加载预训练模型"""
    if not HF_AVAILABLE:
        raise ImportError("请安装 huggingface_hub: pip install huggingface_hub")
    
    print(f"\n从 Hugging Face Hub 加载模型: {model_name}")
    try:
        # 从 Hugging Face Hub 加载预训练模型
        sim_agent = NeuralNet.from_pretrained(model_name)
        sim_agent = sim_agent.to(device)
        sim_agent.eval()
        
        print(f"✅ 模型加载成功")
        print(f"动作维度: {sim_agent.action_dim}")
        print(f"观察维度: {sim_agent.obs_dim}")
        
        # 获取模型信息
        try:
            card = ModelCard.load(model_name)
            print(f"模型标签: {card.data.tags}")
        except Exception as e:
            print(f"无法加载模型卡片: {e}")
        
        return sim_agent
    
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请检查网络连接或模型名称是否正确")
        return None


def create_environment(config, num_envs: int = 4, device: str = "cuda"):
    """创建GPUDrive环境"""
    print("\n创建仿真环境...")
    
    try:
        # 获取项目根目录
        project_root = Path.cwd()
        
        # 创建数据加载器
        data_path = project_root / "data/processed/examples"
        if not data_path.exists():
            # 如果examples不存在，尝试使用validation
            data_path = project_root / "data/processed/validation"
            if not data_path.exists():
                raise FileNotFoundError(f"数据目录不存在: {data_path}")
        
        train_loader = SceneDataLoader(
            root=str(data_path),
            batch_size=num_envs,
            dataset_size=100,
            sample_with_replacement=False,
        )
        
        print(f"数据路径: {data_path}")
        
        # 设置环境参数
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
        
        # 创建环境
        env = GPUDriveTorchEnv(
            config=env_config,
            data_loader=train_loader,
            max_cont_agents=config.max_controlled_agents,
            device=device,
        )
        
        print("✅ 环境创建成功")
        return env
    
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return None


def run_simulation(env, sim_agent, num_envs: int, max_agents: int, device: str = "cuda", 
                  max_steps: Optional[int] = None):
    """运行仿真并收集帧数据"""
    print("\n开始仿真运行...")
    
    try:
        # 重置环境
        next_obs = env.reset()
        control_mask = env.cont_agent_mask
        
        print(f"观察形状: {next_obs.shape}")
        print(f"控制掩码形状: {control_mask.shape}")
        
        # 准备帧收集
        frames = {f"env_{i}": [] for i in range(num_envs)}
        
        # 设置最大步数
        episode_len = min(env.episode_len, max_steps) if max_steps else env.episode_len
        
        for time_step in range(episode_len):
            print(f"\r步骤: {time_step}/{episode_len-1}", end="", flush=True)
            
            # 预测动作
            with torch.no_grad():
                action, logprob, entropy, value = sim_agent(
                    next_obs[control_mask], deterministic=False
                )
            
            # 创建动作模板
            action_template = torch.zeros(
                (num_envs, max_agents), dtype=torch.int64, device=device
            )
            action_template[control_mask] = action.to(device)
            
            # 环境步进
            env.step_dynamics(action_template)
            
            # 渲染状态
            if time_step % 2 == 0:  # 每5步渲染一次以节省内存E
                sim_states = env.vis.plot_simulator_state(
                    env_indices=list(range(num_envs)),
                    time_steps=[time_step] * num_envs,
                    zoom_radius=70,
                )
                
                for i in range(num_envs):
                    frames[f"env_{i}"].append(img_from_fig(sim_states[i]))
            
            # 获取环境反馈
            next_obs = env.get_obs()
            reward = env.get_rewards()
            done = env.get_dones()
            info = env.get_infos()
            
            # 检查是否提前结束
            if done.all():
                print(f"\n仿真在第 {time_step} 步提前结束")
                break
        
        print(f"\n仿真运行完成，总共 {time_step+1} 步")
        return frames
    
    except Exception as e:
        print(f"\n❌ 仿真运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_videos(frames, output_dir: str = "output/huggingface_pretrained_simulation"):
    """保存仿真视频"""
    print(f"\n保存仿真结果到: {output_dir}")
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        from PIL import Image
        
        for env_name, env_frames in frames.items():
            if not env_frames:
                print(f"警告: {env_name} 没有帧数据")
                continue
                
            # 处理帧数据
            images = []
            for frame in env_frames:
                if isinstance(frame, np.ndarray):
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        img = Image.fromarray(frame.astype(np.uint8))
                    else:
                        img = Image.fromarray(frame.astype(np.uint8)).convert('RGB')
                    images.append(img)
            
            if images:
                # 保存 GIF
                output_file = output_path / f"simulation_{env_name}.gif"
                images[0].save(
                    str(output_file),
                    save_all=True,
                    append_images=images[1:],
                    duration=200,  # 约5fps
                    loop=0
                )
                print(f"✅ {env_name} GIF已保存: {output_file}")
        
        print(f"✅ 所有仿真结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"❌ 保存视频失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("=== GPUDrive Hugging Face 预训练模型使用脚本 ===")
    
    # 设置环境
    try:
        project_root = setup_environment()
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        return
    
    # 加载配置
    print("\n1. 加载配置...")
    try:
        config_path = project_root / "examples/experimental/config/reliable_agents_params"
        config = load_config(str(config_path))
        print("✅ 配置加载成功")
        print(f"最大控制智能体数: {config.max_controlled_agents}")
        print(f"观察半径: {config.obs_radius}")
        print(f"奖励类型: {config.reward_type}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return
    
    # 设置参数
    max_agents = config.max_controlled_agents
    num_envs = 4  # 可调整环境数量
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n使用设备: {device}")
    print(f"环境数量: {num_envs}")
    
    # 加载模型
    print("\n2. 加载预训练模型...")
    sim_agent = load_huggingface_model(device=device)
    if sim_agent is None:
        return
    
    # 创建环境
    print("\n3. 创建仿真环境...")
    env = create_environment(config, num_envs, device)
    if env is None:
        return
    
    # 运行仿真
    print("\n4. 运行仿真...")
    frames = run_simulation(env, sim_agent, num_envs, max_agents, device, max_steps=91)
    
    if frames is None:
        return
    
    # 保存结果
    print("\n5. 保存结果...")
    save_videos(frames)
    
    # 清理资源
    env.close()
    print("\n=== 脚本执行完成 ===")


if __name__ == "__main__":
    main()

