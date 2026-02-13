#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 Hugging Face 预训练模型演示脚本
直接对应 04_use_pretrained_sim_agent.ipynb 的内容
"""

# 抑制TensorFlow警告消息
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import dataclasses
from pathlib import Path
import sys

# 设置项目路径
script_dir = Path(__file__).resolve().parent
gpudrive_root = script_dir.parent.parent  # 回到gpudrive根目录
os.chdir(gpudrive_root)
sys.path.insert(0, str(gpudrive_root))

# GPUDrive导入
from huggingface_hub import PyTorchModelHubMixin, ModelCard
from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config

def main():
    print("=== Hugging Face 预训练模型演示 ===\n")
    
    # 1. 加载配置（对应Cell 2）
    print("1. 加载配置...")
    config = load_config("examples/experimental/config/reliable_agents_params")
    print("配置内容:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    max_agents = config.max_controlled_agents
    num_envs = 10 # 与notebook保持一致
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 可以改为cpu测试
    
    print(f"\n参数设置:")
    print(f"  最大智能体数: {max_agents}")
    print(f"  环境数量: {num_envs}")
    print(f"  使用设备: {device}")
    
    # 2. 加载或下载并转换Hugging Face模型为.pt格式（对应Cell 4）
    print("\n2. 加载或下载并转换Hugging Face模型...")
    model_path = "111"
    
    try:
        # 检查是否已存在.pt文件
        if Path(model_path).exists():
            print(f"发现已存在的.pt文件: {model_path}")
            
            # 加载保存的权重
            print("加载保存的模型权重...")
            checkpoint = torch.load(model_path, map_location=device)
            
            # 使用检查点中的架构参数创建模型
            model_arch = checkpoint['model_arch']
            print(f"使用检查点架构参数:")
            print(f"  input_dim: {model_arch.get('input_dim', 64)}")
            print(f"  hidden_dim: {model_arch.get('hidden_dim', 128)}")
            print(f"  action_dim: {checkpoint.get('action_dim', 91)}")
            print(f"  dropout: 0.0 (关闭dropout)")
            
            # 使用检查点的架构参数创建模型，但设置dropout为0
            sim_agent = NeuralNet(
                action_dim=checkpoint.get('action_dim', 91),
                input_dim=model_arch.get('input_dim', 64),
                hidden_dim=model_arch.get('hidden_dim', 128),
                max_controlled_agents=config.max_controlled_agents,
                obs_dim=2984,  # 使用默认的obs_dim
                fusion_type="simple",  # 新增：融合类型选择
                config=config  # 传递配置
            )
            sim_agent = sim_agent.to(device)
            
            # 检查是否是训练检查点格式
            if 'parameters' in checkpoint:
                print("检测到训练检查点格式，提取模型权重...")
                model_state_dict = checkpoint['parameters']
                print(f"找到 {len(model_state_dict)} 个模型参数")
            else:
                # 标准模型权重格式
                model_state_dict = checkpoint
            
            sim_agent.load_state_dict(model_state_dict)
            sim_agent.eval()
            print("✅ 从.pt文件加载模型成功")
            
        else:
            print("未找到.pt文件，从Hugging Face Hub下载模型...")
            
            # 从Hugging Face下载模型
            sim_agent = NeuralNet.from_pretrained("daphne-cornelisse/policy_S10_000_02_27")
            sim_agent = sim_agent.to(device)
            print("✅ 模型下载成功")
            
            # 将模型转换为.pt格式并保存
            print(f"保存模型为: {model_path}")
            torch.save(sim_agent.state_dict(), model_path)
            print("✅ 模型已保存为.pt格式")
        
        # 显示模型信息（对应Cell 5, 6）
        print(f"动作维度: {sim_agent.action_dim}")
        print(f"观察维度: {sim_agent.obs_dim}")
        print(f"使用模型文件: {model_path}")
        
        # 加载模型卡片信息（对应Cell 7）
        try:
            card = ModelCard.load("daphne-cornelisse/policy_S10_000_02_27")
            print(f"模型标签: {card.data.tags}")
        except:
            print("无法获取模型卡片信息")
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请检查网络连接或确保已安装 huggingface_hub")
        return
    
    # 3. 创建环境（对应Cell 11）
    print("\n3. 创建环境...")
    try:
        # 创建数据加载器
        data_path = "data/processed/validation"
        if not Path(data_path).exists():
            data_path = "data/processed/validation"  # 备用路径
            
        train_loader = SceneDataLoader(
            root=data_path,
            batch_size=num_envs,
            dataset_size=100,
            sample_with_replacement=False,
        )
        
        # 设置环境配置
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
        
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return
    
    # 4. 测试模型（对应Cell 14-16）
    print("\n4. 测试模型...")
    try:
        next_obs = env.reset()
        control_mask = env.cont_agent_mask
        
        print(f"观察形状: {next_obs.shape}")
        print(f"控制掩码形状: {control_mask.shape}")
        
        # 测试模型预测
        with torch.no_grad():
            action, logprob, entropy, value = sim_agent(
                next_obs[control_mask], deterministic=False
            )
        
        print(f"动作形状: {action.shape}")
        print(f"对数概率形状: {logprob.shape}")
        print(f"熵形状: {entropy.shape}")
        print(f"价值形状: {value.shape}")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        env.close()
        return
    
    # 5. 运行仿真（对应Cell 18）
    print("\n5. 运行仿真rollout...")
    try:
        next_obs = env.reset()
        control_mask = env.cont_agent_mask
        
        print(f"开始观察形状: {next_obs.shape}")
        
        frames = {f"env_{i}": [] for i in range(num_envs)}
        
        # 限制步数以避免过长运行
        max_steps = min(91, env.episode_len)
        
        for time_step in range(max_steps):
            print(f"\r步骤: {time_step}/{max_steps-1}", end="", flush=True)
            
            # 预测动作
            with torch.no_grad():
                action, _, _, _ = sim_agent(
                    next_obs[control_mask], deterministic=True
                )
            
            action_template = torch.zeros(
                (num_envs, max_agents), dtype=torch.int64, device=device
            )
            action_template[control_mask] = action.to(device)
            
            # 环境步进
            env.step_dynamics(action_template)

            
            sim_states = env.vis.plot_simulator_state(
                env_indices=list(range(num_envs)),
                time_steps=[time_step] * num_envs,
                zoom_radius=70,
            )
            
            for i in range(num_envs):
                frames[f"env_{i}"].append(img_from_fig(sim_states[i]))
            
            # 获取新状态
            next_obs = env.get_obs()
            reward = env.get_rewards()
            done = env.get_dones()
            info = env.get_infos()
            
            if done.all():
                print(f"\n提前结束于步骤 {time_step}")
                break
        
        print(f"\n✅ 仿真完成")
        
        # 6. 保存结果
        print("\n6. 保存结果...")
        try:
            from PIL import Image
            import numpy as np
            
            output_dir = Path("output/huggingface_demo")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for env_name, env_frames in frames.items():
                if env_frames:
                    images = []
                    for frame in env_frames:
                        if isinstance(frame, np.ndarray):
                            img = Image.fromarray(frame.astype(np.uint8))
                            images.append(img)
                    
                    if images:
                        output_file = output_dir / f"{env_name}_demo.gif"
                        images[0].save(
                            str(output_file),
                            save_all=True,
                            append_images=images[1:],
                            duration=200,
                            loop=0
                        )
                        print(f"✅ 保存 {env_name}: {output_file}")
            
            print(f"所有结果保存至: {output_dir}")
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")
        
    except Exception as e:
        print(f"❌ 仿真失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()

