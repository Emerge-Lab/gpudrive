#!/usr/bin/env python3

from gpudrive.networks.late_fusion import NeuralNet
import torch

print("正在加载预训练模型...")
# 加载预训练模型
agent = NeuralNet.from_pretrained("daphne-cornelisse/policy_S10_000_02_27")

print("模型加载成功!")
print(f"模型参数数量: {sum(p.numel() for p in agent.parameters())}")

# 从现有环境中获取动作维度（查看配置文件中的设置）
# 根据 config.py 中的设置：action_space_steer_disc=13, action_space_accel_disc=7
# 总动作数 = steer_disc * accel_disc = 13 * 7 = 91
action_dim = 13 * 7  # 91

print(f"动作空间维度: {action_dim}")

# 保存完整的检查点信息（按照 make_agent 函数期望的格式）
checkpoint = {
    "model_arch": {
        "input_dim": 64,    # 从配置文件中看到的默认值
        "hidden_dim": 128,  # 从配置文件中看到的默认值
    },
    "action_dim": action_dim,
    "parameters": agent.state_dict()  # 注意这里是 "parameters" 不是直接展开
}

print("正在保存完整检查点...")
torch.save(checkpoint, "pretrained_model_complete.pt")
print("保存完成: pretrained_model_complete.pt")

# 验证保存的文件
print("\n验证保存的检查点:")
loaded = torch.load("pretrained_model_complete.pt")
print(f"包含的键: {list(loaded.keys())}")
print(f"model_arch: {loaded['model_arch']}")
print(f"action_dim: {loaded['action_dim']}")
print(f"parameters 键数量: {len(loaded['parameters'])}")
