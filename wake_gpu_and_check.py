#!/usr/bin/env python3
"""尝试唤醒GPU并检查训练进程状态"""
import torch
import sys

print("=== 尝试唤醒 GPU ===")
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f"GPU {device}: {torch.cuda.get_device_name(device)}")
    
    # 尝试创建一个简单的tensor来唤醒GPU
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print("✅ GPU 已被唤醒，可以正常工作")
    except Exception as e:
        print(f"❌ GPU 唤醒失败: {e}")
        sys.exit(1)
else:
    print("❌ CUDA 不可用")
    sys.exit(1)

# 检查GPU状态
print(f"\nGPU 内存使用: {torch.cuda.memory_allocated()/1024**2:.1f} MB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
print(f"GPU 利用率: {torch.cuda.utilization()}%")
