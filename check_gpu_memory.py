#!/usr/bin/env python3
"""
GPU 显存监控和清理脚本
在训练过程中定期检查显存占用
"""

import torch
import gc

def print_gpu_memory():
    """打印当前GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"\n{'='*60}")
        print(f"GPU Memory Status:")
        print(f"  Allocated:     {allocated:.3f} GB")
        print(f"  Reserved:      {reserved:.3f} GB")
        print(f"  Max Allocated: {max_allocated:.3f} GB")
        print(f"  Free (approx): {reserved - allocated:.3f} GB (in reserved)")
        print(f"{'='*60}\n")
        
        return allocated, reserved
    else:
        print("CUDA not available")
        return 0, 0

def clean_gpu_memory():
    """清理GPU显存"""
    print("Cleaning GPU memory...")
    
    # 1. Python垃圾回收
    gc.collect()
    
    # 2. 清空PyTorch缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("GPU memory cleaned!")

if __name__ == "__main__":
    print_gpu_memory()
    
    response = input("Do you want to clean GPU memory? (y/n): ")
    if response.lower() == 'y':
        clean_gpu_memory()
        print("\nAfter cleaning:")
        print_gpu_memory()

