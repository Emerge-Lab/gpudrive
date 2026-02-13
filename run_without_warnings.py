#!/usr/bin/env python3
"""
运行脚本并隐藏CUDA警告的包装器
"""
import sys
import os
from contextlib import redirect_stderr
import io

# 重定向stderr来隐藏所有警告
with redirect_stderr(io.StringIO()):
    # 导入并运行主脚本
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from examples.test.use_training_agen_pufferlib import main
    
    if __name__ == "__main__":
        main()








