# GPUDrive Hugging Face 预训练模型脚本说明

基于 `examples/tutorials/04_use_pretrained_sim_agent.ipynb` 创建的Python脚本版本。

## 📁 文件说明

### 1. `use_huggingface_pretrained_model.py`
**功能完整的生产版本**
- ✅ 完整的错误处理和异常捕获
- ✅ 模块化函数设计，便于维护
- ✅ 支持自定义参数（环境数量、设备选择等）
- ✅ 详细的日志输出和状态反馈
- ✅ 优雅的资源清理
- ✅ 支持不同的数据路径fallback

### 2. `simple_huggingface_demo.py`
**简化的演示版本**
- ✅ 直接对应notebook的Cell结构
- ✅ 更简洁的代码结构
- ✅ 适合学习和理解工作流程
- ✅ 保留了notebook的核心逻辑

## 🚀 运行方法

### 预备条件
```bash
# 安装必要的依赖
pip install huggingface_hub
pip install pillow

# 确保数据目录存在
ls data/processed/examples  # 或者 data/processed/validation
```

### 运行脚本
```bash
# 运行完整版
python examples/test/use_huggingface_pretrained_model.py

# 运行简化版
python examples/test/simple_huggingface_demo.py
```

## 📋 脚本对应的Notebook内容

| Notebook Cell | 简化版函数/部分 | 完整版函数 |
|---------------|----------------|------------|
| Cell 0: 导入库 | 脚本顶部导入 | 脚本顶部导入 |
| Cell 2: 加载配置 | main() 第1部分 | load_config 相关 |
| Cell 4: 加载HF模型 | main() 第2部分 | load_huggingface_model() |
| Cell 5-6: 模型信息 | main() 第2部分 | load_huggingface_model() |
| Cell 11: 创建环境 | main() 第3部分 | create_environment() |
| Cell 14-16: 测试模型 | main() 第4部分 | run_simulation() 部分 |
| Cell 18: Rollout | main() 第5部分 | run_simulation() |
| Cell 19: 保存视频 | main() 第6部分 | save_videos() |

## ⚙️ 主要功能

### 1. 从Hugging Face Hub加载模型
- 模型: `daphne-cornelisse/policy_S10_000_02_27`
- 自动下载和缓存
- 支持GPU/CPU设备选择

### 2. 环境配置
- 使用 `reliable_agents_params` 配置
- 支持多环境并行仿真
- 自动查找数据目录

### 3. 仿真运行
- 实时步骤显示
- 智能帧收集（每5步渲染一次）
- 支持早停条件

### 4. 结果保存
- 生成GIF动画
- 自动创建输出目录
- 支持多环境分别保存

## 🔧 自定义参数

在脚本中可以修改的参数：

```python
# 仿真参数
num_envs = 2          # 环境数量
device = "cuda"       # 设备选择
max_steps = 50        # 最大仿真步数

# 渲染参数
zoom_radius = 70      # 渲染缩放半径
duration = 200        # GIF帧间隔(ms)

# 模型参数
model_name = "daphne-cornelisse/policy_S10_000_02_27"
```

## ❗ 注意事项

1. **网络连接**: 首次运行需要网络连接下载模型
2. **GPU内存**: 多环境仿真需要足够的GPU内存
3. **数据路径**: 确保 `data/processed/examples` 或 `data/processed/validation` 存在
4. **依赖包**: 需要安装 `huggingface_hub` 和 `pillow`

## 🐛 故障排除

### 常见问题
1. **ModuleNotFoundError: huggingface_hub**
   ```bash
   pip install huggingface_hub
   ```

2. **数据目录不存在**
   - 检查 `data/processed/` 下是否有 `examples` 或 `validation` 目录

3. **CUDA内存不足**
   - 减少 `num_envs` 参数
   - 或设置 `device = "cpu"`

4. **网络连接问题**
   - 检查网络连接
   - 或使用代理设置环境变量

## 📊 输出结果

运行成功后，会在 `output/huggingface_demo/` 或 `output/huggingface_pretrained_simulation/` 目录下生成：
- `env_0_demo.gif` - 环境0的仿真动画
- `env_1_demo.gif` - 环境1的仿真动画
- 等等...

每个GIF展示了智能体在仿真环境中的驾驶行为。










