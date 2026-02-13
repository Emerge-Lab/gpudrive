# GPUDrive 项目文档

## 1. 项目简介

GPUDrive 是一个极快的数据驱动自动驾驶仿真平台，核心由 C++ 编写，并通过 Python 提供接口。它支持百万帧级别的仿真速度，适用于大规模智能体开发与评测。项目支持多种智能体类型（车辆、行人、自行车），兼容 Waymo Open Motion Dataset，并集成了主流强化学习库（如 SB3、Pufferlib）。

## 2. 目录结构说明

```
gpudrive/
│
├── assets/           # 仿真相关资源（模型、图片、GIF等）
├── baselines/        # 强化学习基线算法与配置
│   └── ppo/          # PPO 算法实现（SB3、Pufferlib）
├── data/             # 数据集（raw 原始/processed 处理后）
│   └── processed/examples/  # 示例场景数据
├── data_utils/       # 数据处理与分析工具
├── examples/         # 教程与实验示例
│   ├── tutorials/    # 入门教程（Jupyter Notebook）
│   └── experimental/ # 实验脚本与配置
├── external/         # 外部依赖（如 madrona、json 库等）
├── gpudrive/         # 核心 Python 包
│   ├── agents/       # 智能体实现
│   ├── datatypes/    # 数据结构定义
│   ├── env/          # 环境实现（含 gymnasium 封装）
│   ├── integrations/ # 与 RL 框架集成（pufferlib、sb3、vbd）
│   ├── networks/     # 神经网络结构
│   ├── utils/        # 工具函数与配置
│   └── visualize/    # 可视化工具
├── src/              # C++ 源码（核心仿真引擎）
├── tests/            # 测试用例（Python/C++）
├── build_gpudrive.py # 构建脚本
├── Dockerfile        # Docker 镜像构建文件
└── README.md         # 项目说明文档
```

## 3. 主要模块功能简述

### 3.1 gpudrive/agents
- 智能体基类、策略智能体、随机智能体等实现。

### 3.2 gpudrive/datatypes
- 定义动作、观测、路网、轨迹等核心数据结构。

### 3.3 gpudrive/env
- 环境主模块，支持多种动力学模型（如自行车模型、DeltaLocal、StateDynamics）。
- 支持离散/连续动作空间、丰富的观测空间（自车、伙伴、路网、LiDAR）。
- 提供多种配置与批量场景切换，详见 `README.md`。

### 3.4 gpudrive/integrations
- 集成主流 RL 框架（pufferlib、sb3、vbd），便于训练与评测。

### 3.5 gpudrive/networks
- 神经网络结构定义，如基本前馈网络、Late Fusion 等。

### 3.6 gpudrive/utils
- 配置管理、批量任务脚本、几何工具等。

### 3.7 gpudrive/visualize
- 仿真与观测可视化工具。

### 3.8 baselines
- 强化学习基线算法实现与配置（如 PPO）。

### 3.9 data_utils
- 数据处理、行为检测、Waymo 数据转换等工具。

### 3.10 examples
- 教程（Jupyter Notebook）、实验脚本与配置，便于快速上手和复现论文实验。

### 3.11 src
- C++ 仿真引擎源码，负责高性能物理仿真与底层实现。

### 3.12 tests
- Python 与 C++ 单元测试，覆盖核心功能与算法。

## 4. 依赖与环境说明

- Python >= 3.11
- CMake >= 3.24
- CUDA Toolkit >= 12.2 且 <= 12.4
- 推荐使用 uv/pyenv/conda 管理 Python 环境
- 可选 Docker 支持，便于快速部署

详细依赖与安装方法见 `README.md`，支持源码编译与 Docker 镜像两种方式。

## 5. 运行与测试方法

- 编译 C++ 仿真引擎后，安装 Python 包（`pip install -e .`）
- 运行教程与示例脚本（见 `examples/tutorials`）
- 运行单元测试：`pytest`
- 训练/评测 RL 智能体：参考 `baselines/ppo/` 及 `gpudrive/integrations/`

## 6. 其他补充说明

- 支持多种动力学模型与观测空间，适配不同研究需求
- 提供预训练策略（可通过 huggingface_hub 加载）
- 数据集支持 mini/大规模版本，便于快速实验与大规模训练
- 详细文档与教程见 `README.md` 及 `examples/tutorials`

## 7. 教程与示例（examples/tutorials）使用说明

本目录下包含了 GPUDrive 的核心功能、数据加载、环境交互、可视化、智能体行为多样性等方面的 Jupyter Notebook 教程和文档，适合新手快速上手和进阶用户深入理解。主要内容如下：

### 01_scenario_loading.ipynb  
**用途**：介绍如何加载和理解交通场景数据，展示 Waymo Open Motion Dataset (WOMD) 的结构，以及如何用 GPUDrive 处理和迭代场景数据。  
**适用人群**：初学者，想了解数据结构和自定义数据加载流程的用户。  
**主要内容**：
- WOMD 数据集结构与文件说明
- 使用 SceneDataLoader 加载和批量处理场景
- 交通场景的主要字段和可视化

### 02_simulator_demo.ipynb  
**用途**：演示如何在 Python 中直接操作 GPUDrive 仿真器，介绍底层 C++ 仿真引擎的基本用法。  
**适用人群**：需要底层自定义仿真流程或调试底层接口的用户。  
**主要内容**：
- 仿真器对象的创建与参数设置
- 车辆状态、观测、奖励等张量的导出
- 多智能体与多场景的并行仿真

### 03_gym_env_demo.ipynb  
**用途**：展示如何通过 gymnasium 接口与 GPUDrive 环境交互，适合 RL 训练和评测。  
**适用人群**：强化学习研究者、需要标准 RL 环境接口的用户。  
**主要内容**：
- 环境初始化与配置
- 多环境并行 rollout
- 随机动作采样与可视化

### 04_use_pretrained_sim_agent.ipynb  
**用途**：演示如何加载和使用 HuggingFace Hub 上的预训练智能体策略进行仿真。  
**适用人群**：希望直接复现论文结果或基于预训练模型做下游任务的用户。  
**主要内容**：
- 加载预训练策略（如 policy_S10_000_02_27）
- 环境与模型参数对齐
- 智能体推理与 rollout 可视化

### 05_step_with_expert_actions.ipynb  
**用途**：展示如何提取专家（人类驾驶）动作轨迹，并用不同动力学模型复现专家行为。  
**适用人群**：需要 imitation learning、专家演示复现的用户。  
**主要内容**：
- 不同动力学模型下的专家动作提取
- 用专家动作驱动环境并可视化

### 06_visualizer_demo.ipynb  
**用途**：演示 GPUDrive 的可视化工具，包括鸟瞰图、智能体视角等。  
**适用人群**：需要仿真可视化、调试观测空间的用户。  
**主要内容**：
- 环境状态的鸟瞰图渲染
- 智能体第一视角观测渲染

### 07_agent_behavior_diversity.md  
**用途**：介绍如何通过奖励条件化（reward conditioning）实现智能体行为多样性，支持多种行为风格（如保守、激进等）。  
**适用人群**：研究多样性策略、行为调控的用户。  
**主要内容**：
- 奖励条件化的三种模式（随机、预设、固定）
- 不同行为风格的配置与切换
- 奖励权重纳入观测空间

### 08_multiple_policies.ipynb  
**用途**：演示如何在同一环境中并行部署多个不同的智能体策略，实现多策略对比与混合仿真。  
**适用人群**：需要多策略评测、对比实验的用户。  
**主要内容**：
- 多策略 mask 的创建与分配
- 多策略 rollout 与可视化

---

### 使用建议

1. 推荐按顺序阅读 01~03，快速了解数据、仿真器和 RL 环境接口。
2. 04、05 适合进阶用户，分别用于复现预训练智能体和专家演示。
3. 06、07、08 适合探索可视化、多样性行为和多策略仿真等高级功能。
4. 所有 notebook 可直接用 JupyterLab/Notebook 打开运行，部分依赖 GPU/CUDA 环境。

---

如需更详细的 API 说明或某一模块的深入解读，请告知具体需求！ 