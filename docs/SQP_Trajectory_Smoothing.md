# SQP 轨迹平滑 — 技术文档

> 文件：`gpudrive/utils/trajectory_sqp_smoothing.py`
> 求解器：scipy SLSQP（Sequential Least-Squares Quadratic Programming）

---

## 1. 问题概述

自动驾驶仿真中，策略网络输出的**离散动作**经动力学模型产生的轨迹存在锯齿和抖动。
本模块使用 **SQP（序列二次规划）** 对仿真后的完整轨迹做**离线后处理**，
联合优化四个状态量，在保持轨迹保真度的同时提高平滑性和运动学一致性。

---

## 2. 状态量

轨迹包含 $N$ 个路点，每个路点有 4 个状态量，采样间隔 $\Delta t = 0.1\,\text{s}$。

| 符号 | 含义 | 单位 | 来源 |
|------|------|------|------|
| $x_i$ | 全局 x 坐标 | m | `agent_states.pos_x` |
| $y_i$ | 全局 y 坐标 | m | `agent_states.pos_y` |
| $\psi_i$ | 航向角（yaw） | rad | `agent_states.rotation_angle` |
| $v_i$ | 纵向速度 | m/s | `self_observation_tensor[:,:,0]` |

### 决策变量

将所有路点的 4 个状态拼接成一个 $4N$ 维向量：

$$\mathbf{z} = \underbrace{[x_0, \dots, x_{N-1}]}_{N} \oplus \underbrace{[y_0, \dots, y_{N-1}]}_{N} \oplus \underbrace{[\psi_0, \dots, \psi_{N-1}]}_{N} \oplus \underbrace{[v_0, \dots, v_{N-1}]}_{N}$$

### 预处理

- **Yaw 展开**：优化前对 $\psi$ 序列做 `np.unwrap`，消除 $\pm\pi$ 跳变，保证差分有意义。
- **优化后回卷**：将结果 $\psi$ 回卷到 $[-\pi, \pi]$。

---

## 3. 目标函数

目标函数由 **4 类 8 项**组成，全部为连续可微：

$$J(\mathbf{z}) = J_{\text{pos}} + J_{\text{yaw}} + J_{\text{speed}} + J_{\text{kin}} + J_{\text{dev}}$$

### 3.1 位置平滑项 $J_{\text{pos}}$

$$J_{\text{pos}} = w_{pc}\,\|D_2\,\mathbf{x}\|^2 + w_{pc}\,\|D_2\,\mathbf{y}\|^2 + w_{pj}\,\|D_3\,\mathbf{x}\|^2 + w_{pj}\,\|D_3\,\mathbf{y}\|^2$$

| 项 | 有限差分矩阵 | 物理含义 |
|----|-------------|----------|
| $\|D_2\,\mathbf{x}\|^2$ | $D_2$：二阶差分 $(N\!-\!2)\times N$ | **曲率**代理——惩罚路径弯折 |
| $\|D_3\,\mathbf{x}\|^2$ | $D_3$：三阶差分 $(N\!-\!3)\times N$ | **曲率变化率**（jerk）——惩罚曲率突变 |

其中有限差分矩阵定义：

$$D_1: \quad (D_1\mathbf{x})_i = x_{i+1} - x_i$$

$$D_2: \quad (D_2\mathbf{x})_i = x_{i-1} - 2x_i + x_{i+1}$$

$$D_3: \quad (D_3\mathbf{x})_i = -x_{i-1} + 3x_i - 3x_{i+1} + x_{i+2}$$

| 参数 | 符号 | 默认值 | 作用 |
|------|------|--------|------|
| `w_pos_curv` | $w_{pc}$ | 10.0 | 越大路径越平滑 |
| `w_pos_jerk` | $w_{pj}$ | 5.0 | 越大曲率变化越平缓 |

### 3.2 航向角平滑项 $J_{\text{yaw}}$

$$J_{\text{yaw}} = w_{yr}\,\|D_1\,\boldsymbol{\psi}\|^2 + w_{ya}\,\|D_2\,\boldsymbol{\psi}\|^2$$

| 项 | 物理含义 |
|----|----------|
| $\|D_1\,\boldsymbol{\psi}\|^2$ | **横摆角速度**平滑——惩罚快速转向 |
| $\|D_2\,\boldsymbol{\psi}\|^2$ | **横摆角加速度**平滑——惩罚转向突变 |

| 参数 | 符号 | 默认值 |
|------|------|--------|
| `w_yaw_rate` | $w_{yr}$ | 8.0 |
| `w_yaw_accel` | $w_{ya}$ | 3.0 |

### 3.3 速度平滑项 $J_{\text{speed}}$

$$J_{\text{speed}} = w_{sa}\,\|D_1\,\mathbf{v}\|^2 + w_{sj}\,\|D_2\,\mathbf{v}\|^2$$

| 项 | 物理含义 |
|----|----------|
| $\|D_1\,\mathbf{v}\|^2$ | **加速度**平滑——惩罚急加速/急刹 |
| $\|D_2\,\mathbf{v}\|^2$ | **加加速度**（jerk）平滑——惩罚加速度突变 |

| 参数 | 符号 | 默认值 |
|------|------|--------|
| `w_speed_accel` | $w_{sa}$ | 8.0 |
| `w_speed_jerk` | $w_{sj}$ | 3.0 |

### 3.4 运动学一致性项 $J_{\text{kin}}$（核心）

$$J_{\text{kin}} = w_{\text{kin}} \sum_{i=0}^{N-2}\left[\left(\Delta x_i - v_i \cos\psi_i \cdot \Delta t\right)^2 + \left(\Delta y_i - v_i \sin\psi_i \cdot \Delta t\right)^2\right]$$

其中 $\Delta x_i = x_{i+1} - x_i$，$\Delta y_i = y_{i+1} - y_i$。

**物理含义**：这是自行车运动学模型（bicycle kinematic model）的前向约束。
如果车辆严格沿航向以速度 $v$ 行驶 $\Delta t$ 时间，位移应为 $(v\cos\psi\cdot\Delta t,\; v\sin\psi\cdot\Delta t)$。
该项惩罚实际位移与运动学预测的偏差，**将 $(x,y)$ 与 $(\psi, v)$ 耦合**在一起，
确保优化后的四个状态量在物理上是一致的。

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| `w_kinematic` | $w_{\text{kin}}$ | 15.0 | 权重最大，确保物理一致性 |

> **注意**：该项包含 $\cos\psi$、$\sin\psi$，使目标函数为**非线性**的，
> 这正是需要 SQP（而非简单 QP）的原因。

### 3.5 保真项 $J_{\text{dev}}$

$$J_{\text{dev}} = w_{d,xy}\left(\|\mathbf{x}-\mathbf{x}^0\|^2 + \|\mathbf{y}-\mathbf{y}^0\|^2\right) + w_{d,\psi}\,\|\boldsymbol{\psi}-\boldsymbol{\psi}^0\|^2 + w_{d,v}\,\|\mathbf{v}-\mathbf{v}^0\|^2$$

防止优化结果偏离原始轨迹过远。

| 参数 | 符号 | 默认值 |
|------|------|--------|
| `w_deviation_xy` | $w_{d,xy}$ | 1.0 |
| `w_deviation_yaw` | $w_{d,\psi}$ | 2.0 |
| `w_deviation_speed` | $w_{d,v}$ | 2.0 |

---

## 4. 约束条件

### 4.1 盒约束（Box Bounds）

对每个路点 $i$（$1 \le i \le N-2$）施加盒约束，限制每个状态量的偏移范围：

$$x_i^0 - d_{xy} \le x_i \le x_i^0 + d_{xy}$$
$$y_i^0 - d_{xy} \le y_i \le y_i^0 + d_{xy}$$
$$\psi_i^0 - d_{\psi} \le \psi_i \le \psi_i^0 + d_{\psi}$$
$$\max(0,\; v_i^0 - d_v) \le v_i \le v_i^0 + d_v$$

| 参数 | 符号 | 默认值 | 含义 |
|------|------|--------|------|
| `max_deviation_xy` | $d_{xy}$ | 2.0 m | 位置最大偏移 |
| `max_deviation_yaw` | $d_\psi$ | 0.3 rad ≈ 17° | 航向最大偏移 |
| `max_deviation_speed` | $d_v$ | 3.0 m/s | 速度最大偏移 |

> 速度下界取 $\max(0, \cdot)$ 以保证非负。

### 4.2 端点固定约束

当 `fix_endpoints=True`（默认）时，首尾路点被钉死：

$$x_0 = x_0^0, \quad y_0 = y_0^0, \quad \psi_0 = \psi_0^0, \quad v_0 = v_0^0$$
$$x_{N-1} = x_{N-1}^0, \quad y_{N-1} = y_{N-1}^0, \quad \psi_{N-1} = \psi_{N-1}^0, \quad v_{N-1} = v_{N-1}^0$$

实现方式：将端点的盒约束上下界设为相同值（$lb = ub = \text{原始值}$）。

---

## 5. 解析梯度

为加速 SLSQP 收敛，提供了完整的 $4N$ 维解析梯度。

### 5.1 二次项梯度

位置、航向、速度各通道的二次项梯度结构相同，以 $x$ 为例：

$$\frac{\partial J_{\text{quad}}}{\partial \mathbf{x}} = 2\,H_{xy}\,\mathbf{x} - 2\,w_{d,xy}\,\mathbf{x}^0$$

其中 $H_{xy} = w_{pc}\,D_2^\top D_2 + w_{pj}\,D_3^\top D_3 + w_{d,xy}\,I_N$。

### 5.2 运动学项梯度

定义残差：$e_i^x = \Delta x_i - v_i\cos\psi_i\cdot\Delta t$，$e_i^y$ 类似。

$$\frac{\partial J_{\text{kin}}}{\partial \mathbf{x}} = 2\,w_{\text{kin}}\,D_1^\top\,\mathbf{e}^x$$

$$\frac{\partial J_{\text{kin}}}{\partial \psi_i} = 2\,w_{\text{kin}}\,\Delta t\,v_i\left(e_i^x\sin\psi_i - e_i^y\cos\psi_i\right), \quad i < N\!-\!1$$

$$\frac{\partial J_{\text{kin}}}{\partial v_i} = -2\,w_{\text{kin}}\,\Delta t\left(e_i^x\cos\psi_i + e_i^y\sin\psi_i\right), \quad i < N\!-\!1$$

> 梯度中 $\psi_{N-1}$ 和 $v_{N-1}$ 的运动学分量为 0（无后续路点）。

---

## 6. 求解方法

| 项目 | 说明 |
|------|------|
| 算法 | **SLSQP**（Sequential Least-Squares QP） |
| 实现 | `scipy.optimize.minimize(method='SLSQP')` |
| 变量数 | $4N$（典型 $N \approx 91$，即 364 个变量） |
| 约束类型 | 盒约束（bound constraints） |
| 梯度 | 解析梯度（`jac=gradient`） |
| 收敛容差 | `ftol = 1e-9` |
| 最大迭代 | 300 |
| 典型耗时 | < 10 ms / 条轨迹 |

### SQP 工作原理

SLSQP 在每次迭代中：
1. 在当前点处用二阶近似构造一个**二次规划子问题（QP）**
2. 将约束线性化
3. 求解 QP 获得搜索方向
4. 线搜索确定步长
5. 更新解并重复

由于运动学项中的 $\cos\psi$、$\sin\psi$ 使目标非线性，SQP 通过迭代线性化自然处理这一非线性。

---

## 7. 参数调优指南

### 权重平衡关系

```
平滑度 ← w_pos_curv, w_yaw_rate, w_speed_accel → 大
保真度 ← w_deviation_xy, w_deviation_yaw, w_deviation_speed → 大
物理性 ← w_kinematic → 大
```

- **想要更平滑**：增大 `w_pos_curv`（如 20）、`w_yaw_rate`（如 15）
- **想要更贴近原始**：增大 `w_deviation_*`（如 5）、减小 `max_deviation_*`
- **想要更符合物理**：增大 `w_kinematic`（如 25）
- **高速场景**：减小 `max_deviation_yaw`（如 0.1），因为高速时航向变化应更小

### 当前默认值总览

| 参数 | 默认值 | 类别 |
|------|--------|------|
| `w_pos_curv` | 10.0 | 位置曲率 |
| `w_pos_jerk` | 5.0 | 位置 jerk |
| `w_yaw_rate` | 8.0 | 航向变化率 |
| `w_yaw_accel` | 3.0 | 航向加速度 |
| `w_speed_accel` | 8.0 | 速度加速度 |
| `w_speed_jerk` | 3.0 | 速度 jerk |
| `w_kinematic` | **15.0** | 运动学一致性 |
| `w_deviation_xy` | 1.0 | xy 保真 |
| `w_deviation_yaw` | 2.0 | yaw 保真 |
| `w_deviation_speed` | 2.0 | speed 保真 |
| `max_deviation_xy` | 2.0 m | xy 偏移上限 |
| `max_deviation_yaw` | 0.3 rad | yaw 偏移上限 |
| `max_deviation_speed` | 3.0 m/s | speed 偏移上限 |

---

## 8. 数据流

```
仿真循环
  ├─ 每步记录: (x, y, yaw, speed, step) → trajectories dict
  └─ 渲染基础帧（道路 + 车辆）→ base_frame_data

仿真结束
  ├─ SQP 优化（每个智能体一次）→ smoothed_trajectories dict
  ├─ 合成 GIF：base_frame + 轨迹线（绿色/红色）
  └─ 生成 v/yaw 对比图（PNG）
```

---

## 9. 文件结构

| 文件 | 职责 |
|------|------|
| `gpudrive/utils/trajectory_sqp_smoothing.py` | SQP 求解器（2D / 4D） |
| `examples/test/use_training_agen_pufferlib.py` | 仿真主脚本、参数配置、GIF 合成 |
| `output/<model>_gif/simulation_env_*.gif` | 含轨迹的仿真动画 |
| `output/<model>_gif/sqp_v_yaw_env*_agent*.png` | v / yaw 优化前后对比图 |
