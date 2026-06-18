# Battle City RL — 从零搭建强化学习训练框架

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.3-green)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

以 **vibe-coding** 方式从零搭建一套完整的强化学习训练框架，训练一个能在经典**坦克大战 (Battle City)** 中实现一命通关的智能体。

> 教学导向项目 — 通过详细中文注释系统讲解 RL 核心原理：MDP、Bellman 方程、Policy Gradient、GAE、PPO Clip、熵正则化等。

---

## 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [项目架构](#项目架构)
- [训练](#训练)
- [评估](#评估)
- [监控](#监控)
- [课程学习](#课程学习)
- [设计文档](#设计文档)
- [路线图](#路线图)

---

## 特性

- **三层解耦架构** — `envs/` 纯环境层、`rl/` 纯算法层、`training/` 编排层，职责清晰互不依赖
- **PPO-Clip 完整实现** — GAE advantage 估计、clip 约束、熵正则化、mini-batch 梯度更新
- **Curriculum Learning** — 关卡按难度自动分组，Agent 满足通关率阈值后自动晋升阶段
- **Reward Shaping** — 分层奖励（生存/击杀/清波/通关）+ 势能奖励（基于距离变化）
- **TensorBoard 监控** — 实时查看 Reward、Loss、通关率、Stage 变化曲线
- **Checkpoint 管理** — 训练中断可恢复，支持任意 checkpoint 评估和渲染
- **生产级代码** — 完整类型标注、中文 docstring、25 条 pytest 单元测试

---

## 快速开始

### 环境要求

- Python 3.11
- NVIDIA GPU（RTX 4060 推荐，8GB+ VRAM）
- Windows / Linux / macOS

### 安装

```bash
# 1. 创建 Conda 环境
conda create -n tank-rl-teach python=3.11 -y
conda activate tank-rl-teach

# 2. 安装 PyTorch (CUDA 版)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装项目依赖
git clone https://github.com/JYuDanny/battle-city-rl-scratch.git
cd battle-city-rl-scratch
pip install -e .

# 4. 验证安装
python -c "import torch; print(torch.cuda.is_available())"  # True
pytest tests/ -v                                              # 25 passed
```

### 最小化训练验证

```bash
python scripts/train_tile.py --total-steps 51200 --device cuda
python scripts/eval.py --ckpt checkpoints/checkpoint_51200.pt --episodes 5 --render
```

---

## 项目架构

```
battle-city-rl-scratch/
├── envs/                       # 环境层 (Gymnasium Env)
│   ├── tile_env.py             # 13×13 简化 Tile 环境
│   ├── tir_env.py              # tirinox 真实环境包装器 (规划中)
│   └── wrappers.py             # RecordVideo 包装器
├── rl/                         # RL 算法层
│   ├── config.py               # 统一超参数 dataclass
│   ├── network.py              # Actor-Critic 共享网络
│   ├── buffer.py               # GAE Experience Buffer
│   └── ppo.py                  # PPO-Clip Trainer
├── training/                   # 训练编排层
│   ├── trainer.py              # 主训练循环 + 日志 + Checkpoint
│   ├── curriculum.py           # 动态难度课程学习
│   └── reward_shaper.py        # 分层奖励 + 行为塑形 (面朝/射击引导)
├── scripts/                    # 启动脚本
│   ├── train_tile.py           # Tile 环境训练入口
│   └── eval.py                 # 评估 & 录制视频
├── tests/                      # pytest 单元测试
├── docs/                       # 设计文档 & 实施计划
├── objectives.md               # 项目目标
└── AGENTS.md                   # AI 代理工作指南
```

### 设计原则

- **`envs/`** — 纯环境逻辑，不感知 RL 算法；实现标准 `gymnasium.Env` 接口
- **`rl/`** — 纯算法逻辑，不感知具体游戏；可复用到任何 Gymnasium 环境
- **`training/`** — 编排 env + rl；管理训练流程、日志、checkpoint、curriculum
- **游戏本体只读不改** — 后期对接的 tirinox 作为 git submodule，禁止修改游戏原始参数

---

## 训练

### Tile 环境 (当前可用)

```bash
# 基础训练 (使用默认 1000 万步)
python scripts/train_tile.py --device cuda

# 自定义参数
python scripts/train_tile.py \
    --total-steps 5000000 \
    --lr 1e-4 \
    --seed 42 \
    --device cuda

# 从 checkpoint 恢复训练
python scripts/train_tile.py --ckpt checkpoints/checkpoint_1000000.pt

# 手动设置 Curriculum Stage
python scripts/train_tile.py --force-stage 2
```

### 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-steps` | `10000000` | 总训练步数 |
| `--lr` | `3e-4` | Adam 学习率 |
| `--seed` | `42` | 随机种子 |
| `--device` | `auto` | 训练设备 (auto/cuda/cpu) |
| `--force-stage` | — | 手动设置 curriculum stage (1-3) |
| `--ckpt` | — | 恢复训练的 checkpoint 路径 |

### PPO 超参数 (rl/config.py)

| 参数 | 值 | 说明 |
|------|-----|------|
| `rollout_steps` | 2048 | 每次收集的交互步数 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ 参数 |
| `clip_epsilon` | 0.2 | PPO clip 范围 |
| `k_epochs` | 5 | 重用数据的更新轮数 |
| `mini_batch_size` | 256 | 小批次样本数 |
| `entropy_coef` | 0.03 | 熵正则化系数 |

### 网络架构

| 组件 | 配置 | 说明 |
|------|------|------|
| Conv2D backbone | 32→64 channels, 3×3 kernel | 保留空间关系的卷积特征提取 |
| Shared FC | 512 | 共享全连接层 |
| Actor Head | 512→6 | 动作 logits 输出 |
| Critic Head | 512→1 | 状态价值输出 |

---

## 评估

```bash
# 基础评估
python scripts/eval.py --ckpt checkpoints/checkpoint_1000000.pt --episodes 10

# 渲染模式 (实时查看 agent 操作)
python scripts/eval.py --ckpt checkpoints/checkpoint_1000000.pt --episodes 5 --render

# 确定性策略 (取最高概率动作)
python scripts/eval.py --ckpt checkpoints/checkpoint_1000000.pt --deterministic --render
```

---

## 监控

### TensorBoard

训练过程中另开终端：

```bash
conda activate tank-rl-teach
tensorboard --logdir runs
```

浏览器打开 `http://localhost:6006`，可监控：

- `Episode/Reward` — 每局累计奖励
- `Episode/Length` — 每局存活步数
- `Train/Loss` — PPO 训练 loss
- `Curriculum/WinRate` — 滑动窗口通关率
- `Curriculum/Difficulty` — 当前动态难度

### 训练中阶段性评估

训练不中断的情况下，在另一个终端加载最新 checkpoint：

```bash
# 查看最新 checkpoint 的 agent 表现
python scripts/eval.py --ckpt checkpoints/checkpoint_1000000.pt --episodes 3 --render
```

> 提示: `--render` 模式下终端会自动清屏实现动画效果，不会逐行堆叠输出。

---

## 课程学习

动态难度调节 — 不再使用固定 3-stage 分桶，根据 agent 近期通关率连续调节难度：

| 通关率区间 | 动作 | 效果 |
|-----------|------|------|
| > 70% | 升难度 (+0.1) | 增加敌人数量、围墙密度 |
| 30% ~ 70% | 维持 | 保持当前配置 |
| < 30% | 降难度 (-0.1) | 减少敌人数量、降低围墙密度 |

难度参数 (0.0~1.0) 直接影响环境生成：
- `num_enemies`: 2~6 人
- `wall_density`: 5%~35%
- 从最简单开始 (difficulty=0.0)，逐步引导 agent 适应困难局面

---

## 设计文档

完整设计文档和实施计划位于 `docs/` 目录：

- `docs/superpowers/specs/2026-06-18-battle-city-rl-design.md` — 架构设计
- `docs/superpowers/plans/2026-06-18-battle-city-rl-implementation.md` — 逐任务实施计划

---

## 路线图

- [x] 13×13 Tile 环境 + PPO 训练流程
- [x] 分层奖励 + 行为塑形 Reward Shaper (面朝/射击引导)
- [x] Conv2D backbone 保留空间信息
- [x] 动态难度 Curriculum Learning
- [x] 智能敌人 AI (玩家追踪+射击)
- [x] TensorBoard 训练监控
- [x] Checkpoint 保存/恢复
- [x] 评估脚本 + 终端清屏动画
- [ ] tirinox 真实游戏环境对接
- [ ] 并行采样多环境训练
- [ ] 正式训练 + 模型发布

---

## License

MIT
