# Battle City RL — 从零搭建强化学习训练框架

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.3-green)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

以 **vibe-coding** 方式从零搭建一套完整的强化学习训练框架，训练一个能在经典**坦克大战 (Battle City)** 中实现一命通关的智能体。

> 基于 [tirinox/pybattlecity](https://github.com/tirinox/pybattlecity) 原版游戏引擎，84×84 像素帧直接作为 CNN 输入。

---

## 特性

- **真实游戏引擎** — 包装 tirinox/pybattlecity，4 种敌人、6 种道具、子弹互抵、基地保护全部保留
- **像素帧输入** — 84×84×3 RGB 画面直接作为 Conv2D 输入，无需手工特征工程
- **三层解耦架构** — `envs/` 纯环境层、`rl/` 纯算法层、`training/` 编排层
- **PPO-Clip 完整实现** — GAE advantage 估计、clip 约束、熵正则化、mini-batch 梯度更新
- **动态 Curriculum Learning** — 根据通关率连续调节难度参数
- **Reward Shaping** — 基于游戏内置评分系统 + 行为塑形 (面朝/射击引导)
- **TensorBoard 监控** — 实时查看 Reward、Loss、Score、Difficulty 曲线
- **Checkpoint 管理** — 训练中断可恢复，按运行名自动归档到子目录

---

## 快速开始

### 环境要求

- Python 3.11
- NVIDIA GPU (RTX 4060 推荐，8GB+ VRAM)
- Windows / Linux / macOS

### 安装

```bash
# 1. 创建 Conda 环境
conda create -n tank-rl-teach python=3.11 -y
conda activate tank-rl-teach

# 2. 安装 PyTorch (CUDA 版)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 克隆项目 (含子模块)
git clone --recursive https://github.com/JYuDanny/battle-city-rl-scratch.git
cd battle-city-rl-scratch
pip install -e .
pip install opencv-python

# 4. 验证安装
python -c "import torch; print(torch.cuda.is_available())"  # True
pytest tests/ -v  # 31 passed
```

### 最小化训练验证

```bash
python scripts/train_tir.py --total-steps 10240 --device cuda --run-name test --fps 120
```

---

## 项目架构

```
battle-city-rl-scratch/
├── envs/                       # 环境层 (Gymnasium Env)
│   ├── tir_env.py              # tirinox 真实游戏包装器 (唯一环境)
│   └── wrappers.py             # RecordVideo 包装器
├── rl/                         # RL 算法层
│   ├── config.py               # 统一超参数 dataclass
│   ├── network.py              # Conv2D Actor-Critic 网络
│   ├── buffer.py               # GAE Experience Buffer
│   └── ppo.py                  # PPO-Clip Trainer
├── training/                   # 训练编排层
│   ├── trainer.py              # 主训练循环 + 日志 + Checkpoint
│   ├── curriculum.py           # 动态难度课程学习
│   └── reward_shaper.py        # 奖励塑形器
├── scripts/                    # 启动脚本
│   ├── train_tir.py            # tirinox 环境训练入口
│   └── eval.py                 # 评估脚本
├── extern/                     # 第三方子模块
│   └── pybattlecity/           # git submodule: tirinox/pybattlecity
├── tests/                      # pytest 单元测试
├── docs/                       # 设计文档 & 游戏规则
├── objectives.md               # 项目目标
└── AGENTS.md                   # AI 代理工作指南
```

### 设计原则

- **`envs/`** — 纯环境逻辑，不感知 RL 算法；实现标准 `gymnasium.Env` 接口
- **`rl/`** — 纯算法逻辑，不感知具体游戏；可复用到任何 Gymnasium 环境
- **`training/`** — 编排 env + rl；管理训练流程、日志、checkpoint、curriculum
- **游戏本体只读不改** — tirinox 作为 git submodule，禁止修改游戏原始参数

---

## 训练

### tirinox 真实环境

```bash
# 基础训练
python scripts/train_tir.py --device cuda --run-name exp1

# 自定义参数
python scripts/train_tir.py \
    --total-steps 5000000 \
    --lr 1e-4 \
    --seed 42 \
    --device cuda \
    --run-name exp2 \
    --fps 120 \
    --frame-skip 4

# 从 checkpoint 恢复训练
python scripts/train_tir.py --ckpt checkpoints/2026-06-18_exp1/checkpoint_1000000.pt

# 手动设置 Curriculum 难度
python scripts/train_tir.py --force-stage 2
```

### 环境参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-steps` | `10000000` | 总训练步数 |
| `--lr` | `1e-4` | Adam 学习率 |
| `--seed` | `42` | 随机种子 |
| `--device` | `auto` | 训练设备 (auto/cuda/cpu) |
| `--run-name` | — | 运行名称, 自动归档到子目录 |
| `--fps` | `60` | 游戏模拟帧率 (影响游戏内部计时器) |
| `--frame-skip` | `4` | 每 N 帧执行一次动作 |
| `--ckpt` | — | 恢复训练的 checkpoint 路径 |

### PPO 超参数

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
| Conv2D backbone | 32→64→64 channels, 3×3 kernel | 处理 84×84×3 像素帧 |
| Shared FC | 256 | 共享全连接层 |
| Actor Head | 256→10 | Discrete(10) 动作 logits |
| Critic Head | 256→1 | 状态价值输出 |

---

## 评估

```bash
# 仅统计 (最快, 无渲染)
python scripts/eval.py --ckpt checkpoints/2026-06-18_exp1/checkpoint_1000000.pt --episodes 10

# 实时窗口观看 agent 操作 (弹出 pygame 游戏画面)
python scripts/eval.py --ckpt checkpoints/2026-06-18_exp1/checkpoint_1000000.pt --render

# 录制视频保存到 videos/ 目录
python scripts/eval.py --ckpt checkpoints/2026-06-18_exp1/checkpoint_1000000.pt --record

# 确定性策略 + 窗口模式
python scripts/eval.py --ckpt checkpoints/2026-06-18_exp1/checkpoint_1000000.pt --render --deterministic

# 加速模拟 (--fps 越大越快)
python scripts/eval.py --ckpt checkpoints/2026-06-18_exp1/checkpoint_1000000.pt --render --fps 240
```

---

## 监控

### TensorBoard

训练过程中另开终端：

```bash
conda activate tank-rl-teach
tensorboard --logdir runs          # 查看全部训练
tensorboard --logdir runs/2026-06-18_exp1  # 只看某次运行
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
python scripts/eval.py --ckpt checkpoints/2026-06-18_exp1/checkpoint_1000000.pt --episodes 3
```

---

## 游戏环境 (TirEnv)

### 动作空间

| Action | 含义 |
|--------|------|
| 0-3 | ↑ ↓ ← → 移动 |
| 4 | 静止 |
| 5-8 | ↑+Fire ↓+Fire ←+Fire →+Fire |
| 9 | 静止+Fire |

### 奖励设计

基于游戏内置评分系统 (每击杀得分 / 100)：
- 普通坦克 +1.0, 快速坦克 +2.0, 火力坦克 +3.0, 装甲坦克 +4.0
- 生存 +0.005/frame
- 通关 +10.0, 基地被毁 -20.0

### 与真实 NES 游戏的对齐

| 机制 | 状态 |
|------|:---:|
| 20 辆敌人/关, 4 种类型 | ✅ |
| 子弹互抵 (对向消失) | ✅ |
| 道具系统 (6 种) | ✅ |
| 基地砖墙保护 | ✅ |
| 像素级平滑移动 | ✅ |
| 35 关预设地图 | ✅ |
| 3 条命 + 出生无敌 | ✅ |
| 冰面滑行 / 森林遮挡 | ✅ |

---

## 设计文档

完整设计文档位于 `docs/` 目录：
- `docs/battle-city-rules.md` — 完整游戏规则
- `docs/superpowers/specs/2026-06-18-battle-city-rl-design.md` — 架构设计
- `docs/superpowers/plans/2026-06-18-battle-city-rl-implementation.md` — 实施计划

---

## 路线图

- [x] Conv2D 像素帧网络 (84×84×3)
- [x] tirinox 真实游戏环境包装 (TirEnv)
- [x] 游戏评分系统 Reward Shaping
- [x] 动态难度 Curriculum Learning
- [x] TensorBoard 训练监控
- [x] Checkpoint 按运行名归档
- [ ] 正式 PPO 训练
- [ ] 并行采样多环境训练
- [ ] 模型发布

---

## License

MIT
