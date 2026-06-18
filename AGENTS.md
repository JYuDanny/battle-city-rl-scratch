# AGENTS.md - Battle City RL Training Framework

> AI 代理工作指南。本项目使用 superpowers 技能体系进行开发管理。

## 环境

| 平台 | Conda 环境 | 硬件 | 用途 |
|------|-----------|------|------|
| **macOS (本地)** | `tank-rl-mac` (Python 3.11) | Apple Silicon (MPS) | 开发、测试、调试代码 |
| **RunPod (云端)** | `tank-rl-runpod` (Python 3.11) | RTX 4090 24GB | 正式训练 |

- **操作系统**: macOS 15 (本地) / Linux (RunPod)
- **设备策略**:
  - 本地 Mac: `--device auto` → MPS (Apple Silicon GPU) → CPU (fallback)
  - RunPod: `--device cuda` 强制使用 CUDA

## 项目约定

### 代码风格
- 所有模块、类、函数需包含详细中文注释（教学导向）
- 遵循 PEP 8 规范
- 文件职责单一：envs/ 纯环境、rl/ 纯算法、training/ 纯编排

### 游戏本体原则
- **tirinox/pybattlecity 作为 git submodule，只读不改**
- 禁止在 Curriculum 或任何模块中修改游戏原始参数（敌人数量、基地围墙、道具效果等）
- 训练中的所有游戏逻辑由 tirinox 原版控制

### 测试
- 使用 pytest，TDD 模式：先写失败测试，再写实现
- 测试文件放在 `tests/` 目录，与被测模块同名（如 `test_tile_env.py`）

### Git
- Commit message 格式: `feat: / fix: / chore: / init:` + 中文描述
- 子模块管理: `git submodule update --init --recursive`

## 技能体系 (Superpowers)

本项目使用以下技能工作流:

```
brainstorming → writing-plans → subagent-driven-development → finishing-a-development-branch
```

| 技能 | 用途 |
|------|------|
| `brainstorming` | 任何新功能/修改前必须经过设计讨论 |
| `writing-plans` | 设计通过后编写详细实施计划 |
| `subagent-driven-development` | 逐任务执行实施计划 |
| `test-driven-development` | 编写代码前先写测试 |
| `verification-before-completion` | 完成声明前必须运行验证命令 |
| `systematic-debugging` | Bug 修复前必须定位根因 |
| `dispatching-parallel-agents` | 多个独立任务并行处理 |

## 关键命令

```bash
# 激活环境
conda activate tank-rl-mac

# 运行测试
pytest tests/ -v

# 训练 (tirinox 真实环境, 带运行名留存记录)
python scripts/train_tir.py --device auto --run-name exp1

# 评估
python scripts/eval.py --ckpt checkpoints/2026-06-18_exp1/checkpoint_1000000.pt --episodes 10

# TensorBoard (查看全部训练)
tensorboard --logdir runs
# 或指定某次运行
tensorboard --logdir runs/2026-06-18_opt1

# 子模块
git submodule update --init --recursive
```

## 目录规范

### Checkpoint 管理
- 禁止直接在 `checkpoints/` 根目录存放 `.pt` 文件
- 每次正式训练使用 `--run-name` 参数, checkpoint 自动归档到 `checkpoints/<YYYY-MM-DD>_<name>/`
- 示例: `checkpoints/2026-06-18_run1/checkpoint_501760.pt`
- 废弃的 checkpoint 子目录直接删除即可

### 清理规则
- `runs/` 目录存放 TensorBoard 日志, 每次新实验前清空
- `__pycache__/` 和 `.pytest_cache/` 已 gitignore, 无需手动管理

## 架构

```
envs/           # Gymnasium Env 层（不感知 RL）
  tir_env.py    # tirinox 真实游戏包装器 (84×84×3 像素帧, Discrete 10)
  wrappers.py   # RecordVideo 包装器
rl/             # RL 算法层（不感知具体游戏）
  network.py    # Conv2D Actor-Critic 网络
  ppo.py        # PPO Trainer
  buffer.py     # GAE Buffer
  config.py     # 超参数
training/       # 编排层
  trainer.py    # 主训练循环
  curriculum.py # 动态难度课程学习
  reward_shaper.py # 奖励塑形 (面朝/射击引导)
```

## 依赖

```
torch, gymnasium, numpy, tensorboard, matplotlib, pygame, moviepy, pytest
```
