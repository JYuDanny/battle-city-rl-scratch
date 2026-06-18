# AGENTS.md - Battle City RL Training Framework

> AI 代理工作指南。本项目使用 superpowers 技能体系进行开发管理。

## 环境

- **Conda 环境**: `tank-rl-teach` (Python 3.11)
- **硬件**: RTX 4060 8GB VRAM
- **操作系统**: Windows

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
conda activate tank-rl-teach

# 运行测试
pytest tests/ -v

# 训练 (Tile 环境, 带运行名留存记录)
python scripts/train_tile.py --device auto --run-name exp1

# 评估
python scripts/eval.py --ckpt checkpoints/2026-06-18_run1/checkpoint_501760.pt --episodes 10

# 评估并渲染 (终端清屏动画)
python scripts/eval.py --ckpt checkpoints/2026-06-18_run1/checkpoint_501760.pt --episodes 2 --render

# TensorBoard
tensorboard --logdir runs

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
  tile_env.py   # 13×13 简化环境
  tir_env.py    # tirinox 包装器（后期）
rl/             # RL 算法层（不感知具体游戏）
  network.py    # Actor-Critic 共享网络 (Conv2D backbone)
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
