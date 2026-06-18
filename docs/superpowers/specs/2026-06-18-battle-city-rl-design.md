# Battle City RL Training Framework - Design Spec

> 从零构建强化学习训练框架，训练 Battle City 一命通关智能体。教学导向，详细中文注释，系统讲解 RL 核心原理。

## 技术栈

- Python 3.11
- Conda 环境: `tank-rl-teach`
- PyTorch (RL 训练)
- Gymnasium (Env 接口标准)
- Pygame + tirinox/pybattlecity (游戏引擎, git submodule)
- TensorBoard + Matplotlib (训练监控)
- pytest (单元测试)

## 架构总览

```
battle-city-rl-scratch/
├── envs/                       # 环境层 (Gymnasium Env)
│   ├── tile_env.py             # 简化13x13 Tile环境（快速原型）
│   ├── tir_env.py              # tirinox 真实环境包装（后期对接）
│   └── wrappers.py             # RecordVideo / 通用包装器
├── rl/                         # RL算法层
│   ├── network.py              # 共享backbone + Actor/Critic Heads
│   ├── ppo.py                  # PPO Trainer（rollout→advantage→clip→update）
│   ├── buffer.py               # Experience Buffer (GAE + 轨迹存储)
│   └── config.py               # 超参数配置
├── training/                   # 训练编排层
│   ├── trainer.py              # 主训练循环 + curriculum + checkpoint
│   ├── curriculum.py           # 关卡难度阶梯管理
│   └── reward_shaper.py        # 分层奖励 + 势能奖励计算
├── scripts/                    # 启动脚本
│   ├── train_tile.py           # 简化环境训练入口
│   ├── train_tir.py            # 真实环境训练入口
│   └── eval.py                 # 评估/录制视频
├── tests/                      # 测试
│   ├── test_tile_env.py
│   ├── test_buffer.py
│   ├── test_ppo.py
│   ├── test_network.py
│   ├── test_curriculum.py
│   └── test_reward_shaper.py
├── extern/                     # 第三方子模块
│   └── pybattlecity/           # git submodule: tirinox/pybattlecity
├── docs/
│   └── superpowers/
│       ├── specs/              # 设计文档
│       └── plans/              # 实施计划
├── objectives.md
└── AGENTS.md
```

**分层职责:**
- `envs/` — 纯环境逻辑，不感知 RL 算法
- `rl/` — 纯算法逻辑，不感知具体游戏
- `training/` — 编排 env + rl，控制训练流程
- **游戏本体 (tirinox) 只读不改**，所有 curriculum 差异通过选择不同关卡/参数配置实现，游戏引擎完全原版运行

## 组件设计

### 1. TileEnv（简化13×13环境）

| 属性 | 设计 |
|------|------|
| Observation Space | 13×13×(地形通道+实体通道+方向通道) 多通道张量；地形(brick/steel/water/trees/base)，实体(player/enemy/bullet/powerup)，方向 |
| Action Space | `Discrete(6)`: ↑ ↓ ← → 射击 静止 |
| 步进逻辑 | 回合制 tile-based: 玩家→敌人→碰撞→生成，每步视为 RL timestep |
| 终止条件 | 玩家死亡 or 基地被毁 or 所有敌人清理 |
| Info Dict | `enemies_killed`, `base_health`, `wave_number`, `player_alive` |

### 2. 网络架构

```
Observation [13×13×C]
    → Flatten → Embedding → ShareMLP [256, 256]
       ├── Actor Head → Linear(256→6)  → Categorical → action
       └── Critic Head → Linear(256→1) → value
```

后续升级像素输入时，ShareMLP 替换为 `Conv2D(3→32→64→64) → Flatten → FC(512)`。

### 3. PPO Trainer

- GAE: λ=0.95, γ=0.99
- Clip Range: ε=0.2
- Entropy Coef: 0.01
- K Epochs: 10 / rollout batch
- Mini-batch: 64
- Rollout steps: 2048

### 4. Reward Shaper

分层奖励 + 势能奖励:

| 事件 | 奖励值 |
|------|--------|
| 生存 | +0.01/step |
| 击杀敌人 | +2.0 |
| 清波 (wave clear) | +5.0 |
| 通关 (all waves) | +100.0 |
| 死亡 | -10.0 |
| 基地被毁 | -50.0 |
| 势能奖励 | +0.01 × (distance_old - distance_new) to nearest enemy |

### 5. Curriculum Learning

**核心原则：只读 tirinox 关卡数据，按关卡固有特征分组，不注入任何自定义参数。** 环境、敌人、道具、基地围墙等游戏内容始终由 tirinox 原版逻辑控制，不手工设定任何数值。

**关卡分组方式:**
读取 tirinox 全部 35 关的原始数据(敌人总数、墙体密度、地图尺寸等)，统计后按特征分布分位自动归入 3 个 difficulty bucket:
- Stage 1: 低难度关卡 (关卡池 ~1/3)
- Stage 2: 中难度关卡 (关卡池 ~1/3)
- Stage 3: 高难度关卡 (关卡池 ~1/3)

**解锁阈值:**
| 阶段切换 | 条件 |
|----------|------|
| Stage 1 → 2 | 最近 100 episode 通关率 ≥ 80%，且平均击杀数 ≥ 关卡平均敌人数×0.8 |
| Stage 2 → 3 | 最近 100 episode 通关率 ≥ 50% |
| 衰减回退 | 通关率 < 20% 持续 50 episode，自动回退上阶段 |
| 手动覆写 | `--force-stage` flag |

### 6. 训练主循环数据流

```
TileEnv.reset() → obs → PPO.select_action(obs) → action
  → TileEnv.step(action) → (obs', reward, done, info)
  → Buffer.store(obs, action, reward, value, log_prob, done)
  → [collect rollout_steps=2048]
  → Buffer.compute_gae(values, last_value) → advantages
  → for k in 1..K_epochs:
       for each mini-batch:
         PPO.update(obs, action, adv, old_logp, ret)
  → TensorBoard logging + checkpoint
  → curriculum.check(episode_stats) → possible stage switch
  → repeat
```

### 7. 错误处理

| 场景 | 处理 |
|------|------|
| tirinox 子模块缺失 | ImportError → 提示 `git submodule update --init` |
| CUDA OOM | 自动降级 batch_size 并 warn |
| NaN loss / grad | 回滚至上一 checkpoint，学习率 ×0.5 重试 |
| env step 崩溃 | 记录 crash 帧到 `crash_log/`，auto-reset 继续训练 |

### 8. 测试策略

**单元测试 (pytest):**
| 模块 | 测试内容 |
|------|----------|
| `envs/tile_env.py` | 动作→状态转换正确性、reward 计算、done 条件 |
| `rl/buffer.py` | GAE 计算数值正确性 |
| `rl/ppo.py` | loss 下降趋势、clip 机制生效验证 |
| `rl/network.py` | 输入输出维度正确性 |
| `training/curriculum.py` | 通关率统计、阶段切换/回退逻辑 |
| `training/reward_shaper.py` | 各类事件 reward 计算验证 |

**集成验证:**
- 跑 1 个 mini rollout，检查 buffer 存储完整
- 100 episode 小规模训练 loss 无 NaN
- TensorBoard 监控 reward/advantage/loss/clip_fraction/entropy 曲线
- Matplotlib 阶段通关率变化图
- RecordVideo: 每 N 步生成 agent 游戏视频
