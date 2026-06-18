# Battle City RL — 从零搭建强化学习训练框架

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.3-green)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

以 **vibe-coding** 方式从零搭建一套完整的强化学习训练框架，训练一个能在经典**坦克大战 (Battle City)** 中实现一命通关的智能体。

> 基于 [tirinox/pybattlecity](https://github.com/tirinox/pybattlecity) 原版游戏引擎，84×84 像素帧直接作为 CNN 输入。

---

## 快速开始

提供了两套环境配置方案：

| 平台 | 硬件 | Conda 环境 | 用途 |
|------|------|-----------|------|
| **macOS 本地** | Apple Silicon (MPS) | `tank-rl-mac` | 开发、测试、调试 |
| **RunPod 云端** | RTX 4090 24GB (CUDA) | `tank-rl-runpod` | 正式训练 |

### 本地安装 (macOS, Apple Silicon)

```bash
# 1. 创建 Conda 环境
conda create -n tank-rl-mac python=3.11 -y
conda activate tank-rl-mac

# 2. 安装 PyTorch (自动包含 MPS 加速)
pip install torch torchvision torchaudio

# 3. 克隆项目 (含子模块)
git clone --recursive https://github.com/JYuDanny/battle-city-rl-scratch.git
cd battle-city-rl-scratch
git checkout mac
git submodule update --init --recursive

# 4. 安装项目依赖
pip install -e .
pip install opencv-python

# 5. 验证
python -c "import torch; print(torch.backends.mps.is_available())"  # True
pytest tests/ -v  # 31 passed
```

### RunPod 安装 (云端 RTX 4090)

```bash
# 0. 前置步骤: 在 RunPod 上创建 RTX 4090 Pod 并挂载 Network Volume
#    详见下方「RunPod 云端训练」章节

# 1. 克隆项目 (mac 分支, 包含完整的 RunPod 适配代码)
git clone --recurse-submodules -b mac \
    https://github.com/JYuDanny/battle-city-rl-scratch.git \
    /workspace/battle-city-rl
cd /workspace/battle-city-rl

# 2. 一键配置环境
bash scripts/setup_runpod.sh

# 3. 激活环境
conda activate tank-rl-runpod
```

### 最小化训练验证

```bash
# 本地 Mac (MPS)
python scripts/train_tir.py --device mps --total-steps 10240 --run-name test

# RunPod (CUDA)
python scripts/train_tir.py --device cuda --total-steps 10240 --run-name test
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
│   ├── eval.py                 # 评估脚本
│   └── setup_runpod.sh         # RunPod 一键环境配置
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
# 基础训练 (Mac MPS)
python scripts/train_tir.py --device mps --run-name exp1

# 基础训练 (RunPod CUDA)
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
```

### 环境参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-steps` | `10000000` | 总训练步数 |
| `--lr` | `1e-4` | Adam 学习率 |
| `--seed` | `42` | 随机种子 |
| `--device` | `auto` | 训练设备 (`auto`/`cuda`/`mps`/`cpu`) |
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

# 实时窗口观看 agent 操作 (本地 Mac 有屏幕时可用)
python scripts/eval.py --ckpt checkpoints/2026-06-18_exp1/checkpoint_1000000.pt --render

# 录制视频保存到 videos/ 目录 (本地 Mac / RunPod headless 均可用)
python scripts/eval.py --ckpt checkpoints/2026-06-18_exp1/checkpoint_1000000.pt --record

# 确定性策略 + 加速模拟 (--fps 越大越快)
python scripts/eval.py --ckpt checkpoints/2026-06-18_exp1/checkpoint_1000000.pt --deterministic --fps 240
```

> **注意**: `--render` 需要图形界面，**RunPod 上不可用**。在 RunPod 上请使用 `--record` 生成视频后下载观看。

---

## RunPod 云端训练 (RTX 4090 24GB)

> 完整的 RunPod 从零到训练的操作指南。项目代码 (`scripts/setup_runpod.sh`) 已包含一键环境配置脚本。

### 第一步: 注册并充值

1. 访问 [runpod.io](https://www.runpod.io) 注册账号
2. Console → Billing → 充值 **$50-100** (4090 约 $0.34/h, 5M 步预计 $25-35)
3. Settings → API Keys → 生成一个 Key (后续可能需要)

### 第二步: 创建 Network Volume

Network Volume 用于持久化保存 checkpoint——Pod 停止后数据不丢：

1. Console → Storage → **Create Network Volume**
2. 名称: `battle-city-checkpoints`
3. 大小: **20 GB** (checkpoint ~500MB/个, 足够存几十个)
4. 数据中心: **选择一个区域** (记下, 下一步 Pod 必须选同一区域)

### 第三步: 创建 GPU Pod

1. Console → Pods → **Deploy GPU Pod**
2. 选择配置:
   - GPU: **RTX 4090** ($0.34/h, Community Cloud)
   - 或 RTX 3090 ($0.22/h, 同样 24GB VRAM 也可)
3. 模板: 选 **RunPod PyTorch** (预装 CUDA 12.4 + conda)
4. 高级设置 → **Attach Network Volume**: 选择 `battle-city-checkpoints`, 挂载路径 `/workspace`
5. **Deploy** (约 1-2 分钟启动)

### 第四步: 配置训练环境

Pod 就绪后, 通过 **Web Terminal** 或 SSH 登录:

```bash
# 克隆项目到 Pod (mac 分支, 已包含 RunPod 适配)
git clone --recurse-submodules -b mac \
    https://github.com/JYuDanny/battle-city-rl-scratch.git \
    /workspace/battle-city-rl
cd /workspace/battle-city-rl

# 一键配置 (创建环境 + 安装依赖 + 设置持久化符号链接)
bash scripts/setup_runpod.sh
```

脚本会自动:
- 检查 CUDA / GPU 可用性
- 初始化 git 子模块
- 创建 `tank-rl-runpod` conda 环境
- 安装 PyTorch CUDA 版 + 所有依赖
- 检测 Network Volume 并创建符号链接 (`checkpoints/` `runs/` → 持久化目录)
- 运行 pytest 验证

### 第五步: 启动训练

训练命令与本地完全一致，只需指定 `--device cuda`:

```bash
conda activate tank-rl-runpod

# 最小化验证 (10240 步, ~15 分钟)
python scripts/train_tir.py --total-steps 10240 --device cuda --run-name test

# 正式训练 (5M 步, 推荐 nohup 后台运行)
nohup python scripts/train_tir.py --device cuda --run-name runpod_exp1 \
    > train.log 2>&1 &

# 查看实时日志
tail -f train.log

# 查看 GPU 使用率
watch -n 2 nvidia-smi
```

### 第六步: 训练中评估

训练不中断的情况下, 另开一个 Web Terminal 或 SSH 连接:

```bash
conda activate tank-rl-runpod
cd /workspace/battle-city-rl

# 纯统计 (最快)
python scripts/eval.py --ckpt checkpoints/2026-06-18_runpod_exp1/checkpoint_1000000.pt --episodes 10

# 录制视频后在本地观看 (--render 在 RunPod 上不可用)
python scripts/eval.py --ckpt checkpoints/2026-06-18_runpod_exp1/checkpoint_1000000.pt --record
```

> 评估视频保存在 `videos/` 目录, 需下载到本地播放。

### 第七步: 查看 TensorBoard

**方式一**: 下载 `runs/` 目录到本地查看

```bash
# 在 RunPod 上打包
tar -czf runs.tar.gz runs/

# 本地终端下载 (替换为你的 Pod ID 和文件路径)
scp root@<pod-ip>:/workspace/battle-city-rl/runs ./

# 本地启动 TensorBoard
tensorboard --logdir runs
```

**方式二**: 在 RunPod 上启动 TensorBoard, 通过 RunPod Proxy 访问

```bash
# RunPod 上启动
conda activate tank-rl-runpod
tensorboard --logdir runs --host 0.0.0.0 --port 6006

# 在 RunPod UI → Pod → Connect → HTTP Port 6006 即可访问
```

### 第八步: 训练完成后

```bash
# 1. 检查 checkpoint (已自动持久化到 Network Volume)
ls checkpoints/

# 2. 如需将最新 checkpoint 下载到本地
#   (本地终端执行)
scp root@<pod-ip>:/workspace/battle-city-rl/checkpoints/2026-06-18_runpod_exp1/checkpoint_5000000.pt ./

# 3. 停止 Pod (节约费用)
#    RunPod UI → Pods → Stop

# 4. Network Volume 中的数据不会丢失
#    下次创建新 Pod 挂载同一 Volume 即可恢复训练:
#    python scripts/train_tir.py --device cuda \
#        --ckpt checkpoints/2026-06-18_runpod_exp1/checkpoint_5000000.pt
```

### 费用估算

| 训练规模 | 预计耗时 | 4090 费用 | 便捷命令 |
|----------|---------|----------|---------|
| 10K 步 (验证) | ~15 分钟 | ~$0.10 | `--total-steps 10240 --run-name test` |
| 1M 步 | ~6-10 小时 | ~$2-4 | `--total-steps 1000000 --run-name exp_1m` |
| 5M 步 | ~1.5-2.5 天 | ~$15-25 | `--total-steps 5000000 --run-name exp_5m` |
| 10M 步 | ~3-5 天 | ~$30-50 | 默认值, 无需指定 |

> **省钱提示**: 不用时记得 Stop Pod。Network Volume 单独计费 ($0.07/GB/月 ≈ $1.4/月 for 20GB)。

---

## 监控

### TensorBoard

训练过程中另开终端：

```bash
# 本地 Mac
conda activate tank-rl-mac
tensorboard --logdir runs                    # 查看全部训练
tensorboard --logdir runs/2026-06-18_exp1    # 只看某次运行

# RunPod 上 (见上方「查看 TensorBoard」章节)
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
- [x] Mac MPS / RunPod CUDA 双平台适配
- [ ] 正式 PPO 训练
- [ ] 并行采样多环境训练
- [ ] 模型发布

---

## License

MIT
