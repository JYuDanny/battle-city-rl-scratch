# Battle City RL Training Framework - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零搭建完整的 PPO 强化学习训练框架，训练 Battle City 一命通关智能体。

**Architecture:** 三层分离 — envs/ 纯环境层、rl/ 纯算法层、training/ 编排层。先实现简化 13×13 tile 环境快速验证 PPO，后期对接 tirinox 真实游戏。

**Tech Stack:** Python 3.11, PyTorch, Gymnasium, pytest, Conda env `tank-rl-teach`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `envs/tile_env.py` | 13×13 tile-based Gymnasium Env, 回合制步进、碰撞、reward 集成 |
| `envs/tir_env.py` | tirinox 真实游戏 Gymnasium 包装器（后期） |
| `envs/wrappers.py` | RecordVideo / 通用包装器 |
| `rl/config.py` | 统一超参数 dataclass |
| `rl/network.py` | Shared backbone + Actor/Critic heads |
| `rl/buffer.py` | Experience buffer, GAE 计算 |
| `rl/ppo.py` | PPO Trainer: rollout → advantage → clip → update |
| `training/reward_shaper.py` | 分层+势能 reward 计算 |
| `training/curriculum.py` | tirinox 关卡自动分组 + 阶段管理 |
| `training/trainer.py` | 主训练循环：env 交互 → buffer → PPO update → log |
| `scripts/train_tile.py` | 简化环境训练入口 |
| `scripts/train_tir.py` | 真实环境训练入口（后期） |
| `scripts/eval.py` | 评估/录制视频 |

---

### Task 1: 项目初始化与环境搭建

**Files:**
- Modify: `objectives.md` (already done)
- Create: `AGENTS.md`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `setup.py`

- [ ] **Step 1: 创建 Conda 环境并初始化 Git**

```bash
conda create -n tank-rl-teach python=3.11 -y
conda activate tank-rl-teach
git init
git add objectives.md
git commit -m "init: add project objectives"
```

- [ ] **Step 2: 创建 .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.cache/
checkpoints/
runs/
crash_log/
*.mp4
.venv/
```

- [ ] **Step 3: 创建 requirements.txt**

```
torch>=2.0.0
gymnasium>=0.29.0
numpy>=1.24.0
tensorboard>=2.13.0
matplotlib>=3.7.0
pytest>=7.4.0
pygame>=2.5.0
moviepy>=1.0.3
```

- [ ] **Step 4: 创建 setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="battle_city_rl",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "tensorboard>=2.13.0",
        "matplotlib>=3.7.0",
        "pygame>=2.5.0",
        "moviepy>=1.0.3",
    ],
)
```

- [ ] **Step 5: 安装依赖并验证**

```bash
pip install -e .
pytest --version
python -c "import torch; print(torch.__version__)"
python -c "import gymnasium; print(gymnasium.__version__)"
```

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements.txt setup.py
git commit -m "init: project scaffold with conda env configuration"
```

---

### Task 2: 统一超参数配置

**Files:**
- Create: `rl/__init__.py`
- Create: `rl/config.py`

- [ ] **Step 1: 创建 rl/__init__.py**

```python
# rl module
```

- [ ] **Step 2: 创建 rl/config.py**

```python
"""PPO 及训练统一超参数配置

集中管理所有超参数，避免散布在文件各处的魔数。
支持命令行覆写，方便调参实验。
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EnvConfig:
    """环境相关配置
    
    tile_size: 地图网格尺寸 (不含边界)
    action_size: 动作空间大小 [上, 下, 左, 右, 射击, 静止]
    max_steps: 单局最大步数限制, 防止无限循环
    """
    tile_size: Tuple[int, int] = (13, 13)
    action_size: int = 6
    max_steps: int = 2000


@dataclass
class PPOConfig:
    """PPO 算法超参数
    
    rollout_steps: 每次收集的交互步数, 累积后执行一次更新
    gamma: 折扣因子, 控制远期奖励的衰减
    gae_lambda: GAE 中的 λ 参数, 权衡偏差与方差
    clip_epsilon: PPO clip 范围, 限制策略更新的幅度
    k_epochs: 每批 rollout 数据重用的更新轮数
    mini_batch_size: 小批次梯度更新的样本数
    lr: 学习率
    entropy_coef: 熵正则化系数, 鼓励探索
    value_coef: 价值损失权重
    max_grad_norm: 梯度裁剪阈值, 防止梯度爆炸
    """
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    k_epochs: int = 10
    mini_batch_size: int = 64
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class NetworkConfig:
    """网络架构超参数
    
    tile_embedding_dim: tile 嵌入层输出维度
    share_mlp_sizes: 共享 MLP 隐藏层维度列表
    """
    tile_embedding_dim: int = 128
    share_mlp_sizes: Tuple[int, ...] = (256, 256)


@dataclass
class RewardConfig:
    """奖励函数权重
    
    分层奖励体系中各事件的奖励值。
    势能奖励基于到最近敌人的距离变化。
    """
    survival: float = 0.01
    kill_enemy: float = 2.0
    wave_clear: float = 5.0
    level_clear: float = 100.0
    death: float = -10.0
    base_destroyed: float = -50.0
    potential_scale: float = 0.01


@dataclass
class CurriculumConfig:
    """课程学习配置
    
    evaluation_interval: 每隔多少 episode 评估通关率
    evaluation_window: 计算通关率的滑动窗口大小
    promotion_threshold_stage1: S1→S2 通关率阈值
    promotion_threshold_stage2: S2→S3 通关率阈值
    demotion_threshold: 回退触发通关率阈值
    demotion_window: 回退观察窗口
    """
    evaluation_interval: int = 50
    evaluation_window: int = 100
    promotion_threshold_stage1: float = 0.8
    promotion_threshold_stage2: float = 0.5
    demotion_threshold: float = 0.2
    demotion_window: int = 50


@dataclass
class TrainingConfig:
    """训练流程配置
    
    total_timesteps: 总训练步数
    checkpoint_interval: 每隔多少 timestep 保存 checkpoint
    log_interval: 每隔多少 timestep 记录 TensorBoard
    save_dir: checkpoint 存储目录
    log_dir: TensorBoard 日志目录
    device: 训练设备 (auto 表示自动选择 cuda/cpu)
    seed: 随机种子, 确保可复现
    """
    total_timesteps: int = 10_000_000
    checkpoint_interval: int = 100_000
    log_interval: int = 1000
    save_dir: str = "checkpoints"
    log_dir: str = "runs"
    device: str = "auto"
    seed: int = 42


@dataclass
class Config:
    """总配置容器
    
    聚合所有子模块配置, 提供统一的参数访问入口。
    """
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def update_from_dict(self, overrides: dict):
        """根据字典覆写配置项, 支持点分隔键如 'ppo.lr=1e-4'"""
        for key, value in overrides.items():
            parts = key.split(".")
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
```

- [ ] **Step 3: 验证模块可导入**

```bash
python -c "from rl.config import Config; c = Config(); print(c.ppo.lr)"
```

- [ ] **Step 4: Commit**

```bash
git add rl/
git commit -m "feat: add unified hyperparameter config dataclasses"
```

---

### Task 3: 共享网络模型

**Files:**
- Create: `rl/network.py`
- Create: `tests/test_network.py`

- [ ] **Step 1: 编写网络测试**

在 `tests/test_network.py` 中:

```python
"""网络模块单元测试

验证输入输出维度正确性, 确保网络在各种配置下能正常前向传播。
"""

import torch
import pytest
from rl.config import Config, EnvConfig, NetworkConfig
from rl.network import ActorCritic


class TestActorCritic:
    """ActorCritic 网络测试套件"""

    def test_output_shapes(self):
        """验证输出维度: action_logits [batch, 6], value [batch, 1]"""
        config = Config()
        net = ActorCritic(config)
        batch_size = 32

        obs = torch.randn(batch_size, 13, 13, 9)  # 9 通道: 地形+实体+方向
        logits, value = net(obs)

        assert logits.shape == (batch_size, config.env.action_size), \
            f"Expected logits shape {(batch_size, config.env.action_size)}, got {logits.shape}"
        assert value.shape == (batch_size, 1), \
            f"Expected value shape {(batch_size, 1)}, got {value.shape}"

    def test_value_range_reasonable(self):
        """验证 value 输出在合理范围内 (未经训练, 应接近 0)"""
        config = Config()
        net = ActorCritic(config)
        obs = torch.randn(16, 13, 13, 9)

        _, value = net(obs)

        assert value.abs().max() < 100, f"Value output too large: {value.abs().max()}"

    def test_logits_produce_valid_probs(self):
        """验证 logits 经过 softmax 后得到有效的概率分布"""
        config = Config()
        net = ActorCritic(config)
        obs = torch.randn(8, 13, 13, 9)

        logits, _ = net(obs)
        probs = torch.softmax(logits, dim=-1)

        for i in range(probs.shape[0]):
            assert abs(probs[i].sum().item() - 1.0) < 1e-5, \
                f"Probabilities don't sum to 1 for sample {i}"
            assert (probs[i] >= 0).all(), \
                f"Negative probability found for sample {i}"

    def test_deterministic_with_seed(self):
        """验证相同输入和 seed 下输出确定性"""
        torch.manual_seed(42)
        config = Config()
        net1 = ActorCritic(config)

        torch.manual_seed(42)
        net2 = ActorCritic(config)

        obs = torch.randn(4, 13, 13, 9)
        logits1, value1 = net1(obs)
        logits2, value2 = net2(obs)

        assert torch.allclose(logits1, logits2), "Logits not deterministic"
        assert torch.allclose(value1, value2), "Value not deterministic"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_network.py -v
```

Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: 实现 rl/network.py**

```python
"""Actor-Critic 网络模型

采用共享 backbone 架构:
  Observation [B, 13, 13, C] → Flatten → Embedding → ShareMLP
                                  ├── Actor Head → action_distribution
                                  └── Critic Head → state_value

后续升级到像素输入时, ShareMLP 替换为 Conv2D backbone。
"""

import torch
import torch.nn as nn
import numpy as np
from rl.config import Config


class ActorCritic(nn.Module):
    """Actor-Critic 共享网络

    共享底层特征提取, 分叉出策略头(Actor)和价值头(Critic)。
    这种设计减少了参数总量, 且鼓励两个任务学到互补的表征。

    输入: 环境观察 [batch_size, tile_h, tile_w, channels]
    输出: (action_logits [batch_size, action_size], state_value [batch_size, 1])
    """

    def __init__(self, config: Config):
        """初始化网络

        Args:
            config: 包含 network 和 env 模块的全局配置
        """
        super().__init__()

        env_cfg = config.env
        net_cfg = config.network

        # 计算展平后的输入维度: tile_H × tile_W × channels
        obs_channels = 9  # 地形(5) + 实体(3) + 方向(1)
        input_dim = env_cfg.tile_size[0] * env_cfg.tile_size[1] * obs_channels

        # 将 tile grid 展平后映射到嵌入空间
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, net_cfg.tile_embedding_dim),
            nn.ReLU(),
        )

        # 共享 MLP backbone
        share_layers = []
        prev_dim = net_cfg.tile_embedding_dim
        for hidden_dim in net_cfg.share_mlp_sizes:
            share_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        self.share_mlp = nn.Sequential(*share_layers)

        # Actor Head: 输出每个动作的 logit
        self.actor_head = nn.Linear(prev_dim, env_cfg.action_size)

        # Critic Head: 输出状态价值标量
        self.critic_head = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """正交初始化权重, 偏置初始化为 0

        正交初始化有助于保持梯度在前向/反向传播中的稳定性。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            obs: 环境观察张量 [batch_size, H, W, C] 浮点型

        Returns:
            action_logits: [batch_size, action_size] 各动作的未归一化 logits
            state_value: [batch_size, 1] 状态价值估计
        """
        embedded = self.embedding(obs)
        shared = self.share_mlp(embedded)
        action_logits = self.actor_head(shared)
        state_value = self.critic_head(shared)
        return action_logits, state_value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """根据观察采样动作

        Args:
            obs: 单个观察 [1, H, W, C] 或批次
            deterministic: True 时取 argmax, False 时按分布采样

        Returns:
            action: 选中的动作索引
            log_prob: 该动作的对数概率
            value: 状态价值估计
        """
        logits, value = self.forward(obs)
        probs = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        return action, log_prob, value
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_network.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add rl/network.py tests/test_network.py
git commit -m "feat: add shared Actor-Critic network with tile embedding"
```

---

### Task 4: GAE Experience Buffer

**Files:**
- Create: `rl/buffer.py`
- Create: `tests/test_buffer.py`

- [ ] **Step 1: 编写 buffer 测试**

在 `tests/test_buffer.py` 中:

```python
"""Experience Buffer 单元测试

验证 GAE 计算数值正确性 (小批量手动对照公式)。
"""

import torch
import numpy as np
from rl.config import Config
from rl.buffer import RolloutBuffer


class TestRolloutBuffer:
    """RolloutBuffer 测试套件"""

    def test_store_and_compute_gae_shape(self):
        """验证 GAE 计算后 advantages 和 returns 维度正确"""
        config = Config()
        buffer = RolloutBuffer(config)

        buffer.reset()
        for _ in range(config.ppo.rollout_steps):
            buffer.store(
                obs=torch.randn(13, 13, 9),
                action=torch.tensor(2),
                reward=0.01,
                value=torch.tensor([0.0]),
                log_prob=torch.tensor(-1.79),
                done=False,
            )

        last_value = torch.tensor([0.0])
        buffer.compute_gae(last_value)

        assert buffer.advantages.shape[0] == config.ppo.rollout_steps, \
            f"Expected {config.ppo.rollout_steps} advantages, got {buffer.advantages.shape[0]}"
        assert buffer.returns.shape[0] == config.ppo.rollout_steps, \
            f"Expected {config.ppo.rollout_steps} returns, got {buffer.returns.shape[0]}"

    def test_gae_simple_case(self):
        """手动验证简单场景下的 GAE 数值

        场景: 3 步轨迹, γ=0.99, λ=0.95
        奖励: [1.0, 0.0, 0.0], 终止: [False, False, True]
        价值: [0.0, 0.0, 0.0], last_value=0.0

        手动计算:
          δ₀ = 1.0 + 0.99×0.0 - 0.0 = 1.0
          δ₁ = 0.0 + 0.99×0.0 - 0.0 = 0.0
          δ₂ = 0.0 + 0.99×0.0 - 0.0 = 0.0

          A₀ = δ₀ + (γλ)¹ A₁ = 1.0 + 0.9405×0.0 = 1.0
          A₁ = δ₁ + (γλ)¹ A₂ = 0.0 + 0.9405×0.0 = 0.0
          A₂ = δ₂ = 0.0

          回报: G₀ = A₀ + V₀ = 1.0 + 0.0 = 1.0
               G₁ = A₁ + V₁ = 0.0 + 0.0 = 0.0
               G₂ = A₂ + V₂ = 0.0 + 0.0 = 0.0
        """
        # 手动覆盖 PPO 参数便于小批量测试
        config = Config()
        config.ppo.rollout_steps = 3
        config.ppo.gamma = 0.99
        config.ppo.gae_lambda = 0.95

        buffer = RolloutBuffer(config)
        buffer.reset()

        rewards = [1.0, 0.0, 0.0]
        dones = [False, False, True]

        for i in range(3):
            buffer.store(
                obs=torch.randn(13, 13, 9),
                action=torch.tensor(2),
                reward=rewards[i],
                value=torch.tensor([0.0]),
                log_prob=torch.tensor(-1.0),
                done=dones[i],
            )

        last_value = torch.tensor([0.0])
        buffer.compute_gae(last_value)

        expected_advantages = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        expected_returns = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        assert np.allclose(buffer.advantages.numpy(), expected_advantages, atol=1e-5), \
            f"Advantages differ: got {buffer.advantages.numpy()}, expected {expected_advantages}"
        assert np.allclose(buffer.returns.numpy(), expected_returns, atol=1e-5), \
            f"Returns differ: got {buffer.returns.numpy()}, expected {expected_returns}"

    def test_advantage_normalization(self):
        """验证 advantages 归一化后均值≈0, 标准差≈1"""
        config = Config()
        buffer = RolloutBuffer(config)

        buffer.reset()
        for i in range(config.ppo.rollout_steps):
            buffer.store(
                obs=torch.randn(13, 13, 9),
                action=torch.tensor(i % 6),
                reward=float(i % 3) - 1.0,  # 产生 -1, 0, 1 的奖励
                value=torch.tensor([0.0]),
                log_prob=torch.tensor(-1.0),
                done=(i == config.ppo.rollout_steps - 1),
            )

        last_value = torch.tensor([0.0])
        buffer.compute_gae(last_value)

        mean = buffer.advantages.mean().item()
        std = buffer.advantages.std().item()

        assert abs(mean) < 0.1, f"Normalized advantage mean not near 0: {mean}"
        assert abs(std - 1.0) < 0.2, f"Normalized advantage std not near 1: {std}"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_buffer.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: 实现 rl/buffer.py**

```python
"""Experience Buffer (经验回放缓冲)

存储 rollout 轨迹数据, 并在收集完成后通过 GAE 计算 Advantage 估计。

GAE (Generalized Advantage Estimation):
  - 在 TD-residual 上执行指数加权移动平均
  - 通过 λ 参数控制偏差-方差权衡
  - λ=0 → TD(0), 高偏差低方差
  - λ=1 → Monte Carlo, 低偏差高方差
"""

import torch
import numpy as np
from rl.config import Config


class RolloutBuffer:
    """Rollout 缓冲区

    在每轮 rollout 收集中存储 (obs, action, reward, value, log_prob, done),
    收集完成后调用 compute_gae 计算 advantages 和 returns。

    PPO 更新期间可多次全批次或 mini-batch 采样。
    """

    def __init__(self, config: Config):
        """初始化空缓冲区

        Args:
            config: 全局配置, 其中 ppo.rollout_steps 决定容量
        """
        ppo_cfg = config.ppo
        self.gamma = ppo_cfg.gamma
        self.gae_lambda = ppo_cfg.gae_lambda
        self.rollout_steps = ppo_cfg.rollout_steps

        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        self.advantages = None
        self.returns = None

    def reset(self):
        """清空缓冲区, 准备新一轮 rollout 收集"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None

    def store(self, obs: torch.Tensor, action: torch.Tensor, reward: float,
              value: torch.Tensor, log_prob: torch.Tensor, done: bool):
        """存入单步交互数据

        Args:
            obs: 环境观察 [*obs_shape]
            action: 执行的动作索引, 标量 tensor
            reward: 即时奖励, Python 标量
            value: 状态价值 V(s), [1] 或标量 tensor
            log_prob: 动作对数概率 log π(a|s), 标量 tensor
            done: 是否到达终止状态
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(self, last_value: torch.Tensor):
        """计算 GAE advantages 和 returns

        GAE 公式:
          δₜ = rₜ + γ·V(s_{t+1})·(1 - done_t) - V(sₜ)
          Aₜ = δₜ + (γ·λ) · A_{t+1}

        回报 (returns) = Aₜ + V(sₜ), 作为 Value Head 的训练目标。

        advantages 计算后会进行 z-score 归一化 (均值 0, 标准差 1),
        稳定训练并减少策略梯度的噪声。

        Args:
            last_value: 最后状态的 V(s_T), 用于计算最后一个 δ
        """
        # 展开为 tensor 便于向量化计算
        values = torch.stack([v.squeeze() for v in self.values])
        adv = torch.zeros(self.rollout_steps, dtype=torch.float32)

        gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_val = last_value.squeeze()
            else:
                next_val = values[t + 1]

            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + self.gamma * next_val * mask - values[t].item()
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            adv[t] = gae

        # z-score 归一化
        adv_mean = adv.mean()
        adv_std = adv.std() + 1e-8
        self.advantages = (adv - adv_mean) / adv_std
        self.returns = (adv + values).detach()

    def get_batch(self, indices: list[int]):
        """获取指定索引的 mini-batch 数据

        Args:
            indices: 批次内样本索引列表

        Returns:
            obs: [batch, *obs_shape]
            actions: [batch]
            log_probs_old: [batch]
            advantages: [batch]
            returns: [batch]
        """
        obs = torch.stack([self.observations[i] for i in indices])
        actions = torch.tensor([self.actions[i] for i in indices], dtype=torch.long)
        log_probs_old = torch.tensor([self.log_probs[i] for i in indices], dtype=torch.float32)
        adv = self.advantages[indices]
        ret = self.returns[indices]

        return obs, actions, log_probs_old, adv, ret
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_buffer.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add rl/buffer.py tests/test_buffer.py
git commit -m "feat: add experience buffer with GAE advantage computation"
```

---

### Task 5: PPO Trainer

**Files:**
- Create: `rl/ppo.py`
- Create: `tests/test_ppo.py`

- [ ] **Step 1: 编写 PPO 测试**

在 `tests/test_ppo.py` 中:

```python
"""PPO Trainer 单元测试

验证 loss 下降趋势和 clip 机制生效。
"""

import torch
import numpy as np
from rl.config import Config
from rl.network import ActorCritic
from rl.buffer import RolloutBuffer
from rl.ppo import PPOTrainer


class TestPPOTrainer:
    """PPO Trainer 测试套件"""

    @staticmethod
    def _create_mock_rollout(trainer, config, num_good=500, num_bad=500):
        """创建模拟 rollout 数据: 一半是"好"轨迹, 一半是"坏"轨迹

        好轨迹: 正向奖励, 高价值 → 训练后 loss 应下降
        坏轨迹: 负向奖励, 低价值 → 用于让 Critic 区分好坏状态
        """
        buffer = RolloutBuffer(config)
        buffer.reset()

        for i in range(config.ppo.rollout_steps):
            if i < num_good:
                obs = torch.ones(13, 13, 9) * 0.5  # 好状态特征
                reward = 1.0
                value = torch.tensor([5.0])
            else:
                obs = torch.ones(13, 13, 9) * (-0.5)  # 坏状态特征
                reward = -1.0
                value = torch.tensor([-5.0])

            action = torch.tensor(i % 6)
            log_prob = torch.tensor(-1.79)
            done = (i == config.ppo.rollout_steps - 1)
            buffer.store(obs, action, reward, value, log_prob, done)

        last_value = torch.tensor([0.0])
        buffer.compute_gae(last_value)
        return buffer

    def test_loss_decreases_after_update(self):
        """验证经过一次 PPO update 后, loss 应该下降"""
        config = Config()
        config.ppo.rollout_steps = 128
        config.ppo.k_epochs = 1

        net = ActorCritic(config)
        trainer = PPOTrainer(config)

        buffer = self._create_mock_rollout(trainer, config)

        # 记录更新前的 loss
        obs, actions, old_logp, adv, ret = buffer.get_batch(
            list(range(config.ppo.rollout_steps))
        )
        with torch.no_grad():
            logits, values = net(obs)
            _, loss_before = trainer.compute_loss(
                logits, values, actions, old_logp, adv, ret
            )

        # 执行更新
        loss_after = trainer.update(net, buffer)

        assert loss_after < loss_before.item(), \
            f"Loss did not decrease: before={loss_before.item():.4f}, after={loss_after:.4f}"

    def test_clip_prevents_large_policy_change(self):
        """验证 clip 机制限制了策略比率变化在 [1-ε, 1+ε] 范围内"""
        config = Config()
        config.ppo.clip_epsilon = 0.2
        net = ActorCritic(config)
        trainer = PPOTrainer(config)

        # 手动构造: 让 old_logp 和 new_logp 差距很大, 检验 clip 是否生效
        obs = torch.randn(8, 13, 13, 9)
        logits, _ = net(obs)
        probs = torch.softmax(logits, dim=-1)

        old_logp = torch.log(probs + 1e-8).detach()
        old_logp[0, 0] = -10.0  # 极端旧概率

        adv = torch.randn(8)
        actions = torch.zeros(8, dtype=torch.long)

        _, loss = trainer.compute_loss(logits, None, actions, old_logp, adv, None)

        # 如果 clip 没生效, 极端比率会导致巨大 loss
        assert not torch.isnan(loss), "Loss is NaN — clip may not be working"
        assert loss.item() < 100.0, f"Loss too large: {loss.item()}"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_ppo.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: 实现 rl/ppo.py**

```python
"""PPO (Proximal Policy Optimization) Trainer

实现 PPO-Clip 算法:
  1. 与环境交互收集 rollout 轨迹
  2. GAE 计算 advantages
  3. 在 clip 约束下更新策略和价值函数

PPO 核心思想:
  - 限制每次策略更新的幅度, 防止破坏性的大步更新
  - 通过 clip 机制, 当新旧策略比率超出 [1-ε, 1+ε] 时截断 loss
  - 同时优化 Actor loss (clip), Critic loss (MSE) 和 Entropy bonus(鼓励探索)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rl.config import Config
from rl.buffer import RolloutBuffer


class PPOTrainer:
    """PPO 训练器

    管理策略网络的优化过程, 包括:
      - compute_loss: 计算 PPO clip loss + value loss + entropy bonus
      - update: 完整的一次 PPO 更新 (K epochs × mini-batches)
    """

    def __init__(self, config: Config):
        """初始化训练器

        Args:
            config: 全局配置, 包含 ppo 超参数和 training 设置
        """
        ppo_cfg = config.ppo
        self.clip_epsilon = ppo_cfg.clip_epsilon
        self.value_coef = ppo_cfg.value_coef
        self.entropy_coef = ppo_cfg.entropy_coef
        self.max_grad_norm = ppo_cfg.max_grad_norm
        self.k_epochs = ppo_cfg.k_epochs
        self.mini_batch_size = ppo_cfg.mini_batch_size
        self.rollout_steps = ppo_cfg.rollout_steps
        self.lr = ppo_cfg.lr
        self.device = config.training.device

        self.optimizer = None  # 在第一次 update 时初始化
        self._optimizer_created = False

    def _ensure_optimizer(self, net: ActorCritic):
        """延迟创建优化器, 绑定到网络参数"""
        if not self._optimizer_created:
            self.optimizer = optim.Adam(net.parameters(), lr=self.lr)
            self._optimizer_created = True

    def compute_loss(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple:
        """计算 PPO 组合 loss

        Policy Loss (CLIP):
          ratio = exp(new_logp - old_logp)
          surr1 = ratio × A
          surr2 = clip(ratio, 1-ε, 1+ε) × A
          L_policy = -min(surr1, surr2)

        Value Loss:
          L_value = (V(s) - G)²   (MSE between prediction and GAE return)

        Entropy Bonus:
          L_entropy = -H(π)  鼓励策略保持一定的随机性

        总 Loss: L = L_policy + value_coef × L_value + entropy_coef × L_entropy

        Args:
            logits: [batch, action_size] 当前网络输出的动作 logits
            values: [batch, 1] 当前 Critic 输出的状态价值
            actions: [batch] 实际执行的动作索引
            old_log_probs: [batch] rollout 时的旧策略对数概率
            advantages: [batch] GAE 优势估计
            returns: [batch] GAE 回报 (value training target)

        Returns:
            loss: 标量 loss
            debug_info: (policy_loss, value_loss, entropy, clip_fraction) 供日志用
        """
        # Critic loss (MSE)
        value_loss = nn.functional.mse_loss(values.squeeze(), returns)

        # 当前策略的对数概率
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # PPO-Clip 的核心: 限制策略比率变化
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 统计 clip 发生的比例 (用于监控训练稳定性)
        clip_fraction = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()

        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * (-entropy)

        return loss, (policy_loss.detach(), value_loss.detach(), entropy.detach(), clip_fraction)

    def update(self, net: ActorCritic, buffer: RolloutBuffer):
        """执行一轮 PPO 更新

        将 buffer 中的 rollout 数据分成 mini-batch,
        在 K 个 epoch 中反复训练, 每次使用全部数据但顺序随机。

        Args:
            net: 当前策略网络
            buffer: 已填充并计算好 GAE 的 rollout 缓冲区

        Returns:
            average_loss: 本轮更新的平均 loss
        """
        self._ensure_optimizer(net)
        total_loss = 0.0
        num_batches = 0

        all_indices = list(range(self.rollout_steps))
        num_mini_batches = max(1, self.rollout_steps // self.mini_batch_size)

        for epoch in range(self.k_epochs):
            np.random.shuffle(all_indices)

            for batch_start in range(0, self.rollout_steps, self.mini_batch_size):
                batch_indices = all_indices[batch_start:batch_start + self.mini_batch_size]
                obs, actions, old_logp, adv, ret = buffer.get_batch(batch_indices)

                logits, values = net(obs)
                loss, _ = self.compute_loss(logits, values, actions, old_logp, adv, ret)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)
```

注意: `compute_loss` 的 `ActorCritic` 类型引用来自 network 模块, 这里使用延迟导入避免循环依赖 —— 实际实现时通过传参而非类型标注。

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_ppo.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add rl/ppo.py tests/test_ppo.py
git commit -m "feat: add PPO trainer with clip loss and entropy regularization"
```

---

### Task 6: 简化 Tile 环境

**Files:**
- Create: `envs/__init__.py`
- Create: `envs/tile_env.py`
- Create: `tests/test_tile_env.py`

- [ ] **Step 1: 编写环境测试**

在 `tests/test_tile_env.py` 中:

```python
"""TileEnv 环境单元测试

验证动作→状态转换、reward 计算、done 条件。
"""

import gymnasium
import numpy as np
from envs.tile_env import TileEnv


class TestTileEnv:
    """TileEnv 测试套件"""

    def test_reset_returns_valid_obs(self):
        """reset 后观察空间维度与定义一致"""
        env = TileEnv()
        obs, info = env.reset()

        assert obs.shape == (13, 13, 9), f"Expected (13,13,9), got {obs.shape}"
        assert isinstance(info, dict), f"Info should be dict, got {type(info)}"

    def test_step_returns_expected_tuple(self):
        """step 返回 (obs, reward, terminated, truncated, info)"""
        env = TileEnv()
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)

        assert obs.shape == (13, 13, 9)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space_is_discrete_6(self):
        """动作空间为 Discrete(6)"""
        env = TileEnv()
        assert env.action_space.n == 6

    def test_death_causes_termination(self):
        """玩家死亡导致 terminated=True"""
        env = TileEnv()
        env.reset()

        player = env.player_pos

        # 多点几次, 确保碰撞检测正常工作
        for _ in range(1000):
            if not env._player_alive:
                break
            env.step(np.random.randint(0, 6))

    def test_gymnasium_api_compliant(self):
        """验证完整 Gymnasium API 合规性 (check_env)"""
        from gymnasium.utils.env_checker import check_env

        env = TileEnv()
        check_env(env, skip_render_check=True)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_tile_env.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: 实现 envs/tile_env.py**

```python
"""简化 Tile-Based 坦克大战 Gymnasium 环境

13×13 网格地图, 回合制步进:
  1. 玩家执行动作 (6 选: ↑↓←→射击静止)
  2. 敌人执行随机策略
  3. 处理碰撞 (子弹 vs 实体, 玩家 vs 敌人等)
  4. 计算 reward
  5. 检查终止条件

地图编码: 13×13×[C 通道]
  Channel 0: 空地(1) / 砖墙(2) / 水(3) / 树(4) / 钢铁墙(5)
  Channel 1: 基地(0/1)
  Channel 2: 玩家x (归一化到[0,1])
  Channel 3: 玩家y
  Channel 4: 敌人x
  Channel 5: 敌人y
  Channel 6: 子弹x
  Channel 7: 子弹y
  Channel 8: 方向 (0=上 1=下 2=左 3=右)
"""

import gymnasium
from gymnasium import spaces
import numpy as np


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "SHOOT", "IDLE"]
directions = {
    "UP": np.array([0, -1]),
    "DOWN": np.array([0, 1]),
    "LEFT": np.array([-1, 0]),
    "RIGHT": np.array([1, 0]),
}


class TileEnv(gymnasium.Env):
    """简化 13×13 坦克大战环境

    回合制网格环境, 用于快速验证 PPO 训练流程。
    每步 agent 执行一个动作, 环境同步步进所有实体。

    Observation: 13×13×9 多通道张量
    Action: Discrete(6) [UP, DOWN, LEFT, RIGHT, SHOOT, IDLE]
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        """初始化环境

        Args:
            render_mode: 渲染模式, rgb_array 或 None
        """
        super().__init__()

        self.grid_size = 13
        self.render_mode = render_mode

        # 9 通道: [地形(5) + 实体(4)]
        # 地形: empty, brick, water, tree, steel
        # 实体: base, player, enemy, bullet
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(13, 13, 9),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(6)

        self._max_steps = 2000
        self._step_count = 0

        # 内部状态
        self._grid_terrain = None  # [13,13] 地形类型
        self._player_pos = None
        self._player_dir = None
        self._player_alive = True
        self._enemies = []  # list of (x, y, alive)
        self._bullets = []  # list of (x, y, dx, dy, owner)  owner: 'player'/'enemy'
        self._base_alive = True
        self._base_pos = (6, 12)  # 基地在底部中央
        self._wave = 0
        self._enemies_killed = 0
        self._total_enemies = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        """重置环境到初始状态

        Returns:
            observation: 初始观察 [13,13,9]
            info: 额外信息字典
        """
        super().reset(seed=seed)

        # 初始化地形: 周围一圈钢墙, 内部随机砖墙
        self._grid_terrain = np.zeros((13, 13), dtype=np.int32)
        self._grid_terrain[0, :] = 5  # 顶部钢墙
        self._grid_terrain[-1, :] = 5  # 底部钢墙
        self._grid_terrain[:, 0] = 5  # 左侧钢墙
        self._grid_terrain[:, -1] = 5  # 右侧钢墙

        # 内部随机放置砖墙 (约 20% 密度)
        rng = np.random.default_rng(seed)
        for y in range(1, 12):
            for x in range(1, 12):
                if (x, y) == self._base_pos or (x, y) == (6, 11):
                    continue  # 保留基地和入口
                if rng.random() < 0.2:
                    self._grid_terrain[y, x] = 2  # 砖墙

        # 基地
        self._base_alive = True

        # 玩家初始位置: 底部中央
        self._player_pos = np.array([6, 11], dtype=np.int32)
        self._player_dir = np.array([0, -1], dtype=np.int32)  # 朝上
        self._player_alive = True

        # 敌人生成 (第一波)
        self._wave = 1
        self._enemies = self._spawn_enemies()
        self._total_enemies = len(self._enemies)
        self._enemies_killed = 0

        self._bullets = []
        self._step_count = 0

        return self._get_obs(), self._get_info()

    def _spawn_enemies(self):
        """在上半区域生成敌人

        Returns:
            list of (x, y, alive)
        """
        rng = np.random.default_rng(self.np_random.integers(0, 2**31))
        enemies = []
        for _ in range(3):
            for attempt in range(100):
                x = rng.integers(1, 12)
                y = rng.integers(1, 5)
                if self._grid_terrain[y, x] == 0:
                    enemies.append([np.array([x, y]), np.array([0, 1]), True])
                    break
        return enemies

    def step(self, action: int):
        """执行一步环境交互

        处理流程:
        1. 玩家行动 (基于 action)
        2. 敌人行动 (随机策略)
        3. 子弹移动与碰撞检测
        4. 奖励计算
        5. 终止判断

        Args:
            action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=SHOOT, 5=IDLE

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1

        # 1. 玩家行动
        if self._player_alive and action < 4:
            self._player_dir = directions[ACTIONS[action]]
            new_pos = self._player_pos + self._player_dir
            if self._is_walkable(new_pos):
                self._player_pos = new_pos

        if self._player_alive and action == 4:  # SHOOT
            bullet_x = self._player_pos[0] + self._player_dir[0]
            bullet_y = self._player_pos[1] + self._player_dir[1]
            self._bullets.append([bullet_x, bullet_y,
                                  self._player_dir[0], self._player_dir[1], 'player'])

        # 2. 敌人行动 (随机)
        for enemy in self._enemies:
            if not enemy[2]:  # dead
                continue
            # 随机移动或射击
            act = self.np_random.integers(0, 6)
            if act < 4:
                dir_key = ACTIONS[act]
                new_pos = enemy[0] + directions[dir_key]
                if self._is_walkable(new_pos) and not np.array_equal(new_pos, self._player_pos):
                    enemy[0] = directions[dir_key] + enemy[0]
                    enemy[1] = directions[dir_key]
            elif act == 4:  # SHOOT
                bx = enemy[0][0] + enemy[1][0]
                by = enemy[0][1] + enemy[1][1]
                self._bullets.append([bx, by, enemy[1][0], enemy[1][1], 'enemy'])

        # 3. 子弹移动与碰撞
        reward_override = 0.0
        new_bullets = []
        for b in self._bullets:
            bx, by, dx, dy, owner = b
            nx, ny = bx + dx, by + dy

            # 超出边界 → 消失
            if nx < 0 or nx >= 13 or ny < 0 or ny >= 13:
                continue

            # 击中砖墙 → 破坏砖墙 + 子弹消失
            if self._grid_terrain[ny, nx] == 2:
                self._grid_terrain[ny, nx] = 0
                continue

            # 击中钢墙 → 子弹消失
            if self._grid_terrain[ny, nx] == 5:
                continue

            # 击中基地 → 基地被毁
            if (nx, ny) == self._base_pos and self._base_alive:
                self._base_alive = False
                continue

            # 玩家子弹 vs 敌人
            if owner == 'player':
                for enemy in self._enemies:
                    if not enemy[2]:
                        continue
                    if nx == enemy[0][0] and ny == enemy[0][1]:
                        enemy[2] = False  # 击杀敌人
                        self._enemies_killed += 1
                        reward_override += self._get_reward_event('kill')
                        break
                else:
                    new_bullets.append([nx, ny, dx, dy, owner])

            # 敌人子弹 vs 玩家
            elif owner == 'enemy':
                if self._player_alive and nx == self._player_pos[0] and ny == self._player_pos[1]:
                    self._player_alive = False
                    reward_override += self._get_reward_event('death')
                else:
                    new_bullets.append([nx, ny, dx, dy, owner])

        self._bullets = new_bullets

        # 4. 奖励计算
        reward = self._get_reward(reward_override)

        # 5. 终止判断
        terminated = not self._player_alive or not self._base_alive
        if len([e for e in self._enemies if e[2]]) == 0:
            terminated = True  # 清关
            reward += self._get_reward_event('wave_clear')

        truncated = self._step_count >= self._max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_reward_event(self, event: str) -> float:
        """获取事件对应的奖励值 (简化版, 无 config 注入时用硬编码)

        Args:
            event: 事件名 (kill, death, wave_clear, etc.)

        Returns:
            reward value
        """
        rewards = {
            'survival': 0.01,
            'kill': 2.0,
            'wave_clear': 5.0,
            'death': -10.0,
            'base_destroyed': -50.0,
        }
        return rewards.get(event, 0.0)

    def _get_reward(self, event_reward: float) -> float:
        """组合基础生存奖励和事件奖励"""
        survival = self._get_reward_event('survival')
        if not self._base_alive:
            survival += self._get_reward_event('base_destroyed')
        return survival + event_reward

    def _is_walkable(self, pos: np.ndarray) -> bool:
        """检查指定位置是否可通行

        Args:
            pos: [x, y] 坐标

        Returns:
            True if walkable (空地/树), False otherwise
        """
        x, y = int(pos[0]), int(pos[1])
        if x < 0 or x >= 13 or y < 0 or y >= 13:
            return False
        terrain = self._grid_terrain[y, x]
        return terrain == 0 or terrain == 4  # 空地或树可通行

    def _get_obs(self) -> np.ndarray:
        """构建多通道观察张量

        Channel layout (9 channels total):
          0: 空地(1) / 砖墙(0.5) / 水(0.3) / 树(0.7) / 钢墙(0.1)
          1: 基地位置 (1 if base else 0)
          2: 玩家 x (归一化)
          3: 玩家 y (归一化)
          4: 敌人 x (归一化, 叠加)
          5: 敌人 y (归一化, 叠加)
          6: 子弹 x (归一化)
          7: 子弹 y (归一化)
          8: 玩家方向 (0=UP, 0.33=DOWN, 0.67=LEFT, 1.0=RIGHT)

        Returns:
            obs: [13, 13, 9] float32 tensor
        """
        obs = np.zeros((13, 13, 9), dtype=np.float32)

        # Channel 0: 地形 (合并5类为1通道用不同值表示)
        terrain_map = {0: 1.0, 2: 0.5, 3: 0.3, 4: 0.7, 5: 0.1}
        for y in range(13):
            for x in range(13):
                t = self._grid_terrain[y, x]
                obs[y, x, 0] = terrain_map.get(t, 0.0)

        # Channel 1: 基地
        if self._base_alive:
            bx, by = self._base_pos
            obs[by, bx, 1] = 1.0

        # Channel 2-3: 玩家
        if self._player_alive:
            obs[self._player_pos[1], self._player_pos[0], 2] = self._player_pos[0] / 12.0
            obs[self._player_pos[1], self._player_pos[0], 3] = self._player_pos[1] / 12.0

        # Channel 4-5: 敌人 (叠加表示, 多敌人同格时取最大值)
        for enemy in self._enemies:
            if enemy[2]:  # alive
                x, y = enemy[0]
                obs[y, x, 4] = max(obs[y, x, 4], x / 12.0)
                obs[y, x, 5] = max(obs[y, x, 5], y / 12.0)

        # Channel 6-7: 子弹
        for b in self._bullets:
            bx, by = int(b[0]), int(b[1])
            if 0 <= bx < 13 and 0 <= by < 13:
                obs[by, bx, 6] = 1.0
                obs[by, bx, 7] = 1.0 if b[4] == 'player' else 0.5

        # Channel 8: 方向
        if self._player_alive:
            dir_map = {
                (0, -1): 0.0,   # UP
                (0, 1): 0.33,   # DOWN
                (-1, 0): 0.67,  # LEFT
                (1, 0): 1.0,    # RIGHT
            }
            py, px = self._player_pos[1], self._player_pos[0]
            dir_val = dir_map.get(tuple(self._player_dir), 0.5)
            obs[py, px, 8] = dir_val

        return obs

    def _get_info(self) -> dict:
        """返回环境诊断信息"""
        enemies_alive = sum(1 for e in self._enemies if e[2])
        return {
            'enemies_killed': self._enemies_killed,
            'enemies_total': self._total_enemies,
            'enemies_alive': enemies_alive,
            'base_alive': self._base_alive,
            'player_alive': self._player_alive,
            'wave': self._wave,
        }

    def render(self):
        """渲染当前帧 (ascii 文本模式)"""
        grid = [['.' for _ in range(13)] for _ in range(13)]

        # 地形
        terrain_chars = {0: '.', 2: '#', 3: '~', 4: 'T', 5: '@'}
        for y in range(13):
            for x in range(13):
                grid[y][x] = terrain_chars.get(self._grid_terrain[y, x], '?')

        # 基地
        if self._base_alive:
            bx, by = self._base_pos
            grid[by][bx] = 'B'

        # 玩家
        if self._player_alive:
            px, py = self._player_pos
            grid[py][px] = 'P'

        # 敌人
        for enemy in self._enemies:
            if enemy[2]:
                ex, ey = enemy[0]
                grid[ey][ex] = 'E'

        rows = [''.join(row) for row in grid]
        return '\n'.join(rows)


# 注册环境 (供 gymnasium.make 使用)
# 实际使用时直接实例化 TileEnv()
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_tile_env.py -v
```

Expected: tests PASS (或 check_env 可能有小修)

- [ ] **Step 5: Commit**

```bash
git add envs/ tests/test_tile_env.py
git commit -m "feat: add simplified 13x13 tile-based Tank Battle environment"
```

---

### Task 7: Reward Shaper 奖励塑形器

**Files:**
- Create: `training/__init__.py`
- Create: `training/reward_shaper.py`
- Create: `tests/test_reward_shaper.py`

- [ ] **Step 1: 编写 reward_shaper 测试**

在 `tests/test_reward_shaper.py` 中:

```python
"""Reward Shaper 单元测试"""

from training.reward_shaper import RewardShaper
from rl.config import Config


class TestRewardShaper:
    def test_survival_reward(self):
        cfg = Config()
        shaper = RewardShaper(cfg)
        r = shaper.compute(events=['survival'], player_pos=(5, 5), enemy_positions=[])
        assert r == cfg.reward.survival

    def test_kill_reward(self):
        cfg = Config()
        shaper = RewardShaper(cfg)
        r = shaper.compute(events=['kill'], player_pos=(5, 5), enemy_positions=[])
        assert r == cfg.reward.kill_enemy

    def test_potential_decreases_with_closer_enemy(self):
        cfg = Config()
        cfg.reward.potential_scale = 0.01
        shaper = RewardShaper(cfg)

        # 旧位置离敌人远, 新位置离敌人近 → 正奖励
        r = shaper.compute(
            events=['potential'],
            player_pos=(5, 5),
            player_pos_old=(10, 10),
            enemy_positions=[(5, 5)],  # 正好在玩家新位置
        )
        # distance_old - distance_new = sqrt(50) - 0 ≈ 7.07
        # reward = 0.01 * 7.07 ≈ 0.07
        assert r > 0, f"Expected positive potential reward, got {r}"

    def test_potential_increases_with_farther_enemy(self):
        cfg = Config()
        cfg.reward.potential_scale = 0.01
        shaper = RewardShaper(cfg)

        # 靠近敌人后远离 → 负奖励
        r = shaper.compute(
            events=['potential'],
            player_pos=(10, 10),
            player_pos_old=(5, 5),
            enemy_positions=[(5, 5)],
        )
        assert r < 0, f"Expected negative potential reward, got {r}"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_reward_shaper.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现 training/reward_shaper.py**

```python
"""Reward Shaper (奖励塑形器)

实现分层奖励 + 势能奖励:
  分层奖励: 根据游戏事件给予固定值奖励
  势能奖励: 基于向最近敌人靠近/远离的距离变化

潜在奖励的核心原理:
  - 奖励 = scale × (distance_old - distance_new)
  - 靠近敌人 → distance 减小 → reward > 0 (鼓励接近)
  - 远离敌人 → distance 增大 → reward < 0 (惩罚退缩)
"""

import numpy as np
from rl.config import Config, RewardConfig


class RewardShaper:
    """奖励塑形器

    组合事件奖励和势能奖励, 生成每个 timestep 的总奖励值。
    设计上独立于具体环境, 只依赖于 env 提供的 events 列表。
    """

    def __init__(self, config: Config):
        """初始化奖励塑形器

        Args:
            config: 全局配置, 使用其中的 RewardConfig
        """
        rc: RewardConfig = config.reward
        self.survival = rc.survival
        self.kill_enemy = rc.kill_enemy
        self.wave_clear = rc.wave_clear
        self.level_clear = rc.level_clear
        self.death = rc.death
        self.base_destroyed = rc.base_destroyed
        self.potential_scale = rc.potential_scale

    def compute(self, events: list[str], player_pos: tuple,
                enemy_positions: list[tuple],
                player_pos_old: tuple | None = None) -> float:
        """计算总奖励

        Args:
            events: 本步发生的事件列表
                ('survival', 'kill', 'wave_clear', 'level_clear', 'death', 'base_destroyed', 'potential')
            player_pos: 当前玩家坐标 (x, y)
            enemy_positions: 所有存活敌人坐标列表 [(x,y), ...]
            player_pos_old: 上一步玩家坐标, 用于势能计算

        Returns:
            total_reward: 本步总奖励值
        """
        reward = 0.0

        event_map = {
            'survival': self.survival,
            'kill': self.kill_enemy,
            'wave_clear': self.wave_clear,
            'level_clear': self.level_clear,
            'death': self.death,
            'base_destroyed': self.base_destroyed,
        }

        for event in events:
            if event == 'potential':
                reward += self._potential_reward(player_pos, enemy_positions, player_pos_old)
            else:
                reward += event_map.get(event, 0.0)

        return reward

    def _potential_reward(self, player_pos: tuple, enemy_positions: list[tuple],
                           player_pos_old: tuple | None) -> float:
        """计算势能奖励

        势能 = -distance_to_nearest_enemy (更近 = 更高势能)
        奖励 = scale × (势能_new - 势能_old)
             = scale × (distance_old_to_nearest - distance_new_to_nearest)

        Args:
            player_pos: 当前玩家坐标
            enemy_positions: 存活敌人坐标列表
            player_pos_old: 上一步玩家坐标

        Returns:
            势能奖励值, 靠近敌人为正, 远离为负
        """
        if player_pos_old is None or len(enemy_positions) == 0:
            return 0.0

        def nearest_distance(pos: tuple, enemy_positions: list[tuple]) -> float:
            min_dist = float('inf')
            for ex, ey in enemy_positions:
                dist = np.sqrt((pos[0] - ex) ** 2 + (pos[1] - ey) ** 2)
                min_dist = min(min_dist, dist)
            return min_dist if min_dist != float('inf') else 0.0

        dist_old = nearest_distance(player_pos_old, enemy_positions)
        dist_new = nearest_distance(player_pos, enemy_positions)

        return self.potential_scale * (dist_old - dist_new)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_reward_shaper.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add training/ tests/test_reward_shaper.py
git commit -m "feat: add reward shaper with layered rewards and potential-based shaping"
```

---

### Task 8: Curriculum Learning 课程管理

**Files:**
- Create: `training/curriculum.py`
- Create: `tests/test_curriculum.py`

- [ ] **Step 1: 编写 curriculum 测试**

在 `tests/test_curriculum.py` 中:

```python
"""Curriculum Learning 单元测试"""

from training.curriculum import CurriculumManager
from rl.config import Config


class TestCurriculumManager:
    def test_initial_stage_is_1(self):
        cfg = Config()
        cm = CurriculumManager(cfg)
        assert cm.current_stage == 1

    def test_promotion_if_win_rate_high(self):
        cfg = Config()
        cfg.curriculum.evaluation_window = 10
        cfg.curriculum.promotion_threshold_stage1 = 0.8
        cm = CurriculumManager(cfg)

        results = [True] * 9 + [False]  # 90% 通关率
        for r in results:
            cm.record_episode(won=r, enemies_killed=5, total_enemies=5)

        assert cm.should_promote(), "Should promote at 90% win rate"

    def test_no_promotion_if_low_rate(self):
        cfg = Config()
        cfg.curriculum.evaluation_window = 10
        cfg.curriculum.promotion_threshold_stage1 = 0.8
        cm = CurriculumManager(cfg)

        results = [True, False] * 5  # 50% 通关率
        for r in results:
            cm.record_episode(won=r, enemies_killed=3, total_enemies=5)

        assert not cm.should_promote(), "Should not promote at 50% win rate"

    def test_force_stage_override(self):
        cfg = Config()
        cm = CurriculumManager(cfg)
        cm.force_stage(3)
        assert cm.current_stage == 3

    def test_demotion_if_collapse(self):
        cfg = Config()
        cfg.curriculum.demotion_window = 10
        cfg.curriculum.demotion_threshold = 0.2
        cm = CurriculumManager(cfg)
        cm.force_stage(2)

        # 10 episode 全败
        for _ in range(10):
            cm.record_episode(won=False, enemies_killed=0, total_enemies=5)

        assert cm.should_demote(), "Should demote at 0% win rate"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_curriculum.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现 training/curriculum.py**

```python
"""Curriculum Learning (课程学习管理器)

原理:
  1. 按关卡固有特征(敌人数量、墙体密度等)自动分组为 3 个 difficulty stage
  2. Agent 需要满足通关率阈值才能晋升到下一阶段
  3. 若表现崩盘, 自动回退到上一阶段重新巩固

只读 tirinox 关卡数据, 不注入任何自定义游戏参数。
"""

from collections import deque
from rl.config import Config, CurriculumConfig


class CurriculumManager:
    """课程学习管理器

    追踪 agent 在每个 stage 的表现, 管理阶段晋升与回退。
    """

    def __init__(self, config: Config):
        """初始化课程管理器

        Args:
            config: 全局配置
        """
        cc: CurriculumConfig = config.curriculum
        self.eval_interval = cc.evaluation_interval
        self.eval_window = cc.evaluation_window
        self.promo_thresh_s1 = cc.promotion_threshold_stage1
        self.promo_thresh_s2 = cc.promotion_threshold_stage2
        self.demotion_thresh = cc.demotion_threshold
        self.demotion_window = cc.demotion_window

        self.current_stage = 1
        self.max_stage = 3
        self.episode_count = 0

        # 滑动窗口存储最近 episode 的通关结果
        self._recent_results = deque(maxlen=self.eval_window)

        # 关卡池: {stage: [level_indices]} (由 trainer 在读取 tirinox 数据后填充)
        self.level_pools = {1: [], 2: [], 3: []}

    def record_episode(self, won: bool, enemies_killed: int, total_enemies: int):
        """记录一个 episode 的结果

        Args:
            won: 是否通关
            enemies_killed: 本局击杀数
            total_enemies: 本局敌人总数
        """
        self._recent_results.append({
            'won': won,
            'enemies_killed': enemies_killed,
            'total_enemies': total_enemies,
        })
        self.episode_count += 1

    def win_rate(self) -> float:
        """计算滑动窗口内的通关率

        Returns:
            win_rate: 0.0 ~ 1.0 之间的通关比例
        """
        if len(self._recent_results) == 0:
            return 0.0
        wins = sum(1 for r in self._recent_results if r['won'])
        return wins / len(self._recent_results)

    def avg_kill_ratio(self) -> float:
        """计算滑动窗口内的平均击杀率

        Returns:
            avg_kill_ratio: 击杀数/总敌人数 的平均比例
        """
        if len(self._recent_results) == 0:
            return 0.0
        ratios = [r['enemies_killed'] / max(r['total_enemies'], 1)
                  for r in self._recent_results]
        return sum(ratios) / len(ratios)

    def should_promote(self) -> bool:
        """判断是否应该晋升到下一 stage

        条件:
          - S1→S2: 通关率 ≥ 80% 且平均击杀率 ≥ 0.8
          - S2→S3: 通关率 ≥ 50%

        Returns:
            True if promotion is warranted
        """
        if self.current_stage >= self.max_stage:
            return False
        if len(self._recent_results) < self.eval_window:
            return False  # 数据不足

        wr = self.win_rate()

        if self.current_stage == 1:
            kill_ratio = self.avg_kill_ratio()
            return wr >= self.promo_thresh_s1 and kill_ratio >= 0.8

        if self.current_stage == 2:
            return wr >= self.promo_thresh_s2

        return False

    def should_demote(self) -> bool:
        """判断是否应该回退到上一 stage

        条件: 通关率 < demotion_threshold 持续 demotion_window 个 episode

        Returns:
            True if demotion is warranted
        """
        if self.current_stage <= 1:
            return False
        if len(self._recent_results) < self.demotion_window:
            return False
        recent = list(self._recent_results)[-self.demotion_window:]
        wins = sum(1 for r in recent if r['won'])
        return wins / self.demotion_window < self.demotion_thresh

    def check_and_update(self) -> tuple[bool, int]:
        """每 eval_interval 个 episode 执行一次阶段检查

        Returns:
            (changed: bool, new_stage: int)
        """
        if self.episode_count % self.eval_interval != 0:
            return False, self.current_stage

        if self.should_promote():
            self.current_stage = min(self.current_stage + 1, self.max_stage)
            return True, self.current_stage

        if self.should_demote():
            self.current_stage = max(self.current_stage - 1, 1)
            return True, self.current_stage

        return False, self.current_stage

    def force_stage(self, stage: int):
        """手动覆写当前 stage (用于调试)

        Args:
            stage: 目标 stage (1-3)
        """
        self.current_stage = max(1, min(stage, self.max_stage))

    def get_level_pool(self, stage: int | None = None) -> list:
        """获取指定 stage 的关卡池

        Args:
            stage: stage 编号, 默认当前 stage

        Returns:
            关卡索引列表
        """
        s = stage if stage is not None else self.current_stage
        return self.level_pools.get(s, [])

    def classify_levels(self, level_stats: list[dict]):
        """根据关卡固有特征自动分组入 3 个 difficulty bucket

        使用敌人总数作为主要难度指标, 按分布分位数切分。

        Args:
            level_stats: [{'enemy_count': int, 'wall_density': float, ...}, ...]
        """
        if not level_stats:
            return

        enemy_counts = [ls['enemy_count'] for ls in level_stats]

        sorted_indices = sorted(range(len(enemy_counts)), key=lambda i: enemy_counts[i])

        n = len(sorted_indices)
        bucket_size = n // 3

        self.level_pools[1] = sorted_indices[:bucket_size]
        self.level_pools[2] = sorted_indices[bucket_size:2 * bucket_size]
        self.level_pools[3] = sorted_indices[2 * bucket_size:]
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_curriculum.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add training/curriculum.py tests/test_curriculum.py
git commit -m "feat: add curriculum learning manager with auto-promotion and demotion"
```

---

### Task 9: 训练主循环

**Files:**
- Create: `training/trainer.py`
- Create: `scripts/__init__.py`
- Create: `scripts/train_tile.py`

- [ ] **Step 1: 实现 training/trainer.py**

```python
"""训练主循环

编排 env ↔ PPO 的交互流程:
  1. 收集 rollout (env.step × N)
  2. GAE 计算 (buffer.compute_gae)
  3. PPO update (K epochs × mini-batches)
  4. TensorBoard 日志 + checkpoint
  5. Curriculum 评估
"""

import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rl.config import Config
from rl.network import ActorCritic
from rl.buffer import RolloutBuffer
from rl.ppo import PPOTrainer
from training.curriculum import CurriculumManager


class Trainer:
    """RL 训练编排器

    管理完整的训练生命周期:
      - 初始化网络、优化器、缓冲区
      - 执行 rollout 收集与 PPO 更新循环
      - 记录 TensorBoard 日志
      - 管理 checkpoint 和 curriculum
    """

    def __init__(self, config: Config, env):
        """初始化训练器

        Args:
            config: 全局配置
            env: Gymnasium Env 实例
        """
        self.config = config
        self.env = env

        training_cfg = config.training
        self.total_timesteps = training_cfg.total_timesteps
        self.checkpoint_interval = training_cfg.checkpoint_interval
        self.log_interval = training_cfg.log_interval
        self.save_dir = training_cfg.save_dir
        self.log_dir = training_cfg.log_dir
        self.seed = training_cfg.seed
        self.device = training_cfg.device

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs("crash_log", exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.net = ActorCritic(config).to(self.device)
        self.trainer = PPOTrainer(config)
        self.curriculum = CurriculumManager(config)

        self.buffer = RolloutBuffer(config)

        self._timestep = 0
        self._best_reward = float('-inf')

    def train(self):
        """主训练循环

        执行完整的 PPO 训练流程直到达到 total_timesteps。
        每步: 收集 rollout → GAE → PPO update → log → curriculum check
        """
        config = self.config
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_count = 0

        if self.device == "cuda":
            print(f"[Trainer] 使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[Trainer] 使用 CPU 训练")

        print(f"[Trainer] 开始训练, 总步数目标: {self.total_timesteps:,}")

        while self._timestep < self.total_timesteps:
            # === 收集 rollout ===
            for step in range(config.ppo.rollout_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action, log_prob, value = self.net.get_action(obs_tensor)

                action_idx = action.item()
                next_obs, reward, terminated, truncated, info = self.env.step(action_idx)

                self.buffer.store(
                    obs=torch.tensor(obs, dtype=torch.float32),
                    action=action.cpu(),
                    reward=reward,
                    value=value.cpu(),
                    log_prob=log_prob.cpu(),
                    done=terminated or truncated,
                )

                episode_reward += reward
                episode_length += 1
                self._timestep += 1
                obs = next_obs

                if terminated or truncated:
                    # episode 结束, 记录到 curriculum
                    self.curriculum.record_episode(
                        won=(not info.get('player_alive', True) and not info.get('base_alive', True))
                        or info.get('enemies_alive', 1) == 0,
                        enemies_killed=info.get('enemies_killed', 0),
                        total_enemies=info.get('enemies_total', 1),
                    )
                    episode_count += 1

                    # 日志
                    if episode_count % 10 == 0:
                        print(f"[Ep {episode_count}] "
                              f"timestep={self._timestep:,} "
                              f"reward={episode_reward:.1f} "
                              f"len={episode_length} "
                              f"stage={self.curriculum.current_stage}")

                    self.writer.add_scalar("Episode/Reward", episode_reward, self._timestep)
                    self.writer.add_scalar("Episode/Length", episode_length, self._timestep)

                    episode_reward = 0.0
                    episode_length = 0
                    obs, _ = self.env.reset()

                if self._timestep >= self.total_timesteps:
                    break

            # === GAE 计算 ===
            with torch.no_grad():
                last_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                _, _, last_value = self.net.get_action(last_obs)
                self.buffer.compute_gae(last_value.cpu())

            # === PPO Update ===
            start_time = time.time()
            avg_loss = self.trainer.update(self.net, self.buffer)
            update_time = time.time() - start_time

            self.writer.add_scalar("Train/Loss", avg_loss, self._timestep)
            self.writer.add_scalar("Train/UpdateTime", update_time, self._timestep)

            # === Logging ===
            if self._timestep % self.log_interval < config.ppo.rollout_steps:
                wr = self.curriculum.win_rate()
                self.writer.add_scalar("Curriculum/WinRate", wr, self._timestep)
                self.writer.add_scalar("Curriculum/Stage",
                                       self.curriculum.current_stage, self._timestep)

            # === Curriculum Check ===
            changed, new_stage = self.curriculum.check_and_update()
            if changed:
                print(f"[Curriculum] Stage 变更: → {new_stage}")

            # === Checkpoint ===
            if self._timestep % self.checkpoint_interval < config.ppo.rollout_steps:
                self.save_checkpoint()

            self.buffer.reset()

        self.writer.close()
        self.save_checkpoint()
        print(f"[Trainer] 训练完成! 总步数: {self._timestep:,}")

    def save_checkpoint(self):
        """保存模型 checkpoint"""
        path = os.path.join(self.save_dir, f"checkpoint_{self._timestep}.pt")
        torch.save({
            'timestep': self._timestep,
            'network_state_dict': self.net.state_dict(),
            'stage': self.curriculum.current_stage,
        }, path)
        print(f"[Checkpoint] 已保存: {path}")

    def load_checkpoint(self, path: str):
        """加载模型 checkpoint

        Args:
            path: checkpoint 文件路径
        """
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['network_state_dict'])
        self._timestep = ckpt.get('timestep', 0)
        self.curriculum.current_stage = ckpt.get('stage', 1)
        print(f"[Checkpoint] 已加载: {path} (timestep={self._timestep})")
```

- [ ] **Step 2: 实现 scripts/train_tile.py**

```python
"""简化 Tile 环境训练入口

使用简化 13×13 tile 环境快速验证 PPO 训练流程。
"""

import argparse
from rl.config import Config
from envs.tile_env import TileEnv
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="训练 Battle City RL Agent (Tile 环境)")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="总训练步数 (默认 10M)")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率 (默认 3e-4)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--force-stage", type=int, default=None,
                        help="手动设置 curriculum stage (1-3)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="训练设备 (默认 auto)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="恢复训练的 checkpoint 路径")
    args = parser.parse_args()

    config = Config()

    if args.total_steps is not None:
        config.training.total_timesteps = args.total_steps
    if args.lr is not None:
        config.ppo.lr = args.lr
    if args.seed is not None:
        config.training.seed = args.seed
    if args.device is not None:
        config.training.device = args.device

    print("=" * 50)
    print("Battle City RL 训练框架 — Tile 环境")
    print(f"设备: {config.training.device}")
    print(f"学习率: {config.ppo.lr}")
    print(f"总步数: {config.training.total_timesteps:,}")
    print(f"Seed: {config.training.seed}")
    print("=" * 50)

    env = TileEnv()
    trainer = Trainer(config, env)

    if args.force_stage is not None:
        trainer.curriculum.force_stage(args.force_stage)
        print(f"[手动] Curriculum Stage 设为 {args.force_stage}")

    if args.ckpt is not None:
        trainer.load_checkpoint(args.ckpt)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[Trainer] 用户中断训练, 正在保存 checkpoint...")
        trainer.save_checkpoint()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 验证训练器能启动 (小规模)**

```bash
python scripts/train_tile.py --total-steps 2048 --device cpu
```

Expected: 启动训练, 2-3 episode 后完成, 无报错, checkpoint 生成

- [ ] **Step 4: Commit**

```bash
git add training/trainer.py scripts/
git commit -m "feat: add training loop and tile environment training entry point"
```

---

### Task 10: 评估脚本与视频录制

**Files:**
- Create: `scripts/eval.py`
- Create: `envs/wrappers.py`

- [ ] **Step 1: 实现 envs/wrappers.py**

```python
"""Gymnasium 环境包装器

提供 RecordVideo 等通用包装器, 用于训练后录制 agent 表现视频。
"""

import gymnasium
from gymnasium.wrappers import RecordVideo
import os


def make_video_env(env, video_dir: str = "videos", episode_trigger=None):
    """创建可录制视频的环境

    Args:
        env: 原始 Gymnasium Env
        video_dir: 视频输出目录
        episode_trigger: 触发录制的条件 (默认每 10 episode 录制一次)

    Returns:
        包装后的视频录制环境
    """
    if episode_trigger is None:
        episode_trigger = lambda ep: ep % 10 == 0

    os.makedirs(video_dir, exist_ok=True)
    return RecordVideo(env, video_dir, episode_trigger=episode_trigger)
```

- [ ] **Step 2: 实现 scripts/eval.py**

```python
"""评估脚本

加载训练好的模型, 运行指定 episode 并可选录制视频。
"""

import argparse
import time
import torch
from rl.config import Config
from rl.network import ActorCritic
from envs.tile_env import TileEnv
from envs.wrappers import make_video_env


def main():
    parser = argparse.ArgumentParser(description="评估训练好的 Battle City Agent")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--episodes", type=int, default=10,
                        help="评估 episode 数")
    parser.add_argument("--record", action="store_true",
                        help="是否录制视频")
    parser.add_argument("--deterministic", action="store_true",
                        help="是否使用确定性策略 (取最高概率动作)")
    parser.add_argument("--render", action="store_true",
                        help="是否打印 ascii 渲染")
    args = parser.parse_args()

    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = ActorCritic(config).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt['network_state_dict'])
    net.eval()

    env = TileEnv()
    if args.record:
        env = make_video_env(env)

    rewards = []
    wins = 0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = net.get_action(obs_tensor, deterministic=args.deterministic)

            obs, reward, terminated, truncated, info = env.step(action.item())
            ep_reward += reward
            done = terminated or truncated

            if args.render:
                print(env.render())
                time.sleep(0.1)

        rewards.append(ep_reward)
        won = info.get('enemies_alive', 1) == 0
        if won:
            wins += 1
        print(f"[Ep {ep+1}/{args.episodes}] Reward: {ep_reward:.1f}  Win: {won}")

    env.close()
    print(f"\n平均 Reward: {sum(rewards)/len(rewards):.1f}")
    print(f"通关率: {wins}/{args.episodes} ({wins/args.episodes*100:.1f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 验证评估脚本可运行**

```bash
python scripts/train_tile.py --total-steps 4096 --device cpu
python scripts/eval.py --ckpt checkpoints/checkpoint_4096.pt --episodes 2 --render
```

Expected: 加载 model, 运行 2 episode, 打印 ascii 渲染画面

- [ ] **Step 4: Commit**

```bash
git add scripts/eval.py envs/wrappers.py
git commit -m "feat: add evaluation script and video recording wrapper"
```

---

### Task 11: 最终验证与集成测试

**Files:**
- Modify: 无新增文件

- [ ] **Step 1: 运行完整测试套件**

```bash
pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 2: 运行小规模端到端训练验证**

```bash
python scripts/train_tile.py --total-steps 5120 --device cpu
python scripts/eval.py --ckpt checkpoints/checkpoint_5120.pt --episodes 2
```

Expected: 训练和评估均无报错, reward 有变化趋势

- [ ] **Step 3: 添加 tirinox 子模块 (准备后期对接)**

```bash
git submodule add https://github.com/tirinox/pybattlecity.git extern/pybattlecity
git submodule update --init --recursive
```

- [ ] **Step 4: 最终 Commit**

```bash
git add -A
git commit -m "chore: add tirinox submodule, finalize tile training pipeline"
```

---

## Execution Notes

1. 所有代码含完整中文注释，符合教学项目定位
2. Conda 环境 `tank-rl-teach` 需在 Task 1 中创建
3. GPU 训练时注意 batch_size 与 8GB 显存的适配 (当前配置安全)
4. Checkpoint 文件名包含 timestep，方便恢复和版本管理
5. 后期 Tasks (tir_env, train_tir) 在 tile 环境验证完成后再实施
