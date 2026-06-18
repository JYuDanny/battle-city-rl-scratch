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
