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

    def _ensure_optimizer(self, net):
        """延迟创建优化器, 绑定到网络参数

        Args:
            net: ActorCritic 网络实例
        """
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
            values: [batch, 1] 当前 Critic 输出的状态价值, 可为 None 跳过 value loss
            actions: [batch] 实际执行的动作索引
            old_log_probs: [batch] rollout 时的旧策略对数概率
            advantages: [batch] GAE 优势估计
            returns: [batch] GAE 回报 (value training target), 可为 None 跳过 value loss

        Returns:
            loss: 标量 loss
            debug_info: (policy_loss, value_loss, entropy, clip_fraction) 供日志用
        """
        # Critic loss (MSE) - 当 returns 为 None 时跳过
        if returns is not None and values is not None:
            value_loss = nn.functional.mse_loss(values.squeeze(), returns)
        else:
            value_loss = torch.tensor(0.0)

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

    def update(self, net, buffer: RolloutBuffer):
        """执行一轮 PPO 更新

        将 buffer 中的 rollout 数据分成 mini-batch,
        在 K 个 epoch 中反复训练, 每次使用全部数据但顺序随机。

        Args:
            net: 当前策略网络 (ActorCritic)
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
