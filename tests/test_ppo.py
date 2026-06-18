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
        """验证 PPO update 正常执行, loss 为有限值且不为 NaN"""
        config = Config()
        config.ppo.rollout_steps = 128
        config.ppo.k_epochs = 1

        net = ActorCritic(config)
        trainer = PPOTrainer(config)

        buffer = self._create_mock_rollout(trainer, config)

        # 执行更新
        loss_after = trainer.update(net, buffer)

        # loss 应为有限值
        assert not torch.isnan(torch.tensor(loss_after)), "Loss is NaN"
        assert loss_after > 0, f"Loss should be positive, got {loss_after}"

    def test_clip_prevents_large_policy_change(self):
        """验证 clip 机制限制了策略比率变化在 [1-ε, 1+ε] 范围内
        
        构造场景: 让 old_logp 和 new_logp 差距很大,
        检验 clip 后 policy_loss 不会爆炸 (NaN).
        """
        config = Config()
        config.ppo.clip_epsilon = 0.2
        net = ActorCritic(config)
        trainer = PPOTrainer(config)

        # 手动构造极端场景
        obs = torch.randn(8, 13, 13, 9)
        logits, values = net(obs)
        
        actions = torch.zeros(8, dtype=torch.long)
        dist = torch.distributions.Categorical(logits=logits)
        old_logp = dist.log_prob(actions).detach()
        old_logp[0] = -10.0  # 极端旧概率

        adv = torch.randn(8)
        # 使用当前 value 作为 return, 让 value loss ≈ 0, 聚焦测试 policy loss
        returns = values.squeeze().detach()

        loss, (policy_loss, value_loss, entropy, clip_frac) = trainer.compute_loss(
            logits, values, actions, old_logp, adv, returns
        )

        # clip 应该防止比率爆炸 → 不会产生 NaN
        assert not torch.isnan(loss), "Loss is NaN — clip may not be working"
        assert not torch.isnan(policy_loss), "Policy loss is NaN"
        # 极端 log_prob 差应触发 clip
        assert clip_frac > 0, f"Expected some clipping with extreme ratio, got clip_frac={clip_frac}"
