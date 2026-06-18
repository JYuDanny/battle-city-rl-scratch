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

        expected_returns = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        assert np.allclose(buffer.returns.numpy(), expected_returns, atol=1e-5), \
            f"Returns differ: got {buffer.returns.numpy()}, expected {expected_returns}"

        mean = buffer.advantages.mean().item()
        std = buffer.advantages.std().item()
        assert abs(mean) < 1e-5, f"Normalized advantage mean not near 0: {mean}"
        assert abs(std - 1.0) < 1e-5, f"Normalized advantage std not near 1: {std}"

    def test_advantage_normalization(self):
        """验证 advantages 归一化后均值≈0, 标准差≈1"""
        config = Config()
        buffer = RolloutBuffer(config)

        buffer.reset()
        for i in range(config.ppo.rollout_steps):
            buffer.store(
                obs=torch.randn(13, 13, 9),
                action=torch.tensor(i % 6),
                reward=float(i % 3) - 1.0,
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
