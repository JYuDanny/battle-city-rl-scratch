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
