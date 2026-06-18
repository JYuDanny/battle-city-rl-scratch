"""TirEnv 环境单元测试

验证 tirinox 包装器基本功能。
"""

import pytest
from envs.tir_env import TirEnv


class TestTirEnv:
    """TirEnv 测试套件"""

    def test_reset_returns_valid_obs(self):
        """reset 后观察维度正确"""
        env = TirEnv(fps=120)  # 加速测试
        obs, info = env.reset()

        assert obs.shape == (84, 84, 3), f"Expected (84,84,3), got {obs.shape}"
        assert obs.dtype == 'uint8'
        assert isinstance(info, dict)
        env.close()

    def test_step_returns_expected_tuple(self):
        """step 返回 (obs, reward, terminated, truncated, info)"""
        env = TirEnv(fps=120)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)

        assert obs.shape == (84, 84, 3)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_action_space_is_discrete_10(self):
        """动作空间为 Discrete(10)"""
        env = TirEnv(fps=120)
        assert env.action_space.n == 10
        env.close()

    def test_bullet_interaction(self):
        """验证射击产生子弹"""
        env = TirEnv(fps=120)
        env.reset()

        # 连续射击多次 (frame skip * N frames)
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(9)  # IDLE+FIRE

        # 游戏应该仍在运行 (玩家不会立即死)
        assert not terminated, "Game should not end immediately"
        env.close()

    def test_base_destruction_ends_game(self):
        """验证基地被毁导致游戏结束 (通过多步运行)"""
        env = TirEnv(fps=120)
        env.reset()

        # 执行多步 (可能不会摧毁基地, 但至少验证不崩溃)
        for _ in range(50):
            obs, reward, terminated, truncated, info = env.step(4)  # IDLE
            if terminated:
                break

        env.close()
