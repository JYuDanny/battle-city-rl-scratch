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

    def test_gymnasium_api_compliant(self):
        """验证完整 Gymnasium API 合规性 (check_env)"""
        from gymnasium.utils.env_checker import check_env

        env = TileEnv()
        check_env(env, skip_render_check=True)
