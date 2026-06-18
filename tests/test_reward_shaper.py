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

        r = shaper.compute(
            events=['potential'],
            player_pos=(5, 5),
            player_pos_old=(10, 10),
            enemy_positions=[(5, 5)],
        )
        assert r > 0, f"Expected positive potential reward, got {r}"

    def test_potential_increases_with_farther_enemy(self):
        cfg = Config()
        cfg.reward.potential_scale = 0.01
        shaper = RewardShaper(cfg)

        r = shaper.compute(
            events=['potential'],
            player_pos=(10, 10),
            player_pos_old=(5, 5),
            enemy_positions=[(5, 5)],
        )
        assert r < 0, f"Expected negative potential reward, got {r}"
