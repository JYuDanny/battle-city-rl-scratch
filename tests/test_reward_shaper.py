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

    def test_base_destroyed_reward(self):
        cfg = Config()
        shaper = RewardShaper(cfg)
        r = shaper.compute(events=['base_destroyed'], player_pos=(5, 5), enemy_positions=[])
        assert r == cfg.reward.base_destroyed

    def test_facing_enemy_gives_reward(self):
        """面朝敌人时获得 facing_enemy 奖励"""
        cfg = Config()
        shaper = RewardShaper(cfg)

        r = shaper.compute(
            events=['shaping'],
            player_pos=(5, 5),
            enemy_positions=[(5, 0)],
            player_dir=(0, -1),
        )
        assert r > 0, f"Facing enemy should give positive reward, got {r}"

    def test_facing_away_gives_no_reward(self):
        """背对敌人时不获得 facing_enemy 奖励"""
        cfg = Config()
        shaper = RewardShaper(cfg)

        r = shaper.compute(
            events=['shaping'],
            player_pos=(5, 5),
            enemy_positions=[(5, 0)],
            player_dir=(0, 1),
        )
        assert r == 0.0, f"Facing away should give zero reward, got {r}"

    def test_shoot_towards_enemy_gives_reward(self):
        """向敌人方向射击获得 shoot_towards 奖励"""
        cfg = Config()
        shaper = RewardShaper(cfg)

        r = shaper.compute(
            events=['shaping'],
            player_pos=(5, 5),
            enemy_positions=[(5, 0)],
            shot_direction=(0, -1),
        )
        assert r > 0, f"Shooting towards enemy should give positive reward, got {r}"

    def test_no_enemies_no_shaping(self):
        """没有敌人时塑形奖励为 0"""
        cfg = Config()
        shaper = RewardShaper(cfg)
        r = shaper.compute(
            events=['shaping'],
            player_pos=(5, 5),
            enemy_positions=[],
            player_dir=(0, -1),
        )
        assert r == 0.0
