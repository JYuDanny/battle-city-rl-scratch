"""Curriculum Learning 单元测试"""

from training.curriculum import CurriculumManager
from rl.config import Config


class TestCurriculumManager:
    def test_initial_difficulty_is_zero(self):
        cfg = Config()
        cm = CurriculumManager(cfg)
        assert cm.current_difficulty == 0.0

    def test_difficulty_increases_if_win_rate_high(self):
        cfg = Config()
        cfg.curriculum.evaluation_interval = 10
        cfg.curriculum.evaluation_window = 10
        cm = CurriculumManager(cfg)

        results = [True] * 9 + [False]  # 90% 通关率
        for r in results:
            cm.record_episode(won=r, enemies_killed=5, total_enemies=5)

        changed, new_diff = cm.check_and_update()
        assert changed, "Should change difficulty at 90% win rate"
        assert new_diff > 0.0, f"Difficulty should increase, got {new_diff}"

    def test_difficulty_decreases_if_win_rate_low(self):
        cfg = Config()
        cfg.curriculum.evaluation_interval = 10
        cfg.curriculum.evaluation_window = 10
        cm = CurriculumManager(cfg)
        cm.force_difficulty(0.8)  # 从高难度开始

        results = [False] * 10  # 0% 通关率
        for r in results:
            cm.record_episode(won=r, enemies_killed=0, total_enemies=5)

        changed, new_diff = cm.check_and_update()
        assert changed, "Should change difficulty at 0% win rate"
        assert new_diff < 0.8, f"Difficulty should decrease, got {new_diff}"

    def test_no_change_in_mid_range(self):
        cfg = Config()
        cfg.curriculum.evaluation_interval = 10
        cfg.curriculum.evaluation_window = 10
        cm = CurriculumManager(cfg)
        cm.force_difficulty(0.5)

        results = [True, False] * 5  # 50% 通关率
        for r in results:
            cm.record_episode(won=r, enemies_killed=3, total_enemies=5)

        changed, new_diff = cm.check_and_update()
        assert not changed, f"Should not change at 50% win rate"

    def test_force_difficulty_override(self):
        cfg = Config()
        cm = CurriculumManager(cfg)
        cm.force_difficulty(0.8)
        assert cm.current_difficulty == 0.8

    def test_get_params_returns_reasonable_values(self):
        cfg = Config()
        cm = CurriculumManager(cfg)
        cm.force_difficulty(0.5)
        params = cm.get_params()
        assert 1 <= params['num_enemies'] <= 6
        assert 0.05 <= params['wall_density'] <= 0.35
        assert 0.5 <= params['enemy_speed_scale'] <= 1.0

    def test_win_rate_empty(self):
        cfg = Config()
        cm = CurriculumManager(cfg)
        assert cm.win_rate() == 0.0
