"""Curriculum Learning 单元测试"""

from training.curriculum import CurriculumManager
from rl.config import Config


class TestCurriculumManager:
    def test_initial_stage_is_1(self):
        cfg = Config()
        cm = CurriculumManager(cfg)
        assert cm.current_stage == 1

    def test_promotion_if_win_rate_high(self):
        cfg = Config()
        cfg.curriculum.evaluation_window = 10
        cfg.curriculum.promotion_threshold_stage1 = 0.8
        cm = CurriculumManager(cfg)

        results = [True] * 9 + [False]  # 90% 通关率
        for r in results:
            cm.record_episode(won=r, enemies_killed=5, total_enemies=5)

        assert cm.should_promote(), "Should promote at 90% win rate"

    def test_no_promotion_if_low_rate(self):
        cfg = Config()
        cfg.curriculum.evaluation_window = 10
        cfg.curriculum.promotion_threshold_stage1 = 0.8
        cm = CurriculumManager(cfg)

        results = [True, False] * 5  # 50% 通关率
        for r in results:
            cm.record_episode(won=r, enemies_killed=3, total_enemies=5)

        assert not cm.should_promote(), "Should not promote at 50% win rate"

    def test_force_stage_override(self):
        cfg = Config()
        cm = CurriculumManager(cfg)
        cm.force_stage(3)
        assert cm.current_stage == 3

    def test_demotion_if_collapse(self):
        cfg = Config()
        cfg.curriculum.demotion_window = 10
        cfg.curriculum.demotion_threshold = 0.2
        cm = CurriculumManager(cfg)
        cm.force_stage(2)

        for _ in range(10):
            cm.record_episode(won=False, enemies_killed=0, total_enemies=5)

        assert cm.should_demote(), "Should demote at 0% win rate"
