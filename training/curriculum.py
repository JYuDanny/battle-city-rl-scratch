"""Curriculum Learning (课程学习管理器)

原理:
  1. 按关卡固有特征(敌人数量、墙体密度等)自动分组为 3 个 difficulty stage
  2. Agent 需要满足通关率阈值才能晋升到下一阶段
  3. 若表现崩盘, 自动回退到上一阶段重新巩固

只读 tirinox 关卡数据, 不注入任何自定义游戏参数。
"""

from collections import deque
from rl.config import Config, CurriculumConfig


class CurriculumManager:
    """课程学习管理器

    追踪 agent 在每个 stage 的表现, 管理阶段晋升与回退。
    """

    def __init__(self, config: Config):
        """初始化课程管理器

        Args:
            config: 全局配置
        """
        cc: CurriculumConfig = config.curriculum
        self.eval_interval = cc.evaluation_interval
        self.eval_window = cc.evaluation_window
        self.promo_thresh_s1 = cc.promotion_threshold_stage1
        self.promo_thresh_s2 = cc.promotion_threshold_stage2
        self.demotion_thresh = cc.demotion_threshold
        self.demotion_window = cc.demotion_window

        self.current_stage = 1
        self.max_stage = 3
        self.episode_count = 0

        # 滑动窗口存储最近 episode 的通关结果
        self._recent_results = deque(maxlen=self.eval_window)

        # 关卡池: {stage: [level_indices]} (由 trainer 在读取 tirinox 数据后填充)
        self.level_pools = {1: [], 2: [], 3: []}

    def record_episode(self, won: bool, enemies_killed: int, total_enemies: int):
        """记录一个 episode 的结果

        Args:
            won: 是否通关
            enemies_killed: 本局击杀数
            total_enemies: 本局敌人总数
        """
        self._recent_results.append({
            'won': won,
            'enemies_killed': enemies_killed,
            'total_enemies': total_enemies,
        })
        self.episode_count += 1

    def win_rate(self) -> float:
        """计算滑动窗口内的通关率

        Returns:
            win_rate: 0.0 ~ 1.0 之间的通关比例
        """
        if len(self._recent_results) == 0:
            return 0.0
        wins = sum(1 for r in self._recent_results if r['won'])
        return wins / len(self._recent_results)

    def avg_kill_ratio(self) -> float:
        """计算滑动窗口内的平均击杀率"""
        if len(self._recent_results) == 0:
            return 0.0
        ratios = [r['enemies_killed'] / max(r['total_enemies'], 1)
                  for r in self._recent_results]
        return sum(ratios) / len(ratios)

    def should_promote(self) -> bool:
        """判断是否应该晋升到下一 stage"""
        if self.current_stage >= self.max_stage:
            return False
        if len(self._recent_results) < self.eval_window:
            return False

        wr = self.win_rate()
        if self.current_stage == 1:
            kill_ratio = self.avg_kill_ratio()
            return wr >= self.promo_thresh_s1 and kill_ratio >= 0.8
        if self.current_stage == 2:
            return wr >= self.promo_thresh_s2
        return False

    def should_demote(self) -> bool:
        """判断是否应该回退到上一 stage"""
        if self.current_stage <= 1:
            return False
        if len(self._recent_results) < self.demotion_window:
            return False
        recent = list(self._recent_results)[-self.demotion_window:]
        wins = sum(1 for r in recent if r['won'])
        return wins / self.demotion_window < self.demotion_thresh

    def check_and_update(self) -> tuple:
        """每 eval_interval 个 episode 执行一次阶段检查

        Returns:
            (changed: bool, new_stage: int)
        """
        if self.episode_count % self.eval_interval != 0:
            return False, self.current_stage

        if self.should_promote():
            self.current_stage = min(self.current_stage + 1, self.max_stage)
            return True, self.current_stage

        if self.should_demote():
            self.current_stage = max(self.current_stage - 1, 1)
            return True, self.current_stage

        return False, self.current_stage

    def force_stage(self, stage: int):
        """手动覆写当前 stage"""
        self.current_stage = max(1, min(stage, self.max_stage))

    def get_level_pool(self, stage: int | None = None) -> list:
        """获取指定 stage 的关卡池"""
        s = stage if stage is not None else self.current_stage
        return self.level_pools.get(s, [])

    def classify_levels(self, level_stats: list[dict]):
        """根据关卡固有特征自动分组入 3 个 difficulty bucket"""
        if not level_stats:
            return
        enemy_counts = [ls['enemy_count'] for ls in level_stats]
        sorted_indices = sorted(range(len(enemy_counts)), key=lambda i: enemy_counts[i])
        n = len(sorted_indices)
        bucket_size = n // 3
        self.level_pools[1] = sorted_indices[:bucket_size]
        self.level_pools[2] = sorted_indices[bucket_size:2 * bucket_size]
        self.level_pools[3] = sorted_indices[2 * bucket_size:]
