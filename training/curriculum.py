"""Curriculum Learning (课程学习管理器)

动态难度调节:
  - 根据 agent 近期通关率自动调整地图难度
  - 通关率高 → 增加难度(更多敌人、更密围墙)
  - 通关率低 → 降低难度
  - 不使用固定分桶, 连续调节难度参数
"""

from collections import deque
from rl.config import Config, CurriculumConfig


class CurriculumManager:
    """动态难度课程管理器

    持续追踪 agent 表现, 实时调节环境难度。
    difficulty 参数 (0.0~1.0) 决定了环境生成规则:
      0.0 = 最简单 (1-2个敌人, 5%围墙)
      0.5 = 中等   (3-4个敌人, 20%围墙)
      1.0 = 最困难 (5-6个敌人, 35%围墙)
    """

    def __init__(self, config: Config):
        """初始化课程管理器

        Args:
            config: 全局配置
        """
        cc: CurriculumConfig = config.curriculum
        self.eval_interval = cc.evaluation_interval
        self.eval_window = cc.evaluation_window
        self.promo_thresh = 0.7   # 通关率 >70% → 升难度
        self.demote_thresh = 0.3  # 通关率 <30% → 降难度
        self.difficulty_step = 0.1  # 每次调节步长

        self.current_difficulty = 0.0  # 从最简单开始
        self.episode_count = 0

        # 滑动窗口存储最近 episode 的通关结果
        self._recent_results = deque(maxlen=self.eval_window)

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

    def check_and_update(self) -> tuple:
        """每 eval_interval 个 episode 执行一次难度检查

        Returns:
            (changed: bool, new_difficulty: float)
        """
        if self.episode_count % self.eval_interval != 0:
            return False, self.current_difficulty
        if len(self._recent_results) < self.eval_window:
            return False, self.current_difficulty

        wr = self.win_rate()
        old_diff = self.current_difficulty

        if wr > self.promo_thresh:
            self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_step)
        elif wr < self.demote_thresh:
            self.current_difficulty = max(0.0, self.current_difficulty - self.difficulty_step)

        return (self.current_difficulty != old_diff), self.current_difficulty

    def force_difficulty(self, difficulty: float):
        """手动覆写难度

        Args:
            difficulty: 0.0 ~ 1.0 之间的难度值
        """
        self.current_difficulty = max(0.0, min(1.0, difficulty))

    def get_params(self) -> dict:
        """根据当前难度返回环境生成参数
        
        Returns:
            dict with 'num_enemies', 'wall_density', 'enemy_speed_scale'
        """
        d = self.current_difficulty
        return {
            'num_enemies': max(1, int(2 + d * 4)),       # 1~6
            'wall_density': 0.05 + d * 0.30,              # 5%~35%
            'enemy_speed_scale': 0.5 + d * 0.5,           # 50%~100%
        }

    @property
    def current_stage(self):
        """兼容旧接口: 将连续难度映射为整数 stage"""
        if self.current_difficulty < 0.33:
            return 1
        elif self.current_difficulty < 0.66:
            return 2
        else:
            return 3

    @property
    def level_pools(self):
        """兼容旧接口: 返回空字典(不再需要关卡池)"""
        return {1: [], 2: [], 3: []}
