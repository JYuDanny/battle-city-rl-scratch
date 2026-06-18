"""Reward Shaper (奖励塑形器)

实现分层奖励 + 势能奖励:
  分层奖励: 根据游戏事件给予固定值奖励
  势能奖励: 基于向最近敌人靠近/远离的距离变化

潜在奖励的核心原理:
  - 奖励 = scale × (distance_old - distance_new)
  - 靠近敌人 → distance 减小 → reward > 0 (鼓励接近)
  - 远离敌人 → distance 增大 → reward < 0 (惩罚退缩)
"""

import numpy as np
from rl.config import Config, RewardConfig


class RewardShaper:
    """奖励塑形器

    组合事件奖励和势能奖励, 生成每个 timestep 的总奖励值。
    设计上独立于具体环境, 只依赖于 env 提供的 events 列表。
    """

    def __init__(self, config: Config):
        """初始化奖励塑形器

        Args:
            config: 全局配置, 使用其中的 RewardConfig
        """
        rc: RewardConfig = config.reward
        self.survival = rc.survival
        self.kill_enemy = rc.kill_enemy
        self.wave_clear = rc.wave_clear
        self.level_clear = rc.level_clear
        self.death = rc.death
        self.base_destroyed = rc.base_destroyed
        self.potential_scale = rc.potential_scale

    def compute(self, events: list[str], player_pos: tuple,
                enemy_positions: list[tuple],
                player_pos_old: tuple | None = None) -> float:
        """计算总奖励

        Args:
            events: 本步发生的事件列表
            player_pos: 当前玩家坐标 (x, y)
            enemy_positions: 所有存活敌人坐标列表 [(x,y), ...]
            player_pos_old: 上一步玩家坐标, 用于势能计算

        Returns:
            total_reward: 本步总奖励值
        """
        reward = 0.0

        event_map = {
            'survival': self.survival,
            'kill': self.kill_enemy,
            'wave_clear': self.wave_clear,
            'level_clear': self.level_clear,
            'death': self.death,
            'base_destroyed': self.base_destroyed,
        }

        for event in events:
            if event == 'potential':
                reward += self._potential_reward(player_pos, enemy_positions, player_pos_old)
            else:
                reward += event_map.get(event, 0.0)

        return reward

    def _potential_reward(self, player_pos: tuple, enemy_positions: list[tuple],
                           player_pos_old: tuple | None) -> float:
        """计算势能奖励

        势能 = -distance_to_nearest_enemy (更近 = 更高势能)
        奖励 = scale × (势能_new - 势能_old)
             = scale × (distance_old_to_nearest - distance_new_to_nearest)

        Args:
            player_pos: 当前玩家坐标
            enemy_positions: 存活敌人坐标列表
            player_pos_old: 上一步玩家坐标

        Returns:
            势能奖励值, 靠近敌人为正, 远离为负
        """
        if player_pos_old is None or len(enemy_positions) == 0:
            return 0.0

        def nearest_distance(pos: tuple, enemy_positions: list[tuple]) -> float:
            min_dist = float('inf')
            for ex, ey in enemy_positions:
                dist = np.sqrt((pos[0] - ex) ** 2 + (pos[1] - ey) ** 2)
                if dist < min_dist:
                    min_dist = dist
            return min_dist if min_dist != float('inf') else 0.0

        dist_old = nearest_distance(player_pos_old, enemy_positions)
        dist_new = nearest_distance(player_pos, enemy_positions)

        return self.potential_scale * (dist_old - dist_new)
