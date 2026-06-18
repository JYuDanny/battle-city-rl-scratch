"""Reward Shaper (奖励塑形器)

实现分层奖励 + 行为塑形奖励:
  分层奖励: 根据游戏事件给予固定值奖励
  行为塑形: 引导 agent 面朝敌人 + 向敌人方向射击 (而非盲目接近)
"""

import numpy as np
from rl.config import Config, RewardConfig


class RewardShaper:
    """奖励塑形器

    组合事件奖励和行为塑形奖励, 生成每个 timestep 的总奖励值。
    设计核心: 奖励"面朝敌人"和"向敌人射击", 而非"接近敌人"(接近会导致自杀行为)。
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
        self.base_danger = rc.base_danger
        self.facing_enemy = rc.facing_enemy
        self.shoot_towards_enemy = rc.shoot_towards_enemy

    def compute(self, events: list[str], player_pos: tuple,
                enemy_positions: list[tuple],
                player_dir: tuple | None = None,
                shot_direction: tuple | None = None) -> float:
        """计算总奖励

        Args:
            events: 本步发生的事件列表
                ('survival', 'kill', 'wave_clear', 'level_clear',
                 'death', 'base_destroyed', 'base_danger', 'shaping')
            player_pos: 当前玩家坐标 (x, y)
            enemy_positions: 所有存活敌人坐标列表 [(x,y), ...]
            player_dir: 玩家当前朝向 (dx, dy), 用于 facing 奖励
            shot_direction: 子弹方向 (dx, dy), 用于 shoot_towards 奖励

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
            'base_danger': self.base_danger,
        }

        for event in events:
            if event == 'shaping':
                reward += self._shaping_reward(
                    player_pos, enemy_positions, player_dir, shot_direction
                )
            else:
                reward += event_map.get(event, 0.0)

        return reward

    def _shaping_reward(
        self,
        player_pos: tuple,
        enemy_positions: list[tuple],
        player_dir: tuple | None = None,
        shot_direction: tuple | None = None,
    ) -> float:
        """计算行为塑形奖励

        两个信号:
        1. facing_enemy: 玩家朝向是否对准最近敌人 (点积 > 0.5 即为朝向)
        2. shoot_towards_enemy: 射击方向是否对准敌人 (点积 > 0.8)
        
        不使用距离变化(原 potential reward), 因为"接近敌人"会引导自杀行为。

        Args:
            player_pos: 当前玩家坐标
            enemy_positions: 存活敌人坐标列表
            player_dir: 玩家朝向方向
            shot_direction: 射击方向

        Returns:
            塑形奖励值
        """
        reward = 0.0
        if not enemy_positions:
            return 0.0

        nearest = None
        min_dist = float('inf')
        for ex, ey in enemy_positions:
            dist = np.sqrt((player_pos[0] - ex) ** 2 + (player_pos[1] - ey) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = (ex, ey)

        if nearest is None:
            return 0.0

        dx = nearest[0] - player_pos[0]
        dy = nearest[1] - player_pos[1]
        norm = np.sqrt(dx**2 + dy**2) + 1e-8
        enemy_dir = (dx / norm, dy / norm)

        if player_dir is not None:
            dot = player_dir[0] * enemy_dir[0] + player_dir[1] * enemy_dir[1]
            if dot > 0.5:
                reward += self.facing_enemy

        if shot_direction is not None:
            dot = shot_direction[0] * enemy_dir[0] + shot_direction[1] * enemy_dir[1]
            if dot > 0.8:
                reward += self.shoot_towards_enemy

        return reward
