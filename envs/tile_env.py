"""简化 Tile-Based 坦克大战 Gymnasium 环境

13×13 网格地图, 回合制步进:
  1. 玩家执行动作 (6 选: ↑↓←→射击静止)
  2. 敌人执行随机策略
  3. 处理碰撞 (子弹 vs 实体, 玩家 vs 敌人等)
  4. 计算 reward
  5. 检查终止条件

地图编码: 13×13×[C 通道]
  Channel 0-4: 地形 (empty/brick/water/tree/steel)
  Channel 5: 基地
  Channel 6-7: 玩家坐标
  Channel 8: 玩家方向
"""

import gymnasium
from gymnasium import spaces
import numpy as np


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "SHOOT", "IDLE"]
DIRECTIONS = {
    "UP": np.array([0, -1]),
    "DOWN": np.array([0, 1]),
    "LEFT": np.array([-1, 0]),
    "RIGHT": np.array([1, 0]),
}


class TileEnv(gymnasium.Env):
    """简化 13×13 坦克大战环境

    回合制网格环境, 用于快速验证 PPO 训练流程。
    每步 agent 执行一个动作, 环境同步步进所有实体。

    Observation: 13×13×9 多通道张量
    Action: Discrete(6) [UP, DOWN, LEFT, RIGHT, SHOOT, IDLE]
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        """初始化环境

        Args:
            render_mode: 渲染模式, 'ansi' 或 None
        """
        super().__init__()

        self.grid_size = 13
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(13, 13, 9),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(6)

        self._max_steps = 2000
        self._step_count = 0

        self._grid_terrain = None
        self._player_pos = None
        self._player_dir = None
        self._player_alive = True
        self._enemies = []
        self._bullets = []
        self._base_alive = True
        self._base_pos = (6, 12)
        self._wave = 0
        self._enemies_killed = 0
        self._enemies_total = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        """重置环境到初始状态

        Returns:
            observation: 初始观察 [13,13,9]
            info: 额外信息字典
        """
        super().reset(seed=seed)

        self._grid_terrain = np.zeros((13, 13), dtype=np.int32)
        self._grid_terrain[0, :] = 5
        self._grid_terrain[-1, :] = 5
        self._grid_terrain[:, 0] = 5
        self._grid_terrain[:, -1] = 5

        rng = np.random.default_rng(seed)
        for y in range(1, 12):
            for x in range(1, 12):
                if (x, y) == self._base_pos or (x, y) == (6, 11):
                    continue
                if rng.random() < 0.2:
                    self._grid_terrain[y, x] = 2

        self._base_alive = True

        self._player_pos = np.array([6, 11], dtype=np.int32)
        self._player_dir = np.array([0, -1], dtype=np.int32)
        self._player_alive = True

        self._wave = 1
        self._enemies = self._spawn_enemies()
        self._enemies_total = len(self._enemies)
        self._enemies_killed = 0

        self._bullets = []
        self._step_count = 0

        return self._get_obs(), self._get_info()

    def _spawn_enemies(self):
        """在上半区域生成敌人 (最多3个)

        Returns:
            list of [pos_array, dir_array, alive_bool]
        """
        rng = np.random.default_rng(self.np_random.integers(0, 2**31))
        enemies = []
        for _ in range(3):
            for _ in range(100):
                x = rng.integers(1, 12)
                y = rng.integers(1, 5)
                if self._grid_terrain[y, x] == 0:
                    enemies.append([
                        np.array([x, y], dtype=np.int32),
                        np.array([0, 1], dtype=np.int32),
                        True
                    ])
                    break
        return enemies

    def step(self, action: int):
        """执行一步环境交互

        处理流程:
        1. 玩家行动 (基于 action)
        2. 敌人行动 (随机策略)
        3. 子弹移动与碰撞检测
        4. 奖励计算
        5. 终止判断

        Args:
            action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=SHOOT, 5=IDLE

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1
        reward_override = 0.0

        if self._player_alive:
            if action < 4:
                self._player_dir = DIRECTIONS[ACTIONS[action]]
                new_pos = self._player_pos + self._player_dir
                if self._is_walkable(new_pos):
                    blocked = False
                    for enemy in self._enemies:
                        if enemy[2] and np.array_equal(new_pos, enemy[0]):
                            blocked = True
                            break
                    if not blocked:
                        self._player_pos = new_pos

            elif action == 4:
                bx = int(self._player_pos[0]) + int(self._player_dir[0])
                by = int(self._player_pos[1]) + int(self._player_dir[1])
                if 0 <= bx < 13 and 0 <= by < 13:
                    self._bullets.append([bx, by, int(self._player_dir[0]),
                                          int(self._player_dir[1]), 'player'])

        for enemy in self._enemies:
            if not enemy[2]:
                continue
            act = self.np_random.integers(0, 6)
            if act < 4:
                dir_vec = DIRECTIONS[ACTIONS[act]]
                new_pos = enemy[0] + dir_vec
                if self._is_walkable(new_pos):
                    blocked = False
                    if self._player_alive and np.array_equal(new_pos, self._player_pos):
                        blocked = True
                    if not blocked:
                        enemy[0] = new_pos
                        enemy[1] = dir_vec
            elif act == 4:
                bx = int(enemy[0][0]) + int(enemy[1][0])
                by = int(enemy[0][1]) + int(enemy[1][1])
                if 0 <= bx < 13 and 0 <= by < 13:
                    self._bullets.append([bx, by, int(enemy[1][0]),
                                          int(enemy[1][1]), 'enemy'])

        new_bullets = []
        for b in self._bullets:
            bx, by, dx, dy, owner = b
            nx, ny = bx + dx, by + dy

            if nx < 0 or nx >= 13 or ny < 0 or ny >= 13:
                continue

            if self._grid_terrain[ny, nx] == 2:
                self._grid_terrain[ny, nx] = 0
                continue

            if (nx, ny) == self._base_pos and self._base_alive:
                if owner == 'enemy':
                    self._base_alive = False
                    reward_override += self._get_reward_event('base_destroyed')
                continue

            if self._grid_terrain[ny, nx] == 5:
                continue

            hit = False
            if owner == 'player':
                for enemy in self._enemies:
                    if not enemy[2]:
                        continue
                    if nx == enemy[0][0] and ny == enemy[0][1]:
                        enemy[2] = False
                        self._enemies_killed += 1
                        reward_override += self._get_reward_event('kill')
                        hit = True
                        break

            elif owner == 'enemy':
                if self._player_alive and nx == self._player_pos[0] and ny == self._player_pos[1]:
                    self._player_alive = False
                    reward_override += self._get_reward_event('death')
                    hit = True

            if not hit:
                new_bullets.append([nx, ny, dx, dy, owner])

        self._bullets = new_bullets

        reward = self._get_reward_event('survival')

        alive_enemies = [e for e in self._enemies if e[2]]
        if len(alive_enemies) == 0:
            reward += self._get_reward_event('wave_clear')

        reward += reward_override

        terminated = False
        if not self._player_alive or not self._base_alive:
            terminated = True
        if len(alive_enemies) == 0:
            terminated = True

        truncated = self._step_count >= self._max_steps and not terminated

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_reward_event(self, event: str) -> float:
        """获取事件对应的奖励值 (简化版硬编码)

        Args:
            event: 事件名

        Returns:
            奖励值
        """
        rewards = {
            'survival': 0.01,
            'kill': 2.0,
            'wave_clear': 5.0,
            'death': -10.0,
            'base_destroyed': -50.0,
        }
        return rewards.get(event, 0.0)

    def _is_walkable(self, pos: np.ndarray) -> bool:
        """检查指定位置是否可通行

        Args:
            pos: [x, y] 坐标

        Returns:
            True if walkable (空地/树), False otherwise
        """
        x, y = int(pos[0]), int(pos[1])
        if x < 0 or x >= 13 or y < 0 or y >= 13:
            return False
        terrain = self._grid_terrain[y, x]
        return terrain == 0 or terrain == 4

    def _get_obs(self) -> np.ndarray:
        """构建多通道观察张量

        Channel layout (9 channels):
          0-4: 地形 one-hot (empty/brick/water/tree/steel)
          5: 基地位置
          6: 玩家 x (归一化)
          7: 玩家 y (归一化)
          8: 玩家方向 (0=UP, 0.25=DOWN, 0.5=LEFT, 0.75=RIGHT, 1.0=IDLE)
        """
        obs = np.zeros((13, 13, 9), dtype=np.float32)

        for y in range(13):
            for x in range(13):
                t = self._grid_terrain[y, x]
                if 0 <= t <= 4:
                    obs[y, x, t] = 1.0

        if self._base_alive:
            bx, by = self._base_pos
            obs[by, bx, 5] = 1.0

        if self._player_alive:
            px, py = self._player_pos
            obs[py, px, 6] = px / 12.0
            obs[py, px, 7] = py / 12.0

        if self._player_alive:
            dir_map = {
                (0, -1): 0.0,
                (0, 1): 0.25,
                (-1, 0): 0.5,
                (1, 0): 0.75,
            }
            py, px = int(self._player_pos[1]), int(self._player_pos[0])
            dir_val = dir_map.get(tuple(self._player_dir), 1.0)
            obs[py, px, 8] = dir_val

        return obs

    def _get_info(self) -> dict:
        """返回环境诊断信息"""
        enemies_alive = sum(1 for e in self._enemies if e[2])
        return {
            'enemies_killed': self._enemies_killed,
            'enemies_total': self._enemies_total,
            'enemies_alive': enemies_alive,
            'base_alive': self._base_alive,
            'player_alive': self._player_alive,
            'wave': self._wave,
        }

    def render(self):
        """渲染当前帧 (ASCII 文本模式)"""
        grid = [['.' for _ in range(13)] for _ in range(13)]

        terrain_chars = {0: '.', 2: '#', 3: '~', 4: 'T', 5: '@'}
        for y in range(13):
            for x in range(13):
                grid[y][x] = terrain_chars.get(self._grid_terrain[y, x], '?')

        if self._base_alive:
            bx, by = self._base_pos
            grid[by][bx] = 'B'

        if self._player_alive:
            px, py = int(self._player_pos[0]), int(self._player_pos[1])
            grid[py][px] = 'P'

        for enemy in self._enemies:
            if enemy[2]:
                ex, ey = int(enemy[0][0]), int(enemy[0][1])
                grid[ey][ex] = 'E'

        for b in self._bullets:
            bx, by = int(b[0]), int(b[1])
            if 0 <= bx < 13 and 0 <= by < 13:
                grid[by][bx] = '*'

        rows = [''.join(row) for row in grid]
        return '\n'.join(rows)
