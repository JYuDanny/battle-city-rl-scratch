"""tirinox 真实游戏 Gymnasium 包装器

包装 tirinox/pybattlecity 游戏引擎为标准 Gymnasium Env:
  - Observation: 84×84×3 RGB 像素帧 (resized from 540×480)
  - Action: Discrete(10) — 5方向 × 射击/不射击
  - Frame Skip 4: 每 4 帧执行一次动作, 取最后一帧
  - Reward: 基于游戏内置评分系统 + 生存奖励
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # 无头模式, 不需要窗口

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'extern', 'pybattlecity'))

import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
import cv2
from collections import deque

# 必须在 pygame.init() 之后导入 (tirinox 在模块加载时读取 atlas.png)
pygame.init()
pygame.display.set_mode((1, 1))  # 无头模式下仍需 display surface 支持 convert()

# tirinox 使用相对路径加载 data/ 资源, 需要切换到其目录
_PYBATTLECITY_DIR = os.path.join(os.path.dirname(__file__), '..', 'extern', 'pybattlecity')

from game import Game as TirinoxGame
from util import Direction
from config import GAME_WIDTH, GAME_HEIGHT


class TirEnv(gymnasium.Env):
    """Battle City 真实游戏环境

    包装 tirinox 原版游戏引擎, 提供标准 RL 接口。

    Observation: 84×84×3 uint8 像素帧
    Action: Discrete(10) — 0↑ 1↓ 2← 3→ 4IDLE 5↑+F 6↓+F 7←+F 8→+F 9IDLE+F
    Frame Skip: 4 (每 4 帧重复一个动作)
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 15}

    def __init__(self, render_mode: str | None = None, fps: int = 60,
                 frame_skip: int = 4, max_episode_steps: int = 5000):
        """初始化环境

        Args:
            render_mode: 'rgb_array' 或 None
            fps: 游戏模拟帧率 (影响计时器行为)
            frame_skip: 动作重复帧数
            max_episode_steps: 单局最大步数
        """
        super().__init__()

        self.fps = fps
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(10)

        self.render_mode = render_mode
        self._game = None
        self._clock = pygame.time.Clock()
        self._render_surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self._step_count = 0
        self._prev_score = 0
        self._prev_lives = 3

    def reset(self, seed: int | None = None, options: dict | None = None):
        """重置游戏到初始状态

        Returns:
            obs: 84×84×3 RGB 像素帧
            info: 游戏状态字典
        """
        super().reset(seed=seed)

        # tirinox 使用相对路径加载 data/ 资源, 需要切换到其目录
        _prev_cwd = os.getcwd()
        os.chdir(_PYBATTLECITY_DIR)
        try:
            self._game = TirinoxGame()
        finally:
            os.chdir(_prev_cwd)
        self._step_count = 0
        self._prev_score = 0
        self._prev_lives = 3

        # 跳过多余的初始帧让画面稳定
        for _ in range(4):
            self._game.update()
            self._clock.tick(self.fps)

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """执行一步环境交互

        Action 映射 (Discrete 10):
          0: UP,    1: DOWN,  2: LEFT,  3: RIGHT,  4: IDLE
          5: UP+F,  6: DOWN+F, 7: LEFT+F, 8: RIGHT+F, 9: IDLE+F

        Args:
            action: 0-9 的动作索引

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        self._step_count += 1

        # 解码动作
        direction, fire = self._decode_action(action)

        # 收集帧 (用于 frame skip)
        frames = []
        for _ in range(self.frame_skip):
            self._game.my_tank_move_to_direction = direction
            if fire:
                self._game.fire()

            self._game.update()
            self._clock.tick(self.fps)

            if self.render_mode == 'rgb_array':
                self._game.render(self._render_surface)
                frame = pygame.surfarray.array3d(self._render_surface)
                frame = np.transpose(frame, (1, 0, 2))  # (H, W, 3)
                frames.append(frame)

        # 取最后一帧作为观察
        if frames:
            obs_raw = frames[-1]
        else:
            self._game.render(self._render_surface)
            obs_raw = np.transpose(pygame.surfarray.array3d(self._render_surface), (1, 0, 2))

        obs = self._resize_obs(obs_raw)

        # 奖励计算: 基于游戏内置评分
        current_score = self._game.score
        reward = (current_score - self._prev_score) / 100.0  # 归一化到 ~1.0/击杀
        self._prev_score = current_score

        # 生存奖励 (每步微小正向信号)
        reward += 0.005

        # 基地被毁惩罚
        if self._game.my_base.broken:
            reward -= 20.0

        # 玩家死亡 (通过分数不变 + my_tank 为 None 判断)
        # tirinox 在玩家死亡时 my_tank 仍然存在但 is_spawning=True
        if self._game.my_tank.is_spawning and self._prev_lives > 0:
            # 无法直接获取 lives, 通过 spawning 状态判断
            pass

        terminated = self._game.is_game_over
        truncated = self._step_count >= self.max_episode_steps

        # 通关奖励
        if terminated and not self._game.my_base.broken:
            reward += 10.0

        return obs, reward, terminated, truncated, self._get_info()

    def _decode_action(self, action: int) -> tuple:
        """解析动作索引为方向和射击指令

        Args:
            action: 0-9

        Returns:
            (direction: Direction | None, fire: bool)
        """
        dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT, None]
        fire = action >= 5
        idx = action % 5
        return dirs[idx], fire

    def _get_obs(self) -> np.ndarray:
        """获取当前观察帧"""
        self._game.render(self._render_surface)
        frame = np.transpose(pygame.surfarray.array3d(self._render_surface), (1, 0, 2))
        return self._resize_obs(frame)

    def _resize_obs(self, frame: np.ndarray) -> np.ndarray:
        """将 480×540×3 缩放到 84×84×3"""
        resized = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def _get_info(self) -> dict:
        """获取游戏诊断信息"""
        g = self._game
        enemies_alive = sum(1 for e in g.ai.all_enemies if not e.to_destroy)
        return {
            'score': g.score,
            'base_alive': not g.my_base.broken,
            'enemies_alive': enemies_alive,
            'player_spawning': g.my_tank.is_spawning,
            'step_count': self._step_count,
        }

    def render(self):
        """渲染当前帧"""
        if self.render_mode == 'rgb_array':
            return self._get_obs()
        return None

    def close(self):
        """释放资源"""
        if self._game is not None:
            del self._game
            self._game = None
