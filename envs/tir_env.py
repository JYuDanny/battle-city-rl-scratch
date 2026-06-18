"""tirinox 真实游戏 Gymnasium 包装器

包装 tirinox/pybattlecity 游戏引擎为标准 Gymnasium Env:
  - Observation: 84×84×3 RGB 像素帧 (resized from 540×480)
  - Action: Discrete(10) — 5方向 × 射击/不射击
  - Frame Skip 4: 每 4 帧执行一次动作, 取最后一帧
  - Reward: 基于游戏内置评分系统 + 生存奖励
  - render_mode='human': 弹出 pygame 窗口实时显示游戏
  - render_mode='rgb_array': 支持视频录制
"""

import os
import sys
import warnings

# pygame 内部使用已弃用的 pkg_resources, 无害警告, 静默处理
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'extern', 'pybattlecity'))

import gymnasium
from gymnasium import spaces
import numpy as np
import cv2

# pygame 延迟初始化 — SDL_VIDEODRIVER 需在 pygame.init() 之前设置
import pygame

# tirinox 模块延迟加载 (需要 pygame.init() 后才能安全导入)
_TIRINOX_LOADED = False
_TirinoxGame = None
_Direction = None
_GAME_WIDTH = None
_GAME_HEIGHT = None

_PYBATTLECITY_DIR = os.path.join(os.path.dirname(__file__), '..', 'extern', 'pybattlecity')
_PYGAME_DRIVER_SET = False


def _ensure_tirinox_loaded(render_mode: str | None = None):
    """延迟加载 tirinox 模块

    SDL_VIDEODRIVER 必须在 pygame.init() 之前设置。
    无头模式用 'dummy', 窗口模式保持默认。
    """
    global _TIRINOX_LOADED, _TirinoxGame, _Direction, _GAME_WIDTH, _GAME_HEIGHT, _PYGAME_DRIVER_SET

    if _TIRINOX_LOADED:
        return

    if not _PYGAME_DRIVER_SET:
        if render_mode == 'human':
            pass  # 窗口模式: 不设置 SDL_VIDEODRIVER, 使用系统默认显示驱动
        else:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        _PYGAME_DRIVER_SET = True

    pygame.init()

    # tirinox 加载 spritesheet 时需要 display surface 支持 .convert()
    # 窗口模式在 __init__ 中创建窗口; 无头模式创建 1×1 占位 surface
    if render_mode != 'human':
        pygame.display.set_mode((1, 1))

    # tirinox 使用相对路径加载 data/ 资源, 需切换到其目录后导入
    _saved_cwd = os.getcwd()
    os.chdir(_PYBATTLECITY_DIR)
    try:
        from game import Game as _TirinoxGame
        from util import Direction as _Direction
        from config import GAME_WIDTH as _GAME_WIDTH, GAME_HEIGHT as _GAME_HEIGHT
    finally:
        os.chdir(_saved_cwd)

    # 修补: tirinox 的 BonusType.GUN 未实现, 排除它防止拾取时报错
    import bonus as _bonus_mod

    @classmethod
    def _safe_random(cls):
        import random as _rnd
        safe = [bt for bt in cls if bt != cls.GUN]
        return _rnd.choice(safe)

    _bonus_mod.BonusType.random = _safe_random

    _TIRINOX_LOADED = True


class TirEnv(gymnasium.Env):
    """Battle City 真实游戏环境

    包装 tirinox 原版游戏引擎, 提供标准 RL 接口。

    Observation: 84×84×3 uint8 像素帧
    Action: Discrete(10) — 0↑ 1↓ 2← 3→ 4IDLE 5↑+F 6↓+F 7←+F 8→+F 9IDLE+F
    Frame Skip: 4 (每 4 帧重复一个动作)
    Render modes: 'human' (pygame 窗口), 'rgb_array' (返回帧数组, 用于视频录制)
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 15}

    def __init__(self, render_mode: str | None = None, fps: int = 60,
                 frame_skip: int = 4, max_episode_steps: int = 5000):
        """初始化环境

        Args:
            render_mode: 'human' (窗口), 'rgb_array' (帧数组), None (无渲染)
            fps: 游戏模拟帧率 (影响计时器行为)
            frame_skip: 动作重复帧数
            max_episode_steps: 单局最大步数
        """
        super().__init__()

        # 首次实例化时加载 tirinox, SDL 驱动根据 render_mode 设置
        _ensure_tirinox_loaded(render_mode)

        self.fps = fps
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(10)

        self.render_mode = render_mode
        self._game = None
        self._clock = None  # 延迟创建 (pygame.init 后才能用)
        self._render_surface = None
        self._display_surface = None
        self._step_count = 0
        self._prev_score = 0

    def _ensure_game_surfaces(self):
        """创建渲染 surface 和可选显示窗口"""
        if self._render_surface is not None:
            return

        self._render_surface = pygame.Surface((_GAME_WIDTH, _GAME_HEIGHT))
        self._clock = pygame.time.Clock()

        if self.render_mode == 'human':
            # 替换 _ensure_tirinox_loaded 中创建的 1×1 占位窗口
            self._display_surface = pygame.display.set_mode(
                (_GAME_WIDTH, _GAME_HEIGHT))
            pygame.display.set_caption("Battle City — Agent 评估")

    def reset(self, seed: int | None = None, options: dict | None = None):
        """重置游戏到初始状态

        Returns:
            obs: 84×84×3 RGB 像素帧
            info: 游戏状态字典
        """
        super().reset(seed=seed)

        self._ensure_game_surfaces()

        # tirinox 使用相对路径加载 data/ 资源
        _prev_cwd = os.getcwd()
        os.chdir(_PYBATTLECITY_DIR)
        try:
            self._game = _TirinoxGame()
        finally:
            os.chdir(_prev_cwd)

        self._step_count = 0
        self._prev_score = 0

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
        direction, fire = self._decode_action(action)

        frames = []
        for _ in range(self.frame_skip):
            self._game.my_tank_move_to_direction = direction
            if fire:
                self._game.fire()

            self._game.update()
            self._clock.tick(self.fps)

            if self.render_mode in ('rgb_array', 'human'):
                self._game.render(self._render_surface)

                if self.render_mode == 'human':
                    self._display_surface.blit(self._render_surface, (0, 0))
                    pygame.display.flip()

                if self.render_mode == 'rgb_array':
                    frame = pygame.surfarray.array3d(self._render_surface)
                    frames.append(np.transpose(frame, (1, 0, 2)))

        # 取最后一帧作为观察
        if frames:
            obs_raw = frames[-1]
        else:
            self._game.render(self._render_surface)
            obs_raw = np.transpose(pygame.surfarray.array3d(self._render_surface), (1, 0, 2))

        obs = self._resize_obs(obs_raw)

        # 奖励计算: 基于游戏内置评分
        current_score = self._game.score
        reward = (current_score - self._prev_score) / 100.0
        self._prev_score = current_score

        reward += 0.005  # 生存奖励

        terminated = self._game.is_game_over
        truncated = self._step_count >= self.max_episode_steps

        if terminated and not self._game.my_base.broken:
            reward += 10.0  # 通关奖励

        return obs, reward, terminated, truncated, self._get_info()

    def _decode_action(self, action: int) -> tuple:
        """解析动作索引为方向和射击指令"""
        dirs = [_Direction.UP, _Direction.DOWN, _Direction.LEFT, _Direction.RIGHT, None]
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
        """渲染当前帧

        Returns:
            rgb_array 模式下返回 (84, 84, 3) numpy 数组
        """
        if self.render_mode == 'rgb_array':
            return self._get_obs()
        return None

    def close(self):
        """释放资源"""
        if self._game is not None:
            del self._game
            self._game = None
        if self._display_surface is not None or self.render_mode == 'rgb_array':
            try:
                pygame.display.quit()
            except Exception:
                pass
