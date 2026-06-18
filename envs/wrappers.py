"""Gymnasium 环境包装器

提供 RecordVideo 等通用包装器, 用于训练后录制 agent 表现视频。
"""

import gymnasium
from gymnasium.wrappers import RecordVideo
import os


def make_video_env(env, video_dir: str = "videos", episode_trigger=None):
    """创建可录制视频的环境

    Args:
        env: 原始 Gymnasium Env
        video_dir: 视频输出目录
        episode_trigger: 触发录制的条件 (默认每 10 episode 录制一次)

    Returns:
        包装后的视频录制环境
    """
    if episode_trigger is None:
        episode_trigger = lambda ep: ep % 10 == 0

    os.makedirs(video_dir, exist_ok=True)
    return RecordVideo(env, video_dir, episode_trigger=episode_trigger)
