"""PPO 及训练统一超参数配置

集中管理所有超参数，避免散布在文件各处的魔数。
支持命令行覆写，方便调参实验。
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EnvConfig:
    """环境相关配置
    
    tile_size: 地图网格尺寸 (不含边界)
    action_size: 动作空间大小 [上, 下, 左, 右, 射击, 静止]
    max_steps: 单局最大步数限制, 防止无限循环
    """
    tile_size: Tuple[int, int] = (13, 13)
    action_size: int = 6
    max_steps: int = 2000


@dataclass
class PPOConfig:
    """PPO 算法超参数
    
    rollout_steps: 每次收集的交互步数, 累积后执行一次更新
    gamma: 折扣因子, 控制远期奖励的衰减
    gae_lambda: GAE 中的 λ 参数, 权衡偏差与方差
    clip_epsilon: PPO clip 范围, 限制策略更新的幅度
    k_epochs: 每批 rollout 数据重用的更新轮数
    mini_batch_size: 小批次梯度更新的样本数
    lr: 学习率
    entropy_coef: 熵正则化系数, 鼓励探索
    value_coef: 价值损失权重
    max_grad_norm: 梯度裁剪阈值, 防止梯度爆炸
    """
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    k_epochs: int = 10
    mini_batch_size: int = 64
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class NetworkConfig:
    """网络架构超参数
    
    tile_embedding_dim: tile 嵌入层输出维度
    share_mlp_sizes: 共享 MLP 隐藏层维度列表
    """
    tile_embedding_dim: int = 128
    share_mlp_sizes: Tuple[int, ...] = (256, 256)


@dataclass
class RewardConfig:
    """奖励函数权重
    
    分层奖励体系中各事件的奖励值。
    势能奖励基于到最近敌人的距离变化。
    """
    survival: float = 0.01
    kill_enemy: float = 2.0
    wave_clear: float = 5.0
    level_clear: float = 100.0
    death: float = -10.0
    base_destroyed: float = -50.0
    potential_scale: float = 0.01


@dataclass
class CurriculumConfig:
    """课程学习配置
    
    evaluation_interval: 每隔多少 episode 评估通关率
    evaluation_window: 计算通关率的滑动窗口大小
    promotion_threshold_stage1: S1→S2 通关率阈值
    promotion_threshold_stage2: S2→S3 通关率阈值
    demotion_threshold: 回退触发通关率阈值
    demotion_window: 回退观察窗口
    """
    evaluation_interval: int = 50
    evaluation_window: int = 100
    promotion_threshold_stage1: float = 0.8
    promotion_threshold_stage2: float = 0.5
    demotion_threshold: float = 0.2
    demotion_window: int = 50


@dataclass
class TrainingConfig:
    """训练流程配置
    
    total_timesteps: 总训练步数
    checkpoint_interval: 每隔多少 timestep 保存 checkpoint
    log_interval: 每隔多少 timestep 记录 TensorBoard
    save_dir: checkpoint 存储目录
    log_dir: TensorBoard 日志目录
    device: 训练设备 (auto 表示自动选择 cuda/cpu)
    seed: 随机种子, 确保可复现
    """
    total_timesteps: int = 10_000_000
    checkpoint_interval: int = 100_000
    log_interval: int = 1000
    save_dir: str = "checkpoints"
    log_dir: str = "runs"
    device: str = "auto"
    seed: int = 42


@dataclass
class Config:
    """总配置容器
    
    聚合所有子模块配置, 提供统一的参数访问入口。
    """
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def update_from_dict(self, overrides: dict):
        """根据字典覆写配置项, 支持点分隔键如 'ppo.lr=1e-4'"""
        for key, value in overrides.items():
            parts = key.split(".")
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
