"""Actor-Critic 网络模型

采用 Conv2D backbone 架构保留空间信息:
  Observation [B, 13, 13, C] → Conv2D [32, 64] → Flatten → FC(512) → ReLU
                                  ├── Actor Head → action_distribution
                                  └── Critic Head → state_value

相比旧版 Flatten→MLP, Conv2D 保留了"敌人在左边"这类空间关系,
使 agent 能学到位置感知的策略。
"""

import torch
import torch.nn as nn
import numpy as np
from rl.config import Config


class ActorCritic(nn.Module):
    """Actor-Critic 共享网络

    Conv2D backbone 提取空间特征, 分叉出策略头(Actor)和价值头(Critic)。

    输入: 环境观察 [batch_size, tile_h, tile_w, channels]
    输出: (action_logits [batch_size, action_size], state_value [batch_size, 1])
    """

    def __init__(self, config: Config):
        """初始化网络

        Args:
            config: 包含 network 和 env 模块的全局配置
        """
        super().__init__()

        env_cfg = config.env
        net_cfg = config.network

        # NOTE: obs_channels 硬编码为 9, 未来重构将迁移至 EnvConfig 中统一管理
        # 地形(5) + 实体(3) + 方向(1)
        obs_channels = 9

        # Conv2D backbone: 保留空间结构的卷积层
        conv_layers = []
        in_channels = obs_channels
        for out_channels in net_cfg.conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            ])
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers) if conv_layers else nn.Identity()

        # 计算展平后维度
        if conv_layers:
            flattened_dim = in_channels * env_cfg.tile_size[0] * env_cfg.tile_size[1]
        else:
            flattened_dim = obs_channels * env_cfg.tile_size[0] * env_cfg.tile_size[1]

        # 共享 FC 层
        fc_layers = []
        prev_dim = flattened_dim
        for hidden_dim in net_cfg.share_mlp_sizes:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        self.share_fc = nn.Sequential(*fc_layers) if fc_layers else nn.Identity()

        # Actor Head: 输出每个动作的 logit
        self.actor_head = nn.Linear(prev_dim, env_cfg.action_size)

        # Critic Head: 输出状态价值标量
        self.critic_head = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """正交初始化权重, 偏置初始化为 0

        对 Linear 层使用正交初始化, Conv2d 保持默认 Kaiming 初始化。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            obs: 环境观察张量 [batch_size, H, W, C] 浮点型

        Returns:
            action_logits: [batch_size, action_size] 各动作的未归一化 logits
            state_value: [batch_size, 1] 状态价值估计
        """
        # 将 [B, H, W, C] 转换为 [B, C, H, W] 供 Conv2D 使用
        obs = obs.permute(0, 3, 1, 2)

        features = self.conv(obs)
        features = torch.flatten(features, start_dim=1)
        shared = self.share_fc(features)

        action_logits = self.actor_head(shared)
        state_value = self.critic_head(shared)
        return action_logits, state_value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """根据观察采样动作

        Args:
            obs: 单个观察 [1, H, W, C] 或批次
            deterministic: True 时取 argmax, False 时按分布采样

        Returns:
            action: 选中的动作索引
            log_prob: 该动作的对数概率
            value: 状态价值估计
        """
        logits, value = self.forward(obs)
        probs = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        return action, log_prob, value
