"""Actor-Critic 网络模型

采用共享 backbone 架构:
  Observation [B, 13, 13, C] → Flatten → Embedding → ShareMLP
                                  ├── Actor Head → action_distribution
                                  └── Critic Head → state_value

后续升级到像素输入时, ShareMLP 替换为 Conv2D backbone。
"""

import torch
import torch.nn as nn
import numpy as np
from rl.config import Config


class ActorCritic(nn.Module):
    """Actor-Critic 共享网络

    共享底层特征提取, 分叉出策略头(Actor)和价值头(Critic)。
    这种设计减少了参数总量, 且鼓励两个任务学到互补的表征。

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

        # 计算展平后的输入维度: tile_H × tile_W × channels
        # NOTE: obs_channels 硬编码为 9, 未来重构将迁移至 EnvConfig 中统一管理
        obs_channels = 9  # 地形(5) + 实体(3) + 方向(1)
        input_dim = env_cfg.tile_size[0] * env_cfg.tile_size[1] * obs_channels

        # 将 tile grid 展平后映射到嵌入空间
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, net_cfg.tile_embedding_dim),
            nn.ReLU(),
        )

        # 共享 MLP backbone
        share_layers = []
        prev_dim = net_cfg.tile_embedding_dim
        for hidden_dim in net_cfg.share_mlp_sizes:
            share_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        self.share_mlp = nn.Sequential(*share_layers)

        # Actor Head: 输出每个动作的 logit
        self.actor_head = nn.Linear(prev_dim, env_cfg.action_size)

        # Critic Head: 输出状态价值标量
        self.critic_head = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """正交初始化权重, 偏置初始化为 0

        正交初始化有助于保持梯度在前向/反向传播中的稳定性。
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
        embedded = self.embedding(obs)
        shared = self.share_mlp(embedded)
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
