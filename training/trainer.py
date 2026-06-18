"""训练主循环

编排 env ↔ PPO 的交互流程:
  1. 收集 rollout (env.step × N)
  2. GAE 计算 (buffer.compute_gae)
  3. PPO update (K epochs × mini-batches)
  4. TensorBoard 日志 + checkpoint
  5. Curriculum 评估
"""

import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rl.config import Config
from rl.network import ActorCritic
from rl.buffer import RolloutBuffer
from rl.ppo import PPOTrainer
from training.curriculum import CurriculumManager


class Trainer:
    """RL 训练编排器

    管理完整的训练生命周期:
      - 初始化网络、优化器、缓冲区
      - 执行 rollout 收集与 PPO 更新循环
      - 记录 TensorBoard 日志
      - 管理 checkpoint 和 curriculum
    """

    def __init__(self, config: Config, env):
        """初始化训练器

        Args:
            config: 全局配置
            env: Gymnasium Env 实例
        """
        self.config = config
        self.env = env

        training_cfg = config.training
        self.total_timesteps = training_cfg.total_timesteps
        self.checkpoint_interval = training_cfg.checkpoint_interval
        self.log_interval = training_cfg.log_interval
        self.save_dir = training_cfg.save_dir
        self.log_dir = training_cfg.log_dir
        self.seed = training_cfg.seed
        self.device = training_cfg.device

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs("crash_log", exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.net = ActorCritic(config).to(self.device)
        self.trainer = PPOTrainer(config)
        self.curriculum = CurriculumManager(config)

        self.buffer = RolloutBuffer(config)

        self._timestep = 0
        self._best_reward = float('-inf')

    def train(self):
        """主训练循环

        执行完整的 PPO 训练流程直到达到 total_timesteps。
        每步: 收集 rollout → GAE → PPO update → log → curriculum check
        """
        config = self.config
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_count = 0

        if self.device == "cuda":
            print(f"[Trainer] 使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[Trainer] 使用 CPU 训练")

        print(f"[Trainer] 开始训练, 总步数目标: {self.total_timesteps:,}")

        while self._timestep < self.total_timesteps:
            for step in range(config.ppo.rollout_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action, log_prob, value = self.net.get_action(obs_tensor)

                action_idx = action.item()
                next_obs, reward, terminated, truncated, info = self.env.step(action_idx)

                self.buffer.store(
                    obs=torch.tensor(obs, dtype=torch.float32),
                    action=action.cpu().squeeze(),
                    reward=reward,
                    value=value.cpu().squeeze(),
                    log_prob=log_prob.cpu().squeeze(),
                    done=terminated or truncated,
                )

                episode_reward += reward
                episode_length += 1
                self._timestep += 1
                obs = next_obs

                if terminated or truncated:
                    won = bool(info.get('enemies_alive', 1) == 0 and info.get('player_alive', True))
                    self.curriculum.record_episode(
                        won=won,
                        enemies_killed=info.get('enemies_killed', 0),
                        total_enemies=info.get('enemies_total', 1),
                    )
                    episode_count += 1

                    if episode_count % 10 == 0:
                        print(f"[Ep {episode_count}] "
                          f"step={self._timestep:,} "
                          f"reward={episode_reward:.1f} "
                          f"len={episode_length} "
                          f"diff={self.curriculum.current_difficulty:.1f}")

                    self.writer.add_scalar("Episode/Reward", episode_reward, self._timestep)
                    self.writer.add_scalar("Episode/Length", episode_length, self._timestep)

                    episode_reward = 0.0
                    episode_length = 0
                    obs, _ = self.env.reset()

                if self._timestep >= self.total_timesteps:
                    break

            if self._timestep >= self.total_timesteps:
                n_stored = len(self.buffer.rewards)
                if n_stored > 0:
                    with torch.no_grad():
                        last_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                        _, _, last_value = self.net.get_action(last_obs)
                        self._compute_gae_partial(last_value.cpu().squeeze(), n_stored)

                    avg_loss = self.trainer.update(self.net, self.buffer)
                    self.writer.add_scalar("Train/Loss", avg_loss, self._timestep)

                self.buffer.reset()
                break

            with torch.no_grad():
                last_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                _, _, last_value = self.net.get_action(last_obs)
                self.buffer.compute_gae(last_value.cpu())

            start_time = time.time()
            avg_loss = self.trainer.update(self.net, self.buffer)
            update_time = time.time() - start_time

            self.writer.add_scalar("Train/Loss", avg_loss, self._timestep)
            self.writer.add_scalar("Train/UpdateTime", update_time, self._timestep)

            if self._timestep % self.log_interval < config.ppo.rollout_steps:
                wr = self.curriculum.win_rate()
                self.writer.add_scalar("Curriculum/WinRate", wr, self._timestep)
                self.writer.add_scalar("Curriculum/Difficulty",
                                       self.curriculum.current_difficulty, self._timestep)

            changed, new_stage = self.curriculum.check_and_update()
            if changed:
                print(f"[Curriculum] Stage 变更: → {new_stage}")

            if self._timestep % self.checkpoint_interval < config.ppo.rollout_steps:
                self.save_checkpoint()

            self.buffer.reset()

        self.writer.close()
        self.save_checkpoint()
        print(f"[Trainer] 训练完成! 总步数: {self._timestep:,}")

    def _compute_gae_partial(self, last_value: torch.Tensor, n_steps: int):
        """对不完整的缓冲区手动计算 GAE (训练结束时使用)"""
        if n_steps == 0:
            return
        values = torch.stack([self.buffer.values[i].squeeze() for i in range(n_steps)])
        adv = torch.zeros(n_steps, dtype=torch.float32)

        gae = 0.0
        gamma = self.buffer.gamma
        gae_lambda = self.buffer.gae_lambda
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = last_value
            else:
                next_val = values[t + 1]

            mask = 1.0 - float(self.buffer.dones[t])
            delta = self.buffer.rewards[t] + gamma * next_val * mask - values[t].item()
            gae = delta + gamma * gae_lambda * mask * gae
            adv[t] = gae

        if n_steps > 1:
            adv_mean = adv.mean()
            adv_std = adv.std(unbiased=False) + 1e-8
            self.buffer.advantages = (adv - adv_mean) / adv_std
        else:
            self.buffer.advantages = adv

        self.buffer.returns = (adv + values).detach()

    def save_checkpoint(self):
        """保存模型 checkpoint"""
        path = os.path.join(self.save_dir, f"checkpoint_{self._timestep}.pt")
        torch.save({
            'timestep': self._timestep,
            'network_state_dict': self.net.state_dict(),
            'difficulty': self.curriculum.current_difficulty,
        }, path)
        print(f"[Checkpoint] 已保存: {path}")

    def load_checkpoint(self, path: str):
        """加载模型 checkpoint

        Args:
            path: checkpoint 文件路径
        """
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['network_state_dict'])
        self._timestep = ckpt.get('timestep', 0)
        self.curriculum.current_difficulty = ckpt.get('difficulty', 0.0)
        print(f"[Checkpoint] 已加载: {path} (timestep={self._timestep})")
