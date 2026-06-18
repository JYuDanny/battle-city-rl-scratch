"""评估脚本

加载训练好的模型, 运行指定 episode 并可选录制视频。
"""

import argparse
import os
import time
import torch
from rl.config import Config
from rl.network import ActorCritic
from envs.tir_env import TirEnv


def main():
    parser = argparse.ArgumentParser(description="评估训练好的 Battle City Agent")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--episodes", type=int, default=10,
                        help="评估 episode 数")
    parser.add_argument("--deterministic", action="store_true",
                        help="是否使用确定性策略 (取最高概率动作)")
    parser.add_argument("--fps", type=int, default=60,
                        help="游戏模拟帧率")
    args = parser.parse_args()

    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = ActorCritic(config).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt['network_state_dict'])
    net.eval()

    env = TirEnv(fps=args.fps)
    rewards = []
    wins = 0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                action, _, _ = net.get_action(obs_tensor, deterministic=args.deterministic)

            obs, reward, terminated, truncated, info = env.step(action.item())
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        won = info.get('base_alive', False) and info.get('enemies_alive', 99) == 0
        if won:
            wins += 1
        print(f"[Ep {ep+1}/{args.episodes}] Reward: {ep_reward:.1f}  "
              f"Score: {info.get('score', 0)}  Win: {won}")

    env.close()
    print(f"\n平均 Reward: {sum(rewards)/len(rewards):.1f}")
    print(f"通关率: {wins}/{args.episodes} ({wins/args.episodes*100:.1f}%)")


if __name__ == "__main__":
    main()
