"""评估脚本

加载训练好的模型, 运行指定 episode。
支持三种模式:
  --render: 弹出 pygame 窗口实时观看 agent 操作
  --record: 录制视频保存到 videos/ 目录
  不加参数: 仅输出统计信息 (最快)
"""

import argparse
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
    parser.add_argument("--render", action="store_true",
                        help="弹出 pygame 窗口实时显示游戏画面")
    parser.add_argument("--record", action="store_true",
                        help="录制视频保存到 videos/ 目录")
    parser.add_argument("--fps", type=int, default=60,
                        help="游戏模拟帧率 (默认 60, 值越大模拟越快)")
    args = parser.parse_args()

    config = Config()
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    net = ActorCritic(config).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt['network_state_dict'])
    net.eval()

    # 根据参数选择渲染模式
    if args.render:
        render_mode = 'human'
    elif args.record:
        render_mode = 'rgb_array'
    else:
        render_mode = None

    env = TirEnv(render_mode=render_mode, fps=args.fps)

    # 视频录制包装器
    if args.record:
        from envs.wrappers import make_video_env
        env = make_video_env(env)

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
