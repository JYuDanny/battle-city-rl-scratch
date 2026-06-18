"""tirinox 真实环境训练入口

使用 tirinox/pybattlecity 真实游戏引擎进行 PPO 训练。
"""

import argparse
import os
from datetime import datetime
from rl.config import Config
from envs.tir_env import TirEnv
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="训练 Battle City RL Agent (tirinox 环境)")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="总训练步数 (默认 10M)")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率 (默认 1e-4)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--device", type=str, default="auto",
                         choices=["auto", "cuda", "mps", "cpu"],
                        help="训练设备 (默认 auto)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="恢复训练的 checkpoint 路径")
    parser.add_argument("--run-name", type=str, default=None,
                        help="训练运行名称, checkpoint 和日志保存到子目录")
    parser.add_argument("--fps", type=int, default=60,
                        help="游戏模拟帧率 (默认 60, 影响游戏计时器)")
    parser.add_argument("--frame-skip", type=int, default=4,
                        help="帧跳过数 (默认 4, 每 4 帧执行一次动作)")
    args = parser.parse_args()

    config = Config()

    if args.total_steps is not None:
        config.training.total_timesteps = args.total_steps
    if args.lr is not None:
        config.ppo.lr = args.lr
    if args.seed is not None:
        config.training.seed = args.seed
    if args.device is not None:
        config.training.device = args.device

    # 按日期+运行名组织 checkpoint 和 TensorBoard 目录 (绝对路径, 防止 cwd 变更导致失效)
    if args.run_name:
        date_str = datetime.now().strftime("%Y-%m-%d")
        run_dir = f"{date_str}_{args.run_name}"
        config.training.save_dir = os.path.abspath(os.path.join("checkpoints", run_dir))
        config.training.log_dir = os.path.abspath(os.path.join("runs", run_dir))

    print("=" * 50)
    print("Battle City RL 训练框架 — tirinox 真实环境")
    print(f"设备: {config.training.device}")
    print(f"学习率: {config.ppo.lr}")
    print(f"总步数: {config.training.total_timesteps:,}")
    print(f"Seed: {config.training.seed}")
    print(f"保存目录: {config.training.save_dir}")
    print(f"日志目录: {config.training.log_dir}")
    print(f"FPS: {args.fps}, Frame Skip: {args.frame_skip}")
    print("=" * 50)

    env = TirEnv(fps=args.fps, frame_skip=args.frame_skip)
    trainer = Trainer(config, env)

    if args.ckpt is not None:
        trainer.load_checkpoint(args.ckpt)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[Trainer] 用户中断训练, 正在保存 checkpoint...")
        trainer.save_checkpoint()


if __name__ == "__main__":
    main()
