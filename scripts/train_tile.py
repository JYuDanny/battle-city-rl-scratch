"""简化 Tile 环境训练入口

使用简化 13×13 tile 环境快速验证 PPO 训练流程。
"""

import argparse
from rl.config import Config
from envs.tile_env import TileEnv
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="训练 Battle City RL Agent (Tile 环境)")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="总训练步数 (默认 10M)")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率 (默认 3e-4)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--force-stage", type=int, default=None,
                        help="手动设置 curriculum stage (1-3)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="训练设备 (默认 auto)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="恢复训练的 checkpoint 路径")
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

    print("=" * 50)
    print("Battle City RL 训练框架 — Tile 环境")
    print(f"设备: {config.training.device}")
    print(f"学习率: {config.ppo.lr}")
    print(f"总步数: {config.training.total_timesteps:,}")
    print(f"Seed: {config.training.seed}")
    print("=" * 50)

    env = TileEnv()
    trainer = Trainer(config, env)

    if args.force_stage is not None:
        trainer.curriculum.force_stage(args.force_stage)
        print(f"[手动] Curriculum Stage 设为 {args.force_stage}")

    if args.ckpt is not None:
        trainer.load_checkpoint(args.ckpt)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[Trainer] 用户中断训练, 正在保存 checkpoint...")
        trainer.save_checkpoint()


if __name__ == "__main__":
    main()
