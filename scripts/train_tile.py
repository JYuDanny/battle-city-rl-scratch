"""简化 Tile 环境训练入口

使用简化 13×13 tile 环境快速验证 PPO 训练流程。
"""

import argparse
import os
from datetime import datetime
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
    parser.add_argument("--run-name", type=str, default=None,
                        help="训练运行名称, checkpoint 保存到 checkpoints/<date>_<name>/ 目录下")
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

    # 按日期+运行名组织 checkpoint 目录
    if args.run_name:
        date_str = datetime.now().strftime("%Y-%m-%d")
        config.training.save_dir = os.path.join("checkpoints", f"{date_str}_{args.run_name}")

    print("=" * 50)
    print("Battle City RL 训练框架 — Tile 环境")
    print(f"设备: {config.training.device}")
    print(f"学习率: {config.ppo.lr}")
    print(f"总步数: {config.training.total_timesteps:,}")
    print(f"Seed: {config.training.seed}")
    print(f"保存目录: {config.training.save_dir}")
    print("=" * 50)

    env = TileEnv()
    trainer = Trainer(config, env)

    if args.force_stage is not None:
        # 将旧 stage 参数转换为 difficulty: S1=0.0, S2=0.5, S3=1.0
        diff_map = {1: 0.0, 2: 0.5, 3: 1.0}
        diff = diff_map.get(args.force_stage, 0.0)
        trainer.curriculum.force_difficulty(diff)
        print(f"[手动] Curriculum Difficulty 设为 {diff:.1f} (stage {trainer.curriculum.current_stage})")

    if args.ckpt is not None:
        trainer.load_checkpoint(args.ckpt)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[Trainer] 用户中断训练, 正在保存 checkpoint...")
        trainer.save_checkpoint()


if __name__ == "__main__":
    main()
