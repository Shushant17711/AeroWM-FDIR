"""CLI entry point for joint world-model + TD-MPC2 training."""

from __future__ import annotations

import argparse

import torch

from configs.schema import TrainingConfig, WorldModelConfig
from train.trainer import WorldModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AeroWM-FDIR")
    parser.add_argument("--steps", type=int, default=None, help="total environment steps")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--demo-episodes", type=int, default=None,
                        help="PID demonstration episodes used to seed the replay buffer")
    parser.add_argument("--no-demos", action="store_true")
    args = parser.parse_args()

    training = TrainingConfig()
    if args.steps is not None or args.num_envs is not None or args.output_dir is not None:
        total = args.steps or training.total_env_steps
        training = TrainingConfig(
            total_env_steps=total,
            warmup_env_steps=min(training.warmup_env_steps, total),
            num_envs=args.num_envs or training.num_envs,
            output_dir=args.output_dir or training.output_dir,
        )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer = WorldModelTrainer(
        training=training,
        world_model=WorldModelConfig(),
        device=device,
        output_dir=training.output_dir,
        seed=args.seed,
    )
    if args.resume:
        trainer.load(args.resume)
    elif not args.no_demos:
        episodes = args.demo_episodes if args.demo_episodes is not None else training.pid_demo_episodes
        seeded = trainer.seed_with_pid(episodes, training.pid_demo_max_transitions)
        print(f"seeded replay buffer with {seeded} PID demonstration transitions")
    final = trainer.train()
    print("final metrics:", final)
    trainer.save(f"{training.output_dir}/checkpoint_final.pt")


if __name__ == "__main__":
    main()
