"""Evaluate the cascade PID baseline across fault modes (tuning harness)."""

from __future__ import annotations

import argparse

import torch

from agents.pid_baseline import CascadePIDController
from configs.schema import FaultMode
from envs.rocket_vec_env import VectorizedRocketEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--adaptive", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedRocketEnv(num_envs=args.batch, device=device, seed=0)
    controller = CascadePIDController(device=device)

    for mode in [0, 1, 2, 3, 4]:
        env.reset(mode=mode, probability=1.0 if mode else 0.0, onset=40, severity=0.35)
        controller.reset(env.num_envs, device)
        success = crashes = 0
        obs = env._observation()
        for step in range(env.env_cfg.max_episode_steps):
            out = env.step(controller.act(obs))
            obs = out.obs
            term = out.terminated
            success += int((out.info["landing_success"] & term).sum())
            crashes += int((term & ~out.info["landing_success"]).sum())
            if success + crashes >= args.episodes:
                break
        total = max(1, success + crashes)
        print(
            f"{FaultMode(mode).name:14s} success={success:4d}/{total:4d} "
            f"({100.0 * success / total:5.1f}%) crashes={crashes}"
        )


if __name__ == "__main__":
    main()
