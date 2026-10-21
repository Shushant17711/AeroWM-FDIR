"""Measure the frozen-gimbal deflection at the onset vs. the landing outcome.

A seized TVC freezes at whatever angle it happened to hold when the fault
triggered, so a freeze close to the vertical is a *benign* fault: the vehicle
keeps descending on a nearly vertical thrust vector and lands next to the
pad.  This script measures how many of the ``GIMBAL_STUCK`` episodes of a
given protocol are benign and how many of them the classical law still lands,
which is what makes the baseline's ``GIMBAL_STUCK`` rate sample-dependent.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/stuck_freeze_angle.py --episodes 20
"""

from __future__ import annotations

import argparse
import math

import torch

from agents.pid_baseline import CascadePIDController
from configs.schema import FaultMode
from envs.rocket_vec_env import VectorizedRocketEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze angle vs. landing outcome")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--onset", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benign-deg", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedRocketEnv(num_envs=args.envs, device=device, seed=args.seed)
    mode = int(FaultMode.GIMBAL_STUCK)
    env.fixed_fault_mode = mode
    controller = CascadePIDController(device=device)

    rows = []
    attempts = 0
    while len(rows) < args.episodes and attempts < args.episodes:
        attempts += 1
        env.reset(mode=mode, probability=1.0, onset=args.onset, severity=0.5)
        controller.reset(args.envs, device)
        obs = env._observation()
        for _ in range(env.env_cfg.max_episode_steps):
            gimbal = obs[:, 14:16] * env.physics.gimbal_limit
            out = env.step(controller.act(obs))
            obs = out.obs
            done = out.terminated | out.truncated
            if done.any():
                state = out.info["state"]
                for i in torch.nonzero(done).flatten().tolist():
                    rows.append(
                        (
                            float(gimbal[i].abs().max()) * 180.0 / math.pi,
                            bool(out.info["landing_success"][i]),
                            float(torch.linalg.vector_norm(state[i, 0:2])),
                        )
                    )
            if len(rows) >= args.episodes:
                break

    print(f"{'freeze |delta| [deg]':>20s}  {'landed':>7s}  {'lateral [m]':>12s}")
    for angle, landed, lateral in rows:
        print(f"{angle:20.3f}  {str(landed):>7s}  {lateral:12.2f}")

    benign = [r for r in rows if r[0] < args.benign_deg]
    severe = [r for r in rows if r[0] >= args.benign_deg]
    print(
        f"\nbenign freezes (< {args.benign_deg} deg): {len(benign)}/{len(rows)}, "
        f"landed {sum(r[1] for r in benign)}/{len(benign)}"
    )
    print(
        f"severe freezes (>= {args.benign_deg} deg): {len(severe)}/{len(rows)}, "
        f"landed {sum(r[1] for r in severe)}/{len(severe)}"
    )


if __name__ == "__main__":
    main()
