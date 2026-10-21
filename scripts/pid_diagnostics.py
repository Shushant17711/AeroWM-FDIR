"""Diagnostics for the PID baseline: terminal-state statistics per fault mode."""

from __future__ import annotations

import sys

import torch

from agents.pid_baseline import CascadePIDController, PIDGains
from envs.rocket_vec_env import VectorizedRocketEnv
from envs.rocket_dynamics import tilt_angle_deg


@torch.no_grad()
def run(env, controller, episodes: int, mode: int = 0) -> None:
    env.reset(mode=mode, probability=1.0 if mode else 0.0, onset=60, severity=0.35)
    controller.reset(env.num_envs, env.device)
    obs = env._observation()
    rows = []
    for t in range(env.env_cfg.max_episode_steps):
        out = env.step(controller.act(obs))
        obs = out.obs
        done = out.terminated | out.truncated
        if done.any():
            s = out.info["state"]
            for i in torch.nonzero(done).flatten().tolist():
                rows.append(
                    (
                        t,
                        int(out.info["termination_reason"][i]),
                        round(float(s[i, 5]), 2),
                        round(float(torch.linalg.vector_norm(s[i, 3:5])), 2),
                        round(float(torch.linalg.vector_norm(s[i, 0:2])), 1),
                        round(float(tilt_angle_deg(s)[i]), 1),
                        bool(out.info["landing_success"][i]),
                    )
                )
        if len(rows) >= episodes:
            break
    print("step reason vz vlat latd tilt landed")
    for r in rows:
        print(r)
    landed = sum(1 for r in rows if r[-1])
    print(f"landed {landed}/{len(rows)}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedRocketEnv(num_envs=8, device=device, seed=0)
    gains = PIDGains(
        brake_accel=4.0,
        vertical_velocity=1.6,
        lateral_position=0.03,
        tilt_limit_deg=20,
    )
    if len(sys.argv) > 1:
        gains = PIDGains(
            brake_accel=float(sys.argv[1]),
            lateral_position=float(sys.argv[2]),
            lateral_velocity=float(sys.argv[3]),
            max_attitude_rate=float(sys.argv[4]),
            max_lateral_accel=float(sys.argv[5]),
            tilt_limit_deg=float(sys.argv[6]),
            vertical_velocity=float(sys.argv[7]) if len(sys.argv) > 7 else 1.6,
        )
    run(env, CascadePIDController(device=device, gains=gains), 8)


if __name__ == "__main__":
    main()
