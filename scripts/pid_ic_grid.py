"""Deterministic initial-condition sweep for the PID baseline.

Holds altitude/descent fixed and varies lateral offset and outward velocity to
identify which terminal conditions are recoverable.
"""

from __future__ import annotations

import torch

from agents.pid_baseline import CascadePIDController, PIDGains
from envs.rocket_vec_env import VectorizedRocketEnv
from envs.rocket_dynamics import tilt_angle_deg


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedRocketEnv(num_envs=16, device=device, seed=0)
    gains = PIDGains(
        attitude_damping=4.0,
        tilt_limit_deg=30,
        brake_accel=0.8,
        vertical_velocity=3.0,
    )
    c = CascadePIDController(device=device, gains=gains)

    for z0 in (800.0, 1200.0):
        for x0 in (50.0, 100.0, 150.0):
            for vx0 in (-20.0, 0.0, 20.0):
                env.reset(probability=0.0)
                s = env.state.clone()
                s[:, 0] = x0
                s[:, 1] = 0.0
                s[:, 2] = z0
                s[:, 3] = vx0
                s[:, 4] = 0.0
                s[:, 5] = -70.0
                s[:, 6:10] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
                s[:, 10:13] = 0.0
                s[:, 16] = 0.0
                env.state = s
                c.reset(env.num_envs, device)
                obs = env._observation()
                result = None
                for t in range(env.env_cfg.max_episode_steps):
                    out = env.step(c.act(obs))
                    obs = out.obs
                    if out.terminated.any() or out.truncated.any():
                        st = out.info["state"][0:1]
                        result = (
                            int(out.info["termination_reason"][0]),
                            bool(out.info["landing_success"][0]),
                            round(float(st[0, 2]), 2),
                            round(float(st[0, 5]), 2),
                            round(float(torch.linalg.vector_norm(st[0, 3:5])), 2),
                            round(float(torch.linalg.vector_norm(st[0, 0:2])), 2),
                            round(float(tilt_angle_deg(st)[0]), 1),
                            t,
                        )
                        break
                print(f"z0={z0:.0f} x0={x0:5.1f} vx0={vx0:+5.1f} -> {result}")


if __name__ == "__main__":
    main()
