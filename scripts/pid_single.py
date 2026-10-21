"""Dump a single deterministic PID episode for debugging."""

from __future__ import annotations

import sys

import torch

from agents.pid_baseline import CascadePIDController, PIDGains
from envs.rocket_vec_env import VectorizedRocketEnv
from envs.rocket_dynamics import tilt_angle_deg


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    vx0 = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    z0 = float(sys.argv[3]) if len(sys.argv) > 3 else 800.0
    vz0 = float(sys.argv[4]) if len(sys.argv) > 4 else -70.0

    env = VectorizedRocketEnv(num_envs=1, device=device, seed=0)
    env.reset(probability=0.0)
    s = env.state.clone()
    s[0, 0] = x0
    s[0, 1] = 0.0
    s[0, 2] = z0
    s[0, 3] = vx0
    s[0, 4] = 0.0
    s[0, 5] = vz0
    s[0, 6:10] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    s[0, 10:13] = 0.0
    s[0, 16] = 0.0
    env.state = s

    gains = PIDGains(
        tilt_limit_deg=30,
        brake_accel=0.8,
        vertical_velocity=3.0,
        attitude_damping=4.0,
    )
    c = CascadePIDController(device=device, gains=gains)
    c.reset(1, device)
    obs = env._observation()
    for t in range(env.env_cfg.max_episode_steps):
        out = env.step(c.act(obs))
        obs = out.obs
        st = out.info["state"]
        if t % 100 == 0:
            print(
                f"t={t:4d} x={st[0,0]:7.1f} vx={st[0,3]:6.1f} z={st[0,2]:6.1f} vz={st[0,5]:6.1f} "
                f"tilt={tilt_angle_deg(st)[0]:5.1f} T={st[0,16]/1e5:4.1f} u={c.act(obs)[0,2]:5.2f}"
            )
        if out.terminated.any() or out.truncated.any():
            print(
                "END",
                t,
                int(out.info["termination_reason"][0]),
                bool(out.info["landing_success"][0]),
                f"vz={st[0,5]:.2f} vlat={torch.linalg.vector_norm(st[0,3:5]):.2f} "
                f"latd={torch.linalg.vector_norm(st[0,0:2]):.2f} tilt={tilt_angle_deg(st)[0]:.1f}",
            )
            break


if __name__ == "__main__":
    main()
