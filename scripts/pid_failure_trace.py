"""Dump the final seconds of the first failing PID episode in a batch."""

from __future__ import annotations

import torch

from agents.pid_baseline import CascadePIDController, PIDGains
from envs.rocket_vec_env import VectorizedRocketEnv
from envs.rocket_dynamics import tilt_angle_deg


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedRocketEnv(num_envs=32, device=device, seed=0)
    gains = PIDGains(
        tilt_limit_deg=30,
        brake_accel=0.8,
        vertical_velocity=3.0,
        attitude_damping=4.0,
        reposition_altitude=50.0,
        reposition_radius=10.0,
    )
    c = CascadePIDController(device=device, gains=gains)
    env.reset(probability=0.0)
    c.reset(env.num_envs, device)
    obs = env._observation()

    history = []
    for t in range(env.env_cfg.max_episode_steps):
        out = env.step(c.act(obs))
        obs = out.obs
        s = out.info["state"]
        dbg = c.last_debug
        history.append(
            torch.stack(
                [
                    s[:, 0], s[:, 3], s[:, 2], s[:, 5], tilt_angle_deg(s), s[:, 16] / 1e5,
                    dbg["a_xy"][:, 0], dbg["a_z"], dbg["direction"][:, 0],
                ],
                dim=-1,
            ).cpu()
        )
        done = out.terminated | out.truncated
        if done.any():
            for i in torch.nonzero(done).flatten().tolist():
                if out.info["landing_success"][i]:
                    continue
                print(
                    f"FAIL env={i} step={t} reason={int(out.info['termination_reason'][i])} "
                    f"vz={s[i,5]:.2f} vlat={torch.linalg.vector_norm(s[i,3:5]):.2f} "
                    f"latd={torch.linalg.vector_norm(s[i,0:2]):.2f} tilt={tilt_angle_deg(s)[i]:.1f}"
                )
                print(" t  x     vx    z     vz    tilt  T/1e5  a_xy  a_z   dirx")
                for k in range(max(0, len(history) - 500), len(history), 20):
                    row = history[k][i]
                    print(
                        f"{k:5d} {row[0]:6.1f} {row[1]:6.1f} {row[2]:6.1f} {row[3]:6.1f} "
                        f"{row[4]:5.1f} {row[5]:5.1f} {row[6]:6.2f} {row[7]:5.2f} {row[8]:6.2f}"
                    )
                return


if __name__ == "__main__":
    main()
