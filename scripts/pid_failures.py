"""Collect PID failure cases with their initial conditions for analysis."""

from __future__ import annotations

import torch

from agents.pid_baseline import CascadePIDController, PIDGains
from envs.rocket_vec_env import VectorizedRocketEnv
from envs.rocket_dynamics import tilt_angle_deg


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedRocketEnv(num_envs=64, device=device, seed=0)
    gains = PIDGains()
    c = CascadePIDController(device=device, gains=gains)
    env.reset(probability=0.0)
    c.reset(env.num_envs, device)
    init = env.state.clone()
    obs = env._observation()
    failures = []
    total = 0
    for t in range(env.env_cfg.max_episode_steps):
        out = env.step(c.act(obs))
        obs = out.obs
        # Track fresh episodes created by auto-reset.
        reset_mask = out.info["reset_mask"]
        init[reset_mask] = out.info["state"][reset_mask]
        done = out.terminated | out.truncated
        if done.any():
            s = out.info["state"]
            for i in torch.nonzero(done).flatten().tolist():
                total += 1
                if not out.info["landing_success"][i]:
                    st = init[i]
                    failures.append(
                        (
                            round(float(st[0]), 1), round(float(st[1]), 1), round(float(st[2])),
                            round(float(st[3]), 1), round(float(st[4]), 1), round(float(st[5]), 1),
                            round(float(tilt_angle_deg(st.unsqueeze(0))[0]), 1),
                            int(out.info["termination_reason"][i]),
                            round(float(s[i, 5]), 2),
                            round(float(torch.linalg.vector_norm(s[i, 3:5])), 2),
                            round(float(torch.linalg.vector_norm(s[i, 0:2])), 2),
                            round(float(tilt_angle_deg(s)[i]), 1),
                            t,
                        )
                    )
        if total >= 128:
            break
    print(f"failed {len(failures)}/{total}")
    print("x0 y0 z0 vx0 vy0 vz0 tilt0 | reason vz vlat latd tilt t")
    for f in failures[:30]:
        print(f)
    if failures:
        f = failures[0]
        print("first failure initial:", f)


if __name__ == "__main__":
    main()
