"""Grid-search harness for the cascade PID baseline gains."""

from __future__ import annotations

import itertools
import json
import sys

import torch

from agents.pid_baseline import CascadePIDController, PIDGains
from envs.rocket_vec_env import VectorizedRocketEnv


@torch.no_grad()
def evaluate(env, controller, episodes: int, mode: int = 0, onset: int = 60) -> dict:
    controller.use_thrust_feedback = False
    success = crash = timeout = 0
    while success + crash + timeout < episodes:
        env.reset(mode=mode, probability=1.0 if mode else 0.0, onset=onset, severity=0.35)
        controller.reset(env.num_envs, env.device)
        obs = env._observation()
        for _ in range(env.env_cfg.max_episode_steps):
            out = env.step(controller.act(obs))
            obs = out.obs
            done = out.terminated | out.truncated
            if done.any():
                landed = out.info["landing_success"] & out.terminated
                success += int(landed.sum())
                crash += int((out.terminated & ~out.info["landing_success"]).sum())
                timeout += int((out.truncated & ~out.terminated).sum())
            if success + crash + timeout >= episodes:
                break
    total = max(1, success + crash + timeout)
    return {"success": success, "crash": crash, "timeout": timeout, "rate": success / total}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 32
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    env = VectorizedRocketEnv(num_envs=batch, device=device, seed=0)
    best = None
    grid = itertools.product([3.0, 4.0], [1.0, 1.6], [0.08, 0.12], [1.2, 2.0], [0.8, 1.4])
    for brake, kvz, mr, la, lv in grid:
        g = PIDGains(brake_accel=brake, vertical_velocity=kvz, max_attitude_rate=mr, max_lateral_accel=la, lateral_velocity=lv)
        c = CascadePIDController(device=device, gains=g)
        env.reset(probability=0.0)
        r = evaluate(env, c, episodes)
        print(f"brake={brake} kvz={kvz} mr={mr} la={la} lv={lv}: {r['rate']:.2f} (s={r['success']} c={r['crash']} t={r['timeout']})")
        if best is None or r["rate"] > best[0]:
            best = (r["rate"], g, r)
    print("BEST", best[0], best[1], best[2])
    print(json.dumps({"rate": best[0], "gains": best[1].__dict__, "result": best[2]}))


if __name__ == "__main__":
    main()
