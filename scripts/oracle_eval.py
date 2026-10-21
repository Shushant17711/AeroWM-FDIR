"""Evaluate the oracle fault-tolerant PID (demonstration generator) per fault mode."""

from __future__ import annotations

import argparse

import torch

from agents.pid_baseline import CascadePIDController
from configs.schema import FaultMode
from envs.rocket_vec_env import VectorizedRocketEnv


@torch.no_grad()
def run(env, controller, episodes: int, mode: int, oracle: bool) -> dict:
    controller.oracle_fault_compensation = oracle
    success = crash = timeout = 0
    env.fixed_fault_mode = mode
    while success + crash + timeout < episodes:
        env.reset(mode=mode, probability=1.0 if mode else 0.0)
        controller.reset(env.num_envs, env.device)
        obs = env._observation()
        for _ in range(env.env_cfg.max_episode_steps):
            action = controller.act(obs, fault_state=env.fault)
            out = env.step(action)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedRocketEnv(num_envs=args.batch, device=device, seed=0)
    controller = CascadePIDController(device=device)
    for oracle in (False, True):
        print(f"--- oracle_fault_compensation={oracle} ---")
        for mode in range(5):
            result = run(env, controller, args.episodes, mode, oracle)
            print(f"{FaultMode(mode).name:14s} {result}")


if __name__ == "__main__":
    main()
