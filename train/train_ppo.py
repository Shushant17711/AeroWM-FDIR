"""Train the model-free PPO baseline (Gate 5 comparison controller).

Optionally initialises the actor by behaviour cloning from the classical PID
controller, then fine-tunes with PPO on nominal flights.  The benchmark still
tests the trained weights on fault scenarios (the baseline never sees them).

Usage::

    PYTHONPATH=. python -m train.train_ppo --iterations 200 --output runs/ppo/ppo.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from agents.pid_baseline import CascadePIDController
from agents.ppo_baseline import PPOController
from configs.schema import PPOConfig
from envs.rocket_vec_env import VectorizedRocketEnv


@torch.no_grad()
def collect_pid_demos(env, controller, steps: int):
    obs_buf, act_buf = [], []
    env.reset(probability=0.0)
    controller.reset(env.num_envs, env.device)
    obs = env._observation()
    while sum(o.shape[0] for o in obs_buf) < steps:
        action = controller.act(obs)
        obs_buf.append(obs)
        act_buf.append(action)
        obs = env.step(action).obs
    return torch.cat(obs_buf)[:steps], torch.cat(act_buf)[:steps]


def bc_pretrain(agent: PPOController, env, steps: int = 20_000, epochs: int = 6) -> float:
    """Initialise the residual head at zero on states visited by the reference.

    The executed command is ``u_pid + scale * tanh(a_ppo)``; fitting
    ``tanh(a_ppo) -> 0`` therefore starts PPO exactly on the classical law
    instead of on a random perturbation of it.
    """
    controller = CascadePIDController(device=env.device)
    obs, _ = collect_pid_demos(env, controller, steps)
    obs = obs.to(agent.device)
    actions = torch.zeros(obs.shape[0], agent.action_dim, device=agent.device)
    optimizer = torch.optim.Adam(agent.net.actor.parameters(), lr=3e-4)
    batch = 1024
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(obs.shape[0], device=agent.device)
        for start in range(0, obs.shape[0], batch):
            idx = perm[start : start + batch]
            mu, _ = agent.net.actor(obs[idx]).chunk(2, dim=-1)
            loss = F.mse_loss(torch.tanh(mu), actions[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last = float(loss.detach())
    with torch.no_grad():
        agent.net.log_std.fill_(-2.0)
    return last


@torch.no_grad()
def evaluate(env, agent: PPOController, episodes: int = 32) -> float:
    env.fixed_fault_mode = 0
    env.fault_probability = 0.0
    success = total = 0
    while total < episodes:
        env.reset(probability=0.0)
        agent.reset(env.num_envs, env.device)
        obs = env._observation()
        for _ in range(env.env_cfg.max_episode_steps):
            out = env.step(agent.act(obs, deterministic=True))
            obs = out.obs
            done = out.terminated | out.truncated
            if done.any():
                success += int((out.info["landing_success"] & out.terminated).sum())
                total += int(done.sum())
            if total >= episodes:
                break
    return success / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--bc-steps", type=int, default=20_000)
    parser.add_argument("--no-bc", action="store_true")
    parser.add_argument("--output", type=str, default="runs/ppo/ppo.pt")
    parser.add_argument("--log", type=str, default="runs/ppo/metrics.jsonl")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--residual-l2", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedRocketEnv(num_envs=args.envs, device=device, seed=args.seed, fault_probability=0.0)
    overrides = {}
    if args.residual_l2 is not None:
        overrides["residual_l2"] = args.residual_l2
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    cfg = PPOConfig(rollout_steps=args.rollout, **overrides)
    agent = PPOController(cfg=cfg, device=device, seed=args.seed)

    if not args.no_bc:
        loss = bc_pretrain(agent, env, steps=args.bc_steps)
        print(
            f"BC pretrain MSE {loss:.4f}, nominal landing "
            f"{evaluate(env, agent, episodes=args.eval_episodes):.2%}",
            flush=True,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Iteration 0 is the residual head initialised at zero, i.e. exactly the
    # classical reference.  Best-checkpointing from here means the shipped
    # model-free baseline can never be worse than the law it wraps.
    best = evaluate(env, agent, episodes=args.eval_episodes)
    torch.save({"net": agent.net.state_dict()}, out_path)
    print(f"iter    0 residual=0.00 nominal={best:.1%}", flush=True)
    start = time.time()
    env.reset(probability=0.0)
    for iteration in range(1, args.iterations + 1):
        data = agent.collect(env, steps=args.rollout, reset=False)
        stats = agent.update(data)
        stats["iteration"] = iteration
        stats["wall_s"] = time.time() - start
        if iteration % args.eval_every == 0 or iteration == args.iterations:
            rate = evaluate(env, agent, episodes=args.eval_episodes)
            stats["nominal_landing_rate"] = rate
            # A tie keeps the *trained* policy: the behaviour-cloned
            # initialisation is only the fallback for a run that never beats
            # the classical law.
            if rate >= best - 1e-9:
                best = max(best, rate)
                torch.save({"net": agent.net.state_dict()}, out_path)
            print(
                f"iter {iteration:4d} pi={stats['policy_loss']:+.3f} v={stats['value_loss']:.3f} "
                f"H={stats['entropy']:.2f} nominal={rate:.1%} best={best:.1%}",
                flush=True,
            )
        with log_path.open("a") as fh:
            fh.write(json.dumps(stats) + "\n")
        if best >= 0.95 and iteration > args.iterations // 2:
            break

    if best < 0:
        torch.save({"net": agent.net.state_dict()}, out_path)
    print(f"best nominal landing rate {best:.1%} -> {out_path}")


if __name__ == "__main__":
    main()
