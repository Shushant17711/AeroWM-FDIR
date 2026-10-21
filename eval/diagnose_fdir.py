"""Gate 3 check: does the KL surprise spike at fault onset on a trained model?

Runs the world model's posterior/prior on a clean rollout, then on the same
scenario with a mid-flight fault, and reports the surprise ratio and the
detection latency.  Usage::

    PYTHONPATH=. python -m eval.diagnose_fdir --checkpoint runs/aerowm_fdir/checkpoint_final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from agents.world_model_agent import WorldModelAgent
from configs.schema import FaultMode, WorldModelConfig
from envs.rocket_vec_env import VectorizedRocketEnv
from models.world_model import RocketWorldModel


@torch.no_grad()
def rollout_surprise(
    env: VectorizedRocketEnv,
    model: RocketWorldModel,
    agent: WorldModelAgent,
    mode: int,
    onset: int,
    steps: int,
) -> dict:
    env.reset(mode=mode, probability=1.0 if mode else 0.0, onset=onset, severity=0.5)
    batch = env.num_envs
    device = env.device
    h = torch.zeros(batch, model.cfg.gru_dim, device=device)
    z = torch.zeros(batch, model.cfg.latent_dim, device=device)
    a_prev = torch.zeros(batch, model.cfg.action_dim, device=device)
    c = torch.zeros(batch, model.cfg.fault_context_dim, device=device)
    obs_prev = None

    surprise, altitude = [], []
    for _ in range(steps):
        obs = env._observation()
        h = model.recurrence(h, z, a_prev)
        e = model.encode(obs)
        mu_q, log_q = model.posterior_params(h, e)
        z = model.latent_from_params(mu_q, log_q, sample=False)
        mu_p, log_p = model.prior_params(h)
        obs_hat = model.predict_observation(h, mu_p)
        innovation = obs - obs_hat
        signals = model.fault_head(
            h, z, mu_q, log_q, mu_p, log_p, innovation, a_prev, obs_prev=obs_prev, obs=obs
        )
        obs_prev = obs
        logits, c, s = signals.logits, signals.context, signals.surprise
        surprise.append(s.mean().item())
        altitude.append(env.state[:, 2].mean().item())
        action = agent.actor.mean_action(h, z, c)
        out = env.step(action)
        a_prev = action
        if (out.terminated | out.truncated).all():
            break
    return {"surprise": surprise, "altitude": altitude, "logits": logits, "mode": mode, "onset": onset}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/aerowm_fdir/checkpoint_final.pt")
    parser.add_argument("--onset", type=int, default=120)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--envs", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = WorldModelConfig()
    model = RocketWorldModel(cfg).to(device).eval()
    agent = WorldModelAgent(cfg, model).to(device).eval()
    path = Path(args.checkpoint)
    if path.exists():
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model"])
        agent.actor.load_state_dict(state["actor"])
        print(f"loaded {path}")
    else:
        print(f"WARNING: {path} not found; using random weights")

    env = VectorizedRocketEnv(num_envs=args.envs, device=device, seed=0)

    nominal = rollout_surprise(env, model, agent, mode=0, onset=10**9, steps=args.steps)
    model.fdir.detector.calibrate(torch.tensor(nominal["surprise"]))
    threshold = float(model.fdir.detector.threshold)
    print(f"nominal surprise: mean={sum(nominal['surprise']) / len(nominal['surprise']):.3f} threshold={threshold:.3f}")

    for mode in range(1, 5):
        run = rollout_surprise(env, model, agent, mode=mode, onset=args.onset, steps=args.steps)
        pre = run["surprise"][max(0, args.onset - 60) : args.onset]
        post = run["surprise"][args.onset : args.onset + 20]
        base = sum(pre) / max(1, len(pre))
        peak = sum(post) / max(1, len(post))
        latency = next(
            (i for i, v in enumerate(run["surprise"][max(0, args.onset - 1) :]) if v > threshold), None
        )
        print(
            f"{FaultMode(mode).name:14s} pre={base:8.2f} post={peak:8.2f} ratio={peak / max(base, 1e-6):5.2f} "
            f"latency={latency} steps"
        )


if __name__ == "__main__":
    main()
