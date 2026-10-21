"""Task 5.3: PPO baseline rollout-and-update smoke test."""

from __future__ import annotations

import torch

from agents.base import BaseController
from agents.ppo_baseline import PPOController
from configs.schema import PPOConfig
from envs.rocket_vec_env import VectorizedRocketEnv


def test_ppo_controller_contract():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOController(cfg=PPOConfig(rollout_steps=8), device=device)
    assert isinstance(agent, BaseController)
    agent.reset(4, device)
    obs = torch.randn(4, 18, device=device)
    action = agent.act(obs)
    assert action.shape == (4, 3)
    assert torch.all(action.abs() <= 1.0)
    stochastic = agent.act(obs, deterministic=False)
    assert stochastic.shape == (4, 3)


def test_ppo_one_rollout_and_update_epoch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedRocketEnv(num_envs=8, device=device, seed=0)
    cfg = PPOConfig(rollout_steps=16, epochs=2, minibatches=2)
    agent = PPOController(cfg=cfg, device=device)
    data = agent.collect(env, steps=cfg.rollout_steps)
    assert data["obs"].shape == (8, 16, 18)
    assert data["action"].shape == (8, 16, 3)
    assert data["bootstrap"].shape == (8,)
    stats = agent.update(data)
    for key in ("policy_loss", "value_loss", "entropy"):
        assert key in stats
        assert torch.isfinite(torch.tensor(stats[key]))
    # Parameters must have changed after an update.
    before = {k: v.clone() for k, v in agent.net.state_dict().items()}
    agent.update(data)
    assert any(not torch.equal(before[k], v) for k, v in agent.net.state_dict().items())
