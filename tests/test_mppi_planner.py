"""Task 5.2: latent MPPI planner contract, bounds and latency gate."""

from __future__ import annotations

import time

import pytest
import torch

from agents.base import BaseController
from agents.mppi_planner import LatentMPPIPlanner
from agents.world_model_agent import WorldModelAgent
from configs.schema import MPPIConfig, WorldModelConfig
from models.world_model import RocketWorldModel


def small_models(device="cpu"):
    cfg = WorldModelConfig(
        latent_dim=64, gru_dim=64, hidden_dim=64, simnorm_groups=8, simnorm_v=8, horizon=5
    )
    torch.manual_seed(0)
    model = RocketWorldModel(cfg).to(device)
    agent = WorldModelAgent(cfg, model).to(device)
    return cfg, model, agent


def test_planner_contract_and_bounds():
    cfg, model, agent = small_models()
    planner = LatentMPPIPlanner(
        model,
        agent,
        MPPIConfig(num_samples=64, num_policy_samples=8, horizon=5, iterations=2, num_elites=8),
        device="cpu",
        autocast=False,
    )
    assert isinstance(planner, BaseController)
    assert planner.action_dim == 3
    planner.reset(4, torch.device("cpu"))
    obs = torch.randn(4, 18)
    action = planner.act(obs)
    assert action.shape == (4, 3)
    assert torch.all(action.abs() <= 1.0)
    assert torch.isfinite(action).all()


def test_planner_maintains_recurrent_state():
    cfg, model, agent = small_models()
    planner = LatentMPPIPlanner(
        model,
        agent,
        MPPIConfig(num_samples=32, num_policy_samples=4, horizon=4, iterations=1, num_elites=4),
        device="cpu",
        autocast=False,
    )
    planner.reset(2, torch.device("cpu"))
    obs = torch.randn(2, 18)
    planner.act(obs)
    assert planner.h is not None and planner.h.shape == (2, cfg.gru_dim)
    assert planner.context.shape == (2, cfg.fault_context_dim)
    assert planner.surprise.shape == (2,)
    assert torch.all(planner.surprise >= 0)
    assert planner.prev_action.shape == (2, 3)


@pytest.mark.gpu
@pytest.mark.slow
def test_planner_latency_per_environment():
    device = torch.device("cuda")
    cfg = WorldModelConfig()
    model = RocketWorldModel(cfg).to(device).eval()
    agent = WorldModelAgent(cfg, model).to(device).eval()
    planner = LatentMPPIPlanner(model, agent, MPPIConfig(), device=device, autocast=False)
    batch = 16
    planner.reset(batch, device)
    obs = torch.randn(batch, 18, device=device)
    for _ in range(3):
        planner.act(obs)
    torch.cuda.synchronize()
    steps = 5
    start = time.perf_counter()
    for _ in range(steps):
        planner.act(obs)
    torch.cuda.synchronize()
    per_env_ms = (time.perf_counter() - start) / steps / batch * 1e3
    print(f"\nMPPI latency: {per_env_ms:.2f} ms per environment (K=512, H=15, 4 iterations)")
    assert per_env_ms < 15.0, per_env_ms
