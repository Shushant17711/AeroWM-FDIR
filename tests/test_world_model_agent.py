"""Task 5.1: squashed-Gaussian actor, log-prob Jacobian and TD(lambda) targets."""

from __future__ import annotations


import pytest
import torch

from agents.world_model_agent import WorldModelAgent
from configs.schema import WorldModelConfig
from models.world_model import RocketWorldModel


@pytest.fixture(scope="module")
def cfg() -> WorldModelConfig:
    return WorldModelConfig(
        latent_dim=64, gru_dim=64, hidden_dim=64, simnorm_groups=8, simnorm_v=8, horizon=4
    )


@pytest.fixture(scope="module")
def agent(cfg) -> WorldModelAgent:
    torch.manual_seed(0)
    model = RocketWorldModel(cfg)
    return WorldModelAgent(cfg, model)


def states(agent, batch=4):
    cfg = agent.cfg
    device = next(agent.parameters()).device
    return (
        torch.randn(batch, cfg.gru_dim, device=device),
        torch.randn(batch, cfg.latent_dim, device=device),
        torch.randn(batch, cfg.fault_context_dim, device=device),
        torch.zeros(batch, cfg.obs_dim, device=device),
    )


def test_action_bounds_and_shapes(agent):
    h, z, c, o = states(agent)
    action, log_prob, mean = agent.actor.sample(h, z, c, obs=o)
    assert action.shape == (4, 3)
    assert log_prob.shape == (4,)
    assert torch.all(action.abs() <= 1.0)
    assert torch.all(torch.isfinite(log_prob))


def test_log_prob_matches_manual_jacobian(agent):
    h, z, c, o = states(agent, 2)
    dist = agent.actor.distribution(h, z, c, o)
    action = torch.tensor([[0.3, -0.2, 0.9], [0.0, 0.5, -0.7]])
    u = torch.atanh(action)
    expected = dist.log_prob(u).sum(-1) - torch.log(1 - action**2 + 1e-6).sum(-1)
    got = agent.actor.log_prob(h, z, c, action, o)
    assert torch.allclose(got, expected, atol=1e-5)


def test_deterministic_action_is_tanh_mean(agent):
    h, z, c, o = states(agent)
    action, _, _ = agent.actor.sample(h, z, c, deterministic=True, obs=o)
    mu, _ = agent.actor(h, z, c, o)
    assert torch.allclose(action, torch.tanh(mu), atol=1e-6)


def test_entropy_is_finite(agent):
    h, z, c, o = states(agent)
    entropy = agent.actor.entropy(h, z, c, o)
    assert entropy.shape == (4,)
    assert torch.all(torch.isfinite(entropy))


def test_td_lambda_returns_without_dones(agent):
    rewards = torch.ones(1, 4)
    values = torch.zeros(1, 4)
    dones = torch.zeros(1, 4)
    out = agent.td_lambda_returns(rewards, values, dones, bootstrap=torch.zeros(1), gamma=0.9, lam=0.5)
    # gae_t = 1 + 0.9*0.5*gae_{t+1}; returns = gae
    expected = torch.zeros(4)
    running = 0.0
    for t in reversed(range(4)):
        running = 1.0 + 0.45 * running
        expected[t] = running
    assert torch.allclose(out[0], expected, atol=1e-5)


def test_td_lambda_masks_after_done(agent):
    rewards = torch.ones(1, 3)
    values = torch.zeros(1, 3)
    dones = torch.tensor([[0.0, 1.0, 0.0]])
    out = agent.td_lambda_returns(rewards, values, dones, bootstrap=torch.zeros(1), gamma=1.0, lam=1.0)
    assert out[0, 1].item() == pytest.approx(1.0)
    assert out[0, 2].item() == pytest.approx(1.0)
    # step 0 is not terminal, so it bootstraps from the value of step 1
    assert out[0, 0].item() == pytest.approx(2.0)


def test_actor_loss_gradients(agent):
    h, z, c, o = states(agent)
    loss = agent.actor_loss(h, z, c, obs=o)
    loss.backward()
    grads = [p.grad for p in agent.actor.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.gpu
def test_single_imagination_step_gradients(cfg, gpu_device):
    cfg = WorldModelConfig()
    device = str(gpu_device)
    agent = WorldModelAgent(cfg, RocketWorldModel(cfg)).to(gpu_device)
    h = torch.zeros(2, cfg.gru_dim, device=device)
    z = torch.zeros(2, cfg.latent_dim, device=device)
    c = torch.zeros(2, cfg.fault_context_dim, device=device)
    actions = torch.rand(2, 1, 1, 3, device=device) * 2 - 1
    with torch.autocast(gpu_device.type, dtype=torch.bfloat16):
        out = agent.model.imagine(h, z, c, actions)
        loss = out["reward"].float().mean()
    loss.backward()
    assert torch.isfinite(loss)
