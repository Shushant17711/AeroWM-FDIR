"""Task 4.1: RSSM world model shapes, imagination and gradient safety."""

from __future__ import annotations

import pytest
import torch

from configs.schema import WorldModelConfig
from models.networks import NormMLP, SimNorm, gaussian_kl
from models.world_model import RocketWorldModel


@pytest.fixture(scope="module")
def cfg() -> WorldModelConfig:
    return WorldModelConfig(
        latent_dim=64, gru_dim=64, hidden_dim=64, simnorm_groups=8, simnorm_v=8, horizon=6
    )


@pytest.fixture(scope="module")
def model(cfg) -> RocketWorldModel:
    torch.manual_seed(0)
    return RocketWorldModel(cfg)


def test_simnorm_is_a_probability_simplex():
    sim = SimNorm(64, groups=8)
    z = sim(torch.randn(5, 64) * 10)
    assert z.shape == (5, 64)
    groups = z.reshape(5, 8, 8)
    assert torch.allclose(groups.sum(dim=-1), torch.ones(5, 8), atol=1e-6)
    assert torch.all(z >= 0)


def test_gaussian_kl_identities():
    mu = torch.randn(4, 8)
    log_std = torch.randn(4, 8) * 0.1
    kl = gaussian_kl(mu, log_std, mu, log_std)
    assert torch.allclose(kl, torch.zeros(4), atol=1e-6)
    other = gaussian_kl(mu, log_std, mu + 1.0, log_std)
    assert torch.all(other > 0)


def test_forward_sequence_shapes(model, cfg):
    batch, horizon = 4, 6
    obs = torch.randn(batch, horizon, 18)
    action = torch.rand(batch, horizon, 3) * 2 - 1
    out = model.forward_sequence(obs, action)
    assert out["h"].shape == (batch, horizon, cfg.gru_dim)
    assert out["z"].shape == (batch, horizon, cfg.latent_dim)
    assert out["reward"].shape == (batch, horizon)
    assert out["q1"].shape == (batch, horizon)
    assert out["fault_logits"].shape == (batch, horizon, 5)
    assert out["context"].shape == (batch, horizon, cfg.fault_context_dim)
    assert out["surprise"].shape == (batch, horizon)
    assert torch.all(out["surprise"] >= 0)
    assert torch.all(out["consistency"] <= 1.0001)
    assert torch.all(out["consistency"] >= -1.0001)


def test_imagine_rollout_shapes(model):
    batch, samples, horizon = 2, 8, 5
    actions = torch.rand(batch, samples, horizon, 3) * 2 - 1
    out = model.imagine(
        torch.zeros(batch, 64), torch.zeros(batch, 64), torch.zeros(batch, 16), actions, need_q=True
    )
    assert out["reward"].shape == (batch, samples, horizon)
    assert out["q"].shape == (batch, samples, horizon)
    assert out["h"].shape == (batch, samples, 64)
    assert torch.isfinite(out["reward"]).all()


def test_gradients_are_finite(model):
    obs = torch.randn(2, 4, 18)
    action = torch.rand(2, 4, 3) * 2 - 1
    out = model.forward_sequence(obs, action)
    loss = (
        out["reward"].mean()
        + out["q1"].mean()
        + out["q2"].mean()
        + out["consistency"].mean()
        + out["fault_logits"].mean()
        + out["surprise"].mean() * 1e-3
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 10
    assert all(torch.isfinite(g).all() for g in grads)


def test_target_network_soft_update():
    cfg = WorldModelConfig(latent_dim=64, gru_dim=64, hidden_dim=64, simnorm_groups=8, simnorm_v=8)
    model = RocketWorldModel(cfg)
    source = [p.clone() for p in model.q1.parameters()]
    target = [p.clone() for p in model.target_q1.parameters()]
    with torch.no_grad():
        for p in model.q1.parameters():
            p.add_(1.0)
    updated = [p.detach().clone() for p in model.q1.parameters()]
    model.update_targets(tau=0.1)
    for old_target, new_target, new_source in zip(target, model.target_q1.parameters(), updated):
        assert torch.allclose(new_target, 0.9 * old_target + 0.1 * new_source, atol=1e-6)
    assert len(source) == len(target)


@pytest.mark.gpu
def test_imagination_horizon_bf16(gpu_device):
    cfg = WorldModelConfig()  # default: latent 256, horizon 15
    device = str(gpu_device)
    model = RocketWorldModel(cfg).to(gpu_device)
    actions = torch.rand(2, 8, cfg.horizon, 3, device=device) * 2 - 1
    with torch.autocast(gpu_device.type, dtype=torch.bfloat16):
        out = model.imagine(
            torch.zeros(2, cfg.gru_dim, device=device),
            torch.zeros(2, cfg.latent_dim, device=device),
            torch.zeros(2, cfg.fault_context_dim, device=device),
            actions,
        )
    assert out["reward"].shape == (2, 8, cfg.horizon)
    assert torch.isfinite(out["reward"]).all()
    assert out["reward"].dtype in (torch.bfloat16, torch.float32)


def test_norm_mlp_shapes():
    mlp = NormMLP(10, 32, 7, num_layers=3)
    out = mlp(torch.randn(5, 10))
    assert out.shape == (5, 7)
