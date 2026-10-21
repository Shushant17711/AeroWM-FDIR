"""Task 2.4: vectorized environment shapes, auto-reset and reward sanity."""

from __future__ import annotations

import math

import pytest
import torch

from configs.schema import FaultMode
from envs.rocket_vec_env import (
    REASON_CRASH,
    REASON_LANDED,
    REASON_TILT,
    VectorizedRocketEnv,
)


def make_env(batch: int = 8, seed: int = 0, device: str = "cpu") -> VectorizedRocketEnv:
    return VectorizedRocketEnv(num_envs=batch, device=device, seed=seed)


def test_reset_and_step_shapes():
    env = make_env(8)
    obs, info = env.reset()
    assert obs.shape == (8, 18)
    assert torch.isfinite(obs).all()
    for _ in range(5):
        out = env.step(torch.zeros(8, 3))
        assert out.obs.shape == (8, 18)
        assert out.reward.shape == (8,)
        assert out.terminated.shape == (8,)
        assert out.truncated.shape == (8,)
        assert torch.isfinite(out.obs).all()
        assert torch.isfinite(out.reward).all()


def test_observation_normalization_ranges():
    env = make_env(16)
    obs, _ = env.reset()
    pos = obs[:, 0:3]
    vel = obs[:, 3:6]
    quat = obs[:, 6:10]
    mass = obs[:, 13]
    gimbal = obs[:, 14:16]
    thrust = obs[:, 16]
    q_dyn = obs[:, 17]
    assert torch.all(pos.abs() <= 2.0)
    assert torch.all(vel.abs() <= 1.5)
    assert torch.allclose(torch.linalg.vector_norm(quat, dim=-1), torch.ones(16), atol=1e-5)
    assert torch.all(mass >= -1e-6) and torch.all(mass <= 1.0 + 1e-6)
    assert torch.all(gimbal.abs() <= 1.0 + 1e-6)
    assert torch.all(thrust >= -1e-6) and torch.all(thrust <= 1.0 + 1e-6)
    assert torch.all(q_dyn >= 0.0)


def test_action_validation():
    env = make_env(4)
    with pytest.raises(ValueError):
        env.step(torch.zeros(4, 4))
    with pytest.raises(ValueError):
        env.step(torch.zeros(3, 3))
    out = env.step(torch.full((4, 3), 10.0))
    assert torch.isfinite(out.obs).all()


def test_forced_loe_fault_activates_at_onset():
    env = make_env(4)
    env.reset(mode=1, onset=5, severity=0.3, probability=1.0)
    action = torch.zeros(4, 3)
    for step in range(4):
        out = env.step(action)
        assert not bool(out.info["fault_active"].any())
    out = env.step(action)
    assert bool(out.info["fault_active"].all())
    assert torch.allclose(out.info["fault_context"][:, 1], torch.ones(4))
    assert torch.allclose(out.info["fault_params"][:, 0], torch.full((4,), 0.7), atol=1e-5)


def test_auto_reset_on_soft_touchdown():
    env = make_env(4)
    state = env.state.clone()
    state[:, 0:2] = 0.0
    state[:, 2] = 0.01
    state[:, 3:5] = 0.0
    state[:, 5] = -1.0
    state[:, 6:10] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    state[:, 10:13] = 0.0
    state[:, 14:16] = 0.0
    state[:, 16] = 0.0
    env.state = state

    out = env.step(torch.zeros(4, 3))
    assert bool(out.terminated.all())
    assert bool(out.info["landing_success"].all())
    assert torch.all(out.info["termination_reason"] == REASON_LANDED)
    # Auto-reset: returned observations belong to a fresh high-altitude episode.
    assert torch.all(out.obs[:, 2] > 0.2)
    assert torch.all(out.info["reset_mask"])
    assert torch.all(out.info["final_state"][:, 2] <= 0.05)


def test_crash_termination_and_penalty():
    env = make_env(4)
    state = env.state.clone()
    state[:, 0:2] = 0.0
    state[:, 2] = 0.01
    state[:, 5] = -60.0
    state[:, 6:10] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    state[:, 16] = 0.0
    env.state = state
    out = env.step(torch.zeros(4, 3))
    assert bool(out.terminated.all())
    assert not bool(out.info["landing_success"].any())
    assert torch.all(out.info["termination_reason"] == REASON_CRASH)
    assert torch.all(out.reward < -100.0)


def test_tilt_termination():
    env = make_env(2)
    state = env.state.clone()
    # 90 degree tilt about body x.
    q = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
    state[:, 6:10] = q
    env.state = state
    out = env.step(torch.zeros(2, 3))
    assert torch.all(out.info["termination_reason"] == REASON_TILT)
    assert bool(out.terminated.all())


def test_fuel_exhaustion_terminates():
    env = make_env(2)
    state = env.state.clone()
    state[:, 13] = env.physics.dry_mass
    env.state = state
    out = env.step(torch.zeros(2, 3))
    assert bool(out.terminated.all())


def test_timeout_truncation():
    env = VectorizedRocketEnv(num_envs=2, device="cpu", seed=0)
    env.reset()
    env.step_count[:] = env.env_cfg.max_episode_steps - 1
    out = env.step(torch.zeros(2, 3))
    assert bool(out.truncated.all())


def test_thousand_random_steps_stay_finite():
    env = make_env(16, seed=3)
    torch.manual_seed(3)
    env.reset()
    for _ in range(1000):
        action = torch.rand(16, 3) * 2 - 1
        out = env.step(action)
        assert torch.isfinite(out.obs).all()
        assert torch.isfinite(out.reward).all()
    assert env.completed_episodes > 0


def test_episode_counters_track_landings():
    env = make_env(4)
    state = env.state.clone()
    state[:, 0:2] = 0.0
    state[:, 2] = 0.01
    state[:, 3:5] = 0.0
    state[:, 5] = -1.0
    state[:, 6:10] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    state[:, 10:13] = 0.0
    state[:, 14:16] = 0.0
    state[:, 16] = 0.0
    env.state = state
    env.step(torch.zeros(4, 3))
    assert env.completed_episodes == 4
    assert env.completed_landings == 4


def test_deterministic_under_fixed_seed():
    e1 = make_env(8, seed=7)
    e2 = make_env(8, seed=7)
    a = torch.zeros(8, 3)
    for _ in range(20):
        o1 = e1.step(a).obs
        o2 = e2.step(a).obs
        assert torch.allclose(o1, o2)


def test_nominal_probability_override():
    env = make_env(64)
    env.reset(probability=0.0)
    out = env.step(torch.zeros(64, 3))
    assert torch.all(out.info["fault_mode"] == int(FaultMode.NOMINAL))
