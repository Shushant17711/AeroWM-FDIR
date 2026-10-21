"""Task 1.1: frozen contracts, tensor shapes and physical parameter bounds."""

from __future__ import annotations

import dataclasses
import math

import pytest

from configs.schema import (
    ActionSchema,
    EnvironmentConfig,
    FaultInjectionSchema,
    FaultMode,
    FaultParamsLayout,
    MPPIConfig,
    ObservationSchema,
    PPOConfig,
    RewardConfig,
    RocketPhysicsConfig,
    TrainingConfig,
    WorldModelConfig,
    default_configs,
)


def test_physics_config_is_frozen():
    cfg = RocketPhysicsConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.dry_mass = 1.0  # type: ignore[misc]
    assert dataclasses.is_dataclass(cfg)


def test_physics_derived_quantities():
    cfg = RocketPhysicsConfig()
    assert cfg.wet_mass == pytest.approx(cfg.dry_mass + cfg.propellant_mass)
    assert cfg.gimbal_limit == pytest.approx(math.radians(cfg.gimbal_limit_deg))
    assert cfg.thrust_min == pytest.approx(0.40 * cfg.max_thrust)
    assert cfg.exhaust_velocity == pytest.approx(cfg.specific_impulse * cfg.gravity)
    # Engineering sanity: the engine must out-thrust the wet vehicle but the
    # deep-throttle floor must be below hover thrust for a controllable descent.
    assert cfg.max_thrust > cfg.wet_mass * cfg.gravity
    assert cfg.dt == pytest.approx(0.02)


def test_physics_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        RocketPhysicsConfig(dt=0.0)
    with pytest.raises(ValueError):
        RocketPhysicsConfig(min_throttle=0.9, max_throttle=0.2)
    with pytest.raises(ValueError):
        RocketPhysicsConfig(gimbal_limit_deg=90.0)
    with pytest.raises(ValueError):
        RocketPhysicsConfig(act_tau=-1.0)


def test_observation_schema_covers_18_dims():
    schema = ObservationSchema()
    assert schema.dim == 18
    indices = []
    for sl in schema.slices.values():
        indices.extend(range(*sl.indices(schema.dim)))
    assert sorted(indices) == list(range(18))


def test_action_schema_is_3d():
    action = ActionSchema()
    assert action.dim == 3
    assert (action.low, action.high) == (-1.0, 1.0)
    with pytest.raises(ValueError):
        ActionSchema(dim=4)


def test_fault_modes_and_context_dim():
    assert len(FaultMode) == 5
    assert [int(m) for m in FaultMode] == [0, 1, 2, 3, 4]
    fault = FaultInjectionSchema()
    assert fault.fault_context_dim == 5
    assert 0.0 <= fault.probability <= 1.0
    lo, hi = fault.onset_range
    assert 0 <= lo < hi
    with pytest.raises(ValueError):
        FaultInjectionSchema(probability=1.5)
    with pytest.raises(ValueError):
        FaultInjectionSchema(loe_severity=(0.9, 0.1))
    assert FaultParamsLayout.DIM == 7


def test_world_model_contract():
    wm = WorldModelConfig()
    obs = ObservationSchema()
    action = ActionSchema()
    assert wm.obs_dim == obs.dim
    assert wm.action_dim == action.dim
    assert wm.latent_dim == wm.simnorm_groups * wm.simnorm_v
    assert wm.horizon >= 1
    with pytest.raises(ValueError):
        WorldModelConfig(latent_dim=255)
    with pytest.raises(ValueError):
        WorldModelConfig(gamma=1.5)


def test_other_configs_validate():
    with pytest.raises(ValueError):
        EnvironmentConfig(max_episode_steps=0)
    with pytest.raises(ValueError):
        MPPIConfig(num_policy_samples=1000)
    with pytest.raises(ValueError):
        MPPIConfig(num_elites=512)
    with pytest.raises(ValueError):
        TrainingConfig(warmup_env_steps=10, total_env_steps=5)
    with pytest.raises(ValueError):
        TrainingConfig(sequence_length=1)
    assert RewardConfig().soft_landing_bonus > 0
    assert PPOConfig().gae_lambda == pytest.approx(0.95)


def test_default_configs_are_complete():
    cfgs = default_configs()
    assert set(cfgs) == {
        "physics",
        "observation",
        "action",
        "fault",
        "environment",
        "reward",
        "world_model",
        "mppi",
        "ppo",
        "training",
    }
    assert all(dataclasses.is_dataclass(c) for c in cfgs.values())
