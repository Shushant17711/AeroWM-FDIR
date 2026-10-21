"""Task 2.1: 6-DOF dynamics, actuator lag, mass depletion and quaternion safety."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from configs.schema import FaultParamsLayout, RocketPhysicsConfig
from envs.quaternion_ops import quat_normalize
from envs.rk4_integrator import rk4_step
from envs.rocket_dynamics import (
    actuator_commands,
    atmosphere_density,
    compute_derivatives,
    gimbal_from_torque,
    inertia_tensor,
    thrust_vector_body,
)


def make_state(
    batch: int = 1,
    z: float = 1_000.0,
    vz: float = 0.0,
    mass: float | None = None,
    thrust: float = 0.0,
    gimbal: tuple[float, float] = (0.0, 0.0),
    cfg: RocketPhysicsConfig | None = None,
) -> torch.Tensor:
    cfg = cfg or RocketPhysicsConfig()
    state = torch.zeros(batch, 17)
    state[:, 2] = z
    state[:, 5] = vz
    state[:, 6] = 1.0  # identity quaternion
    state[:, 13] = cfg.wet_mass if mass is None else mass
    state[:, 14] = gimbal[0]
    state[:, 15] = gimbal[1]
    state[:, 16] = thrust
    return state


def nominal_fault(batch: int = 1) -> torch.Tensor:
    fault = torch.zeros(batch, FaultParamsLayout.DIM)
    fault[:, FaultParamsLayout.THRUST_SCALE] = 1.0
    return fault


def test_inertia_scales_with_mass():
    cfg = RocketPhysicsConfig()
    m = torch.tensor([cfg.dry_mass, cfg.wet_mass])
    inertia = inertia_tensor(m, cfg)
    assert inertia.shape == (2, 3)
    ratio = inertia[1] / inertia[0]
    assert torch.allclose(ratio, torch.full((3,), cfg.wet_mass / cfg.dry_mass), rtol=1e-6)


def test_atmosphere_density_decreases_with_altitude():
    cfg = RocketPhysicsConfig()
    z = torch.tensor([0.0, cfg.atmosphere_scale_height, 10_000.0])
    rho = atmosphere_density(z, cfg)
    assert rho[0] == pytest.approx(cfg.rho0)
    assert rho[1] == pytest.approx(cfg.rho0 * math.exp(-1.0))
    assert rho[2] < rho[1] < rho[0]


def test_actuator_command_mapping():
    cfg = RocketPhysicsConfig()
    fault = nominal_fault(4)
    gimbal = torch.zeros(4, 2)
    action = torch.tensor(
        [
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    delta, thrust_cmd, rcs = actuator_commands(gimbal, action, fault, cfg)
    assert torch.allclose(delta[0], torch.tensor([cfg.gimbal_limit, -cfg.gimbal_limit]), atol=1e-6)
    assert thrust_cmd[0].item() == pytest.approx(cfg.max_thrust)
    # Bottom of the action range commands engine shutdown (hoverslam).
    assert thrust_cmd[1].item() == pytest.approx(0.0)
    # u=0 lands at 44% of the firing band above the floor.
    # u = 0 -> exactly the minimum sustainable firing thrust.
    assert thrust_cmd[2].item() == pytest.approx(cfg.min_throttle * cfg.max_thrust)
    assert torch.all(delta.abs() <= cfg.gimbal_limit + 1e-6)


def test_engine_cutoff_dead_zone():
    cfg = RocketPhysicsConfig()
    fault = nominal_fault(3)
    gimbal = torch.zeros(3, 2)
    action = torch.tensor(
        [
            [-1.0, 0.0, -1.00],
            [-1.0, 0.0, -0.80],
            [-1.0, 0.0, 0.0],
        ]
    )
    _, thrust_cmd, rcs = actuator_commands(gimbal, action, fault, cfg)
    assert thrust_cmd[0].item() == pytest.approx(0.0)  # u=-1 -> shutdown
    assert thrust_cmd[1].item() == pytest.approx(0.2 * cfg.thrust_min)  # u=-0.8 -> 20%
    assert thrust_cmd[2].item() == pytest.approx(cfg.min_throttle * cfg.max_thrust)


def test_thrust_vector_is_unit_norm():
    delta = torch.tensor([[0.1, -0.05], [0.0, 0.0], [-0.13, 0.13]])
    vec = thrust_vector_body(delta)
    assert torch.allclose(torch.linalg.vector_norm(vec, dim=-1), torch.ones(3), atol=1e-6)


def test_free_fall_acceleration_is_gravity():
    cfg = dataclasses.replace(RocketPhysicsConfig(), drag_coefficient=0.0)
    state = make_state(z=1_000.0, cfg=cfg)
    fault = nominal_fault()
    fault[:, FaultParamsLayout.THRUST_SCALE] = 0.0  # engine off
    action = torch.zeros(1, 3)

    deriv = compute_derivatives(state, action, fault, cfg)
    assert deriv[0, 5].item() == pytest.approx(-cfg.gravity, abs=1e-6)
    assert torch.allclose(deriv[0, 6:10], torch.zeros(4), atol=1e-7)

    nxt = rk4_step(compute_derivatives, state, action, fault, cfg.dt, cfg)
    assert (nxt[0, 5] - state[0, 5]).item() / cfg.dt == pytest.approx(-cfg.gravity, abs=1e-6)
    assert nxt[0, 2].item() == pytest.approx(1_000.0 - 0.5 * cfg.gravity * cfg.dt**2, abs=1e-4)


def test_ballistic_trajectory_matches_analytic_parabola():
    cfg = dataclasses.replace(RocketPhysicsConfig(), drag_coefficient=0.0)
    state = make_state(z=2_000.0, vz=10.0, cfg=cfg)
    fault = nominal_fault()
    fault[:, FaultParamsLayout.THRUST_SCALE] = 0.0
    action = torch.zeros(1, 3)
    steps = 100
    for _ in range(steps):
        state = rk4_step(compute_derivatives, state, action, fault, cfg.dt, cfg)
    t = steps * cfg.dt
    expected_z = 2_000.0 + 10.0 * t - 0.5 * cfg.gravity * t**2
    expected_vz = 10.0 - cfg.gravity * t
    assert state[0, 2].item() == pytest.approx(expected_z, abs=1e-3)
    assert state[0, 5].item() == pytest.approx(expected_vz, abs=1e-4)


def test_mass_conservation_with_engine_off():
    cfg = RocketPhysicsConfig()
    state = make_state(cfg=cfg)
    fault = nominal_fault()
    fault[:, FaultParamsLayout.THRUST_SCALE] = 0.0
    action = torch.zeros(1, 3)
    deriv = compute_derivatives(state, action, fault, cfg)
    assert deriv[0, 13].item() == pytest.approx(0.0, abs=1e-12)
    for _ in range(20):
        state = rk4_step(compute_derivatives, state, action, fault, cfg.dt, cfg)
    assert state[0, 13].item() == pytest.approx(cfg.wet_mass, abs=1e-6)


def test_mass_depletion_rate():
    cfg = RocketPhysicsConfig()
    thrust = 400_000.0
    state = make_state(thrust=thrust, cfg=cfg)
    deriv = compute_derivatives(state, torch.zeros(1, 3), nominal_fault(), cfg)
    expected = -thrust / (cfg.specific_impulse * cfg.gravity)
    assert deriv[0, 13].item() == pytest.approx(expected, rel=1e-6)


def test_engine_and_gimbal_first_order_lag():
    cfg = RocketPhysicsConfig()
    state = make_state(thrust=0.0, gimbal=(0.0, 0.0), cfg=cfg)
    action = torch.tensor([[1.0, 0.0, 1.0]])
    deriv = compute_derivatives(state, action, nominal_fault(), cfg)
    assert deriv[0, 16].item() == pytest.approx((cfg.max_thrust - 0.0) / cfg.engine_tau, rel=1e-6)
    assert deriv[0, 14].item() == pytest.approx(cfg.gimbal_limit / cfg.act_tau, rel=1e-6)
    assert deriv[0, 15].item() == pytest.approx(0.0, abs=1e-9)


def test_gimbal_torque_signs():
    cfg = RocketPhysicsConfig()
    state = make_state(z=5_000.0, thrust=4.0e5, gimbal=(0.1, 0.0), cfg=cfg)
    deriv = compute_derivatives(state, torch.zeros(1, 3), nominal_fault(), cfg)
    # Positive dy tilts thrust towards +y_B -> positive moment about +x.
    assert deriv[0, 10].item() > 0.0

    state = make_state(z=5_000.0, thrust=4.0e5, gimbal=(0.0, 0.1), cfg=cfg)
    deriv = compute_derivatives(state, torch.zeros(1, 3), nominal_fault(), cfg)
    # Positive dz tilts thrust towards +x_B -> negative moment about +y.
    assert deriv[0, 11].item() < 0.0


def test_com_offset_produces_parasitic_moment():
    cfg = RocketPhysicsConfig()
    state = make_state(z=5_000.0, thrust=4.0e5, cfg=cfg)
    fault = nominal_fault()
    fault[:, FaultParamsLayout.COM_OFFSET_Z] = 0.2
    deriv = compute_derivatives(state, torch.zeros(1, 3), fault, cfg)
    nominal = compute_derivatives(state, torch.zeros(1, 3), nominal_fault(), cfg)
    assert deriv[0, 10].abs().item() > 1e-3
    assert nominal[0, 10].abs().item() < 1e-9


def test_gimbal_from_torque_inverts_moment():
    cfg = RocketPhysicsConfig()
    thrust = torch.tensor([4.0e5])
    tau = torch.tensor([[5_000.0, -3_000.0]])
    delta = gimbal_from_torque(tau, thrust, cfg)
    arm = cfg.engine_moment_arm
    reconstructed = torch.stack([arm * thrust[0] * delta[0, 0], -arm * thrust[0] * delta[0, 1]]).unsqueeze(0)
    assert torch.allclose(reconstructed, tau, rtol=1e-4, atol=1e-2)


def test_quaternion_and_bounds_preserved_over_500_steps():
    cfg = RocketPhysicsConfig()
    torch.manual_seed(0)
    state = make_state(batch=4, z=8_000.0, vz=0.0, cfg=cfg)
    fault = nominal_fault(4)
    for _ in range(500):
        action = torch.rand(4, 3) * 2 - 1
        state = rk4_step(compute_derivatives, state, action, fault, cfg.dt, cfg)
    norms = torch.linalg.vector_norm(quat_normalize(state[:, 6:10]), dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-6)
    assert torch.isfinite(state).all()
    assert torch.all(state[:, 13] >= cfg.dry_mass - 1e-3)
    assert torch.all(state[:, 13] <= cfg.wet_mass + 1e-3)
    assert torch.all(state[:, 16] >= 0.0)
    assert torch.all(state[:, 14:16].abs() <= cfg.gimbal_limit + 1e-6)
