"""Task 2.3: deterministic fault scheduling and physical fault effects."""

from __future__ import annotations

import math

import pytest
import torch

from configs.schema import (
    FaultInjectionSchema,
    FaultMode,
    FaultParamsLayout,
    RocketPhysicsConfig,
)
from envs.fault_injector import FaultInjector
from envs.rocket_dynamics import actuator_commands, compute_derivatives


def make_injector() -> FaultInjector:
    return FaultInjector(
        FaultInjectionSchema(), RocketPhysicsConfig(), device="cpu", generator=torch.Generator().manual_seed(0)
    )


def forced(injector: FaultInjector, mode: int, onset: int = 10, severity: float = 0.3):
    return injector.reset(
        1,
        mode=torch.tensor([mode]),
        onset=torch.tensor([onset]),
        severity=torch.tensor([severity], dtype=torch.float32),
    )


def nominal_state(thrust: float = 2.0e5, gimbal=(0.0, 0.0)) -> torch.Tensor:
    state = torch.zeros(1, 17)
    state[0, 2] = 5_000.0
    state[0, 6] = 1.0
    state[0, 13] = 15_000.0
    state[0, 14] = gimbal[0]
    state[0, 15] = gimbal[1]
    state[0, 16] = thrust
    return state


def test_fault_is_inactive_before_onset():
    inj = make_injector()
    for mode in range(1, 5):
        fault = forced(inj, mode, onset=10)
        before = inj.update(fault, torch.tensor([9]))
        at_onset = inj.update(fault, torch.tensor([10]))
        assert not bool(before.active[0])
        assert bool(at_onset.active[0])
        assert before.applied_params()[0, FaultParamsLayout.THRUST_SCALE].item() == pytest.approx(1.0)
        assert torch.all(before.applied_params()[0, 1:] == 0.0)


def test_labels_and_context_before_and_after_onset():
    inj = make_injector()
    fault = forced(inj, FaultMode.LOE_THRUST, onset=5)
    pre = inj.update(fault, torch.tensor([4]))
    post = inj.update(fault, torch.tensor([5]))
    assert FaultInjector.labels(pre)[0].item() == int(FaultMode.NOMINAL)
    assert FaultInjector.labels(post)[0].item() == int(FaultMode.LOE_THRUST)
    assert torch.allclose(FaultInjector.context(pre)[0], torch.eye(5)[0])
    assert torch.allclose(FaultInjector.context(post)[0], torch.eye(5)[1])


def test_loe_scales_thrust_command():
    inj = make_injector()
    cfg = RocketPhysicsConfig()
    fault = forced(inj, FaultMode.LOE_THRUST, onset=0, severity=0.35)
    fault = inj.update(fault, torch.tensor([0]))
    action = torch.tensor([[0.0, 0.0, 1.0]])
    _, thrust_cmd, rcs = actuator_commands(torch.zeros(1, 2), action, fault.params, cfg)
    assert thrust_cmd.item() == pytest.approx(0.65 * cfg.max_thrust, rel=1e-6)
    assert fault.params[0, FaultParamsLayout.THRUST_SCALE].item() == pytest.approx(0.65)


def test_gimbal_bias_offsets_command():
    inj = make_injector()
    cfg = RocketPhysicsConfig()
    fault = forced(inj, FaultMode.GIMBAL_BIAS, onset=0)
    fault = inj.update(fault, torch.tensor([0]))
    action = torch.zeros(1, 3)
    delta, _, rcs = actuator_commands(torch.zeros(1, 2), action, fault.params, cfg)
    expected = math.radians(FaultInjectionSchema().gimbal_bias_deg)
    assert delta.abs().max().item() == pytest.approx(expected, rel=1e-5)
    assert int((delta.abs() > 1e-6).sum()) == 1  # bias is applied to one axis
    assert delta.abs().max() <= cfg.gimbal_limit


def test_gimbal_stuck_freezes_actuator():
    inj = make_injector()
    cfg = RocketPhysicsConfig()
    fault = forced(inj, FaultMode.GIMBAL_STUCK, onset=0)
    fault = inj.update(fault, torch.tensor([0]))
    state = nominal_state(gimbal=(0.05, 0.05))
    action = torch.tensor([[1.0, 1.0, 0.8]])
    deriv = compute_derivatives(state, action, fault.params, cfg)
    stuck_y = bool(fault.params[0, FaultParamsLayout.STUCK_Y] > 0.5)
    if stuck_y:
        assert deriv[0, 14].item() == pytest.approx(0.0, abs=1e-9)
        assert deriv[0, 15].abs().item() > 1e-6
    else:
        assert deriv[0, 15].item() == pytest.approx(0.0, abs=1e-9)
        assert deriv[0, 14].abs().item() > 1e-6


def test_com_shift_produces_parasitic_torque_only_after_onset():
    inj = make_injector()
    cfg = RocketPhysicsConfig()
    fault = forced(inj, FaultMode.COM_SHIFT, onset=3)
    state = nominal_state()
    action = torch.zeros(1, 3)
    before = compute_derivatives(state, action, inj.update(fault, torch.tensor([2])).applied_params(), cfg)
    after = compute_derivatives(state, action, inj.update(fault, torch.tensor([3])).applied_params(), cfg)
    assert before[0, 10].abs().item() < 1e-9
    assert before[0, 11].abs().item() < 1e-9
    assert after[0, 10:12].abs().max().item() > 1e-3


def test_nominal_episodes_are_always_inert():
    inj = make_injector()
    fault = inj.reset(64, probability=0.0)
    assert torch.all(fault.mode == int(FaultMode.NOMINAL))
    updated = inj.update(fault, torch.full((64,), 10_000))
    assert not bool(updated.active.any())
    assert torch.all(updated.params[:, FaultParamsLayout.THRUST_SCALE] == 1.0)
    assert torch.all(updated.applied_params()[:, FaultParamsLayout.THRUST_SCALE] == 1.0)


def test_stochastic_sampling_has_expected_fault_rate():
    inj = FaultInjector(FaultInjectionSchema(probability=0.65), generator=torch.Generator().manual_seed(0))
    fault = inj.reset(20_000)
    rate = float((fault.mode != 0).float().mean())
    assert 0.62 < rate < 0.68
    onset = fault.onset.float()
    assert onset.min() >= FaultInjectionSchema().onset_range[0]
    assert onset.max() < FaultInjectionSchema().onset_range[1]
