"""Task 2.2: RK4 integrator accuracy, generic signature and throughput gate."""

from __future__ import annotations

import dataclasses
import time

import pytest
import torch

from configs.schema import FaultParamsLayout, RocketPhysicsConfig
from envs.rk4_integrator import integrate_rollout, rk4_step, rk4_substeps
from envs.rocket_dynamics import compute_derivatives


def nominal_fault(batch: int = 1) -> torch.Tensor:
    fault = torch.zeros(batch, FaultParamsLayout.DIM)
    fault[:, FaultParamsLayout.THRUST_SCALE] = 1.0
    return fault


def vertical_state(batch: int = 1, z: float = 500.0, vz: float = 0.0) -> torch.Tensor:
    state = torch.zeros(batch, 17)
    state[:, 2] = z
    state[:, 5] = vz
    state[:, 6] = 1.0
    state[:, 13] = RocketPhysicsConfig().wet_mass
    return state


def test_keyword_call_signature():
    cfg = RocketPhysicsConfig()
    state = vertical_state()
    out = rk4_step(
        dynamics_fn=compute_derivatives,
        state=state,
        action=torch.zeros(1, 3),
        fault_params=nominal_fault(),
        dt=cfg.dt,
        cfg=cfg,
    )
    assert out.shape == state.shape
    assert torch.isfinite(out).all()


def test_generic_dynamics_function():
    """The solver must accept an arbitrary ``f(x, u, c, cfg)`` callable."""

    def harmonic(state, action, fault, cfg):
        x = state[..., 0]
        v = state[..., 1]
        dx = torch.stack([v, -x, torch.zeros_like(x)], dim=-1)
        return torch.cat([dx, torch.zeros_like(state[..., 3:])], dim=-1)

    state = torch.zeros(1, 17)
    state[0, 0] = 1.0  # x(0)=1, v(0)=0 -> x(t)=cos(t), period 2*pi
    cfg = RocketPhysicsConfig()
    steps = int(round((2 * torch.pi) / cfg.dt))
    for _ in range(steps):
        state = rk4_step(harmonic, state, torch.zeros(1, 3), nominal_fault(), cfg.dt, cfg)
    t = steps * cfg.dt
    assert state[0, 0].item() == pytest.approx(torch.cos(torch.tensor(t)).item(), abs=1e-5)
    assert state[0, 1].item() == pytest.approx(-torch.sin(torch.tensor(t)).item(), abs=1e-5)
    energy = state[0, 0] ** 2 + state[0, 1] ** 2
    assert energy.item() == pytest.approx(1.0, abs=1e-5)


def test_substeps_improve_accuracy():
    """With aerodynamic drag the RK4 truncation error is measurable; halving
    the step size (via substeps) must reduce the error against a fine
    float64 reference."""
    cfg = dataclasses.replace(RocketPhysicsConfig(), dt=0.2)
    state0 = vertical_state(z=2_000.0, vz=-200.0).double()
    fault = nominal_fault().double()
    fault[:, FaultParamsLayout.THRUST_SCALE] = 0.0
    action = torch.zeros(1, 3, dtype=torch.float64)
    steps = 25

    reference = state0
    for _ in range(steps):
        reference = rk4_substeps(compute_derivatives, reference, action, fault, substeps=32, cfg=cfg)

    coarse = state0
    for _ in range(steps):
        coarse = rk4_step(compute_derivatives, coarse, action, fault, cfg.dt, cfg)

    fine = state0
    for _ in range(steps):
        fine = rk4_substeps(compute_derivatives, fine, action, fault, substeps=4, cfg=cfg)

    coarse_err = torch.linalg.vector_norm(coarse[:, 0:6] - reference[:, 0:6]).item()
    fine_err = torch.linalg.vector_norm(fine[:, 0:6] - reference[:, 0:6]).item()
    assert fine_err < coarse_err / 50.0, (coarse_err, fine_err)


def test_rollout_helper_matches_manual_loop():
    cfg = RocketPhysicsConfig()
    state = vertical_state(batch=2, z=2_000.0)
    actions = torch.zeros(2, 8, 3)
    actions[:, :, 2] = 0.5
    rollout = integrate_rollout(state, actions, nominal_fault(2), cfg)
    assert rollout.shape == (2, 9, 17)
    manual = state
    for t in range(8):
        manual = rk4_step(compute_derivatives, manual, actions[:, t], nominal_fault(2), cfg.dt, cfg)
    assert torch.allclose(rollout[:, -1], manual, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.slow
def test_throughput_gate_at_1024_envs(gpu_device):
    cfg = RocketPhysicsConfig()
    batch = 1024
    state = vertical_state(batch=batch, z=5_000.0)
    state = state.to(gpu_device)
    action = torch.zeros(batch, 3, device=gpu_device)
    action[:, 2] = 0.2
    fault = nominal_fault(batch).to(gpu_device)

    for _ in range(10):  # warm-up / lazy init
        state = rk4_step(compute_derivatives, state, action, fault, cfg.dt, cfg)
    torch.cuda.synchronize()

    steps = 200
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        state = rk4_step(compute_derivatives, state, action, fault, cfg.dt, cfg)
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / 1_000.0
    sps = batch * steps / elapsed
    print(f"\nRK4 throughput: {sps:,.0f} SPS at B={batch}")
    assert sps >= 200_000, f"throughput {sps:,.0f} SPS below the 200k gate"


@pytest.mark.gpu
@pytest.mark.slow
def test_compiled_throughput_gate(gpu_device):
    cfg = RocketPhysicsConfig()
    batch = 1024
    state = vertical_state(batch=batch, z=5_000.0).to(gpu_device)
    action = torch.zeros(batch, 3, device=gpu_device)
    action[:, 2] = 0.2
    fault = nominal_fault(batch).to(gpu_device)

    compiled = torch.compile(compute_derivatives, dynamic=False)
    for _ in range(5):
        state = rk4_step(compiled, state, action, fault, cfg.dt, cfg)
    torch.cuda.synchronize()
    steps = 100
    start = time.perf_counter()
    for _ in range(steps):
        state = rk4_step(compiled, state, action, fault, cfg.dt, cfg)
    torch.cuda.synchronize()
    sps = batch * steps / (time.perf_counter() - start)
    print(f"\nCompiled RK4 throughput: {sps:,.0f} SPS")
    assert sps >= 200_000
