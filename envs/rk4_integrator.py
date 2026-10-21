"""Fourth-order Runge-Kutta integrator for the batched 6-DOF dynamics.

The solver is generic over the right-hand-side function so it can be used for
the rocket, for unit-test ballistic problems, and (potentially) for other
rigid bodies.  It never allocates per-environment Python objects, so a batch
of ``B=1024`` environments integrates with a fixed number of tensor kernels.
"""

from __future__ import annotations

from typing import Callable

import torch

from configs.schema import RocketPhysicsConfig
from envs.quaternion_ops import quat_normalize
from envs.rocket_dynamics import compute_derivatives

DynamicsFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, RocketPhysicsConfig], torch.Tensor]


def post_step_clamp(state: torch.Tensor, cfg: RocketPhysicsConfig) -> torch.Tensor:
    """Renormalize attitude and enforce physical bounds after one RK4 step."""
    q = quat_normalize(state[..., 6:10])
    mass = state[..., 13].clamp(cfg.dry_mass, cfg.wet_mass)
    gimbal = state[..., 14:16].clamp(-cfg.gimbal_limit, cfg.gimbal_limit)
    thrust = state[..., 16].clamp(0.0, cfg.max_thrust)
    return torch.cat(
        [state[..., 0:6], q, state[..., 10:13], mass.unsqueeze(-1), gimbal, thrust.unsqueeze(-1)],
        dim=-1,
    )


def rk4_step(
    dynamics_fn: DynamicsFn = compute_derivatives,
    state: torch.Tensor | None = None,
    action: torch.Tensor | None = None,
    fault_params: torch.Tensor | None = None,
    dt: float | None = None,
    cfg: RocketPhysicsConfig | None = None,
) -> torch.Tensor:
    """Single classical RK4 step with quaternion renormalization.

    Can be called either positionally as
    ``rk4_step(dynamics_fn, state, action, fault_params, dt, cfg)`` or with
    keywords.  The default dynamics is the rocket 6-DOF model.
    """
    if state is None or action is None or fault_params is None:
        raise ValueError("state, action and fault_params are required")
    cfg = cfg or RocketPhysicsConfig()
    h = cfg.dt if dt is None else float(dt)

    k1 = dynamics_fn(state, action, fault_params, cfg)
    k2 = dynamics_fn(state + 0.5 * h * k1, action, fault_params, cfg)
    k3 = dynamics_fn(state + 0.5 * h * k2, action, fault_params, cfg)
    k4 = dynamics_fn(state + h * k3, action, fault_params, cfg)

    next_state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return post_step_clamp(next_state, cfg)


def rk4_substeps(
    dynamics_fn: DynamicsFn = compute_derivatives,
    state: torch.Tensor | None = None,
    action: torch.Tensor | None = None,
    fault_params: torch.Tensor | None = None,
    substeps: int = 1,
    cfg: RocketPhysicsConfig | None = None,
) -> torch.Tensor:
    """Integrate one guidance interval using ``substeps`` smaller RK4 steps."""
    if state is None or action is None or fault_params is None:
        raise ValueError("state, action and fault_params are required")
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    cfg = cfg or RocketPhysicsConfig()
    h = cfg.dt / substeps
    for _ in range(substeps):
        state = rk4_step(dynamics_fn, state, action, fault_params, h, cfg)
    return state


def ballistic_step(
    dynamics_fn: DynamicsFn,
    state: torch.Tensor,
    action: torch.Tensor,
    fault_params: torch.Tensor,
    dt: float,
    cfg: RocketPhysicsConfig,
) -> torch.Tensor:
    """Alias kept for readability in tests of generic integrators."""
    return rk4_step(dynamics_fn, state, action, fault_params, dt, cfg)


def integrate_rollout(
    state: torch.Tensor,
    actions: torch.Tensor,
    fault_params: torch.Tensor,
    cfg: RocketPhysicsConfig | None = None,
) -> torch.Tensor:
    """Integrate a contiguous action sequence ``[B, T, 3]`` -> states ``[B, T+1, 17]``."""
    cfg = cfg or RocketPhysicsConfig()
    states = [state]
    for t in range(actions.shape[1]):
        state = rk4_step(compute_derivatives, state, actions[:, t], fault_params, cfg.dt, cfg)
        states.append(state)
    return torch.stack(states, dim=1)
