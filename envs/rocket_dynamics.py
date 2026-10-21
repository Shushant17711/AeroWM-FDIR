"""Batched, ``torch.compile``-friendly 6-DOF rigid-body rocket dynamics.

State layout (17):

===========  ==================================================
index        quantity
===========  ==================================================
0:3          inertial position ``r`` [m]
3:6          inertial velocity ``v`` [m/s]
6:10         body->inertial quaternion ``q`` (w, x, y, z)
10:13        body angular rate ``omega`` [rad/s]
13           total mass ``m`` [kg]
14:16        actual gimbal deflections ``(dy, dz)`` [rad]
16           actual engine thrust ``T`` [N]
===========  ==================================================

Body frame: ``+z`` is the thrust/longitudinal axis.  A positive gimbal
``dy`` tilts thrust towards ``+y_B`` and produces a positive pitch moment;
``dz`` tilts thrust towards ``+x_B`` and produces a negative yaw moment.
"""

from __future__ import annotations

import torch

from configs.schema import RocketPhysicsConfig
from envs.base_types import RocketState
from envs.quaternion_ops import (
    quat_derivative,
    quat_normalize,
    quat_rotate,
    quat_to_rotmat,
)


def gravity_vector(cfg: RocketPhysicsConfig, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, -cfg.gravity], dtype=dtype, device=device)


def atmosphere_density(altitude: torch.Tensor, cfg: RocketPhysicsConfig) -> torch.Tensor:
    """Exponential atmosphere ``rho(z) = rho0 * exp(-max(z, 0) / H)``."""
    z = altitude.clamp_min(0.0)
    return cfg.rho0 * torch.exp(-z / cfg.atmosphere_scale_height)


def dynamic_pressure(v_rel: torch.Tensor, altitude: torch.Tensor, cfg: RocketPhysicsConfig) -> torch.Tensor:
    speed = torch.linalg.vector_norm(v_rel, dim=-1)
    return 0.5 * atmosphere_density(altitude, cfg) * speed * speed


def inertia_tensor(mass: torch.Tensor, cfg: RocketPhysicsConfig) -> torch.Tensor:
    """Diagonal inertia **[B, 3]** (roll, pitch, yaw), proportional to mass."""
    spec = torch.tensor(cfg.specific_inertia, dtype=mass.dtype, device=mass.device)
    return mass.unsqueeze(-1) * spec


def actuator_commands(
    gimbal: torch.Tensor,
    action: torch.Tensor,
    fault_params: torch.Tensor,
    cfg: RocketPhysicsConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map normalized actions and fault parameters to physical commands.

    ``fault_params`` layout is :class:`configs.schema.FaultParamsLayout`:
    ``[thrust_scale, bias_y, bias_z, com_y, com_z, stuck_y, stuck_z]``.
    Inactive faults must be encoded as ``thrust_scale=1`` and zero
    bias/stuck/offset entries (the injector guarantees this).

    Returns ``(delta_cmd, thrust_cmd, rcs_torque)``.  ``rcs_torque`` is the
    [N*m] torque produced by the backup cold-gas thrusters about the body
    x/y axes.  They latch on an axis only when the primary TVC actuator for
    that axis is stuck or driven into saturation, so the nominal vehicle is
    bit-for-bit unchanged.
    """
    u = action.clamp(-1.0, 1.0)
    raw_cmd = u[..., :2] * cfg.gimbal_limit + fault_params[..., 1:3]
    stuck = fault_params[..., 5:7] > 0.5
    # A stuck actuator ignores the command and stays at its current angle.
    delta_cmd = torch.where(stuck, gimbal, raw_cmd)
    delta_cmd = delta_cmd.clamp(-cfg.gimbal_limit, cfg.gimbal_limit)

    # Backup cold-gas ACS: same normalized command, but *only* on an axis whose
    # primary actuator has locked.  Deliberately not engaged on saturation:
    # adding 500 kN*m of unmodelled authority at the gimbal limit destabilises
    # the nominal attitude loop.  Zero in nominal flight.
    if cfg.rcs_enable:
        engaged = stuck.to(u.dtype)
        rcs_torque = u[..., :2] * cfg.rcs_max_torque * engaged
    else:
        rcs_torque = torch.zeros_like(u[..., :2])

    # Throttle: the usable firing band is [min_throttle, max_throttle]; the
    # bottom ``cutoff_fraction`` of the action range commands a shutdown.
    # Continuous, monotone throttle map: u=-1 shutdown, u=0 minimum sustainable
    # firing thrust, u=+1 full thrust.  The sustainable band is exactly
    # [min_throttle, max_throttle] * T_max as specified, while the shutdown
    # region is a *continuous* ramp so the action remains imitable by a smooth
    # policy (a hard cutoff produced bang-bang chattering at the boundary).
    u_t = u[..., 2]
    thrust_norm = torch.where(
        u_t < 0.0,
        cfg.min_throttle * (u_t + 1.0),
        cfg.min_throttle + u_t * (cfg.max_throttle - cfg.min_throttle),
    )
    thrust_cmd = thrust_norm * cfg.max_thrust * fault_params[..., 0]
    thrust_cmd = thrust_cmd.clamp(0.0, cfg.max_thrust)
    return delta_cmd, thrust_cmd, rcs_torque


def thrust_vector_body(delta: torch.Tensor) -> torch.Tensor:
    """Unit thrust direction in the body frame (norm exactly 1)."""
    dy, dz = delta.unbind(-1)
    return torch.stack(
        [torch.cos(dy) * torch.sin(dz), torch.sin(dy), torch.cos(dy) * torch.cos(dz)], dim=-1
    )


def gimbal_from_torque(
    torque_xy: torch.Tensor, thrust: torch.Tensor, cfg: RocketPhysicsConfig, eps: float = 1.0
) -> torch.Tensor:
    """Invert the small-angle thrust moment for a controller.

    ``tau_x ~= L * T * dy`` and ``tau_y ~= -L * T * dz``.
    """
    denom = (cfg.engine_moment_arm * thrust).clamp_min(eps)
    dy = torque_xy[..., 0] / denom
    dz = -torque_xy[..., 1] / denom
    return torch.stack([dy, dz], dim=-1).clamp(-cfg.gimbal_limit, cfg.gimbal_limit)


def compute_derivatives(
    state: torch.Tensor,
    action: torch.Tensor,
    fault_params: torch.Tensor,
    cfg: RocketPhysicsConfig,
) -> torch.Tensor:
    """Continuous-time state derivative ``dx/dt = f(x, u, c_fault)``.

    Args:
        state: ``[..., 17]`` packed state.
        action: ``[..., 3]`` normalized ``(u_dy, u_dz, u_throttle)``.
        fault_params: ``[..., 7]`` packed fault parameters.
    """
    r = state[..., 0:3]
    v = state[..., 3:6]
    q = state[..., 6:10]
    omega = state[..., 10:13]
    mass = state[..., 13]
    gimbal = state[..., 14:16]
    thrust = state[..., 16]

    delta_cmd, thrust_cmd, rcs_torque = actuator_commands(gimbal, action, fault_params, cfg)

    # Actuator lag.
    d_gimbal = (delta_cmd - gimbal) / cfg.act_tau
    d_thrust = (thrust_cmd - thrust) / cfg.engine_tau

    # Forces.
    f_body = thrust.unsqueeze(-1) * thrust_vector_body(gimbal)
    f_inertial = quat_rotate(q, f_body)

    v_rel = v - torch.tensor(cfg.wind, dtype=v.dtype, device=v.device)
    rho = atmosphere_density(r[..., 2], cfg)
    speed = torch.linalg.vector_norm(v_rel, dim=-1, keepdim=True)
    drag = -0.5 * rho.unsqueeze(-1) * speed * cfg.drag_coefficient * cfg.ref_area * v_rel

    mass_safe = mass.clamp_min(cfg.dry_mass).unsqueeze(-1)
    accel = (f_inertial + drag) / mass_safe + gravity_vector(cfg, v.dtype, v.device)

    # Moments: thrust moment arm (CoM -> gimbal pivot) plus parasitic CoM offset.
    arm = torch.stack(
        [fault_params[..., 3], fault_params[..., 4], torch.full_like(mass, -cfg.engine_moment_arm)],
        dim=-1,
    )
    torque = torch.cross(arm, f_body, dim=-1)
    # Pure-torque cold-gas ACS on the two transverse axes.
    torque[..., :2] = torque[..., :2] + rcs_torque
    damping = torch.tensor(cfg.angular_damping, dtype=v.dtype, device=v.device)
    torque = torque - damping * omega

    inertia = inertia_tensor(mass, cfg)
    gyroscopic = torch.cross(omega, inertia * omega, dim=-1)
    d_omega = (torque - gyroscopic) / inertia

    d_q = quat_derivative(q, omega)
    d_mass = -thrust / cfg.exhaust_velocity

    return torch.cat([v, accel, d_q, d_omega, d_mass.unsqueeze(-1), d_gimbal, d_thrust.unsqueeze(-1)], dim=-1)


def unpack_state(state: torch.Tensor) -> RocketState:
    return RocketState.from_tensor(state)


def pack_state(state: RocketState) -> torch.Tensor:
    return state.to_tensor()


def tilt_angle_deg(state: torch.Tensor) -> torch.Tensor:
    """Angle between the body thrust axis and inertial up, in degrees."""
    q = quat_normalize(state[..., 6:10])
    cos_tilt = (1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_tilt))


def rotation_matrix(state: torch.Tensor) -> torch.Tensor:
    return quat_to_rotmat(quat_normalize(state[..., 6:10]))
