"""Vectorized quaternion algebra on the last tensor dimension.

Quaternions are ordered ``(w, x, y, z)`` and rotate vectors from body to
inertial.  Every function is batched over arbitrary leading dimensions and is
safe under ``torch.compile`` (no Python-level branching on tensor values).
"""

from __future__ import annotations

import torch

EPS = 1e-12


def quat_normalize(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Return ``q / ||q||`` with a numerically safe denominator."""
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    return q / norm.clamp_min(eps)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a ⊗ b`` (rotation composition: apply ``b`` then ``a``)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Body -> inertial direction cosine matrix ``R(q)`` of shape ``[..., 3, 3]``."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    row0 = torch.stack(
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        dim=-1,
    )
    row1 = torch.stack(
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        dim=-1,
    )
    row2 = torch.stack(
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        dim=-1,
    )
    return torch.stack([row0, row1, row2], dim=-2)


def quat_derivative(q: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Quaternion kinematics ``q_dot = 0.5 * Omega(omega) q`` for body rates."""
    w, x, y, z = q.unbind(-1)
    wx, wy, wz = omega.unbind(-1)
    return 0.5 * torch.stack(
        [
            -wx * x - wy * y - wz * z,
            wx * w + wz * y - wy * z,
            wy * w - wz * x + wx * z,
            wz * w + wy * x - wx * y,
        ],
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v_body: torch.Tensor) -> torch.Tensor:
    """Rotate a body-frame vector into the inertial frame."""
    rot = quat_to_rotmat(q)
    return torch.einsum("...ij,...j->...i", rot, v_body)


def quat_rotate_inverse(q: torch.Tensor, v_inertial: torch.Tensor) -> torch.Tensor:
    """Rotate an inertial-frame vector into the body frame."""
    rot = quat_to_rotmat(q)
    return torch.einsum("...ji,...j->...i", rot, v_inertial)


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Quaternion from a (not necessarily unit) axis and an angle in radians."""
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-9)
    half = 0.5 * angle
    return torch.cat([torch.cos(half).unsqueeze(-1), torch.sin(half).unsqueeze(-1) * axis], dim=-1)


def quat_tilt_cosine(q: torch.Tensor) -> torch.Tensor:
    """Cosine of the angle between the body ``+z`` (thrust) axis and inertial ``+z``."""
    w, x, y, z = q.unbind(-1)
    return 1.0 - 2.0 * (x * x + y * y)


def rotation_vector(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Log map ``SO(3) -> R^3`` (axis * angle), with sign continuity at ``w=0``."""
    w = q[..., :1]
    v = q[..., 1:]
    sign = torch.where(w >= 0.0, 1.0, -1.0)
    v = v * sign
    sin_half = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, (w * sign).abs().clamp_min(0.0))
    scale = torch.where(sin_half > eps, angle / sin_half.clamp_min(eps), 2.0 * torch.ones_like(sin_half))
    return v * scale


def quat_error(q_target: torch.Tensor, q_current: torch.Tensor) -> torch.Tensor:
    """Body-frame rotation vector that takes ``q_current`` to ``q_target``."""
    q_current_inv = quat_conjugate(quat_normalize(q_current))
    q_err = quat_multiply(q_current_inv, quat_normalize(q_target))
    return rotation_vector(quat_normalize(q_err))


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Shortest-path spherical interpolation; ``t`` broadcasts over the batch."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp_min(1e-8)
    w0 = torch.sin((1.0 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    return quat_normalize(w0 * q0 + w1 * q1)
