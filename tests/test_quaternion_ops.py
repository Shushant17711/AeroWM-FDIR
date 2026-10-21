"""Batched quaternion algebra: correctness, orthogonality and norm safety."""

from __future__ import annotations

import math

import pytest
import torch

from envs.quaternion_ops import (
    quat_conjugate,
    quat_derivative,
    quat_error,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_rotate_inverse,
    quat_slerp,
    quat_tilt_cosine,
    quat_to_rotmat,
    rotation_vector,
)


def identity(b: int = 1) -> torch.Tensor:
    q = torch.zeros(b, 4)
    q[:, 0] = 1.0
    return q


def test_normalize_handles_zero_vector():
    q = quat_normalize(torch.zeros(3, 4))
    assert torch.isfinite(q).all()
    assert q.shape == (3, 4)


def test_identity_rotation():
    rot = quat_to_rotmat(identity(4))
    assert torch.allclose(rot, torch.eye(3).expand(4, 3, 3), atol=1e-6)


def test_rotmat_is_orthonormal():
    torch.manual_seed(0)
    q = quat_normalize(torch.randn(64, 4))
    rot = quat_to_rotmat(q)
    eye = torch.einsum("bij,bjk->bik", rot, rot.transpose(-1, -2))
    assert torch.allclose(eye, torch.eye(3).expand(64, 3, 3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(rot), torch.ones(64), atol=1e-5)


def test_rotate_matches_rotmat():
    torch.manual_seed(1)
    q = quat_normalize(torch.randn(8, 4))
    v = torch.randn(8, 3)
    assert torch.allclose(quat_rotate(q, v), torch.einsum("bij,bj->bi", quat_to_rotmat(q), v), atol=1e-6)
    back = quat_rotate_inverse(q, quat_rotate(q, v))
    assert torch.allclose(back, v, atol=1e-5)


def test_rotate_about_z_90deg():
    q = quat_from_axis_angle(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([math.pi / 2]))
    out = quat_rotate(q, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_conjugate_is_inverse():
    torch.manual_seed(2)
    q = quat_normalize(torch.randn(16, 4))
    prod = quat_multiply(q, quat_conjugate(q))
    assert torch.allclose(prod, identity(16), atol=1e-6)


def test_multiply_matches_matrix_product():
    torch.manual_seed(3)
    a = quat_normalize(torch.randn(10, 4))
    b = quat_normalize(torch.randn(10, 4))
    ra = quat_to_rotmat(a)
    rb = quat_to_rotmat(b)
    assert torch.allclose(quat_to_rotmat(quat_multiply(a, b)), ra @ rb, atol=1e-5)


def test_derivative_of_identity_is_body_rate_halves():
    dq = quat_derivative(identity(1), torch.tensor([[0.0, 0.0, 2.0]]))
    # d/dt [cos(t/2), 0, 0, sin(t/2)] at t=0 -> [0, 0, 0, 1]
    assert torch.allclose(dq, torch.tensor([[0.0, 0.0, 0.0, 1.0]]), atol=1e-6)


def test_tilt_cosine():
    assert quat_tilt_cosine(identity(1)).item() == pytest.approx(1.0)
    q = quat_from_axis_angle(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([math.pi / 2]))
    assert quat_tilt_cosine(q).item() == pytest.approx(0.0, abs=1e-6)


def test_error_and_rotation_vector_roundtrip():
    q0 = quat_normalize(torch.tensor([[0.9, 0.1, -0.2, 0.3]]))
    q1 = quat_normalize(torch.tensor([[0.2, -0.5, 0.7, 0.1]]))
    err = quat_error(q1, q0)
    recovered = quat_multiply(q0, quat_from_axis_angle(err, torch.linalg.vector_norm(err, dim=-1)))
    assert torch.allclose(quat_normalize(recovered), quat_normalize(q1), atol=1e-5) or torch.allclose(
        quat_normalize(recovered), quat_normalize(-q1), atol=1e-5
    )
    assert torch.allclose(quat_multiply(quat_conjugate(quat_normalize(q1)), quat_normalize(q1)), identity(1), atol=1e-6)


def test_rotation_vector_small_and_pi_angle():
    tiny = quat_normalize(torch.tensor([[1.0, 1e-9, 0.0, 0.0]]))
    assert rotation_vector(tiny).abs().max() < 1e-6
    q = quat_from_axis_angle(torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([math.pi - 1e-3]))
    rv = rotation_vector(q)
    assert rv.shape == (1, 3)
    assert torch.linalg.vector_norm(rv).item() == pytest.approx(math.pi - 1e-3, abs=1e-4)


def test_slerp_endpoints_and_norm():
    torch.manual_seed(4)
    q0 = quat_normalize(torch.randn(5, 4))
    q1 = quat_normalize(torch.randn(5, 4))
    t = torch.rand(5, 1)
    out = quat_slerp(q0, q1, t)
    assert torch.allclose(torch.linalg.vector_norm(out, dim=-1), torch.ones(5), atol=1e-6)
    assert torch.allclose(quat_slerp(q0, q1, torch.zeros(5, 1)), q0, atol=1e-5)
    assert torch.allclose(quat_slerp(q0, q1, torch.ones(5, 1)).abs(), q1.abs(), atol=1e-5)
