"""Shared tensor containers and interfaces for environments and controllers.

Everything here is deliberately plain: named tuples of tensors, no hidden
state, so that the physics kernel can be ``torch.compile``-d and every module
can be unit tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional

import torch

from configs.schema import NUM_FAULT_MODES, FaultParamsLayout

STATE_DIM = 17
"""Packed physics state: r(3) v(3) q(4) omega(3) m(1) gimbal(2) thrust(1)."""


class RocketState(NamedTuple):
    """Batched ground-truth rigid-body state.

    All tensors share the leading batch dimension ``B``.
    """

    r: torch.Tensor  # [B, 3] position in inertial frame [m]
    v: torch.Tensor  # [B, 3] velocity in inertial frame [m/s]
    q: torch.Tensor  # [B, 4] body -> inertial unit quaternion (w, x, y, z)
    omega: torch.Tensor  # [B, 3] body angular rate [rad/s]
    mass: torch.Tensor  # [B] total mass [kg]
    gimbal: torch.Tensor  # [B, 2] actual gimbal deflection (y, z) [rad]
    thrust: torch.Tensor  # [B] actual engine thrust [N]

    def to_tensor(self) -> torch.Tensor:
        return torch.cat(
            [
                self.r,
                self.v,
                self.q,
                self.omega,
                self.mass.unsqueeze(-1),
                self.gimbal,
                self.thrust.unsqueeze(-1),
            ],
            dim=-1,
        )

    @classmethod
    def from_tensor(cls, x: torch.Tensor) -> "RocketState":
        if x.shape[-1] != STATE_DIM:
            raise ValueError(f"expected state dim {STATE_DIM}, got {x.shape[-1]}")
        return cls(
            r=x[..., 0:3],
            v=x[..., 3:6],
            q=x[..., 6:10],
            omega=x[..., 10:13],
            mass=x[..., 13],
            gimbal=x[..., 14:16],
            thrust=x[..., 16],
        )

    @property
    def batch_size(self) -> int:
        return self.r.shape[0]


@dataclass(frozen=True)
class FaultState:
    """Per-environment ground-truth fault bookkeeping.

    ``params`` is the packed tensor consumed by the dynamics kernel; the
    remaining fields are diagnostics used for labels and evaluation.
    """

    mode: torch.Tensor  # [B] int64, FaultMode
    onset: torch.Tensor  # [B] int64, step index where the fault starts
    severity: torch.Tensor  # [B] float
    active: torch.Tensor  # [B] bool
    params: torch.Tensor  # [B, FaultParamsLayout.DIM] float

    def __post_init__(self) -> None:
        if self.params.shape != (self.mode.shape[0], FaultParamsLayout.DIM):
            raise ValueError(
                f"fault params must be [B, {FaultParamsLayout.DIM}], got {tuple(self.params.shape)}"
            )

    def to(self, device: torch.device) -> "FaultState":
        return FaultState(
            mode=self.mode.to(device),
            onset=self.onset.to(device),
            severity=self.severity.to(device),
            active=self.active.to(device),
            params=self.params.to(device),
        )

    def one_hot(self) -> torch.Tensor:
        """[B, 5] fault context (all-zero rows are not possible: NOMINAL=0)."""
        return torch.nn.functional.one_hot(
            self.mode, num_classes=NUM_FAULT_MODES
        ).to(self.params.dtype)

    def applied_params(self) -> torch.Tensor:
        """Fault parameters with inactive faults masked to physically inert values.

        ``params`` always holds the raw ground-truth parameters; this method
        is what the dynamics kernel consumes so that a scheduled fault has no
        effect before its onset.
        """
        inert = torch.zeros_like(self.params)
        inert[..., FaultParamsLayout.THRUST_SCALE] = 1.0
        return torch.where(self.active.unsqueeze(-1), self.params, inert)


class EnvStepReturn(NamedTuple):
    """Contract returned by :meth:`VectorizedRocketEnv.step`."""

    obs: torch.Tensor  # [B, 18] normalized observation
    reward: torch.Tensor  # [B]
    terminated: torch.Tensor  # [B] bool (ground contact / crash / out of bounds)
    truncated: torch.Tensor  # [B] bool (time limit)
    info: Dict[str, Any]


class EnvResetReturn(NamedTuple):
    obs: torch.Tensor  # [B, 18]
    info: Dict[str, Any]


class ControllerContext(NamedTuple):
    """Optional recurrent context a controller may carry across steps."""

    hidden: Optional[torch.Tensor] = None
    latent: Optional[torch.Tensor] = None
    fault_context: Optional[torch.Tensor] = None
