"""Controller interface shared by PID, PPO and the world-model MPPI planner.

The interface is intentionally small so the benchmark can treat every
controller identically: reset per-environment state, then map a normalized
observation batch to a normalized action batch of shape ``[B, 3]``.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch

from configs.schema import ActionSchema


@runtime_checkable
class BaseController(Protocol):
    """Structural protocol for every flight controller in AeroWM-FDIR."""

    @property
    def action_dim(self) -> int:
        ...

    def reset(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> None:
        ...

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        fault_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Map ``obs [B, 18]`` to ``action [B, 3]`` inside ``[-1, 1]``."""
        ...


class ControllerMixin:
    """Small shared helpers for concrete controllers (bounds + validation)."""

    action_schema: ActionSchema = ActionSchema()

    def _check_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[-1] != 18:
            raise ValueError(f"obs must be [B, 18], got {tuple(obs.shape)}")
        return obs

    def _clip_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 2 or action.shape[-1] != 3:
            raise ValueError(f"action must be [B, 3], got {tuple(action.shape)}")
        return action.clamp(self.action_schema.low, self.action_schema.high)
