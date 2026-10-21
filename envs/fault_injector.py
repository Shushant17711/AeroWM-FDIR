"""Mid-flight fault injection for the vectorized rocket environment.

Faults are *not* observable by the agent; the injector exposes ground-truth
labels through ``info`` purely for supervision and evaluation.  Fault
parameters are encoded in the packed tensor consumed by the dynamics kernel
(see :class:`configs.schema.FaultParamsLayout`).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from configs.schema import FaultInjectionSchema, FaultMode, NUM_FAULT_MODES, RocketPhysicsConfig
from envs.base_types import FaultState


class FaultInjector:
    """Samples, schedules and applies the five supported fault modes."""

    def __init__(
        self,
        cfg: FaultInjectionSchema,
        physics: RocketPhysicsConfig | None = None,
        device: torch.device | str = "cpu",
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.cfg = cfg
        self.physics = physics or RocketPhysicsConfig()
        self.device = torch.device(device)
        self.generator = generator

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def reset(
        self,
        batch_size: int,
        mode: Optional[torch.Tensor] = None,
        onset: Optional[torch.Tensor] = None,
        severity: Optional[torch.Tensor] = None,
        probability: Optional[float] = None,
    ) -> FaultState:
        """Sample a fresh fault configuration for ``batch_size`` environments.

        Any of ``mode``/``onset``/``severity`` can be supplied to force a
        deterministic scenario (used by tests and the benchmark suite).
        """
        dev = self.device
        gen = self.generator
        p = self.cfg.probability if probability is None else float(probability)

        if mode is None:
            has_fault = torch.rand(batch_size, device=dev, generator=gen) < p
            sampled = torch.randint(
                1, NUM_FAULT_MODES, (batch_size,), device=dev, generator=gen, dtype=torch.long
            )
            mode = torch.where(has_fault, sampled, torch.zeros_like(sampled))

        if onset is None:
            lo, hi = self.cfg.onset_range
            onset = torch.randint(lo, hi, (batch_size,), device=dev, generator=gen, dtype=torch.long)
        else:
            onset = onset.to(dev, torch.long)

        if severity is None:
            severity = self._sample_severity(mode, batch_size, dev, gen)
        else:
            severity = severity.to(dev, torch.float32)

        params = self._build_params(mode, severity, gen)
        return FaultState(
            mode=mode.to(dev, torch.long),
            onset=onset,
            severity=severity.clamp(0.0, 1.0),
            active=torch.zeros(batch_size, dtype=torch.bool, device=dev),
            params=params,
        )

    def _sample_severity(
        self, mode: torch.Tensor, batch: int, device: torch.device, gen: Optional[torch.Generator]
    ) -> torch.Tensor:
        lo, hi = self.cfg.loe_severity
        loe = torch.rand(batch, device=device, generator=gen) * (hi - lo) + lo
        # Gimbal bias magnitude is fixed at the configured value; severity is
        # the normalised magnitude used for regression supervision.
        bias = torch.full((batch,), self.cfg.gimbal_bias_deg / 8.0, device=device)
        com = torch.rand(batch, device=device, generator=gen)
        stuck = torch.ones(batch, device=device)
        severity = torch.where(
            mode == FaultMode.LOE_THRUST,
            loe,
            torch.where(
                mode == FaultMode.GIMBAL_BIAS,
                bias,
                torch.where(mode == FaultMode.COM_SHIFT, com, torch.where(mode == FaultMode.GIMBAL_STUCK, stuck, torch.zeros_like(loe))),
            ),
        )
        return severity

    def _build_params(
        self, mode: torch.Tensor, severity: torch.Tensor, gen: Optional[torch.Generator]
    ) -> torch.Tensor:
        batch = mode.shape[0]
        dev = mode.device
        dtype = torch.float32

        thrust_scale = torch.where(
            mode == FaultMode.LOE_THRUST, 1.0 - severity, torch.ones(batch, device=dev, dtype=dtype)
        )

        bias_mag = math.radians(self.cfg.gimbal_bias_deg)
        axis_choice = torch.randint(0, 2, (batch,), device=dev, generator=gen)
        sign = torch.where(
            torch.rand(batch, device=dev, generator=gen) < 0.5, -1.0, 1.0
        )
        bias_y = torch.where(
            (mode == FaultMode.GIMBAL_BIAS) & (axis_choice == 0), bias_mag * sign, torch.zeros_like(severity)
        )
        bias_z = torch.where(
            (mode == FaultMode.GIMBAL_BIAS) & (axis_choice == 1), bias_mag * sign, torch.zeros_like(severity)
        )

        stuck_axis = torch.randint(0, 2, (batch,), device=dev, generator=gen)
        stuck_y = ((mode == FaultMode.GIMBAL_STUCK) & (stuck_axis == 0)).to(dtype)
        stuck_z = ((mode == FaultMode.GIMBAL_STUCK) & (stuck_axis == 1)).to(dtype)

        angle = torch.rand(batch, device=dev, generator=gen) * 2.0 * math.pi
        com_mag = self.cfg.com_offset_max * severity
        com_active = (mode == FaultMode.COM_SHIFT).to(dtype)
        com_y = torch.cos(angle) * com_mag * com_active
        com_z = torch.sin(angle) * com_mag * com_active

        return torch.stack([thrust_scale, bias_y, bias_z, com_y, com_z, stuck_y, stuck_z], dim=-1)

    # ------------------------------------------------------------------ #
    # Scheduling
    # ------------------------------------------------------------------ #
    def update(self, fault: FaultState, step: torch.Tensor) -> FaultState:
        """Activate faults whose onset step has been reached.

        Returns a new :class:`FaultState` with updated ``active`` flags.  Raw
        parameters are preserved; :meth:`FaultState.applied_params` performs
        the masking so inactive faults are physically inert.
        """
        active = (step >= fault.onset) & (fault.mode != int(FaultMode.NOMINAL))
        return FaultState(
            mode=fault.mode,
            onset=fault.onset,
            severity=fault.severity,
            active=active,
            params=fault.params,
        )

    # ------------------------------------------------------------------ #
    # Labels
    # ------------------------------------------------------------------ #
    @staticmethod
    def labels(fault: FaultState) -> torch.Tensor:
        """Per-step fault class, NOMINAL before onset."""
        return torch.where(fault.active, fault.mode, torch.zeros_like(fault.mode))

    @staticmethod
    def context(fault: FaultState) -> torch.Tensor:
        """``[B, 5]`` one-hot ground-truth fault context (inactive -> NOMINAL)."""
        return torch.nn.functional.one_hot(
            FaultInjector.labels(fault), num_classes=NUM_FAULT_MODES
        ).to(fault.params.dtype)

    # ------------------------------------------------------------------ #
    # Debug helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def describe(fault: FaultState, index: int = 0) -> dict:
        return {
            "mode": FaultMode(int(fault.mode[index])).name,
            "onset": int(fault.onset[index]),
            "severity": float(fault.severity[index]),
            "active": bool(fault.active[index]),
        }
