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
from envs.counter_rng import hash_uniform

# --------------------------------------------------------------------------- #
# Minimum-magnitude seizure angle
# --------------------------------------------------------------------------- #
def stuck_lock_deflections(
    gimbal: torch.Tensor,
    severity: torch.Tensor,
    min_deflection_deg: float = 1.25,
) -> torch.Tensor:
    """Lock deflection of a seized TVC axis, ``[B, 2]`` [rad].

    A gimbal that seizes *within a fraction of a degree of the vertical* does
    not degrade the flight: the vehicle keeps descending on an essentially
    vertical thrust vector and lands next to the pad.  Such an episode is not
    a fault at all -- it is a healthy flight with a jammed actuator -- so
    freezing the fault model at those angles makes the benchmark measure the
    freeze-angle draw instead of the controller.

    The fault model therefore enforces a **minimum seizure angle**

    ``min_stuck_rad = radians(min_deflection_deg) * (severity / 0.5)``

    on the seized axis.  The lock keeps the sign of the pre-fault deflection
    (so the vehicle is pushed in the direction it was already tilting) and
    raises its magnitude to the floor.  A deflection of exactly zero has no
    sign; the environment index then picks one deterministically so the
    benchmark does not develop a systematic left/right bias.
    """
    severity = severity.to(gimbal.dtype).reshape(-1, 1)
    min_rad = math.radians(min_deflection_deg) * (severity / 0.5)
    sign = torch.where(gimbal > 0.0, 1.0, torch.where(gimbal < 0.0, -1.0, 0.0))
    fallback = torch.where(
        (torch.arange(gimbal.shape[0], device=gimbal.device) % 2) == 0, 1.0, -1.0
    ).unsqueeze(-1)
    sign = torch.where(sign == 0.0, fallback, sign)
    return sign * torch.maximum(gimbal.abs(), min_rad)


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
        seed: Optional[torch.Tensor] = None,
    ) -> FaultState:
        """Sample a fresh fault configuration for ``batch_size`` environments.

        Any of ``mode``/``onset``/``severity`` can be supplied to force a
        deterministic scenario (used by tests and the benchmark suite).
        ``seed`` is an optional per-environment int64 seed; when given, the
        *fault parameters* (which axis is biased, which axis seizes, the
        direction of the CoM offset) are drawn from a counter-based hash of
        that seed instead of the shared generator stream, so a scenario is a
        pure function of its seed and does not depend on the order in which
        episodes are run.
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

        params = self._build_params(mode, severity, gen, seed)
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
        self,
        mode: torch.Tensor,
        severity: torch.Tensor,
        gen: Optional[torch.Generator],
        seed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = mode.shape[0]
        dev = mode.device
        dtype = torch.float32

        thrust_scale = torch.where(
            mode == FaultMode.LOE_THRUST, 1.0 - severity, torch.ones(batch, device=dev, dtype=dtype)
        )

        bias_mag = math.radians(self.cfg.gimbal_bias_deg)
        if seed is not None:
            # Counter-based draws: the scenario depends on the episode seed
            # only, never on how many episodes were generated before it.
            s = seed.to(device=dev, dtype=torch.int64)
            counters = torch.arange(4, device=dev, dtype=torch.int64)
            seed_matrix = s.unsqueeze(-1).expand(batch, 4)
            u = hash_uniform(seed_matrix, counters)  # [B, 4]
            u_axis, u_sign, u_stuck, u_angle = u.unbind(dim=-1)
            axis_choice = (u_axis > 0.5).to(torch.long)
            sign = torch.where(u_sign > 0.5, 1.0, -1.0).to(dtype)
            stuck_axis = (u_stuck > 0.5).to(torch.long)
            angle = u_angle.to(dtype) * 2.0 * math.pi
        else:
            axis_choice = torch.randint(0, 2, (batch,), device=dev, generator=gen)
            sign = torch.where(
                torch.rand(batch, device=dev, generator=gen) < 0.5, -1.0, 1.0
            )
            stuck_axis = torch.randint(0, 2, (batch,), device=dev, generator=gen)
            angle = torch.rand(batch, device=dev, generator=gen) * 2.0 * math.pi
        bias_y = torch.where(
            (mode == FaultMode.GIMBAL_BIAS) & (axis_choice == 0), bias_mag * sign, torch.zeros_like(severity)
        )
        bias_z = torch.where(
            (mode == FaultMode.GIMBAL_BIAS) & (axis_choice == 1), bias_mag * sign, torch.zeros_like(severity)
        )

        stuck_y = ((mode == FaultMode.GIMBAL_STUCK) & (stuck_axis == 0)).to(dtype)
        stuck_z = ((mode == FaultMode.GIMBAL_STUCK) & (stuck_axis == 1)).to(dtype)

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
