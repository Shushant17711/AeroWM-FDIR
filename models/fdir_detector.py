"""Exact one-step actuator / propulsion / rotational residuals for FDIR.

The KL surprise in :mod:`models.fdir_head` answers "did the observation stop
matching the learned prior".  That is a *learned* signal and only as sensitive
as the world model is accurate.  Gate 3 needs a signal that is sensitive to a
*specific* class of fault within one or two samples, so this module adds a
**physics residual** computed from the frozen command and the next telemetry
sample:

``res_gimbal``
    ``delta_meas(t) - delta_expected(delta_meas(t-1), a_{t-1}[0:2])``
    with the exact first-order servo response
    ``delta_expected = delta + (delta_cmd - delta) * (1 - exp(-dt/tau))``.
``res_thrust``
    the same construction for the turbopump / throttle map.
``res_omega``
    ``omega_meas(t) - omega_predicted(t-1 -> t)`` from the nominal rigid-body
    moment balance, integrated over 4 sub-steps with the closed-form actuator
    trajectories.  A CoM shift leaves gimbal and thrust tracking perfect but
    changes the torque, so this channel is what isolates ``COM_SHIFT``.

Two derived quantities are exported alongside the 9-D feature vector:

``bias``
    the **deconvolved** gimbal bias, ``res_gimbal / (1 - exp(-dt/tau))``.
    Because a constant bias enters the command linearly, the residual equals
    ``b * (1 - exp(-dt/tau))`` for *any* command trajectory, so this recovers
    the true bias in a single sample instead of the ~0.57x under-estimate a
    filtered ``delta_meas - delta_cmd`` gives while the servo is slewing.
``frozen``
    a 2-step-confirmed frozen-servo indicator: a stuck axis does not move at
    all (``|Δdelta| < 1e-5``) while the command asks it to (``|Δdelta_expected|
    > 2e-5``); a merely biased axis still tracks, so it never triggers.

The alternating command dither applied by the fault-conditioned controller
(``PIDGains.excitation_dither``) guarantees the command never coincides with a
frozen angle, which is what makes the frozen test valid during vertical flight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from configs.schema import ObservationSchema, RocketPhysicsConfig

# --------------------------------------------------------------------------- #
# Layout of the residual feature vector fed to the isolator.
# --------------------------------------------------------------------------- #
RES_GIMBAL = slice(0, 2)
RES_THRUST = slice(2, 3)
RES_OMEGA = slice(3, 6)
CMD_GIMBAL = slice(6, 8)
CMD_THRUST = slice(8, 9)
RESIDUAL_DIM = 9

# Fault-parameter head layout (physical units, zero when nominal).
PARAM_GIMBAL_Y = 0
PARAM_GIMBAL_Z = 1
PARAM_THRUST_LOSS = 2
PARAM_COM_Y = 3
PARAM_COM_Z = 4
PARAM_DIM = 5

# Movement thresholds for the frozen-servo test [rad].
FREEZE_MOVE_TOL = 1e-5
FREEZE_EXCITE_TOL = 2e-5


def throttle_to_thrust_norm(u: torch.Tensor, cfg: RocketPhysicsConfig) -> torch.Tensor:
    """Inverse of the plant's continuous throttle map (``u`` -> ``T/T_max``)."""
    return torch.where(
        u < 0.0,
        cfg.min_throttle * (u + 1.0),
        cfg.min_throttle + u * (cfg.max_throttle - cfg.min_throttle),
    )


def _thrust_dir(delta: torch.Tensor) -> torch.Tensor:
    """Unit thrust direction in the body frame."""
    dy, dz = delta.unbind(-1)
    return torch.stack(
        [torch.cos(dy) * torch.sin(dz), torch.sin(dy), torch.cos(dy) * torch.cos(dz)],
        dim=-1,
    )


@dataclass
class ResidualOutput:
    """Raw residual vector plus the auxiliary signals used by the isolator."""

    features: torch.Tensor  # [B, 9]  (contract consumed by FDIRModule)
    frozen: torch.Tensor  # [B, 2]  2-step-confirmed frozen indicator
    moving: torch.Tensor  # [B, 2]  |delta_meas - delta_prev| / limit
    bias: Optional[torch.Tensor] = None  # [B, 2] deconvolved gimbal bias [rad]
    expected_move: Optional[torch.Tensor] = None  # [B, 2] commanded move [rad]
    actual_move: Optional[torch.Tensor] = None  # [B, 2] realised move [rad]
    frozen_instant: Optional[torch.Tensor] = None  # [B, 2] 1-step test
    com_estimate: Optional[torch.Tensor] = None  # [B, 2] CoM offset [m]
    thrust_gain: float = 0.0  # [-] 1 - exp(-dt/tau_engine), for deconvolution


class ResidualExtractor(nn.Module):
    """Stateless one-step residual computation + nominal scale calibration."""

    def __init__(
        self,
        physics: Optional[RocketPhysicsConfig] = None,
        observation: Optional[ObservationSchema] = None,
        alpha: float = 0.01,
        eps: float = 1e-6,
        ema: float = 0.35,
    ) -> None:
        super().__init__()
        self.physics = physics or RocketPhysicsConfig()
        self.observation = observation or ObservationSchema()
        self.alpha = alpha
        self.eps = eps
        # 5-step EMA on the feature vector: alpha = 2/(N+1) with N = 5.
        self.ema = ema
        self.register_buffer("mean", torch.zeros(RESIDUAL_DIM))
        self.register_buffer("std", torch.ones(RESIDUAL_DIM))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        # Episode-scoped state (not part of the checkpoint).
        self.register_buffer("ema_state", torch.zeros(0), persistent=False)
        self.register_buffer("bias_ema", torch.zeros(0), persistent=False)
        self.register_buffer("freeze_count", torch.zeros(0), persistent=False)
        self.register_buffer("com_ema", torch.zeros(0), persistent=False)

    # ------------------------------------------------------------------ #
    # Physics
    # ------------------------------------------------------------------ #
    def _unpack(self, obs: torch.Tensor) -> dict:
        phys = self.physics
        s = self.observation.slices
        return {
            "q": obs[..., s["quaternion"]],
            "omega": obs[..., s["angular_rate"]] * self.observation.angular_rate_scale,
            "mass": (phys.dry_mass + obs[..., s["mass"]] * phys.propellant_mass).squeeze(-1),
            "gimbal": obs[..., s["gimbal"]] * phys.gimbal_limit,
            "thrust": (obs[..., s["thrust"]] * phys.max_thrust).squeeze(-1),
        }

    def compute(
        self,
        obs_prev: torch.Tensor,
        obs: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> ResidualOutput:
        """Residuals for the transition ``(o_{t-1}, a_{t-1}) -> o_t``."""
        cfg = self.physics
        prev = self._unpack(obs_prev)
        cur = self._unpack(obs)
        dt = cfg.dt
        u = a_prev

        delta_prev, delta_meas = prev["gimbal"], cur["gimbal"]
        thrust_prev, thrust_meas = prev["thrust"], cur["thrust"]
        omega_prev, omega_meas = prev["omega"], cur["omega"]
        mass = cur["mass"].clamp_min(cfg.dry_mass)

        # --- actuator lag expectations (exact first-order response) ---- #
        alpha_e = 1.0 - math.exp(-dt / cfg.act_tau)
        delta_cmd = u[..., :2] * cfg.gimbal_limit
        delta_expected = delta_prev + (delta_cmd - delta_prev) * alpha_e

        thrust_cmd = throttle_to_thrust_norm(u[..., 2], cfg) * cfg.max_thrust
        thrust_gain = 1.0 - math.exp(-dt / cfg.engine_tau)
        thrust_expected = thrust_prev + (thrust_cmd - thrust_prev) * thrust_gain

        res_gimbal = (delta_meas - delta_expected) / cfg.gimbal_limit
        res_thrust = ((thrust_meas - thrust_expected) / cfg.max_thrust).unsqueeze(-1)

        # --- rigid-body rotational prediction ------------------------- #
        sub = 4
        h = dt / sub
        arm = torch.zeros((*delta_prev.shape[:-1], 3), dtype=delta_prev.dtype, device=delta_prev.device)
        arm[..., 2] = -cfg.engine_moment_arm
        damping = torch.tensor(cfg.angular_damping, dtype=delta_prev.dtype, device=delta_prev.device)
        spec = torch.tensor(cfg.specific_inertia, dtype=delta_prev.dtype, device=delta_prev.device)
        inertia = mass.unsqueeze(-1) * spec

        omega_run = omega_prev
        for i in range(sub):
            frac = (i + 0.5) / sub
            gim_mid = (delta_cmd - delta_prev) * (1.0 - math.exp(-frac * dt / cfg.act_tau)) + delta_prev
            thr_mid = (thrust_cmd - thrust_prev) * (1.0 - math.exp(-frac * dt / cfg.engine_tau)) + thrust_prev
            f_mid = thr_mid.unsqueeze(-1) * _thrust_dir(gim_mid)
            tau_mid = torch.cross(arm, f_mid, dim=-1) - damping * omega_run
            gyro = torch.cross(omega_run, inertia * omega_run, dim=-1)
            omega_run = omega_run + ((tau_mid - gyro) / inertia) * h
        res_omega = (omega_meas - omega_run) / self.observation.angular_rate_scale

        # --- command tracking (5-D contract from the spec) ------------ #
        cmd_gimbal = obs[..., self.observation.slices["gimbal"]] - u[..., :2]
        cmd_thrust = obs[..., self.observation.slices["thrust"]] - throttle_to_thrust_norm(
            u[..., 2], cfg
        ).unsqueeze(-1)

        features = torch.cat(
            [res_gimbal, res_thrust, res_omega, cmd_gimbal, cmd_thrust], dim=-1
        ).clamp(-50.0, 50.0)

        # ------------------------------------------------------------------ #
        # Frozen-servo test with 2-step confirmation.
        # ------------------------------------------------------------------ #
        actual_move = (delta_meas - delta_prev).abs()
        expected_move = (delta_expected - delta_prev).abs()
        frozen_inst = (actual_move < FREEZE_MOVE_TOL) & (expected_move > FREEZE_EXCITE_TOL)
        if self.freeze_count.shape != frozen_inst.shape:
            self.freeze_count = torch.zeros_like(frozen_inst, dtype=torch.long)
        self.freeze_count = torch.where(
            frozen_inst, self.freeze_count + 1, torch.zeros_like(self.freeze_count)
        )
        frozen = (self.freeze_count >= 2).to(features.dtype)

        # ------------------------------------------------------------------ #
        # Deconvolved bias: res_gimbal == b * alpha_e for any command path.
        # ------------------------------------------------------------------ #
        # ``res_gimbal`` is normalised by the gimbal limit; convert to radians.
        bias_step = res_gimbal * cfg.gimbal_limit / alpha_e
        if self.bias_ema.shape != bias_step.shape:
            self.bias_ema = bias_step.detach().clone()
        # 4-sample average: rejects sensor noise only, adds no lag.
        self.bias_ema = 0.75 * self.bias_ema + 0.25 * bias_step.detach()

        # ------------------------------------------------------------------ #
        # CoM estimate.  A CoM offset applies a *persistent* torque
        # ``tau_res = T * com``, so the rotational residual is a small constant
        # (1.5e-4 rad/s at 0.11 m and 400 kN) that is invisible against the
        # per-step noise.  Averaging it over ~0.5 s and dividing by the thrust
        # recovers the offset directly, in metres.
        # ------------------------------------------------------------------ #
        domega_res = (omega_meas - omega_run)  # [rad/s]
        inertia_t = mass.unsqueeze(-1) * spec
        tau_res = inertia_t * domega_res / dt
        thrust_safe = thrust_meas.clamp_min(0.1 * cfg.max_thrust).unsqueeze(-1)
        com_step = (tau_res / thrust_safe).clamp(-1.0, 1.0)
        if self.com_ema.shape != com_step.shape:
            self.com_ema = com_step.detach().clone()
        self.com_ema = 0.70 * self.com_ema + 0.30 * com_step.detach()

        return ResidualOutput(
            features=features,
            frozen=frozen,
            moving=actual_move / cfg.gimbal_limit,
            bias=self.bias_ema,
            expected_move=expected_move,
            actual_move=actual_move,
            frozen_instant=frozen_inst.to(features.dtype),
            com_estimate=self.com_ema,
            thrust_gain=float(thrust_gain),
        )

    # ------------------------------------------------------------------ #
    # Normalisation / calibration
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_nominal(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        x = features.detach().float().reshape(-1, RESIDUAL_DIM)
        if mask is not None:
            x = x[mask.detach().bool().reshape(-1)]
        # Auto-reset jumps produce O(10^3) residuals that would blow up the
        # nominal scale and hide every real fault.
        if x.numel():
            x = x[x.abs().amax(dim=-1) < 100.0]
        if x.numel() == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        if int(self.count) == 0:
            self.mean.copy_(batch_mean)
            self.std.copy_(batch_var.sqrt().clamp_min(max(self.eps, 1e-3)))
        else:
            delta = batch_mean - self.mean
            self.mean.add_(self.alpha * delta)
            var = self.std.pow(2).mul_(1.0 - self.alpha).add_(
                self.alpha * (batch_var + delta * delta)
            )
            self.std.copy_(var.clamp_min(max(self.eps, 1e-3) ** 2).sqrt())
        self.count += 1

    @torch.no_grad()
    def calibrate(self, features: torch.Tensor) -> None:
        x = features.detach().float().reshape(-1, RESIDUAL_DIM)
        if x.numel() == 0:
            return
        self.mean.copy_(x.mean(dim=0))
        self.std.copy_(x.std(dim=0, unbiased=False).clamp_min(max(self.eps, 1e-3)))
        self.count.fill_(1)
        self.ema_state = torch.zeros(0)
        self.freeze_count = torch.zeros(0, dtype=torch.long)

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        # Floor the nominal step-to-step scale.  A channel that is *exactly*
        # predictable (a healthy axis under a stationary command) collapses to
        # ~1e-6, and then a 1e-5 model/prediction mismatch reads as thousands
        # of sigma.  1e-3 of the normalised unit is physically negligible
        # (0.006 deg of gimbal, 0.8 kN of thrust, 0.002 rad/s of rate) while
        # keeping the score interpretable.
        return (features - self.mean) / self.std.clamp_min(max(self.eps, 1e-3))

    def surprise(self, features: torch.Tensor) -> torch.Tensor:
        """Chi-square style anomaly score (``~1`` in nominal flight)."""
        z = self.normalize(features)
        return z.pow(2).mean(dim=-1)

    def smooth(self, features: torch.Tensor) -> torch.Tensor:
        """5-step EMA over the feature vector (state carried across calls)."""
        if self.ema_state.shape != features.shape or self.ema_state.device != features.device:
            self.ema_state = features.detach().clone()
            return features
        state = (1.0 - self.ema) * self.ema_state + self.ema * features
        self.ema_state = state.detach()
        return state

    def forward(
        self, obs_prev: torch.Tensor, obs: torch.Tensor, a_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, ResidualOutput]:
        out = self.compute(obs_prev, obs, a_prev)
        return self.smooth(out.features), out


class FaultParameterHead(nn.Module):
    """Continuous fault-parameter estimate (a learned cross-check).

    Layout (all zero in nominal flight):
    ``[gimbal_bias_y, gimbal_bias_z, thrust_loss, com_offset_y, com_offset_z]``.
    At deployment the magnitudes are read out physically from the residuals
    (see :func:`agents.residual_control.estimate_fault_state`); this head is
    trained as supervision and used only when a physical read-out is missing.
    """

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        from models.networks import NormMLP

        self.net = NormMLP(in_dim, hidden, PARAM_DIM, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        return torch.cat(
            [
                torch.tanh(raw[..., :2]) * 0.15,  # gimbal bias [rad]
                torch.sigmoid(raw[..., 2:3]),  # thrust loss fraction [0, 1]
                torch.tanh(raw[..., 3:5]) * 0.5,  # CoM offset [m]
            ],
            dim=-1,
        )


def isolation_prior(
    features: torch.Tensor,
    frozen: torch.Tensor,
    cfg: Optional[RocketPhysicsConfig] = None,
) -> torch.Tensor:
    """Physics-informed class evidence ``[B, 5]`` for the isolation head.

    Every term is a continuous function of the *normalised* residuals, so this
    is a model of the fault signatures rather than a label lookup:

    * LOE: thrust tracking error while the gimbal tracks.
    * bias: gimbal tracking error with the servo still moving.
    * stuck: gimbal tracking error with a confirmed frozen servo.
    * CoM: rotational innovation that the actuator residuals do not explain.
    """
    gimbal_hit = features[..., RES_GIMBAL].abs().amax(dim=-1) + features[..., CMD_GIMBAL].abs().amax(dim=-1)
    thrust_hit = features[..., RES_THRUST].squeeze(-1).abs()
    omega_hit = features[..., RES_OMEGA].abs().amax(dim=-1)
    frozen_hit = frozen.amax(dim=-1).clamp(0.0, 1.0)

    # CoM evidence is rotational innovation with actuator tracking healthy:
    # every fault perturbs the attitude, so the *unexplained* part is the CoM
    # signature.
    com_excess = omega_hit * (1.0 / (1.0 + 1.5 * thrust_hit)) * (1.0 / (1.0 + 0.2 * gimbal_hit))
    ev_stuck = 10.0 * gimbal_hit * frozen_hit
    ev_bias = 3.0 * gimbal_hit * (1.0 - frozen_hit)
    evidence = torch.stack(
        [
            # The nominal weight must sit above the ~1.5 sigma noise floor of
            # every fault channel, otherwise a single noisy sample flips the
            # argmax to LOE.  A real fault is 20-100 sigma, so the margin below
            # costs nothing in sensitivity.
            torch.full_like(gimbal_hit, 20.0),  # NOMINAL prior weight
            4.0 * thrust_hit / (1.0 + 0.1 * gimbal_hit),
            ev_bias,
            ev_stuck,
            4.0 * com_excess,
        ],
        dim=-1,
    )
    return evidence
