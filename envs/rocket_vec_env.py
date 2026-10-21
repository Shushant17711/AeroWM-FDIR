"""Fully vectorized 6-DOF rocket environment.

* ``reset()`` / ``step()`` operate on ``(B, ...)`` tensors and never loop over
  the batch dimension.
* Auto-resets terminated sub-environments and returns the fresh observation
  for them, while exposing their final state in ``info`` for logging.
* Observations are normalized with fixed physical scales (AC-2.2); the raw
  state remains available for evaluation and plotting.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch

from configs.schema import (
    ActionSchema,
    EnvironmentConfig,
    FaultInjectionSchema,
    ObservationSchema,
    RewardConfig,
    RocketPhysicsConfig,
)
from envs.base_types import EnvResetReturn, EnvStepReturn, FaultState
from envs.fault_injector import FaultInjector
from envs.quaternion_ops import quat_from_axis_angle, quat_multiply, quat_normalize, quat_tilt_cosine
from envs.rk4_integrator import rk4_step
from envs.rocket_dynamics import atmosphere_density, compute_derivatives

# Termination reason codes exposed in ``info["termination_reason"]``.
REASON_NONE = 0
REASON_LANDED = 1
REASON_CRASH = 2
REASON_TILT = 3
REASON_FUEL = 4
REASON_OUT_OF_BOUNDS = 5
REASON_TIMEOUT = 6


class VectorizedRocketEnv:
    """Batched rocket landing environment with fault injection."""

    def __init__(
        self,
        num_envs: int = 1024,
        device: torch.device | str = "cpu",
        physics: Optional[RocketPhysicsConfig] = None,
        observation: Optional[ObservationSchema] = None,
        action: Optional[ActionSchema] = None,
        fault: Optional[FaultInjectionSchema] = None,
        environment: Optional[EnvironmentConfig] = None,
        reward: Optional[RewardConfig] = None,
        seed: int = 0,
        generator: Optional[torch.Generator] = None,
        fault_probability: Optional[float] = None,
        obs_noise_std: float = 0.0,
    ) -> None:
        self.physics = physics or RocketPhysicsConfig()
        self.observation = observation or ObservationSchema()
        self.action_schema = action or ActionSchema()
        self.fault_cfg = fault or FaultInjectionSchema()
        self.env_cfg = environment or EnvironmentConfig()
        self.reward_cfg = reward or RewardConfig()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.obs_noise_std = float(obs_noise_std)
        self.fault_probability = fault_probability
        # When enabled, every reset draws 50% nominal episodes and 50% faults
        # spread uniformly over the four failure modes.  Explicit ``mode``
        # arguments (benchmarks, tests) always take precedence.
        self.balanced_fault_mix = False
        # When set, every auto-reset (not just an explicit ``reset``) uses this
        # fault mode.  Evaluation harnesses use it to keep per-mode metrics
        # clean across auto-resets.
        self.fixed_fault_mode: Optional[int] = None

        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        self.generator = generator
        self._phase = 0  # deterministic draw counter for the RNG stream

        self.injector = FaultInjector(self.fault_cfg, self.physics, self.device, generator)

        # Mutable per-environment buffers (persistent, never reallocated).
        self.state = torch.zeros(self.num_envs, 17, device=self.device, dtype=self.dtype)
        self.prev_action = torch.zeros(self.num_envs, self.action_schema.dim, device=self.device, dtype=self.dtype)
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        self.episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.fault = self.injector.reset(
            self.num_envs, probability=self._effective_fault_probability()
        )
        self._last_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Counters used by the trainer / evaluator.
        self.completed_episodes = 0
        self.completed_landings = 0

        self.reset()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def obs_dim(self) -> int:
        return self.observation.dim

    @property
    def action_dim(self) -> int:
        return self.action_schema.dim

    def _effective_fault_probability(self) -> Optional[float]:
        return self.fault_probability

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(
        self,
        mask: Optional[torch.Tensor] = None,
        mode: Optional[torch.Tensor | int] = None,
        onset: Optional[torch.Tensor | int] = None,
        severity: Optional[torch.Tensor | float] = None,
        probability: Optional[float] = None,
    ) -> EnvResetReturn:
        """Reset all (or a masked subset) of the environments."""
        dev = self.device
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=dev)
        else:
            mask = mask.to(dev, torch.bool).reshape(self.num_envs)
        if not bool(mask.any()):
            return EnvResetReturn(self._observation(), self._info())

        n = int(mask.sum())
        if mode is None and self.fixed_fault_mode is not None:
            mode = torch.full((n,), int(self.fixed_fault_mode), dtype=torch.long, device=dev)
        if mode is None and self.balanced_fault_mix:
            modes = torch.zeros(n, dtype=torch.long, device=dev)
            n_fault = n // 2
            if n_fault > 0:
                fault_modes = 1 + (torch.arange(n_fault, device=dev) % 4)
                perm = torch.randperm(n, device=dev, generator=self.generator)
                modes[perm[:n_fault]] = fault_modes
            mode = modes
        lo_a, hi_a = self.env_cfg.initial_altitude
        lo_xy, hi_xy = self.env_cfg.initial_lateral
        lo_vz, hi_vz = self.env_cfg.initial_descent_rate
        lo_vxy, hi_vxy = self.env_cfg.initial_lateral_rate

        def uniform(lo: float, hi: float) -> torch.Tensor:
            return torch.rand(n, device=dev, generator=self.generator) * (hi - lo) + lo

        r = torch.stack([uniform(lo_xy, hi_xy), uniform(lo_xy, hi_xy), uniform(lo_a, hi_a)], dim=-1)
        v = torch.stack(
            [uniform(lo_vxy, hi_vxy), uniform(lo_vxy, hi_vxy), uniform(lo_vz, hi_vz)], dim=-1
        )

        # Random small tilt about a horizontal axis, then a random yaw.
        tilt_axis = torch.stack(
            [uniform(-1.0, 1.0), uniform(-1.0, 1.0), torch.zeros(n, device=dev)], dim=-1
        )
        tilt_axis = tilt_axis / torch.linalg.vector_norm(tilt_axis, dim=-1, keepdim=True).clamp_min(1e-6)
        tilt = uniform(0.0, math.radians(self.env_cfg.initial_tilt_deg))
        yaw_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        q = quat_normalize(
            quat_multiply(quat_from_axis_angle(tilt_axis, tilt), quat_from_axis_angle(yaw_axis, uniform(-math.pi, math.pi)))
        )

        omega = torch.stack(
            [uniform(-0.02, 0.02), uniform(-0.02, 0.02), uniform(-0.02, 0.02)], dim=-1
        )
        mass = torch.full((n,), self.physics.wet_mass, device=dev)
        gimbal = torch.zeros(n, 2, device=dev)
        thrust = torch.zeros(n, device=dev)

        new_state = torch.cat([r, v, q, omega, mass.unsqueeze(-1), gimbal, thrust.unsqueeze(-1)], dim=-1)
        self.state[mask] = new_state
        self.prev_action[mask] = 0.0
        self.step_count[mask] = 0
        self.episode_return[mask] = 0.0
        self.episode_length[mask] = 0
        self._last_reason[mask] = REASON_NONE

        # Fault schedule for the reset environments only.
        def expand(value, cast):
            if value is None:
                return None
            t = value if isinstance(value, torch.Tensor) else torch.tensor(value, device=dev)
            t = t.to(dev, cast)
            if t.ndim == 0:
                t = t.expand(n)
            return t[:n]

        fresh = self.injector.reset(
            n,
            mode=expand(mode, torch.long),
            onset=expand(onset, torch.long),
            severity=expand(severity, torch.float),
            probability=probability if probability is not None else self._effective_fault_probability(),
        )
        self.fault = FaultState(
            mode=self._scatter(self.fault.mode, fresh.mode, mask),
            onset=self._scatter(self.fault.onset, fresh.onset, mask),
            severity=self._scatter(self.fault.severity, fresh.severity, mask),
            active=self._scatter(self.fault.active, fresh.active, mask),
            params=self._scatter(self.fault.params, fresh.params, mask),
        )
        return EnvResetReturn(self._observation(), self._info())

    @staticmethod
    def _scatter(old: torch.Tensor, new: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = old.clone()
        out[mask] = new.to(old.dtype)
        return out

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, action: torch.Tensor) -> EnvStepReturn:
        """Advance all environments by one guidance interval."""
        if action.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"action must be [{self.num_envs}, {self.action_dim}], got {tuple(action.shape)}")
        action = action.to(self.device, self.dtype).clamp(self.action_schema.low, self.action_schema.high)

        # 1. Schedule faults, integrate physics.  ``onset`` is 1-based: a
        #    fault with onset=5 is applied during the 5th environment step.
        self.fault = self.injector.update(self.fault, self.step_count + 1)
        self.state = rk4_step(
            compute_derivatives,
            self.state,
            action,
            self.fault.applied_params(),
            self.physics.dt,
            self.physics,
        )
        self.step_count = self.step_count + 1
        self.episode_length = self.episode_length + 1

        # 2. Reward and termination on the new state.
        reward, components = self._reward(action)
        terminated, truncated, reason, landed = self._termination()

        self.episode_return = self.episode_return + reward
        reward = torch.where(terminated, self._terminal_reward(reason, landed), reward)

        done = terminated | truncated
        n_done = int(done.sum())
        if n_done:
            self.completed_episodes += n_done
            self.completed_landings += int(landed.sum())

        info = self._info()
        info.update(
            {
                "reward_components": components,
                "termination_reason": reason,
                "landing_success": landed,
                "episode_return": self.episode_return.clone(),
                "episode_length": self.episode_length.clone(),
                "reset_mask": done,
                "final_state": torch.where(done.unsqueeze(-1), self.state, torch.zeros_like(self.state)),
                "fault_mode": self.fault.mode,
                "fault_active": self.fault.active,
                "fault_context": FaultInjector.context(self.fault),
                "fault_severity": self.fault.severity,
                "fault_onset": self.fault.onset,
                "fault_params": self.fault.params,
                "fault_applied_params": self.fault.applied_params(),
                "step_count": self.step_count.clone(),
            }
        )

        # 3. Auto-reset finished environments (obs returned below is fresh).
        if n_done:
            self.reset(mask=done)

        self.prev_action = action
        return EnvStepReturn(
            obs=self._observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    def _observation(self) -> torch.Tensor:
        state = self.state
        r = state[:, 0:3]
        v = state[:, 3:6]
        q = state[:, 6:10]
        omega = state[:, 10:13]
        mass = state[:, 13]
        gimbal = state[:, 14:16]
        thrust = state[:, 16]
        v_rel = v - torch.tensor(self.physics.wind, device=self.device, dtype=self.dtype)
        q_dyn = 0.5 * atmosphere_density(r[:, 2], self.physics) * torch.sum(v_rel * v_rel, dim=-1)

        obs = torch.cat(
            [
                r / self.observation.position_scale,
                v / self.observation.velocity_scale,
                q,
                omega / self.observation.angular_rate_scale,
                ((mass - self.physics.dry_mass) / self.physics.propellant_mass).unsqueeze(-1),
                gimbal / self.physics.gimbal_limit,
                (thrust / self.physics.max_thrust).unsqueeze(-1),
                (q_dyn / self.observation.dynamic_pressure_scale).unsqueeze(-1),
            ],
            dim=-1,
        )
        if self.obs_noise_std > 0.0:
            obs = obs + torch.randn_like(obs) * self.obs_noise_std
        return obs

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #
    def _reward(self, action: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.reward_cfg
        r = self.state[:, 0:3]
        v = self.state[:, 3:6]
        q = self.state[:, 6:10]
        omega = self.state[:, 10:13]
        thrust = self.state[:, 16]
        gimbal = self.state[:, 14:16]

        dist = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
        v_targ = -cfg.velocity_target_max * r / (dist + 10.0)
        r_track = -cfg.track_velocity_weight * torch.linalg.vector_norm(v - v_targ, dim=-1) - cfg.track_position_weight * dist.squeeze(-1)

        cos_tilt = quat_tilt_cosine(q).clamp(-1.0, 1.0)
        r_att = -cfg.tilt_penalty * (1.0 - cos_tilt) ** 2 - cfg.angular_rate_penalty * torch.sum(omega * omega, dim=-1)

        r_ctrl = -cfg.action_rate_penalty * torch.sum((action - self.prev_action) ** 2, dim=-1)
        r_ctrl = r_ctrl - cfg.gimbal_penalty * torch.sum(gimbal * gimbal, dim=-1)
        r_fuel = -cfg.fuel_penalty * (thrust / self.physics.max_thrust)

        # Reward rescaling (Gate 4): the raw dense terms are O(-10) per step,
        # which makes the discounted return O(-340) and the Q regression
        # ill-conditioned.  Dividing the dense penalties by 50 puts Q in the
        # O(1-10) range, and the glide-slope bonus rewards staying near the
        # velocity-to-go profile (a small positive signal the value head can
        # resolve).
        dense = (r_track + r_att + r_ctrl + r_fuel) / 50.0
        vel_err = torch.linalg.vector_norm(v - v_targ, dim=-1)
        glide_bonus = 0.05 * torch.exp(-vel_err)
        reward = dense + glide_bonus
        components = {
            "glide": glide_bonus,
            "track": r_track,
            "attitude": r_att,
            "control": r_ctrl,
            "fuel": r_fuel,
        }
        return reward, components

    def _terminal_reward(self, reason: torch.Tensor, landed: torch.Tensor) -> torch.Tensor:
        cfg = self.reward_cfg
        speed = torch.linalg.vector_norm(self.state[:, 3:6], dim=-1)
        mass = self.state[:, 13]
        prop_margin = ((mass - self.physics.dry_mass) / self.physics.propellant_mass).clamp(0.0, 1.0)
        success = cfg.soft_landing_bonus + cfg.propellant_margin_bonus * prop_margin
        failure = -(cfg.crash_penalty + cfg.crash_velocity_penalty * speed)
        oob = torch.full_like(failure, -cfg.out_of_bounds_penalty)
        timeout = torch.full_like(failure, -cfg.truncation_penalty)
        reward = torch.where(reason == REASON_OUT_OF_BOUNDS, oob, failure)
        reward = torch.where(reason == REASON_TIMEOUT, timeout, reward)
        return torch.where(landed, success, reward)

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #
    def _termination(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.env_cfg
        phys = self.physics
        s = self.state
        r = s[:, 0:3]
        v = s[:, 3:6]
        q = s[:, 6:10]
        omega = s[:, 10:13]
        mass = s[:, 13]

        z = r[:, 2]
        lateral = torch.linalg.vector_norm(r[:, :2], dim=-1)
        v_horiz = torch.linalg.vector_norm(v[:, :2], dim=-1)
        cos_tilt = quat_tilt_cosine(q).clamp(-1.0, 1.0)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt))
        rate = torch.linalg.vector_norm(omega, dim=-1)

        landed = (
            (z <= cfg.touchdown_altitude)
            & (lateral < cfg.success_lateral_radius)
            & (v[:, 2].abs() < cfg.success_vertical_speed)
            & (v_horiz < cfg.success_lateral_speed)
            & (tilt_deg < cfg.success_tilt_deg)
            & (rate < cfg.success_angular_rate)
        )
        ground = z <= cfg.touchdown_altitude
        tilt_fail = tilt_deg > cfg.crash_tilt_deg
        fuel_out = mass <= phys.dry_mass + 1e-3
        oob = (lateral > cfg.out_of_bounds_xy) | (z > cfg.altitude_ceiling)

        reason = torch.full_like(self.step_count, REASON_NONE)
        reason = torch.where(ground & landed, torch.full_like(reason, REASON_LANDED), reason)
        reason = torch.where(ground & ~landed, torch.full_like(reason, REASON_CRASH), reason)
        reason = torch.where(~ground & tilt_fail, torch.full_like(reason, REASON_TILT), reason)
        reason = torch.where(~ground & ~tilt_fail & fuel_out, torch.full_like(reason, REASON_FUEL), reason)
        reason = torch.where(oob & (reason == REASON_NONE), torch.full_like(reason, REASON_OUT_OF_BOUNDS), reason)

        terminated = (ground | tilt_fail | fuel_out | oob)
        truncated = (self.step_count >= cfg.max_episode_steps) & ~terminated
        reason = torch.where(truncated, torch.full_like(reason, REASON_TIMEOUT), reason)
        return terminated, truncated, reason, landed

    # ------------------------------------------------------------------ #
    # Info
    # ------------------------------------------------------------------ #
    def _info(self) -> Dict[str, Any]:
        labels = FaultInjector.labels(self.fault)
        return {
            "fault_label": labels,
            "fault_context": torch.nn.functional.one_hot(labels, num_classes=5).to(self.dtype),
            "fault_mode": self.fault.mode.clone(),
            "fault_active": self.fault.active.clone(),
            "fault_severity": self.fault.severity.clone(),
            "fault_onset": self.fault.onset.clone(),
            "step_count": self.step_count.clone(),
            "state": self.state.clone(),
        }

    def close(self) -> None:  # gymnasium API compatibility
        pass
