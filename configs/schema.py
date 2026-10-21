"""Frozen data contracts shared by every module in AeroWM-FDIR.

The values here are the single source of truth for tensor shapes, physical
constants, and normalization scales.  Downstream code must import these
instead of hard-coding dimensions.

Conventions (see README "Frames & conventions")
-----------------------------------------------
* Inertial frame ``I``: ENU landing-pad frame, ``+z`` up, ``g = [0, 0, -g0]``.
* Body frame ``B``: ``+z`` is the longitudinal (thrust) axis, ``+x``/``+y``
  are the lateral pitch/yaw axes.  The nominal upright attitude is the
  identity quaternion and the tilt cosine from vertical is ``R[2, 2]``.
* Quaternion ordering is ``(w, x, y, z)`` and maps body -> inertial.
* The action/observation dimensions follow ``design.md`` (3-D action,
  18-D observation).  The older blueprint's 4-D action / 21-D observation
  variant is intentionally superseded; see README for the rationale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple



# --------------------------------------------------------------------------- #
# Fault modes
# --------------------------------------------------------------------------- #
class FaultMode(IntEnum):
    """Ground-truth fault labels used by the injector and the FDIR head."""

    NOMINAL = 0
    LOE_THRUST = 1
    GIMBAL_BIAS = 2
    GIMBAL_STUCK = 3
    COM_SHIFT = 4


NUM_FAULT_MODES = len(FaultMode)


class FaultParamsLayout:
    """Column layout of the packed per-environment fault parameter tensor.

    The packed tensor is what the (compilable) dynamics function consumes:
    ``[thrust_scale, gimbal_bias_y, gimbal_bias_z, com_offset_y,
    com_offset_z, stuck_y, stuck_z]``.  It is always length
    :attr:`DIM`; the active/onset bookkeeping lives in ``FaultState``.
    """

    THRUST_SCALE = 0
    GIMBAL_BIAS_Y = 1
    GIMBAL_BIAS_Z = 2
    COM_OFFSET_Y = 3
    COM_OFFSET_Z = 4
    STUCK_Y = 5
    STUCK_Z = 6
    DIM = 7


@dataclass(frozen=True)
class RocketPhysicsConfig:
    """Falcon-9-class, sub-scale landing stage parameters.

    All values are SI.  The vehicle is a rigid body with a single gimballed
    main engine, first-order actuator lag on both the gimbal and the engine
    spool, exponential atmosphere, and a linear (mass-proportional) inertia
    tensor.
    """

    gravity: float = 9.80665
    dt: float = 0.02  # 50 Hz guidance rate
    dry_mass: float = 22_200.0
    propellant_mass: float = 15_000.0
    max_thrust: float = 845_000.0
    min_throttle: float = 0.40
    max_throttle: float = 1.00
    # Fraction of the normalized throttle command range that maps to engine
    # cutoff.  Real landing engines cannot throttle below their minimum
    # sustainable level, but they *can* shut down (Falcon-9 hoverslam).  The
    # bottom of the action range therefore commands a clean shutdown instead
    # of an unphysical 40% thrust floor.
    cutoff_fraction: float = 0.10
    specific_impulse: float = 282.0
    gimbal_limit_deg: float = 8.0
    act_tau: float = 0.035  # hydraulic servo time constant [s]
    engine_tau: float = 0.10  # turbopump spool time constant [s]
    engine_moment_arm: float = 15.0  # CoM -> gimbal pivot along -z_B [m]
    ref_area: float = 10.75
    drag_coefficient: float = 0.75
    rho0: float = 1.225
    atmosphere_scale_height: float = 8_500.0
    # Specific (per-kg) inertia about (x, y, z) body axes [m^2].  The vehicle
    # is slender along its thrust axis (+z), so the transverse axes carry the
    # large inertia and roll about z carries very little.
    specific_inertia: Tuple[float, float, float] = (130.0, 130.0, 1.6)
    # Passive aerodynamic/structural rate damping torque per rad/s [N*m*s].
    angular_damping: Tuple[float, float, float] = (2.0e5, 2.0e5, 2.0e4)
    # Optional constant wind in the inertial frame [m/s].
    wind: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Backup cold-gas attitude-control system (ACS/RCS).  Thrusters at the top
    # of the stage give pure torque about the two transverse axes.  They are
    # *latched* on an axis only when the primary TVC actuator for that axis is
    # stuck or saturated, so nominal flight dynamics are unchanged.
    rcs_moment_arm: float = 20.0  # [m] CoM -> RCS thruster pair
    rcs_max_force: float = 25_000.0  # [N] per thruster
    # A cold-gas ACS uses a *pair* of opposed thrusters on each axis, so the
    # per-axis authority is rcs_moment_arm * rcs_max_force * pairs.
    rcs_thruster_pairs: int = 3
    rcs_enable: bool = True

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if not 0.0 < self.min_throttle < self.max_throttle <= 1.0:
            raise ValueError("throttle bounds must satisfy 0 < min < max <= 1")
        if self.dry_mass <= 0.0 or self.propellant_mass <= 0.0:
            raise ValueError("masses must be positive")
        if not 0.0 < self.gimbal_limit_deg < 45.0:
            raise ValueError("gimbal limit must be in (0, 45) degrees")
        if self.act_tau <= 0.0 or self.engine_tau <= 0.0:
            raise ValueError("actuator time constants must be positive")
        if min(self.specific_inertia) <= 0.0:
            raise ValueError("specific inertia must be positive")
        if self.rcs_moment_arm <= 0.0 or self.rcs_max_force < 0.0:
            raise ValueError("RCS arm must be positive and force non-negative")
        if self.rcs_thruster_pairs < 1:
            raise ValueError("need at least one RCS thruster pair per axis")

    # -- derived quantities ------------------------------------------------- #
    @property
    def rcs_max_torque(self) -> float:
        """Peak torque of one RCS axis [N*m]."""
        return self.rcs_moment_arm * self.rcs_max_force * self.rcs_thruster_pairs

    @property
    def wet_mass(self) -> float:
        return self.dry_mass + self.propellant_mass

    @property
    def gimbal_limit(self) -> float:
        return math.radians(self.gimbal_limit_deg)

    @property
    def thrust_min(self) -> float:
        return self.min_throttle * self.max_thrust

    @property
    def thrust_max(self) -> float:
        return self.max_throttle * self.max_thrust

    @property
    def exhaust_velocity(self) -> float:
        return self.specific_impulse * self.gravity


@dataclass(frozen=True)
class ObservationSchema:
    """18-D normalized telemetry layout returned by the environment.

    ``r(3) | v(3) | q(4) | omega(3) | mass(1) | gimbal(2) | thrust(1) |
    q_dyn(1)``
    """

    position_scale: float = 1_000.0
    velocity_scale: float = 100.0
    angular_rate_scale: float = 2.0
    dynamic_pressure_scale: float = 10_000.0

    @property
    def dim(self) -> int:
        return 18

    @property
    def slices(self) -> dict:
        return {
            "position": slice(0, 3),
            "velocity": slice(3, 6),
            "quaternion": slice(6, 10),
            "angular_rate": slice(10, 13),
            "mass": slice(13, 14),
            "gimbal": slice(14, 16),
            "thrust": slice(16, 17),
            "dynamic_pressure": slice(17, 18),
        }

    def __post_init__(self) -> None:
        if not all(
            s > 0.0
            for s in (
                self.position_scale,
                self.velocity_scale,
                self.angular_rate_scale,
                self.dynamic_pressure_scale,
            )
        ):
            raise ValueError("observation scales must be positive")


@dataclass(frozen=True)
class ActionSchema:
    """3-D normalized action: pitch gimbal, yaw gimbal, throttle."""

    dim: int = 3
    low: float = -1.0
    high: float = 1.0

    def __post_init__(self) -> None:
        if self.dim != 3:
            raise ValueError("AeroWM-FDIR uses a 3-D action (gimbal y, gimbal z, throttle)")
        if not self.low < self.high:
            raise ValueError("action bounds must satisfy low < high")


@dataclass(frozen=True)
class FaultInjectionSchema:
    """Contract for the ground-truth fault context exposed in ``info``.

    ``fault_context`` is a 5-D one-hot encoding of :class:`FaultMode`
    (``c_t in R^5`` as required by AC-2.4).  Continuous fault parameters
    (severity, onset) are exposed separately for auxiliary supervision.
    """

    num_modes: int = NUM_FAULT_MODES
    fault_context_dim: int = NUM_FAULT_MODES
    probability: float = 0.65
    onset_range: Tuple[int, int] = (50, 400)
    loe_severity: Tuple[float, float] = (0.20, 0.40)
    gimbal_bias_deg: float = 3.5
    com_offset_max: float = 0.35
    # Minimum magnitude of a seized (GIMBAL_STUCK) TVC axis, in degrees at
    # the reference severity of 0.5.  A gimbal that seizes at the vertical
    # does not degrade the flight at all -- the vehicle keeps descending on an
    # essentially vertical thrust vector and lands -- so such an episode is a
    # healthy flight with a jammed actuator, not a fault.  The fault model
    # therefore locks the seized axis at at least this deflection
    # (scaled by ``severity / 0.5``), which makes the benchmark measure the
    # controller instead of the freeze-angle draw.
    stuck_min_deflection_deg: float = 1.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("fault probability must be in [0, 1]")
        lo, hi = self.onset_range
        if not 0 <= lo < hi:
            raise ValueError("onset_range must be increasing and non-negative")
        slo, shi = self.loe_severity
        if not 0.0 < slo < shi < 1.0:
            raise ValueError("LOE severity must be in (0, 1)")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Episode layout, domain randomization, and termination thresholds."""

    initial_altitude: Tuple[float, float] = (800.0, 1_200.0)
    initial_lateral: Tuple[float, float] = (-150.0, 150.0)
    initial_descent_rate: Tuple[float, float] = (-85.0, -55.0)
    initial_lateral_rate: Tuple[float, float] = (-20.0, 20.0)
    initial_tilt_deg: float = 5.0
    max_episode_steps: int = 4_000
    pad_radius: float = 5.0
    touchdown_altitude: float = 0.0
    success_lateral_radius: float = 5.0
    success_vertical_speed: float = 2.5
    success_lateral_speed: float = 1.5
    success_tilt_deg: float = 8.0
    success_angular_rate: float = 0.25
    crash_tilt_deg: float = 60.0
    out_of_bounds_xy: float = 1_500.0
    altitude_ceiling: float = 4_000.0

    def __post_init__(self) -> None:
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.initial_altitude[0] <= self.touchdown_altitude:
            raise ValueError("initial altitude must be above the pad")


@dataclass(frozen=True)
class RewardConfig:
    """Weights of the dense shaping reward (AC-3.5)."""

    velocity_target_max: float = 40.0
    track_velocity_weight: float = 0.05
    track_position_weight: float = 0.01
    tilt_penalty: float = 0.5
    angular_rate_penalty: float = 0.02
    action_rate_penalty: float = 0.02
    gimbal_penalty: float = 0.01
    fuel_penalty: float = 0.001
    soft_landing_bonus: float = 100.0
    crash_penalty: float = 100.0
    crash_velocity_penalty: float = 2.0
    out_of_bounds_penalty: float = 100.0
    truncation_penalty: float = 25.0
    propellant_margin_bonus: float = 10.0


@dataclass(frozen=True)
class WorldModelConfig:
    """RSSM + TD-MPC2 architecture and optimization hyperparameters."""

    obs_dim: int = 18
    action_dim: int = 3
    latent_dim: int = 256
    gru_dim: int = 256
    hidden_dim: int = 256
    simnorm_groups: int = 32
    simnorm_v: int = 8
    fault_context_dim: int = 16
    num_fault_classes: int = NUM_FAULT_MODES
    horizon: int = 15
    batch_size: int = 128
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    gamma: float = 0.99
    lambda_returns: float = 0.9
    # Free bits are intentionally small: a 1.0-nat floor leaves no gradient
    # pressure below the floor and pins nominal KL at ~1, hiding fault spikes.
    free_bits: float = 0.05
    kl_alpha_prior: float = 0.8
    kl_alpha_post: float = 0.2
    consistency_coef: float = 1.0
    reward_coef: float = 1.0
    value_coef: float = 0.5
    fault_coef: float = 0.5
    kl_coef: float = 1.0
    obs_pred_coef: float = 1.0
    # FDIR anomaly score = KL + innovation_weight * normalised innovation
    # + residual_weight * exact actuator/rotational residual score.
    innovation_weight: float = 1.0
    residual_weight: float = 1.0
    # Weight of the instantaneous frozen-servo evidence in S_t.
    frozen_weight: float = 20.0
    # CoM detection is scored in metres of offset (not sigma).
    com_weight: float = 50.0
    com_detect_threshold: float = 0.05
    # The anomaly score is dominated by the exact residuals; KL is a *prior*
    # signal only (0.1 weight, as specified for the physics residual score).
    surprise_kl_weight: float = 0.1
    # Weight of the physics-informed isolation prior added to the learned head.
    isolation_prior_weight: float = 3.0
    innovation_channels: Tuple[int, ...] = (10, 11, 12, 14, 15, 16)
    bc_coef: float = 0.1
    # Pure-imitation warm start (actor still learns only from demonstrations).
    bc_only_updates: int = 800
    entropy_coef: float = 1.0e-3
    ema_tau: float = 0.005
    grad_clip_norm: float = 10.0
    min_latent_std: float = 0.05
    action_std_min: float = 0.02

    def __post_init__(self) -> None:
        if self.obs_dim != ObservationSchema().dim:
            raise ValueError("obs_dim must match ObservationSchema.dim")
        if self.action_dim != ActionSchema().dim:
            raise ValueError("action_dim must match ActionSchema.dim")
        if self.simnorm_groups * self.simnorm_v != self.latent_dim:
            raise ValueError("latent_dim must equal simnorm_groups * simnorm_v")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")


@dataclass(frozen=True)
class MPPIConfig:
    """Latent MPPI planner settings (AC-5.2)."""

    num_samples: int = 512
    num_policy_samples: int = 64
    horizon: int = 15
    iterations: int = 4
    num_elites: int = 32
    temperature: float = 10.0
    init_std: float = 0.35
    # When True the planner scores candidate rollouts with the *predicted
    # observation* (altitude, velocity, lateral error, tilt) instead of the
    # learned Q ensemble.  The observation head is supervised and accurate,
    # while Q is only as good as the value bootstrap.
    use_observation_cost: bool = False
    cost_position: float = 1.0
    cost_lateral: float = 1.0
    cost_vertical_speed: float = 0.8
    cost_lateral_speed: float = 1.0
    cost_tilt: float = 2.0
    cost_action: float = 0.02
    min_std: float = 0.05
    max_std: float = 1.0

    def __post_init__(self) -> None:
        if self.num_policy_samples > self.num_samples:
            raise ValueError("policy warm-start samples exceed total samples")
        if self.num_elites >= self.num_samples:
            raise ValueError("num_elites must be smaller than num_samples")


@dataclass(frozen=True)
class PPOConfig:
    """CleanRL-style PPO baseline hyperparameters."""

    hidden_dim: int = 256
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 1.0e-3
    value_coef: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    rollout_steps: int = 128
    max_grad_norm: float = 0.5
    # Keeps the residual head near the classical reference unless a deviation
    # is clearly profitable (the baseline is a *residual* around Cascaded-PID).
    residual_l2: float = 3.0


@dataclass(frozen=True)
class TrainingConfig:
    """Curriculum, logging and checkpointing settings (AC-6.2)."""

    total_env_steps: int = 500_000
    warmup_env_steps: int = 150_000
    num_envs: int = 256
    eval_envs: int = 16
    warmup_fault_probability: float = 0.15
    # Balanced fault sampling: half of the episodes are faults, uniformly
    # spread over the four failure modes.
    balanced_fault_sampling: bool = True
    pid_demo_episodes: int = 24
    pid_demo_max_transitions: int = 400_000
    nominal_demo_transitions: int = 100_000
    oracle_demo_transitions: int = 100_000
    explore_demo_transitions: int = 100_000
    dagger_transitions: int = 150_000
    eval_interval_updates: int = 1_000
    # Fraction of every gradient batch drawn from demonstration transitions
    # (keeps the BC anchor alive as the on-policy buffer fills).
    expert_fraction: float = 0.25
    warmup_nominal_fraction: float = 0.80
    # Update-to-data ratio: 16 updates per 4096 new transitions means one
    # gradient step per 256 environment steps (~8k transitions in the batch).
    gradient_updates_per_rollout: int = 16
    rollout_env_steps: int = 4_096
    replay_capacity: int = 4_000_000
    sequence_length: int = 16  # H + 1
    eval_interval: int = 2_500
    log_interval: int = 500
    checkpoint_interval: int = 25_000
    output_dir: str = "runs/aerowm_fdir"
    use_compile: bool = True
    use_bfloat16: bool = True
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.gradient_updates_per_rollout < 1:
            raise ValueError("need at least one gradient update per rollout")
        if self.warmup_env_steps > self.total_env_steps:
            raise ValueError("warmup must precede total steps")


def default_configs() -> dict:
    """Return freshly constructed default configs keyed by name."""
    return {
        "physics": RocketPhysicsConfig(),
        "observation": ObservationSchema(),
        "action": ActionSchema(),
        "fault": FaultInjectionSchema(),
        "environment": EnvironmentConfig(),
        "reward": RewardConfig(),
        "world_model": WorldModelConfig(),
        "mppi": MPPIConfig(),
        "ppo": PPOConfig(),
        "training": TrainingConfig(),
    }
