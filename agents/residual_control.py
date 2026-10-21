"""Fault-conditioned residual controller.

Structure
---------
``u_total = clip(u_base(o_t, fdir_estimate) + delta_u_wm, -1, 1)``

``u_base``
    The cascaded feedback law, augmented with the **FDIR-predicted** fault
    class and parameters (never ground truth).  Using the same compensation
    law as the oracle keeps the base controller close to the demonstration
    policy, so the learned residual only has to make a small correction.

``delta_u_wm``
    A bounded residual (``0.15 * tanh(mu)``) from the world-model actor, or a
    bounded deviation of the latent MPPI plan from ``u_base``.  The actor's
    output layer is zero-initialised, so an untrained / early-checkpoint
    policy reproduces ``u_base`` exactly instead of destroying it.
"""

from __future__ import annotations

from typing import Optional

import torch

from agents.base import ControllerMixin
from agents.pid_baseline import CascadePIDController
from configs.schema import ActionSchema, FaultParamsLayout
from models.world_model import RocketWorldModel


class EstimatedFaultState:
    """Minimal ``FaultState``-compatible container for predicted parameters."""

    def __init__(self, params: torch.Tensor, active: torch.Tensor) -> None:
        self.params = params
        self.active = active

    def applied_params(self) -> torch.Tensor:
        """Zero out inactive faults so they are physically inert."""
        p = self.params.clone()
        p = torch.where(self.active.unsqueeze(-1), p, torch.zeros_like(p))
        idx = FaultParamsLayout.THRUST_SCALE
        p[..., idx] = torch.where(
            self.active, p[..., idx], torch.ones_like(p[..., idx])
        )
        return p


@torch.no_grad()
def estimate_fault_state(
    logits: torch.Tensor,
    param_hat: torch.Tensor,
    frozen: Optional[torch.Tensor],
    confidence: float = 0.6,
    residuals: Optional[torch.Tensor] = None,
    gimbal_limit: float = 0.13962634015954636,
    bias: Optional[torch.Tensor] = None,
    obs: Optional[torch.Tensor] = None,
    com_estimate: Optional[torch.Tensor] = None,
    thrust_gain: float = 1.0 - __import__("math").exp(-0.02 / 0.10),
    a_prev: Optional[torch.Tensor] = None,
) -> EstimatedFaultState:
    """Map FDIR outputs to the packed fault-parameter layout.

    Classes follow :class:`configs.schema.FaultMode`: 1 LOE, 2 bias, 3 stuck,
    4 CoM.  Continuous magnitudes come from the parameter head, so the
    compensation degrades gracefully when the severity is mis-estimated.
    """
    probs = torch.softmax(logits, dim=-1)
    cls = probs.argmax(dim=-1)
    confident = probs.max(dim=-1).values > confidence
    batch = cls.shape[0]
    device = logits.device

    zeros = torch.zeros(batch, device=device, dtype=logits.dtype)
    is_loe = (cls == 1) & confident
    is_bias = (cls == 2) & confident
    is_stuck = (cls == 3) & confident
    is_com = (cls == 4) & confident

    if bias is not None:
        bias_y = torch.where(is_bias, bias[..., 0], zeros)
        bias_z = torch.where(is_bias, bias[..., 1], zeros)
    if residuals is not None:
        # Direct physics read-out: the gimbal residual *is* the bias (radians,
        # after undoing the normalisation), and the thrust residual *is* the
        # fractional loss.  Exact, and independent of the learned head.
        from models.fdir_detector import RES_GIMBAL, RES_THRUST

        # ``RES_GIMBAL`` is (measured - expected) / gimbal_limit, so multiply
        # back to radians; ``RES_THRUST`` is a fraction already.
        bias_y_raw = residuals[..., RES_GIMBAL][..., 0] * gimbal_limit
        bias_z_raw = residuals[..., RES_GIMBAL][..., 1] * gimbal_limit
        # Deconvolve the turbopump lag: the residual only contains
        # ``thrust_gain * (s_true - 1)`` of the commanded thrust, so divide it
        # back out (without this the loss was under-estimated ~5x and the
        # controller under-compensated, which is the LOE landing shortfall).
        # The residual carries ``gain * thrust_norm * (s_true - 1)``, so it
        # needs both the spool gain and the commanded thrust fraction to
        # deconvolve.  The direct measured/commanded ratio is exact at steady
        # state and is used when the command is available.
        loss_res = -residuals[..., RES_THRUST].squeeze(-1) / max(thrust_gain, 1e-3)
        loss_raw = loss_res.clamp(0.0, 0.9)
        if a_prev is not None:
            from models.fdir_detector import throttle_to_thrust_norm

            norm = throttle_to_thrust_norm(a_prev[..., 2], __import__("configs.schema", fromlist=["RocketPhysicsConfig"]).RocketPhysicsConfig())
            measured = (obs[..., 16] if obs is not None else torch.zeros_like(norm))
            ratio = measured / norm.clamp_min(0.08)
            loss_ratio = (1.0 - ratio).clamp(0.0, 0.9)
            # Trust the direct ratio wherever the engine is actually firing.
            loss_raw = torch.where(norm > 0.15, loss_ratio, loss_raw)
    else:
        bias_y_raw = param_hat[..., 0]
        bias_z_raw = param_hat[..., 1]
        loss_raw = param_hat[..., 2]
    thrust_scale = torch.where(is_loe, (1.0 - loss_raw).clamp(0.2, 1.0), torch.ones_like(zeros))
    if bias is None:
        bias_y = torch.where(is_bias, bias_y_raw, zeros)
        bias_z = torch.where(is_bias, bias_z_raw, zeros)
    com_y = torch.where(is_com, param_hat[..., 3], zeros)
    com_z = torch.where(is_com, param_hat[..., 4], zeros)
    if com_estimate is not None:
        # Physical read-out (metres), preferred over the learned head.
        # tau_x = T (com_z + L dy), tau_y = -T (com_y + L dz), so the
        # residual torque / thrust is [com_z, -com_y].
        #
        # ``com_estimate`` is the *averaged* inversion of the rotational
        # residual (``com = J * domega_res / dt / T``), which is the only
        # usable form: the per-sample rotational residual sits far below the
        # measurement noise floor, so inverting it sample by sample produces a
        # jittery trim command and the attitude loop diverges to the 60 deg
        # limit (measured: CoM recovery 60-80 % with the instantaneous
        # read-out, 95 % with the averaged one).
        com_y = torch.where(is_com, -com_estimate[..., 1], zeros)
        com_z = torch.where(is_com, com_estimate[..., 0], zeros)
    elif residuals is not None and obs is not None:
        # Fallback: invert the rotational residual into the unmodelled CoM
        # offset.  tau_res = J * domega_res / dt and tau_x = T*(com_z + L dy),
        # tau_y = -T*(L dz + com_y).  The gimbal terms are already in the
        # prediction, so tau_res/T is the offset directly.  ``res_omega`` is
        # stored as ``domega / angular_rate_scale``.
        from configs.schema import RocketPhysicsConfig
        phys = RocketPhysicsConfig()
        mass = (phys.dry_mass + obs[:, 13] * phys.propellant_mass).clamp_min(phys.dry_mass)
        inertia = mass * phys.specific_inertia[0]
        thrust = (obs[:, 16] * phys.max_thrust).clamp_min(1e3)
        tau_res = residuals[..., 3:6] * 2.0 * inertia.unsqueeze(-1) / phys.dt
        com_z_direct = (tau_res[:, 0] / thrust).clamp(-0.5, 0.5)
        com_y_direct = (-tau_res[:, 1] / thrust).clamp(-0.5, 0.5)
        com_y = torch.where(is_com, com_y_direct, zeros)
        com_z = torch.where(is_com, com_z_direct, zeros)

    if frozen is None:
        stuck_y = is_stuck & (param_hat[..., 0].abs() >= param_hat[..., 1].abs())
        stuck_z = is_stuck & ~stuck_y
    else:
        stuck_y = is_stuck & (frozen[..., 0] >= frozen[..., 1])
        stuck_z = is_stuck & (frozen[..., 1] > frozen[..., 0])

    params = torch.stack(
        [thrust_scale, bias_y, bias_z, com_y, com_z, stuck_y.to(logits.dtype), stuck_z.to(logits.dtype)],
        dim=-1,
    )
    active = (cls != 0) & confident
    return EstimatedFaultState(params, active)


class FDIRResidualController(ControllerMixin):
    """``u_base`` (FDIR-conditioned) + bounded learned residual."""

    action_schema = ActionSchema()

    def __init__(
        self,
        model: RocketWorldModel,
        agent,
        device: torch.device | str = "cuda",
        base: Optional[CascadePIDController] = None,
        residual_scale: float = 0.15,
        planner_scale: float = 0.02,
        mode: str = "actor",
        planner=None,
        use_fdir: bool = True,
    ) -> None:
        self.model = model
        self.agent = agent
        self.device = torch.device(device)
        self.residual_scale = float(residual_scale)
        self.planner_scale = float(planner_scale)
        self.mode = mode
        self.planner = planner
        self.use_fdir = use_fdir
        self.base = base or CascadePIDController(
            device=self.device, fault_estimation_mode=True
        )
        self.reset(1, self.device)

    @property
    def action_dim(self) -> int:
        return self.action_schema.dim

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        if device is not None:
            self.device = torch.device(device)
        cfg = self.model.cfg
        self.h = torch.zeros(batch_size, cfg.gru_dim, device=self.device)
        self.z = torch.zeros(batch_size, cfg.latent_dim, device=self.device)
        self.c = torch.zeros(batch_size, cfg.fault_context_dim, device=self.device)
        self.prev_action = torch.zeros(batch_size, cfg.action_dim, device=self.device)
        self.prev_obs: Optional[torch.Tensor] = None
        self.surprise = torch.zeros(batch_size, device=self.device)
        self.logits = torch.zeros(batch_size, cfg.num_fault_classes, device=self.device)
        self.fault_params = torch.zeros(batch_size, 5, device=self.device)
        self.frozen = torch.zeros(batch_size, 2, device=self.device)
        # Latched fault estimate: a fault does not heal, so once the detector
        # fires the parameters are held for the rest of the episode.  This also
        # removes class flicker between adjacent samples.
        self.latched_params = torch.zeros(batch_size, 7, device=self.device)
        self.latched_active = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        # Fault-class latch.  The alarm is raised on the onset transient, but
        # closed-loop compensation then nulls the very residual that isolates
        # the fault -- so the class must be decided once, two samples after
        # detection (long enough for the 2-step frozen-servo confirmation), and
        # never overwritten back to NOMINAL.
        self.detected = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.detect_age = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.latched_cls = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.cls_set = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.cls_streak = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.cls_prev = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.com_streak = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.frozen_seen = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.last_debug = {}
        self.base.reset(batch_size, self.device)
        if hasattr(self.model, 'fdir'):
            self.model.fdir.reset_state()
        if self.planner is not None and hasattr(self.planner, "reset"):
            self.planner.reset(batch_size, self.device)

    def reset_mask(self, mask: torch.Tensor) -> None:
        mask = mask.to(self.device, torch.bool)
        if not bool(mask.any()):
            return
        self.h[mask] = 0.0
        self.z[mask] = 0.0
        self.c[mask] = 0.0
        self.prev_action[mask] = 0.0
        self.surprise[mask] = 0.0
        self.logits[mask] = 0.0
        self.frozen[mask] = 0.0
        self.latched_params[mask] = 0.0
        self.latched_active[mask] = False
        self.detected[mask] = False
        self.detect_age[mask] = 0
        self.latched_cls[mask] = 0
        self.cls_set[mask] = False
        self.cls_streak[mask] = 0
        self.cls_prev[mask] = 0
        self.com_streak[mask] = 0
        self.frozen_seen[mask] = False
        if self.prev_obs is not None:
            self.prev_obs = self.prev_obs.clone()
            self.prev_obs[mask] = 0.0
        for obj in (self.base, self.planner):
            if obj is not None and hasattr(obj, "reset_mask"):
                obj.reset_mask(mask)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(self, obs: torch.Tensor) -> None:
        obs = obs.to(self.device)
        self.h = self.model.recurrence(self.h, self.z, self.prev_action)
        e = self.model.encode(obs)
        mu_q, log_q = self.model.posterior_params(self.h, e)
        self.z = self.model.latent_from_params(mu_q, log_q, sample=False)
        mu_p, log_p = self.model.prior_params(self.h)
        obs_hat = self.model.predict_observation(self.h, mu_p)
        innovation = obs - obs_hat
        # ``None`` on the first step of an episode: there is no (o_{t-1},
        # a_{t-1}) pair yet, and a zero action maps to *minimum thrust*, so
        # inventing one produces a large spurious residual.
        obs_prev = self.prev_obs
        signals = self.model.fault_head(
            self.h,
            self.z,
            mu_q,
            log_q,
            mu_p,
            log_p,
            innovation,
            self.prev_action,
            obs_prev=obs_prev,
            obs=obs,
        )
        self.logits = signals.logits
        self.surprise = signals.surprise
        self.fault_params = signals.fault_params
        self.frozen = signals.frozen if signals.frozen is not None else torch.zeros_like(self.frozen)
        self.features = signals.features
        self.head_inputs = signals.head_inputs
        self.bias_estimate = signals.bias_estimate
        self.com_estimate = signals.com_estimate
        self.c = signals.context if self.use_fdir else torch.zeros_like(signals.context)
        self.prev_obs = obs.detach()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, fault_context=None) -> torch.Tensor:
        obs = self._check_obs(obs).to(self.device)
        if self.h.shape[0] != obs.shape[0]:
            self.reset(obs.shape[0], obs.device)
        self.encode(obs)

        # --- fault-class latch (see __init__) --------------------------- #
        threshold = float(self.model.fdir.detector.threshold)
        alarm = self.surprise > threshold
        self.detected = self.detected | alarm
        self.detect_age = torch.where(
            self.detected, self.detect_age + 1, torch.zeros_like(self.detect_age)
        )
        cls_now = self.logits.argmax(dim=-1)
        # Require the class to persist for two consecutive samples: a nominal
        # transient can produce a one-sample argmax but never a stable one.
        same = (cls_now == self.cls_prev) & (cls_now != 0)
        self.cls_streak = torch.where(same, self.cls_streak + 1, torch.zeros_like(self.cls_streak))
        self.cls_prev = cls_now
        # CoM gets a dedicated latch: its evidence is a physical offset in
        # metres, which the sigma-based argmax can miss on a low-thrust sample.
        com_thr = float(self.model.cfg.com_detect_threshold)
        if self.com_estimate is not None:
            # A CoM offset is only diagnosable when the *actuators* track:
            # a gimbal bias/stuck (or the RCS that substitutes for it) also
            # produces an unexplained torque, which would masquerade as a CoM
            # shift.  Require the actuator residuals and the frozen test to be
            # nominal before attributing torque to the CoM.
            act_z = self.model.fdir.residuals.normalize(self.features)
            act_ok = (act_z[..., 0:3].abs().amax(dim=-1) < 3.0) & (
                self.frozen.amax(dim=-1) < 0.5
            )
            com_flag = (self.com_estimate.abs().amax(dim=-1) > 1.2 * com_thr) & act_ok
            self.com_streak = torch.where(
                com_flag, self.com_streak + 1, torch.zeros_like(self.com_streak)
            )
        else:
            com_flag = torch.zeros_like(self.cls_set)
        # A confirmed frozen servo always wins: it is a direct measurement,
        # whereas a CoM attribution is an inference from unexplained torque.
        self.frozen_seen = self.frozen_seen | (self.frozen.amax(dim=-1) > 0.5)
        # The frozen indicator is already 2-sample confirmed inside the
        # extractor, so one confirmed sample is enough to latch STUCK.
        take_frozen = (
            self.detected
            & ~self.cls_set
            & (self.frozen.amax(dim=-1) > 0.5)
        )
        take_com = (
            self.detected
            & ~self.cls_set
            & ~take_frozen
            & (self.com_streak >= 8)
            & ~self.frozen_seen
        )
        take = (
            self.detected
            & ~self.cls_set
            & (self.detect_age >= 2)
            & (self.cls_streak >= 2)
            & (cls_now != 0)
            & ~take_com
        )
        new_cls = torch.where(take_com, torch.full_like(cls_now, 4), cls_now)
        new_cls = torch.where(take_frozen, torch.full_like(cls_now, 3), new_cls)
        take_any = take | take_com | take_frozen
        self.latched_cls = torch.where(take_any, new_cls, self.latched_cls)
        self.cls_set = self.cls_set | take_any
        if bool(self.cls_set.any()):
            report_logits = torch.nn.functional.one_hot(
                self.latched_cls, num_classes=self.logits.shape[-1]
            ).to(self.logits.dtype) * 10.0
            report_logits = torch.where(
                self.cls_set.unsqueeze(-1), report_logits, self.logits
            )
        else:
            report_logits = self.logits

        estimate = estimate_fault_state(
            report_logits,
            self.fault_params,
            self.frozen,
            residuals=self.features,
            bias=self.bias_estimate,
            obs=obs,
            com_estimate=self.com_estimate,
            a_prev=self.prev_action,
            confidence=0.5,
        )
        fresh = estimate.active.unsqueeze(-1)
        self.latched_params = torch.where(fresh, estimate.params, self.latched_params)
        self.latched_active = self.latched_active | estimate.active
        estimate = EstimatedFaultState(self.latched_params, self.latched_active)
        u_base = self.base.act(obs, fault_state=estimate)

        # The world-model policy prior emits a *residual* around the
        # fault-conditioned base law, so the absolute action it implies is
        # ``u_base + delta_policy``.
        delta_policy = self.agent.residual_mean(self.h, self.z, self.c, obs) * self.residual_scale
        if self.mode == "mppi" and self.planner is not None:
            # The planner only *refines* the law: its deviation is bounded by
            # planner_scale, so an under-trained plan degrades gracefully to
            # the fault-conditioned base controller instead of crashing.
            # As in TD-MPC2 the plan is warm-started from the policy prior, so
            # the anchor is the *absolute* action the prior implies.
            anchor = (u_base + delta_policy).clamp(-1.0, 1.0)
            planned = self.planner.act(obs, anchor=anchor, latent=(self.h, self.z, self.c))
            # A latched fault means the base law is carrying the precision
            # compensation (bias / CoM cancellation, thrust re-scaling) and the
            # world model does not know about the fault, so the planner must
            # not perturb the actuator: its deviation is switched off *per
            # environment* (an episode that has latched a fault hands the
            # actuator back to the fault-conditioned law, while a nominal
            # episode in the same batch keeps the full planner refinement).
            latched = self.cls_set.to(planned.dtype).unsqueeze(-1)
            att_scale = (1.0 - latched) * float(self.planner_scale)
            thr_scale = (1.0 - latched) * float(self.planner_scale) * 0.25
            # Only the two attitude channels are delegated: the throttle sets
            # the descent profile and is far more sensitive to plan noise, so
            # it stays on the base law (bounded refinement).
            raw = planned - anchor
            delta = torch.cat(
                [
                    raw[..., :2].clamp(-att_scale, att_scale),
                    raw[..., 2:3].clamp(-thr_scale, thr_scale),
                ],
                dim=-1,
            )
            action = (anchor + delta).clamp(-1.0, 1.0)
        else:
            delta = delta_policy
            action = (u_base + delta).clamp(-1.0, 1.0)
        self.prev_action = action
        self.last_debug = {
            "fault_logits": report_logits,
            "surprise": self.surprise,
            "fault_params": self.fault_params,
            "u_base": u_base,
            "delta": delta,
        }
        return action
