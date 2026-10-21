"""Batched latent-space MPPI / CEM planner (TD-MPC2 style).

The planner keeps the RSSM posterior state of every environment, then optimises
a horizon of actions entirely inside the learned prior.  The first action of
the best sequence is executed.  The policy prior supplies a warm start.
"""

from __future__ import annotations

from typing import Optional

import torch

from agents.base import ControllerMixin
from agents.world_model_agent import WorldModelAgent
from configs.schema import ActionSchema, MPPIConfig, ObservationSchema, WorldModelConfig
from models.world_model import RocketWorldModel


class LatentMPPIPlanner(ControllerMixin):
    """Model-predictive path integral planner in the world model's latent space."""

    action_schema = ActionSchema()

    def __init__(
        self,
        model: RocketWorldModel,
        agent: WorldModelAgent,
        cfg: Optional[MPPIConfig] = None,
        observation: Optional[ObservationSchema] = None,
        device: torch.device | str = "cuda",
        use_fdir: bool = True,
        autocast: bool = False,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        self.model = model
        self.agent = agent
        self.cfg = cfg or MPPIConfig()
        self.wm_cfg: WorldModelConfig = model.cfg
        self.observation = observation or ObservationSchema()
        self.device = torch.device(device)
        self.use_fdir = use_fdir
        # NOTE: bfloat16 autocast is used for *training*; for the small
        # sequential GEMMs inside the planner fp32 is measurably faster on the
        # RTX 4060, so the planner defaults to fp32.
        self.autocast = autocast and self.device.type == "cuda"
        self.action_low = action_low
        self.action_high = action_high

        self.h: Optional[torch.Tensor] = None
        self.z: Optional[torch.Tensor] = None
        self.context: Optional[torch.Tensor] = None
        self.prev_action: Optional[torch.Tensor] = None
        self.mean_plan: Optional[torch.Tensor] = None
        self.surprise: Optional[torch.Tensor] = None
        self.last_debug: dict = {}
        self.last_obs: Optional[torch.Tensor] = None
        self.prev_obs: Optional[torch.Tensor] = None

    @property
    def action_dim(self) -> int:
        return self.action_schema.dim

    # ------------------------------------------------------------------ #
    # State management
    # ------------------------------------------------------------------ #
    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        if device is not None:
            self.device = device
        dtype = next(self.model.parameters()).dtype
        self.h = torch.zeros(batch_size, self.wm_cfg.gru_dim, device=self.device, dtype=dtype)
        self.z = torch.zeros(batch_size, self.wm_cfg.latent_dim, device=self.device, dtype=dtype)
        self.context = torch.zeros(batch_size, self.wm_cfg.fault_context_dim, device=self.device, dtype=dtype)
        self.prev_action = torch.zeros(batch_size, self.action_dim, device=self.device, dtype=dtype)
        self.mean_plan = torch.zeros(
            batch_size, self.cfg.horizon, self.action_dim, device=self.device, dtype=dtype
        )
        self.surprise = torch.zeros(batch_size, device=self.device, dtype=dtype)
        self.prev_obs = None
        if hasattr(self.model, 'fdir'):
            self.model.fdir.reset_state()

    def reset_mask(self, mask: torch.Tensor) -> None:
        """Clear the recurrent carry-over of auto-reset environments."""
        mask = mask.to(self.device, torch.bool)
        if not bool(mask.any()):
            return
        self.h[mask] = 0.0
        self.z[mask] = 0.0
        self.context[mask] = 0.0
        self.prev_action[mask] = 0.0
        self.mean_plan[mask] = 0.0
        if self.prev_obs is not None:
            self.prev_obs[mask] = 0.0

    def _encode(self, obs: torch.Tensor) -> None:
        obs = obs.to(self.device)
        self.last_obs = obs
        self.h = self.model.recurrence(self.h, self.z, self.prev_action)
        e = self.model.encode(obs)
        mu_q, log_q = self.model.posterior_params(self.h, e)
        self.z = self.model.latent_from_params(mu_q, log_q, sample=False)
        mu_p, log_p = self.model.prior_params(self.h)
        obs_hat = self.model.predict_observation(self.h, mu_p)
        innovation = obs - obs_hat
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
        self.prev_obs = obs.detach()
        logits, context, surprise = signals.logits, signals.context, signals.surprise
        if self.use_fdir:
            self.context = context
        else:
            self.context = torch.zeros_like(context)
        self.surprise = surprise
        self.last_debug["fault_logits"] = logits
        self.last_debug["surprise"] = surprise
        self.last_debug["kl"] = signals.kl
        self.last_debug["innovation"] = signals.innovation
        self.last_debug["obs_innovation"] = innovation

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        fault_context: Optional[torch.Tensor] = None,
        anchor: Optional[torch.Tensor] = None,
        latent: Optional[tuple] = None,
    ) -> torch.Tensor:
        obs = self._check_obs(obs)
        if self.h is None or self.h.shape[0] != obs.shape[0]:
            self.reset(obs.shape[0], obs.device)
        if latent is None:
            self._encode(obs)
        else:
            # Share the caller's recurrent state instead of encoding twice.
            # A second ``fault_head`` call per step would advance the FDIR
            # residual EMAs with the *planned* action rather than the one the
            # plant executed, inflating nominal surprise (and therefore the
            # calibration threshold) by ~10x and delaying LOE detection from
            # 0.9 to 14.6 steps.
            h, z, context = latent
            self.h, self.z, self.context = h, z, context
            self.last_obs = obs.to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast):
            action = self._plan(anchor=anchor)
        self.prev_action = action.to(self.prev_action.dtype)
        return self._clip_action(action.float())

    def _plan(self, anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
        cfg = self.cfg
        device = self.device
        dtype = self.h.dtype
        batch = self.h.shape[0]
        horizon = cfg.horizon
        num = cfg.num_samples

        gamma = self.wm_cfg.gamma
        discounts = torch.pow(
            torch.tensor(gamma, device=device, dtype=dtype), torch.arange(horizon, device=device)
        )
        gamma_h = gamma**horizon

        # Warm start from the policy prior, shifted by one step.
        shifted = torch.zeros_like(self.mean_plan)
        shifted[:, :-1] = self.mean_plan[:, 1:]
        mean = shifted
        std = torch.full_like(mean, cfg.init_std)

        warm_samples = min(cfg.num_policy_samples, num)
        if anchor is not None:
            # The policy prior emits a *residual* around the fault-conditioned
            # base law, so it cannot warm-start an absolute-action planner.
            # Anchor the search on the base action instead: the planner then
            # only searches its neighbourhood.
            anchor = anchor.to(device=device, dtype=dtype).clamp(self.action_low, self.action_high)
            mean = anchor.unsqueeze(1).expand(batch, horizon, self.action_dim).clone()
            std = torch.full_like(mean, cfg.init_std * 0.5)
            warm = anchor.unsqueeze(1).expand(batch, horizon, self.action_dim).clone()
        else:
            warm = self._policy_rollout(horizon)  # [B, H, 3]

        best_action = mean[:, 0]
        for iteration in range(cfg.iterations):
            noise = torch.randn(batch, num, horizon, self.action_dim, device=device, dtype=dtype)
            actions = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(self.action_low, self.action_high)
            if warm_samples > 0:
                jitter = torch.randn(
                    batch, warm_samples, horizon, self.action_dim, device=device, dtype=dtype
                ) * (0.15 if iteration else 0.05)
                actions[:, :warm_samples] = (warm.unsqueeze(1) + jitter).clamp(
                    self.action_low, self.action_high
                )

            scores = self._score(actions, discounts, gamma_h, horizon)
            elites = torch.topk(scores, cfg.num_elites, dim=1).indices
            elite_actions = actions.gather(
                1, elites.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, horizon, self.action_dim)
            )
            elite_scores = scores.gather(1, elites)
            weights = torch.softmax(
                cfg.temperature * (elite_scores - elite_scores.max(dim=1, keepdim=True).values), dim=1
            )
            mean = (weights.unsqueeze(-1).unsqueeze(-1) * elite_actions).sum(dim=1)
            var = (weights.unsqueeze(-1).unsqueeze(-1) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1)
            std = var.sqrt().clamp(cfg.min_std, cfg.max_std)
            best_action = mean[:, 0]

        self.mean_plan = torch.cat([mean[:, 1:], mean[:, -1:]], dim=1)
        self.last_debug["plan_mean"] = mean
        return best_action

    def _score(
        self,
        actions: torch.Tensor,
        discounts: torch.Tensor,
        gamma_h: float,
        horizon: int,
    ) -> torch.Tensor:
        if self.cfg.use_observation_cost:
            out = self.model.imagine(
                self.h, self.z, self.context, actions, need_obs=True
            )
            return self._observation_score(out["obs"], actions, discounts)
        out = self.model.imagine(self.h, self.z, self.context, actions)
        rewards = out["reward"]
        returns = (rewards * discounts.reshape(1, 1, -1)).sum(dim=-1)
        h_terminal = out["h"]
        z_terminal = out["z"]
        context = out["context"]
        batch, samples = h_terminal.shape[:2]
        # The current observation is the best available estimate of the state
        # at the end of the rollout; broadcast it across candidates.
        obs_final = self.last_obs.unsqueeze(1).expand(batch, samples, -1).reshape(
            batch * samples, -1
        )
        final_action = self.agent.actor.mean_action(
            h_terminal.reshape(batch * samples, -1),
            z_terminal.reshape(batch * samples, -1),
            context.reshape(batch * samples, -1),
            obs_final,
        )
        q1, q2 = self.model.q_values(h_terminal, z_terminal, context, final_action)
        terminal_value = torch.minimum(q1, q2).reshape(batch, samples)
        return returns + gamma_h * terminal_value

    def _observation_score(
        self, obs_pred: torch.Tensor, actions: torch.Tensor, discounts: torch.Tensor
    ) -> torch.Tensor:
        """Model-predictive cost in (normalised) observation space.

        Uses the supervised one-step observation predictor, so the planner
        steers the *predicted physical state* rather than relying on a value
        function that may not have converged.
        """
        cfg = self.cfg
        pos = obs_pred[..., 0:3]
        vel = obs_pred[..., 3:6]
        quat = obs_pred[..., 6:10]
        lateral = pos[..., 0:2].pow(2).sum(-1).sqrt()
        alt = pos[..., 2]
        v_lat = vel[..., 0:2].pow(2).sum(-1).sqrt()
        v_up = vel[..., 2]
        tilt = (1.0 - (1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)).clamp(-1.0, 1.0))
        cost = (
            cfg.cost_lateral * lateral
            + cfg.cost_position * alt.clamp_min(0.0) * 0.1
            + cfg.cost_vertical_speed * v_up.abs() * 0.2
            + cfg.cost_lateral_speed * v_lat * 0.2
            + cfg.cost_tilt * tilt * 10.0
            + cfg.cost_action * actions.pow(2).sum(-1)
        )
        return -(cost * discounts.reshape(1, 1, -1)).sum(dim=-1)

    def _policy_rollout(self, horizon: int) -> torch.Tensor:
        """One prior rollout of the mean policy, used as a warm start."""
        h, z = self.h, self.z
        actions = []
        for _ in range(horizon):
            a = self.agent.actor.mean_action(h, z, self.context, self.last_obs)
            actions.append(a)
            h, z = self.model.prior_state(h, z, a)
        return torch.stack(actions, dim=1)
