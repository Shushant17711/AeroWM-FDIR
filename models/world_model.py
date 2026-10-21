"""RSSM + TD-MPC2 latent world model for the 6-DOF rocket.

The model is decoder-free: a posterior encoder grounds the latent state in
telemetry, a GRU carries history, and a prior predicts the next latent before
the observation arrives.  The KL between the two is the FDIR surprise signal.
Reward and a twin-Q ensemble are conditioned on the fault context produced by
:class:`models.fdir_head.FDIRModule`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.schema import WorldModelConfig
from models.fdir_head import FDIRModule, FDIRSignals
from models.networks import GRUCell, NormMLP, SimNorm, reparameterize


@dataclass
class LatentState:
    h: torch.Tensor
    z: torch.Tensor
    context: torch.Tensor


class RocketWorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig, use_simnorm: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_simnorm = use_simnorm
        dim = cfg.latent_dim
        hidden = cfg.hidden_dim
        gru = cfg.gru_dim
        ctx = cfg.fault_context_dim

        self.obs_encoder = NormMLP(cfg.obs_dim, hidden, dim, num_layers=2)
        self.posterior = NormMLP(gru + dim, hidden, 2 * dim, num_layers=2)
        self.prior_net = NormMLP(gru, hidden, 2 * dim, num_layers=2)
        self.gru = GRUCell(dim + cfg.action_dim, gru)
        self.simnorm = SimNorm(dim, cfg.simnorm_groups) if use_simnorm else nn.Identity()

        # One-step-ahead observation predictor: uses only (h_t, prior latent),
        # so its residual is a genuine innovation rather than a reconstruction.
        self.prior_obs_head = NormMLP(gru + dim, hidden, cfg.obs_dim, num_layers=2)

        self.reward_head = NormMLP(gru + dim + ctx + cfg.action_dim, hidden, 1, num_layers=2)
        self.q1 = NormMLP(gru + dim + ctx + cfg.action_dim, hidden, 1, num_layers=2)
        self.q2 = NormMLP(gru + dim + ctx + cfg.action_dim, hidden, 1, num_layers=2)
        # Predicts the *next* observation embedding from (h, z, a).
        self.consistency_head = NormMLP(gru + dim + cfg.action_dim, hidden, dim, num_layers=2)

        self.fdir = FDIRModule(cfg)

        # Target ensemble (EMA, frozen gradients).
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        for p in list(self.target_q1.parameters()) + list(self.target_q2.parameters()):
            p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.cfg.gru_dim, device=device, dtype=dtype)
        z = torch.zeros(batch_size, self.cfg.latent_dim, device=device, dtype=dtype)
        return h, z

    def _split(self, out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = out.chunk(2, dim=-1)
        return mu, log_std.clamp(-5.0, 2.0)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obs_encoder(obs)

    # ------------------------------------------------------------------ #
    # Single-step transitions
    # ------------------------------------------------------------------ #
    def recurrence(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.gru(torch.cat([z, action], dim=-1), h)

    def posterior_params(self, h: torch.Tensor, obs_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._split(self.posterior(torch.cat([h, obs_embedding], dim=-1)))

    def prior_params(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._split(self.prior_net(h))

    def latent_from_params(
        self, mu: torch.Tensor, log_std: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        z = reparameterize(mu, log_std) if sample else mu
        return self.simnorm(z)

    def predict_observation(self, h: torch.Tensor, mu_prior: torch.Tensor) -> torch.Tensor:
        """One-step-ahead observation prediction from the prior only."""
        return self.prior_obs_head(torch.cat([h, mu_prior], dim=-1))

    def fault_head(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        mu_post: torch.Tensor,
        log_std_post: torch.Tensor,
        mu_prior: torch.Tensor,
        log_std_prior: torch.Tensor,
        obs_innovation: torch.Tensor,
        a_prev: torch.Tensor,
        obs_prev: Optional[torch.Tensor] = None,
        obs: Optional[torch.Tensor] = None,
    ) -> FDIRSignals:
        features, frozen, raw, bias, frozen_instant, com_est = self.fdir.residual_features(
            obs_prev, obs, a_prev
        )
        return self.fdir(
            h,
            z,
            mu_post,
            log_std_post,
            mu_prior,
            log_std_prior,
            obs_innovation,
            a_prev,
            residual_features=features,
            frozen=frozen,
            raw_residuals=raw,
            bias_estimate=bias,
            frozen_instant_input=frozen_instant,
            com_input=com_est,
            com_estimate=com_est,
        )

    def reward(self, h: torch.Tensor, z: torch.Tensor, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat([h, z, context, action], dim=-1)).squeeze(-1)

    @staticmethod
    def _flat(h, z, context, action):
        """Flatten leading dims; return (flat tensors, original leading shape)."""
        shape = h.shape[:-1]
        return (
            h.reshape(-1, h.shape[-1]),
            z.reshape(-1, z.shape[-1]),
            context.reshape(-1, context.shape[-1]),
            action.reshape(-1, action.shape[-1]),
            shape,
        )

    def q_values(
        self, h: torch.Tensor, z: torch.Tensor, context: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, z, context, action, shape = self._flat(h, z, context, action)
        x = torch.cat([h, z, context, action], dim=-1)
        return self.q1(x).squeeze(-1).reshape(shape), self.q2(x).squeeze(-1).reshape(shape)

    def target_values(
        self, h: torch.Tensor, z: torch.Tensor, context: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            h, z, context, action, shape = self._flat(h, z, context, action)
            x = torch.cat([h, z, context, action], dim=-1)
            return (
                self.target_q1(x).squeeze(-1).reshape(shape),
                self.target_q2(x).squeeze(-1).reshape(shape),
            )

    def consistency(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicted embedding of the *next* observation (decoder-free target)."""
        return self.consistency_head(torch.cat([h, z, action], dim=-1))

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        for src, dst in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            for p_src, p_dst in zip(src.parameters(), dst.parameters()):
                p_dst.mul_(1.0 - tau).add_(tau * p_src)

    # ------------------------------------------------------------------ #
    # Sequence unroll (training)
    # ------------------------------------------------------------------ #
    def forward_sequence(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        z0: Optional[torch.Tensor] = None,
        a0: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Unroll the RSSM over ``[B, T, ...]`` observations and actions."""
        if obs.ndim != 3 or action.ndim != 3:
            raise ValueError("obs and action must be [B, T, dim]")
        batch, horizon, _ = obs.shape
        device, dtype = obs.device, obs.dtype
        h = h0 if h0 is not None else torch.zeros(batch, self.cfg.gru_dim, device=device, dtype=dtype)
        z_prev = z0 if z0 is not None else torch.zeros(batch, self.cfg.latent_dim, device=device, dtype=dtype)
        a_prev = a0 if a0 is not None else torch.zeros(batch, self.cfg.action_dim, device=device, dtype=dtype)

        out: Dict[str, list] = {
            key: []
            for key in (
                "h", "z", "z_prior", "mu_post", "log_std_post", "mu_prior", "log_std_prior",
                "reward", "q1", "q2", "fault_logits", "context", "surprise", "consistency",
                "embedding", "obs_hat", "obs_innovation", "kl", "innovation",
                "residual_score", "fault_params", "residual_features",
            )
        }
        for k in range(horizon):
            h = self.recurrence(h, z_prev, a_prev)
            e = self.encode(obs[:, k])
            mu_q, log_std_q = self.posterior_params(h, e)
            z = self.latent_from_params(mu_q, log_std_q, sample=sample)
            mu_p, log_std_p = self.prior_params(h)
            z_prior = self.simnorm(mu_p) if self.use_simnorm else mu_p
            obs_hat = self.predict_observation(h, mu_p)
            obs_innovation = obs[:, k] - obs_hat
            signals = self.fault_head(
                h,
                z,
                mu_q,
                log_std_q,
                mu_p,
                log_std_p,
                obs_innovation,
                a_prev,
                obs_prev=obs[:, k - 1] if k > 0 else obs[:, 0],
                obs=obs[:, k],
            )
            logits, context, surprise = signals.logits, signals.context, signals.surprise

            out["h"].append(h)
            out["z"].append(z)
            out["z_prior"].append(z_prior)
            out["mu_post"].append(mu_q)
            out["log_std_post"].append(log_std_q)
            out["mu_prior"].append(mu_p)
            out["log_std_prior"].append(log_std_p)
            out["fault_logits"].append(logits)
            out["context"].append(context)
            out["surprise"].append(surprise)
            out["kl"].append(signals.kl)
            out["innovation"].append(signals.innovation)
            out["residual_score"].append(signals.residual_score)
            out["fault_params"].append(signals.fault_params)
            out["residual_features"].append(signals.features)
            out["obs_hat"].append(obs_hat)
            out["obs_innovation"].append(obs_innovation)
            out["embedding"].append(e)
            out["reward"].append(self.reward(h, z, context, action[:, k]))
            q1, q2 = self.q_values(h, z, context, action[:, k])
            out["q1"].append(q1)
            out["q2"].append(q2)
            # Latent consistency: (h_t, z_t, a_t) must predict e_{t+1}, which
            # forces the posterior latent to carry observation content that the
            # action history alone cannot explain.
            if k + 1 < horizon:
                target = self.encode(obs[:, k + 1]).detach()
                sh = self.consistency(h, z, action[:, k])
                out["consistency"].append(F.cosine_similarity(sh, target, dim=-1))
            else:
                out["consistency"].append(torch.zeros(batch, device=device, dtype=dtype))

            z_prev, a_prev = z, action[:, k]

        return {key: torch.stack(value, dim=1) for key, value in out.items()}

    # ------------------------------------------------------------------ #
    # Imagination (planning / actor-critic)
    # ------------------------------------------------------------------ #
    def imagine(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
        deterministic: bool = True,
        need_q: bool = False,
        need_obs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Roll the prior forward under a batch of candidate action sequences.

        All inputs have shape ``[B, ..., dim]`` and ``actions`` is
        ``[B, ..., H, 3]``; the leading dimensions are flattened internally.
        """
        if actions.ndim != 4:
            raise ValueError("actions must be [B, K, H, 3]")
        batch, num_samples, horizon, _ = actions.shape
        def expand(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(1).expand(batch, num_samples, -1).reshape(batch * num_samples, -1)

        h = expand(h)
        z = expand(z)
        context = expand(context)

        rewards, qs, obs_pred = [], [], []
        for k in range(horizon):
            a = actions[:, :, k].reshape(batch * num_samples, -1)
            h = self.recurrence(h, z, a)
            mu_p, log_std_p = self.prior_params(h)
            z = self.latent_from_params(mu_p, log_std_p, sample=not deterministic)
            rewards.append(self.reward(h, z, context, a))
            if need_obs:
                obs_pred.append(self.predict_observation(h, mu_p))
            if need_q:
                q1, q2 = self.q_values(h, z, context, a)
                qs.append(torch.minimum(q1, q2))

        if need_obs:
            # obs_pred is a list of [B*K, obs_dim] -> [B, K, H, obs_dim]
            out_obs = torch.stack(obs_pred, dim=1).reshape(
                batch, num_samples, horizon, -1
            )
        out = {
            "reward": torch.stack(rewards, dim=-1).reshape(batch, num_samples, horizon),
            "h": h.reshape(batch, num_samples, -1),
            "z": z.reshape(batch, num_samples, -1),
            "context": context.reshape(batch, num_samples, -1),
        }
        if need_q:
            out["q"] = torch.stack(qs, dim=-1).reshape(batch, num_samples, horizon)
        if need_obs:
            out["obs"] = out_obs
        return out

    def prior_state(
        self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One prior step: new ``h`` and (mean) ``z``."""
        h_next = self.recurrence(h, z, action)
        mu, _ = self.prior_params(h_next)
        z_next = self.simnorm(mu) if self.use_simnorm else mu
        return h_next, z_next
