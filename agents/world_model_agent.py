"""Fault-conditioned squashed-Gaussian actor and TD(lambda) targets."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from configs.schema import WorldModelConfig
from models.networks import NormMLP
from models.world_model import RocketWorldModel


class SquashedGaussianActor(nn.Module):
    """``pi(a | h, z, c)`` with tanh squashing and correct log-prob Jacobian."""

    def __init__(self, cfg: WorldModelConfig, residual_scale: float = 0.15) -> None:
        super().__init__()
        self.cfg = cfg
        # Residual controller: the policy emits ``residual_scale * tanh(mu)``
        # added to the fault-conditioned base law rather than a raw action.
        self.residual_scale = float(residual_scale)
        # The policy is observation-augmented: the RSSM latent is trained for
        # dynamics prediction and is not a sufficient statistic for control
        # (measured R^2 0.27 on the observation vs 0.86 for the GRU state), so
        # pi(a | h, z, c, o) keeps full state information at deployment.
        in_dim = cfg.gru_dim + cfg.latent_dim + cfg.fault_context_dim + cfg.obs_dim
        self.net = NormMLP(in_dim, cfg.hidden_dim, 2 * cfg.action_dim, num_layers=2)
        self.log_std_min = math.log(cfg.action_std_min)
        self.log_std_max = 2.0
        # Zero the output layer: an untrained actor reproduces ``u_base``
        # exactly, so Gate 4 never depends on an under-trained residual.
        last = None
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                last = module
        if last is not None:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs is None:
            obs = torch.zeros(h.shape[0], self.cfg.obs_dim, device=h.device, dtype=h.dtype)
        out = self.net(torch.cat([h, z, context, obs], dim=-1))
        mu, log_std = out.chunk(2, dim=-1)
        return mu, log_std.clamp(self.log_std_min, self.log_std_max)

    def distribution(
        self, h: torch.Tensor, z: torch.Tensor, context: torch.Tensor, obs: Optional[torch.Tensor] = None
    ) -> torch.distributions.Normal:
        mu, log_std = self(h, z, context, obs)
        return torch.distributions.Normal(mu, log_std.exp())

    def sample(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor,
        deterministic: bool = False,
        obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob, mean_action)``."""
        mu, _ = self(h, z, context, obs)
        if deterministic:
            action = torch.tanh(mu)
            return action, self.log_prob(h, z, context, action, obs), action
        dist = self.distribution(h, z, context, obs)
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = _squashed_log_prob(dist, u, action)
        return action, log_prob, torch.tanh(mu)

    def mean_action(
        self, h: torch.Tensor, z: torch.Tensor, context: torch.Tensor, obs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        mu, _ = self(h, z, context, obs)
        return torch.tanh(mu)

    def residual_mean(
        self, h: torch.Tensor, z: torch.Tensor, context: torch.Tensor, obs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Bounded residual in ``[-residual_scale, residual_scale]``."""
        return self.mean_action(h, z, context, obs) * self.residual_scale

    def log_prob(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dist = self.distribution(h, z, context, obs)
        eps = 1e-6
        a = action.clamp(-1.0 + eps, 1.0 - eps)
        u = torch.atanh(a)
        return _squashed_log_prob(dist, u, a)

    def entropy(
        self, h: torch.Tensor, z: torch.Tensor, context: torch.Tensor, obs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Monte-Carlo differential entropy estimate (per batch element)."""
        _, log_prob, _ = self.sample(h, z, context, obs=obs)
        return -log_prob


def _squashed_log_prob(dist: torch.distributions.Normal, u: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    log_prob = dist.log_prob(u).sum(dim=-1)
    log_det = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
    return log_prob - log_det


class WorldModelAgent(nn.Module):
    """Actor plus TD(lambda) utilities; critics live in the world model."""

    def __init__(self, cfg: WorldModelConfig, model: RocketWorldModel) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.actor = SquashedGaussianActor(cfg)

    @torch.no_grad()
    def act(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor,
        deterministic: bool = True,
        obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action, _, _ = self.actor.sample(h, z, context, deterministic=deterministic, obs=obs)
        return action.clamp(-1.0, 1.0)

    def residual_mean(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bounded residual for the fault-conditioned controller."""
        return self.actor.residual_mean(h, z, context, obs)

    def td_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        bootstrap: Optional[torch.Tensor] = None,
        gamma: Optional[float] = None,
        lam: Optional[float] = None,
    ) -> torch.Tensor:
        """TD(lambda) value targets over the sequence dimension.

        Args:
            rewards: ``[B, T]``
            values: ``[B, T]`` 1-step Q values (or value estimates)
            dones: ``[B, T]`` float/bool
            bootstrap: ``[B]`` value of the state after the last step
        """
        gamma = self.cfg.gamma if gamma is None else gamma
        lam = self.cfg.lambda_returns if lam is None else lam
        batch, horizon = rewards.shape
        returns = torch.zeros_like(rewards)
        gae = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
        bootstrap = bootstrap if bootstrap is not None else values[:, -1]
        for t in reversed(range(horizon)):
            next_value = bootstrap if t == horizon - 1 else values[:, t + 1]
            not_done = 1.0 - dones[:, t].to(rewards.dtype)
            delta = rewards[:, t] + gamma * not_done * next_value - values[:, t]
            gae = delta + gamma * lam * not_done * gae
            returns[:, t] = gae + values[:, t]
        return returns

    def bc_loss(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor,
        expert_action: torch.Tensor,
        expert_mask: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Regress the pre-tanh activation: with a tanh-squashed output the
        # gradient vanishes for saturated (|a| -> 1) targets, which is exactly
        # where the attitude loop needs full authority.
        mu, _ = self.actor(h, z, context, obs)
        target = torch.atanh(expert_action.clamp(-0.995, 0.995))
        err = (mu - target) ** 2
        mask = expert_mask.to(err.dtype)
        denom = mask.sum().clamp_min(1.0)
        var = (
            ((target - (target * mask.unsqueeze(-1)).sum(0) / denom) ** 2 * mask.unsqueeze(-1)).sum(0)
            / denom
        ).clamp_min(1e-2)
        err = (err / var).mean(dim=-1)
        return (err * mask).sum() / denom

    def actor_loss(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor,
        entropy_coef: Optional[float] = None,
        expert_action: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None,
        bc_coef: float = 0.0,
        obs: Optional[torch.Tensor] = None,
        normalize_q: bool = True,
    ) -> torch.Tensor:
        """Max-Q with entropy bonus plus optional behavioural cloning.

        ``bc_coef > 0`` anchors the policy to demonstration actions on the
        transitions flagged ``expert_mask`` (advantage-weighted by construction:
        only successful demonstration trajectories are labelled expert).
        """
        action, log_prob, _ = self.actor.sample(h, z, context, obs=obs)
        q1, q2 = self.model.q_values(h, z, context, action)
        q = torch.minimum(q1, q2)
        coef = self.cfg.entropy_coef if entropy_coef is None else entropy_coef
        # Q normalisation: the critic's absolute scale is arbitrary (and drifts
        # with the reward rescale), but only the *relative* ranking of actions
        # matters.  Normalising keeps the policy gradient O(1).
        if normalize_q and q.numel() > 1:
            q = (q - q.mean()) / (q.std() + 1e-5)
        loss = (-q + coef * log_prob).mean()
        if bc_coef > 0.0 and expert_action is not None and expert_mask is not None:
            pred = self.actor.mean_action(h, z, context, obs)
            err = ((pred - expert_action) ** 2).mean(dim=-1)
            mask = expert_mask.to(err.dtype)
            bc = (err * mask).sum() / mask.sum().clamp_min(1.0)
            loss = loss + bc_coef * bc
        return loss
