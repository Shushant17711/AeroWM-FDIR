"""CleanRL-style vectorized PPO baseline (model-free comparison)."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import ControllerMixin
from agents.pid_baseline import CascadePIDController
from configs.schema import ActionSchema, ObservationSchema, PPOConfig, RocketPhysicsConfig
from models.networks import NormMLP


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.actor = NormMLP(obs_dim, hidden, 2 * action_dim, num_layers=2)
        self.critic = NormMLP(obs_dim, hidden, 1, num_layers=2)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor):
        mu, _ = self.actor(obs).chunk(2, dim=-1)
        value = self.critic(obs).squeeze(-1)
        return mu, self.log_std.clamp(math.log(0.02), 0.5).expand_as(mu), value

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mu, log_std, _ = self(obs)
        return torch.distributions.Normal(mu, log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


def squashed_log_prob(dist, u: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return dist.log_prob(u).sum(-1) - torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)


class PPOController(ControllerMixin):
    """Model-free PPO agent satisfying the shared controller interface.

    The policy is a *residual* around the non-adaptive classical stabiliser:

    ``u_total = clip(u_pid(o_t) + residual_scale * tanh(a_ppo), -1, 1)``

    It receives no fault information, no detector signals and no world model,
    so it isolates the value of the learned policy itself: the residual head
    can trim the nominal trajectory, but it has no mechanism to identify a
    mid-flight actuator fault.
    """

    action_schema = ActionSchema()

    def __init__(
        self,
        physics: Optional[RocketPhysicsConfig] = None,
        observation: Optional[ObservationSchema] = None,
        cfg: Optional[PPOConfig] = None,
        device: torch.device | str = "cpu",
        seed: int = 0,
        residual_scale: float = 0.15,
        base_controller: Optional[CascadePIDController] = None,
    ) -> None:
        self.physics = physics or RocketPhysicsConfig()
        self.observation = observation or ObservationSchema()
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device)
        self.net = ActorCritic(self.observation.dim, self.action_schema.dim, self.cfg.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self._batch = 1
        # Non-adaptive reference law.  Deliberately *not* FDIR-conditioned:
        # the model-free baseline must not see fault estimates.
        self.residual_scale = float(residual_scale)
        self.base = base_controller or CascadePIDController(
            physics=self.physics,
            observation=self.observation,
            device=self.device,
            fault_estimation_mode=False,
            oracle_fault_compensation=False,
        )

    @property
    def action_dim(self) -> int:
        return self.action_schema.dim

    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        self._batch = batch_size
        if device is not None and torch.device(device) != self.device:
            self.device = torch.device(device)
            self.net.to(self.device)
        self.base.reset(batch_size, self.device)

    def _base_action(self, obs: torch.Tensor) -> torch.Tensor:
        if self._batch != obs.shape[0]:
            self.base.reset(obs.shape[0], self.device)
            self._batch = obs.shape[0]
        return self.base.act(obs)

    def _compose(self, obs: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        u_base = self._base_action(obs)
        return (u_base + self.residual_scale * residual).clamp(-1.0, 1.0)

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        fault_context: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        obs = self._check_obs(obs).to(self.device)
        mu, log_std, _ = self.net(obs)
        if deterministic:
            residual = torch.tanh(mu)
        else:
            eps = torch.randn(mu.shape, device=self.device, generator=generator)
            residual = torch.tanh(mu + log_std.exp() * eps)
        return self._clip_action(self._compose(obs, residual))

    @torch.no_grad()
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.net.distribution(obs)
        u = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
        log_prob = squashed_log_prob(dist, u, actions)
        entropy = dist.entropy().sum(-1)
        value = self.net.value(obs)
        return log_prob, entropy, value

    # ------------------------------------------------------------------ #
    # Rollout collection
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def collect(self, env, steps: int, reset: bool = True) -> Dict[str, torch.Tensor]:
        """Collect ``steps`` transitions from a vectorized environment.

        ``reset=False`` continues from the environment's current state, which is
        what a long training run wants (partial episodes are stitched across
        rollouts by the ``done`` flags).
        """
        if reset:
            obs, _ = env.reset()
        else:
            obs = env._observation()
        obs = obs.to(self.device)
        self.reset(obs.shape[0], self.device)
        buf = {k: [] for k in ("obs", "action", "log_prob", "reward", "done", "value")}
        for _ in range(steps):
            dist = self.net.distribution(obs)
            u = dist.sample()
            action = torch.tanh(u)
            log_prob = squashed_log_prob(dist, u, action)
            value = self.net.value(obs)
            # The stored action is the *residual* (the policy's variable); the
            # plant receives the bounded deviation around the classical law.
            out = env.step(self._compose(obs, action))
            buf["obs"].append(obs)
            buf["action"].append(action)
            buf["log_prob"].append(log_prob)
            buf["reward"].append(out.reward.to(self.device))
            buf["done"].append((out.terminated | out.truncated).float().to(self.device))
            buf["value"].append(value)
            obs = out.obs.to(self.device)
        data = {k: torch.stack(v, dim=1) for k, v in buf.items()}
        data["bootstrap"] = self.net.value(obs)
        return data

    # ------------------------------------------------------------------ #
    # PPO update
    # ------------------------------------------------------------------ #
    def update(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        rewards = data["reward"]
        dones = data["done"]
        values = data["value"]
        batch, horizon = rewards.shape

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(batch, device=self.device)
        for t in reversed(range(horizon)):
            next_value = data["bootstrap"] if t == horizon - 1 else values[:, t + 1]
            not_done = 1.0 - dones[:, t]
            delta = rewards[:, t] + cfg.gamma * not_done * next_value - values[:, t]
            gae = delta + cfg.gamma * cfg.gae_lambda * not_done * gae
            advantages[:, t] = gae
        returns = advantages + values

        obs = data["obs"].reshape(batch * horizon, -1)
        actions = data["action"].reshape(batch * horizon, -1)
        old_logp = data["log_prob"].reshape(-1)
        adv = advantages.reshape(-1)
        ret = returns.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        minibatch = max(1, n // cfg.minibatches)
        indices = torch.randperm(n, device=self.device)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}
        for _ in range(cfg.epochs):
            perm = indices[torch.randperm(n, device=self.device)]
            for start in range(0, n, minibatch):
                mb = perm[start : start + minibatch]
                dist = self.net.distribution(obs[mb])
                u = torch.atanh(actions[mb].clamp(-1 + 1e-6, 1 - 1e-6))
                logp = squashed_log_prob(dist, u, actions[mb])
                entropy = dist.entropy().sum(-1).mean()
                value = self.net.value(obs[mb])

                ratio = (logp - old_logp[mb]).exp()
                surr1 = ratio * adv[mb]
                surr2 = ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, ret[mb])
                mu_mb, _ = self.net.actor(obs[mb]).chunk(2, dim=-1)
                mean_reg = cfg.residual_l2 * mu_mb.pow(2).mean()
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy
                    + mean_reg
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                stats["policy_loss"] += policy_loss.detach().item()
                stats["value_loss"] += value_loss.detach().item()
                stats["entropy"] += entropy.detach().item()
                stats["updates"] += 1
        n_updates = max(1, stats.pop("updates"))
        return {k: v / n_updates for k, v in stats.items()}
