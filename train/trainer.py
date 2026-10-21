"""Joint world-model + TD-MPC2 training loop with curriculum and FDIR supervision."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from agents.pid_baseline import CascadePIDController
from agents.world_model_agent import WorldModelAgent
from configs.schema import (
    EnvironmentConfig,
    MPPIConfig,
    RocketPhysicsConfig,
    TrainingConfig,
    WorldModelConfig,
)
from envs.rocket_vec_env import VectorizedRocketEnv
from models.world_model import RocketWorldModel
from train.replay_buffer import PinnedSequenceReplayBuffer


class WorldModelTrainer:
    """Rollout -> replay -> joint latent/reward/Q/actor/FDIR optimisation."""

    def __init__(
        self,
        training: Optional[TrainingConfig] = None,
        world_model: Optional[WorldModelConfig] = None,
        mppi: Optional[MPPIConfig] = None,
        physics: Optional[RocketPhysicsConfig] = None,
        environment: Optional[EnvironmentConfig] = None,
        device: Optional[torch.device | str] = None,
        output_dir: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self.cfg = training or TrainingConfig()
        self.wm_cfg = world_model or WorldModelConfig()
        self.mppi_cfg = mppi or MPPIConfig()
        self.physics = physics or RocketPhysicsConfig()
        self.env_cfg = environment or EnvironmentConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.output_dir = Path(output_dir or self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        torch.manual_seed(seed)

        self.env = VectorizedRocketEnv(
            num_envs=self.cfg.num_envs,
            device=self.device,
            physics=self.physics,
            environment=self.env_cfg,
            seed=seed,
            fault_probability=self.cfg.warmup_fault_probability,
        )
        self.model = RocketWorldModel(self.wm_cfg).to(self.device)
        self.agent = WorldModelAgent(self.wm_cfg, self.model).to(self.device)
        self.buffer = PinnedSequenceReplayBuffer(
            capacity=self.cfg.replay_capacity,
            sequence_length=self.cfg.sequence_length,
            device=self.device,
            seed=seed,
            num_envs=self.cfg.num_envs,
        )
        self.buffer.reset_episode_tracking(self.cfg.num_envs, self.device)

        self.wm_optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + [p for p in self.agent.actor.parameters()],
            lr=self.wm_cfg.learning_rate,
            weight_decay=self.wm_cfg.weight_decay,
        )
        # The policy prior is refreshed against frozen world-model features.
        self.actor_optimizer = torch.optim.AdamW(
            self.agent.actor.parameters(),
            lr=self.wm_cfg.learning_rate,
            weight_decay=self.wm_cfg.weight_decay,
        )
        self._obs = self.env.reset()[0].to(self.device)
        self._total_steps = 0
        self._updates = 0
        self._demo_controller = CascadePIDController(
            physics=self.physics, environment=self.env_cfg, device=self.device
        )
        self._log_path = self.output_dir / "metrics.jsonl"

    # ------------------------------------------------------------------ #
    # Rollout
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def collect(self, steps: int, explore: bool = True) -> Dict[str, float]:
        rewards, landed, faults = [], [], []
        for _ in range(steps):
            h, z, c = self._encode_current()
            action, _, _ = self.agent.actor.sample(h, z, c, deterministic=not explore, obs=self._obs)
            if explore:
                action = (action + torch.randn_like(action) * 0.1).clamp(-1.0, 1.0)
            out = self.env.step(action)
            done = out.terminated | out.truncated
            self.buffer.add(
                self._obs,
                action,
                out.reward,
                done,
                out.info["fault_context"],
            )
            self._obs = out.obs.to(self.device)
            self._prev_action = action.detach()
            # Reset the recurrent carry-over of finished environments.
            if done.any():
                self._h = self._h.clone()
                self._z = self._z.clone()
                self._c = self._c.clone()
                self._h[done] = 0.0
                self._z[done] = 0.0
                self._c[done] = 0.0
                self._prev_action = self._prev_action.clone()
                self._prev_action[done] = 0.0
            self._total_steps += self.env.num_envs
            rewards.append(out.reward.mean().item())
            landed.append(out.info["landing_success"].float().mean().item())
            faults.append(out.info["fault_active"].float().mean().item())
        return {
            "rollout_reward": float(sum(rewards) / max(1, len(rewards))),
            "rollout_landing_rate": float(sum(landed) / max(1, len(landed))),
            "rollout_fault_rate": float(sum(faults) / max(1, len(faults))),
        }

    def _encode_current(self):
        """Posterior encode of the current observation (recurrent carry-over)."""
        if not hasattr(self, "_h") or self._h is None:
            self._h, self._z = self.model.initial_state(
                self._obs.shape[0], self.device, self._obs.dtype
            )
            self._c = torch.zeros(
                self._obs.shape[0], self.wm_cfg.fault_context_dim, device=self.device
            )
            self._prev_action = torch.zeros(
                self._obs.shape[0], self.wm_cfg.action_dim, device=self.device
            )
        self._h = self.model.recurrence(self._h, self._z, self._prev_action)
        e = self.model.encode(self._obs)
        mu_q, log_q = self.model.posterior_params(self._h, e)
        self._z = self.model.latent_from_params(mu_q, log_q, sample=False)
        mu_p, log_p = self.model.prior_params(self._h)
        obs_hat = self.model.predict_observation(self._h, mu_p)
        innovation = self._obs - obs_hat
        signals = self.model.fault_head(
            self._h, self._z, mu_q, log_q, mu_p, log_p, innovation, self._prev_action
        )
        self._c = signals.context
        self._surprise = signals.surprise
        return self._h, self._z, self._c

    # ------------------------------------------------------------------ #
    # Demonstration seeding
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _seed_with_controller(
        self,
        transitions: int,
        controller,
        oracle: bool,
        balanced: bool,
        expert: bool,
        fault_probability=None,
    ) -> int:
        collected = 0
        self.env.fixed_fault_mode = None
        self.env.balanced_fault_mix = balanced
        self.env.fault_probability = fault_probability
        while collected < transitions:
            self.env.reset()
            controller.reset(self.env.num_envs, self.device)
            self.buffer.reset_episode_tracking(self.env.num_envs, self.device)
            obs = self.env._observation()
            for _ in range(self.env_cfg.max_episode_steps):
                fault_state = self.env.fault if oracle else None
                action = controller.act(obs, fault_state=fault_state)
                out = self.env.step(action)
                n = self.env.num_envs
                self.buffer.add(
                    obs,
                    action,
                    out.reward,
                    out.terminated | out.truncated,
                    out.info["fault_context"],
                    expert=torch.full((n,), expert, dtype=torch.bool),
                )
                obs = out.obs
                collected += n
                if collected >= transitions:
                    break
        return collected

    @torch.no_grad()
    def dagger_seed(self, transitions: int) -> int:
        """DAgger: label the *actor's* state distribution with oracle actions."""
        collected = 0
        self.env.fixed_fault_mode = None
        self.env.balanced_fault_mix = True
        self.env.fault_probability = None
        self._oracle_controller = CascadePIDController(
            physics=self.physics,
            environment=self.env_cfg,
            oracle_fault_compensation=True,
            device=self.device,
        )
        while collected < transitions:
            self.env.reset()
            self.buffer.reset_episode_tracking(self.env.num_envs, self.device)
            self._reset_carry()
            obs = self.env._observation()
            for _ in range(self.env_cfg.max_episode_steps):
                h, z, c = self._encode_current()
                actor_action = self.agent.actor.mean_action(h, z, c, obs)
                expert_action = self._oracle_controller.act(obs, fault_state=self.env.fault)
                out = self.env.step(actor_action)
                n = self.env.num_envs
                self.buffer.add(
                    obs,
                    expert_action,
                    out.reward,
                    out.terminated | out.truncated,
                    out.info["fault_context"],
                    expert=torch.full((n,), True, dtype=torch.bool),
                )
                obs = out.obs
                self._obs = obs.to(self.device)
                done = out.terminated | out.truncated
                self._prev_action = actor_action.detach()
                if done.any():
                    self._h = self._h.clone()
                    self._z = self._z.clone()
                    self._c = self._c.clone()
                    self._h[done] = 0.0
                    self._z[done] = 0.0
                    self._c[done] = 0.0
                    self._prev_action = self._prev_action.clone()
                    self._prev_action[done] = 0.0
                collected += n
                if collected >= transitions:
                    break
        self._reset_carry()
        return collected

    @torch.no_grad()
    def seed_demonstrations(self) -> Dict[str, int]:
        """Nominal PID + oracle fault-tolerant PID + noisy exploration."""
        cfg = self.cfg
        demos: Dict[str, int] = {}
        demos["nominal_pid"] = self._seed_with_controller(
            cfg.nominal_demo_transitions,
            self._demo_controller,
            oracle=False,
            balanced=False,
            expert=True,
            fault_probability=0.0,
        )
        self._oracle_controller = CascadePIDController(
            physics=self.physics,
            environment=self.env_cfg,
            oracle_fault_compensation=True,
            device=self.device,
        )
        demos["oracle_fault"] = self._seed_with_controller(
            cfg.oracle_demo_transitions, self._oracle_controller, oracle=True, balanced=True, expert=True
        )
        self._explore_controller = CascadePIDController(physics=self.physics, environment=self.env_cfg, device=self.device)
        demos["exploration"] = self._seed_with_controller(
            cfg.explore_demo_transitions, self._explore_controller, oracle=False, balanced=True, expert=False
        )
        if cfg.dagger_transitions > 0:
            demos["dagger"] = self.dagger_seed(cfg.dagger_transitions)
        self.env.balanced_fault_mix = cfg.balanced_fault_sampling
        self._reset_carry()
        return demos

    def seed_with_pid(self, episodes: int = 32, max_transitions: int = 400_000) -> int:
        """Backward-compatible single-controller seeding helper."""
        return self._seed_with_controller(max_transitions, self._demo_controller, False, False, True)

    def _reset_carry(self) -> None:
        self._h = None
        self._z = None
        self._obs = self.env.reset()[0].to(self.device)

    # ------------------------------------------------------------------ #
    # Imitation warm start (iterative DAgger)
    # ------------------------------------------------------------------ #
    def pretrain_actor(
        self,
        rounds: int = 6,
        bc_updates_per_round: int = 300,
        dagger_transitions: int = 60_000,
        verbose: bool = True,
    ) -> list:
        """Behaviour-cloning warm start with iterative DAgger aggregation.

        Round 0 fits the actor to the PID/oracle demonstrations.  Each later
        round rolls out the *current* actor, relabels every visited state with
        the oracle action, and refits.  This removes the covariate shift that
        makes pure offline BC diverge in closed loop (the policy steers into
        states the demonstrations never visited, such as the engine spool-up
        transient at episode start).
        """
        history = []
        for rnd in range(rounds):
            for _ in range(bc_updates_per_round):
                batch = self.buffer.sample(self.wm_cfg.batch_size, expert_fraction=1.0)
                terms = self.loss_terms(batch)
                for p in self.model.parameters():
                    p.requires_grad_(False)
                bc = self.agent.bc_loss(
                    terms["h_flat"], terms["z_flat"], terms["c_flat"],
                    terms["a_expert"], terms["expert_mask"], obs=terms["obs_flat"],
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                bc.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.actor.parameters(), self.wm_cfg.grad_clip_norm
                )
                self.actor_optimizer.step()
                for p in self.model.parameters():
                    p.requires_grad_(True)
                self._updates += 1
            rate = self.evaluate_landing_rate(episodes=32, fixed_mode=0)
            history.append({"round": rnd, "bc_loss": float(bc.detach()), "landing_rate": rate})
            if verbose:
                print(
                    f"[pretrain] round {rnd}: bc={float(bc.detach()):.4f} "
                    f"landing={rate:.1%}",
                    flush=True,
                )
            if rnd + 1 < rounds:
                self.dagger_seed(dagger_transitions)
            if rate >= 0.97:
                break
        return history

    # ------------------------------------------------------------------ #
    # Losses / optimisation
    # ------------------------------------------------------------------ #
    def loss_terms(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cfg = self.wm_cfg
        obs = batch["obs"]
        actions = batch["action"]
        rewards = batch["reward"]
        dones = batch["done"].float()
        labels = batch["fault_context"].argmax(dim=-1)

        out = self.model.forward_sequence(obs, actions)
        horizon = obs.shape[1] - 1  # last state bootstraps the Q target

        mu_q = out["mu_post"][:, :horizon]
        ls_q = out["log_std_post"][:, :horizon]
        mu_p = out["mu_prior"][:, :horizon]
        ls_p = out["log_std_prior"][:, :horizon]

        kl_prior = self.model.fdir.surprise(mu_q.detach(), ls_q.detach(), mu_p, ls_p)
        kl_post = self.model.fdir.surprise(mu_q, ls_q, mu_p.detach(), ls_p.detach())
        free = cfg.free_bits
        dyn = (
            cfg.kl_alpha_prior * torch.clamp(kl_prior, min=free)
            + cfg.kl_alpha_post * torch.clamp(kl_post, min=free)
        ).mean()

        disc = torch.cumprod(
            torch.full((horizon,), cfg.lambda_returns, device=obs.device), dim=0
        )
        reward_loss = (
            F.smooth_l1_loss(out["reward"][:, :horizon], rewards[:, :horizon], reduction="none")
            * disc.reshape(1, -1)
        ).mean()

        h_next = out["h"][:, 1 : horizon + 1]
        z_next = out["z"][:, 1 : horizon + 1]
        c_next = out["context"][:, 1 : horizon + 1]
        with torch.no_grad():
            next_action = self.agent.actor.mean_action(
                h_next, z_next, c_next, obs[:, 1 : horizon + 1]
            )
            tq1, tq2 = self.model.target_values(h_next, z_next, c_next, next_action)
            next_value = torch.minimum(tq1, tq2)
            target = rewards[:, :horizon] + cfg.gamma * (1.0 - dones[:, :horizon]) * next_value

        q1 = out["q1"][:, :horizon]
        q2 = out["q2"][:, :horizon]
        q_loss = (
            F.smooth_l1_loss(q1, target, reduction="none") * disc.reshape(1, -1)
        ).mean() + (
            F.smooth_l1_loss(q2, target, reduction="none") * disc.reshape(1, -1)
        ).mean()

        # Class-balanced fault cross-entropy: faults are rare during the
        # warm-up curriculum (15%), so unweighted CE collapses to "NOMINAL".
        fault_logits = out["fault_logits"][:, :horizon].reshape(-1, self.wm_cfg.num_fault_classes)
        flat_labels = labels[:, :horizon].reshape(-1)
        counts = torch.bincount(flat_labels, minlength=self.wm_cfg.num_fault_classes).float()
        weights = torch.where(
            counts > 0,
            1.0 / counts.clamp_min(1.0),
            torch.zeros_like(counts),
        )
        weights = weights / weights.sum().clamp_min(1e-8) * self.wm_cfg.num_fault_classes
        fault_loss = F.cross_entropy(fault_logits, flat_labels, weight=weights)
        consistency = (1.0 - out["consistency"][:, :horizon]).mean()
        obs_loss = F.smooth_l1_loss(
            out["obs_hat"][:, :horizon], obs[:, :horizon], reduction="none"
        ).mean()

        # FDIR statistics are fitted on nominal transitions only.
        nominal = (labels[:, :horizon] == 0)
        with torch.no_grad():
            self.model.fdir.normalizer.update(out["obs_innovation"][:, :horizon], nominal)
            self.model.fdir.residuals.update_nominal(
                out["residual_features"][:, :horizon], nominal
            )
            self.model.fdir.detector.update(out["surprise"][:, :horizon].detach(), nominal)

        total = (
            cfg.kl_coef * dyn
            + cfg.consistency_coef * consistency
            + cfg.reward_coef * reward_loss
            + cfg.value_coef * q_loss
            + cfg.fault_coef * fault_loss
            + cfg.obs_pred_coef * obs_loss
        )
        return {
            "loss": total,
            "h_flat": out["h"][:, :horizon].detach().reshape(-1, self.wm_cfg.gru_dim),
            "z_flat": out["z"][:, :horizon].detach().reshape(-1, self.wm_cfg.latent_dim),
            "c_flat": out["context"][:, :horizon].detach().reshape(-1, self.wm_cfg.fault_context_dim),
            "a_expert": actions[:, :horizon].detach().reshape(-1, self.wm_cfg.action_dim),
            "obs_flat": obs[:, :horizon].detach().reshape(-1, self.wm_cfg.obs_dim),
            "expert_mask": batch["expert"][:, :horizon].detach().reshape(-1),
            "dyn": dyn,
            "consistency": consistency,
            "reward": reward_loss,
            "q": q_loss,
            "fault": fault_loss,
            "surprise": out["surprise"][:, :horizon].mean(),
            "kl": out["kl"][:, :horizon].mean(),
            "innovation": out["innovation"][:, :horizon].mean(),
            "obs_loss": obs_loss,
            "q_mean": q1.mean(),
        }

    def optimize(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cfg = self.wm_cfg
        self.model.train()
        terms = self.loss_terms(batch)
        self.wm_optimizer.zero_grad(set_to_none=True)
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.agent.actor.parameters()), cfg.grad_clip_norm
        )
        self.wm_optimizer.step()
        self.model.update_targets(cfg.ema_tau)

        # Policy prior: frozen world model.  A pure-imitation warm start
        # guarantees a landing-capable policy before max-Q fine-tuning (the
        # Q scale is O(100) while the BC MSE is O(0.01), so a single weighted
        # loss would otherwise ignore the demonstrations entirely).
        for p in self.model.parameters():
            p.requires_grad_(False)
        if self._updates < cfg.bc_only_updates:
            actor_terms = self.agent.bc_loss(
                terms["h_flat"],
                terms["z_flat"],
                terms["c_flat"],
                terms["a_expert"],
                terms["expert_mask"],
                obs=terms["obs_flat"],
            )
        else:
            actor_terms = self.agent.actor_loss(
                terms["h_flat"],
                terms["z_flat"],
                terms["c_flat"],
                expert_action=terms["a_expert"],
                expert_mask=terms["expert_mask"],
                bc_coef=cfg.bc_coef,
                obs=terms["obs_flat"],
            )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), cfg.grad_clip_norm)
        self.actor_optimizer.step()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self._updates += 1

        return {
            "loss": float(terms["loss"].detach()),
            "dyn": float(terms["dyn"].detach()),
            "consistency": float(terms["consistency"].detach()),
            "reward_loss": float(terms["reward"].detach()),
            "q_loss": float(terms["q"].detach()),
            "fault_loss": float(terms["fault"].detach()),
            "actor_loss": float(actor_terms.detach()),
            "surprise": float(terms["surprise"].detach()),
            "kl": float(terms["kl"].detach()),
            "innovation": float(terms["innovation"].detach()),
            "obs_loss": float(terms["obs_loss"].detach()),
            "q_mean": float(terms["q_mean"].detach()),
            "updates": self._updates,
        }

    def sample_batch(
        self, batch_size: Optional[int] = None, expert_fraction: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        if expert_fraction is None:
            expert_fraction = (
                1.0 if self._updates < self.wm_cfg.bc_only_updates else self.cfg.expert_fraction
            )
        return self.buffer.sample(batch_size or self.wm_cfg.batch_size, expert_fraction=expert_fraction)

    def update(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        return self.optimize(self.sample_batch(batch_size))

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train(self, total_steps: Optional[int] = None, max_updates: Optional[int] = None) -> Dict[str, float]:
        total = total_steps or self.cfg.total_env_steps
        steps_per_rollout = self.cfg.rollout_env_steps // self.cfg.num_envs
        last: Dict[str, float] = {}
        start = time.time()
        updates = 0
        best_rate = -1.0
        self.env.balanced_fault_mix = self.cfg.balanced_fault_sampling
        if self.cfg.balanced_fault_sampling:
            self.env.fault_probability = None
        while self._total_steps < total:
            stats = self.collect(steps_per_rollout)
            if len(self.buffer) >= self.cfg.sequence_length:
                for _ in range(self.cfg.gradient_updates_per_rollout):
                    last = self.update()
                    updates += 1
                    if updates % self.cfg.eval_interval_updates == 0:
                        rate = self.evaluate_landing_rate()
                        last["eval_landing_rate"] = rate
                        if rate > best_rate:
                            best_rate = rate
                            self.save(self.output_dir / "checkpoint_best.pt")
                        if max_updates is not None and updates >= max_updates:
                            break
                    if max_updates is not None and updates >= max_updates:
                        break
                last["updates"] = updates
            last.update(stats)
            last["env_steps"] = self._total_steps
            last["sps"] = self._total_steps / max(1e-6, time.time() - start)
            if updates % 100 == 0:
                self._log(last)
            if self._total_steps % self.cfg.checkpoint_interval < self.cfg.num_envs:
                self.save(self.output_dir / f"checkpoint_{self._total_steps}.pt")
            if max_updates is not None and updates >= max_updates:
                break
        self.save(self.output_dir / "checkpoint_final.pt")
        return last

    @torch.no_grad()
    def evaluate_landing_rate(self, episodes: int = 32, fixed_mode: Optional[int] = None) -> float:
        """Deterministic policy evaluation (balanced faults or one fixed mode)."""
        if not hasattr(self, "_eval_env"):
            self._eval_env = VectorizedRocketEnv(
                num_envs=16, device=self.device, seed=self.seed + 1234
            )
        env = self._eval_env
        env.fixed_fault_mode = fixed_mode
        env.balanced_fault_mix = fixed_mode is None
        env.fault_probability = None
        successes = 0
        total = 0
        while total < episodes:
            env.reset()
            h, z = self.model.initial_state(16, self.device, torch.float32)
            c = torch.zeros(16, self.wm_cfg.fault_context_dim, device=self.device)
            prev_action = torch.zeros(16, self.wm_cfg.action_dim, device=self.device)
            obs = env._observation()
            for _ in range(self.env_cfg.max_episode_steps):
                h = self.model.recurrence(h, z, prev_action)
                e = self.model.encode(obs)
                mu_q, log_q = self.model.posterior_params(h, e)
                z = self.model.latent_from_params(mu_q, log_q, sample=False)
                mu_p, log_p = self.model.prior_params(h)
                obs_hat = self.model.predict_observation(h, mu_p)
                signals = self.model.fault_head(
                    h, z, mu_q, log_q, mu_p, log_p, obs - obs_hat, prev_action
                )
                c = signals.context
                action = self.agent.actor.mean_action(h, z, c, obs)
                out = env.step(action)
                obs = out.obs
                prev_action = action
                done = out.terminated | out.truncated
                if done.any():
                    successes += int((out.info["landing_success"] & out.terminated).sum())
                    total += int(done.sum())
                    if done.any():
                        h[done] = 0.0
                        z[done] = 0.0
                        c[done] = 0.0
                        prev_action[done] = 0.0
                if total >= episodes:
                    break
        return successes / max(1, total)

    def _log(self, metrics: Dict[str, float]) -> None:
        with self._log_path.open("a") as fh:
            fh.write(json.dumps(metrics) + "\n")

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save(self, path: Path | str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "actor": self.agent.actor.state_dict(),
                "wm_optimizer": self.wm_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "env_steps": self._total_steps,
                "config": {
                    "world_model": self.wm_cfg.__dict__ if hasattr(self.wm_cfg, "__dict__") else {},
                },
            },
            path,
        )

    def load(self, path: Path | str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.agent.actor.load_state_dict(state["actor"])
        self.wm_optimizer.load_state_dict(state["wm_optimizer"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self._total_steps = int(state.get("env_steps", 0))
