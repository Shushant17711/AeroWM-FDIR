"""Budgeted end-to-end training for AeroWM-FDIR.

Three stages, all inside a ~2 minute wall-clock budget on an RTX 4060:

1. **Oracle demonstrations** (balanced 50 % fault mix) seed the sequence
   replay buffer.
2. **World model + FDIR** joint optimisation (KL, reward, Q, consistency,
   observation prediction, fault classification).  Sequences are unrolled with
   ``T_burnin = 8`` steps so the recurrent state is warm before the loss
   window starts.
3. **Residual policy** (``delta_u_wm``) trained by DAgger against the oracle on
   the *fault-conditioned base controller's* state distribution, with a
   normalised-Q max-Q term.  The actor's output layer is zero-initialised, so
   ``u_total == u_base`` until the residual data says otherwise.

Usage::

    PYTHONPATH=. python -m train.train_fast --updates 2500 --ppo-iterations 80
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from agents.pid_baseline import CascadePIDController
from agents.residual_control import FDIRResidualController, estimate_fault_state
from configs.schema import TrainingConfig, WorldModelConfig
from train.trainer import WorldModelTrainer

BURN_IN = 8


def build_trainer(seed: int = 0, output_dir: str = "runs/aerowm_fdir") -> WorldModelTrainer:
    training = TrainingConfig(
        num_envs=8,
        sequence_length=BURN_IN + 9,
        replay_capacity=400_000,
        expert_fraction=1.0,
        output_dir=output_dir,
        seed=seed,
    )
    wm_cfg = WorldModelConfig(batch_size=16, bc_only_updates=0)
    return WorldModelTrainer(training=training, world_model=wm_cfg, output_dir=output_dir, seed=seed)


def train_world_model(trainer: WorldModelTrainer, updates: int, verbose: bool = True) -> dict:
    last: dict = {}
    start = time.time()
    for i in range(updates):
        batch = trainer.sample_batch()
        terms = trainer.loss_terms(batch)
        trainer.wm_optimizer.zero_grad(set_to_none=True)
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.wm_cfg.grad_clip_norm)
        trainer.wm_optimizer.step()
        trainer.model.update_targets(trainer.wm_cfg.ema_tau)
        trainer._updates += 1
        last = {k: float(v.detach()) for k, v in terms.items() if torch.is_tensor(v) and v.numel() == 1}
        if verbose and (i + 1) % max(1, updates // 4) == 0:
            print(
                f"  [wm {i + 1:5d}] loss={last.get('loss', 0):.3f} kl={last.get('kl', 0):.2f} "
                f"fault={last.get('fault', 0):.3f} q_mean={last.get('q_mean', 0):.2f} "
                f"({time.time() - start:.0f}s)"
            )
    return last


@torch.no_grad()
def collect_residual_dataset(
    trainer: WorldModelTrainer,
    controller: FDIRResidualController,
    oracle: CascadePIDController,
    transitions: int = 6000,
) -> dict:
    """DAgger on the base controller's state distribution.

    For every visited state we record the latent the FDIR stack computed, the
    base action, the oracle action, and the ground-truth fault parameters (used
    only to supervise the parameter head head here -- never at deployment).
    """
    env = trainer.env
    env.fixed_fault_mode = None
    env.balanced_fault_mix = True
    env.fault_probability = None
    store = {
        "h": [],
        "z": [],
        "c": [],
        "obs": [],
        "target": [],
        "params": [],
        "features": [],
    }
    collected = 0
    scale = controller.residual_scale
    while collected < transitions:
        env.reset()
        controller.reset(env.num_envs, trainer.device)
        oracle.reset(env.num_envs, trainer.device)
        obs = env._observation()
        for _ in range(env.env_cfg.max_episode_steps):
            controller.encode(obs)
            estimate = estimate_fault_state(
                controller.logits,
                controller.fault_params,
                controller.frozen,
                residuals=controller.features,
                bias=controller.bias_estimate,
                obs=obs,
            )
            u_base = controller.base.act(obs, fault_state=estimate)
            u_oracle = oracle.act(obs, fault_state=env.fault)
            target = ((u_oracle - u_base) / scale).clamp(-1.0, 1.0)
            store["h"].append(controller.h.detach())
            store["z"].append(controller.z.detach())
            store["c"].append(controller.c.detach())
            store["obs"].append(obs.detach())
            store["target"].append(target.detach())
            store["params"].append(env.fault.applied_params().detach())
            store["features"].append(controller.head_inputs.detach())
            controller.prev_action = u_base.detach()
            out = env.step(u_base)
            obs = out.obs
            collected += env.num_envs
            done = out.terminated | out.truncated
            if done.any():
                controller.reset_mask(done)
            if collected >= transitions:
                break
    return {k: torch.cat(v, dim=0) for k, v in store.items()}


def train_residual_actor(
    trainer: WorldModelTrainer,
    data: dict,
    updates: int = 800,
    batch_size: int = 128,
    bc_coef: float = 4.0,
    verbose: bool = True,
) -> dict:
    model, agent = trainer.model, trainer.agent
    actor_opt = torch.optim.AdamW(agent.actor.parameters(), lr=1e-3, weight_decay=1e-4)
    param_opt = torch.optim.AdamW(model.fdir.param_head.parameters(), lr=1e-3)
    n = data["h"].shape[0]
    last = {}
    for i in range(updates):
        idx = torch.randint(0, n, (batch_size,), device=trainer.device)
        h, z, c, obs = data["h"][idx], data["z"][idx], data["c"][idx], data["obs"][idx]
        target, params = data["target"][idx], data["params"][idx]
        feats = data["features"][idx]

        # --- residual policy: BC + normalised-Q max-Q ------------------- #
        bc = agent.bc_loss(h, z, c, target, torch.ones_like(target[..., 0], dtype=torch.bool), obs)
        rl = agent.actor_loss(h, z, c, obs=obs)
        loss = bc_coef * bc + rl
        actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
        actor_opt.step()

        # --- continuous fault-parameter head -------------------------- #
        # Targets: gimbal bias (rad), thrust loss fraction, CoM offset (m).
        p_target = torch.stack(
            [
                params[:, 1],
                params[:, 2],
                1.0 - params[:, 0],
                params[:, 3],
                params[:, 4],
            ],
            dim=-1,
        )
        # Supervised on the *actual* head features captured during the DAgger
        # rollout (never on a synthetic zero vector).
        p_pred = model.fdir.param_head(feats)
        p_loss = F.mse_loss(p_pred, p_target)
        param_opt.zero_grad(set_to_none=True)
        p_loss.backward()
        param_opt.step()

        last = {"bc": float(bc.detach()), "rl": float(rl.detach()), "param": float(p_loss.detach())}
        if verbose and (i + 1) % max(1, updates // 3) == 0:
            print(
                f"  [actor {i + 1:4d}] bc={last['bc']:.4f} rl={last['rl']:.3f} "
                f"param_mse={last['param']:.4f}"
            )
    return last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=2500)
    parser.add_argument("--actor-updates", type=int, default=400)
    parser.add_argument("--demo-transitions", type=int, default=20_000)
    parser.add_argument("--dagger-transitions", type=int, default=6_000)
    parser.add_argument("--ppo-iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="runs/aerowm_fdir/checkpoint_final.pt")
    parser.add_argument("--skip-ppo", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    trainer = build_trainer(args.seed)
    oracle = CascadePIDController(
        physics=trainer.physics, environment=trainer.env_cfg,
        oracle_fault_compensation=True, device=trainer.device,
    )
    print(f"seeding {args.demo_transitions} oracle transitions ...")
    n = trainer._seed_with_controller(
        args.demo_transitions, oracle, oracle=True, balanced=True, expert=True
    )
    print(f"  seeded {n} transitions in {time.time() - t0:.0f}s")

    print(f"training world model + FDIR for {args.updates} updates ...")
    train_world_model(trainer, args.updates)

    scale = 0.15
    controller = FDIRResidualController(
        trainer.model, trainer.agent, device=trainer.device, residual_scale=scale, mode="actor"
    )
    print(f"collecting {args.dagger_transitions} DAgger transitions for the residual ...")
    data = collect_residual_dataset(trainer, controller, oracle, args.dagger_transitions)
    print(f"  residual dataset: {data['h'].shape[0]} samples")
    train_residual_actor(trainer, data, updates=max(1, args.actor_updates))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": trainer.model.state_dict(), "actor": trainer.agent.actor.state_dict()},
        out,
    )
    torch.save(
        {"model": trainer.model.state_dict(), "actor": trainer.agent.actor.state_dict()},
        out.parent / "checkpoint_best.pt",
    )
    print(f"saved {out} after {time.time() - t0:.0f}s")

    if args.ppo_iterations > 0 and not args.skip_ppo:
        from train.train_ppo import main as ppo_main  # noqa: F401  (documented entry point)
        import subprocess
        import sys

        cmd = [
            sys.executable, "-m", "train.train_ppo",
            "--iterations", str(args.ppo_iterations),
            "--output", "runs/ppo/ppo.pt",
        ]
        print("running PPO:", " ".join(cmd))
        subprocess.run(cmd, check=False)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
