"""Deterministic FDIR benchmark across fault modes and controllers.

Usage::

    python -m eval.benchmark_fdir --episodes 10 --output eval_results.json

The benchmark evaluates the cascaded PID controller, the model-free PPO
baseline and the AeroWM-FDIR stack (world model + KL surprise + latent MPPI)
on identical scenarios.  For the world-model controller it additionally
reports KL-detection latency (in steps and milliseconds) and fault-isolation
accuracy.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from agents.mppi_planner import LatentMPPIPlanner
from agents.pid_baseline import CascadePIDController
from agents.residual_control import FDIRResidualController
from agents.ppo_baseline import PPOController
from agents.world_model_agent import WorldModelAgent
from configs.schema import (
    ActionSchema,
    FaultMode,
    MPPIConfig,
    PPOConfig,
    WorldModelConfig,
)
from envs.rocket_vec_env import VectorizedRocketEnv
from envs.rocket_dynamics import tilt_angle_deg
from models.world_model import RocketWorldModel


@dataclass
class EpisodeResult:
    mode: str
    success: bool
    terminated: bool
    truncated: bool
    steps: int
    final_vz: float
    final_vlat: float
    final_lateral: float
    final_tilt: float
    propellant_margin: float
    detection_latency_steps: Optional[int] = None
    detection_latency_ms: Optional[float] = None
    predicted_class: Optional[int] = None
    true_class: int = 0
    surprise_peak: Optional[float] = None


def build_controllers(
    device: torch.device,
    wm_checkpoint: Optional[str] = None,
    ppo_checkpoint: Optional[str] = None,
    mppi: Optional[MPPIConfig] = None,
    world_model_cfg: Optional[WorldModelConfig] = None,
    use_mppi: bool = True,
) -> Dict[str, object]:
    controllers: Dict[str, object] = {
        "Cascaded-PID": CascadePIDController(device=device),
        "PPO (model-free)": PPOController(cfg=PPOConfig(), device=device),
    }
    if ppo_checkpoint and Path(ppo_checkpoint).exists():
        state = torch.load(ppo_checkpoint, map_location=device)
        controllers["PPO (model-free)"].net.load_state_dict(state["net"] if "net" in state else state)

    cfg = world_model_cfg or WorldModelConfig()
    model = RocketWorldModel(cfg).to(device).eval()
    agent = WorldModelAgent(cfg, model).to(device).eval()
    if wm_checkpoint and Path(wm_checkpoint).exists():
        state = torch.load(wm_checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        agent.actor.load_state_dict(state["actor"])

    # Fault-conditioned residual controllers: u_total = clip(u_base + du, -1, 1)
    # where u_base is the FDIR-conditioned feedback law (estimated fault only).
    controllers["AeroWM-FDIR (actor)"] = FDIRResidualController(
        model,
        agent,
        device=device,
        base=CascadePIDController(device=device, fault_estimation_mode=True),
        mode="actor",
    )
    if use_mppi:
        planner = LatentMPPIPlanner(model, agent, mppi or MPPIConfig(), device=device, autocast=False)
        controllers["AeroWM-FDIR (MPPI)"] = FDIRResidualController(
            model,
            agent,
            device=device,
            base=CascadePIDController(device=device, fault_estimation_mode=True),
            mode="mppi",
            planner=planner,
        )
    return controllers


class _ActorController:
    """Fast AeroWM-FDIR controller: FDIR context + policy prior (no planning)."""

    action_dim = 3
    action_schema = ActionSchema()

    def __init__(self, model: RocketWorldModel, agent: WorldModelAgent, device: torch.device) -> None:
        self.model = model
        self.agent = agent
        self.device = device
        self.reset(1, device)
        self.last_debug = {}

    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        if device is not None:
            self.device = torch.device(device)
        cfg = self.model.cfg
        self.h = torch.zeros(batch_size, cfg.gru_dim, device=self.device)
        self.z = torch.zeros(batch_size, cfg.latent_dim, device=self.device)
        self.c = torch.zeros(batch_size, cfg.fault_context_dim, device=self.device)
        self.prev_action = torch.zeros(batch_size, cfg.action_dim, device=self.device)
        self.surprise = torch.zeros(batch_size, device=self.device)
        self.prev_obs: Optional[torch.Tensor] = None
        if hasattr(self.model, 'fdir'):
            self.model.fdir.reset_state()

    def reset_mask(self, mask: torch.Tensor) -> None:
        mask = mask.to(self.device, torch.bool)
        if not bool(mask.any()):
            return
        self.h[mask] = 0.0
        self.z[mask] = 0.0
        self.c[mask] = 0.0
        self.prev_action[mask] = 0.0
        if self.prev_obs is not None:
            self.prev_obs = self.prev_obs.clone()
            self.prev_obs[mask] = 0.0

    @torch.no_grad()
    def act(self, obs: torch.Tensor, fault_context=None) -> torch.Tensor:
        self.h = self.model.recurrence(self.h, self.z, self.prev_action)
        e = self.model.encode(obs.to(self.device))
        mu_q, log_q = self.model.posterior_params(self.h, e)
        self.z = self.model.latent_from_params(mu_q, log_q, sample=False)
        mu_p, log_p = self.model.prior_params(self.h)
        obs_hat = self.model.predict_observation(self.h, mu_p)
        obs_now = obs.to(self.device)
        innovation = obs_now - obs_hat
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
            obs=obs_now,
        )
        self.prev_obs = obs_now.detach()
        logits, self.c, self.surprise = signals.logits, signals.context, signals.surprise
        self.last_debug["fault_logits"] = logits
        self.last_debug["surprise"] = self.surprise
        action = self.agent.actor.mean_action(self.h, self.z, self.c, obs.to(self.device))
        self.prev_action = action
        return action.clamp(-1.0, 1.0)


@torch.no_grad()
def evaluate_controller(
    env: VectorizedRocketEnv,
    controller,
    episodes: int,
    mode: int,
    onset: int = 150,
    max_batches: int = 100,
    is_world_model: bool = False,
) -> Dict[str, object]:
    device = env.device
    batch = env.num_envs
    results: List[EpisodeResult] = []
    surprises = torch.zeros(env.env_cfg.max_episode_steps, batch, device=device)
    threshold: Optional[float] = None

    # Calibrate the surprise detector on the first nominal batch.
    env.fixed_fault_mode = mode
    if is_world_model:
        # Calibrate on nominal flight, skipping the initial transient of each
        # episode (h=0 makes the first predictions atypically uncertain).
        env.fixed_fault_mode = 0
        nominal = []
        for _ in range(3):
            env.reset(mode=0, probability=0.0)
            controller.reset(batch, device)
            obs = env._observation()
            for t in range(80):
                out = env.step(controller.act(obs))
                obs = out.obs
                if t >= 20 and hasattr(controller, "surprise"):
                    nominal.append(controller.surprise.detach())
                if (out.terminated | out.truncated).any():
                    break
        if nominal:
            controller.model.fdir.detector.calibrate(torch.cat(nominal))
            threshold = float(controller.model.fdir.detector.threshold)
        env.fixed_fault_mode = mode

    attempts = 0
    while len(results) < episodes and attempts < max_batches:
        attempts += 1
        env.reset(mode=mode, probability=1.0 if mode else 0.0, onset=onset, severity=0.35 if mode == 1 else 0.5)
        controller.reset(batch, device)
        obs = env._observation()
        surprises.zero_()
        for t in range(env.env_cfg.max_episode_steps):
            action = controller.act(obs)
            out = env.step(action)
            obs = out.obs
            if is_world_model and hasattr(controller, "surprise"):
                surprises[t] = controller.surprise.detach()
            done = out.terminated | out.truncated
            if done.any():
                s = out.info["state"]
                for i in torch.nonzero(done).flatten().tolist():
                    latency = None
                    peak = None
                    if is_world_model and threshold is not None:
                        # The injector activates the fault on the ``onset``-th
                        # call to step, i.e. at loop index ``onset - 1``.
                        start = max(0, onset - 1)
                        window = surprises[start : t + 1, i]
                        peak = float(window.max()) if window.numel() else None
                        hits = (window > threshold).nonzero()
                        latency = int(hits[0]) if hits.numel() else None
                    logits = controller.last_debug.get("fault_logits") if is_world_model else None
                    predicted = int(logits[i].argmax()) if logits is not None else None
                    results.append(
                        EpisodeResult(
                            mode=FaultMode(mode).name,
                            success=bool(out.info["landing_success"][i]),
                            terminated=bool(out.terminated[i]),
                            truncated=bool(out.truncated[i]),
                            steps=t + 1,
                            final_vz=float(s[i, 5]),
                            final_vlat=float(torch.linalg.vector_norm(s[i, 3:5])),
                            final_lateral=float(torch.linalg.vector_norm(s[i, 0:2])),
                            final_tilt=float(tilt_angle_deg(s)[i]),
                            propellant_margin=float((s[i, 13] - env.physics.dry_mass) / env.physics.propellant_mass),
                            detection_latency_steps=latency,
                            detection_latency_ms=None if latency is None else latency * env.physics.dt * 1000.0,
                            predicted_class=predicted,
                            true_class=int(mode),
                            surprise_peak=peak,
                        )
                    )
                # Keep recurrent state consistent with auto-resets.
                if hasattr(controller, "reset_mask"):
                    controller.reset_mask(done)
            if len(results) >= episodes:
                break
    results = results[:episodes]
    return summarize(results, env.physics.dt, mode)


def summarize(results: List[EpisodeResult], dt: float, mode: int) -> Dict[str, object]:
    n = max(1, len(results))
    success = sum(r.success for r in results)
    latencies = [r.detection_latency_steps for r in results if r.detection_latency_steps is not None]
    classes = [r.predicted_class for r in results if r.predicted_class is not None]
    accuracy = (
        sum(int(c == mode) for c in classes) / len(classes) if classes and mode != 0 else None
    )
    return {
        "episodes": len(results),
        "success_rate": success / n,
        "successes": success,
        "crashes": sum((r.terminated and not r.success) for r in results),
        "timeouts": sum(r.truncated for r in results),
        "mean_final_vz": sum(r.final_vz for r in results) / n,
        "mean_final_vlat": sum(r.final_vlat for r in results) / n,
        "mean_final_lateral": sum(r.final_lateral for r in results) / n,
        "mean_final_tilt": sum(r.final_tilt for r in results) / n,
        "mean_propellant_margin": sum(r.propellant_margin for r in results) / n,
        "detection_latency_steps": (
            None if mode == 0 or not latencies else sum(latencies) / len(latencies)
        ),
        "detection_latency_ms": (
            None if mode == 0 or not latencies else sum(latencies) / len(latencies) * dt * 1000.0
        ),
        "detection_rate": None if mode == 0 else len(latencies) / n,
        "fault_classification_accuracy": accuracy,
        "mean_surprise_peak": (
            None
            if not results or results[0].surprise_peak is None
            else sum(r.surprise_peak for r in results if r.surprise_peak is not None) / n
        ),
        "episode_details": [asdict(r) for r in results],
    }


def run_benchmark(
    episodes_per_mode: int = 10,
    device: Optional[torch.device | str] = None,
    num_envs: int = 8,
    seed: int = 0,
    onset: int = 150,
    wm_checkpoint: Optional[str] = None,
    ppo_checkpoint: Optional[str] = None,
    mppi: Optional[MPPIConfig] = None,
    world_model_cfg: Optional[WorldModelConfig] = None,
    use_mppi: bool = True,
    output: Optional[str] = None,
    verbose: bool = True,
    controller_filter: Optional[str] = None,
) -> Dict[str, object]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    env = VectorizedRocketEnv(num_envs=num_envs, device=device, seed=seed)
    controllers = build_controllers(device, wm_checkpoint, ppo_checkpoint, mppi, world_model_cfg, use_mppi)
    if controller_filter:
        wanted = [c.strip().lower() for c in controller_filter.split(",") if c.strip()]
        controllers = {
            name: ctrl
            for name, ctrl in controllers.items()
            if any(w in name.lower() for w in wanted)
        }
        if not controllers:
            raise SystemExit(f"no controller matches {controller_filter!r}")
    results: Dict[str, Dict[str, object]] = {}
    start = time.time()
    for name, controller in controllers.items():
        results[name] = {}
        for mode in range(5):
            is_wm = "AeroWM" in name
            metrics = evaluate_controller(
                env, controller, episodes_per_mode, mode, onset=onset, is_world_model=is_wm
            )
            results[name][FaultMode(mode).name] = metrics
            if verbose:
                print(
                    f"{name:22s} {FaultMode(mode).name:14s} "
                    f"success={metrics['success_rate'] * 100:5.1f}% "
                    f"latency={metrics['detection_latency_steps']} steps"
                )

    summary = []
    for name, per_mode in results.items():
        rates = [per_mode[FaultMode(m).name]["success_rate"] for m in range(5)]
        summary.append(
            {
                "controller": name,
                "mean_success_rate": sum(rates) / len(rates),
                "success_by_mode": {FaultMode(m).name: rates[m] for m in range(5)},
            }
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "seed": seed,
        "episodes_per_mode": episodes_per_mode,
        "onset_step": onset,
        "dt_ms": env.physics.dt * 1000.0,
        "num_envs": num_envs,
        "wm_checkpoint": wm_checkpoint,
        "ppo_checkpoint": ppo_checkpoint,
        "controllers": list(controllers.keys()),
        "results": results,
        "summary": summary,
        "wall_clock_s": time.time() - start,
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        if verbose:
            print(f"wrote {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="AeroWM-FDIR benchmark")
    parser.add_argument("--episodes", type=int, default=10, help="episodes per fault mode")
    parser.add_argument("--envs", type=int, default=8, help="parallel environments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--onset", type=int, default=150)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--wm-checkpoint", type=str, default="runs/aerowm_fdir/checkpoint_final.pt")
    parser.add_argument("--ppo-checkpoint", type=str, default="runs/ppo/ppo.pt")
    parser.add_argument("--no-mppi", action="store_true")
    parser.add_argument("--fast-mppi", action="store_true", help="small planner for quick smoke runs")
    parser.add_argument(
        "--controllers",
        type=str,
        default=None,
        help="comma-separated case-insensitive name filter (e.g. 'pid')",
    )
    args = parser.parse_args()
    mppi = (
        MPPIConfig(num_samples=64, num_policy_samples=8, horizon=5, iterations=2, num_elites=8)
        if args.fast_mppi
        else MPPIConfig()
    )
    run_benchmark(
        episodes_per_mode=args.episodes,
        num_envs=args.envs,
        seed=args.seed,
        onset=args.onset,
        wm_checkpoint=args.wm_checkpoint,
        ppo_checkpoint=args.ppo_checkpoint,
        mppi=mppi,
        use_mppi=not args.no_mppi,
        output=args.output,
        controller_filter=args.controllers,
    )


if __name__ == "__main__":
    main()
