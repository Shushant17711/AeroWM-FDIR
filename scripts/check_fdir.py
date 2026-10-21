"""Gate 3 check: detection latency + 5-way isolation accuracy on the deployed stack.

Runs the *actual* FDIR pipeline (world model + residual detector + fault head +
class latch) on the fault modes and reports, per mode:

* detection rate  (fraction of episodes whose surprise crosses the calibrated
  alarm threshold within the first steps after the onset),
* detection latency in steps (and milliseconds),
* isolation accuracy (the latched fault class vs. the true mode).

Usage::

    PYTHONPATH=. .venv/bin/python -m scripts.check_fdir --episodes 20
"""

from __future__ import annotations

import argparse
import time

import torch

from agents.pid_baseline import CascadePIDController
from agents.residual_control import FDIRResidualController
from agents.world_model_agent import WorldModelAgent
from configs.schema import FaultMode, WorldModelConfig
from envs.rocket_vec_env import VectorizedRocketEnv
from eval.benchmark_fdir import evaluate_controller
from models.world_model import RocketWorldModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 3 detector check")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--onset", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="runs/aerowm_fdir/checkpoint_final.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = WorldModelConfig()
    model = RocketWorldModel(cfg).to(device).eval()
    agent = WorldModelAgent(cfg, model).to(device).eval()
    try:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        agent.actor.load_state_dict(state["actor"])
    except FileNotFoundError:
        print(f"! checkpoint {args.checkpoint} not found -- running an untrained model")

    env = VectorizedRocketEnv(num_envs=args.envs, device=device, seed=args.seed)
    controller = FDIRResidualController(
        model,
        agent,
        device=device,
        base=CascadePIDController(device=device, fault_estimation_mode=True),
        mode="actor",
    )

    start = time.time()
    latencies, isolations, rates = [], [], []
    print(f"{'mode':14s} {'detected':>9s} {'latency[steps]':>14s} {'latency[ms]':>12s} {'isolation':>10s}")
    for mode in range(1, 5):
        metrics = evaluate_controller(
            env, controller, args.episodes, mode, onset=args.onset, is_world_model=True
        )
        name = FaultMode(mode).name
        lat = metrics["detection_latency_steps"]
        iso = metrics["fault_classification_accuracy"]
        rate = metrics["detection_rate"]
        latencies.append(lat)
        isolations.append(iso)
        rates.append(rate)
        print(
            f"{name:14s} {rate * 100:8.1f}% {lat:14.2f} {lat * 20.0:12.1f} {iso * 100:9.1f}%"
        )

    mean_lat = sum(latencies) / len(latencies)
    mean_iso = sum(isolations) / len(isolations)
    mean_rate = sum(rates) / len(rates)
    print(
        f"\nmean: detection {mean_rate * 100:.1f} %, latency {mean_lat:.2f} steps "
        f"({mean_lat * 20.0:.1f} ms), isolation {mean_iso * 100:.1f}%  [{time.time() - start:.0f} s]"
    )
    ok_lat = mean_lat <= 3.0
    ok_iso = mean_iso >= 0.92
    print(f"Gate 3 latency  <= 3 steps : {'PASS' if ok_lat else 'FAIL'}")
    print(f"Gate 3 isolation >= 92 %   : {'PASS' if ok_iso else 'FAIL'}")


if __name__ == "__main__":
    main()
