"""Print the four verification gates from a benchmark result file.

Usage::

    PYTHONPATH=. .venv/bin/python -m scripts.gate_report results/eval_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODES = ("NOMINAL", "LOE_THRUST", "GIMBAL_BIAS", "GIMBAL_STUCK", "COM_SHIFT")


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/eval_results.json")
    payload = json.loads(path.read_text())
    results = payload["results"]
    print(f"benchmark: {path}  ({payload['generated_at']}, seed {payload['seed']}, "
          f"{payload['episodes_per_mode']} episodes/mode, onset {payload['onset_step']}, "
          f"{payload.get('num_envs', '?')} envs)\n")

    header = f"{'controller':24s}" + "".join(f"{m:>14s}" for m in MODES)
    print(header)
    for name, per_mode in results.items():
        row = f"{name:24s}"
        for m in MODES:
            row += f"{per_mode[m]['success_rate'] * 100:13.1f}%"
        print(row)

    # Gate 1 -- throughput is measured separately (scripts/measure_sps.py).
    # Gate 2 -- classical reference: nominal >= 90 %, every fault <= 50 %
    #           (the blueprint's stricter bound is 30 %).
    pid = results.get("Cascaded-PID")
    if pid:
        nominal = pid["NOMINAL"]["success_rate"]
        faults = {m: pid[m]["success_rate"] for m in MODES[1:]}
        worst = max(faults.values())
        print(f"\nGate 2  nominal {nominal * 100:.1f} % (>= 90 %: "
              f"{'PASS' if nominal >= 0.90 else 'FAIL'}), "
              f"worst fault {worst * 100:.1f} % (<= 50 %: {'PASS' if worst <= 0.50 else 'FAIL'}, "
              f"<= 30 %: {'PASS' if worst <= 0.30 else 'FAIL'})")

    # Gate 3 -- detection latency and isolation for the FDIR controllers.
    for name, per_mode in results.items():
        if "AeroWM" not in name:
            continue
        lat = [per_mode[m]["detection_latency_steps"] for m in MODES[1:]]
        iso = [per_mode[m]["fault_classification_accuracy"] for m in MODES[1:]]
        rate = [per_mode[m]["detection_rate"] for m in MODES[1:]]
        mean_lat = sum(lat) / len(lat)
        mean_iso = sum(iso) / len(iso)
        mean_rate = sum(rate) / len(rate)
        print(
            f"Gate 3  {name:22s} detection {mean_rate * 100:.1f} %, latency {mean_lat:.2f} steps "
            f"({mean_lat * 20:.1f} ms) <= 3: {'PASS' if mean_lat <= 3.0 else 'FAIL'}, "
            f"isolation {mean_iso * 100:.1f} % >= 92: {'PASS' if mean_iso >= 0.92 else 'FAIL'}"
        )
        fa = per_mode["NOMINAL"].get("false_alarm_rate")
        if fa is not None:
            print(
                f"Gate 3  {name:22s} nominal false alarms {fa * 100:.1f} % "
                f"<= 5: {'PASS' if fa <= 0.05 else 'FAIL'}"
            )

    # Gate 4 -- FDIR stack lands >= 85 % in every mode.
    for name, per_mode in results.items():
        if "AeroWM" not in name:
            continue
        worst_mode = min(MODES, key=lambda m: per_mode[m]["success_rate"])
        worst = per_mode[worst_mode]["success_rate"]
        print(
            f"Gate 4  {name:22s} worst mode {worst_mode} {worst * 100:.1f} % "
            f">= 85 %: {'PASS' if worst >= 0.85 else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
