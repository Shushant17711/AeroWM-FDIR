# Changelog

All notable changes to this project are documented here.  The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-21

First public research release.  All four verification gates pass on the
published benchmark (`--episodes 20 --fast-mppi`, seed 0, 8 envs, onset 150).

### Added

* **Exact one-step physics residuals** (`models/fdir_detector.py`): actuator
  tracking (`res_gimbal`), thrust tracking (`res_thrust`), rigid-body rate
  residual (`res_omega`), a 5-D command-tracking vector, nominal step-to-step
  standard-deviation normalisation, a 5-step EMA, and a 2-sample confirmed
  frozen-servo indicator.
* **Fault detector and isolator** (`models/fdir_head.py`): anomaly score
  `S_t = 0.1·D_KL + innovation + residual + frozen + CoM`, EMA-calibrated
  alarm threshold, 5-way isolation head with a physics-informed prior and
  bounded logits, latched class decision, continuous fault-parameter head.
* **Fault-conditioned residual controller** (`agents/residual_control.py`):
  `u_total = clip(u_base(o_t, fdir_estimate) + Δu, -1, 1)` with the fault
  magnitudes read out physically (deconvolved bias, thrust ratio, averaged
  CoM offset), per-episode latching, and a supervisor mode that blends the
  base law, the world-model residual and the MPPI plan.
* **Fault-tolerant classical law** (`agents/pid_baseline.py`): Section 6.3
  terminal divert with an online `a_res` estimator (same-filter EMA of the
  measured/commanded lateral acceleration), ACS/RCS takeover on a frozen
  gimbal axis, exact force feed-forward with the corrected axis mapping, a
  saturation-aware tilt limiter, and a terminal flare.
* **Model-free PPO baseline** (`agents/ppo_baseline.py`,
  `train/train_ppo.py`): a bounded residual around the non-adaptive
  classical law, behaviour-cloned initialisation, checkpoint selection on
  held-out nominal flights.
* **Latent MPPI planner** (`agents/mppi_planner.py`): batched CEM/MPPI inside
  the learned prior, warm-started from the policy prior and anchored on the
  fault-conditioned base law.
* **Benchmark and artefacts** (`eval/benchmark_fdir.py`,
  `eval/plot_fdir_telemetry.py`): deterministic 5-mode × 4-controller
  benchmark writing `results/eval_results.json`, and a four-panel telemetry
  figure.
* **Gate scripts** (`scripts/`): `measure_sps.py` (throughput),
  `check_fdir.py` (latency/isolation on the deployed stack),
  `gate_report.py` (verdict table from the JSON),
  `stuck_freeze_angle.py` (freeze angle vs. landing outcome).
* **Documentation**: research-oriented `README.md`, `LIMITATIONS.md`,
  `CONTRIBUTING.md`, `CITATION.cff`, `CHANGELOG.md`,
  `docs/REPRODUCTION.md`.

### Changed

* Reward rescaled (dense penalties ÷ 50 plus a glide-slope bonus) so that
  Q-learning and PPO are numerically well posed.
* `Cascaded-PID` terminal flare level 0.9 → 0.8 (measured sweep in
  `results/TRAINING_REPORT.md` §9.3).
* MPPI plan warm-started from the policy prior; planner deviation switched
  off per environment once a fault is latched.
* CoM offset read out from the **averaged** rotational residual.

### Fixed

* Swapped-axis bug in the frozen-nozzle feed-forward (`Fx = T·sin(dz)`,
  `Fy = T·sin(dy)`); the nominal `GIMBAL_STUCK` recovery of the oracle
  went from 19 % to 92 %.
* Euler actuator step (`dt/τ = 0.571`) replaced by the exact first-order
  response (`1 − e^{−dt/τ} = 0.435`).
* `MPPI` re-encoded the observation and called the fault head twice per step,
  which delayed `LOE_THRUST` detection from 0.9 to 14.6 steps; the recurrent
  state is now shared with the controller.

### Known issues

See `LIMITATIONS.md`.  The most important one: the classical baseline lands
40 % of the `GIMBAL_STUCK` flights of the 20-episode benchmark sample
because eight of those episodes are *benign* freezes (the nozzle seizes
within 1° of the vertical); it lands 0/16 of the severe freezes and 0 % on
the 48-episode unit-test protocol.
