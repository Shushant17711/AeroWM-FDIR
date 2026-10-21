# Changelog

All notable changes to this project are documented here.  The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-21

Revision 7: the three open items of revision 6 are closed, and the two
measurement caveats (the benign freeze and the nominal false alarms) are
removed from the fault model and from the detector.

### Added

* **Minimum seizure angle** (`envs/fault_injector.py`,
  `FaultInjectionSchema.stuck_min_deflection_deg`): a `GIMBAL_STUCK` fault
  locks the seized axis at at least `radians(1.25) * (severity / 0.5)` from
  the vertical, so the fault model no longer contains "faults" that do not
  degrade the flight at all.
* **Seizure lock** (`envs/rocket_vec_env.py`): the environment applies the
  lock at the onset step; the plant keeps its freeze-in-place semantics.
* **Per-episode randomness** (`envs/counter_rng.py`): a counter-based hash
  (SplitMix64) makes the episode scenario and the observation noise a pure
  function of `(seed, environment index, episode counter)`.
* **`false_alarm_rate`** in the benchmark output: the fraction of episodes
  that raise an alarm outside the detection window (before the onset for a
  faulted episode, anywhere for a nominal one).
* **Greedy trust-region acceptance** for the MPPI variant
  (`agents/residual_control.py`): the planner stays active under a latched
  fault inside a 0.5 % trust region, and the plan is only applied where the
  world model's own value prefers it over the policy prior.

### Changed

* The anomaly score now scores only the six *physics* residual channels; the
  command-tracking channels are large during any fast transient (the engine
  spooling down in the terminal phase reaches 6 sigma) and are not a fault
  signature.  They are still part of the isolation head's input.
* The alarm requires *physical* evidence (a residual, a frozen servo or a CoM
  offset) and two consecutive samples above the threshold, and it is
  suppressed during the first samples of an episode and across an
  auto-reset.
* The frozen-servo test is confirmed over four consecutive bit-stationary
  samples, which separates a seized axis from a biased one without a
  mis-isolation.
* The isolation evidence is used instantaneously instead of a 250-step
  moving average: the class is now decided 7 samples after the onset instead
  of 65-90.

### Fixed

* A `GIMBAL_BIAS` episode could be isolated as `GIMBAL_STUCK` (the frozen
  test also fired on a converged, biased servo).
* A `GIMBAL_STUCK` episode could be latched as `COM_SHIFT` (the CoM
  attribution used the smoothed residuals, which lag the fault onset); the
  class is now latched from the instantaneous actuator residuals.
* The nominal false alarms: the episode-boundary jump and the terminal-phase
  transients no longer raise an alarm (measured false-alarm rate: 0 %).

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
