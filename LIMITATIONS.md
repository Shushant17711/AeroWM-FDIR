# Limitations

This repository is **research software for a simulated vehicle**.  It is
published so that the measurements can be inspected and reproduced, not as a
flight-ready product.  The list below is deliberately explicit about what the
results do and do not establish.

---

## 1. Simulation, not flight

* Everything runs inside a 6-DOF rigid-body simulator (`envs/`) with RK4
  integration at 50 Hz, exponential atmosphere, quadratic drag, mass/inertia
  depletion, a first-order TVC servo (`τ = 35 ms`) and a first-order engine
  spool (`τ = 100 ms`).
* There is **no hardware-in-the-loop, no aerodynamic identification, no
  sensor model beyond additive Gaussian noise, and no flight test**.
* The terminal success criterion (`lateral < 5 m`, `|v_z| < 2.5 m/s`,
  `v_lat < 1.5 m/s`, `tilt < 8°`, `|ω| < 0.25 rad/s`) is a *simulator*
  criterion, not a certification basis.

## 2. The fault model is narrow and deterministic

Five fault modes are injected: `LOE_THRUST`, `GIMBAL_BIAS`, `GIMBAL_STUCK`,
`COM_SHIFT` and nominal.  They are *static* after onset (a constant thrust
scale, a constant angle bias, a frozen angle, a static mass offset) and they
never heal.  Real actuator faults are stochastic, intermittent, rate-limited,
and often coupled (e.g. a leak that also changes the centre of mass and
induces vibration).

Consequences that show up in the numbers:

* **A frozen TVC is not necessarily a failure.**  A seized gimbal freezes at
  whatever angle it held at the onset, so a freeze within 1° of the vertical
  is an almost healthy flight.  Measured over 20 episodes
  (`scripts/stuck_freeze_angle.py`): the classical baseline lands 4/4
  benign freezes and 0/16 severe ones.  The published 20-episode benchmark
  sample contains eight benign freezes, which is why `Cascaded-PID` lands
  40 % of the `GIMBAL_STUCK` episodes.  This is a property of the fault
  draw, not of the controller.
* The **detector assumes the plant model is exact except for the fault**.
  The alarm is dominated by exact one-step residuals (actuator tracking,
  thrust tracking, rigid-body rate residual) computed from the *known* servo
  and engine models.  Unmodelled aerodynamics, sensor bias, flexible
  structures or plume interaction would raise the residual floor and the
  calibration would have to be re-tuned; the current thresholds would not
  transfer unchanged.
* The fault class and its parameters are **latched** at the first confirmed
  detection and never revised, because the compensation erases the evidence
  that identified the fault.  A fault that partially heals, or a second
  simultaneous fault, is out of scope.

## 3. The detector is physics-based, not purely learned

The headline "FDIR" numbers (0.9-step latency, 99-100 % isolation) come from
an **exact model-based residual** front-end plus a learned head that only
refines the decision, plus a physics-informed isolation prior.  The latent
KL divergence contributes only `0.1 × D_KL` to the anomaly score, because the
posterior latent collapses onto the GRU state (linear probe R² = 0.27 for
`z` versus 0.86 for `h`).  A reader looking for a "pure KL anomaly detector"
result should note that this is not what the code does.

## 4. The controller is a classical law plus a bounded learned residual

* `Cascaded-PID` is a cascaded position/velocity/attitude law with a
  saturation-aware tilt limiter.  It is a competent classical baseline, not
  a state-of-the-art flight controller (no tube MPC, no disturbance
  observer, no gain scheduling, no adaptive estimation).
* `AeroWM-FDIR` adds (a) a fault-conditioned feed-forward/trim law driven by
  the *measured* actuator state, and (b) a bounded learned residual
  (`0.15·tanh(·)`) around it.  The learned part cannot take the vehicle out
  of the classical law's basin, by construction.
* The MPPI variant is the policy prior plus a **bounded** model-based
  refinement (≤ 2 % on the attitude channels, ≤ 0.5 % on the throttle) that
  is **switched off per environment as soon as a fault is latched**, because
  the world model does not know about the fault.  The benchmark therefore
  compares two *refinement* strategies on top of the same FDIR front end;
  it is not a comparison of "model-based control" against "model-free
  control".
* The fault compensation in the MPPI variant and the actor variant is
  identical by construction, so their small differences (90-95 % versus
  95-100 %) are sampling noise at 20 episodes per mode.

## 5. Statistical strength

* Every published number is a **single seed, 20 episodes per mode**
  (8 parallel environments).  The 95 % Wilson interval on a 19/20 success
  rate is roughly 75-99 %, so differences of one or two episodes between
  configurations are not significant.
* The **same 20-episode benchmark was used to tune** the terminal flare
  level, the flare band, the residual scale and the planner scale.  There is
  no held-out set, so the published rates are optimistic.
* The observation noise is drawn from the *global* torch RNG, so a benchmark
  is reproducible only when the process performs the same sequence of
  allocations (same code path, same flags).  A per-run seeded generator
  would make the numbers independent of the controller order; this is a known
  wart, documented rather than hidden.

## 6. Scope of the published artefacts

* The published `results/eval_results.json` comes from
  `--episodes 20 --fast-mppi` (K = 64, H = 5, 2 CEM iterations).  The full
  planner from the design document (K = 512, H = 15, 4 iterations) is
  ~13 ms per environment and is not used for the benchmark sweep.
* The world model is trained on nominal and synthetic-fault data from this
  simulator only; no real flight data, no other vehicle, no transfer study.
* Throughput was measured on a single machine (RTX 4060 Laptop, 8 GB,
  PyTorch 2.11 + CUDA 12.8).  Other GPUs will differ.
* Training is not bit-exact reproducible (cuDNN autotuning, non-deterministic
  GPU reductions).  The released checkpoints are the exact ones that produced
  the published numbers.

## 7. What this work does *not* claim

* It does not claim that a latent world model is needed for fault detection
  on a rocket: exact physics residuals already do the job, and they do it
  better.  The world model earns its place by supplying the fault-conditioned
  context and the refinement policy.
* It does not claim flight readiness, fault coverage beyond the four injected
  modes, or a comparison against published fault-tolerant control methods.
* It does not claim the classical baseline is a fair "industry" reference
  beyond the role defined in the design document (AC-5.3).

---

## License

MIT (see `LICENSE`).  Research use; no warranty; do not use it to command a
real vehicle.

> **Research software notice.**  This project is research software for a
> simulated 6-DOF powered-descent vehicle.  It is released for reproducibility
> and for teaching/portfolio use.  It is NOT flight software, it has not been
> validated against hardware, and it carries no warranty of fitness for any
> purpose.  See the sections above for the full statement of scope.
