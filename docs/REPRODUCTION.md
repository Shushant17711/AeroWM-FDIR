# Reproducing every published artefact

Every number, table and figure in `README.md` and
`results/TRAINING_REPORT.md` comes from one of the commands on this page.
Nothing is hard-coded.

**Reference machine**: RTX 4060 Laptop (8 GB), i7 13th gen, 24 GB RAM,
Linux, Python 3.12, PyTorch 2.11.0+cu128.  CUDA is used automatically when
available; the fast test subset also runs on CPU.

---

## 0. Environment

```bash
cd AeroWM-FDIR
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch
uv pip install --python .venv/bin/python -e ".[dev]" numpy matplotlib typer pyyaml
```

## 1. Test suite

```bash
# fast subset (~6 s)
.venv/bin/python -m pytest tests -q -m "not slow"

# full suite, including the PID gate benchmarks (~8 min)
.venv/bin/python -m pytest tests -q
```

The full suite prints the Gate-2 measurements used in the report:

```
PID nominal:      {'success': 47, 'crash': 1, 'timeout': 0, 'rate': 0.979}
PID stuck gimbal: {'success': 0,  'crash': 16, 'timeout': 0, 'rate': 0.0}
PID gimbal bias:  {'success': 0,  'crash': 1,  'timeout': 15, 'rate': 0.0}
PID CoM shift:    {'success': 1,  'crash': 14, 'timeout': 1, 'rate': 0.0625}
PID LOE:          {'success': 2,  'crash': 14, 'timeout': 0, 'rate': 0.125}
```

## 2. Gate 1 — throughput

```bash
PYTHONPATH=. .venv/bin/python scripts/measure_sps.py
# -> full env throughput: 267,915 SPS at B=1024   (262-278k across runs)
```

The RK4 integrator gates are asserted by `tests/test_rk4_integrator.py`
(365,196 SPS eager, 2,376,420 SPS compiled at B=1024).

## 3. Gate 2 — the classical reference

```bash
# 20 episodes/mode, the benchmark sample
PYTHONPATH=. .venv/bin/python scripts/_oracle_bench.py 20 150 0

# 24 episodes/mode, the oracle (ground-truth fault parameters) reference
PYTHONPATH=. .venv/bin/python scripts/_oracle_bench.py 24 150 1
```

The 48/16-episode unit-test protocol lives in `tests/test_pid_baseline.py`
(`run_episodes(..., onset=60)`).

## 4. Gate 3 — detection latency and isolation

```bash
PYTHONPATH=. .venv/bin/python scripts/check_fdir.py --episodes 20
```

```
mode            detected latency[steps]  latency[ms]  isolation
LOE_THRUST        100.0%           2.00         40.0      95.0%
GIMBAL_BIAS       100.0%           2.00         40.0     100.0%
GIMBAL_STUCK      100.0%           2.00         40.0      85.0%
COM_SHIFT         100.0%           2.00         40.0     100.0%
mean: detection 100.0 %, latency 2.00 steps (40.0 ms), isolation 95.0%
```

The latency is measured on the controller's **alarm** (the debounced,
physics-gated decision), not on a single sample above the threshold; that is
why it is one step larger than the raw crossing.

## 5. Gate 4 — the full benchmark (the published artefact)

```bash
PYTHONPATH=. .venv/bin/python -m eval.benchmark_fdir \
    --episodes 20 --fast-mppi --output results/eval_results.json
```

Protocol: 20 episodes per fault mode, 8 parallel environments, seed 0,
onset 150, severity 0.35 for `LOE_THRUST` and 0.5 otherwise, `--fast-mppi`
(CEM/MPPI with K = 64, H = 5, 2 iterations).  Wall clock ≈ 33 min on the
reference machine.

The benchmark loads `runs/aerowm_fdir/checkpoint_final.pt` and
`runs/ppo/ppo.pt` by default; pass `--wm-checkpoint` / `--ppo-checkpoint` to
override.  A run with untrained weights reports ~0 % for the FDIR variants,
which is the intended behaviour of the harness.

Verdict table:

```bash
PYTHONPATH=. .venv/bin/python scripts/gate_report.py results/eval_results.json
```

## 6. The telemetry figure

```bash
PYTHONPATH=. .venv/bin/python -m eval.plot_fdir_telemetry \
    --output results/fdir_recovery_demo.png --mode 2 --onset 120
```

Panel 2 uses a logarithmic axis because the anomaly score spans three
decades (nominal ≈ 1, a 40-σ residual signature ≈ 10²); the alarm threshold
is calibrated on the measured nominal surprise of the plotted flight.

## 7. Retraining the two models

Both training runs are budgeted to stay under two minutes of wall clock on
the reference machine.

```bash
# world model + FDIR head + residual actor (1,600 WM/FDIR updates,
# 500 actor updates, 6,000 DAgger transitions, T_burnin = 8)
PYTHONPATH=. .venv/bin/python -m train.train_fast

# model-free PPO baseline (residual around the classical law)
PYTHONPATH=. .venv/bin/python -m train.train_ppo \
    --iterations 80 --envs 128 --rollout 128 --bc-steps 20000 \
    --eval-every 10 --eval-episodes 48 \
    --output runs/ppo/ppo.pt --log runs/ppo/metrics.jsonl
```

Training is not bit-exact reproducible (cuDNN autotuning, non-deterministic
GPU reductions).  The checkpoints released with the repository are the exact
ones that produced the published benchmark.

## 8. Optional diagnostics

```bash
# why the classical baseline lands some stuck-gimbal flights
PYTHONPATH=. .venv/bin/python scripts/stuck_freeze_angle.py --episodes 20

# residual separation per fault mode on any checkpoint
PYTHONPATH=. .venv/bin/python -m eval.diagnose_fdir \
    --checkpoint runs/aerowm_fdir/checkpoint_final.pt

# stuck-gimbal reachability bound (oracle, ground-truth fault knowledge)
PYTHONPATH=. .venv/bin/python -m eval.analyze_stuck_gimbal
```
