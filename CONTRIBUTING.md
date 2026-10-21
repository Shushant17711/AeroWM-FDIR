# Contributing

Thanks for looking at AeroWM-FDIR.  This is a research repository: the
measurements in `results/` and in `results/TRAINING_REPORT.md` are the
deliverable, so a change that alters a number must come with the
measurement that justifies it.

## Ground rules

1. **Never hard-code a metric.**  Every number in the README, the report and
   the figure must be reproducible by a command in this repository.  If you
   quote a number, quote the command that produced it.
2. **Keep the contracts frozen.**  `configs/schema.py` defines the 18-D
   observation, the 3-D action, the 5 fault modes and the physics
   constants; `design.md` is authoritative over the older
   `aerowm_fdir_master_engineering_blueprint.md`.  Changing a contract
   invalidates the published benchmark and needs a new release.
3. **Show the failure.**  A gate that is not met is reported as not met
   (see `LIMITATIONS.md` §5 and `results/TRAINING_REPORT.md` §9.8 for an
   example).  Do not move a threshold to make a test pass.
4. **One change, one measurement.**  A pull request that changes a
   controller or a detector must include the before/after numbers for the
   affected mode.

## Development setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch
uv pip install --python .venv/bin/python -e ".[dev]" numpy matplotlib typer pyyaml
```

## Checks to run before opening a pull request

```bash
# fast unit tests (~6 s, CPU is fine)
.venv/bin/python -m pytest tests -q -m "not slow"

# the full suite, including the PID gate benchmarks (~8 min on an RTX 4060)
.venv/bin/python -m pytest tests -q

# lint
.venv/bin/python -m ruff check --select F,E9 agents models envs train eval configs scripts
```

If your change touches the controller, the detector or the reward, also
re-run the benchmark and attach the new `results/eval_results.json`:

```bash
PYTHONPATH=. .venv/bin/python -m eval.benchmark_fdir --episodes 20 --fast-mppi \
    --output results/eval_results.json
PYTHONPATH=. .venv/bin/python -m scripts.gate_report.py results/eval_results.json
```

## Style

* Keep the module docstrings honest: they state the measured effect of the
  mechanism they implement, and they are updated when the measurement
  changes.
* Prefer physics-explicit code over learned magic; every "learned" quantity
  in this project has a physical read-out next to it.
* No new runtime dependencies without a discussion in an issue.

## Reporting bugs

Open an issue with the command you ran, the seed, the GPU, and the output of
`python scripts/gate_report.py results/eval_results.json`.  A minimal
reproduction is worth more than a description.
