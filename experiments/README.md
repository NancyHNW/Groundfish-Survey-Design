# Experiments

Each module here is **one research question**, run directly from the repo root.
The module name selects the experiment — there are no `--mc` / `--sc` / `--bc`
flags any more.

```bash
python -m experiments.buffers      --ns 100 --nv 2 --cf 125 --time-limit 60
python -m experiments.monte_carlo  --ns 100 --nv 2 --cf 125 --time-limit 60
python -m experiments.strategies   --ns 100 --nv 2 --cf 125 --time-limit 60
python -m experiments.realisation  --ns 20  --nv 2 --cf 62.5 --seed 7
```

Every module supports `--help`.

---

## The four experiments

| Module | Question | Sweeps | Re-solves? |
|---|---|---|---|
| `buffers` | Does planning below full capacity pay for itself? | `capacity_buffer` | yes |
| `monte_carlo` | Which solver gives the most robust routes? | `method` | yes |
| `strategies` | What should a boat do when the hold fills? | overflow response | **no** |
| `realisation` | What does one season actually look like? | nothing — single draw | yes |

`strategies` solves **once** and re-scores that one solution under each
response. That is deliberate: the overflow response is chosen at sea, after the
plan is fixed, so every row must share the same route and the same planned
time. `buffers` and `monte_carlo` change the plan itself, so they re-solve.

### `buffers`
Solves at each buffer, scores all of them against one shared scenario set.
Planned time rises as the buffer tightens, so a buffer only pays if mean
realised time falls by more than planned time rose. Extra flag: `--buffers 0.7
0.8 0.9 1.0`.

### `monte_carlo`
Solves with each method, scores against shared scenarios. A solver that wins on
deterministic time need not win once catch is uncertain. Extra flags:
`--methods grasp_only grasp_swap tabu_move`, `--no-histograms`.

### `strategies`
Main table over `backtrack` / `forward` / `preemptive_0.8` / `preemptive_0.7`,
then a threshold sweep over 0.5–0.9. Extra flag: `--no-threshold-sweep`.

### `realisation`
Draws a single catch scenario and plots the planned route beside the route
actually sailed, plus a per-trip time comparison. This is the one that makes
overflow behaviour legible. Extra flags: `--seed`, `--full` (581 stations),
`--no-catch-table`.

---

## Shared flags

Defined once in `common.base_parser()` and inherited by all four, so the same
flag always means the same thing:

| Group | Flags |
|---|---|
| problem | `--ns --nv --cf --instance` |
| solver | `--method --time-limit --catch-source --capacity-buffer` |
| stochastic | `--n-scenarios --scenario-seed --strategy --threshold` |

`--strategy` reads its choices from `unified.stochastic_eval.STRATEGIES`, so
adding a strategy needs no CLI edit.

**`--catch-source` matters more than it looks.** The solver plans against this
data, but scenarios are *always* drawn from the historical distributions.
Planning against `gfsp` (mean 531 kg/station) while being scored against
historical (mean 812 kg) is a systematic bias, not just noise — it makes any
solver look worse than it is. Default is `historical`; `gfsp` is for when the
specifically want to study that mismatch.

---

## `common.py`

| Function | Purpose |
|---|---|
| `output_path(name, tag, inst, ext)` | One naming scheme for everything in `tests/outputs/` |
| `save_csv(rows, path)` | Write a table; if Excel has the file locked, writes a timestamped `_LOCKED-*` copy instead |
| `make_evaluator(seed, n)` | One shared scenario set per experiment run |
| `solve(**kw)` | Wraps the solver, adds `planned_time` |
| `summarise(result, planned, n_trips, **id)` | Evaluator result → one table row |
| `sweep_solve(param, values, ...)` | Re-solve per value; yields `(value, det, result, row)` |
| `sweep_eval(trips, inst, cases, ...)` | Re-evaluate one fixed solution per case |
| `add_baseline_delta(rows, key, ref, col)` | Re-centre `mean` on a reference row |
| `print_table(rows, columns, title)` | Fixed-width table |
| `base_parser()` | The shared flags above |

`sweep_solve` and `sweep_eval` are **generators**, not table builders. They
yield the raw solve result alongside the row so each experiment can add its own
per-iteration output — `monte_carlo` draws a histogram, `buffers` prints a
one-line summary — without the shared code needing to know about it.

---

## Reading the output tables

| Column | Meaning |
|---|---|
| `planned` | Deterministic total time, summed across vessels. A hard floor: realised time can only be ≥ this, since strategies only ever add detours. |
| `trips` | Number of planned trips |
| `P(exceed)` | Fraction of scenarios where **any** trip overflowed at any point |
| `E[ret]` | Mean number of unscheduled port returns per season |
| `ret/trip` | `E[ret]` ÷ `trips`. Needed because a tighter buffer plans more trips, so the raw count can fall just by spreading the same risk more thinly. |
| `mean time` | Mean realised time across scenarios |
| `p95 time` | 95th percentile — the bad-but-not-freak season |
| `vs …` | That row's `mean` minus the reference row's `mean`. Positive = worse. The reference row is 0.0 by construction. |

**Objective is summed vessel-hours, not calendar duration.** Two boats working
150 h each gives 300, not 150. It is not a makespan.

`p5 == planned` exactly means at least 5% of scenarios ran with zero overflow.
If `p5 > planned`, that plan never runs clean.

---

## Adding an experiment

1. New module in `experiments/`.
2. `parser = argparse.ArgumentParser(parents=[base_parser()])`, then add only the flags specific to that experiment.
3. Use `sweep_solve` if the swept parameter changes the routes, `sweep_eval` if it
   does not.
4. `print_table` → `save_csv` → plot.

## Adding an overflow strategy

1. Write `_walk_yours(...)` in `unified/stochastic_eval.py`, matching the
   signature of `_walk_backtrack`.
2. Add it to the `STRATEGIES` dict.

Validation, CLI choices, and the parametrised tests in
`tests/test_strategies.py` all read from that dict, so nothing else needs
touching.
