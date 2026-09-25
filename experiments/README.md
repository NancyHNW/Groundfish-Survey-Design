# Experiments

Each module here is **one research question**, run directly from the repo root.
The module name selects the experiment — there are no `--mc` / `--sc` / `--bc`
flags any more.

```bash
python -m experiments.buffers      --ns 100 --nv 2 --cf 125 --time-limit 60
python -m experiments.monte_carlo  --ns 100 --nv 2 --cf 125 --time-limit 60
python -m experiments.strategies   --ns 100 --nv 2 --cf 125 --time-limit 60
python -m experiments.realisation  --ns 20  --nv 2 --cf 62.5 --seed 7
python -m experiments.paired       --compare strategy --instances 1-30
```

Every module supports `--help`.

---

## The experiments

| Module | Question | Sweeps | Re-solves? |
|---|---|---|---|
| `buffers` | Does planning below full capacity pay for itself? | `capacity_buffer` | yes |
| `monte_carlo` | Which solver gives the most robust routes? | `method` | yes |
| `strategies` | What should a boat do when the hold fills? | overflow response | **no** |
| `realisation` | What does one season actually look like? | nothing — single draw | yes |
| `paired` | Is a difference real, or is it solver noise? | any of the above | as needed |
| `full_sweep` | Buffer crossed with overflow response, on the full problem | both | one per buffer |

`paired` is the one to reach for when a single solve cannot settle the
question. Restart noise on the full problem spans tens of hours, wider than
most of the effects being looked for, so it measures every setting on the same
(instance, seed) block and differences within the block.

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
Main table over eight rows — `backtrack`, `forward`, `preemptive_0.8`,
`preemptive_0.7`, and `repair` crossed with its two scopes and two
re-planners — then a threshold sweep over 0.5–0.9. Extra flag:
`--no-threshold-sweep`.

**`backtrack` and `forward` disagree about what an overflow is**, and the
difference is not a detail.

Under `backtrack` the station that tips the hold over is **not fished**: its
catch would not fit. The boat lands what it is carrying, sails back, tows that
station, and carries on. So the out-and-back is real at every station, a
trip's last one included, because the boat must still return for it. Two
consequences: the hold leaves port carrying that station's catch rather than
empty, and the catch is *not* re-drawn on the return — it belongs to the
station in that scenario, every strategy has to see the same draw, and
re-drawing would only ever follow a high draw and so flatter backtrack through
regression to the mean.

Under `forward` the catch **is** aboard, which is what lets the boat carry on
from the port to the next station without losing it. That makes an overflow on
a trip's last station nearly free — the next thing in the route was the port
anyway, so only which port differs:

```
t(station → diverted port) + t(diverted port → end) − t(station → end)
```

exactly zero when the nearest port is the end port.

`forward` used to charge a full out-and-back there, copied from backtrack, for
a journey it never makes. Correcting that is worth several hours a season and
is why `forward` now beats `repair` on the test problems.

Repair re-plans against **true** capacity, not the buffered planning capacity
the routes were built with. That is what a skipper would do at sea, but it
means repair partially undoes the buffer wherever the two are crossed.

### `paired`
`--compare strategy | buffer | method`, over `--instances 1-30` and
`--solver-seeds`. Reports the mean difference against a baseline with a 95%
interval, and groups the settings into beats / reliably worse / cannot
separate — a setting can separate from the baseline by being worse.

Settings that differ only in scoring share one solve, so eight strategies
over 30 instances costs 30 solves, not 240.

### `realisation`
Draws a single catch scenario and plots the planned route beside the route
actually sailed, plus a per-trip time comparison. This is the one that makes
overflow behaviour legible. Extra flags: `--seed`, `--full` (581 stations),
`--no-catch-table`.

---

## Shared flags

Defined once in `common.base_parser()` and inherited by every module, so the
same flag always means the same thing:

| Group | Flags |
|---|---|
| problem | `--ns --nv --cf --instance` |
| solver | `--method --time-limit --catch-source --capacity-buffer` |
| stochastic | `--n-scenarios --scenario-seed --strategy --threshold` |
| repair | `--repair-scope --repair-planner --repair-solver-time` |

`--strategy`, `--repair-scope` and `--repair-planner` read their choices from
the registries in `unified.stochastic_eval`, so adding one needs no CLI edit.

The repair flags only do anything with `--strategy repair`, and only in the
modules that sweep a single strategy — `buffers`, `monte_carlo`, `realisation`.
`strategies`, `full_sweep` and `paired` carry their own fixed lists of repair
settings and ignore them.

`--repair-scope trip` re-plans the overflowed trip; `boat` re-plans everything
that boat has left. `--repair-planner` is `nn`, `2opt` or `solver`, worst to
best; `solver` is thousands of times slower and is for one-off runs only.

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
| `planned` | Deterministic total time, summed across vessels. A floor for the detour strategies, which only ever add to an unchanged route. `repair` replaces routes, so it can come in under it. |
| `trips` | Number of planned trips |
| `P(exceed)` | Fraction of scenarios where **any** trip overflowed at any point |
| `E[ret]` | Mean number of unscheduled port returns per season |
| `ret/trip` | `E[ret]` ÷ `trips`. Needed because a tighter buffer plans more trips, so the raw count can fall just by spreading the same risk more thinly. |
| `mean time` | Mean realised time across scenarios |
| `+/-95%` | Monte Carlo 95% interval on that mean. Scenario sampling only — it says nothing about solver restart noise, which is the larger term whenever two rows come from different solves. |
| `p95 time` | 95th percentile — the bad-but-not-freak season |
| `E[rep]` | Mean re-plans per season. Zero for every strategy but `repair`. |
| `cap hit` | Fraction of scenarios that hit `max_repairs` and finished on backtrack detours instead. Anything above zero means those rows are not pure repair. |
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
2. Add it to the `STRATEGIES` dict, and to `DETOUR_STRATEGIES` if it only adds
   to an unchanged route rather than replacing one.
3. If it needs more than the walker signature carries, give it a branch in
   `_run_strategy` — that is where `preemptive`, `forward` and
   `repair` get their extra arguments.

Validation, CLI choices, and the parametrised tests in
`tests/test_strategies.py` all read from those two, so nothing else needs
touching. Add the row to `CASES` in `experiments/strategies.py` and to the
lists in `full_sweep.py` and `paired.py` to have it appear in the tables.

`_STRATEGY_COLOURS` in `stochastic_eval.py` holds nine, which is what the sweep
uses. A tenth wraps the palette and gives two strategies the same colour, so
extend it and re-check the separation first. `plot_sweep_grid` warns when it
runs short, but the figure is already wrong by then.
