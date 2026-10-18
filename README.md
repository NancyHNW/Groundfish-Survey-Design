# Groundfish Survey Design under Catch Uncertainty

Route planning for the Icelandic groundfish survey (IceGFSDP) when catch is
uncertain. Four vessels tow 581 stations; a vessel whose hold fills mid-trip
must divert to port, and that costs time the plan never accounted for.

This repository connects two inherited planners to a catch model and a
simulator, so a plan can be judged by the time it is **likely to take** rather
than the time it was planned to take.

---

## Layout

| Folder | Contents |
|---|---|
| `unified/` | The layer this project adds: problem loaders, `ProblemInstance`, adapters to the inherited solvers, the deterministic evaluator, the catch model and the stochastic evaluator |
| `experiments/` | Nine command-line experiment runners plus `common.py`, which holds everything they share |
| `tests/` | The automated test suite. `tests/outputs/` holds generated CSVs and figures and `tests/logs/` holds run logs; both are gitignored. `tests/plans/` holds reusable saved plans and is kept visible because plans are reproducibility inputs |
| `archive/report-reproduction/` | Archived Chapter 4 batch scripts and duplicate reproducibility plans; retained for provenance, not normal operation |
| `final_code/` | Inherited heuristic planners (GRASP, tabu search, simulated annealing) and the Gurobi TSP |
| `gfsp_code/` | Inherited MILP formulations, the LKH solver and 1,620 test problems |
| `simulations/` | Historical survey data and the catch analysis built from it |

## Requirements

Python 3.12, Gurobi 13 with a licence, plus `numpy`, `scipy`, `pandas`,
`matplotlib`. Developed against NumPy 2.2.4, SciPy 1.15.2, pandas 2.2.3,
Matplotlib 3.10.5, Gurobi 13.0.1.

Gurobi runs single-threaded (`Threads = 1`, set in `final_code/better_tsp.py`):
the sub-problems are 20–60 node TSPs where parallelism is pure overhead, and a
single thread keeps memory flat across the thousands of solves a Monte Carlo
run performs.

---

## Architecture

Four layers, each of which can be varied while the others are held fixed.

**Data layer.** Three loaders read problems from either inherited codebase into
one `ProblemInstance` holding stations, ports, fleet and limits. A station is a
**tow**, so it occupies two nodes and may be entered from either end:
`station_nodes` maps station `s` to nodes `(13 + 2s, 13 + 2s + 1)` and
`node_to_station` inverts it. `N_PORTS = 13` is the single source of that
offset.

**Planning layer.** `run_heuristic_on_gfsp` wraps the inherited heuristics and
can scale each vessel's capacity by the planning buffer β. `solution_to_trips`
converts raw planner output into a standard trip format.

**Evaluation layer.** `evaluate_single_realisation` sails one plan through one
catch scenario under a chosen recourse policy; `StochasticEvaluator` aggregates
across scenarios. Scenarios never reach the planner, so no future catch
information leaks into planning.

**Experiment layer.** Nine runners, all taking their flags from one
`base_parser()` so a flag means the same thing everywhere.

### Saved plans

`--plan PATH` solves and saves on first use, loads on every later use, and
**refuses** to load a plan solved with different parameters. This is what makes
"solve once, score many ways" true across separate runs, and why the scoring
experiments take seconds rather than minutes.

```bash
# first call solves and writes; later calls load
python -m experiments.overflow_map --full --home-ports 0 6 10 12 \
    --time-limit 300 --n-scenarios 1000 --plan tests/plans/full_default.json
```

---

## Running experiments

Scenario seed (123), planner seed (42) and policy (`forward`) are defaults.
Prefix each with `python -m experiments.`

| Runner | Question |
|---|---|
| `overflow_map` | Where do overflows happen and what do they cost? |
| `realisation` | What does a single season look like? |
| `strategies` | How should a vessel respond when its hold overflows? |
| `monte_carlo` | Which planner gives the shortest realised time? |
| `variance` | How much variation comes from the planner rather than the catch? |
| `full_sweep` | Every planning buffer crossed with every policy |
| `paired` | Compare settings across many instances, differenced within block |
| `combine_blocks` | Re-reference a paired table without re-solving |
| `buffers` | Superseded by `full_sweep` |

Representative commands:

```bash
# cost of uncertainty on the full survey
python -m experiments.overflow_map --full --home-ports 0 6 10 12 \
    --time-limit 300 --n-scenarios 1000 --plan tests/plans/full_default.json

# planner comparison, paired across 30 test problems
python -m experiments.paired --compare method --instances 1-30 \
    --time-limit 60 --n-scenarios 500

# recourse policies on the full survey
python -m experiments.strategies --full --home-ports 0 6 10 12 \
    --time-limit 300 --n-scenarios 500 --plan tests/plans/full_default.json

# planner variability
python -m experiments.variance --full --home-ports 0 6 10 12 \
    --time-limit 300 --n-scenarios 1000 \
    --solver-seeds 42 52 62 72 82 92 102 112 122 132
```

### Two things that will bite you

**The restart loop stops on wall-clock, not iterations.** The number of
restarts a budget buys depends on machine load, so a plan is reproducible given
a seed **and** a restart count — not a seed alone. GRASP alone recorded 383,
360 and 308 restarts across three runs of identical settings; tabu search
(move) recorded six in one run and four in another, returning 760.9 h and
767.0 h. Use `--plan` wherever a fixed plan matters.

**Scenario sets at different N are not nested.** A 500-scenario draw is not the
first 500 of a 1,000-scenario draw, even at the same seed. Never difference a
result scored at one N against a result scored at another.

---

## Data

| File | Location | Used for |
|---|---|---|
| `spring_historical_english.xlsx` | `simulations/data/` | Historical spring-survey catches, 1985–2022; every scenario is drawn from fits to these |
| `smb.2019.dat` | `gfsp_code/data/` | Station tow endpoints and station labels |
| `ports.csv` | `gfsp_code/data/` | Port positions |
| `true_dist.npy` | `gfsp_code/data/` | Distance matrix used by the inherited GFSP MIP/LKH code |
| `true_time.npy` | `final_code/local data/` | Active travel-time matrix used by the heuristic planner and unified evaluator |
| `updated_true_dist.npy` | `final_code/local data/` | Clean upper-triangular distance source used by `unified.rebuild_matrix` |
| `true_dist_fixed.npy` | `final_code/local data/` | Rebuilt, distance-closed matrix written by `unified.rebuild_matrix --write`; retained as provenance/output |

**Catch model.** 518 of the 581 stations are fitted to their own records; the
other 63 take the distribution of the nearest station that has them. Draws are
normal with negatives clipped to zero, which raises the mean simulated catch
from 811.1 kg to 897.2 kg — 10.6% above the means the planner uses.

**Catch-source defaults differ by entry point.** The experiment runners default
to `--catch-source historical`, because their results are scored against
historical scenarios. The low-level `unified.solve` API defaults to
`--catch-source gfsp` for compatibility with the inherited test problems.
`final_code/sample data/RandomMatchingCatches.dat` and the per-instance GFSP
catch values remain available so the original solvers and compatibility
workflows still run.

### Travel-matrix files in `final_code/local data/`

The active route planner needs travel **time**, while the inherited MIP/LKH
code needs travel **distance**. The local directory contains several files
from the original distance-generation workflow, so similarly named files are
not interchangeable:

| File | Status and purpose |
|---|---|
| `true_time.npy` | **Active.** 1175 x 1175 symmetric hours matrix. Loaded by `final_code/classes.py` and `unified.evaluate`. |
| `updated_true_dist.npy` | **Rebuild input.** Upper-triangular nautical-mile distances with `-1` for entries not stored. Used by `unified.rebuild_matrix`; not loaded during normal solving. |
| `true_dist_fixed.npy` | **Rebuild output.** Shortest-path-closed nautical-mile distances corresponding to the current `true_time.npy`; useful for provenance and later rebuilds, but not the normal GFSP distance loader. |
| `true_dist(temporary).npy` | **Legacy/broken source.** Already symmetric, but the old `final_code` loader treated it as upper-triangular and doubled distances. Do not use it to regenerate current results. |
| `true_dist.npy` | Older local distance export. The active inherited GFSP distance loader uses `gfsp_code/data/true_dist.npy` instead. |
| `default_dist.npy`, `alt_dist.npy` | Earlier alternative distance exports; no active loader references them. |
| `binary_matrix.npy`, `updated_binary_matrix.npy`, `binary_matrix(temporary).npy` | Earlier land/edge availability masks used while constructing distances; no active solver loads them. |
| `land_station_point.npy` | Earlier land-routing/waypoint provenance array; no active solver loads it. |
| `tour.npy` | Earlier serialized route/object array; no active solver loads it. |

The archived pre-rebuild time matrix is in
`final_code/local data/matrix_archive/`, with its provenance in that folder's
`README.md`.

---

## Defects found and corrected

Three defects were found in inherited or project code during this work. All are
fixed; the numbers are recorded here because they determine which results are
comparable with which.

### 1. The travel-time matrix (inherited) — fixed 2026-10-08

The shipped `true_time.npy` had three independent faults:

**Every distance doubled.** `final_code/data_loader.py:load_true_dist` does
`dist[dist == -1] = 0; dist += dist.T`, which mirrors an upper-triangular
matrix. That is correct for `true_dist.npy`, but the loader reads
`true_dist(temporary).npy`, which is **already symmetric** — so the mirror
doubles instead of mirroring. Maximum distance went 507.25 → 1014.50 nm, i.e.
the fleet appeared to steam at 7 kn against the true 14 kn.

**Port 0 (Vestmannaeyjar) corrupted.** In the same file, 135 station legs to
port 0 were under 1 nm against a true 170–185 nm, and 270 station–port legs
came out *shorter than the great circle*, which is geometrically impossible.
All 270 involved port 0.

**Not a metric.** Floyd–Warshall closure improved 52% of cells, by a mean of
12.8 h and a maximum of 45.9 h, because land avoidance had been applied
inconsistently: roughly 11% of legs sat at exactly great-circle distance and
still crossed land, so they were billed a straight line through Iceland.

The corrected `true_time.npy` was rebuilt from `updated_true_dist.npy`; the
third defect needs the closure that `unified/rebuild_matrix.py` applies. The
legacy `final_code/data_loader.py` still contains the old loader for historical
compatibility, so it should not be used to regenerate the current time matrix.

**Closure is applied to distance, not time, and that matters.** Distance is
pure geometry, so a shorter path passing a station node is a legitimate
correction. Time is not speed-homogeneous — tows run at 7 kn and transit at
14 kn — so closing time would let a route "shortcut" through another station's
tow. Closing distance first and applying speeds afterwards avoids this, and is
verified not to shorten any tow leg.

**448 residual violations remain in the finished time matrix, all of them tow
legs.** That is correct and must not be closed away: a tow has to be towed, so
being slower than sailing around it is a real property rather than an error.

> **Results produced before 2026-10-08 are not comparable with later ones.**

To re-run: `python -m unified.rebuild_matrix` reports without writing;
`--write` backs up the current matrix and replaces it.

### 2. Diversions measured from the wrong end of a tow — fixed 2026-10-16

`station_nodes` stores each tow's **entry** node, and the recourse policies used
it as the point the vessel turns for port from. That is correct for exactly one
policy:

| policy | should divert from | why |
|---|---|---|
| Backtrack | **entry** | has not towed the station — its catch would not fit — so it turns back from where it arrived |
| Forward, Pre-emptive, Repair | **exit** | the tow is completed and the catch is aboard |

Repair was worse than wrong: it walked the prefix to the exit and then charged
a leg measured from the entry, billing the vessel from an end it was not at.
Six sites were corrected.

Measured bias **+0.43 h per season** (0.06% of a 761 h survey), with the wrong
port chosen entirely in **3% of events**. It reversed the sign of the
trip-scope repair comparison; no other ranking changed.

**Why the tests missed it.** Every fixture built its matrix with
`np.full((1175, 1175), travel)`. On a uniform matrix both ends of a tow are
equidistant from every port, so no uniform fixture can tell them apart — the
golden tripwire pinning the policies' exact output passed before and after the
fix. Six tests on a non-uniform fixture were added: three asserting which end
each policy departs from, three reconciling the time charged against the route
sailed.

### 3. Dropped repositioning legs and a Gurobi leak

`solution_to_trips` silently dropped port-to-port repositioning legs, which
have two nodes like an empty padding trip but carry real travel time. Every
reported planned time was too low, and the cost was hidden from the evaluator
entirely. Regression cover is in `tests/test_trip_extraction.py`.

The Gurobi TSP never called `dispose()`, so memory grew across a Monte Carlo
run until it crashed after about three hours. Fixed with `dispose()` in a
`finally` block plus `Threads = 1`, which is also faster for problems this
size.

---

## Tests

```bash
python -m pytest -m "not full" -q   # 227 tests, a few seconds
python -m pytest -q                 # all 236, including the full survey
```

| File | Tests | Covers |
|---|---|---|
| `test_repair.py` | 66 | The Repair policy: every station visited once, trips ending at the home port, sub-trips within capacity and the freshness limit, termination |
| `test_unified.py` | 42 | Loader and adapter correctness, plus golden baselines that fail if a fixed seed stops reproducing |
| `test_strategies.py` | 40 | The recourse policies, including which end of a tow each diverts from and reconciliation of time charged against route sailed |
| `test_paired.py` | 32 | The pairing harness: a constant effect is recovered exactly, scoring-only settings share one solve, intervals use *t* rather than the normal, infeasible blocks are counted |
| `test_intervals.py` | 31 | The intervals reported: Wilson against the normal approximation where P(overflow) saturates, and the √(n−1) treatment of the penalty mean |
| `test_single_realisation.py` | 11 | Capacity accounting, vessel-to-capacity mapping, station extraction, the Monte Carlo wrapper |
| `test_full_problem.py` | 9 | The real 581-station survey rather than a scaled-down proxy |
| `test_trip_extraction.py` | 5 | Regression cover for the dropped repositioning legs |

Assertions are mutation-tested: the input is broken in turn and the test must
fail.

---

## Headline results

Full survey, 581 stations, four vessels (14/45/200/200 t), home ports 0, 6, 10
and 12, tabu search (move) at 300 s, scored under Forward over 1,000 scenarios.

| | |
|---|---|
| Planned time | 760.9 h over 26 trips |
| Cost of uncertainty | 19.3 h (2.5% of planned) |
| P(at least one overflow) | 99.2% [98.4, 99.6] |
| Unplanned returns | 3.90 per season |
| Stations that ever overflow | 88 of 581 |
| Mean overflow position in trip | 0.85 (67% in the last fifth) |
| Share of cost on vessel 1 (14 t) | 94.2% — vessels 3 and 4 never overflow |
| Variance from the planner, not the catch | 81.6% |

Three findings worth knowing before using this code:

- **The shortest plan is not the most robust one.** Tabu search (move) gives
  the shortest survey and is significantly the least robust; ranking planners
  by planned time overstates the best planner's advantage.
- **Scope matters, the re-planner does not.** Re-planning the trip that
  overflowed is worth between −0.16 and +0.50 h; re-planning the vessel's whole
  remaining schedule costs 1.85 to 12.10 h.
- **Prevention works and does not pay.** Pre-emptive returns cut overflow from
  100% to 4.4% and lengthen the survey by up to 88.6 h, because here the
  failure *is* a return and a voluntary one costs the same port call.

Generated CSVs and figures land in `tests/outputs/`; saved plans in
`tests/plans/`; run logs in `tests/logs/`.
