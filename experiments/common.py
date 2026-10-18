"""Bits every experiment needs: output paths, scenarios, solving, tables.

Nothing in here is an experiment itself.

base_parser() is the important one. Every experiment gets its flags from it,
so the same flag cannot end up meaning different things in different scripts.
"""

import csv
import math
import os

import numpy as np

# CLI method names -> the names run_heuristic_on_gfsp uses.
# Only place that translates between the two.
METHODS = {
    "grasp_only": "grasp_only",   # construction only, no improvement
    "grasp_swap": "grasp",        # GRASP + next-descent swap
    "tabu_swap": "tabu_swap",     # GRASP + tabu search over swaps
    "tabu_move": "tabu_move",     # GRASP + tabu search over moves
    "tabu_combined": "tabu_combined",  # GRASP + tabu over swaps and moves
    "sa": "sa",                   # simulated annealing
}

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUT_DIR = os.path.join(_ROOT, "tests", "outputs")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def output_path(name, tag, inst=None, ext=".png", home_ports=None):
    """Build a path under tests/outputs/ from experiment, tag and problem size.

    e.g. output_path("buffer-comparison", "tabu_move-backtrack", inst, ".csv")
      -> tests/outputs/buffer-comparison_tabu_move-backtrack_ns100-nv2.csv

    home_ports adds an hp0-6-10-12 segment so a custom-port run cannot
    overwrite a default one. Omitted when the ports are the default.

    Folder is gitignored, these all get regenerated.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parts = [name, tag]
    if inst is not None:
        parts.append(f"ns{inst.ns}-nv{inst.n_boats}")
    if home_ports:
        parts.append("hp" + "-".join(str(int(p)) for p in home_ports))
    return os.path.join(OUTPUT_DIR, "_".join(parts) + ext)


def save_csv(rows, path):
    """Write rows to CSV. Falls back to a timestamped name if locked.

    Excel locks a CSV while it is open, which would otherwise crash a long
    experiment right at the last step.

    The fallback name is timestamped on purpose. A fixed suffix like _v2 looks
    exactly like a current file a few days later, and its numbers belong to
    whatever version of the model produced them. One such leftover, from before
    boats had to return home, was mistaken for a trip-extraction bug.
    """
    if not rows:
        print("No rows to save.")
        return None
    try:
        _write_csv(rows, path)
    except PermissionError:
        from datetime import datetime

        locked = os.path.basename(path)
        stem, ext = os.path.splitext(path)
        path = f"{stem}_LOCKED-{datetime.now():%Y%m%d-%H%M}{ext}"
        _write_csv(rows, path)
        print(f"!! {locked} was locked (open in Excel?).\n"
              f"!! Wrote a timestamped copy instead -- {locked} is now STALE.")
    print(f"Table saved to {path}")
    return path


def _write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Solving and evaluating
# ---------------------------------------------------------------------------

def make_evaluator(scenario_seed=123, n_scenarios=500):
    """Build a StochasticEvaluator over one fixed set of catch scenarios.

    Every experiment shares one scenario set across the things it compares, so
    row differences come from the solutions and not from resampling the catch.
    """
    from unified.stochastic_catch import CatchSimulator
    from unified.stochastic_eval import StochasticEvaluator

    sim = CatchSimulator(seed=scenario_seed)
    sim.fit()
    return StochasticEvaluator(scenarios=sim.sample(n_scenarios=n_scenarios))


def solve(ns=100, nv=2, cf=125, instance=1, method="tabu_move",
          time_limit=10, catch_source="gfsp", capacity_buffer=1.0,
          **kwargs):
    """Solve one gfsp instance, with planned_time added to the result.

    Wraps run_heuristic_on_gfsp, resolves the method alias and sums the planned
    time (every caller wanted that anyway).

    Pass full=True through kwargs for the 581-station survey, in which case
    ns/nv/cf/instance are ignored.
    """
    from unified.solve import run_heuristic_on_gfsp

    result = run_heuristic_on_gfsp(
        ns=ns, nv=nv, cf=cf, instance=instance,
        method=METHODS.get(method, method),
        time_limit=time_limit,
        catch_source=catch_source,
        capacity_buffer=capacity_buffer,
        **kwargs
    )
    result["planned_time"] = sum(t["total_time"] for t in result["trips"])
    return result


# ---------------------------------------------------------------------------
# Saved plans
# ---------------------------------------------------------------------------

PLAN_DIR = os.path.join(_ROOT, "tests", "plans")

# What has to match before a saved plan may be reused. Scoring keys are not in
# here -- the whole point is to score one plan many ways.
_PLAN_KEYS = ("ns", "nv", "cf", "instance", "method", "time_limit",
              "catch_source", "capacity_buffer", "full", "home_ports", "seed")


# What solve() uses when a caller leaves a key out. Without this an experiment
# that omits seed records seed=None while one that passes 42 records 42, for
# the same solve -- which is the mismatch this whole mechanism is here to stop.
_PLAN_DEFAULTS = {"seed": 42, "full": False, "home_ports": None}


def _plan_params(kwargs):
    """Fill in solve()'s defaults so two callers describe one solve the same."""
    import inspect

    out = dict(_PLAN_DEFAULTS)
    for name, p in inspect.signature(solve).parameters.items():
        if p.default is not inspect.Parameter.empty:
            out[name] = p.default
    out.update({k: v for k, v in kwargs.items() if k in _PLAN_KEYS})
    return {k: out.get(k) for k in _PLAN_KEYS}


def _plain(x):
    """numpy -> json. Trip nodes come back as np ints and json refuses them."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def save_plan(result, path, params):
    """Write a solved plan plus the parameters that produced it."""
    import datetime
    import json

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blob = {
        "params": _plain(_plan_params(params)),
        "saved": datetime.datetime.now().isoformat(timespec="seconds"),
        "trips": _plain(result["trips"]),
        "planned_time": float(result["planned_time"]),
        "planned_catch": _plain(result.get("planned_catch")),
        "feasible": bool(result.get("feasible", True)),
        # Provenance for Table 3.3 and the Appendix B claim
        "restarts": int(result.get("iterations", 0)),
        "solve_s": float(result.get("elapsed", 0.0)),
    }
    with open(path, "w") as f:
        json.dump(blob, f, indent=1)
    return path


def load_plan(path, params=None):
    """Read a saved plan back into the shape solve() returns.

    Rebuilds the instance from the saved parameters rather than storing it, so
    the instance cannot drift from the plan. If params is given, every key in
    _PLAN_KEYS must match -- scoring the wrong plan silently is the failure
    this whole mechanism exists to stop.
    """
    import json

    from unified.loaders import load_gfsp_full_problem, load_gfsp_problem

    with open(path) as f:
        blob = json.load(f)
    saved = blob["params"]

    if params is not None:
        want = _plain(_plan_params(params))
        bad = {k: (saved.get(k), want[k]) for k in _PLAN_KEYS
               if saved.get(k) != want[k]}
        if bad:
            lines = "\n".join(f"    {k}: saved {s!r}, asked for {w!r}"
                              for k, (s, w) in bad.items())
            raise ValueError(
                f"{path} was solved with different parameters:\n{lines}\n"
                "  Delete the file to re-solve, or point --plan elsewhere.")

    inst = (load_gfsp_full_problem() if saved.get("full")
            else load_gfsp_problem(ns=saved["ns"], nv=saved["nv"],
                                   cf=saved["cf"],
                                   instance=saved["instance"]))
    catch = blob.get("planned_catch")
    return {
        "trips": blob["trips"],
        "instance": inst,
        "planned_time": blob["planned_time"],
        "planned_catch": np.asarray(catch, dtype=float) if catch else None,
        "feasible": blob["feasible"],
        "iterations": blob.get("restarts", 0),
        "elapsed": blob.get("solve_s", 0.0),
        "loaded_from": path,
    }


def solve_cached(plan_path, **kwargs):
    """Load the plan at plan_path, or solve once and save it there.

    This is what makes 'solve once, score many ways' true across separate
    runs. First call solves and writes; every later call reads. Pass
    plan_path=None to solve without caching.
    """
    if not plan_path:
        return solve(**kwargs)
    if os.path.exists(plan_path):
        det = load_plan(plan_path, params=kwargs)
        print(f"Loaded plan from {plan_path} "
              f"({det['planned_time']:.1f}h, {len(det['trips'])} trips, "
              f"{det['iterations']} restarts)")
        return det
    det = solve(**kwargs)
    save_plan(det, plan_path, kwargs)
    print(f"Saved plan to {plan_path}")
    return det


def _mc_se(std, n):
    """Standard error of a mean over n scenarios.

    std is a population sd (np.std, ddof=0), so the error is std/sqrt(n-1).
    Scenario sampling only -- says nothing about solver restart noise.
    """
    n = n or 0
    return std / math.sqrt(n - 1) if n > 1 else 0.0


def _wilson_ci(k, n, z=1.96):
    """95% interval for a proportion k/n, Wilson rather than normal.

    P(exceed) saturates near 1.0 on the full problem, where the textbook
    p +/- z*sqrt(p(1-p)/n) is both too narrow and capable of reaching past 1.
    Wilson stays inside [0, 1] and does not collapse to zero width when every
    scenario overflows, which is exactly the regime these runs sit in.

    Returns (lo, hi). Both zero when n is zero.
    """
    if not n:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _returns_error(result, e_returns, n_trips):
    """Error on E[returns], absolute and per trip.

    Empty when the per-scenario counts are absent, so a stubbed result works.
    """
    counts = [r["n_unscheduled_returns"] for r in result.get("all_results") or []]
    if len(counts) < 2:
        return {}
    var = sum((c - e_returns) ** 2 for c in counts) / len(counts)
    se = _mc_se(math.sqrt(var), len(counts))
    return {
        "returns_se": se,
        "returns_ci95": 1.96 * se,
        "returns_per_trip_ci95": 1.96 * se / n_trips if n_trips else 0.0,
    }


def _plan_balance(trips):
    """Stations on the busiest and quietest vessel, and the spread in hours.

    The station cap is a per-plan property, so it cannot be read off the
    scenarios. Reported because a planner that breaks the cap is not
    comparable with one that holds it, and feasible=True does not check it.
    """
    from unified.problem import N_PORTS
    stations, hours = {}, {}
    for t in trips:
        b = t["boat_id"]
        stations[b] = stations.get(b, 0) + len(
            {(n - N_PORTS) // 2 for n in t["nodes"] if n >= N_PORTS})
        hours[b] = hours.get(b, 0.0) + t["total_time"]
    if not stations:
        return {}
    return {"max_stations": max(stations.values()),
            "min_stations": min(stations.values()),
            "makespan_planned": max(hours.values()),
            "vessel_spread": max(hours.values()) - min(hours.values())}


def summarise(result, planned, n_trips, feasible=None, trips=None, **extra):
    """Turn an evaluator result into one table row.

    Every experiment reports the same metrics. **extra is the column naming the
    row (buffer, strategy, method), and lands first in the dict.

    Pass `trips` to get the plan's station balance as well -- the cap check for
    the planner comparison, which no per-scenario measure can supply.
    """
    td = result["total_time_distribution"]
    e_returns = result["expected_unscheduled_returns"]
    n_sims = result.get("n_simulations")
    se = _mc_se(td["std"], n_sims)

    # Realised minus planned: what uncertainty cost this plan. Already computed
    # by the evaluator, previously thrown away. .get() keeps a stubbed result
    # working, and the tests stub one.
    tp = result.get("time_penalty_distribution") or {}
    penalty_se = _mc_se(tp.get("std", 0.0), n_sims)
    # Wilson needs the count, and the evaluator only reports the proportion.
    # Recovering it by multiplying back is exact -- p is a mean of booleans, so
    # p*n is a whole number -- and it keeps the interval consistent with the
    # p_exceed the table actually prints.
    p_exceed = result["p_capacity_exceedance"]
    exceed_lo, exceed_hi = _wilson_ci(round(p_exceed * (n_sims or 0)),
                                      n_sims or 0)

    row = dict(extra)
    row.update({
        "planned": planned,
        "trips": n_trips,
        "p_exceed": p_exceed,
        "p_exceed_lo": exceed_lo,
        "p_exceed_hi": exceed_hi,
        "e_returns": e_returns,
        # Per trip, not absolute. A tighter buffer plans more trips, so the
        # raw count can drop just by spreading the same risk more thinly.
        "returns_per_trip": e_returns / n_trips if n_trips else 0.0,
        "mean": td["mean"],
        "sd": td["std"],
        "mc_se": se,
        "mc_ci95": 1.96 * se,
        "p5": td["p5"],
        "p95": td["p95"],
        "penalty": tp.get("mean", 0.0),
        "penalty_sd": tp.get("std", 0.0),
        "penalty_ci95": 1.96 * penalty_se,
        # Repair only, and zero everywhere else. Carried so a capped run is
        # visible: past the cap repair gives up and sails on backtrack, which
        # otherwise looks like a clean result.
        "e_repairs": result.get("expected_repairs", 0.0),
        "p_cap_hit": result.get("p_repair_cap_hit", 0.0),
    })
    # Delay at the 95th percentile, and U as a share of the plan. The
    # evaluator computes both halves; only the mean was being kept.
    row["penalty_p95"] = tp.get("p95", 0.0)
    row["u_pct"] = (row["penalty"] / planned) if planned else 0.0

    # Realised makespan: the sum hides it, and the two can move in opposite
    # directions -- loosening the station cap cut the sum and raised this.
    row.update(_makespan(result))

    row.update(_returns_error(result, e_returns, n_trips))
    if trips is not None:
        row.update(_plan_balance(trips))
    if feasible is not None:
        row["feasible"] = feasible
    return row


def _makespan(result):
    """Mean and p95 of the longest single vessel's time, over scenarios.

    The objective is summed vessel-hours, so nothing else reports this. It
    matters if the survey is bounded by its season rather than by total hours.
    """
    per_scenario = []
    for res in (result.get("all_results") or []):
        hours = {}
        # .get on both levels: a stubbed result carries neither, and the tests
        # stub exactly that.
        for d in (res.get("trip_details") or []):
            hours[d["boat_id"]] = hours.get(d["boat_id"], 0.0) + \
                d["adjusted_time"]
        if hours:
            per_scenario.append(max(hours.values()))
    if not per_scenario:
        return {}
    arr = np.asarray(per_scenario)
    return {"makespan": float(arr.mean()),
            "makespan_p95": float(np.percentile(arr, 95))}


def sweep_solve(param, values, evaluator, key=None, fmt=None,
                strategy="backtrack", preemptive_threshold=0.8,
                repair_scope="trip", repair_planner="nn",
                repair_solver_time=0.0, **fixed):
    """Re-solve for each value of param. Use when the sweep changes the routes.

    buffers sweeps capacity_buffer, monte_carlo sweeps method.

    Yields (value, det, result, row) instead of returning rows, so each
    experiment can print or plot its own per-value extras without this function
    needing to know about them.

    Parameters
    ----------
    param : str, the solve() keyword that varies
    values : iterable, the values to try
    evaluator : StochasticEvaluator, shared across every value
    key : str, column name in the row (defaults to param)
    fmt : str, format spec for the banner, e.g. ".0%"
    **fixed : everything else, passed straight to solve()
    """
    key = key or param
    for value in values:
        shown = format(value, fmt) if fmt else value
        print(f"\n{'=' * 60}\n{key}: {shown}\n{'=' * 60}")

        det = solve(**{param: value}, **fixed)
        result = evaluator.evaluate(
            det["trips"], det["instance"], strategy=strategy,
            preemptive_threshold=preemptive_threshold,
            repair_scope=repair_scope, repair_planner=repair_planner,
            repair_solver_time=repair_solver_time,
            planned_catch=det.get("planned_catch"))
        row = summarise(result, det["planned_time"], len(det["trips"]),
                        feasible=det["feasible"], trips=det["trips"],
                        **{key: value})
        yield value, det, result, row


def sweep_eval(trips, inst, cases, evaluator, planned, key="case",
               planned_catch=None):
    """Re-score one fixed solution under each case. Routes stay the same.

    The overflow strategy gets picked at sea, after the plan is fixed, so every
    row has to share one route. Solving once keeps planned time constant, which
    means the differences are purely the cost of reacting.

    Yields (label, result, row).

    Parameters
    ----------
    cases : list of (label, kwargs), kwargs go to evaluator.evaluate
    planned : float, planned time, the same for every row
    """
    for label, kwargs in cases:
        result = evaluator.evaluate(trips, inst, planned_catch=planned_catch,
                                    **kwargs)
        row = summarise(result, planned, len(trips), trips=trips,
                        **{key: label})
        yield label, result, row


def add_baseline_delta(rows, key, baseline_value, column="vs_baseline"):
    """Add a column of each row mean minus the baseline row mean.

    The means sit in a narrow band, so the differences hide in the third digit.
    Re-centring on a reference row makes the size of the effect readable.
    Positive = worse (more time) than the baseline.
    """
    baseline = next((r["mean"] for r in rows if r[key] == baseline_value), None)
    if baseline is None:
        return rows
    for row in rows:
        row[column] = row["mean"] - baseline
    return rows


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def strategy_tag(strategy, repair_scope="trip", repair_planner="nn"):
    """Name a run's strategy for its output filename.

    Scope and planner have to be in the name, or two repair runs write over
    each other's CSV and figure.
    """
    if strategy != "repair":
        return strategy
    return f"repair-{repair_scope}-{repair_planner}"


# Evaluator kwargs; everything else in a setting goes to solve(). Settings
# differing only in these share one solve -- the strategy is picked at sea
# against a fixed plan, so re-solving per strategy would break the pairing.
EVAL_KEYS = ("strategy", "preemptive_threshold", "max_repairs",
             "repair_scope", "repair_planner", "repair_solver_time")


def parse_instances(text):
    """Parse an instance spec: "1-30", "1,4,7", "1-5,9" all work."""
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


# Two-tailed 95% critical values. At n=8 blocks, t is 2.365 against the normal
# 1.96 -- using 1.96 there would overstate significance by a fifth.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980}


def _t_critical(df):
    """Two-tailed 95% t value. Falls back to the next df down, so it errs wide."""
    if df < 1:
        return 0.0
    if df in _T95:
        return _T95[df]
    below = [d for d in _T95 if d <= df]
    return _T95[max(below)] if below else 1.96


def _split_setting(kwargs):
    """Split one setting's kwargs into (solve kwargs, evaluate kwargs)."""
    eval_kw = {k: v for k, v in kwargs.items() if k in EVAL_KEYS}
    solve_kw = {k: v for k, v in kwargs.items() if k not in EVAL_KEYS}
    return solve_kw, eval_kw


def _group_by_solve(settings):
    """Group settings sharing solve kwargs -> (solve_kw, [(label, eval_kw)]).

    Keyed on a repr because home_ports arrives as a list, which is unhashable.
    """
    groups = {}
    for label, kwargs in settings:
        solve_kw, eval_kw = _split_setting(kwargs)
        key = repr(sorted(solve_kw.items()))
        groups.setdefault(key, (solve_kw, []))[1].append((label, eval_kw))
    return list(groups.values())


def paired_compare(settings, baseline, instances=(1,), solver_seeds=(42,),
                   evaluator=None, scenario_seed=123, n_scenarios=500,
                   progress=True, **fixed):
    """Measure every setting on every block, differencing inside the block.

    A block is one (instance, solver seed) pair. Differencing within it cancels
    instance difficulty, and the shared restarts cancel much of the solver noise.

    Parameters
    ----------
    settings : list of (label, kwargs), kwargs split on EVAL_KEYS
    baseline : str, the label every difference is taken against
    instances, solver_seeds : iterable, their cross product is the blocks
    progress : bool, per-block progress. Not called `verbose` -- solve() has one
        of those already, and the two shadowing silences the wrong thing
    **fixed : passed to every solve(). No `instance` or `seed`

    Returns (summary_rows, block_rows).
    """
    labels = [label for label, _ in settings]
    if baseline not in labels:
        raise ValueError(f"baseline {baseline!r} is not one of {labels}")
    for clash in ("instance", "seed"):
        if clash in fixed:
            raise ValueError(f"pass {clash} through its own argument, "
                             f"not through **fixed")

    if evaluator is None:
        evaluator = make_evaluator(scenario_seed, n_scenarios)

    groups = _group_by_solve(settings)
    blocks = [(i, s) for i in instances for s in solver_seeds]
    if progress:
        print(f"{len(blocks)} blocks x {len(groups)} solves = "
              f"{len(blocks) * len(groups)} solves for "
              f"{len(blocks) * len(settings)} measurements")

    block_rows = []
    for b, (inst_no, seed) in enumerate(blocks, 1):
        for solve_kw, members in groups:
            det = solve(instance=inst_no, seed=seed, verbose=False,
                        **solve_kw, **fixed)
            for label, eval_kw in members:
                result = evaluator.evaluate(
                    det["trips"], det["instance"],
                    planned_catch=det.get("planned_catch"), **eval_kw)
                block_rows.append(summarise(
                    result, det["planned_time"], len(det["trips"]),
                    feasible=det["feasible"],
                    instance=inst_no, seed=seed, setting=label))
        if progress:
            print(f"  block {b}/{len(blocks)}: instance {inst_no}, seed {seed}",
                  flush=True)

    return _paired_summary(block_rows, labels, baseline), block_rows


def _paired_summary(block_rows, labels, baseline):
    """Within-block differences against the baseline, one row per setting."""
    by_block = {}
    for r in block_rows:
        by_block.setdefault((r["instance"], r["seed"]), {})[r["setting"]] = r

    rows = []
    for label in labels:
        paired = [(b[label], b[baseline]) for b in by_block.values()
                  if label in b and baseline in b]
        diffs = [mine["mean"] - base["mean"] for mine, base in paired]
        n = len(diffs)
        mean_level = sum(m["mean"] for m, _ in paired) / n if n else 0.0
        diff = sum(diffs) / n if n else 0.0

        # n-1: a sample of instances, not the population of them
        if n > 1:
            sd = math.sqrt(sum((d - diff) ** 2 for d in diffs) / (n - 1))
            se = sd / math.sqrt(n)
            half = _t_critical(n - 1) * se
        else:
            sd = se = half = 0.0

        lo, hi = diff - half, diff + half
        rows.append({
            "setting": label,
            "n": n,
            "mean": mean_level,
            "diff": diff,
            "sd": sd,
            "se": se,
            "ci_lo": lo,
            "ci_hi": hi,
            # Plain bool: a numpy one leaks into the CSV and fails `is False`
            "significant": bool(n > 1 and (lo > 0 or hi < 0)),
            "infeasible_blocks": sum(1 for m, _ in paired
                                     if m.get("feasible") is False),
        })
    return rows


def paired_scenarios(times, baseline):
    """Paired differences across scenarios, for one fixed solution.

    The block version pairs on instances. This pairs on scenarios, which is
    what is left once the solve is fixed -- the solver is deterministic at the
    budgets these runs use, so the catch draw is the only randomness.

    It matters because the strategies are strongly correlated: a heavy
    scenario slows all of them, and that shared bad luck cancels on
    subtraction. Two overlapping per-strategy error bars say nothing about
    whether the difference between them is real; this does.

    times : {label: array of per-scenario realised time}
    """
    base = times[baseline]
    rows = []
    for label, arr in times.items():
        d = np.asarray(arr) - base
        n = len(d)
        mean = float(d.mean())
        if n > 1:
            sd = float(d.std(ddof=1))
            se = sd / math.sqrt(n)
            half = _t_critical(n - 1) * se
        else:
            sd = se = half = 0.0
        lo, hi = mean - half, mean + half
        rows.append({
            "setting": label,
            "n": n,
            "mean": float(np.mean(arr)),
            "diff": mean,
            "sd": sd,
            "se": se,
            "ci_lo": lo,
            "ci_hi": hi,
            "significant": bool(n > 1 and (lo > 0 or hi < 0)),
            "infeasible_blocks": 0,
        })
    return rows


PAIRED_COLUMNS = [
    ("setting", "setting", 20, ""),
    ("n", "n", 5, "d"),
    ("mean", "mean", 10, ".1f"),
    ("diff", "vs base", 10, "+.2f"),
    ("sd", "sd", 8, ".2f"),
    ("ci_lo", "ci lo", 9, "+.2f"),
    ("ci_hi", "ci hi", 9, "+.2f"),
    ("significant", "sig", 7, ""),
]


def print_paired(rows, baseline, title=None):
    """Print the paired table, then which way each result went.

    Sign matters as much as significance: a setting can separate from the
    baseline by being reliably worse, which a "best setting" line reads as a win.
    """
    print_table(rows, PAIRED_COLUMNS, title=title)

    infeasible = sum(r["infeasible_blocks"] for r in rows)
    if infeasible:
        # An infeasible solve still returns a solution -- the last restart
        # tried, not a best-of. Averaging those in compares plans to failures.
        print(f"\n!! {infeasible} block-measurements came from solves with no "
              f"feasible solution. Those are last-restart plans, not best-of.")

    better = [r for r in rows if r["significant"] and r["diff"] < 0]
    worse = [r for r in rows if r["significant"] and r["diff"] > 0]
    flat = [r for r in rows
            if not r["significant"] and r["setting"] != baseline]

    def show(heading, group):
        if not group:
            return
        print(f"\n{heading}")
        for r in sorted(group, key=lambda r: r["diff"]):
            print(f"    {r['setting']:<20s} {r['diff']:+8.2f} h   "
                  f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]")

    show(f"Beats {baseline}:", better)
    show(f"Reliably worse than {baseline}:", worse)
    show(f"Cannot separate from {baseline}:", flat)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# (key, header, width, format spec), used by every experiment table
CORE_COLUMNS = [
    ("planned", "planned", 9, ".1f"),
    ("trips", "trips", 7, "d"),
    ("p_exceed", "P(exceed)", 11, ".1%"),
    ("e_returns", "E[ret]", 8, ".2f"),
    ("returns_per_trip", "ret/trip", 10, ".3f"),
    ("mean", "mean time", 11, ".1f"),
    # Alongside the mean, always. A mean with no error is what let three
    # contradictory buffer tables all look equally believable in August.
    ("mc_ci95", "+/-95%", 9, ".1f"),
    ("p95", "p95 time", 10, ".1f"),
]

# Add to a table that scores repair. Zero on every other strategy, so they are
# noise in tables that have none.
REPAIR_COLUMNS = [
    ("e_repairs", "E[rep]", 8, ".2f"),
    ("p_cap_hit", "cap hit", 9, ".1%"),
]

# Opt-in, for tables that have to be quotable in the report. CORE_COLUMNS gives
# the mean an interval; these give one to every other headline number, at the
# cost of a much wider table. Interval on a mean is +/- half-width; on
# P(exceed) it is a Wilson [lo, hi], because that one saturates and an
# asymmetric interval is the honest way to show it.
BALANCE_COLUMNS = [
    ("penalty", "U", 8, ".1f"),
    ("u_pct", "U/plan", 8, ".1%"),
    ("penalty_p95", "U p95", 8, ".1f"),
    ("makespan", "makespan", 10, ".1f"),
    ("max_stations", "max stn", 9, "d"),
]

INTERVAL_COLUMNS = [
    ("p_exceed_lo", "exc lo", 9, ".1%"),
    ("p_exceed_hi", "exc hi", 9, ".1%"),
    ("returns_ci95", "+/-ret", 9, ".2f"),
    ("penalty", "penalty", 10, ".1f"),
    ("penalty_ci95", "+/-pen", 9, ".1f"),
]


def print_table(rows, columns, title=None):
    """Print rows as a fixed-width table.

    columns : list of (key, header, width, fmt). fmt is a format spec without
    the colon, e.g. ".1f" or ".1%". Use "" for plain str().
    """
    if not rows:
        print("(no rows)")
        return

    width = sum(c[2] for c in columns)
    if title:
        print(f"\n{'=' * width}")
        print(title)
        print("=" * width)

    header = "".join(f"{head:>{w}}" for _, head, w, _ in columns)
    print(header)
    print("-" * width)

    for row in rows:
        line = ""
        for key, _, w, fmt in columns:
            value = row.get(key, "")
            # Format first, then pad. Building ">{w}{fmt}" breaks on specs
            # with a sign flag, since the sign has to come before the width.
            text = format(value, fmt) if (fmt and value != "") else str(value)
            line += f"{text:>{w}}"
        print(line)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def base_parser():
    """Flags shared by every experiment.

    Use as ArgumentParser(parents=[base_parser()]), then add only the flags
    specific to that experiment. Defining them once here is what stops the same
    flag meaning different things in different scripts.
    """
    import argparse

    p = argparse.ArgumentParser(add_help=False)

    problem = p.add_argument_group("problem")
    problem.add_argument("--ns", type=int, default=100,
                         help="Number of stations (default: 100)")
    problem.add_argument("--nv", type=int, default=2,
                         help="Number of vessels (default: 2)")
    problem.add_argument("--cf", type=float, default=125,
                         help="Capacity factor (default: 125)")
    problem.add_argument("--instance", type=int, default=1,
                         help="Instance number 1-30 (default: 1)")
    problem.add_argument("--instances", default=None,
                         help="Instances to pair over, e.g. 1-30 or 1,4,7. "
                              "Default: just --instance")
    problem.add_argument("--full", action="store_true",
                         help="Run on the real 581-station survey. --ns/--nv/"
                              "--cf/--instance are ignored")
    problem.add_argument("--home-ports", nargs="+", type=int, default=None,
                         help="Home port per vessel, in boat order, e.g. "
                              "--home-ports 4 6 9 11. Default: every vessel "
                              "shares the instance's home port")

    solver = p.add_argument_group("solver")
    solver.add_argument("--method", default="tabu_move", choices=list(METHODS),
                        help="Solver method (default: tabu_move)")
    solver.add_argument("--time-limit", type=float, default=10,
                        help="Solver time limit in seconds (default: 10)")
    solver.add_argument("--solver-seeds", nargs="+", type=int, default=[42],
                        help="Solver seeds to pair over. On --full this is the "
                             "only axis pairing has (default: 42)")
    solver.add_argument("--catch-source", default="historical",
                        choices=["gfsp", "heuristic", "historical"],
                        help="Catch data the solver plans against. Scenarios "
                             "always come from historical distributions, so "
                             "'gfsp' introduces a systematic bias "
                             "(default: historical)")
    solver.add_argument("--capacity-buffer", type=float, default=1.0,
                        help="Fraction of capacity the solver plans to, e.g. "
                             "0.8 leaves 20%% headroom (default: 1.0)")
    solver.add_argument("--plan", default=None, metavar="PATH",
                        help="Reuse a saved plan. Solves and writes it the "
                             "first time, loads it after, so every experiment "
                             "scores the same routes. Refuses to load a plan "
                             "solved with different parameters")

    stoch = p.add_argument_group("stochastic")
    stoch.add_argument("--n-scenarios", type=int, default=500,
                       help="Monte Carlo scenarios (default: 500)")
    stoch.add_argument("--scenario-seed", type=int, default=123,
                       help="Seed for the shared scenario set (default: 123)")
    # Choices come from the registry, so adding a strategy needs no CLI edit
    from unified.stochastic_eval import (REPAIR_PLANNERS, REPAIR_SCOPES,
                                         STRATEGIES)
    stoch.add_argument("--strategy", default="forward",
                       choices=sorted(STRATEGIES),
                       help="Overflow response (default: forward, the policy "
                            "that matches the survey -- a tow is one event)")
    stoch.add_argument("--threshold", type=float, default=0.8,
                       help="Preemptive return threshold (default: 0.8)")
    stoch.add_argument("--repair-scope", default="trip",
                       choices=list(REPAIR_SCOPES),
                       help="How much repair re-plans: the overflowed trip, or "
                            "everything that boat has left (default: trip)")
    stoch.add_argument("--repair-planner", default="nn",
                       choices=list(REPAIR_PLANNERS),
                       help="What repair re-plans with, worst to best: nn, "
                            "2opt (nn then 2-opt), solver (greedy+TSP, the "
                            "same heuristic that built the routes, ~1000x "
                            "slower). Default: nn")
    stoch.add_argument("--repair-solver-time", type=float, default=0.0,
                       help="Seconds of restarts per repair when "
                            "--repair-planner solver (default: 0, one pass)")

    return p


def describe_run(name, args, omit=(), **extra):
    """Print the banner every experiment opens with.

    omit drops shared flags an experiment does not use. monte_carlo sweeps
    --methods, so printing the inherited --method would just be misleading.
    """
    if getattr(args, "full", False):
        # ns/nv/cf/instance are ignored on the full problem.
        shared = {"problem": "full 581-station survey"}
    else:
        shared = {"ns": args.ns, "nv": args.nv, "cf": args.cf,
                  "instance": args.instance}
    shared.update({
        "method": args.method,
        "catch": args.catch_source, "scenarios": args.n_scenarios,
        "time_limit": f"{args.time_limit}s",
    })
    # Must appear, or a custom-port run looks identical to a default one.
    if getattr(args, "home_ports", None):
        shared["home_ports"] = args.home_ports
    bits = [f"{k}={v}" for k, v in shared.items() if k not in omit]
    bits += [f"{k}={v}" for k, v in extra.items()]
    print("=" * 78)
    print(name)
    print(", ".join(bits))
    print("=" * 78)
