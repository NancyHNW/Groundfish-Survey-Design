"""Bits every experiment needs: output paths, scenarios, solving, tables.

Nothing in here is an experiment itself.

base_parser() is the important one. Every experiment gets its flags from it,
so the same flag cannot end up meaning different things in different scripts.
"""

import csv
import math
import os

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


def _mc_se(std, n_simulations):
    """Standard error of a Monte Carlo mean.

    The evaluator reports a population sd over n scenarios (np.std, ddof=0), so
    the sample-sd standard error is std/sqrt(n-1) rather than std/sqrt(n).

    This is scenario-sampling error only. It says nothing about solver restart
    noise, which is the larger term whenever two rows come from different
    solves -- see the note on plot_sweep_grid.
    """
    n = n_simulations or 0
    return std / math.sqrt(n - 1) if n > 1 else 0.0


def _returns_error(result, e_returns, n_trips):
    """Monte Carlo error on E[unscheduled returns], absolute and per trip.

    Needs the per-scenario counts, which only a real evaluator result carries.
    Returns an empty dict when they are absent so a stubbed result still works.
    """
    per_scenario = result.get("all_results")
    if not per_scenario:
        return {}
    counts = [r["n_unscheduled_returns"] for r in per_scenario]
    n = len(counts)
    if n < 2:
        return {}
    var = sum((c - e_returns) ** 2 for c in counts) / n
    se = _mc_se(math.sqrt(var), n)
    return {
        "returns_se": se,
        "returns_ci95": 1.96 * se,
        "returns_per_trip_ci95": 1.96 * se / n_trips if n_trips else 0.0,
    }


def summarise(result, planned, n_trips, feasible=None, **extra):
    """Turn an evaluator result into one table row.

    Every experiment reports the same core metrics. The only difference is the
    column naming the row (buffer, strategy, method), passed in through extra
    so it lands first in the dict.

    Parameters
    ----------
    result : dict, output of StochasticEvaluator.evaluate
    planned : float, deterministic planned time (the floor of the distribution)
    n_trips : int, number of planned trips
    feasible : bool or None, feasibility against the true capacity
    **extra : the column naming this row, e.g. buffer=0.8
    """
    td = result["total_time_distribution"]
    e_returns = result["expected_unscheduled_returns"]
    row = dict(extra)
    row.update({
        "planned": planned,
        "trips": n_trips,
        "p_exceed": result["p_capacity_exceedance"],
        "e_returns": e_returns,
        # Per trip, not absolute. A tighter buffer plans more trips, so the
        # raw count can drop just by spreading the same risk more thinly.
        "returns_per_trip": (e_returns / n_trips if n_trips else 0.0),
        "mean": td["mean"],
        "sd": td["std"],
        "mc_se": _mc_se(td["std"], result.get("n_simulations")),
        "mc_ci95": 1.96 * _mc_se(td["std"], result.get("n_simulations")),
        "p5": td["p5"],
        "p95": td["p95"],
    })
    row.update(_returns_error(result, e_returns, n_trips))
    if feasible is not None:
        row["feasible"] = feasible
    return row


def sweep_solve(param, values, evaluator, key=None, fmt=None,
                strategy="backtrack", preemptive_threshold=0.8, **fixed):
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
            preemptive_threshold=preemptive_threshold)
        row = summarise(result, det["planned_time"], len(det["trips"]),
                        feasible=det["feasible"], **{key: value})
        yield value, det, result, row


def sweep_eval(trips, inst, cases, evaluator, planned, key="case"):
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
        result = evaluator.evaluate(trips, inst, **kwargs)
        row = summarise(result, planned, len(trips), **{key: label})
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

# Evaluator kwargs; everything else in a setting goes to solve(). Settings
# differing only in these share one solve -- the strategy is picked at sea
# against a fixed plan, so re-solving per strategy would break the pairing.
EVAL_KEYS = ("strategy", "preemptive_threshold", "max_repairs")


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
                result = evaluator.evaluate(det["trips"], det["instance"],
                                            **eval_kw)
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

    stoch = p.add_argument_group("stochastic")
    stoch.add_argument("--n-scenarios", type=int, default=500,
                       help="Monte Carlo scenarios (default: 500)")
    stoch.add_argument("--scenario-seed", type=int, default=123,
                       help="Seed for the shared scenario set (default: 123)")
    # Choices come from the registry, so adding a strategy needs no CLI edit
    from unified.stochastic_eval import STRATEGIES
    stoch.add_argument("--strategy", default="backtrack",
                       choices=sorted(STRATEGIES),
                       help="Overflow response (default: backtrack)")
    stoch.add_argument("--threshold", type=float, default=0.8,
                       help="Preemptive return threshold (default: 0.8)")

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
