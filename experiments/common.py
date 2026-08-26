"""Bits every experiment needs: output paths, scenarios, solving, tables.

Nothing in here is an experiment itself.

base_parser() is the important one. Every experiment gets its flags from it,
so the same flag cannot end up meaning different things in different scripts.
"""

import csv
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

def output_path(name, tag, inst=None, ext=".png"):
    """Build a path under tests/outputs/ from experiment, tag and problem size.

    e.g. output_path("buffer-comparison", "tabu_move-backtrack", inst, ".csv")
      -> tests/outputs/buffer-comparison_tabu_move-backtrack_ns100-nv2.csv

    Folder is gitignored, these all get regenerated.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parts = [name, tag]
    if inst is not None:
        parts.append(f"ns{inst.ns}-nv{inst.n_boats}")
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
    row = dict(extra)
    row.update({
        "planned": planned,
        "trips": n_trips,
        "p_exceed": result["p_capacity_exceedance"],
        "e_returns": result["expected_unscheduled_returns"],
        # Per trip, not absolute. A tighter buffer plans more trips, so the
        # raw count can drop just by spreading the same risk more thinly.
        "returns_per_trip": (result["expected_unscheduled_returns"] / n_trips
                             if n_trips else 0.0),
        "mean": td["mean"],
        "p5": td["p5"],
        "p95": td["p95"],
    })
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

    solver = p.add_argument_group("solver")
    solver.add_argument("--method", default="tabu_move", choices=list(METHODS),
                        help="Solver method (default: tabu_move)")
    solver.add_argument("--time-limit", type=float, default=10,
                        help="Solver time limit in seconds (default: 10)")
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
    shared = {
        "ns": args.ns, "nv": args.nv, "cf": args.cf,
        "instance": args.instance, "method": args.method,
        "catch": args.catch_source, "scenarios": args.n_scenarios,
        "time_limit": f"{args.time_limit}s",
    }
    bits = [f"{k}={v}" for k, v in shared.items() if k not in omit]
    bits += [f"{k}={v}" for k, v in extra.items()]
    print("=" * 78)
    print(name)
    print(", ".join(bits))
    print("=" * 78)
