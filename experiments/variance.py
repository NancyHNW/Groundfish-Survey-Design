"""Compare solver variation with catch variation.

Each route is scored on the same scenario set. Between-route variation
measures solver noise; within-route variation measures catch noise.
"""

import math
import os

from experiments.common import (OUTPUT_DIR, base_parser, describe_run,
                                make_evaluator, output_path, print_table,
                                save_csv, solve)

COLUMNS = [
    ("seed", "seed", 7, "d"),
    ("planned", "planned", 10, ".1f"),
    ("trips", "trips", 7, "d"),
    ("feasible", "feas", 7, ""),
    ("mean", "mean", 10, ".1f"),
    ("sd", "sd within", 11, ".2f"),
    ("p_exceed", "P(exceed)", 11, ".1%"),
    ("e_returns", "E[ret]", 8, ".2f"),
]


def plot_planned_vs_uncertainty(csv_path, save_path):
    """Plot planned time against the uncertainty cost for each route."""
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        import csv
        rows = list(csv.DictReader(handle))

    planned = np.array([float(row["planned"]) for row in rows])
    uncertainty = np.array([
        float(row["mean"]) - float(row["planned"]) for row in rows
    ])
    correlation = float(np.corrcoef(planned, uncertainty)[0, 1])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(planned, uncertainty, s=58, color="#2c7fb8",
               edgecolors="white", linewidths=0.8, zorder=3)
    for row, x, y in zip(rows, planned, uncertainty):
        ax.annotate(f"seed {row['seed']}", (x, y),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)

    slope, intercept = np.polyfit(planned, uncertainty, 1)
    xline = np.array([planned.min(), planned.max()])
    ax.plot(xline, intercept + slope * xline, color="#444444",
            linestyle="--", linewidth=1.1,
            label=f"linear trend  r = {correlation:.2f}")
    ax.set_xlabel("Planned route time (h)")
    ax.set_ylabel("Cost of uncertainty, realised − planned (h)")
    ax.set_title("Planned time versus cost of uncertainty")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Plot saved to {save_path}")
    return correlation


def run(ns=100, nv=2, cf=125, instance=1, method="tabu_move", time_limit=60,
        catch_source="historical", capacity_buffer=1.0, strategy="forward",
        repair_scope="trip", repair_planner="nn", n_scenarios=500,
        scenario_seed=123, solver_seeds=(1, 2, 3, 4, 5), full=False,
        home_ports=None, out_path=None):
    """Solve once per seed, score each on one shared scenario set."""
    import numpy as np

    evaluator = make_evaluator(scenario_seed, n_scenarios)
    rows, per_route = [], []

    for i, seed in enumerate(solver_seeds, 1):
        det = solve(ns=ns, nv=nv, cf=cf, instance=instance, method=method,
                    time_limit=time_limit, catch_source=catch_source,
                    capacity_buffer=capacity_buffer, full=full,
                    home_ports=home_ports, seed=seed, verbose=False)
        result = evaluator.evaluate(
            det["trips"], det["instance"], strategy=strategy,
            repair_scope=repair_scope, repair_planner=repair_planner,
            planned_catch=det.get("planned_catch"))

        times = np.array([r["total_time"] for r in result["all_results"]])
        rows.append({
            "seed": seed,
            "planned": det["planned_time"],
            "trips": len(det["trips"]),
            "feasible": det["feasible"],
            "mean": float(times.mean()),
            "sd": float(times.std(ddof=1)),
            "p_exceed": result["p_capacity_exceedance"],
            "e_returns": result["expected_unscheduled_returns"],
        })
        # Two seeds can converge on the same solution. If they do, the
        # between-route term is measuring fewer routes than it looks.
        fingerprint = tuple(sorted(tuple(t["nodes"]) for t in det["trips"]))
        per_route.append({"seed": seed, "feasible": det["feasible"],
                          "planned": det["planned_time"], "times": times,
                          "route": fingerprint})
        print(f"  [{i}/{len(solver_seeds)}] seed {seed}: "
              f"planned {det['planned_time']:.1f}h, "
              f"mean {times.mean():.1f}h, feasible={det['feasible']}",
              flush=True)

    print_table(rows, COLUMNS,
                title=f"ONE ROUTE PER SEED  ({n_scenarios} shared scenarios)")
    _decompose(per_route, n_scenarios)

    if out_path:
        save_csv(rows, out_path)
        plot_planned_vs_uncertainty(
            out_path, os.path.splitext(out_path)[0] + ".png")
    return rows


def _decompose(per_route, n_scenarios):
    """Split realised-time variance into solver and catch components."""
    import numpy as np

    # An infeasible solve returns the last restart tried, not a best-of.
    # Leaving those in would charge the solver term with failures rather than
    # with genuine route-to-route variation.
    dropped = [r["seed"] for r in per_route if not r["feasible"]]
    usable = [r for r in per_route if r["feasible"]]
    if dropped:
        print(f"\n!! dropped {len(dropped)} infeasible solve(s): {dropped}")
    if len(usable) < 2:
        print("\nNeed at least two feasible solves to decompose. Stopping.")
        return

    distinct = len({r["route"] for r in usable})
    if distinct < len(usable):
        print(f"\n!! {len(usable)} solves found only {distinct} distinct "
              f"routes. The solver term is averaging over {distinct} of them,"
              f"\n   so it understates the true spread between routes.")

    means = np.array([r["times"].mean() for r in usable])
    within = np.array([r["times"].var(ddof=1) for r in usable])
    planned = np.array([r["planned"] for r in usable])

    between_var = means.var(ddof=1)          # spread of per-route means
    within_var = within.mean()               # mean spread within a route
    total = between_var + within_var

    sd = math.sqrt
    print(f"\n{'=' * 72}\nVARIANCE DECOMPOSITION  "
          f"({len(usable)} routes x {n_scenarios} shared scenarios)\n"
          f"{'=' * 72}")
    print(f"  {'source':<34} {'sd (h)':>10} {'share of var':>14}")
    print("  " + "-" * 60)
    print(f"  {'solver  (between routes)':<34} {sd(between_var):>10.2f} "
          f"{between_var / total:>13.1%}")
    print(f"  {'catch   (within a route)':<34} {sd(within_var):>10.2f} "
          f"{within_var / total:>13.1%}")
    print(f"  {'total':<34} {sd(total):>10.2f} {1.0:>13.1%}")

    ratio = sd(between_var) / sd(within_var) if within_var else float("inf")
    print(f"\n  solver sd is {ratio:.2f}x the catch sd")

    # Route quality and route robustness are different things, and the
    # realised decomposition mixes them. Planned time isolates the first.
    print(f"\n  Planned time across routes: sd {planned.std(ddof=1):.2f} h, "
          f"range {planned.max() - planned.min():.2f} h")
    print(f"  Realised means across routes: sd {sd(between_var):.2f} h")
    print("  The first is pure routing quality. The gap between them is how"
          "\n  differently those routes respond to the same catch.")

    if ratio > 1:
        print(f"\n  Solver noise dominates. A single-solve comparison of two"
              f"\n  settings is measuring the solver unless the effect is"
              f"\n  well over {2 * sd(between_var):.1f} h. Pair across seeds.")
    else:
        print(f"\n  Catch noise dominates. Single-solve comparisons are"
              f"\n  defensible, and more scenarios buy more than more seeds.")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], parents=[base_parser()])
    args = parser.parse_args()

    seeds = args.solver_seeds
    if len(seeds) < 2:
        parser.error("need at least two --solver-seeds to separate the two "
                     "sources; 20 is a reasonable default")

    describe_run("SOLVER VS CATCH VARIABILITY", args, strategy=args.strategy,
                 seeds=len(seeds))

    size = "full" if args.full else f"ns{args.ns}-nv{args.nv}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Seeds in the filename, same reason as experiments/paired.py: the run
    # can be split and the per-seed rows combined afterwards.
    seedtag = ("_s" + "-".join(str(s) for s in args.solver_seeds)
               if len(args.solver_seeds) <= 5 else "")
    out = os.path.join(OUTPUT_DIR,
                       f"variance_{args.method}_{size}{seedtag}.csv")

    run(ns=args.ns, nv=args.nv, cf=args.cf, instance=args.instance,
        method=args.method, time_limit=args.time_limit,
        catch_source=args.catch_source,
        capacity_buffer=args.capacity_buffer, strategy=args.strategy,
        repair_scope=args.repair_scope, repair_planner=args.repair_planner,
        n_scenarios=args.n_scenarios, scenario_seed=args.scenario_seed,
        solver_seeds=seeds, full=args.full, home_ports=args.home_ports,
        out_path=out)
