"""Which solver gives the most robust routes?

Solves the same instance with each heuristic and scores every solution against
one shared set of catch scenarios, so the comparison is about the solutions and
not the sampling. A solver that wins on deterministic time does not necessarily
win once the catch is uncertain.

    python -m experiments.monte_carlo --ns 100 --nv 2 --cf 125 --time-limit 60
    python -m experiments.monte_carlo --all-methods --full --time-limit 300 \
        --n-scenarios 1000 --strategy forward --no-histograms

Unlike `strategies`, every row here comes from its **own solve** -- the method
is what is being swept, and the method is what builds the routes. So the
differences between rows carry solver noise as well as the catch, and the
+/-95% column, which is scenario sampling only, understates them. Two defences,
both needed:

  * the paired-across-scenarios table, which prices the difference between two
    fixed plans far more precisely than two overlapping per-row error bars can;
  * `experiments.variance`, which is what establishes that the solver is
    effectively deterministic at these budgets and so that the pairing above is
    measuring the methods rather than the restarts. Run it before quoting any
    of this.
"""

import time

from experiments.common import (CORE_COLUMNS, INTERVAL_COLUMNS, REPAIR_COLUMNS,
                                add_baseline_delta, base_parser, describe_run,
                                make_evaluator, output_path, paired_scenarios,
                                print_paired, print_table, save_csv,
                                strategy_tag, sweep_solve)

# Every planner the solver exposes, weakest first, so `vs first` reads as the
# gain over construction alone and the baseline of the paired table is the
# method nobody would actually ship.
ALL_METHODS = ("grasp_only", "grasp_swap", "tabu_swap", "tabu_move",
               "tabu_combined", "sa")

DEFAULT_METHODS = ("grasp_only", "grasp_swap", "tabu_move")

# solve_s is wall clock, not vessel-hours. It belongs in this table and in no
# other: --time-limit caps the restart loop, so a method coming in well under
# the cap has run out of things to try rather than been cut off, and that is a
# result about the method.
def _columns(strategy):
    """Table columns. Wide already, so the repair pair is earned, not default.

    Six methods crossed with an interval on every metric runs to 150 characters
    before E[rep] and cap hit, which are identically zero under every strategy
    but repair.
    """
    return ([("method", "method", 15, ""), ("solve_s", "solve s", 9, ".0f")]
            + CORE_COLUMNS + INTERVAL_COLUMNS
            + (REPAIR_COLUMNS if strategy == "repair" else [])
            + [("feasible", "feas", 7, ""),
               ("vs_first", "vs first", 10, "+.1f")])


def _figure_context(method, strategy, inst, capacity_buffer):
    """What the figure needs to identify itself once it is a file on disk."""
    size = ("full 581-station survey" if inst.ns > 500
            else f"ns{inst.ns}, nv{inst.n_boats}")
    bits = [size, f"{method}", f"{strategy}"]
    if capacity_buffer != 1.0:
        bits.append(f"buffer {capacity_buffer:.0%}")
    return ", ".join(bits)


def run(methods=DEFAULT_METHODS, ns=100, nv=2, cf=125, instance=1,
        time_limit=10, catch_source="historical", strategy="backtrack",
        preemptive_threshold=0.8, repair_scope="trip", repair_planner="nn",
        repair_solver_time=0.0, n_scenarios=500, scenario_seed=123,
        capacity_buffer=1.0, histograms=True, full=False,
        home_ports=None, baseline=None, out_path=None):
    """Solve with each method, evaluate against shared scenarios, return rows."""
    from unified.stochastic_eval import plot_monte_carlo, print_stochastic_summary

    t0 = time.time()
    evaluator = make_evaluator(scenario_seed, n_scenarios)
    print(f"Scenario set ready ({n_scenarios} scenarios, "
          f"seed {scenario_seed}) at {time.time() - t0:.0f}s\n", flush=True)

    baseline = baseline or methods[0]
    if baseline not in methods:
        raise ValueError(f"baseline {baseline!r} is not one of {list(methods)}")

    rows, times = [], {}
    inst = None
    for method, det, result, row in sweep_solve(
            "method", methods, evaluator,
            strategy=strategy, preemptive_threshold=preemptive_threshold,
            repair_scope=repair_scope,
            repair_planner=repair_planner,
            repair_solver_time=repair_solver_time,
            ns=ns, nv=nv, cf=cf, instance=instance, time_limit=time_limit,
            catch_source=catch_source, capacity_buffer=capacity_buffer,
            full=full, home_ports=home_ports):
        # Straight off the solver, so a method that finished early is visible
        row["solve_s"] = det.get("elapsed", 0.0)
        row["restarts"] = det.get("iterations", 0)
        rows.append(row)
        times[method] = [r["total_time"] for r in result["all_results"]]
        inst = det["instance"]

        print_stochastic_summary(result, deterministic_time=det["planned_time"])

        if histograms:
            plot_monte_carlo(
                result, save_path=output_path("monte-carlo-hist", method,
                                              inst, home_ports=home_ports),
                deterministic_time=det["planned_time"],
                title_suffix=_figure_context(method, strategy, inst,
                                             capacity_buffer))
        # Written every method: six solves at 300s is half an hour, and a crash
        # on the last one should not cost the first five.
        if out_path:
            save_csv(add_baseline_delta(rows, "method", methods[0], "vs_first"),
                     out_path)

    # Re-centre on the first method listed, usually the weakest one
    add_baseline_delta(rows, "method", methods[0], "vs_first")

    print_table(rows, _columns(strategy),
                title=f"HEURISTIC COMPARISON  (strategy={strategy}, "
                      f"{n_scenarios} shared scenarios, {time_limit}s per solve)")

    infeasible = [r["method"] for r in rows if r.get("feasible") is False]
    if infeasible:
        # Same trap as the paired harness: an infeasible solve hands back the
        # last restart tried, not a best-of, so its row is a failure and not a
        # plan. Quoting it beside the others compares unlike things.
        print(f"\n!! no feasible solution from: {', '.join(infeasible)}. "
              f"Those rows are last-restart plans -- do not quote them as "
              f"method results.")

    # Each method brought its own routes, so the realised times differ for two
    # reasons at once. Pairing on scenarios removes only the catch: a heavy
    # draw slows every plan, and that shared bad luck cancels on subtraction.
    # What is left is the difference between the plans, which is the question.
    paired_rows = paired_scenarios(times, baseline)
    print_paired(paired_rows, baseline,
                 title=f"PAIRED ACROSS SCENARIOS  ({n_scenarios} scenarios, "
                       f"one solve per method, vs {baseline})")
    print("\n  Pairs on the catch draw only, so the interval prices the gap "
          "between\n  these particular plans. It does not include solver "
          "restart noise --\n  see experiments.variance for whether that term "
          "matters at this budget.")

    best = min(rows, key=lambda r: r["mean"])
    print(f"\nLowest mean realised time: {best['method']} "
          f"({best['mean']:.1f} h +/- {best['mc_ci95']:.1f})")

    tag = f"methods-{strategy_tag(strategy, repair_scope, repair_planner)}"
    path = out_path or output_path("method-comparison", tag, inst, ".csv",
                                   home_ports=home_ports)
    save_csv(rows, path)
    save_csv(paired_rows, path.replace(".csv", "_paired.csv"))

    print(f"\nTotal {time.time() - t0:.0f}s")
    return rows


if __name__ == "__main__":
    import argparse
    import os
    from datetime import datetime

    from experiments.common import METHODS, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], parents=[base_parser()])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                        choices=list(METHODS),
                        help=f"Methods to compare (default: "
                             f"{' '.join(DEFAULT_METHODS)})")
    parser.add_argument("--all-methods", action="store_true",
                        help=f"Compare all six planners: "
                             f"{' '.join(ALL_METHODS)}. Overrides --methods")
    parser.add_argument("--baseline", default=None, choices=list(METHODS),
                        help="Method every paired difference is taken against "
                             "(default: the first one compared)")
    parser.add_argument("--no-histograms", dest="histograms",
                        action="store_false",
                        help="Skip the per-method histogram figures")
    args = parser.parse_args()

    methods = list(ALL_METHODS) if args.all_methods else args.methods
    if args.baseline and args.baseline not in methods:
        parser.error(f"--baseline {args.baseline} is not in --methods "
                     f"({' '.join(methods)})")

    describe_run("HEURISTIC COMPARISON", args, omit=("method",),
                 strategy=args.strategy, methods=" ".join(methods))
    print(f"{len(methods)} solves, rough estimate "
          f"{len(methods) * (args.time_limit + 15) / 60:.0f} min plus ~2 min "
          f"to fit the scenarios")

    # Timestamped on the full problem. tests/outputs/ already mixes results
    # from two different overflow models and only the dated filenames can be
    # told apart; a method table is going straight into the report, so it gets
    # a date.
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    size = "full" if args.full else f"ns{args.ns}-nv{args.nv}-cf{args.cf:g}"
    tag = strategy_tag(args.strategy, args.repair_scope, args.repair_planner)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(
        OUTPUT_DIR,
        f"method-comparison_{tag}_{size}_sc{args.n_scenarios}_{stamp}.csv")
    print(f"writing to {os.path.basename(out)}")

    run(methods=tuple(methods), ns=args.ns, nv=args.nv, cf=args.cf,
        instance=args.instance, time_limit=args.time_limit,
        catch_source=args.catch_source, strategy=args.strategy,
        preemptive_threshold=args.threshold,
        repair_scope=args.repair_scope, repair_planner=args.repair_planner,
        repair_solver_time=args.repair_solver_time,
        n_scenarios=args.n_scenarios,
        scenario_seed=args.scenario_seed,
        capacity_buffer=args.capacity_buffer, histograms=args.histograms,
        full=args.full, home_ports=args.home_ports,
        baseline=args.baseline, out_path=out)
