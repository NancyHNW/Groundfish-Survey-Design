"""Which solver gives the most robust routes?

Solves the same instance with each heuristic and scores every solution against
one shared set of catch scenarios, so the comparison is about the solutions and
not the sampling. A solver that wins on deterministic time does not necessarily
win once the catch is uncertain.

    python -m experiments.monte_carlo --ns 100 --nv 2 --cf 125 --time-limit 60
"""

from experiments.common import (CORE_COLUMNS, add_baseline_delta, base_parser,
                                describe_run, make_evaluator, output_path,
                                print_table, save_csv, sweep_solve)

COLUMNS = ([("method", "method", 14, "")] + CORE_COLUMNS
           + [("vs_first", "vs first", 10, "+.1f")])

DEFAULT_METHODS = ("grasp_only", "grasp_swap", "tabu_move")


def run(methods=DEFAULT_METHODS, ns=100, nv=2, cf=125, instance=1,
        time_limit=10, catch_source="historical", strategy="backtrack",
        preemptive_threshold=0.8, n_scenarios=500, scenario_seed=123,
        capacity_buffer=1.0, histograms=True, full=False,
        home_ports=None):
    """Solve with each method, evaluate against shared scenarios, return rows."""
    from unified.stochastic_eval import plot_monte_carlo, print_stochastic_summary

    evaluator = make_evaluator(scenario_seed, n_scenarios)

    rows = []
    inst = None
    for method, det, result, row in sweep_solve(
            "method", methods, evaluator,
            strategy=strategy, preemptive_threshold=preemptive_threshold,
            ns=ns, nv=nv, cf=cf, instance=instance, time_limit=time_limit,
            catch_source=catch_source, capacity_buffer=capacity_buffer,
            full=full, home_ports=home_ports):
        rows.append(row)
        inst = det["instance"]

        print_stochastic_summary(result, deterministic_time=det["planned_time"])

        if histograms:
            plot_monte_carlo(
                result, save_path=output_path("monte-carlo-hist", method,
                                              inst, home_ports=home_ports),
                deterministic_time=det["planned_time"],
                title_suffix=f"method: {method}")

    # Re-centre on the first method listed, usually the weakest one
    add_baseline_delta(rows, "method", methods[0], "vs_first")

    print_table(rows, COLUMNS,
                title=f"HEURISTIC COMPARISON  (strategy={strategy}, "
                      f"{n_scenarios} shared scenarios)")

    best = min(rows, key=lambda r: r["mean"])
    print(f"\nLowest mean realised time: {best['method']}")

    tag = f"methods-{strategy}"
    save_csv(rows, output_path("method-comparison", tag, inst, ".csv",
                               home_ports=home_ports))

    return rows


if __name__ == "__main__":
    import argparse

    from experiments.common import METHODS

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], parents=[base_parser()])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                        choices=list(METHODS),
                        help=f"Methods to compare (default: "
                             f"{' '.join(DEFAULT_METHODS)})")
    parser.add_argument("--no-histograms", dest="histograms",
                        action="store_false",
                        help="Skip the per-method histogram figures")
    args = parser.parse_args()

    describe_run("HEURISTIC COMPARISON", args, omit=("method",),
                 strategy=args.strategy, methods=" ".join(args.methods))
    run(methods=tuple(args.methods), ns=args.ns, nv=args.nv, cf=args.cf,
        instance=args.instance, time_limit=args.time_limit,
        catch_source=args.catch_source, strategy=args.strategy,
        preemptive_threshold=args.threshold, n_scenarios=args.n_scenarios,
        scenario_seed=args.scenario_seed,
        capacity_buffer=args.capacity_buffer, histograms=args.histograms,
        full=args.full, home_ports=args.home_ports)
