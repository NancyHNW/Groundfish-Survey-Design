"""What should a boat do when its hold fills up mid-trip?

Solves once, then scores that one solution under each overflow response. The
route is fixed, so every row shares the same planned time and the differences
are purely the cost of reacting.

    python -m experiments.strategies --ns 100 --nv 2 --cf 125 --time-limit 60
"""

from experiments.common import (CORE_COLUMNS, add_baseline_delta, base_parser,
                                describe_run, make_evaluator, output_path,
                                print_table, save_csv, solve, sweep_eval)

COLUMNS = ([("case", "case", 18, "")] + CORE_COLUMNS
           + [("vs_backtrack", "vs backtr", 11, "+.1f")])

THRESHOLD_COLUMNS = ([("case", "alpha", 10, "")] + CORE_COLUMNS)

# (label, evaluate kwargs), one row each in the main table
CASES = [
    ("backtrack", {"strategy": "backtrack"}),
    ("forward", {"strategy": "forward"}),
    ("preemptive_0.8", {"strategy": "preemptive", "preemptive_threshold": 0.8}),
    ("preemptive_0.7", {"strategy": "preemptive", "preemptive_threshold": 0.7}),
]

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def run(ns=100, nv=2, cf=125, instance=1, method="tabu_move", time_limit=10,
        catch_source="historical", n_scenarios=500, scenario_seed=123,
        capacity_buffer=1.0, sweep_thresholds=True):
    """Solve once, evaluate under each strategy, return the main table rows."""
    evaluator = make_evaluator(scenario_seed, n_scenarios)

    print(f"Solving once with {method}...")
    det = solve(ns=ns, nv=nv, cf=cf, instance=instance, method=method,
                time_limit=time_limit, catch_source=catch_source,
                capacity_buffer=capacity_buffer)
    trips, inst = det["trips"], det["instance"]
    planned = det["planned_time"]
    print(f"Planned {planned:.1f}h over {len(trips)} trips "
          f"(feasible={det['feasible']})\n")

    rows = [row for _, _, row
            in sweep_eval(trips, inst, CASES, evaluator, planned)]

    # Every row shares one route, so backtrack is the natural reference
    add_baseline_delta(rows, "case", "backtrack", "vs_backtrack")

    print_table(rows, COLUMNS,
                title=f"OVERFLOW STRATEGY COMPARISON  ({n_scenarios} scenarios, "
                      f"planned {planned:.1f}h)")

    best = min(rows, key=lambda r: r["mean"])
    print(f"\nLowest mean realised time: {best['case']}")

    tag = f"{method}-strategies"
    save_csv(rows, output_path("strategy-comparison", tag, inst, ".csv"))

    if sweep_thresholds:
        threshold_cases = [
            (f"{a:.1f}", {"strategy": "preemptive", "preemptive_threshold": a})
            for a in THRESHOLDS
        ]
        sweep_rows = [row for _, _, row
                      in sweep_eval(trips, inst, threshold_cases, evaluator,
                                    planned)]

        print_table(sweep_rows, THRESHOLD_COLUMNS,
                    title="PREEMPTIVE THRESHOLD SWEEP")
        save_csv(sweep_rows,
                 output_path("preemptive-threshold", tag, inst, ".csv"))

    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], parents=[base_parser()])
    parser.add_argument("--no-threshold-sweep", dest="sweep_thresholds",
                        action="store_false",
                        help="Skip the preemptive threshold sweep")
    args = parser.parse_args()

    describe_run("OVERFLOW STRATEGY COMPARISON", args)
    run(ns=args.ns, nv=args.nv, cf=args.cf, instance=args.instance,
        method=args.method, time_limit=args.time_limit,
        catch_source=args.catch_source, n_scenarios=args.n_scenarios,
        scenario_seed=args.scenario_seed,
        capacity_buffer=args.capacity_buffer,
        sweep_thresholds=args.sweep_thresholds)
