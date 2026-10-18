"""What should a boat do when its hold fills up mid-trip?

Solves once, then scores that one solution under each overflow response. The
route is fixed, so every row shares the same planned time and the differences
are purely the cost of reacting.

    python -m experiments.strategies --ns 100 --nv 2 --cf 125 --time-limit 60
"""

from experiments.common import (CORE_COLUMNS, REPAIR_COLUMNS,
                                add_baseline_delta, base_parser,
                                describe_run, make_evaluator, output_path,
                                print_table, save_csv, solve_cached,
                                sweep_eval)


def plot_threshold_sweep(csv_path, save_path, forward_mean=None):
    """Plot realised time and returns against the pre-emptive threshold."""
    import csv
    import matplotlib.pyplot as plt

    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["case"]))

    alpha = [float(row["case"]) for row in rows]
    realised = [float(row["mean"]) for row in rows]
    returns = [float(row["e_returns"]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(7.4, 5.0))
    ax2 = ax1.twinx()
    time_line, = ax1.plot(alpha, realised, marker="o", linewidth=2,
                          color="#2c7fb8", label="Mean realised time")
    return_line, = ax2.plot(alpha, returns, marker="s", linewidth=2,
                            color="#d95f02", label="Expected returns")
    lines = [time_line, return_line]
    if forward_mean is not None:
        baseline = ax1.axhline(
            forward_mean, color="#555555", linestyle="--", linewidth=1.1,
            label=f"Forward ({forward_mean:.1f} h)")
        lines.append(baseline)

    ax1.set_xlabel("Pre-emptive threshold, α")
    ax1.set_ylabel("Mean realised time (h)", color="#2c7fb8")
    ax2.set_ylabel("Expected unscheduled returns", color="#d95f02")
    ax1.set_xticks(alpha)
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.set_title("Pre-emptive threshold trade-off")
    ax1.legend(lines, [line.get_label() for line in lines],
               frameon=False, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


# Width 20 fits the longest label, repair_trip_solver
COLUMNS = ([("case", "case", 20, "")] + CORE_COLUMNS + REPAIR_COLUMNS
           + [("vs_forward", "vs fwd", 11, "+.1f")])

THRESHOLD_COLUMNS = ([("case", "alpha", 10, "")] + CORE_COLUMNS)

# (label, evaluate kwargs), one row each in the main table
CASES = [
    ("backtrack", {"strategy": "backtrack"}),
    ("forward", {"strategy": "forward"}),
    ("preemptive_0.8", {"strategy": "preemptive", "preemptive_threshold": 0.8}),
    ("preemptive_0.7", {"strategy": "preemptive", "preemptive_threshold": 0.7}),
    # 0.9 is the best threshold the sweep found; without it the main table
    # shows preemptive at ~14h worse than its own best setting.
    ("preemptive_0.9", {"strategy": "preemptive", "preemptive_threshold": 0.9}),
    ("repair_trip", {"strategy": "repair", "repair_scope": "trip"}),
    ("repair_boat", {"strategy": "repair", "repair_scope": "boat"}),
    # keep_order keeps the planner's station order and re-decides only the
    # trip breaks, so vs nn it prices the routing thrown away.
    ("repair_trip_keep", {"strategy": "repair", "repair_scope": "trip",
                          "repair_planner": "keep_order"}),
    ("repair_boat_keep", {"strategy": "repair", "repair_scope": "boat",
                          "repair_planner": "keep_order"}),
    ("repair_trip_2opt", {"strategy": "repair", "repair_scope": "trip",
                          "repair_planner": "2opt"}),
    ("repair_boat_2opt", {"strategy": "repair", "repair_scope": "boat",
                          "repair_planner": "2opt"}),
    # Much slower than nearest neighbour; use only on small problems.
    ("repair_trip_solver", {"strategy": "repair", "repair_scope": "trip",
                            "repair_planner": "solver"}),
    ("repair_boat_solver", {"strategy": "repair", "repair_scope": "boat",
                            "repair_planner": "solver"}),
    # Tabu can move stations between the boat's trips after an overflow.
    ("repair_trip_tabu", {"strategy": "repair", "repair_scope": "trip",
                          "repair_planner": "tabu"}),
    ("repair_boat_tabu", {"strategy": "repair", "repair_scope": "boat",
                          "repair_planner": "tabu"}),
]

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def run(ns=100, nv=2, cf=125, instance=1, method="tabu_move", time_limit=10,
        catch_source="historical", n_scenarios=500, scenario_seed=123,
        capacity_buffer=1.0, sweep_thresholds=True, full=False,
        home_ports=None, repair_solver_time=0.0, plan=None,
        solver_seed=42):
    """Solve once, evaluate under each strategy, return the main table rows."""
    evaluator = make_evaluator(scenario_seed, n_scenarios)

    cases = CASES
    if repair_solver_time:
        # Per repair, not per run. Only the solver rows read it.
        cases = [(label, dict(kw, repair_solver_time=repair_solver_time)
                  if kw.get("repair_planner") == "solver" else kw)
                 for label, kw in CASES]

    print(f"Solving once with {method}...")
    det = solve_cached(plan, ns=ns, nv=nv, cf=cf, instance=instance,
                       method=method, time_limit=time_limit,
                       catch_source=catch_source,
                       capacity_buffer=capacity_buffer, full=full,
                       home_ports=home_ports, seed=solver_seed)
    trips, inst = det["trips"], det["instance"]
    planned = det["planned_time"]
    print(f"Planned {planned:.1f}h over {len(trips)} trips "
          f"(feasible={det['feasible']})\n")

    rows = [row for _, _, row
            in sweep_eval(trips, inst, cases, evaluator, planned,
                          planned_catch=det.get("planned_catch"))]

    # Forward is the operational reference policy.
    add_baseline_delta(rows, "case", "forward", "vs_forward")

    print_table(rows, COLUMNS,
                title=f"OVERFLOW STRATEGY COMPARISON  ({n_scenarios} scenarios, "
                      f"planned {planned:.1f}h)")

    best = min(rows, key=lambda r: r["mean"])
    print(f"\nLowest mean realised time: {best['case']}")

    tag = f"{method}-strategies"
    save_csv(rows, output_path("strategy-comparison", tag, inst, ".csv",
                               home_ports=home_ports))

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
                 output_path("preemptive-threshold", tag, inst, ".csv",
                             home_ports=home_ports))

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
        sweep_thresholds=args.sweep_thresholds, full=args.full,
        home_ports=args.home_ports,
        repair_solver_time=args.repair_solver_time,
        plan=args.plan, solver_seed=args.solver_seeds[0])
