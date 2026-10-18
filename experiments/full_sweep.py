"""Every buffer crossed with every overflow strategy, on the full problem.

The buffer decides the routes, so each buffer needs its own solve. The strategy
is chosen at sea once the plan is fixed, so every strategy scores the same
solution. That means the whole grid costs one solve per buffer, not one per
cell -- 4 solves for 36 rows rather than 36.

Everything shares one scenario set, so differences between cells come from the
plans and the responses rather than from resampling the catch.

    python -m experiments.full_sweep --full --time-limit 300 --n-scenarios 1000

Long runs: results are written after every buffer, so a crash or a locked file
late on does not cost the whole sweep.
"""

import os
import time
from datetime import datetime

from experiments.common import (CORE_COLUMNS, OUTPUT_DIR, PAIRED_COLUMNS,
                                REPAIR_COLUMNS, base_parser,
                                make_evaluator, paired_scenarios,
                                print_paired, print_table, save_csv,
                                summarise, solve)

BUFFERS = (0.7, 0.8, 0.9, 1.0)

# (label, evaluate kwargs). Same set as experiments.strategies.
STRATEGIES = [
    ("backtrack", {"strategy": "backtrack"}),
    ("forward", {"strategy": "forward"}),
    ("preemptive_0.8", {"strategy": "preemptive", "preemptive_threshold": 0.8}),
    ("preemptive_0.7", {"strategy": "preemptive", "preemptive_threshold": 0.7}),
    ("preemptive_0.9", {"strategy": "preemptive", "preemptive_threshold": 0.9}),
    ("repair_trip", {"strategy": "repair", "repair_scope": "trip"}),
    ("repair_boat", {"strategy": "repair", "repair_scope": "boat"}),
    ("repair_trip_keep", {"strategy": "repair", "repair_scope": "trip",
                          "repair_planner": "keep_order"}),
    ("repair_boat_keep", {"strategy": "repair", "repair_scope": "boat",
                          "repair_planner": "keep_order"}),
    ("repair_trip_2opt", {"strategy": "repair", "repair_scope": "trip",
                          "repair_planner": "2opt"}),
    ("repair_boat_2opt", {"strategy": "repair", "repair_scope": "boat",
                          "repair_planner": "2opt"}),
    # The solver re-planner adds no solves, but a lot of scoring: it re-plans
    # ~150 stations per repair under boat scope. Budget ~2.5 h on top of a
    # 1000-scenario four-buffer run. --no-solver-repair drops both.
    ("repair_trip_solver", {"strategy": "repair", "repair_scope": "trip",
                            "repair_planner": "solver"}),
    ("repair_boat_solver", {"strategy": "repair", "repair_scope": "boat",
                            "repair_planner": "solver"}),
]

# Width 20 fits the longest label, repair_trip_2opt
COLUMNS = ([("buffer", "buffer", 8, ".0%"), ("case", "strategy", 20, "")]
           + CORE_COLUMNS + REPAIR_COLUMNS
           + [("feasible", "feas", 7, ""),
              ("vs_baseline", "vs 100%/fwd", 12, "+.1f")])


def run(time_limit=300, n_scenarios=1000, scenario_seed=123,
        catch_source="historical", method="tabu_move", buffers=BUFFERS,
        full=True, ns=100, nv=2, cf=125, instance=1, home_ports=None,
        out_path=None, plot=True, strategies=None, baseline="forward",
        repair_solver_time=0.0, buffer_ref="forward"):
    """Solve at each buffer, score each solution under each strategy.

    Returns the full grid of rows. Writes the CSV after every buffer.
    """
    from unified.stochastic_eval import plot_sweep_grid

    t0 = time.time()
    evaluator = make_evaluator(scenario_seed, n_scenarios)
    print(f"Scenario set ready ({n_scenarios} scenarios, "
          f"seed {scenario_seed}) at {time.time() - t0:.0f}s\n")

    strategies = strategies or STRATEGIES
    if repair_solver_time:
        # Per repair, not per run. Only the solver rows read it.
        strategies = [(label, dict(kw, repair_solver_time=repair_solver_time)
                       if kw.get("repair_planner") == "solver" else kw)
                      for label, kw in strategies]
    rows, paired_rows = [], []
    # Kept across the whole loop so the buffers can be paired against each
    # other afterwards, not just the strategies within a buffer.
    by_buffer = {}
    for i, buffer in enumerate(buffers, 1):
        print(f"{'=' * 70}\n[{i}/{len(buffers)}] buffer {buffer:.0%}  "
              f"-- solving, {time_limit}s limit\n{'=' * 70}", flush=True)

        t1 = time.time()
        det = solve(ns=ns, nv=nv, cf=cf, instance=instance, method=method,
                    time_limit=time_limit, catch_source=catch_source,
                    capacity_buffer=buffer, full=full, home_ports=home_ports,
                    verbose=False)
        print(f"  solved in {time.time() - t1:.0f}s: "
              f"planned {det['planned_time']:.1f}h over {len(det['trips'])} "
              f"trips, feasible={det['feasible']}", flush=True)

        times = {}
        by_buffer[buffer] = times
        for label, kwargs in strategies:
            result = evaluator.evaluate(det["trips"], det["instance"],
                                        planned_catch=det.get("planned_catch"),
                                        **kwargs)
            times[label] = [r["total_time"] for r in result["all_results"]]
            row = summarise(result, det["planned_time"], len(det["trips"]),
                            feasible=det["feasible"], trips=det["trips"],
                            buffer=buffer, case=label)
            rows.append(row)
            print(f"    {label:20s} mean {row['mean']:8.1f}h  "
                  f"p95 {row['p95']:8.1f}h  "
                  f"ret/trip {row['returns_per_trip']:.3f}", flush=True)

        # The solve is fixed within a buffer, so scenarios are the only
        # randomness left and pairing on them is the right comparison. Two
        # overlapping per-strategy error bars cannot answer this.
        for r in paired_scenarios(times, baseline):
            r["buffer"] = buffer
            paired_rows.append(r)
        print_paired([r for r in paired_rows if r["buffer"] == buffer],
                     baseline,
                     title=f"PAIRED ACROSS SCENARIOS  (buffer {buffer:.0%}, "
                           f"{n_scenarios} scenarios, one solve)")

        # Written every buffer, so a crash late on does not lose the earlier work
        _finalise(rows, baseline)
        save_csv(rows, out_path)
        print(f"  elapsed {time.time() - t0:.0f}s\n", flush=True)

    _finalise(rows, baseline)
    print_table(rows, COLUMNS,
                title=f"FULL SWEEP  ({n_scenarios} scenarios, {method}, "
                      f"{time_limit}s per solve)")
    save_csv(rows, out_path)
    if out_path:
        save_csv(paired_rows, out_path.replace(".csv", "_paired.csv"))

    buffer_rows = _paired_across_buffers(by_buffer, strategies, buffer_ref,
                                         n_scenarios)
    if buffer_rows and out_path:
        save_csv(buffer_rows, out_path.replace(".csv", "_buffers_paired.csv"))

    if plot and len(buffers) > 1:
        size = "full 581-station" if full else f"ns{ns}-nv{nv}"
        plot_sweep_grid(
            rows, save_path=os.path.splitext(out_path)[0] + ".png",
            title_suffix=f"{size}, {method}, {n_scenarios} scenarios")

    print(f"\nTotal {time.time() - t0:.0f}s")
    return rows


def _paired_across_buffers(by_buffer, strategies, ref, n_scenarios):
    """Pair each buffer against 100%, policy held fixed. No extra solves.

    The other table pairs policies inside one buffer. This is the axis the
    buffer question actually asks about, and it is only sound because the
    planner is deterministic at these budgets: one plan per buffer, so the
    difference between two buffers is the effect of the buffer rather than of
    re-solving. Verify that with experiments.variance before quoting it.

    Prints the reference policy and returns every policy, so the printout
    stays readable and the CSV stays complete.
    """
    if 1.0 not in by_buffer or len(by_buffer) < 2:
        return []

    out = []
    for label, _ in strategies:
        times = {f"buffer_{b:.0%}": by_buffer[b][label]
                 for b in sorted(by_buffer) if label in by_buffer[b]}
        if len(times) < 2:
            continue
        for r in paired_scenarios(times, "buffer_100%"):
            r["case"] = label
            out.append(r)

    shown = [r for r in out if r["case"] == ref]
    if shown:
        print_paired(shown, "buffer_100%",
                     title=f"PAIRED ACROSS BUFFERS  ({ref}, {n_scenarios} "
                           f"scenarios, one solve per buffer)")
    else:
        print(f"\n!! --buffer-ref-strategy {ref!r} is not one of the rows; "
              f"the CSV still has every policy.")
    return out


def _finalise(rows, case="forward"):
    """Re-centre every row on the 100% buffer under `case`.

    Means sit in a narrow band, so the differences hide in the third digit.
    Positive is worse. Forward by default, matching the reference used
    everywhere a policy is compared.
    """
    baseline = next((r["mean"] for r in rows
                     if r["buffer"] == 1.0 and r["case"] == case), None)
    for row in rows:
        row["vs_baseline"] = (row["mean"] - baseline
                              if baseline is not None else 0.0)
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], parents=[base_parser()])
    parser.add_argument("--buffers", nargs="+", type=float, default=list(BUFFERS),
                        help="Buffers to sweep (default: 0.7 0.8 0.9 1.0)")
    parser.add_argument("--buffer-ref-strategy", default="forward",
                        help="Which policy the paired-across-buffers table "
                             "is printed for (default: forward). Every "
                             "policy is written to the CSV regardless)")
    parser.add_argument("--no-plot", dest="plot", action="store_false",
                        help="Skip the figure; table and CSV only")
    parser.add_argument("--no-solver-repair", dest="solver_repair",
                        action="store_false",
                        help="Drop the two solver re-planner rows, which cost "
                             "roughly 2.5 h on a 1000-scenario run")
    args = parser.parse_args()

    # Timestamped
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    size = "full" if args.full else f"ns{args.ns}-nv{args.nv}-cf{args.cf:g}"
    hp = ("_hp" + "-".join(str(p) for p in args.home_ports)
          if args.home_ports else "")
    name = (f"full-sweep_{args.method}_{size}{hp}"
            f"_sc{args.n_scenarios}_{stamp}.csv")
    out_path = os.path.join(OUTPUT_DIR, name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_solves = len(args.buffers)
    print("=" * 78)
    print("BUFFER x STRATEGY SWEEP")
    print(f"problem={'full 581-station survey' if args.full else f'ns{args.ns}'}"
          f", method={args.method}, catch={args.catch_source}, "
          f"scenarios={args.n_scenarios}, time_limit={args.time_limit}s")
    n_str = len(STRATEGIES) if args.solver_repair else len(STRATEGIES) - 2
    print(f"{n_solves} solves x {n_str} strategies = {n_solves * n_str} rows")
    print(f"rough estimate: {n_solves * (args.time_limit + 15) / 60:.0f} min "
          f"plus ~2 min to fit the scenarios")
    if args.solver_repair and args.repair_solver_time:
        # ~5 repairs a scenario across the two solver rows, and the budget is
        # spent on every one of them. Worth spelling out before a run starts.
        days = (n_solves * args.n_scenarios * 5 * args.repair_solver_time
                / 86400)
        print(f"!! --repair-solver-time {args.repair_solver_time:g}s is PER "
              f"REPAIR. Across {n_solves} buffers x {args.n_scenarios} "
              f"scenarios,")
        print(f"   that is roughly {days:.1f} DAYS of extra solving. Reduce "
              f"it, cut --n-scenarios, or pass --no-solver-repair.")
    print(f"writing to {name}")
    print("=" * 78)

    chosen = ([s for s in STRATEGIES if "solver" not in s[0]]
              if not args.solver_repair else STRATEGIES)

    run(strategies=chosen, repair_solver_time=args.repair_solver_time,
        buffer_ref=args.buffer_ref_strategy,
        plot=args.plot, time_limit=args.time_limit,
        n_scenarios=args.n_scenarios,
        scenario_seed=args.scenario_seed, catch_source=args.catch_source,
        method=args.method, buffers=tuple(args.buffers), full=args.full,
        ns=args.ns, nv=args.nv, cf=args.cf, instance=args.instance,
        home_ports=args.home_ports, out_path=out_path)
