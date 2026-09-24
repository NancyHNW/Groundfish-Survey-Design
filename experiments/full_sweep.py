"""Every buffer crossed with every overflow strategy, on the full problem.

The buffer decides the routes, so each buffer needs its own solve. The strategy
is chosen at sea once the plan is fixed, so all four strategies score the same
solution. That means the whole grid costs one solve per buffer, not one per
cell -- 4 solves for 16 rows rather than 16.

Everything shares one scenario set, so differences between cells come from the
plans and the responses rather than from resampling the catch.

    python -m experiments.full_sweep --full --time-limit 300 --n-scenarios 1000

Long runs: results are written after every buffer, so a crash or a locked file
late on does not cost the whole sweep.
"""

import os
import time
from datetime import datetime

from experiments.common import (CORE_COLUMNS, OUTPUT_DIR, base_parser,
                                make_evaluator, print_table, save_csv,
                                summarise, solve)

BUFFERS = (0.7, 0.8, 0.9, 1.0)

# (label, evaluate kwargs). Same four as experiments.strategies.
STRATEGIES = [
    ("backtrack", {"strategy": "backtrack"}),
    ("forward", {"strategy": "forward"}),
    ("preemptive_0.8", {"strategy": "preemptive", "preemptive_threshold": 0.8}),
    ("preemptive_0.7", {"strategy": "preemptive", "preemptive_threshold": 0.7}),
]

COLUMNS = ([("buffer", "buffer", 8, ".0%"), ("case", "strategy", 16, "")]
           + CORE_COLUMNS
           + [("feasible", "feas", 7, ""),
              ("vs_baseline", "vs 100%/bt", 12, "+.1f")])


def run(time_limit=300, n_scenarios=1000, scenario_seed=123,
        catch_source="historical", method="tabu_move", buffers=BUFFERS,
        full=True, ns=100, nv=2, cf=125, instance=1, home_ports=None,
        out_path=None, plot=True):
    """Solve at each buffer, score each solution under each strategy.

    Returns the full grid of rows. Writes the CSV after every buffer.
    """
    from unified.stochastic_eval import plot_sweep_grid

    t0 = time.time()
    evaluator = make_evaluator(scenario_seed, n_scenarios)
    print(f"Scenario set ready ({n_scenarios} scenarios, "
          f"seed {scenario_seed}) at {time.time() - t0:.0f}s\n")

    rows = []
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

        for label, kwargs in STRATEGIES:
            result = evaluator.evaluate(det["trips"], det["instance"], **kwargs)
            row = summarise(result, det["planned_time"], len(det["trips"]),
                            feasible=det["feasible"],
                            buffer=buffer, case=label)
            rows.append(row)
            print(f"    {label:16s} mean {row['mean']:8.1f}h  "
                  f"p95 {row['p95']:8.1f}h  ret/trip {row['returns_per_trip']:.3f}",
                  flush=True)

        # Written every buffer, so a crash late on does not lose the earlier work
        _finalise(rows)
        save_csv(rows, out_path)
        print(f"  elapsed {time.time() - t0:.0f}s\n", flush=True)

    _finalise(rows)
    print_table(rows, COLUMNS,
                title=f"FULL SWEEP  ({n_scenarios} scenarios, {method}, "
                      f"{time_limit}s per solve)")
    save_csv(rows, out_path)

    if plot and len(buffers) > 1:
        size = "full 581-station" if full else f"ns{ns}-nv{nv}"
        plot_sweep_grid(
            rows, save_path=os.path.splitext(out_path)[0] + ".png",
            title_suffix=f"{size}, {method}, {n_scenarios} scenarios")

    print(f"\nTotal {time.time() - t0:.0f}s")
    return rows


def _finalise(rows):
    """Re-centre every row on the 100% buffer under backtrack.

    Means sit in a narrow band, so the differences hide in the third digit.
    Positive is worse.
    """
    baseline = next((r["mean"] for r in rows
                     if r["buffer"] == 1.0 and r["case"] == "backtrack"), None)
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
    parser.add_argument("--no-plot", dest="plot", action="store_false",
                        help="Skip the figure; table and CSV only")
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
    print(f"{n_solves} solves x {len(STRATEGIES)} strategies = "
          f"{n_solves * len(STRATEGIES)} rows")
    print(f"rough estimate: {n_solves * (args.time_limit + 15) / 60:.0f} min "
          f"plus ~2 min to fit the scenarios")
    print(f"writing to {name}")
    print("=" * 78)

    run(plot=args.plot, time_limit=args.time_limit,
        n_scenarios=args.n_scenarios,
        scenario_seed=args.scenario_seed, catch_source=args.catch_source,
        method=args.method, buffers=tuple(args.buffers), full=args.full,
        ns=args.ns, nv=args.nv, cf=args.cf, instance=args.instance,
        home_ports=args.home_ports, out_path=out_path)
