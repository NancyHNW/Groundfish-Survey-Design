"""Does planning to less than full capacity pay for itself?

Solves the same instance at several buffers and scores every solution against
one shared set of catch scenarios. Planned time goes up as the buffer tightens,
so a buffer is only worth it if mean realised time drops by more than planned
time rose.

    python -m experiments.buffers --ns 100 --nv 2 --cf 125 \
        --method tabu_move --time-limit 60 --buffers 0.7 0.8 0.9 1.0
"""

from experiments.common import (CORE_COLUMNS, add_baseline_delta, base_parser,
                                describe_run, make_evaluator, output_path,
                                print_table, save_csv, sweep_solve)

COLUMNS = ([("buffer", "buffer", 9, ".0%")] + CORE_COLUMNS
           + [("vs_full", "vs 100%", 9, "+.1f")])


def run(ns=100, nv=2, cf=125, instance=1, method="tabu_move", time_limit=10,
        catch_source="historical", strategy="backtrack",
        preemptive_threshold=0.8, n_scenarios=500, scenario_seed=123,
        buffers=(0.7, 0.8, 0.9, 1.0)):
    """Solve at each buffer, evaluate against shared scenarios, return rows."""
    evaluator = make_evaluator(scenario_seed, n_scenarios)

    rows = []
    inst = None
    for _, det, result, row in sweep_solve(
            "capacity_buffer", buffers, evaluator, key="buffer", fmt=".0%",
            strategy=strategy, preemptive_threshold=preemptive_threshold,
            ns=ns, nv=nv, cf=cf, instance=instance, method=method,
            time_limit=time_limit, catch_source=catch_source):
        rows.append(row)
        inst = det["instance"]
        print(f"Planned {det['planned_time']:.1f}h over {len(det['trips'])} "
              f"trips, P(exceed)={result['p_capacity_exceedance']:.1%}")

    # Re-centre on full capacity, the do-nothing baseline
    add_baseline_delta(rows, "buffer", max(buffers), "vs_full")

    print_table(rows, COLUMNS,
                title=f"BUFFER COMPARISON  (strategy={strategy}, "
                      f"{n_scenarios} scenarios)")

    best = min(rows, key=lambda r: r["mean"])
    print(f"\nLowest mean realised time: buffer {best['buffer']:.0%}")
    if best["buffer"] == max(buffers):
        print("No buffer beat planning to full capacity on this instance.")

    tag = f"{method}-{strategy}"
    save_csv(rows, output_path("buffer-comparison", tag, inst, ".csv"))

    from unified.stochastic_eval import plot_buffer_comparison
    plot_buffer_comparison(
        rows, save_path=output_path("buffer-comparison", tag, inst),
        title_suffix=(f"ns={ns}, nv={nv}, cf={cf}, {method}, {strategy}, "
                      f"{n_scenarios} scenarios, catch={catch_source}"))

    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], parents=[base_parser()])
    parser.add_argument("--buffers", nargs="+", type=float,
                        default=[0.7, 0.8, 0.9, 1.0],
                        help="Buffers to compare (default: 0.7 0.8 0.9 1.0)")
    args = parser.parse_args()

    describe_run("BUFFER COMPARISON", args, strategy=args.strategy,
                 buffers=[f"{b:.0%}" for b in args.buffers])
    run(ns=args.ns, nv=args.nv, cf=args.cf, instance=args.instance,
        method=args.method, time_limit=args.time_limit,
        catch_source=args.catch_source, strategy=args.strategy,
        preemptive_threshold=args.threshold, n_scenarios=args.n_scenarios,
        scenario_seed=args.scenario_seed, buffers=tuple(args.buffers))
