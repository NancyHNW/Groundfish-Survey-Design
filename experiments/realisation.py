"""What does one season actually look like?

Solves the instance, draws one catch realisation from the historical spring
survey distributions, and plots what happens when the holds fill up: planned
route next to the route actually sailed, plus a per-trip time comparison.

The other experiments average over hundreds of scenarios. This one shows a
single concrete season, which is what makes the overflow behaviour readable.

    python -m experiments.realisation --ns 20 --nv 2 --cf 62.5 --seed 7
    python -m experiments.realisation --full --method tabu_combined
"""

from experiments.common import (base_parser, describe_run, output_path, solve)


def run(ns=20, nv=2, cf=62.5, instance=1, method="tabu_move", time_limit=10,
        catch_source="historical", capacity_buffer=1.0, strategy="backtrack",
        preemptive_threshold=0.8, seed=None, full=False, catch_table=True):
    """Solve, draw one catch scenario, report and plot the realisation."""
    from unified.stochastic_catch import CatchSimulator
    from unified.stochastic_eval import (evaluate_single_realisation,
                                         plot_realisation, plot_time_comparison,
                                         print_catch_table)

    det = solve(ns=ns, nv=nv, cf=cf, instance=instance, method=method,
                time_limit=time_limit, catch_source=catch_source,
                capacity_buffer=capacity_buffer, full=full)
    trips, inst = det["trips"], det["instance"]

    print(f"\nGot {len(trips)} trips, planned {det['planned_time']:.1f}h "
          f"(feasible={det['feasible']})")
    _print_trips(trips, inst)

    # The solver planned against catch_source, but the realisation always
    # comes from the historical distributions. So a solver that planned with
    # lighter catch data will look worse here than it deserves.
    sim = CatchSimulator(seed=seed)
    sim.fit()
    catch = sim.sample(n_scenarios=1)[0]

    result = evaluate_single_realisation(
        trips, inst, catch, strategy=strategy,
        preemptive_threshold=preemptive_threshold)

    _print_result(result, strategy, preemptive_threshold, seed)

    tag = f"{method}-{strategy}"
    if capacity_buffer != 1.0:
        tag = f"{method}-buf{capacity_buffer:g}-{strategy}"

    plot_realisation(trips, inst, result,
                     save_path=output_path("realisation", tag, inst))
    plot_time_comparison(trips, result,
                         save_path=output_path("time-comparison", tag, inst))
    if catch_table:
        print_catch_table(trips, catch, inst)

    return result


def _print_trips(trips, inst):
    """Print station count and load per trip, numbered within each boat."""
    unique_bids = sorted({t["boat_id"] for t in trips})
    bid_to_idx = {bid: i for i, bid in enumerate(unique_bids)}
    counter = {}
    for t in trips:
        bid = t["boat_id"]
        idx = counter.get(bid, 0)
        counter[bid] = idx + 1
        cap = inst.capacities[bid_to_idx[bid]]
        n_stations = sum(1 for n in t["nodes"] if n >= 13) // 2
        print(f"  Boat {bid}, Trip {idx}: {n_stations} stations, "
              f"catch={t['catch']:.0f}, capacity={cap:.0f}")


def _print_result(result, strategy, threshold, seed):
    label = strategy
    if strategy == "preemptive":
        label += f" (threshold {threshold:g})"
    print(f"\n--- Simulated realisation: {label}, seed={seed} ---")
    print(f"  capacity exceeded:    {result['capacity_exceeded']}")
    print(f"  unscheduled returns:  {result['n_unscheduled_returns']}")
    print(f"  overflow events:      {len(result['overflow_events'])}")
    print(f"  planned total time:   {result['original_total_time']:.1f} h")
    print(f"  actual total time:    {result['total_time']:.1f} h "
          f"(+{result['time_penalty']:.1f} h, "
          f"+{result['time_penalty'] / result['original_total_time']:.1%})")

    print("\n  Per-trip breakdown:")
    for i, td in enumerate(result["trip_details"]):
        flag = "  *** OVERFLOW" if td["exceeded"] else ""
        print(f"    Trip {i} (Boat {td['boat_id']}): "
              f"planned={td['original_time']:.1f}, "
              f"actual={td['adjusted_time']:.1f}, "
              f"detour=+{td['detour_time']:.1f}{flag}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], parents=[base_parser()])
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the single catch draw "
                             "(default: a new realisation each run)")
    parser.add_argument("--full", action="store_true",
                        help="Use the full 581-station problem "
                             "(ignores --ns/--nv/--cf)")
    parser.add_argument("--no-catch-table", dest="catch_table",
                        action="store_false",
                        help="Skip the per-station catch table")
    args = parser.parse_args()

    describe_run("SINGLE REALISATION", args, strategy=args.strategy,
                 seed=args.seed, full=args.full)
    run(ns=args.ns, nv=args.nv, cf=args.cf, instance=args.instance,
        method=args.method, time_limit=args.time_limit,
        catch_source=args.catch_source,
        capacity_buffer=args.capacity_buffer, strategy=args.strategy,
        preemptive_threshold=args.threshold, seed=args.seed, full=args.full,
        catch_table=args.catch_table)
