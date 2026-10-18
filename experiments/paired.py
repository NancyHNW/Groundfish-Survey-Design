"""Compare settings across many instances, differencing inside each block.

The other experiments solve once, which is fine when the comparison shares that
solve and useless when it does not -- restart noise on the full problem spans
tens of hours, wider than most of the effects being looked for. This runs every
setting on every (instance, seed) block and differences within the block, so
what comes out is a mean difference with an interval around it.

    python -m experiments.paired --compare strategy --instances 1-30
    python -m experiments.paired --compare buffer   --instances 1-30
    python -m experiments.paired --compare method   --instances 1-20
"""

import os

from experiments.common import (OUTPUT_DIR, base_parser, describe_run,
                                make_evaluator, paired_compare, parse_instances,
                                print_paired, save_csv)

# (label, kwargs) per setting. Scoring keys go to the evaluator, the rest force
# their own solve.
COMPARISONS = {
    "strategy": {
        # Forward is the reference everywhere a policy is compared: it is
        # the policy that matches the survey (a tow is one event), and
        # differencing against it directly gives an interval on the gap to
        # each alternative, which subtracting two backtrack-relative
        # intervals does not.
        "baseline": "forward",
        "settings": [
            ("backtrack", {"strategy": "backtrack"}),
            ("forward", {"strategy": "forward"}),
            ("preemptive_0.8", {"strategy": "preemptive",
                                "preemptive_threshold": 0.8}),
            ("preemptive_0.7", {"strategy": "preemptive",
                                "preemptive_threshold": 0.7}),
            ("preemptive_0.9", {"strategy": "preemptive",
                                "preemptive_threshold": 0.9}),
            ("repair_trip", {"strategy": "repair",
                             "repair_scope": "trip"}),
            ("repair_boat", {"strategy": "repair",
                             "repair_scope": "boat"}),
            ("repair_trip_2opt", {"strategy": "repair",
                                  "repair_scope": "trip",
                                  "repair_planner": "2opt"}),
            ("repair_boat_2opt", {"strategy": "repair",
                                  "repair_scope": "boat",
                                  "repair_planner": "2opt"}),
            # The solver re-planner adds no solves -- repair_planner is an
            # EVAL_KEY, so these share the block's solve. It does add a lot of
            # scoring: trip scope ~25 min over 30 instances, boat scope ~9 h,
            # because it re-plans ~150 stations per repair instead of a few.
            ("repair_trip_solver", {"strategy": "repair",
                                    "repair_scope": "trip",
                                    "repair_planner": "solver"}),
            ("repair_boat_solver", {"strategy": "repair",
                                    "repair_scope": "boat",
                                    "repair_planner": "solver"}),
            # Scoring-only, so these share the block solve and cost no
            # extra solves -- only scoring time.
            ("repair_trip_keep", {"strategy": "repair",
                                  "repair_scope": "trip",
                                  "repair_planner": "keep_order"}),
            ("repair_boat_keep", {"strategy": "repair",
                                  "repair_scope": "boat",
                                  "repair_planner": "keep_order"}),
            # Most freedom, strongest search: the ceiling of what re-planning
            # can do. The answer to "you only tried weak re-planners". Also
            # the slowest row here by a wide margin -- if it will not finish,
            # drop it and cite the single-solve full-problem figure instead.
            ("repair_boat_tabu", {"strategy": "repair",
                                  "repair_scope": "boat",
                                  "repair_planner": "tabu"}),
        ],
    },
    "buffer": {
        # The policy is not swept here, so without an explicit strategy the
        # evaluator would fall back to its own default. Pinned to forward so
        # the held-fixed policy matches Table 3.5.
        "baseline": "buffer_1.0",
        "settings": [
            ("buffer_1.0", {"capacity_buffer": 1.0, "strategy": "forward"}),
            ("buffer_0.9", {"capacity_buffer": 0.9, "strategy": "forward"}),
            ("buffer_0.8", {"capacity_buffer": 0.8, "strategy": "forward"}),
            ("buffer_0.7", {"capacity_buffer": 0.7, "strategy": "forward"}),
        ],
    },
    "method": {
        "baseline": "grasp_only",
        "settings": [
            ("grasp_only", {"method": "grasp_only", "strategy": "forward"}),
            ("grasp_swap", {"method": "grasp_swap", "strategy": "forward"}),
            ("tabu_swap", {"method": "tabu_swap", "strategy": "forward"}),
            ("tabu_move", {"method": "tabu_move", "strategy": "forward"}),
            ("tabu_combined", {"method": "tabu_combined", "strategy": "forward"}),
            ("sa", {"method": "sa", "strategy": "forward"}),
        ],
    },
}


def run(compare="strategy", instances=(1,), solver_seeds=(42,),
        n_scenarios=500, scenario_seed=123, out_path=None, **fixed):
    """Run one comparison and print it. Returns (summary_rows, block_rows)."""
    spec = COMPARISONS[compare]
    evaluator = make_evaluator(scenario_seed, n_scenarios)

    summary, blocks = paired_compare(
        spec["settings"], spec["baseline"], instances=instances,
        solver_seeds=solver_seeds, evaluator=evaluator, **fixed)

    print_paired(summary, spec["baseline"],
                 title=f"PAIRED {compare.upper()} COMPARISON  "
                       f"({len(instances)} instances x {len(solver_seeds)} "
                       f"seeds, {n_scenarios} scenarios)")

    if out_path:
        save_csv(summary, out_path)
        save_csv(blocks, out_path.replace(".csv", "_blocks.csv"))
    return summary, blocks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], parents=[base_parser()])
    parser.add_argument("--compare", default="strategy",
                        choices=sorted(COMPARISONS),
                        help="What to compare (default: strategy)")
    args = parser.parse_args()

    instances = (parse_instances(args.instances) if args.instances
                 else [args.instance])
    seeds = args.solver_seeds

    if args.full:
        # One instance, so seeds are the only axis pairing has left. Warn rather
        # than let --instances 1-30 look like it did something.
        if args.instances:
            print("!! --full is a single instance; --instances is ignored.")
        instances = [1]
        if len(seeds) < 2:
            print("!! --full with one solver seed gives n=1 blocks: a "
                  "difference, but no interval. Pass several --solver-seeds.")

    # Whatever is swept must not also print as fixed, or --compare method reads
    # as though every solve used tabu_move.
    swept = {k for _, kw in COMPARISONS[args.compare]["settings"] for k in kw}
    omit = ("instance",) + (("method",) if "method" in swept else ())

    describe_run("PAIRED COMPARISON", args, omit=omit,
                 compare=args.compare, blocks=len(instances) * len(seeds))

    tag = args.compare if "method" in swept else f"{args.compare}-{args.method}"
    size = "full" if args.full else f"ns{args.ns}-nv{args.nv}"
    # Seeds in the filename so a run can be split one seed at a time and the
    # blocks files combined afterwards. A 20-solve run is a single point of
    # failure; five 4-solve runs are not.
    seedtag = ("_s" + "-".join(str(s) for s in seeds)) if len(seeds) <= 5 else ""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"paired_{tag}_{size}{seedtag}.csv")

    fixed = dict(ns=args.ns, nv=args.nv, cf=args.cf, method=args.method,
                 time_limit=args.time_limit, catch_source=args.catch_source,
                 full=args.full, home_ports=args.home_ports)
    # Same reason, but here it would pass solve() the keyword twice
    for key in swept:
        fixed.pop(key, None)

    run(compare=args.compare, instances=instances, solver_seeds=seeds,
        n_scenarios=args.n_scenarios, scenario_seed=args.scenario_seed,
        out_path=out, **fixed)
