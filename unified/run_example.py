"""Example: Run heuristic solvers on a gfsp_code test problem.

Usage:
    cd StartingMaterials
    python -m unified.run_example

This script demonstrates the full workflow:
  1. Load a test problem from gfsp_code
  2. Convert it to Saahil's Problem format
  3. Run GRASP + neighbourhood search heuristics
  4. Evaluate and display the solution
"""

import time
import numpy as np

from unified.loaders import load_gfsp_problem, load_heuristic_problem, list_gfsp_problems
from unified.adapters import heuristic_context, to_heuristic_problem
from unified.evaluate import evaluate_heuristic_solution, compare_results, solution_to_trips


def run_heuristic_on_gfsp(ns=20, nv=2, cf=62.5, instance=1, method="grasp",
                          time_limit=10, seed=42, init="grasp",
                          catch_source="gfsp"):
    """Run heuristic solvers on a gfsp test problem.

    Parameters
    ----------
    ns, nv, cf, instance : gfsp problem identifiers
    method : str — "grasp", "tabu_swap", "tabu_move", "sa", or "grasp_only"
    time_limit : float — seconds to run the improvement loop
    seed : int — random seed
    init : str — "grasp" (probabilistic, may violate capacity) or
                 "greedy" (strict constraint enforcement)
    catch_source : str — "gfsp", "heuristic", or "historical"

    Returns
    -------
    dict — evaluation results
    """
    # 1. Load the gfsp test problem
    inst = load_gfsp_problem(ns=ns, nv=nv, cf=cf, instance=instance)
    print(f"Loaded: {inst}")

    # 2. Convert and run inside Saahil's context
    with heuristic_context():
        from classes import Problem, Trip
        from neighbourhood_rules import (
            next_descent_swap_improved,
            next_descent_move,
            tabu_search_swap,
            tabu_search_move,
            simulated_annealing,
        )

        prob = to_heuristic_problem(inst, catch_source=catch_source)
        print(f"Created Problem: {prob}")

        # Penalty parameters (same logic as final_testing_heur.py)
        k_catch = max(1.0, (400000 - np.mean(inst.capacities)) * 0.001)
        k_fish = np.mean(inst.capacities) / inst.fish_time_limit * k_catch
        upper_bound = inst.upper_bound if np.isfinite(inst.upper_bound) else 10000

        # 3. Run the heuristic
        best_obj = np.inf
        best_sol = None
        t0 = time.time()
        iteration = 0

        while time.time() - t0 < time_limit:
            iteration += 1
            prob.reset()

            # Construct initial solution
            if init == "greedy":
                prob.generate_initial_solution(seed=seed + iteration)
            else:
                prob.GRASP(rcl_size=2, seed=seed + iteration)

            # Save/restore cycle: this is required because add_to_route
            # stores 1-indexed boat IDs in boat_dict, but neighbourhood
            # rules expect 0-indexed (as set by restore_solution_from_list).
            init_sol = prob.save_solution_as_list()
            prob.restore_solution_from_list(init_sol)

            obj, pen = prob.evaluate_solution(k_catch, k_fish, upper_bound)
            cost = obj if pen < 1e-8 else obj + pen + upper_bound

            if method == "grasp_only":
                pass
            else:
                # Improve route ordering first
                for boat in prob.boats:
                    boat.improve_route(prob)
                for boat in prob.boats:
                    if len(boat.route) == 1:
                        boat.route.append(Trip(boat.home_port, [], boat.home_port))

                # Apply neighbourhood search
                if method == "grasp":
                    obj, pen, cost = next_descent_swap_improved(
                        prob, k_catch, k_fish, upper_bound, seed=seed + iteration, tsp=0.02
                    )
                elif method == "tabu_swap":
                    obj, pen, cost = tabu_search_swap(
                        prob, k_catch, k_fish, upper_bound, inst.work_limit,
                        history_size=max(1, len(prob.stations) // 10),
                        max_iter=10, max_time=300, tsp=0.02,
                    )
                elif method == "tabu_move":
                    obj, pen, cost = tabu_search_move(
                        prob, k_catch, k_fish, upper_bound, inst.work_limit,
                        history_size=max(1, len(prob.stations) // 10),
                        max_iter=10, max_time=300, tsp=0.02,
                    )
                elif method == "sa":
                    obj, pen, cost = simulated_annealing(
                        prob, k_catch, k_fish, upper_bound,
                        temp_init=50, alpha=0.99, stopping_temp=1e-10,
                        max_iter=10, tsp=0.02, seed=seed + iteration,
                    )

            # Final TSP improvement if feasible
            if pen < 1e-8:
                for boat in prob.boats:
                    boat.improve_route(prob)
                obj = sum(boat.total_time for boat in prob.boats)

                if obj < best_obj:
                    best_obj = obj
                    best_sol = prob.save_solution_as_list()

            elapsed = time.time() - t0
            print(f"  Iter {iteration}: obj={obj:.2f}, pen={pen:.2f}, "
                  f"best={best_obj:.2f}, elapsed={elapsed:.1f}s")

        # Restore best solution
        if best_sol is not None:
            prob.restore_solution_from_list(best_sol)

        # 4. Evaluate
        result = evaluate_heuristic_solution(prob, inst)
        result["iterations"] = iteration
        result["elapsed"] = time.time() - t0
        result["method"] = method
        # return the trips too so the stochastic evaluator can reuse this solution
        # (extract here while prob is still valid)
        result["trips"] = solution_to_trips(prob)
        result["instance"] = inst

        print(f"\n--- Final Result ---")
        print(f"Method: {method}")
        print(f"Objective (total time): {result['objective']:.2f}")
        print(f"Feasible: {result['feasible']}")
        print(f"Iterations: {iteration}")
        print(f"Elapsed: {result['elapsed']:.1f}s")
        prob.display_solution()

        return result


def compare_methods_on_gfsp(ns=20, nv=2, cf=62.5, instance=1, time_limit=10):
    """Run multiple methods on the same gfsp problem and compare."""
    methods = ["grasp_only", "grasp", "sa"]
    results = []
    for m in methods:
        print(f"\n{'='*60}")
        print(f"Running method: {m}")
        print(f"{'='*60}")
        r = run_heuristic_on_gfsp(
            ns=ns, nv=nv, cf=cf, instance=instance,
            method=m, time_limit=time_limit,
        )
        results.append(r)

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(compare_results(*results, labels=methods))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run heuristic solvers on a gfsp test problem"
    )
    parser.add_argument("--ns", type=int, default=20, help="Number of stations")
    parser.add_argument("--nv", type=int, default=2, help="Number of vessels")
    parser.add_argument("--cf", type=float, default=62.5, help="Capacity factor")
    parser.add_argument("--instance", type=int, default=1, help="Instance number")
    parser.add_argument("--method", default="grasp",
                        choices=["grasp_only", "grasp", "tabu_swap", "tabu_move", "sa"],
                        help="Heuristic method to use")
    parser.add_argument("--time-limit", type=float, default=10,
                        help="Time limit in seconds")
    parser.add_argument("--init", default="greedy", choices=["grasp", "greedy"],
                        help="Initial solution method")
    parser.add_argument("--compare", action="store_true",
                        help="Run multiple methods and compare")
    args = parser.parse_args()

    if args.compare:
        compare_methods_on_gfsp(
            ns=args.ns, nv=args.nv, cf=args.cf,
            instance=args.instance, time_limit=args.time_limit,
        )
    else:
        run_heuristic_on_gfsp(
            ns=args.ns, nv=args.nv, cf=args.cf,
            instance=args.instance, method=args.method,
            time_limit=args.time_limit, init=args.init,
        )
