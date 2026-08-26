"""Run the heuristic solvers on a gfsp test problem.

The solver entry point the rest of the code builds on: load a problem, convert
it to the heuristic Problem class, run GRASP plus a neighbourhood search, and
return the evaluated solution.

No plotting and no experiments in here. Those live in the experiments package,
which calls in through experiments.common.solve.

Run one solve directly:

    python -m unified.solve --ns 100 --nv 2 --cf 125 --method tabu_move
"""

import time

import numpy as np

from unified.loaders import load_gfsp_problem
from unified.adapters import heuristic_context, to_heuristic_problem
from unified.evaluate import evaluate_heuristic_solution, solution_to_trips

# Neighbourhood searches available as --method.
# experiments.common.METHODS maps the friendlier CLI names onto these.
METHOD_CHOICES = ("grasp_only", "grasp", "tabu_swap", "tabu_move",
                  "tabu_combined", "sa")


def run_heuristic_on_gfsp(ns=20, nv=2, cf=62.5, instance=1, method="grasp",
                          time_limit=10, seed=42, init="grasp",
                          catch_source="gfsp", home_ports=None,
                          capacity_buffer=1.0, full=False, verbose=True):
    """Solve one gfsp test problem and return the evaluated solution.

    Parameters
    ----------
    ns, nv, cf, instance : gfsp problem identifiers
    method : str, one of METHOD_CHOICES
    time_limit : float, seconds to run the restart loop
    seed : int, base random seed. Each restart uses seed + iteration
    init : str, "grasp" (probabilistic) or "greedy" (strict constraints)
    catch_source : str, "gfsp", "heuristic" or "historical". The evaluator
        always draws scenarios from the historical distributions, so planning
        against "gfsp" is a systematic bias, not just extra noise.
    home_ports : list[int] or None, per-vessel home ports. None = all share
        instance.home_port
    capacity_buffer : float, fraction of true capacity the solver plans to
        (0.9 leaves 10% headroom for catch heavier than planned)
    full : bool, solve the 581-station survey. ns/nv/cf are ignored then
    verbose : bool, print progress and the final solution. False silences the
        whole function, which is what the experiments want when they print
        their own summary

    Returns
    -------
    dict, evaluation results plus 'trips', 'instance', 'iterations',
    'elapsed', 'method' and 'capacity_buffer'
    """
    # 1. Load the problem
    if full:
        from unified.loaders import load_gfsp_full_problem
        inst = load_gfsp_full_problem()
    else:
        inst = load_gfsp_problem(ns=ns, nv=nv, cf=cf, instance=instance)
    if verbose:
        print(f"Loaded: {inst}")

    # 2. Convert and run inside the heuristic code's import context
    with heuristic_context():
        from classes import Trip

        prob = to_heuristic_problem(inst, catch_source=catch_source,
                                    home_ports=home_ports,
                                    capacity_buffer=capacity_buffer)
        if verbose and capacity_buffer != 1.0:
            planned_vs_true = [f"{b.planning_capacity:.0f}/{b.capacity:.0f}"
                               for b in prob.boats]
            print(f"Planning to {capacity_buffer:.0%} of capacity: "
                  f"{planned_vs_true}")

        # Penalty weights for infeasibility: k_catch prices going over
        # capacity, k_fish prices going over the fishing time limit. Scaled to
        # the instance so both matter about equally. Same formulas as
        # final_testing_heur.py, kept identical so results stay comparable.
        k_catch = max(1.0, (400000 - np.mean(inst.capacities)) * 0.001)
        k_fish = np.mean(inst.capacities) / inst.fish_time_limit * k_catch
        upper_bound = (inst.upper_bound if np.isfinite(inst.upper_bound)
                       else 10000)

        # 3. Restart loop until the time limit
        best_obj = np.inf
        best_sol = None
        t0 = time.time()
        iteration = 0

        while time.time() - t0 < time_limit:
            iteration += 1
            prob.reset()

            if init == "greedy":
                prob.generate_initial_solution(seed=seed + iteration)
            else:
                prob.GRASP(rcl_size=2, seed=seed + iteration)

            # Save/restore cycle. Needed because add_to_route stores 1-indexed
            # boat IDs in boat_dict, but the neighbourhood rules expect
            # 0-indexed (which is what restore_solution_from_list sets).
            init_sol = prob.save_solution_as_list()
            prob.restore_solution_from_list(init_sol)

            obj, pen = prob.evaluate_solution(k_catch, k_fish, upper_bound)

            if method != "grasp_only":
                for boat in prob.boats:
                    boat.improve_route(prob)
                    # A swap or move needs somewhere to move stations to, so a
                    # boat left with a single trip gets an empty second one.
                    # Boats end where they started, hence home port at both ends.
                    if len(boat.route) == 1:
                        boat.route.append(
                            Trip(boat.home_port, [], boat.home_port))

                # The searches also return a penalised cost, which nothing here
                # uses: the best solution is picked on obj among feasible ones.
                obj, pen, _ = _improve(method, prob, k_catch, k_fish,
                                       upper_bound, inst, seed + iteration)

            # Only feasible solutions can win. Re-order each trip with the TSP
            # solver first, then re-read the objective, since reordering
            # changes the travel time the searches were working from.
            if pen < 1e-8:
                for boat in prob.boats:
                    boat.improve_route(prob)
                obj = sum(boat.total_time for boat in prob.boats)

                if obj < best_obj:
                    best_obj = obj
                    best_sol = prob.save_solution_as_list()

            if verbose:
                print(f"  Iter {iteration}: obj={obj:.2f}, pen={pen:.2f}, "
                      f"best={best_obj:.2f}, elapsed={time.time() - t0:.1f}s")

        if best_sol is not None:
            prob.restore_solution_from_list(best_sol)

        # 4. Evaluate
        result = evaluate_heuristic_solution(prob, inst)
        result["iterations"] = iteration
        result["elapsed"] = time.time() - t0
        result["method"] = method
        result["capacity_buffer"] = capacity_buffer
        # Pull the trips out here while prob is still valid, so the stochastic
        # evaluator can reuse this solution outside the context
        result["trips"] = solution_to_trips(prob)
        result["instance"] = inst

        if verbose:
            print("\n--- Final Result ---")
            print(f"Method: {method}")
            print(f"Objective (total time): {result['objective']:.2f}")
            print(f"Feasible (true capacity): {result['feasible']}")
            if capacity_buffer != 1.0:
                print(f"Feasible ({capacity_buffer:.0%} buffered capacity): "
                      f"{result['feasible_planning']}")
            print(f"Iterations: {iteration}, "
                  f"elapsed {result['elapsed']:.1f}s")
            prob.display_solution()

    return result


def _improve(method, prob, k_catch, k_fish, upper_bound, inst, seed):
    """Run the chosen neighbourhood search. Returns (obj, pen, cost).

    The searches are imported here rather than passed in. They are only
    importable with final_code on sys.path, and re-entering heuristic_context
    while already inside it is a no-op, so this is safe from either place.
    """
    with heuristic_context():
        from neighbourhood_rules import (next_descent_swap_improved,
                                         simulated_annealing,
                                         tabu_search_combined,
                                         tabu_search_move, tabu_search_swap)

    # Every tabu variant takes the same arguments, only the search differs
    tabu_args = (prob, k_catch, k_fish, upper_bound, inst.work_limit)
    tabu_kwargs = dict(history_size=max(1, len(prob.stations) // 10),
                       max_iter=10, max_time=300, tsp=0.02)

    if method == "grasp":
        return next_descent_swap_improved(
            prob, k_catch, k_fish, upper_bound, seed=seed, tsp=0.02)
    if method == "tabu_swap":
        return tabu_search_swap(*tabu_args, **tabu_kwargs)
    if method == "tabu_move":
        return tabu_search_move(*tabu_args, **tabu_kwargs)
    if method == "tabu_combined":
        return tabu_search_combined(*tabu_args, **tabu_kwargs)
    if method == "sa":
        return simulated_annealing(
            prob, k_catch, k_fish, upper_bound, temp_init=50, alpha=0.99,
            stopping_temp=1e-10, max_iter=10, tsp=0.02, seed=seed)
    raise ValueError(f"Unknown method: {method!r}. "
                     f"Choose from {list(METHOD_CHOICES)}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Solve one gfsp test problem with the heuristics")
    parser.add_argument("--ns", type=int, default=20, help="Number of stations")
    parser.add_argument("--nv", type=int, default=2, help="Number of vessels")
    parser.add_argument("--cf", type=float, default=62.5, help="Capacity factor")
    parser.add_argument("--instance", type=int, default=1, help="Instance number")
    parser.add_argument("--method", default="grasp", choices=METHOD_CHOICES,
                        help="Heuristic method (default: grasp)")
    parser.add_argument("--time-limit", type=float, default=10,
                        help="Time limit in seconds (default: 10)")
    parser.add_argument("--init", default="greedy", choices=["grasp", "greedy"],
                        help="Initial solution method (default: greedy)")
    parser.add_argument("--catch-source", default="gfsp",
                        choices=["gfsp", "heuristic", "historical"],
                        help="Catch data the solver plans against")
    parser.add_argument("--capacity-buffer", type=float, default=1.0,
                        help="Fraction of capacity to plan to (default: 1.0)")
    parser.add_argument("--full", action="store_true",
                        help="Solve the 581-station problem")
    parser.add_argument("--quiet", dest="verbose", action="store_false",
                        help="Suppress per-iteration progress")
    args = parser.parse_args()

    run_heuristic_on_gfsp(
        ns=args.ns, nv=args.nv, cf=args.cf, instance=args.instance,
        method=args.method, time_limit=args.time_limit, init=args.init,
        catch_source=args.catch_source,
        capacity_buffer=args.capacity_buffer, full=args.full,
        verbose=args.verbose,
    )
