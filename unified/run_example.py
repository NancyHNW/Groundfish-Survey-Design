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

import os
import time
import numpy as np

from unified.loaders import load_gfsp_problem, load_heuristic_problem, list_gfsp_problems
from unified.adapters import heuristic_context, to_heuristic_problem
from unified.evaluate import evaluate_heuristic_solution, compare_results, solution_to_trips

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def plot_output_path(plot_name, solver_tag, inst=None):
    """Path under tests/outputs/ named by plot, solver and instance size."""
    out_dir = os.path.join(_ROOT, "tests", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    parts = [plot_name, solver_tag]
    if inst is not None:
        parts.append(f"ns{inst.ns}-nv{inst.n_boats}")
    return os.path.join(out_dir, "_".join(parts) + ".png")


def run_heuristic_on_gfsp(ns=20, nv=2, cf=62.5, instance=1, method="grasp",
                          time_limit=10, seed=42, init="grasp",
                          catch_source="gfsp", home_ports=None,
                          capacity_buffer=1.0, plot=False, simulate=True,
                          strategy="backtrack", preemptive_threshold=0.8,
                          sim_seed=None):
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
    capacity_buffer : float — fraction of true capacity the solver plans to
                     (e.g. 0.9 leaves 10% headroom for heavier-than-planned catch)
    plot : bool — save figures to tests/outputs/
    simulate : bool — when plotting, also draw one catch realisation and plot
               planned vs simulated (set False for the planned map only)
    strategy : str — overflow response used in the simulation
    preemptive_threshold : float — trigger fraction for strategy="preemptive"
    sim_seed : int or None — seed for the simulated catch draw

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

        prob = to_heuristic_problem(inst, catch_source=catch_source,
                                    home_ports=home_ports,
                                    capacity_buffer=capacity_buffer)
        print(f"Created Problem: {prob}")
        if capacity_buffer != 1.0:
            print(f"Planning to {capacity_buffer:.0%} of capacity: "
                  f"{[f'{b.planning_capacity:.0f}/{b.capacity:.0f}' for b in prob.boats]}")

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
                # boats end at their home port, so the padding trip starts there
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
        result["capacity_buffer"] = capacity_buffer
        # return the trips too so the stochastic evaluator can reuse this solution
        # (extract here while prob is still valid)
        result["trips"] = solution_to_trips(prob)
        result["instance"] = inst

        print(f"\n--- Final Result ---")
        print(f"Method: {method}")
        print(f"Objective (total time): {result['objective']:.2f}")
        print(f"Feasible (true capacity): {result['feasible']}")
        if capacity_buffer != 1.0:
            print(f"Feasible ({capacity_buffer:.0%} buffered capacity): "
                  f"{result['feasible_planning']}")
        print(f"Iterations: {iteration}")
        print(f"Elapsed: {result['elapsed']:.1f}s")
        prob.display_solution()

    if plot:
        # outside the heuristic context — the plots only need the trip dicts
        tag = method if capacity_buffer == 1.0 else f"{method}-buf{capacity_buffer:g}"

        if simulate:
            result["realisation"] = simulate_and_plot(
                result, tag, strategy=strategy,
                preemptive_threshold=preemptive_threshold, sim_seed=sim_seed,
            )
        else:
            from unified.stochastic_eval import plot_solution

            title = (f"{method} — {inst.ns} stations, {inst.n_boats} boats, "
                     f"{len(result['trips'])} trips, "
                     f"total time {result['objective']:.1f} h")
            if capacity_buffer != 1.0:
                title += f" (planning to {capacity_buffer:.0%} capacity)"
            plot_solution(result["trips"], inst,
                          save_path=plot_output_path("route-map", tag, inst),
                          title=title)

    return result


def simulate_and_plot(result, tag, strategy="backtrack",
                      preemptive_threshold=0.8, sim_seed=None):
    """Draw one catch realisation for a solved route and plot what happens.

    Produces the same pair of figures as ``tests/test_single_realisation.py
    --plot``: the planned-vs-simulated map, and the per-trip time comparison.

    Parameters
    ----------
    result : dict — output of run_heuristic_on_gfsp (needs 'trips', 'instance')
    tag : str — filename tag, e.g. "tabu_move-buf0.9"
    strategy : str — overflow response: "backtrack", "forward" or "preemptive"
    preemptive_threshold : float — trigger fraction for the preemptive strategy
    sim_seed : int or None — seed for the catch draw (None = a different
               realisation each run)

    Returns
    -------
    dict — output of evaluate_single_realisation
    """
    from unified.stochastic_catch import CatchSimulator
    from unified.stochastic_eval import (evaluate_single_realisation,
                                         plot_realisation, plot_time_comparison)

    trips = result["trips"]
    inst = result["instance"]

    # The solver planned against its catch_source; the realisation is always
    # drawn from the historical spring-survey distributions, so a solver that
    # planned with lighter catch data will look worse here than it deserves.
    sim = CatchSimulator(seed=sim_seed)
    sim.fit()
    catch = sim.sample(n_scenarios=1)[0]

    realisation = evaluate_single_realisation(
        trips, inst, catch, strategy=strategy,
        preemptive_threshold=preemptive_threshold,
    )

    print(f"\n--- Simulated realisation (strategy={strategy}"
          f"{f', threshold={preemptive_threshold:g}' if strategy == 'preemptive' else ''}"
          f", seed={sim_seed}) ---")
    print(f"Capacity exceeded:  {realisation['capacity_exceeded']}")
    print(f"Unscheduled returns: {realisation['n_unscheduled_returns']}")
    print(f"Planned total time:  {realisation['original_total_time']:.1f} h")
    print(f"Actual total time:   {realisation['total_time']:.1f} h "
          f"(+{realisation['time_penalty']:.1f} h, "
          f"+{realisation['time_penalty'] / realisation['original_total_time']:.1%})")

    sim_tag = f"{tag}-{strategy}"
    plot_realisation(trips, inst, realisation,
                     save_path=plot_output_path("realisation", sim_tag, inst))
    plot_time_comparison(trips, realisation,
                         save_path=plot_output_path("time-comparison", sim_tag, inst))

    return realisation


def compare_methods_on_gfsp(ns=20, nv=2, cf=62.5, instance=1, time_limit=10,
                            capacity_buffer=1.0):
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
            capacity_buffer=capacity_buffer,
        )
        results.append(r)

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(compare_results(*results, labels=methods))


def compare_buffers_on_gfsp(ns=20, nv=2, cf=62.5, instance=1, method="grasp",
                            time_limit=10, buffers=(0.7, 0.8, 0.9, 1.0)):
    """Solve the same problem at several planning buffers and compare.

    Answers "is the extra travel time of planning to alpha*Q worth the
    overflows it avoids?" — the deterministic half of that question.  Run the
    returned solutions through the stochastic evaluator for the other half.
    """
    results = []
    for alpha in buffers:
        print(f"\n{'='*60}")
        print(f"Capacity buffer: {alpha:.0%}")
        print(f"{'='*60}")
        r = run_heuristic_on_gfsp(
            ns=ns, nv=nv, cf=cf, instance=instance,
            method=method, time_limit=time_limit,
            capacity_buffer=alpha,
        )
        results.append(r)

    print(f"\n{'='*60}")
    print("BUFFER COMPARISON")
    print(f"{'='*60}")
    print(compare_results(*results, labels=[f"{a:.0%}" for a in buffers]))
    return results


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
    parser.add_argument("--capacity-buffer", type=float, default=1.0,
                        help="Fraction of capacity to plan to, e.g. 0.9 for a "
                             "10%% headroom buffer (default 1.0 = full capacity)")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Save figures to tests/outputs/ (on by default)")
    parser.add_argument("--no-plot", dest="plot", action="store_false",
                        help="Skip all figures")
    parser.add_argument("--no-sim", dest="simulate", action="store_false",
                        help="Plot the planned route only, without simulating "
                             "a catch realisation")
    parser.add_argument("--strategy", default="backtrack",
                        choices=["backtrack", "forward", "preemptive"],
                        help="Overflow response in the simulation "
                             "(default: backtrack)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Preemptive return threshold, 0.0-1.0 "
                             "(only used with --strategy preemptive)")
    parser.add_argument("--sim-seed", type=int, default=None,
                        help="Seed for the simulated catch draw "
                             "(default: a new realisation each run)")
    parser.add_argument("--catch-source", default="gfsp",
                        choices=["gfsp", "heuristic", "historical"],
                        help="Catch data the solver plans against. The "
                             "simulation always draws from historical "
                             "distributions (default: gfsp)")
    parser.add_argument("--compare-buffers", nargs="*", type=float,
                        metavar="ALPHA",
                        help="Solve at several planning buffers and compare "
                             "(default: 0.7 0.8 0.9 1.0)")
    args = parser.parse_args()

    if args.compare_buffers is not None:
        compare_buffers_on_gfsp(
            ns=args.ns, nv=args.nv, cf=args.cf,
            instance=args.instance, method=args.method,
            time_limit=args.time_limit,
            buffers=tuple(args.compare_buffers) or (0.7, 0.8, 0.9, 1.0),
        )
    elif args.compare:
        compare_methods_on_gfsp(
            ns=args.ns, nv=args.nv, cf=args.cf,
            instance=args.instance, time_limit=args.time_limit,
            capacity_buffer=args.capacity_buffer,
        )
    else:
        run_heuristic_on_gfsp(
            ns=args.ns, nv=args.nv, cf=args.cf,
            instance=args.instance, method=args.method,
            time_limit=args.time_limit, init=args.init,
            capacity_buffer=args.capacity_buffer,
            catch_source=args.catch_source,
            plot=args.plot, simulate=args.simulate,
            strategy=args.strategy, preemptive_threshold=args.threshold,
            sim_seed=args.sim_seed,
        )
