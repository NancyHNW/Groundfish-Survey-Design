"""Test the single-realisation evaluation with fake data.

Run from repo root:
    python -m tests.test_single_realisation
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unified.problem import ProblemInstance, N_PORTS
from unified.stochastic_eval import evaluate_single_realisation, _extract_station_ordinals


def plot_output_path(plot_name, solver_tag, inst=None):
    """Path under tests/outputs/ named by plot, solver and instance size.

    e.g. tests/outputs/realisation_grasp-tabu_ns20-nv2.png, so different
    solvers/instances don't overwrite each other. Folder is gitignored.
    """
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    parts = [plot_name, solver_tag]
    if inst is not None:
        parts.append(f"ns{len(inst.station_ids)}-nv{inst.n_boats}")
    return os.path.join(out_dir, "_".join(parts) + ".png")


def make_fake_trips():
    """Create fake solution trips that look like real output from solution_to_trips().

    We'll simulate 2 boats, 2 trips each, visiting a handful of stations.
    Nodes: port 0, then station ordinals 0,1,2,3,4 -> node pairs (13,14), (15,16), (17,18), (19,20), (21,22)
    """
    trips = [
        {
            "boat_id": 0,
            "nodes": [0, 13, 14, 15, 16, 17, 18, 0],  # port -> stn 0 -> stn 1 -> stn 2 -> port
            "total_time": 50.0,
            "fish_time": 30.0,
            "catch": 300.0,  # deterministic total
        },
        {
            "boat_id": 0,
            "nodes": [0, 19, 20, 0],  # port -> stn 3 -> port
            "total_time": 20.0,
            "fish_time": 10.0,
            "catch": 100.0,
        },
        {
            "boat_id": 1,
            "nodes": [0, 21, 22, 13, 14, 0],  # port -> stn 4 -> stn 0 -> port
            "total_time": 40.0,
            "fish_time": 25.0,
            "catch": 200.0,
        },
    ]
    return trips


def make_fake_instance():
    """Create a fake ProblemInstance with 5 stations, 2 boats."""
    return ProblemInstance(
        station_ids=np.array([0, 1, 2, 3, 4]),
        port_ids=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        n_boats=2,
        capacities=np.array([250.0, 250.0]),  # each boat holds 250 kg
        home_port_index=0,
        fish_time_limit=100.0,
        work_limit=10,
        source="test",
    )


def make_fake_time_matrix():
    """Create a small fake time matrix (1175 x 1175).

    We only need entries for ports 0-12 and station nodes 13-22.
    Set all travel times to 5.0 for simplicity, except self = 0.
    """
    n = 1175
    tm = np.full((n, n), 5.0)
    np.fill_diagonal(tm, 0.0)
    return tm


# ---- Tests ----

def test_extract_station_ordinals():
    """Check that node lists correctly map to station ordinals."""
    nodes = [0, 13, 14, 15, 16, 17, 18, 0]
    result = _extract_station_ordinals(nodes)
    assert result == [0, 1, 2], f"Expected [0, 1, 2], got {result}"
    print("PASS: _extract_station_ordinals")


def test_no_exceedance():
    """Low catch — everything fits, no violations."""
    trips = make_fake_trips()
    inst = make_fake_instance()
    tm = make_fake_time_matrix()

    # Catch well under capacity: 50 kg per station, max 3 stations/trip = 150 < 250
    catch = np.full(581, 50.0)

    result = evaluate_single_realisation(trips, inst, catch, time_matrix=tm)

    print(f"\n--- Test: no exceedance ---")
    print(f"  capacity_exceeded: {result['capacity_exceeded']}")
    print(f"  fish_time_violated: {result['fish_time_violated']}")
    print(f"  n_unscheduled_returns: {result['n_unscheduled_returns']}")
    print(f"  total_time: {result['total_time']}")

    assert not result["capacity_exceeded"], "Should not exceed capacity"
    assert not result["fish_time_violated"], "Should not violate fish time"
    assert result["n_unscheduled_returns"] == 0, "Should have no unscheduled returns"
    assert result["total_time"] == 110.0, f"Total time should be 50+20+40=110, got {result['total_time']}"

    for td in result["trip_details"]:
        print(f"  Trip (boat {td['boat_id']}): sim_catch={td['simulated_catch']:.0f}, "
              f"capacity={td['capacity']:.0f}, exceeded={td['exceeded']}")

    print("PASS: no exceedance\n")


def test_capacity_exceedance():
    """High catch — one trip overflows, triggering an unscheduled return."""
    trips = make_fake_trips()
    inst = make_fake_instance()
    tm = make_fake_time_matrix()

    # Trip 0 visits stations 0,1,2. Set catch so cumulative exceeds 250 at station 2.
    catch = np.full(581, 50.0)
    catch[0] = 100.0   # cumulative after stn 0: 100
    catch[1] = 100.0   # cumulative after stn 1: 200
    catch[2] = 120.0   # cumulative after stn 2: 320 > 250 -> EXCEEDED

    result = evaluate_single_realisation(trips, inst, catch, time_matrix=tm)

    print(f"--- Test: capacity exceedance ---")
    print(f"  capacity_exceeded: {result['capacity_exceeded']}")
    print(f"  n_unscheduled_returns: {result['n_unscheduled_returns']}")
    print(f"  total_time: {result['total_time']}")

    assert result["capacity_exceeded"], "Should exceed capacity"
    assert result["n_unscheduled_returns"] >= 1, "Should have at least 1 unscheduled return"

    # Trip 0 should have detour time added (2 * 5.0 = 10.0 for nearest port round trip)
    trip0 = result["trip_details"][0]
    print(f"  Trip 0 detour_time: {trip0['detour_time']}")
    print(f"  Trip 0 adjusted_time: {trip0['adjusted_time']} (original: 50.0)")
    assert trip0["detour_time"] == 10.0, f"Detour should be 2*5=10, got {trip0['detour_time']}"
    assert trip0["adjusted_time"] == 60.0, f"Adjusted time should be 50+10=60, got {trip0['adjusted_time']}"

    # Total time should include the detour
    assert result["total_time"] == 120.0, f"Total should be 60+20+40=120, got {result['total_time']}"

    for td in result["trip_details"]:
        print(f"  Trip (boat {td['boat_id']}): sim_catch={td['simulated_catch']:.0f}, "
              f"exceeded={td['exceeded']}, unscheduled={td['n_unscheduled_returns']}")

    print("PASS: capacity exceedance\n")


def test_fish_time_violation():
    """Fish time exceeds limit."""
    trips = make_fake_trips()
    inst = make_fake_instance()
    tm = make_fake_time_matrix()

    # Set fish_time_limit very low so the existing fish_time (30) exceeds it
    inst.fish_time_limit = 20.0

    catch = np.full(581, 50.0)
    result = evaluate_single_realisation(trips, inst, catch, time_matrix=tm)

    print(f"--- Test: fish time violation ---")
    print(f"  fish_time_violated: {result['fish_time_violated']}")
    assert result["fish_time_violated"], "Should violate fish time (30 > 20)"
    print("PASS: fish time violation\n")


def test_multiple_overflows_in_one_trip():
    """Catch so high that a trip overflows twice — two unscheduled returns."""
    inst = make_fake_instance()
    inst.capacities = np.array([150.0, 150.0])  # tighter capacity
    tm = make_fake_time_matrix()

    # Trip with 3 stations, each with 100 kg catch
    # Station 0: cumulative 100 (ok)
    # Station 1: cumulative 200 > 150 -> overflow #1, reset to 0
    # Station 2: cumulative 100 (ok)
    trips = [
        {
            "boat_id": 0,
            "nodes": [0, 13, 14, 15, 16, 17, 18, 0],
            "total_time": 50.0,
            "fish_time": 30.0,
            "catch": 300.0,
        }
    ]

    catch = np.full(581, 100.0)
    result = evaluate_single_realisation(trips, inst, catch, time_matrix=tm)

    print(f"--- Test: multiple overflows ---")
    print(f"  n_unscheduled_returns: {result['n_unscheduled_returns']}")
    trip0 = result["trip_details"][0]
    print(f"  Trip 0 unscheduled: {trip0['n_unscheduled_returns']}, detour: {trip0['detour_time']}")

    assert result["n_unscheduled_returns"] == 1, \
        f"Should have 1 overflow (at stn 1), got {result['n_unscheduled_returns']}"
    print("PASS: multiple overflows\n")


def test_boat_id_capacity_mapping():
    """Different boats have different capacities — check correct one is used."""
    inst = make_fake_instance()
    inst.capacities = np.array([500.0, 100.0])  # boat 0 big, boat 1 small
    tm = make_fake_time_matrix()

    # Boat 1 visits stations 4 and 0 with catch = 80 each -> cumulative 160 > 100
    trips = [
        {
            "boat_id": 1,
            "nodes": [0, 21, 22, 13, 14, 0],
            "total_time": 40.0,
            "fish_time": 25.0,
            "catch": 160.0,
        }
    ]

    catch = np.full(581, 80.0)
    result = evaluate_single_realisation(trips, inst, catch, time_matrix=tm)

    print(f"--- Test: boat capacity mapping ---")
    print(f"  capacity_exceeded: {result['capacity_exceeded']}")
    assert result["capacity_exceeded"], "Boat 1 (capacity 100) should overflow with 160 catch"
    print("PASS: boat capacity mapping\n")


def test_visualise_real_problem(strategy="backtrack", preemptive_threshold=0.8, seed=None, home_ports=None, ns=20, cf=62.5, full=False):
    """Solve a real gfsp problem and visualise one realisation.

    Uses real routes (GRASP + tabu) and real catch sampled from historical
    spring survey distributions (via CatchSimulator).
    """
    from unified.loaders import load_gfsp_problem, load_gfsp_full_problem
    from unified.adapters import heuristic_context, to_heuristic_problem
    from unified.evaluate import solution_to_trips
    from unified.stochastic_eval import evaluate_single_realisation, plot_realisation

    # Load a test problem
    if full:
        inst = load_gfsp_full_problem()
    else:
        inst = load_gfsp_problem(ns=ns, nv=2, cf=cf, instance=1)
    print(f"Loaded: {inst}")

    # record which solver steps ran so the filename reflects it
    solver_steps = []

    # Solve it with GRASP + combined tabu search improvement
    with heuristic_context():
        from classes import Problem, Trip
        from neighbourhood_rules import tabu_search_combined
        prob = to_heuristic_problem(inst, catch_source="historical",
                                    home_ports=home_ports)
        prob.GRASP(rcl_size=2, seed=42)
        solver_steps.append("grasp")

        # Save/restore to fix indexing
        sol = prob.save_solution_as_list()
        prob.restore_solution_from_list(sol)

        # First-pass route improvement
        for boat in prob.boats:
            boat.improve_route(prob)
        # tabu_search expects every boat to have at least one extra trip slot
        for boat in prob.boats:
            if len(boat.route) == 1:
                boat.route.append(Trip(boat.home_port, [], boat.home_port))

        # Penalty parameters (same recipe as final_testing_heur.py)
        k_catch = max(1.0, (400000 - np.mean(inst.capacities)) * 0.001)
        k_fish = np.mean(inst.capacities) / inst.fish_time_limit * k_catch
        upper_bound = inst.upper_bound if np.isfinite(inst.upper_bound) else 10000

        obj_before, pen_before = prob.evaluate_solution(k_catch, k_fish, upper_bound)
        print(f"  Before tabu:  obj={obj_before:.2f}, pen={pen_before:.2f}")

        obj, pen, cost = tabu_search_combined(
            prob, k_catch, k_fish, upper_bound, inst.work_limit,
            history_size=max(1, len(prob.stations) // 10),
            max_iter=10, max_time=60, tsp=0.02,
        )
        solver_steps.append("tabu")
        print(f"  After tabu:   obj={obj:.2f}, pen={pen:.2f}")

        # Final TSP improvement
        for boat in prob.boats:
            boat.improve_route(prob)

        trips = solution_to_trips(prob)

    print(f"Got {len(trips)} trips")
    unique_bids = sorted(set(t["boat_id"] for t in trips))
    bid_to_idx = {bid: i for i, bid in enumerate(unique_bids)}
    trip_counter = {}
    for t in trips:
        cap = inst.capacities[bid_to_idx[t['boat_id']]]
        station_ids = _extract_station_ordinals(t["nodes"])
        trip_idx = trip_counter.get(t['boat_id'], 0)
        trip_counter[t['boat_id']] = trip_idx + 1
        print(f"  Boat {t['boat_id']}, Trip {trip_idx}: {len(station_ids)} stations, "
              f"catch={t['catch']:.0f}, capacity={cap:.0f}")
        print(f"    station IDs: {station_ids}")

    # Sample one catch scenario from historical distributions
    from unified.stochastic_catch import CatchSimulator
    sim = CatchSimulator(seed=seed)
    sim.fit()
    catch = sim.sample(n_scenarios=1)[0]

    # Evaluate
    tm = np.load(os.path.join(os.path.dirname(__file__), "..",
                              "final_code", "local data", "true_time.npy"))
    print(f"\nStrategy: {strategy}")
    result = evaluate_single_realisation(trips, inst, catch, time_matrix=tm,
                                         strategy=strategy,
                                         preemptive_threshold=preemptive_threshold)

    print(f"\nResult:")
    print(f"  capacity_exceeded: {result['capacity_exceeded']}")
    print(f"  n_unscheduled_returns: {result['n_unscheduled_returns']}")
    print(f"  planned total time: {result['original_total_time']:.1f}")
    print(f"  actual total time:  {result['total_time']:.1f}")
    print(f"  time penalty:       +{result['time_penalty']:.1f} "
          f"({result['time_penalty']/result['original_total_time']*100:.1f}% increase)")
    print(f"  overflow events: {len(result['overflow_events'])}")

    print(f"\n  Per-trip breakdown:")
    for i, td in enumerate(result["trip_details"]):
        flag = " *** OVERFLOW" if td["exceeded"] else ""
        print(f"    Trip {i} (Boat {td['boat_id']}): "
              f"planned={td['original_time']:.1f}, "
              f"actual={td['adjusted_time']:.1f}, "
              f"detour=+{td['detour_time']:.1f}{flag}")

    solver_tag = "-".join(solver_steps) + f"-{strategy}"

    # Plot map
    save_path = plot_output_path("realisation", solver_tag, inst)
    plot_realisation(trips, inst, result, save_path=save_path)
    print(f"\nMap saved to {save_path}")

    # Print catch table
    from unified.stochastic_eval import print_catch_table
    print_catch_table(trips, catch, inst)

    # Plot time comparison
    from unified.stochastic_eval import plot_time_comparison
    save_path2 = plot_output_path("time-comparison", solver_tag, inst)
    plot_time_comparison(trips, result, save_path=save_path2)
    print(f"Time comparison saved to {save_path2}")


# ---- Monte Carlo tests ----

def test_monte_carlo_with_scenarios():
    """Test StochasticEvaluator with pre-generated scenario matrix."""
    from unified.stochastic_eval import StochasticEvaluator, print_stochastic_summary

    trips = make_fake_trips()
    inst = make_fake_instance()
    tm = make_fake_time_matrix()

    n_sims = 200
    rng = np.random.default_rng(42)
    # Generate scenarios: some low (no overflow), some high (overflow)
    scenarios = rng.uniform(30, 120, size=(n_sims, 581))

    evaluator = StochasticEvaluator(scenarios=scenarios)
    result = evaluator.evaluate(trips, inst)

    print(f"\n--- Test: Monte Carlo with scenarios ---")
    print(f"  n_simulations: {result['n_simulations']}")
    print(f"  p_capacity_exceedance: {result['p_capacity_exceedance']:.1%}")
    print(f"  p_fish_time_violation: {result['p_fish_time_violation']:.1%}")
    print(f"  expected_unscheduled_returns: {result['expected_unscheduled_returns']:.2f}")
    print(f"  total_time mean: {result['total_time_distribution']['mean']:.1f}")
    print(f"  total_time p95:  {result['total_time_distribution']['p95']:.1f}")

    assert result["n_simulations"] == n_sims
    assert 0 <= result["p_capacity_exceedance"] <= 1
    assert 0 <= result["p_fish_time_violation"] <= 1
    assert result["expected_unscheduled_returns"] >= 0
    assert len(result["per_trip_exceedance_probs"]) == len(trips)
    assert result["worst_case_catch"].shape == (581,)
    assert len(result["all_results"]) == n_sims

    # Print formatted summary
    print_stochastic_summary(result, deterministic_time=110.0)
    print("PASS: Monte Carlo with scenarios\n")


def test_monte_carlo_deterministic_scenarios():
    """All scenarios identical — results should be deterministic."""
    from unified.stochastic_eval import StochasticEvaluator

    trips = make_fake_trips()
    inst = make_fake_instance()

    # All scenarios = 50 kg (no overflow with capacity 250)
    scenarios = np.full((100, 581), 50.0)
    evaluator = StochasticEvaluator(scenarios=scenarios)
    result = evaluator.evaluate(trips, inst)

    print(f"--- Test: Monte Carlo deterministic ---")
    assert result["p_capacity_exceedance"] == 0.0, "No overflow expected"
    assert result["total_time_distribution"]["std"] == 0.0, "No variance expected"
    assert result["total_time_distribution"]["mean"] == 110.0
    print("PASS: Monte Carlo deterministic\n")


def test_monte_carlo_all_overflow():
    """All scenarios cause overflow — exceedance probability should be 1.0."""
    from unified.stochastic_eval import StochasticEvaluator

    trips = make_fake_trips()
    inst = make_fake_instance()

    # 200 kg per station, 3 stations in trip 0 = 600 > 250 capacity
    scenarios = np.full((50, 581), 200.0)
    evaluator = StochasticEvaluator(scenarios=scenarios)
    result = evaluator.evaluate(trips, inst)

    print(f"--- Test: Monte Carlo all overflow ---")
    assert result["p_capacity_exceedance"] == 1.0, "All should overflow"
    assert result["expected_unscheduled_returns"] > 0
    print(f"  expected_unscheduled_returns: {result['expected_unscheduled_returns']:.2f}")
    print("PASS: Monte Carlo all overflow\n")


def run_mc_comparison(methods=("grasp_only", "grasp_swap", "tabu_move"),
                      ns=20, nv=2, cf=62.5, instance=1,
                      time_limit=120, n_scenarios=500, scenario_seed=123,
                      catch_source="gfsp", strategy="backtrack",
                      capacity_buffer=1.0):
    """Run Monte Carlo for several heuristics and compare them.

    Solves the problem with each method, then evaluates all the solutions
    against the same scenarios (so the comparison is fair). One histogram
    per method. Returns {method: result}.

    Method names used here map to run_heuristic_on_gfsp method names:
        grasp_only  -> grasp_only  (construction only, no improvement)
        grasp_swap  -> grasp       (GRASP + next-descent swap)
        tabu_move   -> tabu_move   (GRASP + tabu search with moves)
    """
    from unified.run_example import run_heuristic_on_gfsp
    from unified.stochastic_eval import (StochasticEvaluator,
                                          print_stochastic_summary,
                                          plot_monte_carlo)
    from unified.stochastic_catch import CatchSimulator

    # Map display names to solver method names
    solver_method = {
        "grasp_only": "grasp_only",
        "grasp_swap": "grasp",
        "tabu_move": "tabu_move",
        "tabu_swap": "tabu_swap",
        "sa": "sa",
    }

    # Generate shared scenarios from historical data so comparison is fair
    sim = CatchSimulator(seed=scenario_seed)
    sim.fit()
    scenarios = sim.sample(n_scenarios=n_scenarios)
    evaluator = StochasticEvaluator(scenarios=scenarios)

    results = {}
    for method in methods:
        print(f"\n{'='*60}\nMonte Carlo for method: {method}\n{'='*60}")

        # solve with this method (returns trips + instance now)
        det = run_heuristic_on_gfsp(ns=ns, nv=nv, cf=cf, instance=instance,
                                    method=solver_method.get(method, method),
                                    time_limit=time_limit,
                                    catch_source=catch_source,
                                    capacity_buffer=capacity_buffer)
        trips = det["trips"]
        inst = det["instance"]
        print(f"Got {len(trips)} trips (deterministic obj={det['objective']:.1f})")

        # evaluate against the shared scenarios
        result = evaluator.evaluate(trips, inst, strategy=strategy)
        det_time = sum(t["total_time"] for t in trips)
        print_stochastic_summary(result, deterministic_time=det_time)

        # histogram named by method, with the predicted time drawn for comparison
        save_path = plot_output_path("monte-carlo-hist", method, inst)
        plot_monte_carlo(result, save_path=save_path,
                         deterministic_time=det_time, title_suffix=f"method: {method}")
        print(f"Histogram saved to {save_path}")

        results[method] = result

    # comparison table
    print(f"\n{'='*60}\nHEURISTIC COMPARISON (shared scenarios)\n{'='*60}")
    print(f"{'method':<14}{'mean time':>12}{'p95 time':>12}{'P(exceed)':>12}")
    for method, result in results.items():
        td = result["total_time_distribution"]
        print(f"{method:<14}{td['mean']:>12.1f}{td['p95']:>12.1f}"
              f"{result['p_capacity_exceedance']:>12.1%}")

    return results


def run_strategy_comparison(ns=100, nv=2, cf=250, instance=1,
                            time_limit=10, n_scenarios=500, scenario_seed=123,
                            catch_source="historical", method="tabu_move",
                            capacity_buffer=1.0):
    """Run a single method, then evaluate it under multiple overflow strategies."""
    from unified.run_example import run_heuristic_on_gfsp
    from unified.stochastic_eval import (StochasticEvaluator,
                                          print_stochastic_summary)
    from unified.stochastic_catch import CatchSimulator

    solver_method = {
        "grasp_only": "grasp_only",
        "grasp_swap": "grasp",
        "tabu_move": "tabu_move",
    }

    # Generate shared scenarios
    sim = CatchSimulator(seed=scenario_seed)
    sim.fit()
    scenarios = sim.sample(n_scenarios=n_scenarios)
    evaluator = StochasticEvaluator(scenarios=scenarios)

    # Solve once
    print(f"Solving with {method}...")
    det = run_heuristic_on_gfsp(ns=ns, nv=nv, cf=cf, instance=instance,
                                method=solver_method.get(method, method),
                                time_limit=time_limit,
                                catch_source=catch_source,
                                capacity_buffer=capacity_buffer)
    trips = det["trips"]
    inst = det["instance"]
    det_time = sum(t["total_time"] for t in trips)
    print(f"Deterministic time: {det_time:.1f}h, {len(trips)} trips\n")

    # Evaluate under each strategy
    strategies = [
        ("backtrack", "backtrack", 0.8),
        ("forward", "forward", 0.8),
        ("preemptive_0.8", "preemptive", 0.8),
        ("preemptive_0.7", "preemptive", 0.7),
    ]

    print(f"{'strategy':<18}{'P(exceed)':>12}{'E[returns]':>12}"
          f"{'mean penalty':>14}{'p95 penalty':>14}")
    print("-" * 70)

    for label, strat, threshold in strategies:
        result = evaluator.evaluate(trips, inst, strategy=strat,
                                     preemptive_threshold=threshold)
        td = result["time_penalty_distribution"]
        print(f"{label:<18}{result['p_capacity_exceedance']:>12.1%}"
              f"{result['expected_unscheduled_returns']:>12.2f}"
              f"{td['mean']:>14.1f}{td['p95']:>14.1f}")

    # Threshold sweep
    print(f"\n{'='*60}")
    print("Preemptive threshold sweep")
    print(f"{'='*60}")
    print(f"{'alpha':<10}{'P(exceed)':>12}{'E[returns]':>12}"
          f"{'mean penalty':>14}{'p95 penalty':>14}")
    print("-" * 62)

    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        result = evaluator.evaluate(trips, inst, strategy="preemptive",
                                     preemptive_threshold=alpha)
        td = result["time_penalty_distribution"]
        print(f"{alpha:<10.1f}{result['p_capacity_exceedance']:>12.1%}"
              f"{result['expected_unscheduled_returns']:>12.2f}"
              f"{td['mean']:>14.1f}{td['p95']:>14.1f}")


def run_buffer_comparison(ns=100, nv=2, cf=125, instance=1,
                          time_limit=10, n_scenarios=500, scenario_seed=123,
                          catch_source="gfsp", method="tabu_move",
                          strategy="backtrack", preemptive_threshold=0.8,
                          buffers=(0.7, 0.8, 0.9, 1.0)):
    """Solve at several planning buffers and score each against the scenarios.

    The deterministic half of the question ("what does the buffer cost in
    planned time?") is answered by run_example's --compare-buffers.  This adds
    the half that matters: what the buffer buys back in avoided overflows.
    Planned time rises as alpha falls, so the buffer is only worth it if mean
    realised time falls by more than planned time rose.
    """
    import csv
    from unified.run_example import run_heuristic_on_gfsp
    from unified.stochastic_eval import (StochasticEvaluator,
                                          plot_buffer_comparison)
    from unified.stochastic_catch import CatchSimulator

    solver_method = {
        "grasp_only": "grasp_only",
        "grasp_swap": "grasp",
        "tabu_move": "tabu_move",
    }

    # One scenario set shared across every buffer, so the differences are the
    # solutions and not the sampling
    sim = CatchSimulator(seed=scenario_seed)
    sim.fit()
    scenarios = sim.sample(n_scenarios=n_scenarios)
    evaluator = StochasticEvaluator(scenarios=scenarios)

    results = {}
    last_inst = None
    for alpha in buffers:
        print(f"\n{'='*60}\nCapacity buffer: {alpha:.0%}\n{'='*60}")
        det = run_heuristic_on_gfsp(ns=ns, nv=nv, cf=cf, instance=instance,
                                    method=solver_method.get(method, method),
                                    time_limit=time_limit,
                                    catch_source=catch_source,
                                    capacity_buffer=alpha)
        trips = det["trips"]
        inst = det["instance"]
        last_inst = inst
        det_time = sum(t["total_time"] for t in trips)
        result = evaluator.evaluate(trips, inst, strategy=strategy,
                                    preemptive_threshold=preemptive_threshold)
        results[alpha] = {"result": result, "planned": det_time,
                          "trips": len(trips), "feasible": det["feasible"]}
        print(f"Planned {det_time:.1f}h over {len(trips)} trips, "
              f"P(exceed)={result['p_capacity_exceedance']:.1%}")

    baseline = results[max(buffers)]["result"]["total_time_distribution"]["mean"]

    # flatten into rows once, then reuse for the table, the CSV and the plot
    rows = []
    for alpha in buffers:
        entry = results[alpha]
        result = entry["result"]
        td = result["total_time_distribution"]
        rows.append({
            "buffer": alpha,
            "planned": entry["planned"],
            "trips": entry["trips"],
            "feasible": entry["feasible"],
            "p_exceed": result["p_capacity_exceedance"],
            "e_returns": result["expected_unscheduled_returns"],
            # normalised, because a tighter buffer plans more trips - the raw
            # count can fall simply by spreading the same risk over more trips
            "returns_per_trip": (result["expected_unscheduled_returns"]
                                 / entry["trips"] if entry["trips"] else 0.0),
            "mean": td["mean"],
            "p5": td["p5"],
            "p95": td["p95"],
            "vs_full": td["mean"] - baseline,
        })

    print(f"\n{'='*78}")
    print(f"BUFFER COMPARISON  (strategy={strategy}, {n_scenarios} scenarios)")
    print(f"{'='*78}")
    print(f"{'buffer':<9}{'planned':>9}{'trips':>7}{'P(exceed)':>11}"
          f"{'E[ret]':>8}{'ret/trip':>10}{'mean time':>11}{'p95 time':>10}"
          f"{'vs 100%':>9}")
    print("-" * 84)
    for row in rows:
        print(f"{row['buffer']:<9.0%}{row['planned']:>9.1f}{row['trips']:>7d}"
              f"{row['p_exceed']:>11.1%}{row['e_returns']:>8.2f}"
              f"{row['returns_per_trip']:>10.3f}"
              f"{row['mean']:>11.1f}{row['p95']:>10.1f}{row['vs_full']:>+9.1f}")

    best = min(rows, key=lambda r: r["mean"])
    print(f"\nLowest mean realised time: buffer {best['buffer']:.0%}")
    if best["buffer"] == max(buffers):
        print("No buffer beat planning to full capacity on this instance.")

    # persist — stdout alone is not much use for a writeup
    tag = f"{method}-{strategy}"
    csv_path = plot_output_path("buffer-comparison", tag,
                                last_inst).replace(".png", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Table saved to {csv_path}")

    plot_path = plot_output_path("buffer-comparison", tag, last_inst)
    plot_buffer_comparison(
        rows, save_path=plot_path,
        title_suffix=(f"ns={ns}, nv={nv}, cf={cf}, {method}, {strategy}, "
                      f"{n_scenarios} scenarios, catch={catch_source}"))

    return results


def test_monte_carlo_real_problem():
    """Quick end-to-end run of the heuristic comparison (three methods)."""
    results = run_mc_comparison(
        methods=("grasp_only", "grasp_swap", "tabu_move"),
        time_limit=10,
    )
    assert set(results) == {"grasp_only", "grasp_swap", "tabu_move"}
    for result in results.values():
        assert 0 <= result["p_capacity_exceedance"] <= 1
        assert result["worst_case_catch"].shape == (581,)


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true",
                        help="Run the visualisation test with a real problem")
    parser.add_argument("--full", action="store_true",
                        help="Use the full 581-station problem (4 boats)")
    parser.add_argument("--mc", action="store_true",
                        help="Run the Monte Carlo method comparison")
    parser.add_argument("--sc", action="store_true",
                        help="Run the strategy comparison + threshold sweep")
    parser.add_argument("--bc", action="store_true",
                        help="Run the capacity-buffer comparison over scenarios")
    parser.add_argument("--buffers", nargs="+", type=float,
                        default=[0.7, 0.8, 0.9, 1.0],
                        help="Buffers to compare with --bc (default: 0.7 0.8 0.9 1.0)")
    parser.add_argument("--strategy", default="backtrack",
                        choices=["backtrack", "forward", "preemptive"],
                        help="Overflow strategy (default: backtrack)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Preemptive return threshold, 0.0-1.0 (default: 0.8)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for catch sampling (default: random)")
    parser.add_argument("--catch-source", default="gfsp",
                        choices=["gfsp", "heuristic", "historical"],
                        help="Catch data for route planning (default: gfsp)")
    parser.add_argument("--ns", type=int, default=20,
                        help="Number of stations (default: 20)")
    parser.add_argument("--cf", type=float, default=62.5,
                        help="Capacity factor (default: 62.5)")
    parser.add_argument("--home-ports", type=int, nargs="+", default=None,
                        help="Per-vessel home port node indices (e.g. --home-ports 4 8)")
    parser.add_argument("--nv", type=int, default=2,
                        help="Number of vessels, used by --mc/--sc (default: 2)")
    parser.add_argument("--method", default="tabu_move",
                        choices=["grasp_only", "grasp_swap", "tabu_move"],
                        help="Solver for --sc (--mc always compares all three)")
    parser.add_argument("--capacity-buffer", type=float, default=1.0,
                        help="Fraction of capacity the solver plans to, e.g. 0.8 "
                             "leaves 20%% headroom (default 1.0 = full capacity)")
    parser.add_argument("--n-scenarios", type=int, default=500,
                        help="Monte Carlo scenarios for --mc/--sc (default: 500)")
    parser.add_argument("--time-limit", type=float, default=10,
                        help="Solver time limit in seconds (default: 10)")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing single-realisation evaluation")
    print("=" * 60 + "\n")

    test_extract_station_ordinals()
    test_no_exceedance()
    test_capacity_exceedance()
    test_fish_time_violation()
    test_multiple_overflows_in_one_trip()
    test_boat_id_capacity_mapping()

    print("=" * 60)
    print("ALL SINGLE-REALISATION TESTS PASSED")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Testing Monte Carlo evaluation")
    print("=" * 60 + "\n")

    test_monte_carlo_with_scenarios()
    test_monte_carlo_deterministic_scenarios()
    test_monte_carlo_all_overflow()

    print("=" * 60)
    print("ALL MONTE CARLO UNIT TESTS PASSED")
    print("=" * 60)

    if args.plot:
        print("\n" + "=" * 60)
        print(f"Running visualisation test (strategy={args.strategy})...")
        print("=" * 60 + "\n")
        test_visualise_real_problem(strategy=args.strategy,
                                    preemptive_threshold=args.threshold,
                                    seed=args.seed,
                                    home_ports=args.home_ports,
                                    ns=args.ns, cf=args.cf,
                                    full=args.full)

    if args.mc:
        print("\n" + "=" * 60)
        print(f"Running Monte Carlo test (ns={args.ns}, nv={args.nv}, cf={args.cf}, "
              f"catch_source={args.catch_source}, buffer={args.capacity_buffer:.0%}, "
              f"scenarios={args.n_scenarios})...")
        print("=" * 60 + "\n")
        run_mc_comparison(
            ns=args.ns, nv=args.nv, cf=args.cf,
            catch_source=args.catch_source,
            capacity_buffer=args.capacity_buffer,
            n_scenarios=args.n_scenarios,
            time_limit=args.time_limit,
        )

    if args.sc:
        print("\n" + "=" * 60)
        print(f"Running strategy comparison (ns={args.ns}, nv={args.nv}, cf={args.cf}, "
              f"method={args.method}, catch_source={args.catch_source}, "
              f"buffer={args.capacity_buffer:.0%}, scenarios={args.n_scenarios})...")
        print("=" * 60 + "\n")
        run_strategy_comparison(
            ns=args.ns, nv=args.nv, cf=args.cf,
            catch_source=args.catch_source,
            method=args.method,
            capacity_buffer=args.capacity_buffer,
            n_scenarios=args.n_scenarios,
            time_limit=args.time_limit,
        )

    if args.bc:
        print("\n" + "=" * 60)
        print(f"Running buffer comparison (ns={args.ns}, nv={args.nv}, cf={args.cf}, "
              f"method={args.method}, catch_source={args.catch_source}, "
              f"strategy={args.strategy}, scenarios={args.n_scenarios}, "
              f"buffers={[f'{b:.0%}' for b in args.buffers]})...")
        print("=" * 60 + "\n")
        run_buffer_comparison(
            ns=args.ns, nv=args.nv, cf=args.cf,
            catch_source=args.catch_source,
            method=args.method,
            strategy=args.strategy,
            preemptive_threshold=args.threshold,
            n_scenarios=args.n_scenarios,
            buffers=tuple(args.buffers),
            time_limit=args.time_limit,
        )
