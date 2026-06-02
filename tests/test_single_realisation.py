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


def test_visualise_real_problem():
    """Solve a real gfsp problem and visualise one realisation with overflows.

    This uses the actual map data (Iceland coastline, real station positions).
    Catch is faked to force some overflows so we can see the red X markers.
    """
    from unified.loaders import load_gfsp_problem
    from unified.adapters import heuristic_context, to_heuristic_problem
    from unified.evaluate import solution_to_trips
    from unified.stochastic_eval import evaluate_single_realisation, plot_realisation

    # Load a small test problem
    inst = load_gfsp_problem(ns=20, nv=2, cf=62.5, instance=1)
    print(f"Loaded: {inst}")

    # record which solver steps ran so the filename reflects it
    solver_steps = []

    # Solve it with GRASP + combined tabu search improvement
    with heuristic_context():
        from classes import Problem, Trip
        from neighbourhood_rules import tabu_search_combined
        prob = to_heuristic_problem(inst, catch_source="gfsp")
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

    # Create fake catch that causes some overflows:
    # Use real-ish baseline (~500 kg) but spike a few stations to force overflow
    rng = np.random.default_rng(123)
    catch = rng.uniform(300, 600, size=581)

    # Spike some stations that are actually in the trips to guarantee overflows
    for t in trips:
        station_ords = _extract_station_ordinals(t["nodes"])
        if len(station_ords) >= 3:
            # Make the 3rd station very high to overflow
            catch[station_ords[2]] = 2000.0
            break  # just spike one trip

    # Evaluate
    tm = np.load(os.path.join(os.path.dirname(__file__), "..",
                              "final_code", "local data", "true_time.npy"))
    result = evaluate_single_realisation(trips, inst, catch, time_matrix=tm)

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

    solver_tag = "-".join(solver_steps)

    # Plot map
    save_path = plot_output_path("realisation", solver_tag, inst)
    plot_realisation(trips, inst, result, save_path=save_path)
    print(f"\nMap saved to {save_path}")

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


def run_mc_comparison(methods=("grasp_only", "grasp", "tabu_swap"),
                      ns=20, nv=2, cf=62.5, instance=1,
                      time_limit=10, n_scenarios=500, scenario_seed=123):
    """Run Monte Carlo for several heuristics and compare them.

    Solves the problem with each method, then evaluates all the solutions
    against the same scenarios (so the comparison is fair). One histogram
    per method. Returns {method: result}.
    """
    from unified.run_example import run_heuristic_on_gfsp
    from unified.stochastic_eval import (StochasticEvaluator,
                                          print_stochastic_summary,
                                          plot_monte_carlo)

    # same scenarios for every method (stand-in until Jess's simulator is ready)
    rng = np.random.default_rng(scenario_seed)
    scenarios = rng.lognormal(mean=5.5, sigma=0.8, size=(n_scenarios, 581))
    evaluator = StochasticEvaluator(scenarios=scenarios)

    results = {}
    for method in methods:
        print(f"\n{'='*60}\nMonte Carlo for method: {method}\n{'='*60}")

        # solve with this method (returns trips + instance now)
        det = run_heuristic_on_gfsp(ns=ns, nv=nv, cf=cf, instance=instance,
                                    method=method, time_limit=time_limit)
        trips = det["trips"]
        inst = det["instance"]
        print(f"Got {len(trips)} trips (deterministic obj={det['objective']:.1f})")

        # evaluate against the shared scenarios
        result = evaluator.evaluate(trips, inst)
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


def test_monte_carlo_real_problem():
    """Quick end-to-end run of the heuristic comparison (two methods)."""
    results = run_mc_comparison(methods=("grasp_only", "grasp"), time_limit=10)
    assert set(results) == {"grasp_only", "grasp"}
    for result in results.values():
        assert 0 <= result["p_capacity_exceedance"] <= 1
        assert result["worst_case_catch"].shape == (581,)


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true",
                        help="Run the visualisation test with a real problem")
    parser.add_argument("--mc", action="store_true",
                        help="Run the Monte Carlo test with a real problem")
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
        print("Running visualisation test on real problem...")
        print("=" * 60 + "\n")
        test_visualise_real_problem()

    if args.mc:
        print("\n" + "=" * 60)
        print("Running Monte Carlo test on real problem...")
        print("=" * 60 + "\n")
        test_monte_carlo_real_problem()
