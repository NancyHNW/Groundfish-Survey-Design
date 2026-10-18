"""Tests for the real 581-station survey problem.

The module is marked ``full`` because it loads the historical catch fit.
Run it with:

    python -m pytest tests/test_full_problem.py -v -m full
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unified.adapters import heuristic_context, to_heuristic_problem
from unified.evaluate import evaluate_heuristic_solution, solution_to_trips
from unified.loaders import load_gfsp_full_problem
from unified.problem import N_PORTS
from unified.stochastic_catch import CatchSimulator
from unified.stochastic_eval import (DETOUR_STRATEGIES, STRATEGIES,
                                     StochasticEvaluator)

pytestmark = pytest.mark.full

SEED = 42
N_STATIONS = 581


def _node_to_station(node):
    return (node - N_PORTS) // 2


@pytest.fixture(scope="module")
def solved():
    """Solve the full problem once and share the result across tests."""
    inst = load_gfsp_full_problem()

    with heuristic_context():
        prob = to_heuristic_problem(inst, catch_source="gfsp")
        prob.generate_initial_solution(seed=SEED)

        sol = prob.save_solution_as_list()
        prob.restore_solution_from_list(sol)

        k_catch = max(1.0, (400000 - np.mean(inst.capacities)) * 0.001)
        k_fish = np.mean(inst.capacities) / inst.fish_time_limit * k_catch
        obj_raw, penalty = prob.evaluate_solution(k_catch, k_fish, 10000)

        # Worst trip against the fishing time limit, recorded before the TSP
        # reorder. See test_fishing_time_limit_has_little_slack.
        worst_fish_time = max(t.fish_time for b in prob.boats for t in b.route)

        for boat in prob.boats:
            boat.improve_route(prob)

        result = evaluate_heuristic_solution(prob, inst)
        result["trips"] = solution_to_trips(prob)
        result["instance"] = inst
        result["penalty_raw"] = penalty
        result["objective_raw"] = obj_raw
        result["worst_fish_time"] = worst_fish_time
        result["home_ports"] = [b.home_port for b in prob.boats]
        result["routes"] = [
            [(t.start_port, list(t.stations), t.end_port) for t in b.route]
            for b in prob.boats
        ]

    return result


@pytest.fixture(scope="module")
def scenarios():
    """A small shared scenario set. Enough to test an invariant, not a mean."""
    sim = CatchSimulator(seed=123)
    sim.fit()
    return sim.sample(n_scenarios=30)
# Construction
def test_greedy_terminates_and_is_feasible(solved):
    """Check that greedy construction finds a feasible full-problem route."""
    assert solved["penalty_raw"] == 0.0, (
        f"greedy produced an infeasible solution, penalty "
        f"{solved['penalty_raw']:.1f}"
    )
    assert solved["feasible"], "solution infeasible against true capacity"


def test_every_station_visited_exactly_once(solved):
    """All 581 stations, none repeated, none invented."""
    ordinals = [_node_to_station(n)
                for trip in solved["trips"]
                for n in trip["nodes"] if n >= N_PORTS]

    counts = {}
    for o in ordinals:
        counts[o] = counts.get(o, 0) + 1

    expected = set(range(N_STATIONS))
    assert set(counts) == expected, (
        f"missing {sorted(expected - set(counts))[:5]}, "
        f"unexpected {sorted(set(counts) - expected)[:5]}"
    )
    # Each station is a towed pair, so both of its nodes appear, and only once.
    repeated = {o: c for o, c in counts.items() if c != 2}
    assert not repeated, f"stations not visited exactly once: {list(repeated)[:5]}"


def test_no_boat_is_left_idle(solved):
    """Check that every boat receives work."""
    per_boat = [b["n_stations"] for b in solved["boats_summary"]]
    assert all(n > 0 for n in per_boat), f"a boat got no stations: {per_boat}"
# Route constraints
def test_every_boat_starts_and_ends_at_its_home_port(solved):
    """Check that every boat starts and ends at its home port."""
    for home, route in zip(solved["home_ports"], solved["routes"]):
        assert route, "boat has no route at all"
        assert route[0][0] == home, (
            f"boat starts at port {route[0][0]}, home is {home}")
        assert route[-1][2] == home, (
            f"boat ends at port {route[-1][2]}, home is {home}")


def test_extracted_trip_times_match_the_objective(solved):
    """Check that extracted trip times still match the solver objective."""
    extracted = sum(t["total_time"] for t in solved["trips"])
    assert extracted == pytest.approx(solved["objective"], abs=1e-9), (
        f"objective {solved['objective']:.4f} but trips sum to "
        f"{extracted:.4f}, a gap of {solved['objective'] - extracted:.4f} h"
    )
# Stochastic evaluation
@pytest.mark.parametrize("strategy", sorted(DETOUR_STRATEGIES))
def test_realised_time_is_never_below_planned(solved, scenarios, strategy):
    """Check that detour strategies never lose planned work."""
    evaluator = StochasticEvaluator(scenarios=scenarios)
    result = evaluator.evaluate(solved["trips"], solved["instance"],
                                strategy=strategy)

    planned = sum(t["total_time"] for t in solved["trips"])
    # Every scenario, not a percentile -- one breach is a bug.
    realised = [r["total_time"] for r in result["all_results"]]
    worst = min(realised)

    assert worst >= planned - 1e-6, (
        f"{strategy}: a scenario realised {worst:.2f} h against a planned "
        f"{planned:.2f} h, which is {planned - worst:.2f} h of work lost"
    )
# Instance parameters
def test_fishing_time_limit_has_little_slack(solved):
    """Check that the full-problem fish-time limit has little slack."""
    limit = solved["instance"].fish_time_limit
    slack = limit - solved["worst_fish_time"]

    assert slack >= 0, (
        f"worst trip fishes {solved['worst_fish_time']:.1f} h against a "
        f"{limit} h limit -- infeasible")
    assert slack < 5.0, (
        f"slack is now {slack:.1f} h, not the ~0.7 h this was written against. "
        f"If fish_time_limit changed on purpose, update this test; if not, "
        f"check where {limit} came from.")
