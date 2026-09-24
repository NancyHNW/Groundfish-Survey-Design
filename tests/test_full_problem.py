"""Tests on the real 581-station survey, not a scaled-down proxy.

The gfsp subproblems need a catch adjustment at ns == 10 to be solvable at
all, and their capacities are scaled by a factor while the catches are not.
The full problem has neither issue: the largest single station is 7,908 kg
against a smallest hold of 14,000 kg, so greedy construction finishes with a
zero penalty on the first attempt.

Solving is cheap. The whole file runs in a few seconds, and does so only
because the catch fit is cached -- an uncached fit costs 138 s on its own.

Marked `full` so the everyday run stays fast:

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
from unified.stochastic_eval import STRATEGIES, StochasticEvaluator

pytestmark = pytest.mark.full

SEED = 42
N_STATIONS = 581


def _node_to_station(node):
    return (node - N_PORTS) // 2


@pytest.fixture(scope="module")
def solved():
    """Solve the full problem once and share it across the file.

    Greedy rather than GRASP on purpose: GRASP never checks whether a station
    fits before adding it, so it cannot show that a feasible solution exists.
    """
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


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_greedy_terminates_and_is_feasible(solved):
    """The whole point: the full problem has a feasible solution.

    Greedy only crosses a station off when it manages to add one, so a station
    that fits in no boat leaves it sailing to port and back forever. Reaching
    this assertion at all proves that did not happen.
    """
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
    """Every boat does some work.

    Total time is summed across boats, so a solution that idles one scores a
    *lower* objective while being worse. Greedy nearest-boat selection once
    left a boat with 0 stations and another with 652 for exactly this reason.
    """
    per_boat = [b["n_stations"] for b in solved["boats_summary"]]
    assert all(n > 0 for n in per_boat), f"a boat got no stations: {per_boat}"


# ---------------------------------------------------------------------------
# Route constraints
# ---------------------------------------------------------------------------

def test_every_boat_starts_and_ends_at_its_home_port(solved):
    """Closes a long-standing gap: this was enforced but never asserted.

    The penalty term that would catch a violation only fires when
    check_stations=True, which no caller passes, so the constraint rests
    entirely on the construction heuristics appending a return leg.
    """
    for home, route in zip(solved["home_ports"], solved["routes"]):
        assert route, "boat has no route at all"
        assert route[0][0] == home, (
            f"boat starts at port {route[0][0]}, home is {home}")
        assert route[-1][2] == home, (
            f"boat ends at port {route[-1][2]}, home is {home}")


def test_extracted_trip_times_match_the_objective(solved):
    """Nothing is lost between the solved Problem and the extracted trips.

    The stochastic evaluator only ever sees the extracted trips, so anything
    dropped here is invisible downstream. Port-to-port repositioning legs have
    two nodes like empty padding trips but carry real travel time, and were
    silently discarded once before.
    """
    extracted = sum(t["total_time"] for t in solved["trips"])
    assert extracted == pytest.approx(solved["objective"], abs=1e-9), (
        f"objective {solved['objective']:.4f} but trips sum to "
        f"{extracted:.4f}, a gap of {solved['objective'] - extracted:.4f} h"
    )


# ---------------------------------------------------------------------------
# Stochastic evaluation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_realised_time_is_never_below_planned(solved, scenarios, strategy):
    """The one invariant no overflow strategy may break.

    Every strategy adds detours and none removes work, so realised time is
    bounded below by planned time in every scenario. A strategy that comes in
    under it has lost work somewhere.
    """
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


# ---------------------------------------------------------------------------
# Instance parameters
# ---------------------------------------------------------------------------

def test_fishing_time_limit_has_little_slack(solved):
    """Documents how tight the fishing time limit is, rather than asserting it.

    fish_time_limit=120 and work_limit=150 are hardcoded defaults in
    load_gfsp_full_problem's signature, not read from full_problem.txt the way
    the subproblem values come from the gfsp formulas. The best trip lands
    around 119.3 h against the 120 h limit -- feasible by well under an hour.

    Those two numbers are load-bearing and unvalidated. This test fails if the
    slack changes materially, which is the signal to go and check them rather
    than to adjust the number here.
    """
    limit = solved["instance"].fish_time_limit
    slack = limit - solved["worst_fish_time"]

    assert slack >= 0, (
        f"worst trip fishes {solved['worst_fish_time']:.1f} h against a "
        f"{limit} h limit -- infeasible")
    assert slack < 5.0, (
        f"slack is now {slack:.1f} h, not the ~0.7 h this was written against. "
        f"If fish_time_limit changed on purpose, update this test; if not, "
        f"check where {limit} came from.")
