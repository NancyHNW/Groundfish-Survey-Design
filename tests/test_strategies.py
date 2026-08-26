"""Tests for the overflow response strategies.

The existing single-realisation tests exercise `backtrack` implicitly through
`evaluate_single_realisation`. These target the registry itself, so a broken
`forward` or `preemptive` fails here rather than silently producing plausible
numbers in an experiment.

Run from repo root:
    python -m pytest tests/test_strategies.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unified.problem import ProblemInstance
from unified.stochastic_eval import STRATEGIES, evaluate_single_realisation

STRATEGY_NAMES = sorted(STRATEGIES)


def make_instance(capacity=250.0):
    return ProblemInstance(
        station_ids=np.array([0, 1, 2, 3, 4]),
        port_ids=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        n_boats=2,
        capacities=np.array([capacity, capacity]),
        home_port_index=0,
        fish_time_limit=100.0,
        work_limit=10,
        source="test",
    )


def make_trips():
    return [
        {"boat_id": 0, "nodes": [0, 13, 14, 15, 16, 17, 18, 0],
         "total_time": 50.0, "fish_time": 30.0, "catch": 300.0},
        {"boat_id": 1, "nodes": [0, 19, 20, 21, 22, 0],
         "total_time": 40.0, "fish_time": 25.0, "catch": 200.0},
    ]


def make_time_matrix(travel=5.0):
    tm = np.full((1175, 1175), travel)
    np.fill_diagonal(tm, 0.0)
    return tm


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_lists_the_three_strategies():
    assert set(STRATEGIES) == {"backtrack", "forward", "preemptive"}
    assert all(callable(fn) for fn in STRATEGIES.values())


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown strategy"):
        evaluate_single_realisation(
            make_trips(), make_instance(), np.full(581, 50.0),
            time_matrix=make_time_matrix(), strategy="teleport")


# ---------------------------------------------------------------------------
# Invariants that must hold for every strategy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_realised_time_never_below_planned(strategy):
    """Strategies only ever add detours, so realised >= planned always."""
    result = evaluate_single_realisation(
        make_trips(), make_instance(), np.full(581, 90.0),
        time_matrix=make_time_matrix(), strategy=strategy)
    assert result["total_time"] >= result["original_total_time"] - 1e-9
    assert result["time_penalty"] >= -1e-9


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_light_catch_costs_nothing(strategy):
    """Well under capacity: no overflow, no detour, no added time.

    Preemptive is included deliberately — 30 kg/station against a 250 kg hold
    stays below even a 0.8 trigger, so it must not fire either.
    """
    result = evaluate_single_realisation(
        make_trips(), make_instance(), np.full(581, 30.0),
        time_matrix=make_time_matrix(), strategy=strategy)
    assert not result["capacity_exceeded"]
    assert result["n_unscheduled_returns"] == 0
    assert result["total_time"] == pytest.approx(90.0)
    assert result["overflow_events"] == []


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_heavy_catch_forces_returns(strategy):
    """200 kg/station against a 250 kg hold must overflow on every strategy."""
    result = evaluate_single_realisation(
        make_trips(), make_instance(), np.full(581, 200.0),
        time_matrix=make_time_matrix(), strategy=strategy)
    assert result["capacity_exceeded"]
    assert result["n_unscheduled_returns"] > 0
    assert result["total_time"] > result["original_total_time"]


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_reports_its_own_name(strategy):
    result = evaluate_single_realisation(
        make_trips(), make_instance(), np.full(581, 50.0),
        time_matrix=make_time_matrix(), strategy=strategy)
    assert result["strategy"] == strategy


# ---------------------------------------------------------------------------
# Behaviour specific to each strategy
# ---------------------------------------------------------------------------

def test_backtrack_detour_is_there_and_back():
    """Boat 0 fills at station 2; backtrack costs 2 x travel to nearest port."""
    catch = np.full(581, 50.0)
    catch[0] = catch[1] = 100.0
    catch[2] = 120.0                      # cumulative 320 > 250

    result = evaluate_single_realisation(
        make_trips(), make_instance(), catch,
        time_matrix=make_time_matrix(travel=5.0), strategy="backtrack")

    trip0 = result["trip_details"][0]
    assert trip0["detour_time"] == pytest.approx(10.0)   # 2 x 5.0
    assert trip0["adjusted_time"] == pytest.approx(60.0)


def test_preemptive_triggers_before_capacity_is_reached():
    """A load that never overflows still forces a return once past the trigger.

    Two stations at 110 kg reach 220 of a 250 kg hold — under capacity, but
    over the 0.8 trigger (200 kg). Backtrack should do nothing here;
    preemptive should return.
    """
    catch = np.full(581, 110.0)
    inst, trips, tm = make_instance(), make_trips(), make_time_matrix()

    quiet = evaluate_single_realisation(trips, inst, catch, time_matrix=tm,
                                        strategy="backtrack")
    early = evaluate_single_realisation(trips, inst, catch, time_matrix=tm,
                                        strategy="preemptive",
                                        preemptive_threshold=0.8)

    assert early["n_unscheduled_returns"] > quiet["n_unscheduled_returns"]
    assert any(e.get("preemptive") for e in early["overflow_events"])


def test_lower_preemptive_threshold_returns_more_often():
    """Monotonicity: a tighter trigger cannot produce fewer returns."""
    catch = np.full(581, 90.0)
    inst, trips, tm = make_instance(), make_trips(), make_time_matrix()

    returns = [
        evaluate_single_realisation(
            trips, inst, catch, time_matrix=tm, strategy="preemptive",
            preemptive_threshold=alpha)["n_unscheduled_returns"]
        for alpha in (0.5, 0.7, 0.9)
    ]
    assert returns[0] >= returns[1] >= returns[2]


def test_forward_and_backtrack_differ_on_a_mid_trip_overflow():
    """Forward carries on from the port; backtrack resumes where it left off.

    Trip 0 visits stations 0, 1, 2. The catch is set to overflow at station 1
    so that station 2 is still ahead — that is the only situation where the two
    strategies can diverge (see the final-station test below). Station 2's
    nodes are made expensive to reach, so continuing there from the port costs
    visibly more than backtracking.
    """
    tm = make_time_matrix(travel=5.0)
    tm[17, :] = tm[:, 17] = 40.0
    tm[18, :] = tm[:, 18] = 40.0
    np.fill_diagonal(tm, 0.0)

    catch = np.full(581, 50.0)
    catch[0] = 150.0
    catch[1] = 150.0          # cumulative 300 > 250, and station 2 still ahead

    inst, trips = make_instance(), make_trips()
    back = evaluate_single_realisation(trips, inst, catch, time_matrix=tm,
                                       strategy="backtrack")
    fwd = evaluate_single_realisation(trips, inst, catch, time_matrix=tm,
                                      strategy="forward")

    assert back["capacity_exceeded"] and fwd["capacity_exceeded"]
    # they must not silently be the same code path
    assert back["total_time"] != fwd["total_time"]


def test_forward_matches_backtrack_when_overflow_is_at_the_last_station():
    """With nothing left to continue to, forward has to behave as backtrack.

    Documented behaviour rather than a coincidence: a boat that fills at the
    final station of a trip was heading to port regardless.
    """
    tm = make_time_matrix(travel=5.0)
    catch = np.full(581, 50.0)
    catch[0] = catch[1] = 100.0
    catch[2] = 120.0          # overflow at station 2, the last in trip 0

    inst, trips = make_instance(), make_trips()
    back = evaluate_single_realisation(trips, inst, catch, time_matrix=tm,
                                       strategy="backtrack")
    fwd = evaluate_single_realisation(trips, inst, catch, time_matrix=tm,
                                      strategy="forward")

    assert back["total_time"] == pytest.approx(fwd["total_time"])
