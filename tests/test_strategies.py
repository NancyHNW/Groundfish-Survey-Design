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
from unified.stochastic_eval import (DETOUR_STRATEGIES, STRATEGIES,
                                     evaluate_single_realisation)

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
    """Two trips whose declared times match what their nodes actually walk.

    At travel=5.0: trip 0 is 7 legs (35.0) with 6 after the first station
    (30.0), trip 1 is 5 legs (25.0) with 4 after the first (20.0). The detour
    strategies never check, but repair recomputes from the nodes, so a fixture
    that disagreed with itself made a correct implementation look broken.
    """
    return [
        {"boat_id": 0, "nodes": [0, 13, 14, 15, 16, 17, 18, 0],
         "total_time": 35.0, "fish_time": 30.0, "catch": 300.0},
        {"boat_id": 1, "nodes": [0, 19, 20, 21, 22, 0],
         "total_time": 25.0, "fish_time": 20.0, "catch": 200.0},
    ]


def make_time_matrix(travel=5.0):
    tm = np.full((1175, 1175), travel)
    np.fill_diagonal(tm, 0.0)
    return tm
# Registry
# Exact output of the three detour strategies. A tripwire, not a spec: if these
# move, find what changed the model before editing the numbers.
GOLDEN = {
    ("light", "backtrack"): (60.0, 0, [0.0, 0.0]),
    ("light", "forward"): (60.0, 0, [0.0, 0.0]),
    ("light", "preemptive"): (60.0, 0, [0.0, 0.0]),
    ("heavy", "backtrack"): (90.0, 3, [20.0, 10.0]),
    # Trip 0 overflows mid-trip (5.0); trip 1 overflows on its LAST station,
    # where the nearest port is the end port, so forward pays nothing (0.0).
    ("heavy", "forward"): (65.0, 2, [5.0, 0.0]),
    ("heavy", "preemptive"): (110.0, 5, [30.0, 20.0]),
    ("mid", "backtrack"): (70.0, 1, [10.0, 0.0]),
    # Single overflow, on trip 0's last station -> change of port only, 0.0.
    ("mid", "forward"): (60.0, 1, [0.0, 0.0]),
    ("mid", "preemptive"): (70.0, 1, [10.0, 0.0]),
}


def golden_catch(case):
    if case == "light":
        return np.full(581, 30.0)
    if case == "heavy":
        return np.full(581, 200.0)
    catch = np.full(581, 50.0)
    catch[0] = catch[1] = 100.0
    catch[2] = 120.0
    return catch


@pytest.mark.parametrize("case,strategy", sorted(GOLDEN))
def test_detour_strategies_are_unchanged(case, strategy):
    """The three detour strategies must not move when repair is added."""
    total, returns, detours = GOLDEN[(case, strategy)]
    result = evaluate_single_realisation(
        make_trips(), make_instance(), golden_catch(case),
        time_matrix=make_time_matrix(), strategy=strategy)

    assert result["total_time"] == pytest.approx(total)
    assert result["n_unscheduled_returns"] == returns
    assert [d["detour_time"] for d in result["trip_details"]] == \
        pytest.approx(detours)


def test_fixture_times_match_its_own_nodes():
    """Guards the fixture itself. Repair recomputes time from the nodes."""
    tm = make_time_matrix(travel=5.0)
    for trip in make_trips():
        nodes = trip["nodes"]
        walked = sum(tm[a, b] for a, b in zip(nodes, nodes[1:]))
        assert trip["total_time"] == pytest.approx(walked)

        first_station = next(i for i, n in enumerate(nodes) if n >= 13)
        fished = sum(tm[a, b] for a, b in
                     zip(nodes[first_station:], nodes[first_station + 1:]))
        assert trip["fish_time"] == pytest.approx(fished)


def test_registry_lists_the_detour_strategies():
    assert {"backtrack", "forward", "preemptive"} <= set(STRATEGIES)
    assert all(callable(fn) for fn in STRATEGIES.values())


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown strategy"):
        evaluate_single_realisation(
            make_trips(), make_instance(), np.full(581, 50.0),
            time_matrix=make_time_matrix(), strategy="teleport")
# Invariants that must hold for every strategy
@pytest.mark.parametrize("strategy", DETOUR_STRATEGIES)
def test_realised_time_never_below_planned(strategy):
    """Detour strategies only add to an unchanged route, so realised >= planned.

    Not asked of repair, which replaces the route and so may come in under it.
    """
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
    assert result["total_time"] == pytest.approx(60.0)
    assert result["overflow_events"] == []


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_heavy_catch_forces_returns(strategy):
    """200 kg/station against a 250 kg hold must force returns on every strategy.

    It does not force an *exceedance* on preemptive: the 0.8 trigger sits at
    exactly 200, so it lands after every station and the hold never passes 250.
    Avoiding the exceedance is the whole point of the policy, so that assertion
    belongs to the detour strategies only.
    """
    result = evaluate_single_realisation(
        make_trips(), make_instance(), np.full(581, 200.0),
        time_matrix=make_time_matrix(), strategy=strategy)
    if strategy != "preemptive":
        assert result["capacity_exceeded"]
    assert result["n_unscheduled_returns"] > 0
    assert result["total_time"] > result["original_total_time"]


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_reports_its_own_name(strategy):
    result = evaluate_single_realisation(
        make_trips(), make_instance(), np.full(581, 50.0),
        time_matrix=make_time_matrix(), strategy=strategy)
    assert result["strategy"] == strategy
# Behaviour specific to each strategy
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
    assert trip0["adjusted_time"] == pytest.approx(45.0)  # 35.0 planned + 10.0


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


def test_forward_is_cheaper_than_backtrack_at_a_trip_last_station():
    """They used to be identical here. Now only backtrack goes back.

    Forward has the station's catch aboard and the port was its next stop
    anyway, so it pays only the change of port. Backtrack did not fish the
    station, so it must return for it and pays the full out-and-back.

    On this fixture the nearest port IS the end port (uniform matrix, so
    argmin picks port 0, which is where trip 0 ends), making forward's detour
    exactly zero against backtrack's 2 x 5.0.
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

    assert fwd["total_time"] < back["total_time"]
    assert fwd["total_time"] == pytest.approx(back["total_time"] - 10.0)


def test_preemptive_returns_are_not_counted_as_exceedances():
    """Returning early is what avoids exceeding, so it cannot mean exceeding.

    A load that passes the 0.8 trigger but never the hold must report
    capacity_exceeded False while still recording the return. Counting the
    two together made P(exceed) read 100% for preemptive against 98.6% for
    every other method -- backwards.
    """
    catch = np.full(581, 110.0)        # 2 x 110 = 220: over 0.8 x 250, under 250
    inst, trips, tm = make_instance(), make_trips(), make_time_matrix()

    early = evaluate_single_realisation(trips, inst, catch, time_matrix=tm,
                                        strategy="preemptive",
                                        preemptive_threshold=0.8)

    assert early["n_unscheduled_returns"] > 0, "fixture must trigger a return"
    assert all(e["preemptive"] for e in early["overflow_events"])
    assert early["capacity_exceeded"] is False
    assert not any(d["exceeded"] for d in early["trip_details"])


def test_preemptive_still_reports_a_real_exceedance():
    """The guard must not hide genuine overflows under preemptive.

    One tow has to clear the hold on its own, or the 0.8 trigger fires first
    and there is no exceedance left to report. 260 > 250 does it; 200 does not,
    because 200 is exactly the trigger.
    """
    catch = np.full(581, 260.0)        # one tow alone clears the 250 hold
    result = evaluate_single_realisation(
        make_trips(), make_instance(), catch, time_matrix=make_time_matrix(),
        strategy="preemptive", preemptive_threshold=0.8)

    assert any(not e["preemptive"] for e in result["overflow_events"])
    assert result["capacity_exceeded"] is True
# Which end of the tow does a strategy leave from?
def _tow_geometry():
    """A tow whose two ends have different nearest ports.

    make_time_matrix is flat, so entry and exit are equidistant from every
    port and no flat fixture can tell the two apart. Station 1 is nodes 15
    (entry) and 16 (exit); port 3 is close to the entry, port 4 to the exit.
    """
    tm = make_time_matrix()
    tm[15, 3] = tm[3, 15] = 1.0        # entry end is 1.0 from port 3
    tm[16, 4] = tm[4, 16] = 2.0        # exit end is 2.0 from port 4
    return tm


def _overflow_on_station_1():
    """Catch that fills the hold exactly at station 1 of trip 0."""
    catch = np.zeros(581)
    catch[0] = 150.0
    catch[1] = 150.0                   # 300 > 250 capacity
    return catch


def test_forward_diverts_from_the_tow_exit():
    """Forward tows the station, so it leaves from the far end.

    Catching this needs a matrix where the two ends differ; with a flat one
    the bug is invisible, which is how it survived.
    """
    r = evaluate_single_realisation(
        make_trips(), make_instance(), _overflow_on_station_1(),
        time_matrix=_tow_geometry(), strategy="forward")
    e = r["overflow_events"][0]
    assert e["station_node"] == 16, (
        f"forward should divert from the exit node 16, got {e['station_node']}")
    assert e["nearest_port"] == 4, (
        "port 4 is nearest the exit; port 3 is nearest the entry")


def test_backtrack_diverts_from_the_tow_entry():
    """Backtrack has NOT towed the station, so the entry is correct here."""
    r = evaluate_single_realisation(
        make_trips(), make_instance(), _overflow_on_station_1(),
        time_matrix=_tow_geometry(), strategy="backtrack")
    e = r["overflow_events"][0]
    assert e["station_node"] == 15, (
        f"backtrack should divert from the entry node 15, "
        f"got {e['station_node']}")
    assert e["nearest_port"] == 3
    assert e["detour_time"] == 2.0, "out and back to the same spot: 2 x 1.0"


def test_repair_prices_the_diversion_from_where_it_sailed():
    """Repair's prefix ends at the exit, so its leg must start there too."""
    r = evaluate_single_realisation(
        make_trips(), make_instance(), _overflow_on_station_1(),
        time_matrix=_tow_geometry(), strategy="repair", repair_scope="trip")
    e = r["overflow_events"][0]
    assert e["station_node"] == 16
    assert e["detour_time"] == 2.0, (
        "the leg is exit -> port 4 = 2.0; charging the entry would give 1.0 "
        "and bill the boat from an end it is not at")
# Does the time charged equal the route sailed?
#
# The strongest check available, because it needs no knowledge of the right
# answer -- it only asks the evaluator to agree with itself. The entry/exit
# defect of 2026-10-16 was exactly this disagreement: a route walked to the
# tow's exit, a time billed from its entry. Run on the geometry fixture, since
# a flat matrix cannot tell the two ends apart.

def _walk(nodes, tm):
    return sum(tm[a, b] for a, b in zip(nodes, nodes[1:]))


def test_forward_charges_exactly_the_route_it_sails():
    """Forward tows the station, then calls at a port before the next one."""
    tm = _tow_geometry()
    trips = make_trips()
    r = evaluate_single_realisation(
        trips, make_instance(), _overflow_on_station_1(),
        time_matrix=tm, strategy="forward")

    nodes = trips[0]["nodes"]
    port = r["overflow_events"][0]["nearest_port"]
    # station 1 is nodes[3:5]; the port call goes after its exit
    sailed = nodes[:5] + [port] + nodes[5:]

    charged = r["trip_details"][0]["adjusted_time"]
    assert charged == pytest.approx(_walk(sailed, tm)), (
        f"forward charged {charged} for a route costing "
        f"{_walk(sailed, tm)}")


def test_backtrack_charges_exactly_the_route_it_sails():
    """Backtrack reaches the station, turns back, then returns and tows it."""
    tm = _tow_geometry()
    trips = make_trips()
    r = evaluate_single_realisation(
        trips, make_instance(), _overflow_on_station_1(),
        time_matrix=tm, strategy="backtrack")

    nodes = trips[0]["nodes"]
    port = r["overflow_events"][0]["nearest_port"]
    # out and back from the entry, which is where it resumes
    sailed = nodes[:4] + [port] + nodes[3:]

    charged = r["trip_details"][0]["adjusted_time"]
    assert charged == pytest.approx(_walk(sailed, tm)), (
        f"backtrack charged {charged} for a route costing "
        f"{_walk(sailed, tm)}")


def test_repair_charges_exactly_the_routes_it_records():
    """Repair records what it sailed, so the two must reconcile directly."""
    tm = _tow_geometry()
    trips = make_trips()
    r = evaluate_single_realisation(
        trips, make_instance(), _overflow_on_station_1(),
        time_matrix=tm, strategy="repair", repair_scope="trip",
        record_routes=True)

    recorded = [leg for e in r["repaired_routes"] if e["trip_idx"] == 0
                for leg in e["nodes"]]
    assert recorded, "repair recorded no routes to reconcile against"

    charged = r["trip_details"][0]["adjusted_time"]
    assert charged == pytest.approx(sum(_walk(leg, tm) for leg in recorded)), (
        f"repair charged {charged} for routes costing "
        f"{sum(_walk(leg, tm) for leg in recorded)}")
