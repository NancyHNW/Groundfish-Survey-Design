"""Tests for the repair strategy and its honest baseline.

Repair rebuilds routes rather than adding detours to them, so it can lose or
repeat a station, dock at the wrong port, or quietly break the limits
the re-split is supposed to respect. Any of those would read as a result rather
than a bug, which is what most of this file guards.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unified.problem import N_PORTS
from unified.stochastic_eval import (_leg_time, _nn_resplit, _solver_resplit,
                                     _two_opt,
                                     evaluate_single_realisation)
from test_strategies import make_instance, make_time_matrix, make_trips


def stations_of(nodes):
    """Station ordinals in a node sequence, in visit order."""
    return [(n - N_PORTS) // 2 for n in nodes if n >= N_PORTS][::2]


def run(catch, strategy="repair", trips=None, instance=None, tm=None, **kw):
    return evaluate_single_realisation(
        trips or make_trips(), instance or make_instance(), catch,
        time_matrix=tm if tm is not None else make_time_matrix(),
        strategy=strategy, **kw)


# ---- the null case --------------------------------------------------------

def test_no_overflow_costs_nothing():
    result = run(np.full(581, 30.0))
    assert result["n_unscheduled_returns"] == 0
    assert result["time_penalty"] == pytest.approx(0.0)
    assert result["n_repairs"] == 0


def test_no_overflow_matches_backtrack_exactly():
    catch = np.full(581, 30.0)
    assert run(catch)["total_time"] == pytest.approx(
        run(catch, strategy="backtrack")["total_time"])


# ---- what repair must not break -------------------------------------------

def test_every_station_still_visited_exactly_once():
    """Repair rebuilds routes, so it can drop or repeat a station."""
    catch = np.full(581, 90.0)
    result = run(catch)
    assert result["n_unscheduled_returns"] > 0, "fixture must overflow"

    planned = sorted(s for t in make_trips() for s in stations_of(t["nodes"]))
    tm = make_time_matrix()
    for trip in make_trips():
        stations = stations_of(trip["nodes"])
        plan = _nn_resplit(0, stations, np.full(581, 100.0), 250.0, 0,
                           tm, 100.0)
        rebuilt = sorted(s for sub in plan for s in sub["stations"])
        assert rebuilt == sorted(stations)
    assert planned == sorted(planned)


def test_resplit_covers_its_stations_exactly_once():
    tm = make_time_matrix()
    stations = [0, 1, 2, 3, 4, 5, 6]
    plan = _nn_resplit(0, stations, np.full(581, 100.0), 250.0, 0, tm, 100.0)
    seen = [s for sub in plan for s in sub["stations"]]
    assert sorted(seen) == sorted(stations)
    assert len(seen) == len(set(seen))


def test_every_repaired_trip_docks_where_the_plan_did():
    """The next planned trip departs from where this one docked."""
    tm = make_time_matrix()
    plan = _nn_resplit(5, [0, 1, 2, 3, 4], np.full(581, 100.0), 250.0,
                       end_port=3, time_matrix=tm, fish_time_limit=100.0)
    assert all(sub["nodes"][-1] == 3 for sub in plan)


def test_first_repaired_trip_leaves_the_port_it_diverted_to():
    tm = make_time_matrix()
    plan = _nn_resplit(5, [0, 1, 2, 3, 4], np.full(581, 100.0), 250.0,
                       end_port=3, time_matrix=tm, fish_time_limit=100.0)
    assert plan[0]["nodes"][0] == 5
    assert all(sub["nodes"][0] == 3 for sub in plan[1:])


def test_resplit_nodes_are_station_pairs():
    """Both nodes of a station, adjacent, so the tow is actually sailed."""
    tm = make_time_matrix()
    plan = _nn_resplit(0, [0, 1, 2], np.full(581, 100.0), 250.0, 0, tm, 100.0)
    for sub in plan:
        inner = sub["nodes"][1:-1]
        assert len(inner) == 2 * len(sub["stations"])
        for a, b in zip(inner[::2], inner[1::2]):
            assert abs(a - b) == 1
            assert (a - N_PORTS) // 2 == (b - N_PORTS) // 2


# ---- the re-split's own limits --------------------------------------------

def test_resplit_respects_the_planned_capacity():
    tm = make_time_matrix()
    belief = np.full(581, 100.0)
    plan = _nn_resplit(0, list(range(8)), belief, 250.0, 0, tm, 1000.0)
    for sub in plan:
        if len(sub["stations"]) > 1:
            assert sum(belief[s] for s in sub["stations"]) <= 250.0


def test_resplit_respects_the_fish_time_limit():
    tm = make_time_matrix(travel=5.0)
    plan = _nn_resplit(0, list(range(8)), np.full(581, 1.0), 1e9, 0, tm,
                       fish_time_limit=30.0)
    for sub in plan:
        if len(sub["stations"]) > 1:
            fish = _leg_time(sub["nodes"][1:], tm)
            assert fish <= 30.0


def test_a_single_station_over_the_limit_still_gets_a_trip():
    """Refusing the first station of a trip would never close it."""
    tm = make_time_matrix()
    plan = _nn_resplit(0, [0, 1], np.full(581, 9999.0), 250.0, 0, tm, 1.0)
    assert [s for sub in plan for s in sub["stations"]] != []
    assert all(len(sub["stations"]) == 1 for sub in plan)


def test_resplit_of_nothing_is_nothing():
    assert _nn_resplit(0, [], np.full(581, 1.0), 250.0, 0,
                       make_time_matrix(), 100.0) == []


# ---- the free trip-boundary return ----------------------------------------

def test_overflow_on_the_last_station_is_not_charged_an_out_and_back():
    """The boat was heading to port anyway, so only the port differs."""
    catch = np.full(581, 50.0)
    catch[0] = catch[1] = 100.0
    catch[2] = 120.0                       # trip 0's last station

    tm = make_time_matrix(travel=5.0)
    repaired = run(catch, tm=tm)
    back = run(catch, strategy="backtrack", tm=tm)

    assert repaired["trip_details"][0]["detour_time"] < \
        back["trip_details"][0]["detour_time"]


def test_backtrack_last_free_only_changes_the_last_station_case():
    """Mid-trip overflows must cost exactly what backtrack charges."""
    catch = np.full(581, 50.0)
    catch[0] = 150.0
    catch[1] = 150.0                       # station 2 still ahead

    tm = make_time_matrix(travel=5.0)
    assert run(catch, strategy="backtrack_last_free", tm=tm)["total_time"] == \
        pytest.approx(run(catch, strategy="backtrack", tm=tm)["total_time"])


def test_backtrack_last_free_is_cheaper_on_a_last_station_overflow():
    catch = np.full(581, 50.0)
    catch[0] = catch[1] = 100.0
    catch[2] = 120.0

    tm = make_time_matrix(travel=5.0)
    free = run(catch, strategy="backtrack_last_free", tm=tm)
    back = run(catch, strategy="backtrack", tm=tm)
    assert free["total_time"] < back["total_time"]
    assert free["n_unscheduled_returns"] == back["n_unscheduled_returns"]


# ---- termination ----------------------------------------------------------

def test_absurd_catch_terminates_and_reports_the_cap():
    """Every station overflowing on its own must still finish the season.

    One repair is not enough to clear trip 0's three stations, so the cap
    binds and the rest is sailed charging backtrack detours.
    """
    result = run(np.full(581, 1e6), max_repairs=1)
    assert result["repair_cap_hit"] is True
    assert np.isfinite(result["total_time"])


def test_the_cap_does_not_bind_when_repairs_run_out_naturally():
    """Exhausting the stations is not the cap binding."""
    result = run(np.full(581, 1e6), max_repairs=10)
    assert result["repair_cap_hit"] is False
    assert np.isfinite(result["total_time"])


def test_returns_past_the_cap_are_still_counted():
    """Past the cap repair falls back to backtrack, which still makes returns.

    Leaving them out would let a capped run report fewer returns than an
    uncapped one on the same scenario.
    """
    catch = np.full(581, 1e6)
    capped = run(catch, max_repairs=1)
    uncapped = run(catch, max_repairs=10)
    assert capped["repair_cap_hit"] is True
    assert capped["n_unscheduled_returns"] == uncapped["n_unscheduled_returns"]


def test_repair_count_is_reported():
    result = run(np.full(581, 200.0))
    assert result["n_repairs"] >= 1


def test_a_higher_cap_is_never_worse_on_time():
    """More re-planning should not cost more than falling back to backtrack."""
    catch = np.full(581, 200.0)
    tight = run(catch, max_repairs=1)["total_time"]
    loose = run(catch, max_repairs=10)["total_time"]
    assert loose <= tight + 1e-9


# ---- the catch belief -----------------------------------------------------

def test_planned_catch_is_used_when_given():
    """A belief that says every station is huge forces one station per trip."""
    tm = make_time_matrix()
    many = _nn_resplit(0, list(range(6)), np.full(581, 240.0), 250.0, 0, tm,
                       1000.0)
    few = _nn_resplit(0, list(range(6)), np.full(581, 10.0), 250.0, 0, tm,
                      1000.0)
    assert len(many) > len(few)


def test_missing_planned_catch_falls_back_to_the_trip_average():
    """Still runs, just with a cruder belief than the solver had."""
    result = run(np.full(581, 90.0), planned_catch=None)
    assert np.isfinite(result["total_time"])


def test_per_station_belief_changes_the_split():
    """Flattening the belief is what packs heavy stations together."""
    tm = make_time_matrix()
    flat = np.full(581, 100.0)
    peaked = np.full(581, 100.0)
    peaked[0] = peaked[1] = 240.0

    assert len(_nn_resplit(0, [0, 1, 2, 3], peaked, 250.0, 0, tm, 1000.0)) > \
        len(_nn_resplit(0, [0, 1, 2, 3], flat, 250.0, 0, tm, 1000.0))


# ---- accounting -----------------------------------------------------------

def test_an_overflow_is_reported_even_when_repair_saves_time():
    """trip_exceeded comes from the events, not the sign of the time change."""
    catch = np.full(581, 50.0)
    catch[0] = catch[1] = 100.0
    catch[2] = 120.0

    result = run(catch, tm=make_time_matrix(travel=5.0))
    assert result["capacity_exceeded"] is True
    assert result["n_unscheduled_returns"] == 1


def test_trip_details_still_sum_to_the_total():
    result = run(np.full(581, 90.0))
    assert sum(d["adjusted_time"] for d in result["trip_details"]) == \
        pytest.approx(result["total_time"])


# ---- route recording ------------------------------------------------------

def test_recording_is_off_by_default():
    assert run(np.full(581, 90.0))["repaired_routes"] == []


def test_recording_does_not_change_any_number():
    """The whole point: turning it on must be observationally inert."""
    catch = np.full(581, 90.0)
    off = run(catch)
    on = run(catch, record_routes=True)
    assert on["repaired_routes"] != []
    for key in off:
        if key == "repaired_routes":
            continue
        assert off[key] == on[key], key


def test_detour_strategies_record_nothing():
    """They never replace a route, so there is nothing to record."""
    catch = np.full(581, 200.0)
    for strategy in ("backtrack", "backtrack_last_free", "forward",
                     "preemptive"):
        assert run(catch, strategy=strategy,
                   record_routes=True)["repaired_routes"] == []


def test_recorded_routes_cover_the_whole_trip_exactly_once():
    catch = np.full(581, 90.0)
    result = run(catch, record_routes=True)
    entry = result["repaired_routes"][0]

    planned = stations_of(make_trips()[entry["trip_idx"]]["nodes"])
    sailed = [s for route in entry["nodes"] for s in stations_of(route)]
    assert sorted(set(sailed)) == sorted(set(planned))
    assert len(sailed) == len(set(sailed)), "a station was sailed twice"


def test_recorded_routes_start_and_end_at_ports():
    result = run(np.full(581, 90.0), record_routes=True)
    for entry in result["repaired_routes"]:
        for route in entry["nodes"]:
            assert route[0] < N_PORTS and route[-1] < N_PORTS


def test_recorded_routes_are_plain_ints():
    """numpy ints leak into plotting and JSON badly."""
    result = run(np.full(581, 90.0), record_routes=True)
    for entry in result["repaired_routes"]:
        for route in entry["nodes"]:
            assert all(type(n) is int for n in route)


# ---- boat scope -----------------------------------------------------------

def make_multi_trips():
    """One boat, three trips of three stations each.

    The shared fixture gives every boat a single trip, which makes boat scope
    and trip scope identical by construction. Boat scope only means anything
    when there are later trips to absorb.

    At travel=5.0 each trip walks 7 legs (35.0), 6 of them after the first
    station (30.0).
    """
    return [
        {"boat_id": 0, "nodes": [0, 13, 14, 15, 16, 17, 18, 0],
         "total_time": 35.0, "fish_time": 30.0, "catch": 300.0},
        {"boat_id": 0, "nodes": [0, 19, 20, 21, 22, 23, 24, 0],
         "total_time": 35.0, "fish_time": 30.0, "catch": 300.0},
        {"boat_id": 0, "nodes": [0, 25, 26, 27, 28, 29, 30, 0],
         "total_time": 35.0, "fish_time": 30.0, "catch": 300.0},
    ]


def multi_run(catch, **kw):
    """Run the multi-trip fixture. 130 kg/station overflows mid-trip."""
    return run(catch, trips=make_multi_trips(), **kw)


def test_an_unknown_scope_is_rejected():
    """Silently falling back to trip scope would read as a result."""
    with pytest.raises(ValueError, match="repair_scope"):
        multi_run(np.full(581, 130.0), repair_scope="bost")


def test_boat_scope_does_not_change_trip_scope():
    """Adding the wider scope must leave the narrow one exactly as it was."""
    catch = np.full(581, 130.0)
    default = multi_run(catch)
    explicit = multi_run(catch, repair_scope="trip")
    assert default["total_time"] == pytest.approx(explicit["total_time"])
    assert default["n_repairs"] == explicit["n_repairs"]


def test_boat_scope_replans_more_than_trip_scope():
    """Trip scope leaves the later trips alone; boat scope takes them too."""
    catch = np.full(581, 130.0)
    narrow = multi_run(catch)
    wide = multi_run(catch, repair_scope="boat")
    assert wide["total_time"] != pytest.approx(narrow["total_time"])


def test_boat_scope_no_overflow_costs_nothing():
    result = multi_run(np.full(581, 30.0), repair_scope="boat")
    assert result["n_unscheduled_returns"] == 0
    assert result["time_penalty"] == pytest.approx(0.0)
    assert result["n_repairs"] == 0


def test_boat_scope_trip_details_still_sum_to_the_total():
    """One repair spans several trips, so the per-trip books must still close."""
    result = multi_run(np.full(581, 130.0), repair_scope="boat")
    assert sum(d["adjusted_time"] for d in result["trip_details"]) == \
        pytest.approx(result["total_time"])


def test_boat_scope_swallows_the_later_trips():
    """Trips consumed by the re-plan no longer exist, so they cost nothing."""
    result = multi_run(np.full(581, 130.0), repair_scope="boat")
    details = result["trip_details"]
    assert details[0]["adjusted_time"] > 0
    assert all(d["adjusted_time"] == pytest.approx(0.0) for d in details[1:])


def test_trip_scope_leaves_the_later_trips_alone():
    """The contrast: under trip scope only the overflowed trip changes."""
    result = multi_run(np.full(581, 130.0), repair_scope="trip")
    details = result["trip_details"]
    assert all(d["adjusted_time"] > 0 for d in details)


def test_boat_scope_covers_every_station_exactly_once():
    result = multi_run(np.full(581, 130.0), repair_scope="boat",
                       record_routes=True)
    planned = sorted(s for t in make_multi_trips()
                     for s in stations_of(t["nodes"]))
    for entry in result["repaired_routes"]:
        sailed = [s for route in entry["nodes"] for s in stations_of(route)]
        assert sorted(sailed) == planned
        assert len(sailed) == len(set(sailed))


def test_boat_scope_finishes_where_the_season_planned_to():
    result = multi_run(np.full(581, 130.0), repair_scope="boat",
                       record_routes=True)
    last_port = make_multi_trips()[-1]["nodes"][-1]
    for entry in result["repaired_routes"]:
        assert entry["nodes"][-1][-1] == last_port


def test_boat_scope_triggers_once_per_boat():
    """One boat, one trigger -- later trips are absorbed, not re-triggered."""
    result = multi_run(np.full(581, 130.0), repair_scope="boat")
    triggers = {}
    for e in result["overflow_events"]:
        triggers.setdefault(e["boat_id"], []).append(e["trip_idx"])
    for boat, idxs in triggers.items():
        assert len(set(idxs)) == 1, f"boat {boat} repaired from two trips"


def test_boat_scope_terminates_on_absurd_catch():
    result = multi_run(np.full(581, 1e6), repair_scope="boat", max_repairs=1)
    assert result["repair_cap_hit"] is True
    assert np.isfinite(result["total_time"])


# ---- 2-opt ----------------------------------------------------------------

def test_two_opt_is_off_by_default():
    catch = np.full(581, 130.0)
    assert multi_run(catch)["total_time"] == pytest.approx(
        multi_run(catch, repair_planner="nn")["total_time"])


def test_two_opt_never_lengthens_a_route():
    """It only accepts improving moves, so it cannot make a trip worse."""
    tm = make_time_matrix(travel=5.0)
    rng = np.random.default_rng(0)
    tm = tm + rng.uniform(0, 20, tm.shape)
    tm = (tm + tm.T) / 2                    # 2-opt assumes symmetry
    np.fill_diagonal(tm, 0.0)

    for stations in ([0, 1, 2, 3, 4, 5], [2, 7, 1, 9, 4], [0, 3, 6, 8]):
        plan = _nn_resplit(0, stations, np.full(581, 1.0), 1e9, 0, tm, 1e9)
        for sub in plan:
            after = _two_opt(sub["nodes"], tm, 1e9)
            assert _leg_time(after, tm) <= _leg_time(sub["nodes"], tm) + 1e-9


def test_two_opt_keeps_the_same_stations():
    tm = make_time_matrix(travel=5.0)
    rng = np.random.default_rng(1)
    tm = tm + rng.uniform(0, 20, tm.shape)
    tm = (tm + tm.T) / 2
    np.fill_diagonal(tm, 0.0)

    plan = _nn_resplit(0, [0, 1, 2, 3, 4, 5], np.full(581, 1.0), 1e9, 0, tm, 1e9)
    for sub in plan:
        after = _two_opt(sub["nodes"], tm, 1e9)
        assert sorted(stations_of(after)) == sorted(stations_of(sub["nodes"]))
        assert after[0] == sub["nodes"][0] and after[-1] == sub["nodes"][-1]


def test_two_opt_keeps_tow_pairs_adjacent():
    """Reversing a segment flips each tow, but both nodes must stay together."""
    tm = make_time_matrix(travel=5.0)
    rng = np.random.default_rng(2)
    tm = tm + rng.uniform(0, 20, tm.shape)
    tm = (tm + tm.T) / 2
    np.fill_diagonal(tm, 0.0)

    plan = _nn_resplit(0, [0, 1, 2, 3, 4], np.full(581, 1.0), 1e9, 0, tm, 1e9)
    for sub in plan:
        inner = _two_opt(sub["nodes"], tm, 1e9)[1:-1]
        for a, b in zip(inner[::2], inner[1::2]):
            assert abs(a - b) == 1
            assert (a - N_PORTS) // 2 == (b - N_PORTS) // 2


def test_two_opt_respects_the_fish_time_limit():
    """A shorter route can still push fish time over, since it drops the first leg."""
    tm = make_time_matrix(travel=5.0)
    plan = _nn_resplit(0, [0, 1, 2, 3, 4, 5], np.full(581, 1.0), 1e9, 0, tm,
                       fish_time_limit=40.0, two_opt=True)
    for sub in plan:
        if len(sub["stations"]) > 1:
            assert _leg_time(sub["nodes"][1:], tm) <= 40.0


def test_two_opt_leaves_a_route_it_cannot_improve():
    """Every leg costs the same here, so no reversal wins."""
    tm = make_time_matrix(travel=5.0)
    route = [0, 13, 14, 15, 16, 0]
    assert _two_opt(route, tm, 1e9) == route


def test_two_opt_reverses_a_two_station_trip():
    """Short trips are the common case for repair, so they must be tried.

    With different start and end ports the reversal is a real move, which is
    why the guard cannot skip anything with fewer than three stations.
    """
    tm = make_time_matrix(travel=5.0)
    tm[0, 16] = tm[16, 0] = 1.0            # cheap if station 1 goes first
    tm[13, 1] = tm[1, 13] = 1.0            # cheap if station 0 ends the trip

    route = [0, 13, 14, 15, 16, 1]
    out = _two_opt(route, tm, 1e9)
    assert _leg_time(out, tm) < _leg_time(route, tm)
    assert sorted(stations_of(out)) == sorted(stations_of(route))


def test_two_opt_flips_a_single_tow():
    """Which end of the tow the boat enters is a choice worth making."""
    tm = make_time_matrix(travel=5.0)
    tm[0, 14] = tm[14, 0] = 1.0            # cheaper to enter from the far end

    route = [0, 13, 14, 1]
    out = _two_opt(route, tm, 1e9)
    assert out == [0, 14, 13, 1]


# ---- solver re-planner ----------------------------------------------------

@pytest.mark.slow
def test_solver_resplit_covers_its_stations_exactly_once():
    plan = _solver_resplit(5, [0, 1, 2, 3, 4, 5, 6, 7], np.full(581, 4000.0),
                           14000.0, end_port=3, port_nodes=list(range(13)),
                           fish_time_limit=120.0)
    seen = [s for sub in plan for s in sub["stations"]]
    assert sorted(seen) == list(range(8))
    assert len(seen) == len(set(seen))


@pytest.mark.slow
def test_solver_resplit_starts_and_ends_where_asked():
    """The sub-problem docks where it started, so the last leg is re-pointed."""
    plan = _solver_resplit(5, [0, 1, 2, 3], np.full(581, 4000.0), 14000.0,
                           end_port=3, port_nodes=list(range(13)),
                           fish_time_limit=120.0)
    assert plan[0]["nodes"][0] == 5
    assert plan[-1]["nodes"][-1] == 3


@pytest.mark.slow
def test_solver_resplit_respects_capacity():
    belief = np.full(581, 4000.0)
    plan = _solver_resplit(5, list(range(10)), belief, 14000.0, end_port=5,
                           port_nodes=list(range(13)), fish_time_limit=120.0)
    for sub in plan:
        assert sum(belief[s] for s in sub["stations"]) <= 14000.0 + 1e-6


@pytest.mark.slow
def test_solver_resplit_of_nothing_is_nothing():
    assert _solver_resplit(5, [], np.full(581, 1.0), 14000.0, 5,
                           list(range(13)), 120.0) == []


@pytest.mark.slow
def test_solver_beats_nearest_neighbour_on_routing():
    """The point of using it: better routes than the greedy re-split."""
    from unified.evaluate import _load_time_matrix
    tm = _load_time_matrix()
    stations = list(range(40))
    belief = np.full(581, 2000.0)

    nn = _nn_resplit(5, stations, belief, 14000.0, 5, tm, 120.0)
    sol = _solver_resplit(5, stations, belief, 14000.0, 5, list(range(13)),
                          120.0)
    nn_time = sum(_leg_time(s["nodes"], tm) for s in nn)
    sol_time = sum(_leg_time(s["nodes"], tm) for s in sol)
    assert sol_time < nn_time
