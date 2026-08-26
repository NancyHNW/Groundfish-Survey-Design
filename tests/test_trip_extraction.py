"""Tests for solution_to_trips — what survives extraction from a solved Problem.

Regression cover for a bug where port-to-port repositioning legs were silently
dropped. Those legs have exactly two nodes, like an empty padding trip, but
unlike padding they carry real travel time. Dropping them made every reported
planned time too low, and hid the cost from the stochastic evaluator entirely,
since the evaluator only ever sees the extracted trips.

Uses lightweight stubs rather than a real solve, so these stay fast.

Run from repo root:
    python -m pytest tests/test_trip_extraction.py -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unified.evaluate import solution_to_trips


class FakeTrip:
    """Stands in for classes.Trip — only what solution_to_trips touches."""

    def __init__(self, nodes, total_time, fish_time=0.0, total_catch=0.0):
        self._nodes = list(nodes)
        self.total_time = total_time
        self.fish_time = fish_time
        self.total_catch = total_catch

    def get_full_trip(self):
        return list(self._nodes)


class FakeBoat:
    def __init__(self, boat_id, route):
        self.id = boat_id
        self.route = route


class FakeProblem:
    def __init__(self, boats):
        self.boats = boats


def test_working_trips_are_kept():
    prob = FakeProblem([FakeBoat(1, [
        FakeTrip([11, 13, 14, 15, 16, 11], total_time=40.0, total_catch=900.0),
    ])])
    trips = solution_to_trips(prob)
    assert len(trips) == 1
    assert trips[0]["total_time"] == 40.0
    assert trips[0]["boat_id"] == 1


def test_repositioning_leg_is_kept():
    """A two-node port-to-port leg with real travel time must survive.

    This is the regression: a boat sailing home from its last working port.
    """
    prob = FakeProblem([FakeBoat(1, [
        FakeTrip([11, 13, 14, 12], total_time=40.0, total_catch=900.0),
        FakeTrip([12, 11], total_time=11.32),      # sail home, no stations
    ])])
    trips = solution_to_trips(prob)

    assert len(trips) == 2, "the repositioning leg was dropped"
    assert sum(t["total_time"] for t in trips) == 51.32


def test_empty_padding_trip_is_dropped():
    """A zero-time two-node trip is padding for the tabu search, not a leg."""
    prob = FakeProblem([FakeBoat(1, [
        FakeTrip([11, 13, 14, 11], total_time=40.0, total_catch=900.0),
        FakeTrip([11, 11], total_time=0.0),
    ])])
    trips = solution_to_trips(prob)

    assert len(trips) == 1
    assert sum(t["total_time"] for t in trips) == 40.0


def test_extracted_time_matches_the_boat_total():
    """The invariant that failed in production: no time may go missing.

    sum over extracted trips must equal sum over the boats' routes, which is
    what the solver reports as its objective.
    """
    routes = {
        1: [FakeTrip([11, 13, 14, 12], 40.0, total_catch=900.0),
            FakeTrip([12, 15, 16, 12], 25.0, total_catch=800.0),
            FakeTrip([12, 11], 11.32),
            FakeTrip([11, 11], 0.0)],
        2: [FakeTrip([11, 17, 18, 12], 33.0, total_catch=700.0),
            FakeTrip([12, 11], 11.32)],
    }
    prob = FakeProblem([FakeBoat(bid, r) for bid, r in routes.items()])

    boat_total = sum(t.total_time for r in routes.values() for t in r)
    trip_total = sum(t["total_time"] for t in solution_to_trips(prob))

    assert trip_total == boat_total == 120.64


def test_repositioning_leg_has_no_stations():
    """Downstream code must cope with a trip that visits nothing.

    The evaluator derives station ordinals from nodes >= 13; for these legs
    that list is empty, so it must not assume every trip has stations.
    """
    prob = FakeProblem([FakeBoat(1, [FakeTrip([12, 11], total_time=11.32)])])
    trips = solution_to_trips(prob)

    assert len(trips) == 1
    assert [n for n in trips[0]["nodes"] if n >= 13] == []
    assert trips[0]["catch"] == 0.0
