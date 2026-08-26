"""Shared fixtures and configuration for unified layer tests."""

import os
import sys
import json
import numpy as np
import pytest

# Ensure the repo root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: tests that run solvers (may take 10-60s each)"
    )


BASELINES_PATH = os.path.join(os.path.dirname(__file__), "golden_baselines.json")


def load_baselines():
    if not os.path.isfile(BASELINES_PATH):
        return {}
    try:
        with open(BASELINES_PATH) as f:
            return json.load(f)
    except json.JSONDecodeError:
        # A half-written file from an older run. Start clean rather than
        # failing every test that wants baselines.
        print(f"\n{BASELINES_PATH} is not valid JSON, ignoring it.")
        return {}


def _json_safe(obj):
    """Convert numpy scalars json cannot serialise into Python ones.

    Solutions come back from the solver as lists of numpy int64, which
    json.dump rejects.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def save_baselines(data):
    # Serialise fully before touching the file. json.dump writes as it goes, so
    # dumping straight to the open file leaves a truncated, unloadable file
    # behind if anything raises part way through.
    text = json.dumps(data, indent=2, default=_json_safe)
    with open(BASELINES_PATH, "w") as f:
        f.write(text)


@pytest.fixture(scope="session")
def baselines():
    """Load or initialise golden baselines dict.

    After the session, any new baselines are written back to disk.
    """
    data = load_baselines()
    yield data
    save_baselines(data)


@pytest.fixture(autouse=True, scope="session")
def _patch_time_matrix_caching():
    """Prevent Trip/Boat from re-loading the 11 MB time matrix on every __init__.

    The original heuristic code (classes.py) calls np.load() in every
    Trip.__init__ and Boat.__init__, creating a new 10.5 MB array each time.
    This exhausts memory when many objects are created across tests.

    This fixture patches them to reuse the module-level ``time_matrix``
    that is loaded once when ``classes`` is first imported.
    """
    from unified.adapters import heuristic_context

    with heuristic_context():
        import classes

        _cached_time = classes.time_matrix  # loaded once at import time

        _orig_trip_init = classes.Trip.__init__
        _orig_boat_init = classes.Boat.__init__

        def _patched_trip_init(self, start_port, stations=None, end_port=None, id=0):
            # keep in sync with classes.Trip.__init__ (this version skips the time reload)
            self.start_port = start_port
            self.end_port = start_port if end_port is None else end_port
            self.stations = list(stations) if stations is not None else []
            self.nice_trip = []
            self.actual_trip = []
            self.total_dist = 0
            self.total_catch = 0
            self.total_time = 0
            self.fish_time = 0
            self.route_lines = []
            self.id = id
            self.time = _cached_time
            self.catch = classes.load_catch_fixed_random_assignment().to_numpy().reshape(-1)

        def _patched_boat_init(self, id, capacity, home_port, trip_w_port=True,
                               capacity_buffer=1.0):
            self.id = id
            self.capacity = capacity
            self.planning_capacity = capacity * capacity_buffer
            self.home_port = home_port
            if trip_w_port:
                self.route = [classes.Trip(start_port=home_port)]
            else:
                self.route = [classes.Trip(start_port=home_port, id=f'trip:{id}:1')]
            self.stations = []
            self.station_dict = {}
            self.current_node = home_port
            self.current_load = 0
            self.current_fish_time = 0
            self.total_time = 0
            self.time = _cached_time
            self.catch = classes.load_catch_fixed_random_assignment().to_numpy().reshape(-1)

        classes.Trip.__init__ = _patched_trip_init
        classes.Boat.__init__ = _patched_boat_init

    yield

    # Restore originals (not strictly necessary since tests end here)
    with heuristic_context():
        import classes
        classes.Trip.__init__ = _orig_trip_init
        classes.Boat.__init__ = _orig_boat_init
