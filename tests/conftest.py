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
    config.addinivalue_line(
        "markers", "full: tests against the real 581-station survey"
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


_CATCH_LOADER_ATTR = "load_catch_fixed_random_assignment"
_CATCH_LOADER_MODULES = ["data_loader", "classes", "neighbourhood_rules",
                         "ant_colony_optimisation",
                         "formulation_tester_functions"]


@pytest.fixture(autouse=True, scope="session")
def _pristine_catch_loader():
    """Capture the unpatched catch loader once, before any test runs.

    Importing has to happen inside heuristic_context, since data_loader is
    only importable with final_code on sys.path.
    """
    from unified.adapters import heuristic_context

    with heuristic_context():
        import data_loader  # noqa: F401  imported for the side effect

    return {name: getattr(sys.modules[name], _CATCH_LOADER_ATTR)
            for name in _CATCH_LOADER_MODULES
            if name in sys.modules
            and hasattr(sys.modules[name], _CATCH_LOADER_ATTR)}


@pytest.fixture(autouse=True)
def _restore_catch_loader(_pristine_catch_loader):
    """Undo the catch-loader monkey-patch after every test.

    to_heuristic_problem patches load_catch_fixed_random_assignment in
    data_loader and in every module that did `from data_loader import *`, and
    never restores it. That is deliberate in production -- Trip.__init__
    reloads catch from disk, so the patch has to outlive the call that set it.

    In a test session it leaks. Once any test asks for catch_source="gfsp",
    every later test gets gfsp data whatever it asked for, so
    catch_source="heuristic" silently returns the wrong array. The failure
    depends on file ordering, which is why it surfaced only when a new test
    file sorted ahead of test_unified.py.

    Restoring has to target the *pristine* loader captured at session start,
    not whatever was in place when this test began -- by then an earlier test
    may already have patched it, and restoring that just reinstates the leak.
    """
    try:
        yield
    finally:
        for name, original in _pristine_catch_loader.items():
            mod = sys.modules.get(name)
            if mod is not None:
                setattr(mod, _CATCH_LOADER_ATTR, original)


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
