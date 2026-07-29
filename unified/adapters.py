"""Adapters that convert a ProblemInstance into solver-specific objects.

The heuristic code (final_code/) relies on hardcoded relative paths for data
files, so we temporarily change the working directory when constructing
his Problem objects.  Use the ``heuristic_context`` context manager to
handle this safely.
"""

import os
import sys
import contextlib
import numpy as np

from .problem import ProblemInstance

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_FINAL_CODE = os.path.join(_ROOT, "final_code")
_GFSP_SRC = os.path.join(_ROOT, "gfsp_code", "src")


# ---------------------------------------------------------------------------
# Context manager for The heuristic code
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def heuristic_context():
    """Context manager that sets up CWD and sys.path for The heuristic code.

    Usage::

        with heuristic_context():
            from classes import Problem, Boat, Trip
            from neighbourhood_rules import *
            prob = to_heuristic_problem(instance)
            prob.GRASP(rcl_size=2, seed=42)
    """
    old_cwd = os.getcwd()
    added_path = False
    try:
        os.chdir(_FINAL_CODE)
        if _FINAL_CODE not in sys.path:
            sys.path.insert(0, _FINAL_CODE)
            added_path = True
        yield
    finally:
        os.chdir(old_cwd)
        if added_path and _FINAL_CODE in sys.path:
            sys.path.remove(_FINAL_CODE)


# ---------------------------------------------------------------------------
# Convert to the heuristic code's Problem
# ---------------------------------------------------------------------------

def to_heuristic_problem(instance: ProblemInstance, catch_source="heuristic"):
    """Convert a ProblemInstance to the heuristic ``Problem`` class.

    Must be called inside a ``heuristic_context()`` block so that relative
    data paths resolve correctly.

    Parameters
    ----------
    instance : ProblemInstance
    catch_source : str
        Which catch data to use:
        - "heuristic" (RandomMatchingCatches.dat, mean 859 kg)
        - "gfsp" (LimitedCatch.csv, mean 531 kg)
        - "historical" (historical station means from CatchSimulator, mean 812 kg)

    Returns
    -------
    Problem  (from final_code/classes.py)
    """
    # Import here because it only works when CWD is final_code/
    from classes import Problem

    node_indices = list(instance.station_ids_to_node_indices())
    home_ports = [instance.home_port] * instance.n_boats

    prob = Problem(
        stations=node_indices,
        ports=list(instance.port_ids),
        fish_time_limit=instance.fish_time_limit,
        n_boats=instance.n_boats,
        boat_capacities=list(instance.capacities),
        home_ports=home_ports,
    )

    if catch_source == "gfsp":
        _patch_catch_loader_gfsp()
        override_catch_data(prob)
    elif catch_source == "historical":
        hist_means = _load_historical_catch_array()
        _patch_catch_loader_historical(hist_means)
        override_catch_data(prob, catch_array=hist_means)

    return prob


def _load_gfsp_catch_array():
    """Load catch data from gfsp_code/data/LimitedCatch.csv as a 1D array."""
    import pandas as pd

    catch_file = os.path.join(_ROOT, "gfsp_code", "data", "LimitedCatch.csv")
    with open(catch_file) as f:
        lines = [line.split(",") for line in f.readlines()]
    catch_df = pd.DataFrame(np.array(lines)[1:, [1]], columns=["catch"])
    catch_df["catch"] = catch_df["catch"].astype(float)
    return catch_df


def _patch_catch_loader_gfsp():
    """Monkey-patch the catch loader so that new Boat/Trip objects created
    during reset() and GRASP() also use gfsp catch data.

    This patches ``load_catch_fixed_random_assignment`` in all modules
    that imported it via ``from data_loader import *``.
    """
    gfsp_catch_df = _load_gfsp_catch_array()

    def patched_loader():
        return gfsp_catch_df

    # Patch in data_loader itself
    import data_loader
    data_loader.load_catch_fixed_random_assignment = patched_loader

    # Patch in modules that did `from data_loader import *`
    for mod_name in ["classes", "neighbourhood_rules", "ant_colony_optimisation",
                     "formulation_tester_functions"]:
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "load_catch_fixed_random_assignment"):
            mod.load_catch_fixed_random_assignment = patched_loader


def _load_historical_catch_array():
    """Load historical mean catch per station from CatchSimulator.

    Returns a 1D numpy array of shape (581,) with the fitted mean catch
    for each station ordinal.
    """
    from .stochastic_catch import CatchSimulator
    sim = CatchSimulator()
    sim.fit()
    return sim.means


def _patch_catch_loader_historical(historical_means=None):
    """Monkey-patch the catch loader to use historical station means."""
    import pandas as pd
    if historical_means is None:
        historical_means = _load_historical_catch_array()
    catch_df = pd.DataFrame({"catch": historical_means})

    def patched_loader():
        return catch_df

    import data_loader
    data_loader.load_catch_fixed_random_assignment = patched_loader

    for mod_name in ["classes", "neighbourhood_rules", "ant_colony_optimisation",
                     "formulation_tester_functions"]:
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "load_catch_fixed_random_assignment"):
            mod.load_catch_fixed_random_assignment = patched_loader


def override_catch_data(prob, catch_file=None, catch_array=None):
    """Replace catch data in a heuristic Problem (and all its Boats/Trips).

    By default loads gfsp_code's LimitedCatch.csv.  This is needed when
    running heuristics on gfsp test problems, because the boat capacities
    were designed around that catch data.

    Parameters
    ----------
    prob : Problem (from final_code/classes.py)
    catch_file : str or None — path to a CSV with columns [station_number, catch].
                 If None, uses gfsp_code/data/LimitedCatch.csv.
    catch_array : np.ndarray or None — pre-loaded catch array (shape 581,).
                  If provided, catch_file is ignored.
    """
    if catch_array is not None:
        pass  # use the provided array directly
    elif catch_file is None:
        catch_array = _load_gfsp_catch_array().to_numpy().reshape(-1)
    else:
        import pandas as pd
        with open(catch_file) as f:
            lines = [line.split(",") for line in f.readlines()]
        catch_df = pd.DataFrame(np.array(lines)[1:, [1]], columns=["catch"])
        catch_df["catch"] = catch_df["catch"].astype(float)
        catch_array = catch_df.to_numpy().reshape(-1)

    # Override on Problem
    prob.catch = catch_array
    # Override on all Boats and their Trips
    for boat in prob.boats:
        boat.catch = catch_array
        for trip in boat.route:
            trip.catch = catch_array


# ---------------------------------------------------------------------------
# Convert to gfsp MIP model
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def gfsp_context():
    """Context manager that sets up sys.path for gfsp_code imports."""
    added_path = False
    try:
        parent = os.path.join(_ROOT, "gfsp_code")
        if parent not in sys.path:
            sys.path.insert(0, parent)
            added_path = True
        yield
    finally:
        if added_path and parent in sys.path:
            sys.path.remove(parent)


def to_gfsp_model(instance: ProblemInstance, formulation: int = 2):
    """Convert a ProblemInstance to a gfsp MIP model.

    Must be called inside a ``gfsp_context()`` block.

    Parameters
    ----------
    instance : ProblemInstance
    formulation : int
        1 = ThreeIndexUncompressed
        2 = ThreeIndexCompressed
        3 = TwoIndexUncompressed
        4 = TwoIndexCompressed

    Returns
    -------
    A model object with ``.build_model()`` and ``.optimize()`` methods.
    """
    from src.opt.ThreeIndUncomp import ThreeIndexUncompressedModel
    from src.opt.ThreeIndComp import ThreeIndexCompressedModel
    from src.opt.TwoIndUncomp import TwoIndexUncompressedModel
    from src.opt.TwoIndComp import TwoIndexCompressedModel

    # gfsp models load data via (ns, nv, cf, t_i) — we need to write a
    # temporary test problem file in gfsp format so the constructor can
    # load it through its normal path.
    ns = instance.ns
    nv = instance.n_boats
    # Reverse-engineer the capacity factor from the first vessel's capacity
    cf = instance.capacities[0] / ns

    # Write temporary test problem file
    base_cap = int(cf * ns)
    tmp_dir = os.path.join(
        _ROOT, "gfsp_code", "data", "test_problems",
        str(ns), str(nv), str(base_cap),
    )
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"ins_{ns}_{nv}_{base_cap}_tmp.txt")

    with open(tmp_path, "w") as f:
        f.write("PORT_SECTION\n")
        f.write(",".join(map(str, instance.port_ids)) + "\n")
        f.write("STATION_SECTION\n")
        f.write(",".join(map(str, instance.station_ids)) + "\n")
        f.write("HOME_PORT_INDEX_SECTION\n")
        f.write(f"{instance.home_port_index}\n")
        f.write("CAPACITIES_SECTION\n")
        f.write(",".join(str(c) for c in instance.capacities))

    model_classes = {
        1: ThreeIndexUncompressedModel,
        2: ThreeIndexCompressedModel,
        3: TwoIndexUncompressedModel,
        4: TwoIndexCompressedModel,
    }
    cls = model_classes[formulation]

    # The constructor uses the instance number to find the file via
    # load_test_problem(ns, nv, base_cap, t_i).  We wrote "tmp" as
    # the instance identifier, so we pass t_i="tmp".
    model = cls(ns=ns, nv=nv, cf=cf, t_i="tmp")

    # Clean up the temp file
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return model
