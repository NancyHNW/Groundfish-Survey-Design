"""Load test problems from either gfsp_code or final_code (Saahil) format."""

import os
import glob
import numpy as np
import math

from .problem import ProblemInstance

# Root of the StartingMaterials directory (parent of unified/)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_GFSP_DATA = os.path.join(_ROOT, "gfsp_code", "data")
_HEURISTIC_DIR = os.path.join(_ROOT, "final_code")


# ---------------------------------------------------------------------------
# gfsp_code test problems
# ---------------------------------------------------------------------------

def load_gfsp_problem(ns, nv, cf, instance) -> ProblemInstance:
    """Load a test problem from gfsp_code/data/test_problems/.

    Parameters
    ----------
    ns : int   — number of stations (10, 20, 40, 60, 80, 100)
    nv : int   — number of vessels (2, 3, 4)
    cf : float — capacity factor (62.5, 125, 250)
    instance : int — instance number (1-30)

    Returns
    -------
    ProblemInstance
    """
    base_capacity = int(cf * ns)
    prob_path = os.path.join(
        _GFSP_DATA, "test_problems",
        str(ns), str(nv), str(base_capacity),
        f"ins_{ns}_{nv}_{base_capacity}_{instance}.txt",
    )
    if not os.path.isfile(prob_path):
        raise FileNotFoundError(f"Test problem not found: {prob_path}")

    with open(prob_path) as f:
        lines = f.readlines()

    port_ids = np.array(lines[1].strip().split(","), dtype=int)
    station_ids = np.array(lines[3].strip().split(","), dtype=int)
    home_port_index = int(lines[5].strip())
    capacities = np.array(lines[7].strip().split(","), dtype=float)

    # gfsp conventions for limits
    fish_time_limit = ns * 7.5 + 450
    work_limit = int(math.ceil(ns * ((1 / nv) + (1 / (3 * nv ** 2)))) * 2)

    return ProblemInstance(
        station_ids=station_ids,
        port_ids=port_ids,
        n_boats=nv,
        capacities=capacities,
        home_port_index=home_port_index,
        fish_time_limit=fish_time_limit,
        work_limit=work_limit,
        source="gfsp",
    )


def load_gfsp_full_problem(capacities=None, n_boats=4, home_port_index=0,
                           fish_time_limit=120, work_limit=150) -> ProblemInstance:
    """Load the full 581-station problem from gfsp_code."""
    prob_path = os.path.join(
        _GFSP_DATA, "test_problems", "full_problem", "full_problem.txt"
    )
    with open(prob_path) as f:
        lines = f.readlines()

    port_ids = np.array(lines[1].strip().split(","), dtype=int)
    station_ids = np.array(lines[3].strip().split(","), dtype=int)
    home_idx = int(lines[5].strip())
    caps = np.array(lines[7].strip().split(","), dtype=float)

    return ProblemInstance(
        station_ids=station_ids,
        port_ids=port_ids,
        n_boats=n_boats if capacities is not None else len(caps),
        capacities=capacities if capacities is not None else caps,
        home_port_index=home_idx,
        fish_time_limit=fish_time_limit,
        work_limit=work_limit,
        source="gfsp",
    )


# ---------------------------------------------------------------------------
# Heuristic test problems (final_code/better_test_problems)
# ---------------------------------------------------------------------------

def load_heuristic_problem(prob_id) -> ProblemInstance:
    """Load a test problem from final_code/better_test_problems/.

    Parameters
    ----------
    prob_id : int or str — problem number (e.g. 6, 12, 24, 34)

    Returns
    -------
    ProblemInstance
    """
    if isinstance(prob_id, int):
        prob_id = f"{prob_id:02d}"
    pattern = os.path.join(
        _HEURISTIC_DIR, "better_test_problems", f"test_problem{prob_id}*"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No test problem matching: {pattern}")

    with open(matches[0]) as f:
        result = f.read()

    split = result.split("; ")
    station_ids = np.array(eval(split[0]))   # station ordinals
    port_ids = np.array(eval(split[1]))       # port node indices
    n_boats = int(split[2])
    capacities = np.array(eval(split[3]))
    fish_time_limit = int(split[4])
    work_limit = int(split[5])
    upper_bound = float(split[6]) if len(split) > 6 else np.inf

    return ProblemInstance(
        station_ids=station_ids,
        port_ids=port_ids,
        n_boats=n_boats,
        capacities=capacities,
        home_port_index=0,  # Heuristic code uses port_ids[0] as home
        fish_time_limit=fish_time_limit,
        work_limit=work_limit,
        source="heuristic",
        upper_bound=upper_bound,
    )


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def list_gfsp_problems():
    """Return a list of available (ns, nv, capacity, instance) tuples."""
    base = os.path.join(_GFSP_DATA, "test_problems")
    results = []
    for ns_dir in sorted(os.listdir(base)):
        ns_path = os.path.join(base, ns_dir)
        if not os.path.isdir(ns_path) or ns_dir == "full_problem":
            continue
        ns = int(ns_dir)
        for nv_dir in sorted(os.listdir(ns_path)):
            nv_path = os.path.join(ns_path, nv_dir)
            if not os.path.isdir(nv_path):
                continue
            nv = int(nv_dir)
            for cap_dir in sorted(os.listdir(nv_path)):
                cap_path = os.path.join(nv_path, cap_dir)
                if not os.path.isdir(cap_path):
                    continue
                cap = int(cap_dir)
                for f in sorted(os.listdir(cap_path)):
                    if f.startswith("ins_") and f.endswith(".txt"):
                        ins = int(f.split("_")[-1].replace(".txt", ""))
                        results.append((ns, nv, cap, ins))
    return results


def list_heuristic_problems():
    """Return a list of available problem IDs (as ints)."""
    pattern = os.path.join(
        _HEURISTIC_DIR, "better_test_problems", "test_problem*.txt"
    )
    results = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        # extract digits after "test_problem"
        num = name.replace("test_problem", "").replace(".txt", "")
        try:
            results.append(int(num))
        except ValueError:
            pass
    return results
