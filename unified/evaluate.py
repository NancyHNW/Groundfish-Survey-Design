"""Common solution evaluation and comparison utilities."""

import numpy as np
import os

from .problem import ProblemInstance, N_PORTS

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _load_time_matrix():
    path = os.path.join(_ROOT, "final_code", "local data", "true_time.npy")
    return np.load(path)


def _load_dist_matrix():
    path = os.path.join(_ROOT, "gfsp_code", "data", "true_dist.npy")
    dist = np.load(path)
    dist[dist == -1] = 0
    dist += dist.T
    dist[dist == 0] = np.inf
    np.fill_diagonal(dist, 0)
    return dist


def evaluate_heuristic_solution(prob, instance: ProblemInstance) -> dict:
    """Evaluate a solved heuristic Problem and return a results dict.

    Parameters
    ----------
    prob : Problem  (from final_code/classes.py, already solved)
    instance : ProblemInstance

    Returns
    -------
    dict with keys: objective, penalty, feasible, trips, boats_summary
    """
    k_catch = max(1.0, (400000 - np.mean(instance.capacities)) * 0.001)
    k_fish = np.mean(instance.capacities) / instance.fish_time_limit * k_catch
    upper_bound = instance.upper_bound if np.isfinite(instance.upper_bound) else 10000

    # "feasible" is judged against the true boat capacity; when the solver planned
    # to a buffered capacity we also report whether it stayed inside that buffer
    obj, pen = prob.evaluate_solution(k_catch, k_fish, upper_bound,
                                      use_planning_capacity=False)
    feasible = pen < 1e-8
    _, pen_planning = prob.evaluate_solution(k_catch, k_fish, upper_bound)

    trips = solution_to_trips(prob)
    boats_summary = []
    for boat in prob.boats:
        boats_summary.append({
            "id": boat.id,
            "total_time": boat.total_time,
            "n_trips": len(boat.route),
            "n_stations": len(boat.stations) // 2,
        })

    return {
        "objective": obj,
        "penalty": pen,
        "feasible": feasible,
        "penalty_planning": pen_planning,
        "feasible_planning": pen_planning < 1e-8,
        "total_time": obj,
        "trips": trips,
        "boats_summary": boats_summary,
    }


def solution_to_trips(prob) -> list:
    """Extract trips from a solved heuristic Problem as a list of node-index lists."""
    all_trips = []
    for boat in prob.boats:
        for trip in boat.route:
            full = trip.get_full_trip()
            # A repositioning leg (boat sailing home from its last working
            # port) has 2 nodes but real travel time, so it has to be kept or
            # the total comes out too low. Empty padding trips also have 2
            # nodes but cost 0, so the time check is what tells them apart.
            if len(full) > 2 or trip.total_time > 0:
                all_trips.append({
                    "boat_id": boat.id,
                    "nodes": full,
                    "total_time": trip.total_time,
                    "fish_time": trip.fish_time,
                    "catch": trip.total_catch,
                })
    return all_trips


def compute_total_distance(trips, dist_matrix=None):
    """Compute total distance for a list of trips (node sequences)."""
    if dist_matrix is None:
        dist_matrix = _load_dist_matrix()

    total = 0.0
    for trip in trips:
        nodes = trip["nodes"] if isinstance(trip, dict) else trip
        for i in range(len(nodes) - 1):
            total += dist_matrix[nodes[i], nodes[i + 1]]
    return total


def compute_total_time(trips, time_matrix=None):
    """Compute total time for a list of trips (node sequences)."""
    if time_matrix is None:
        time_matrix = _load_time_matrix()

    total = 0.0
    for trip in trips:
        nodes = trip["nodes"] if isinstance(trip, dict) else trip
        for i in range(len(nodes) - 1):
            total += time_matrix[nodes[i], nodes[i + 1]]
    return total


def compare_results(*results, labels=None) -> str:
    """Print a comparison table of multiple result dicts.

    Parameters
    ----------
    *results : dicts from evaluate_heuristic_solution or similar
    labels : list[str], optional names for each result

    Returns
    -------
    str — formatted comparison table
    """
    if labels is None:
        labels = [f"Result {i+1}" for i in range(len(results))]

    header = f"{'':20s} | " + " | ".join(f"{l:>15s}" for l in labels)
    sep = "-" * len(header)
    rows = [sep, header, sep]

    keys = ["objective", "penalty", "feasible", "total_time"]
    for key in keys:
        vals = []
        for r in results:
            v = r.get(key, "N/A")
            if isinstance(v, float):
                vals.append(f"{v:>15.2f}")
            elif isinstance(v, bool):
                vals.append(f"{'Yes':>15s}" if v else f"{'No':>15s}")
            else:
                vals.append(f"{str(v):>15s}")
        rows.append(f"{key:20s} | " + " | ".join(vals))

    # Boat summaries
    for r, label in zip(results, labels):
        if "boats_summary" in r:
            rows.append(sep)
            rows.append(f"  {label} — per-boat breakdown:")
            for b in r["boats_summary"]:
                rows.append(
                    f"    Boat {b['id']}: time={b['total_time']:.2f}, "
                    f"trips={b['n_trips']}, stations={b['n_stations']}"
                )

    rows.append(sep)
    return "\n".join(rows)
