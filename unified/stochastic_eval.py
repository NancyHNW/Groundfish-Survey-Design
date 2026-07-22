"""Stochastic evaluation of routing solutions under catch uncertainty."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from .problem import ProblemInstance, N_PORTS
from .evaluate import _load_time_matrix

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _node_to_station(node):
    """Convert a raw node index (13+) to a station ordinal (0-580)."""
    return (node - N_PORTS) // 2


def _extract_station_ordinals(trip_nodes):
    """Extract unique station ordinals visited in a trip (skip ports).

    Stations come in pairs in the node list, so we step by 2 through
    the non-port nodes to avoid double-counting.

    Returns list of station ordinals in visit order.
    """
    stations = []
    non_port = [n for n in trip_nodes if n >= N_PORTS]
    for i in range(0, len(non_port), 2):
        stations.append(_node_to_station(non_port[i]))
    return stations


def evaluate_single_realisation(solution_trips, instance, catch_vector,
                                time_matrix=None, strategy="backtrack",
                                preemptive_threshold=0.8):
    """Evaluate a routing solution under one catch realisation.

    Parameters
    ----------
    solution_trips : list[dict]
        Trip dicts from evaluate.py's solution_to_trips().
        Each has keys: boat_id, nodes, total_time, fish_time, catch.
    instance : ProblemInstance
        Problem parameters (capacities, fish_time_limit, port_ids).
    catch_vector : np.ndarray, shape (581,)
        Simulated catch per station ordinal for this realisation.
    time_matrix : np.ndarray, optional
        Pre-loaded time matrix (1175x1175). Loaded if not provided.
    strategy : str
        How to handle capacity overflow:
        - "backtrack": go to nearest port, return to where you left off,
          continue the planned route. (default)
        - "forward": go to nearest port, then continue to the nearest
          unvisited station from that port (no backtracking).
        - "preemptive": return to nearest port when load reaches
          preemptive_threshold fraction of capacity, before overflow.
    preemptive_threshold : float
        Fraction of capacity that triggers a preemptive return (0.0–1.0).
        Only used when strategy="preemptive". Default 0.8.

    Returns
    -------
    dict with keys:
        capacity_exceeded : bool — did ANY trip exceed capacity?
        fish_time_violated : bool — did ANY trip exceed fish_time_limit?
        n_unscheduled_returns : int — number of forced port returns
        total_time : float — total survey time (with detour penalties)
        trip_details : list[dict] — per-trip breakdown
        overflow_events : list[dict] — where each overflow happened
        strategy : str — which strategy was used
    """
    if strategy not in ("backtrack", "forward", "preemptive"):
        raise ValueError(f"Unknown strategy: {strategy!r}. "
                         f"Use 'backtrack', 'forward', or 'preemptive'.")

    if time_matrix is None:
        time_matrix = _load_time_matrix()

    # Build mapping from boat_id to capacity index.
    # Solver boat IDs may be 1-indexed (1,2,...) while capacities are 0-indexed.
    all_boat_ids = set(t["boat_id"] for t in solution_trips)
    max_bid = max(all_boat_ids)
    if max_bid >= instance.n_boats:
        # 1-indexed IDs: subtract 1
        boat_id_to_cap_idx = {bid: bid - min(all_boat_ids) for bid in all_boat_ids}
    else:
        # 0-indexed IDs: use directly
        boat_id_to_cap_idx = {bid: bid for bid in all_boat_ids}

    port_nodes = instance.port_ids
    capacity_exceeded = False
    fish_time_violated = False
    n_unscheduled_returns = 0
    original_total_time = 0.0
    total_time = 0.0
    trip_details = []
    overflow_events = []

    for trip_idx, trip in enumerate(solution_trips):
        nodes = trip["nodes"]
        boat_id = trip["boat_id"]
        boat_capacity = instance.capacities[boat_id_to_cap_idx[boat_id]]

        # Extract station ordinals and their corresponding node indices
        non_port = [n for n in nodes if n >= N_PORTS]
        station_ordinals = []
        station_nodes = []
        for i in range(0, len(non_port), 2):
            station_ordinals.append(_node_to_station(non_port[i]))
            station_nodes.append(non_port[i])

        if strategy == "forward":
            detour_time = _walk_forward(
                station_ordinals, station_nodes, catch_vector,
                boat_capacity, port_nodes, time_matrix,
                trip_idx, boat_id, overflow_events,
            )
        elif strategy == "preemptive":
            detour_time = _walk_preemptive(
                station_ordinals, station_nodes, catch_vector,
                boat_capacity, preemptive_threshold, port_nodes, time_matrix,
                trip_idx, boat_id, overflow_events,
            )
        else:  # backtrack
            detour_time = _walk_backtrack(
                station_ordinals, station_nodes, catch_vector,
                boat_capacity, port_nodes, time_matrix,
                trip_idx, boat_id, overflow_events,
            )

        trip_exceeded = detour_time > 0
        trip_unscheduled = sum(
            1 for e in overflow_events if e["trip_idx"] == trip_idx
        )

        trip_fish_time = trip["fish_time"]
        adjusted_time = trip["total_time"] + detour_time

        if trip_fish_time > instance.fish_time_limit:
            fish_time_violated = True
        if trip_exceeded:
            capacity_exceeded = True
        n_unscheduled_returns += trip_unscheduled
        original_total_time += trip["total_time"]
        total_time += adjusted_time

        trip_details.append({
            "boat_id": boat_id,
            "original_time": trip["total_time"],
            "adjusted_time": adjusted_time,
            "detour_time": detour_time,
            "original_catch": trip["catch"],
            "simulated_catch": sum(catch_vector[s] for s in station_ordinals),
            "capacity": boat_capacity,
            "exceeded": trip_exceeded,
            "n_unscheduled_returns": trip_unscheduled,
        })

    return {
        "capacity_exceeded": capacity_exceeded,
        "fish_time_violated": fish_time_violated,
        "n_unscheduled_returns": n_unscheduled_returns,
        "original_total_time": original_total_time,
        "total_time": total_time,
        "time_penalty": total_time - original_total_time,
        "trip_details": trip_details,
        "overflow_events": overflow_events,
        "strategy": strategy,
    }


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _walk_backtrack(station_ordinals, station_nodes, catch_vector,
                    boat_capacity, port_nodes, time_matrix,
                    trip_idx, boat_id, overflow_events):
    """Original strategy: nearest port, then backtrack to resume the route."""
    cumulative_catch = 0.0
    detour_time = 0.0

    for i, stn in enumerate(station_ordinals):
        cumulative_catch += catch_vector[stn]
        if cumulative_catch > boat_capacity:
            stn_node = station_nodes[i]
            nearest_port, return_time = _find_nearest_port(
                stn_node, port_nodes, time_matrix
            )
            detour = 2 * return_time  # go to port and come back
            detour_time += detour

            overflow_events.append({
                "trip_idx": trip_idx,
                "boat_id": boat_id,
                "station_ordinal": stn,
                "station_node": stn_node,
                "cumulative_catch": cumulative_catch,
                "capacity": boat_capacity,
                "nearest_port": nearest_port,
                "detour_time": detour,
            })
            cumulative_catch = 0.0

    return detour_time


def _walk_forward(station_ordinals, station_nodes, catch_vector,
                  boat_capacity, port_nodes, time_matrix,
                  trip_idx, boat_id, overflow_events):
    """Forward strategy: nearest port, then continue to nearest unvisited
    station from that port instead of backtracking.

    Detour cost = time(overflow_station -> port) + time(port -> next_station)
                  - time(overflow_station -> next_station)
    i.e. the extra time compared to going directly to the next station.
    """
    cumulative_catch = 0.0
    detour_time = 0.0
    n_stations = len(station_ordinals)

    for i, stn in enumerate(station_ordinals):
        cumulative_catch += catch_vector[stn]
        if cumulative_catch > boat_capacity:
            stn_node = station_nodes[i]
            nearest_port, time_to_port = _find_nearest_port(
                stn_node, port_nodes, time_matrix
            )

            if i + 1 < n_stations:
                # There's a next station — compute the forward detour
                next_node = station_nodes[i + 1]
                direct_time = time_matrix[stn_node, next_node]
                via_port_time = (time_to_port +
                                 time_matrix[nearest_port, next_node])
                detour = via_port_time - direct_time
            else:
                # Last station in trip — just go to port and back
                # (same as backtrack for the final station)
                detour = 2 * time_to_port

            detour_time += detour

            overflow_events.append({
                "trip_idx": trip_idx,
                "boat_id": boat_id,
                "station_ordinal": stn,
                "station_node": stn_node,
                "cumulative_catch": cumulative_catch,
                "capacity": boat_capacity,
                "nearest_port": nearest_port,
                "detour_time": detour,
            })
            cumulative_catch = 0.0

    return detour_time


def _walk_preemptive(station_ordinals, station_nodes, catch_vector,
                     boat_capacity, threshold, port_nodes, time_matrix,
                     trip_idx, boat_id, overflow_events):
    """Preemptive strategy: return to port when load reaches threshold
    fraction of capacity, before overflow actually happens.

    Uses backtrack-style detour (nearest port, return to same spot).
    """
    cumulative_catch = 0.0
    detour_time = 0.0
    trigger_level = boat_capacity * threshold

    for i, stn in enumerate(station_ordinals):
        cumulative_catch += catch_vector[stn]

        # Check if we've hit the threshold OR actually overflowed
        exceeded = cumulative_catch > boat_capacity
        preemptive = (not exceeded) and (cumulative_catch >= trigger_level)

        if exceeded or preemptive:
            stn_node = station_nodes[i]
            nearest_port, return_time = _find_nearest_port(
                stn_node, port_nodes, time_matrix
            )
            detour = 2 * return_time
            detour_time += detour

            overflow_events.append({
                "trip_idx": trip_idx,
                "boat_id": boat_id,
                "station_ordinal": stn,
                "station_node": stn_node,
                "cumulative_catch": cumulative_catch,
                "capacity": boat_capacity,
                "nearest_port": nearest_port,
                "detour_time": detour,
                "preemptive": preemptive,
            })
            cumulative_catch = 0.0

    return detour_time


def _find_nearest_port(station_node, port_nodes, time_matrix):
    """Find the nearest port to a station node by travel time.

    Returns (port_node, travel_time).
    """
    best_port = port_nodes[0]
    best_time = time_matrix[station_node, port_nodes[0]]
    for p in port_nodes[1:]:
        t = time_matrix[station_node, p]
        if t < best_time:
            best_time = t
            best_port = p
    return best_port, best_time


# ---------------------------------------------------------------------------
# Monte Carlo evaluation
# ---------------------------------------------------------------------------


class StochasticEvaluator:
    """Run Monte Carlo evaluation of a routing solution under catch uncertainty.

    Accepts either a CatchSimulator (Jess's module) or a pre-generated
    scenario matrix so you can test without the simulator.

    Parameters
    ----------
    simulator : object, optional
        Any object with a .sample(station_ids, n_samples) method returning
        shape (n_samples, len(station_ids)). Plug in CatchSimulator when ready.
    scenarios : np.ndarray, optional
        Pre-generated catch scenarios, shape (n_simulations, 581).
        If provided, simulator is ignored and n_simulations is inferred.
    n_simulations : int
        Number of Monte Carlo realisations (used with simulator only).
    seed : int, optional
        Random seed for reproducibility (used with simulator only).
    """

    def __init__(self, simulator=None, scenarios=None, n_simulations=1000,
                 seed=None):
        if scenarios is not None:
            self.scenarios = np.asarray(scenarios)
            self.n_simulations = self.scenarios.shape[0]
            self.simulator = None
        elif simulator is not None:
            self.simulator = simulator
            self.scenarios = None
            self.n_simulations = n_simulations
        else:
            raise ValueError("Provide either simulator or scenarios")
        self.seed = seed

    def evaluate(self, solution_trips, instance):
        """Run n_simulations catch realisations against the fixed route.

        Parameters
        ----------
        solution_trips : list[dict]
            Trip dicts from evaluate.py's solution_to_trips().
        instance : ProblemInstance
            Problem parameters.

        Returns
        -------
        dict with keys:
            p_capacity_exceedance : float
                P(any trip exceeds capacity) across simulations.
            p_fish_time_violation : float
                P(any trip exceeds fish time) across simulations.
            expected_unscheduled_returns : float
                E[number of forced port returns].
            total_time_distribution : dict
                mean, std, p5, p50, p95 of total survey time.
            time_penalty_distribution : dict
                mean, std, p5, p50, p95 of extra time from detours.
            per_trip_exceedance_probs : list[float]
                Per-trip probability of capacity exceedance.
            worst_case_catch : np.ndarray
                95th percentile catch per station across simulations.
            all_results : list[dict]
                Raw per-realisation results (for further analysis).
        """
        time_matrix = _load_time_matrix()

        # Pre-generate all scenarios if using simulator
        if self.scenarios is not None:
            all_catch = self.scenarios
        else:
            all_catch = self._generate_scenarios(instance)

        n_trips = len(solution_trips)
        results = []

        for i in range(self.n_simulations):
            catch_vector = all_catch[i]
            result = evaluate_single_realisation(
                solution_trips, instance, catch_vector, time_matrix=time_matrix
            )
            results.append(result)

        # --- Aggregate ---
        capacity_exceeded = [r["capacity_exceeded"] for r in results]
        fish_time_violated = [r["fish_time_violated"] for r in results]
        unscheduled_returns = [r["n_unscheduled_returns"] for r in results]
        total_times = np.array([r["total_time"] for r in results])
        time_penalties = np.array([r["time_penalty"] for r in results])

        # Per-trip exceedance probabilities
        per_trip_exceeded = np.zeros(n_trips)
        for r in results:
            for t_idx, td in enumerate(r["trip_details"]):
                if td["exceeded"]:
                    per_trip_exceeded[t_idx] += 1
        per_trip_exceedance_probs = (per_trip_exceeded / self.n_simulations).tolist()

        # Worst-case catch (95th percentile per station)
        worst_case_catch = np.percentile(all_catch, 95, axis=0)

        return {
            "n_simulations": self.n_simulations,
            "p_capacity_exceedance": float(np.mean(capacity_exceeded)),
            "p_fish_time_violation": float(np.mean(fish_time_violated)),
            "expected_unscheduled_returns": float(np.mean(unscheduled_returns)),
            "total_time_distribution": {
                "mean": float(np.mean(total_times)),
                "std": float(np.std(total_times)),
                "p5": float(np.percentile(total_times, 5)),
                "p50": float(np.percentile(total_times, 50)),
                "p95": float(np.percentile(total_times, 95)),
            },
            "time_penalty_distribution": {
                "mean": float(np.mean(time_penalties)),
                "std": float(np.std(time_penalties)),
                "p5": float(np.percentile(time_penalties, 5)),
                "p50": float(np.percentile(time_penalties, 50)),
                "p95": float(np.percentile(time_penalties, 95)),
            },
            "per_trip_exceedance_probs": per_trip_exceedance_probs,
            "worst_case_catch": worst_case_catch,
            "all_results": results,
        }

    def _generate_scenarios(self, instance):
        """Generate catch scenarios using the simulator.

        The simulator handles its own seeding (set in its constructor).
        """
        all_catch = self.simulator.sample(
            n_scenarios=self.n_simulations,
            station_ids=instance.station_ids,
        )
        return all_catch


def print_stochastic_summary(stoch_result, deterministic_time=None):
    """Print a formatted summary of Monte Carlo evaluation results.

    Parameters
    ----------
    stoch_result : dict
        Output from StochasticEvaluator.evaluate().
    deterministic_time : float, optional
        Deterministic total time for comparison.
    """
    sep = "=" * 60
    print(sep)
    print(f"  STOCHASTIC EVALUATION SUMMARY  ({stoch_result['n_simulations']} simulations)")
    print(sep)

    print(f"\n  Risk metrics:")
    print(f"    P(capacity exceedance)     = {stoch_result['p_capacity_exceedance']:.1%}")
    print(f"    P(fish time violation)     = {stoch_result['p_fish_time_violation']:.1%}")
    print(f"    E[unscheduled returns]     = {stoch_result['expected_unscheduled_returns']:.2f}")

    td = stoch_result["total_time_distribution"]
    print(f"\n  Total survey time distribution:")
    print(f"    Mean   = {td['mean']:.1f}")
    print(f"    Std    = {td['std']:.1f}")
    print(f"    5th %%  = {td['p5']:.1f}")
    print(f"    Median = {td['p50']:.1f}")
    print(f"    95th %% = {td['p95']:.1f}")

    tp = stoch_result["time_penalty_distribution"]
    print(f"\n  Time penalty (from detours):")
    print(f"    Mean   = +{tp['mean']:.1f}")
    print(f"    95th %% = +{tp['p95']:.1f}")

    if deterministic_time is not None:
        print(f"\n  Deterministic vs stochastic:")
        print(f"    Deterministic time = {deterministic_time:.1f}")
        print(f"    Stochastic mean    = {td['mean']:.1f}  ({(td['mean']/deterministic_time - 1)*100:+.1f}%)")
        print(f"    Stochastic p95     = {td['p95']:.1f}  ({(td['p95']/deterministic_time - 1)*100:+.1f}%)")

    # Per-trip breakdown
    probs = stoch_result["per_trip_exceedance_probs"]
    if probs:
        print(f"\n  Per-trip exceedance probability:")
        for i, p in enumerate(probs):
            bar = "#" * int(p * 40)
            print(f"    Trip {i:2d}: {p:.1%}  {bar}")

    print(sep)


def plot_monte_carlo(stoch_result, save_path=None, deterministic_time=None,
                     title_suffix=None):
    """Histogram of total survey time across all simulations.

    Parameters
    ----------
    stoch_result : dict
        Output from StochasticEvaluator.evaluate().
    save_path : str, optional
        If given, save figure to this path instead of showing.
    deterministic_time : float, optional
        Predicted total time the route was planned for. Drawn as a reference
        line to compare against the simulated distribution.
    title_suffix : str, optional
        Extra text for the figure title (e.g. the heuristic name).
    """
    total_times = [r["total_time"] for r in stoch_result["all_results"]]
    td = stoch_result["total_time_distribution"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: total time histogram ---
    ax1.hist(total_times, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
    if deterministic_time is not None:
        ax1.axvline(deterministic_time, color="green", linestyle="-", linewidth=2.5,
                    label=f"Predicted: {deterministic_time:.1f}")
    ax1.axvline(td["mean"], color="red", linestyle="-", linewidth=2,
                label=f"Simulated mean: {td['mean']:.1f}")
    ax1.axvline(td["p5"], color="orange", linestyle="--", linewidth=1.5,
                label=f"5th: {td['p5']:.1f}")
    ax1.axvline(td["p95"], color="orange", linestyle="--", linewidth=1.5,
                label=f"95th: {td['p95']:.1f}")
    ax1.set_xlabel("Total Survey Time")
    ax1.set_ylabel("Frequency")
    hist_title = f"Total Time Distribution ({stoch_result['n_simulations']} simulations)"
    if title_suffix:
        hist_title += f"\n{title_suffix}"
    ax1.set_title(hist_title)
    ax1.legend()

    # --- Right: unscheduled returns histogram ---
    returns = [r["n_unscheduled_returns"] for r in stoch_result["all_results"]]
    max_ret = max(returns) if returns else 0
    bins = np.arange(-0.5, max_ret + 1.5, 1)
    ax2.hist(returns, bins=bins, color="#DD8452", alpha=0.8, edgecolor="white")
    ax2.axvline(stoch_result["expected_unscheduled_returns"], color="red",
                linestyle="-", linewidth=2,
                label=f"Mean: {stoch_result['expected_unscheduled_returns']:.1f}")
    ax2.set_xlabel("Number of Unscheduled Port Returns")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Unscheduled Returns Distribution")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _load_nodes():
    """Load all node coordinates (ports + stations) as (1175, 2) array.

    Returns (nodes, n_ports, n_stations).
    Coordinates are in degrees (lat, lon) — lon is positive for west (matching station convention).
    """
    import pandas as pd

    # Ports
    ports_path = os.path.join(_ROOT, "gfsp_code", "data", "ports.csv")
    ports_df = pd.read_csv(ports_path)
    ports_df["longitude"] = -ports_df["longitude"]  # flip sign to match station convention
    ports = ports_df[["latitude", "longitude"]].to_numpy()

    # Stations — degrees-minutes format, need conversion
    station_path = os.path.join(_ROOT, "gfsp_code", "data", "smb.2019.dat")
    with open(station_path) as f:
        lines = [line.split('\t') for line in f.readlines()]
    stn_raw = np.array(lines)[:, [3, 4, 5, 6]].astype(int)

    # Convert degree-minutes to decimal degrees (same as degmin2deg)
    def dm2deg(val):
        min_ = (val / 100) - np.floor(val / 10000.0) * 100.0
        return (val + (200.0 / 3.0) * min_) / 10000.0

    stn_deg = dm2deg(stn_raw.astype(float))
    # Each station has 2 nodes: (i_lat, i_lon) and (f_lat, f_lon)
    station_nodes = stn_deg.reshape(-1, 2)  # (n_stations*2, 2)

    nodes = np.concatenate([ports, station_nodes], axis=0)
    return nodes, ports.shape[0], len(lines)


def _load_island():
    """Load Iceland coastline boundary data."""
    island_path = os.path.join(_ROOT, "final_code", "data", "island.bin")
    with open(island_path, 'rb') as f:
        landata = np.fromfile(f, dtype=np.float32)
    half = int(landata.size / 2)
    return np.vstack((landata[:half], -landata[half:])).T


def _draw_base_map(ax, nodes, n_ports, island, instance):
    """Draw the shared base layer: coastline, tow lines, ports."""
    ax.plot(island[:, 1], island[:, 0], color="grey", linewidth=0.5)

    for s in instance.station_ids:
        n1 = s * 2 + n_ports
        n2 = s * 2 + n_ports + 1
        ax.plot([nodes[n1, 1], nodes[n2, 1]],
                [nodes[n1, 0], nodes[n2, 0]],
                color="#cccccc", linewidth=1, alpha=0.5)

    for p in instance.port_ids:
        ax.scatter(nodes[p, 1], nodes[p, 0],
                   c="#22A884", marker="o", s=80, zorder=5)


def _draw_routes(ax, solution_trips, nodes):
    """Draw trip routes coloured by boat."""
    boat_colors = list(mcolors.TABLEAU_COLORS.values())
    trip_styles = ["-", "--", ":", "-."]
    for trip_idx, trip in enumerate(solution_trips):
        trip_nodes = trip["nodes"]
        boat_id = trip["boat_id"]
        color = boat_colors[boat_id % len(boat_colors)]
        style = trip_styles[trip_idx % len(trip_styles)]

        lats = [nodes[n, 0] for n in trip_nodes]
        lons = [nodes[n, 1] for n in trip_nodes]
        ax.plot(lons, lats, color=color, linewidth=1.5, linestyle=style,
                alpha=0.8, label=f"Boat {boat_id}, Trip {trip_idx}")

        non_port = [n for n in trip_nodes if n >= N_PORTS]
        for i in range(0, len(non_port), 2):
            mid_lat = (nodes[non_port[i], 0] + nodes[non_port[i+1], 0]) / 2
            mid_lon = (nodes[non_port[i], 1] + nodes[non_port[i+1], 1]) / 2
            ax.scatter(mid_lon, mid_lat, color=color, s=30, zorder=4, alpha=0.7)


def _draw_overflows(ax, result, nodes):
    """Draw overflow markers and detour lines.

    Preemptive returns (threshold-triggered) are shown in orange with a
    triangle marker; actual overflows are shown in red with an X marker.
    """
    drew_overflow = False
    drew_preemptive = False

    for event in result["overflow_events"]:
        stn_node = event["station_node"]
        stn_lat = nodes[stn_node, 0]
        stn_lon = nodes[stn_node, 1]
        port_node = event["nearest_port"]
        port_lat = nodes[port_node, 0]
        port_lon = nodes[port_node, 1]

        is_preemptive = event.get("preemptive", False)

        if is_preemptive:
            color = "#e040fb"
            edge_color = "#aa00ff"
            marker = "^"
            label_prefix = "preemptive"
            legend_label = "Preemptive return" if not drew_preemptive else None
            drew_preemptive = True
        else:
            color = "red"
            edge_color = "darkred"
            marker = "X"
            label_prefix = "overflow"
            legend_label = "Overflow return" if not drew_overflow else None
            drew_overflow = True

        ax.scatter(stn_lon, stn_lat, color=color, marker=marker, s=200,
                   zorder=10, edgecolors=edge_color, linewidths=1,
                   label=legend_label)

        ax.plot([stn_lon, port_lon], [stn_lat, port_lat],
                color=color, linewidth=2, linestyle="--", alpha=0.7, zorder=9)

        ax.annotate(
            f"{label_prefix}: {event['cumulative_catch']:.0f}/{event['capacity']:.0f} kg",
            xy=(stn_lon, stn_lat), xytext=(10, 10),
            textcoords="offset points", fontsize=8,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
        )


def plot_realisation(solution_trips, instance, result, save_path=None):
    """Side-by-side map: planned route (left) vs simulated route (right).

    Left panel shows the clean planned route with no overflows.
    Right panel shows the same route with overflow markers and detour lines.

    Parameters
    ----------
    solution_trips : list[dict]
        The trip dicts (with 'nodes', 'boat_id').
    instance : ProblemInstance
        Problem instance for port/station info.
    result : dict
        Output from evaluate_single_realisation (needs 'overflow_events').
    save_path : str, optional
        If given, save figure to this path instead of showing.
    """
    nodes, n_ports, n_stations = _load_nodes()
    island = _load_island()

    fig, (ax_plan, ax_sim) = plt.subplots(1, 2, figsize=(28, 10))

    # --- Left: planned route ---
    _draw_base_map(ax_plan, nodes, n_ports, island, instance)
    _draw_routes(ax_plan, solution_trips, nodes)
    ax_plan.set_xlabel("Longitude")
    ax_plan.set_ylabel("Latitude")
    ax_plan.set_title(f"Planned Route — total time: {result['original_total_time']:.1f}")
    ax_plan.legend(loc="lower right", fontsize=8)

    # --- Right: simulated route with overflows ---
    _draw_base_map(ax_sim, nodes, n_ports, island, instance)
    _draw_routes(ax_sim, solution_trips, nodes)
    _draw_overflows(ax_sim, result, nodes)
    ax_sim.set_xlabel("Longitude")
    ax_sim.set_ylabel("Latitude")

    strategy_label = result.get("strategy", "backtrack")
    title = (f"Simulated ({strategy_label}) — "
             f"{result['n_unscheduled_returns']} overflow(s), "
             f"actual time: {result['total_time']:.1f} "
             f"(+{result['time_penalty']:.1f})")
    ax_sim.set_title(title)
    ax_sim.legend(loc="lower right", fontsize=8)

    # Match axis limits so the two maps are directly comparable
    xlim = (min(ax_plan.get_xlim()[0], ax_sim.get_xlim()[0]),
            max(ax_plan.get_xlim()[1], ax_sim.get_xlim()[1]))
    ylim = (min(ax_plan.get_ylim()[0], ax_sim.get_ylim()[0]),
            max(ax_plan.get_ylim()[1], ax_sim.get_ylim()[1]))
    ax_plan.set_xlim(xlim)
    ax_plan.set_ylim(ylim)
    ax_sim.set_xlim(xlim)
    ax_sim.set_ylim(ylim)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def print_catch_table(solution_trips, catch_vector, instance):
    """Print a table of simulated catch per station in visit order.

    Labels each station as (boat, trip, letter) where the letter indicates
    visit order within that trip (a=first, b=second, ...).

    Parameters
    ----------
    solution_trips : list[dict]
        The trip dicts (with 'nodes', 'boat_id').
    catch_vector : np.ndarray, shape (581,)
        Simulated catch per station ordinal.
    instance : ProblemInstance
        Problem instance (for capacities).
    """
    unique_bids = sorted(set(t["boat_id"] for t in solution_trips))
    bid_to_idx = {bid: i for i, bid in enumerate(unique_bids)}
    trip_counter = {}

    print(f"\n{'Label':<10} {'Station':<10} {'Catch (kg)':<12} {'Cumulative':<12} {'Capacity':<10}")
    print("-" * 54)

    for trip in solution_trips:
        boat_id = trip["boat_id"]
        trip_idx = trip_counter.get(boat_id, 0)
        trip_counter[boat_id] = trip_idx + 1
        capacity = instance.capacities[bid_to_idx[boat_id]]

        station_ords = _extract_station_ordinals(trip["nodes"])
        cumulative = 0.0

        for i, stn in enumerate(station_ords):
            letter = chr(ord('a') + i) if i < 26 else str(i)
            label = f"{boat_id},{trip_idx},{letter}"
            catch = catch_vector[stn]
            cumulative += catch
            flag = " ***" if cumulative > capacity else ""
            print(f"{label:<10} {stn:<10} {catch:<12.1f} {cumulative:<12.1f} {capacity:<10.0f}{flag}")

        print()


def plot_time_comparison(solution_trips, result, save_path=None):
    """Bar chart comparing planned vs actual time per trip.

    Each trip gets a pair of bars: blue (planned) and red (actual with detours).
    The red bar extends beyond the blue when there are overflow detours.

    Parameters
    ----------
    solution_trips : list[dict]
        The trip dicts.
    result : dict
        Output from evaluate_single_realisation.
    save_path : str, optional
        If given, save figure to this path instead of showing.
    """
    details = result["trip_details"]
    n_trips = len(details)

    labels = [f"Boat {d['boat_id']}\nTrip {i}" for i, d in enumerate(details)]
    original_times = [d["original_time"] for d in details]
    adjusted_times = [d["adjusted_time"] for d in details]
    detour_times = [d["detour_time"] for d in details]

    x = np.arange(n_trips)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # --- Left: per-trip bar chart ---
    bars1 = ax1.bar(x - width/2, original_times, width, label="Planned",
                    color="#4C72B0", alpha=0.85)
    bars2 = ax1.bar(x + width/2, adjusted_times, width, label="Actual (with detours)",
                    color="#DD8452", alpha=0.85)

    # Label detour time on bars that have it
    for i, dt in enumerate(detour_times):
        if dt > 0:
            ax1.annotate(f"+{dt:.1f}",
                         xy=(x[i] + width/2, adjusted_times[i]),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", fontsize=8, color="red", fontweight="bold")

    ax1.set_xlabel("Trip")
    ax1.set_ylabel("Time")
    ax1.set_title("Per-Trip: Planned vs Actual Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.legend()

    # --- Right: total summary ---
    totals = [result["original_total_time"], result["total_time"]]
    bar_labels = ["Planned\nTotal", "Actual\nTotal"]
    colors = ["#4C72B0", "#DD8452"]
    bars = ax2.bar([0, 1], totals, width=0.5, color=colors, alpha=0.85)

    # Label the penalty
    penalty = result["time_penalty"]
    if penalty > 0:
        ax2.annotate(f"+{penalty:.1f}\n({penalty/result['original_total_time']*100:.1f}% increase)",
                     xy=(1, totals[1]), xytext=(0, 10),
                     textcoords="offset points", ha="center",
                     fontsize=10, color="red", fontweight="bold")

    ax2.set_ylabel("Time")
    ax2.set_title("Total Survey Time")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(bar_labels)

    # Add value labels on bars
    for bar, val in zip(bars, totals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                 f"{val:.1f}", ha="center", va="center",
                 fontsize=12, fontweight="bold", color="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)
