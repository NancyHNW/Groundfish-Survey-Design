"""Stochastic evaluation of routing solutions under catch uncertainty."""

import os
import time
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
                                preemptive_threshold=0.8, planned_catch=None,
                                max_repairs=10, repair_scope="trip",
                                repair_planner="nn",
                                repair_solver_time=0.0, record_routes=False):
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
        - "backtrack": go to nearest port, then return and tow the station
          that overflowed, which was not fished on the first pass. (default)
        - "forward": go to nearest port, then continue to the nearest
          unvisited station from that port (no backtracking).
        - "preemptive": return to nearest port when load reaches
          preemptive_threshold fraction of capacity, before overflow.
        - "repair": re-plan the rest of the overflowed trip from the port
          the boat diverted to.
    preemptive_threshold : float
        Fraction of capacity that triggers a preemptive return (0.0–1.0).
        Only used when strategy="preemptive". Default 0.8.
    planned_catch : np.ndarray, optional
        Per-station catch the solver planned against, from solve(). Repair
        re-plans with it. Falls back to the trips' own average, which makes
        the re-planner cruder than the planner it is repairing.
    max_repairs : int
        Cap on re-plans per trip. Repair is re-simulated under the same
        scenario so it can overflow again; past the cap the rest is sailed
        charging backtrack detours. Default 10, against ~2 expected.
    repair_scope : str
        "trip" re-plans only the trip that overflowed; "boat" re-plans
        everything that boat has left, across all its remaining trips.
        Only used when strategy="repair".
    repair_planner : str
        What repair re-plans with, worst to best:
        - "nn" (default): nearest-neighbour re-split, ~0.06 ms
        - "2opt": nearest neighbour then 2-opt over each new trip
        - "solver": a one-boat sub-problem solved with the same greedy
          construction and TSP re-order the planner uses, ~1000x the cost
        The re-planner must stay worse than the planner that built the original
        routes, or repair wins by re-optimising rather than by reacting.
    repair_solver_time : float
        Seconds of restarts per repair when repair_planner="solver". 0 runs a
        single pass.
    record_routes : bool
        Collect the routes repair actually sailed, into 'repaired_routes'.
        Off by default -- a Monte Carlo has no use for them and would keep
        hundreds of thousands. Ignored by the detour strategies, which never
        change a route.

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
        n_repairs, repair_cap_hit : repair only
        repaired_routes : list[dict] — only when record_routes=True
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r}. "
                         f"Choose from {sorted(STRATEGIES)}.")
    if repair_planner not in REPAIR_PLANNERS:
        raise ValueError(f"Unknown repair_planner: {repair_planner!r}. "
                         f"Choose from {list(REPAIR_PLANNERS)}.")
    # Unchecked, a typo here reads as trip scope and looks like a result
    if repair_scope not in REPAIR_SCOPES:
        raise ValueError(f"Unknown repair_scope: {repair_scope!r}. "
                         f"Choose from {list(REPAIR_SCOPES)}.")

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
    stats = {"n_repairs": 0, "repair_cap_hit": False}
    repaired_routes = []

    # Repair needs a per-station belief. Without one from the solver it can
    # only average the trips, which packs heavy stations together because on
    # average they look fine.
    if planned_catch is None:
        n_stations = sum(len([n for n in t["nodes"] if n >= N_PORTS]) // 2
                         for t in solution_trips)
        mean_catch = (sum(t["catch"] for t in solution_trips) / n_stations
                      if n_stations else 0.0)
        planned_catch = np.full(581, mean_catch)

    # Boat scope spans several planned trips, so it runs as a pre-pass grouped
    # by boat. Everything else stays on the per-trip path below untouched.
    boat_deltas = None
    if strategy == "repair" and repair_scope == "boat":
        capacities = [instance.capacities[boat_id_to_cap_idx[t["boat_id"]]]
                      for t in solution_trips]
        boat_deltas = _repair_by_boat(
            solution_trips, capacities, catch_vector, port_nodes, time_matrix,
            planned_catch, instance.fish_time_limit, max_repairs, stats,
            overflow_events, repaired_routes if record_routes else None,
            planner=repair_planner, solver_time=repair_solver_time)

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

        if boat_deltas is not None:
            detour_time = boat_deltas[trip_idx]
        else:
            record = [] if record_routes else None
            detour_time = _run_strategy(
                strategy, preemptive_threshold,
                station_ordinals, station_nodes, catch_vector, boat_capacity,
                port_nodes, time_matrix, trip_idx, boat_id, overflow_events,
                trip=trip, planned_catch=planned_catch,
                fish_time_limit=instance.fish_time_limit,
                max_repairs=max_repairs, stats=stats, record=record,
                planner=repair_planner,
                solver_time=repair_solver_time,
            )
            if record:
                repaired_routes.append({"boat_id": boat_id,
                                        "trip_idx": trip_idx,
                                        "nodes": record})

        trip_unscheduled = sum(
            1 for e in overflow_events if e["trip_idx"] == trip_idx
        )
        # From the events, not the sign of the time change: repair re-routes
        # and can come in under the plan, so a negative change is still an
        # overflow that happened.
        trip_exceeded = trip_unscheduled > 0

        # The planned fish time, even under repair, which replaced the route.
        # The re-split enforces the limit itself, so this reports the plan.
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
        "n_repairs": stats["n_repairs"],
        "repair_cap_hit": stats["repair_cap_hit"],
        "repaired_routes": repaired_routes,
    }


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _walk_backtrack(station_ordinals, station_nodes, catch_vector,
                    boat_capacity, port_nodes, time_matrix,
                    trip_idx, boat_id, overflow_events):
    """Original strategy: nearest port, then backtrack to tow the station.

    The station that triggers the overflow is **not** fished on the first pass:
    its catch would not fit. The boat lands what it is carrying, returns, tows
    that station, and carries on. So the out-and-back is real at every station
    including a trip's last one, where the boat must still come back for it.

    That makes the hold non-empty leaving port. It unloads what it had, sails
    back, and tows the station -- so it resumes carrying that station's catch,
    not nothing. Getting this wrong hides later overflows by giving the boat a
    free station's worth of headroom.

    The catch is **not** re-drawn on the return. It is a property of the
    station in this scenario, every strategy has to see the same draw for the
    comparison to mean anything, and re-drawing would only ever happen after a
    high draw -- so it would flatter this strategy through regression to the
    mean.

    Contrast `_walk_forward`, which does not go back: there the catch *is*
    aboard, which is why only that one needs a trip-boundary correction.
    """
    cumulative_catch = 0.0
    detour_time = 0.0

    for i, stn in enumerate(station_ordinals):
        cumulative_catch += catch_vector[stn]
        if cumulative_catch > boat_capacity:
            stn_node = station_nodes[i]
            nearest_port, return_time = _find_nearest_port(
                stn_node, port_nodes, time_matrix
            )
            detour = 2 * return_time  # to port, then back to tow the station
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
            # Empty at the port, then tow this station -- so the hold carries
            # its catch onward. A station that fills a hold on its own is
            # landed instead, or the boat would return to it forever.
            station_catch = catch_vector[stn]
            cumulative_catch = (station_catch
                                if station_catch <= boat_capacity else 0.0)

    return detour_time


def _walk_forward(station_ordinals, station_nodes, catch_vector,
                  boat_capacity, port_nodes, time_matrix,
                  trip_idx, boat_id, overflow_events, end_port):
    """Forward strategy: nearest port, then carry on to the next station
    instead of going back.

    Detour cost = time(overflow_station -> port) + time(port -> next_station)
                  - time(overflow_station -> next_station)
    i.e. the extra time compared to going directly to the next station.

    Not going back is what fixes what an overflow means *here*: the catch is
    already aboard. If it were not, this strategy would sail past that station
    and never fish it. `_walk_backtrack` takes the other reading, goes back,
    and tows it.

    Because the catch is aboard, an overflow on a trip's **last** station costs
    almost nothing -- the next thing in the route is the port the boat was
    already heading for, so only which port differs. That case used to charge a
    full out-and-back, copied from backtrack, for a journey never made.
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
                # Last station of the trip: the boat was sailing to its end
                # port next, so only the change of port costs anything.
                detour = (time_to_port
                          + time_matrix[nearest_port, end_port]
                          - time_matrix[stn_node, end_port])

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


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def _leg_time(nodes, time_matrix):
    """Travel time along a node sequence."""
    return sum(time_matrix[a, b] for a, b in zip(nodes, nodes[1:]))


def _two_opt(route, time_matrix, fish_time_limit, max_passes=3):
    """Re-order one trip's stations by 2-opt. Returns a node list.

    The time matrix is exactly symmetric (checked: max asymmetry 0.0), so
    reversing a segment leaves every link inside it unchanged and re-prices
    only the two at its ends. Each move is therefore O(1), which is what makes
    this affordable inside a Monte Carlo where nearest neighbour alone is not
    the bottleneck.

    Reversing also flips which end of each tow the boat enters, which is what
    the (b, a) swap below accounts for. The tow itself costs the same either
    way, again by symmetry.

    j starts at i, so a single pair on its own can be reversed. That is not a
    re-ordering but an entry-side flip, and it is worth real time here.
    """
    start, end = route[0], route[-1]
    pairs = [(route[i], route[i + 1]) for i in range(1, len(route) - 1, 2)]
    if not pairs:
        return route

    improved, passes = True, 0
    while improved and passes < max_passes:
        improved, passes = False, passes + 1
        for i in range(len(pairs)):
            for j in range(i, len(pairs)):
                prev_exit = start if i == 0 else pairs[i - 1][1]
                next_entry = end if j == len(pairs) - 1 else pairs[j + 1][0]
                before = (time_matrix[prev_exit, pairs[i][0]]
                          + time_matrix[pairs[j][1], next_entry])
                after = (time_matrix[prev_exit, pairs[j][1]]
                         + time_matrix[pairs[i][0], next_entry])
                if after < before - 1e-12:
                    pairs[i:j + 1] = [(b, a) for a, b in
                                      reversed(pairs[i:j + 1])]
                    improved = True

    out = [start]
    for a, b in pairs:
        out.extend((int(a), int(b)))
    out.append(end)

    # 2-opt minimises the whole route, but fish time is measured from the first
    # station only, so a shorter route can still push it over. Keep the
    # un-optimised order rather than break the constraint.
    if _leg_time(out[1:], time_matrix) > fish_time_limit:
        return route
    return out


def _nn_resplit(start_port, stations, planned_catch, capacity, end_port,
                time_matrix, fish_time_limit, two_opt=False):
    """Re-plan `stations` from `start_port` as nearest-neighbour trips.

    Takes the nearest unvisited station while the planned catch fits and the
    fish time stays under the limit, then closes back to port. The first trip
    leaves the port the boat diverted to and the rest leave end_port, which is
    where the planned trip was going to dock -- trips do not always start and
    end at the same port, so the next planned trip's departure has to be met.

    No solver, because ~240,000 re-plans a comparison at 100ms each is 6.7
    hours. Returns a list of {"nodes", "stations"}.
    """
    stations = np.asarray(list(stations), dtype=int)
    if stations.size == 0:
        return []

    pairs = np.stack([N_PORTS + 2 * stations, N_PORTS + 2 * stations + 1],
                     axis=1)
    alive = np.ones(stations.size, dtype=bool)

    trips = []
    depart = start_port
    while alive.any():
        route = [depart]
        taken = []
        load = 0.0

        while alive.any():
            costs = np.where(alive[:, None], time_matrix[route[-1], pairs],
                             np.inf)
            k, side = np.unravel_index(np.argmin(costs), costs.shape)
            entry, exit_ = pairs[k, side], pairs[k, 1 - side]
            stn = int(stations[k])

            # The first station of a trip is always accepted. Refusing it when
            # one station alone breaks a limit would never close the trip.
            if taken:
                if load + planned_catch[stn] > capacity:
                    break
                trial = route + [int(entry), int(exit_), end_port]
                if _leg_time(trial[1:], time_matrix) > fish_time_limit:
                    break

            route.extend((int(entry), int(exit_)))
            taken.append(stn)
            load += planned_catch[stn]
            alive[k] = False

        route.append(end_port)
        if two_opt:
            route = _two_opt(route, time_matrix, fish_time_limit)
        trips.append({"nodes": route, "stations": taken})
        depart = end_port

    return trips


def _sub_feasible(prob, capacity, fish_time_limit):
    """Does every trip in a sub-problem respect capacity and fish time?

    GRASP is penalty-driven and will happily return a solution that breaks
    both, so a restart has to be checked before it can beat the greedy one.
    """
    return all(trip.total_catch <= capacity + 1e-6
               and trip.fish_time <= fish_time_limit + 1e-6
               for boat in prob.boats for trip in boat.route)


def _solver_resplit(start_port, stations, planned_catch, capacity, end_port,
                    port_nodes, fish_time_limit, solver_time=0.0, seed=0):
    """Re-plan with the heuristic that built the original routes.

    A one-boat sub-problem over the remaining stations, based at the port the
    boat diverted to, solved with the same greedy construction and Gurobi TSP
    re-order the planner uses. Restarts until solver_time is spent, keeping the
    best.

    Measured ~0.2 s for 150 stations against ~0.06 ms for _nn_resplit, so a few
    thousand times the cost. For single-solve comparisons, not a 30-instance
    paired run.

    The heuristic Problem gives a boat one home port, so the sub-problem docks
    where it started and the final leg is re-pointed to end_port afterwards.

    Uses the time matrix classes.py loads, not a caller-supplied one, so a
    synthetic matrix passed to the evaluator will not reach this planner.
    """
    stations = [int(s) for s in stations]
    if not stations:
        return []

    from .adapters import heuristic_context, override_catch_data
    from .evaluate import solution_to_trips

    nodes = [n for s in stations
             for n in (N_PORTS + 2 * s, N_PORTS + 2 * s + 1)]

    with heuristic_context():
        from classes import Problem

        prob = Problem(stations=nodes, ports=[int(p) for p in port_nodes],
                       fish_time_limit=float(fish_time_limit), n_boats=1,
                       boat_capacities=[float(capacity)],
                       home_ports=[int(start_port)], capacity_buffer=1.0)
        override_catch_data(prob,
                            catch_array=np.asarray(planned_catch, dtype=float))

        # Greedy first: deterministic, and it always respects the limits, so
        # there is always a feasible answer to fall back on.
        prob.generate_initial_solution(seed=seed)
        for boat in prob.boats:
            boat.improve_route(prob)
        best_obj = sum(boat.total_time for boat in prob.boats)
        best_sol = prob.save_solution_as_list()

        # Restarts use GRASP, not the greedy: generate_initial_solution ignores
        # its seed, so restarting it would recompute the same answer. GRASP is
        # randomised but may break capacity or fish time, hence the check.
        t0, it = time.time(), 0
        while time.time() - t0 < solver_time:
            it += 1
            prob.reset()
            prob.GRASP(rcl_size=2, seed=seed + it)
            for boat in prob.boats:
                boat.improve_route(prob)
            if not _sub_feasible(prob, capacity, fish_time_limit):
                continue
            obj = sum(boat.total_time for boat in prob.boats)
            if obj < best_obj:
                best_obj, best_sol = obj, prob.save_solution_as_list()

        prob.restore_solution_from_list(best_sol)
        for boat in prob.boats:
            boat.improve_route(prob)
        raw = solution_to_trips(prob)

    trips = []
    for trip in raw:
        stns = _extract_station_ordinals(trip["nodes"])
        if stns:                          # skip empty padding trips
            trips.append({"nodes": [int(n) for n in trip["nodes"]],
                          "stations": stns})
    if trips:
        trips[-1]["nodes"][-1] = int(end_port)
    return trips


# What repair re-plans with. One knob, not a planner plus a refinement flag --
# they are three points on one axis of "how good is the re-planner", and the
# combinations that a separate flag allows are meaningless.
REPAIR_PLANNERS = ("nn", "2opt", "solver")

# How much repair re-plans. The CLIs read their choices from this.
REPAIR_SCOPES = ("trip", "boat")


def _replan(planner, start_port, stations, planned_catch, capacity, end_port,
            port_nodes, time_matrix, fish_time_limit, solver_time):
    """Pick a re-planner. One place, so every caller stays the same."""
    if planner == "solver":
        return _solver_resplit(start_port, stations, planned_catch, capacity,
                               end_port, port_nodes, fish_time_limit,
                               solver_time)
    return _nn_resplit(start_port, stations, planned_catch, capacity, end_port,
                       time_matrix, fish_time_limit, two_opt=planner == "2opt")


def _sail(trips, catch_vector, capacity, port_nodes, time_matrix, record=None):
    """Sail planned trips until the hold overflows.

    Returns (time, port, stations still unvisited, event). `event` is None when
    every trip completed, in which case port is where the boat finished.

    Pass a list as `record` to collect the node sequences actually sailed. Off
    by default: the Monte Carlo runs this hundreds of thousands of times and
    has no use for the routes.
    """
    accrued = 0.0
    for t, trip in enumerate(trips):
        nodes, load = trip["nodes"], 0.0
        for i, stn in enumerate(trip["stations"]):
            load += catch_vector[stn]
            if load > capacity:
                entry = nodes[2 * i + 1]
                port, leg = _find_nearest_port(entry, port_nodes, time_matrix)
                accrued += _leg_time(nodes[:2 * i + 3], time_matrix) + leg
                if record is not None:
                    record.append([int(n) for n in nodes[:2 * i + 3]]
                                  + [int(port)])
                rest = list(trip["stations"][i + 1:])
                for later in trips[t + 1:]:
                    rest.extend(later["stations"])
                return accrued, port, rest, (stn, entry, load, port, leg)
        accrued += _leg_time(nodes, time_matrix)
        if record is not None:
            record.append([int(n) for n in nodes])

    return accrued, trips[-1]["nodes"][-1], [], None


def _sail_charging_detours(trips, catch_vector, capacity, port_nodes,
                           time_matrix, trip_idx, boat_id, overflow_events,
                           record=None):
    """Sail planned trips to the end, charging backtrack detours on overflow.

    How repair finishes the season once max_repairs binds, so a pathological
    scenario still terminates rather than re-planning forever.

    These returns are logged like any other. They are still returns the boat
    made, and leaving them out would let a capped run report fewer of them
    than an uncapped one.
    """
    total = 0.0
    for trip in trips:
        total += _leg_time(trip["nodes"], time_matrix)
        if record is not None:
            record.append([int(n) for n in trip["nodes"]])
        load = 0.0
        for i, stn in enumerate(trip["stations"]):
            load += catch_vector[stn]
            if load > capacity:
                node = trip["nodes"][2 * i + 1]
                port, leg = _find_nearest_port(node, port_nodes, time_matrix)
                total += 2 * leg
                overflow_events.append({
                    "trip_idx": trip_idx, "boat_id": boat_id,
                    "station_ordinal": stn, "station_node": node,
                    "cumulative_catch": load, "capacity": capacity,
                    "nearest_port": port, "detour_time": 2 * leg,
                })
                load = 0.0
    return total


def _repair_remainder(start_port, remaining, end_port, catch_vector, capacity,
                      planned_catch, port_nodes, time_matrix, fish_time_limit,
                      max_repairs, trip_idx, boat_id, overflow_events, stats,
                      record, planner="nn", solver_time=0.0):
    """Re-plan and sail `remaining` from `start_port`, repeating on overflow.

    Shared by both scopes -- they differ only in what goes into `remaining`.
    Returns the time it took.
    """
    realised = 0.0
    port = start_port
    n_repairs = 0

    while remaining and n_repairs < max_repairs:
        n_repairs += 1
        plan = _replan(planner, port, remaining, planned_catch, capacity,
                       end_port, port_nodes, time_matrix, fish_time_limit,
                       solver_time)
        added, port, remaining, event = _sail(plan, catch_vector, capacity,
                                              port_nodes, time_matrix,
                                              record=record)
        realised += added
        if event:
            stn, node, load, port2, leg = event
            overflow_events.append({
                "trip_idx": trip_idx, "boat_id": boat_id,
                "station_ordinal": stn, "station_node": node,
                "cumulative_catch": load, "capacity": capacity,
                "nearest_port": port2, "detour_time": leg,
            })

    if remaining:
        stats["repair_cap_hit"] = True
        plan = _replan(planner, port, remaining, planned_catch, capacity,
                       end_port, port_nodes, time_matrix, fish_time_limit,
                       solver_time)
        realised += _sail_charging_detours(plan, catch_vector, capacity,
                                           port_nodes, time_matrix, trip_idx,
                                           boat_id, overflow_events,
                                           record=record)

    stats["n_repairs"] += n_repairs
    return realised


def _repair_by_boat(solution_trips, capacities, catch_vector, port_nodes,
                    time_matrix, planned_catch, fish_time_limit, max_repairs,
                    stats, overflow_events, records, planner="nn",
                    solver_time=0.0):
    """Boat-scope repair: on the first overflow, re-plan all the boat has left.

    Wider than trip scope -- it takes the rest of the overflowed trip *and*
    every later trip that boat was going to make. One repair therefore spans
    several planned trips, which is why this runs as a pre-pass grouped by boat
    rather than inside the per-trip loop.

    Returns {trip_idx: change to that trip's planned time}. The whole
    re-planned remainder is charged to the trip where the overflow happened;
    the trips it swallows are zeroed, since they no longer exist.
    """
    deltas = {i: 0.0 for i in range(len(solution_trips))}

    by_boat = {}
    for idx, trip in enumerate(solution_trips):
        by_boat.setdefault(trip["boat_id"], []).append(idx)

    for boat_id, idxs in by_boat.items():
        capacity = capacities[idxs[0]]
        # The boat must still finish where its season was planned to finish.
        end_port = solution_trips[idxs[-1]]["nodes"][-1]

        hit = None
        for pos, idx in enumerate(idxs):
            stations = _extract_station_ordinals(solution_trips[idx]["nodes"])
            load = 0.0
            for i, stn in enumerate(stations):
                load += catch_vector[stn]
                if load > capacity:
                    hit = (pos, idx, i, stn, load, stations)
                    break
            if hit:
                break

        if hit is None:
            continue                      # this boat never overflowed

        pos, idx, i, stn, load, stations = hit
        nodes = solution_trips[idx]["nodes"]
        record = [] if records is not None else None

        entry = nodes[2 * i + 1]
        port, leg = _find_nearest_port(entry, port_nodes, time_matrix)
        realised = _leg_time(nodes[:2 * i + 3], time_matrix) + leg
        overflow_events.append({
            "trip_idx": idx, "boat_id": boat_id, "station_ordinal": stn,
            "station_node": entry, "cumulative_catch": load,
            "capacity": capacity, "nearest_port": port, "detour_time": leg,
        })
        if record is not None:
            record.append([int(n) for n in nodes[:2 * i + 3]] + [int(port)])

        remaining = list(stations[i + 1:])
        for later in idxs[pos + 1:]:
            remaining.extend(
                _extract_station_ordinals(solution_trips[later]["nodes"]))

        realised += _repair_remainder(
            port, remaining, end_port, catch_vector, capacity, planned_catch,
            port_nodes, time_matrix, fish_time_limit, max_repairs, idx,
            boat_id, overflow_events, stats, record, planner,
            solver_time)

        deltas[idx] = realised - solution_trips[idx]["total_time"]
        for later in idxs[pos + 1:]:
            deltas[later] = -solution_trips[later]["total_time"]

        if records is not None and record:
            records.append({"boat_id": boat_id, "trip_idx": idx,
                            "nodes": record})

    return deltas


def _walk_repair(station_ordinals, station_nodes, catch_vector, boat_capacity,
                 port_nodes, time_matrix, trip_idx, boat_id, overflow_events,
                 nodes, planned_catch, fish_time_limit, planned_time,
                 max_repairs, stats, record=None, planner="nn",
                 solver_time=0.0):
    """Re-plan the rest of the trip from the port the boat diverted to.

    Unlike the detour strategies this replaces a route rather than adding to
    one, so it returns the change against the planned time and that change can
    be negative. Scope is the overflowed trip only: boat assignment is fixed
    and the replacement trips still dock at home, so no other trip moves.

    Pass a list as `record` to collect the node sequences actually sailed.
    """
    # Where the planned trip was going to dock. Not always the home port:
    # a boat's trips can run port-to-port, and the next planned trip departs
    # from wherever this one ended.
    end_port = nodes[-1]

    load = 0.0
    for i, stn in enumerate(station_ordinals):
        load += catch_vector[stn]
        if load > boat_capacity:
            break
    else:
        return 0.0                      # no overflow, the plan stands

    # Priced from the same node backtrack uses, so the comparison against it is
    # the re-planning rule alone and not a change of convention.
    entry = station_nodes[i]
    port, leg = _find_nearest_port(entry, port_nodes, time_matrix)
    realised = _leg_time(nodes[:2 * i + 3], time_matrix) + leg
    overflow_events.append({
        "trip_idx": trip_idx, "boat_id": boat_id, "station_ordinal": stn,
        "station_node": entry, "cumulative_catch": load,
        "capacity": boat_capacity, "nearest_port": port, "detour_time": leg,
    })
    if record is not None:
        record.append([int(n) for n in nodes[:2 * i + 3]] + [int(port)])

    realised += _repair_remainder(
        port, list(station_ordinals[i + 1:]), end_port, catch_vector,
        boat_capacity, planned_catch, port_nodes, time_matrix,
        fish_time_limit, max_repairs, trip_idx, boat_id, overflow_events,
        stats, record, planner, solver_time)

    return realised - planned_time


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

# The overflow responses a boat can take, by name. Single place that knows
# which strategies exist: evaluate_single_realisation validates against it and
# the experiment CLIs read their --strategy choices from it.
STRATEGIES = {
    "backtrack": _walk_backtrack,
    "forward": _walk_forward,
    "preemptive": _walk_preemptive,
    "repair": _walk_repair,
}

# The ones that model an overflow as a detour on an unchanged route, and so can
# never come in under the plan. Repair re-routes, so it can.
DETOUR_STRATEGIES = ("backtrack", "forward", "preemptive")


def _run_strategy(strategy, preemptive_threshold, station_ordinals,
                  station_nodes, catch_vector, boat_capacity, port_nodes,
                  time_matrix, trip_idx, boat_id, overflow_events, trip=None,
                  planned_catch=None, fish_time_limit=None, max_repairs=10,
                  stats=None, record=None, planner="nn",
                  solver_time=0.0):
    """Walk one trip under strategy, returning the change to its planned time.

    For the detour strategies that change is a detour and is never negative.
    Repair replaces the route, so it needs the trip's nodes and the catch the
    solver planned against, and it may return less than zero.
    """
    walk = STRATEGIES[strategy]
    head = (station_ordinals, station_nodes, catch_vector, boat_capacity)
    tail = (port_nodes, time_matrix, trip_idx, boat_id, overflow_events)

    if strategy == "preemptive":
        return walk(*head, preemptive_threshold, *tail)
    if strategy == "forward":
        return walk(*head, *tail, trip["nodes"][-1])
    if strategy == "repair":
        return walk(*head, *tail, trip["nodes"], planned_catch,
                    fish_time_limit, trip["total_time"], max_repairs, stats,
                    record, planner, solver_time)
    return walk(*head, *tail)


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
    scenario matrix, which allows testing without the simulator.

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

    def evaluate(self, solution_trips, instance, strategy="backtrack",
                 preemptive_threshold=0.8, planned_catch=None,
                 max_repairs=10, repair_scope="trip",
                 repair_planner="nn", repair_solver_time=0.0):
        """Run n_simulations catch realisations against the fixed route.

        Parameters
        ----------
        solution_trips : list[dict]
            Trip dicts from evaluate.py's solution_to_trips().
        instance : ProblemInstance
            Problem parameters.
        strategy : str
            Overflow strategy: "backtrack", "forward", or "preemptive".
        preemptive_threshold : float
            Threshold for preemptive strategy (0.0-1.0).

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
                solution_trips, instance, catch_vector, time_matrix=time_matrix,
                strategy=strategy, preemptive_threshold=preemptive_threshold,
                planned_catch=planned_catch, max_repairs=max_repairs,
                repair_scope=repair_scope, repair_planner=repair_planner,
                repair_solver_time=repair_solver_time,
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
            "expected_repairs": float(np.mean([r["n_repairs"] for r in results])),
            "p_repair_cap_hit": float(np.mean([r["repair_cap_hit"]
                                               for r in results])),
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
    ax1.set_xlabel("Total survey time (hours)")
    ax1.set_ylabel("Scenarios")
    ax1.set_title("Realised survey time", fontsize=11)
    ax1.legend(fontsize=8)

    # --- Right: unscheduled returns histogram ---
    returns = [r["n_unscheduled_returns"] for r in stoch_result["all_results"]]
    max_ret = max(returns) if returns else 0
    bins = np.arange(-0.5, max_ret + 1.5, 1)
    ax2.hist(returns, bins=bins, color="#DD8452", alpha=0.8, edgecolor="white")
    ax2.axvline(stoch_result["expected_unscheduled_returns"], color="red",
                linestyle="-", linewidth=2,
                label=f"Mean: {stoch_result['expected_unscheduled_returns']:.1f}")
    ax2.set_xlabel("Unscheduled port returns per season")
    ax2.set_ylabel("Scenarios")
    ax2.set_title("Overflow returns", fontsize=11)
    ax2.legend(fontsize=8)

    # One heading over both panels carrying the run that produced them --
    # panel titles alone leave a saved figure with no record of which run.
    heading = f"Monte Carlo over {stoch_result['n_simulations']} catch scenarios"
    if title_suffix:
        heading += f"  --  {title_suffix}"
    fig.suptitle(heading, fontsize=12)

    # Both y-axes count scenarios, so fractional ticks are meaningless.
    from matplotlib.ticker import MaxNLocator
    for ax in (ax1, ax2):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25, linewidth=0.6, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# Categorical palette, fixed order, one hue per strategy. Nine of them, which
# is how many the sweep now holds -- a wrapping palette would give two
# strategies the same colour.
#
# Worst pair over all pairs and all three dichromacies (Vienot, CIE76) is
# dE 8.9, the orange/amber pair under tritanopia. The last three additions did
# not lower it. Do not reorder or substitute without re-checking.
_STRATEGY_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8b5cf6",
                     "#8c564b", "#d62728", "#2b3a55", "#005f73"]
_INK = "#0b0b0b"
_MUTED = "#898781"


def _dodge(x, n_series):
    """Horizontal offset per series, so intervals at one x sit side by side.

    Spread over a third of the gap between adjacent x values, which keeps each
    buffer's group visibly a group rather than drifting into its neighbour. A
    single series is not dodged at all.
    """
    if n_series < 2:
        return [0.0]
    gaps = [b - a for a, b in zip(x, x[1:])]
    width = (min(gaps) if gaps else 1.0) * 0.34
    step = width / (n_series - 1)
    return [-width / 2 + i * step for i in range(n_series)]


def _draw_series(ax, x, y, yerr, colour, label):
    """One series: a connecting line, and the 95% interval as the mark.

    The interval replaces the point marker rather than decorating it. Where the
    rows carry no error columns there is nothing to mark the point with, so a
    small dot comes back -- otherwise the series would be a bare line.
    """
    ax.plot(x, y, linestyle="-", color=colour, linewidth=1.6, zorder=3,
            marker="" if yerr else "o", markersize=4, label=label)
    if yerr:
        # Caps sized to stay legible when the interval is shorter than the line
        # is thick, which is the usual case at 1000 scenarios.
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=colour, elinewidth=1.6,
                    capsize=5, capthick=1.6, zorder=4)


def _errors(series, buffers, key):
    """Per-point error bar heights, or None when the rows do not carry them.

    Rows predating the Monte Carlo error columns still plot, just without bars,
    rather than raising halfway through a long sweep.
    """
    vals = [series[b].get(key) for b in buffers]
    if any(v is None for v in vals):
        return None
    return [float(v) for v in vals]


def plot_sweep_grid(rows, save_path=None, title_suffix=None):
    """Buffer against realised time, one line per overflow strategy.

    The point of the grid is the interaction: whether the best strategy changes
    as the buffer tightens. Lines crossing say it does.

    Parameters
    ----------
    rows : list of dict
        One per (buffer, strategy) cell, with keys buffer, case, planned,
        mean, returns_per_trip.
    """
    buffers = sorted({r["buffer"] for r in rows})
    cases = list(dict.fromkeys(r["case"] for r in rows))  # keeps sweep order
    x = [b * 100 for b in buffers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Planned time is the floor every strategy is measured against, so it is a
    # reference line rather than a fifth series -- hence grey and dashed.
    planned = [next(r["planned"] for r in rows if r["buffer"] == b)
               for b in buffers]
    ax1.plot(x, planned, "s--", color=_MUTED, linewidth=1.5, markersize=6,
             label="Planned (no overflow)", zorder=2)

    # The interval is the mark -- no separate dot. A point estimate drawn as a
    # dot invites reading a 2 h gap as real when the interval is 3 h wide.
    #
    # Series are dodged apart within each buffer because backtrack and forward
    # often coincide exactly -- they make the same returns, differing only in
    # what happens after -- and two intervals at identical (x, y) would hide
    # one another completely. Distinct markers used to carry that job.
    # Loud, not silent: the palette wrapping would give two strategies the
    # same colour, which reads as one series moving rather than two.
    if len(cases) > len(_STRATEGY_COLOURS):
        print(f"!! {len(cases)} strategies but {len(_STRATEGY_COLOURS)} "
              f"colours; some will repeat. Extend _STRATEGY_COLOURS.")

    offsets = _dodge(x, len(cases))
    ends = []

    for i, case in enumerate(cases):
        colour = _STRATEGY_COLOURS[i % len(_STRATEGY_COLOURS)]
        series = {r["buffer"]: r for r in rows if r["case"] == case}
        xi = [xv + offsets[i] for xv in x]

        means = [series[b]["mean"] for b in buffers]
        _draw_series(ax1, xi, means, _errors(series, buffers, "mc_ci95"),
                     colour, case)
        ends.append((means[-1], case, colour))

        rpt = [series[b]["returns_per_trip"] for b in buffers]
        _draw_series(ax2, xi, rpt,
                     _errors(series, buffers, "returns_per_trip_ci95"),
                     colour, case)

    # Direct labels as well as the legend, so identity never rests on colour
    # alone. Nudged apart where the series end too close to read.
    span = max(r["mean"] for r in rows) - min(r["mean"] for r in rows)
    gap = span * 0.045
    placed = []
    for y, case, colour in sorted(ends):
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax1.annotate(case, (x[-1] + max(offsets), y),
                     textcoords="offset points",
                     xytext=(10, 0), fontsize=8, color=_INK,
                     va="center", annotation_clip=False)

    best = min(rows, key=lambda r: r["mean"])
    ax1.set_title(f"Mean realised time  (best: {best['case']} at "
                  f"{best['buffer']:.0%}, {best['mean']:.0f}h)", fontsize=10)
    ax1.set_xlabel("Planning buffer (% of true capacity)")
    ax1.set_ylabel("Survey time (hours)")
    ax1.legend(fontsize=8, loc="best")

    ax2.set_title("Unscheduled returns per trip", fontsize=10)
    ax2.set_xlabel("Planning buffer (% of true capacity)")
    ax2.set_ylabel("Returns per trip")
    ax2.legend(fontsize=8, loc="best")

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # Room for the direct labels hanging off the right of the left panel
    ax1.margins(x=0.12)

    # What the bars cover, and what they do not. Points in one column share a
    # solve, so the strategy contrast is properly paired and the bars are the
    # whole story. Across columns each point has its own solve, and restart
    # noise there runs to tens of hours -- far wider than anything drawn.
    has_bars = any("mc_ci95" in r for r in rows)
    if has_bars:
        fig.text(0.5, 0.015,
                 "Error bars: 95% Monte Carlo CI on the mean (scenario "
                 "sampling only). Solver restart noise, which dominates "
                 "comparisons between buffers, is not shown.",
                 ha="center", fontsize=7.5, color=_MUTED)

    bottom = 0.10 if has_bars else 0
    if title_suffix:
        fig.suptitle(title_suffix, fontsize=10)
        plt.tight_layout(rect=[0, bottom, 1, 0.94])
    else:
        plt.tight_layout(rect=[0, bottom, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_buffer_comparison(rows, save_path=None, title_suffix=None):
    """Planned vs realised time across planning buffers, and what they cost.

    Parameters
    ----------
    rows : list of dict
        One per buffer, with keys: buffer, planned, mean, p5, p95,
        p_exceed, e_returns.  Ordered by buffer.
    save_path : str, optional
        If given, save figure to this path instead of showing.
    title_suffix : str, optional
        Extra text for the figure title (e.g. instance size and method).
    """
    buffers = [r["buffer"] for r in rows]
    x = [b * 100 for b in buffers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: the tradeoff in hours ---
    ax1.fill_between(x, [r["p5"] for r in rows], [r["p95"] for r in rows],
                     color="#4C72B0", alpha=0.15, label="Realised 5th-95th pct")
    ax1.plot(x, [r["mean"] for r in rows], "o-", color="#4C72B0", linewidth=2,
             label="Mean realised time")
    ax1.plot(x, [r["planned"] for r in rows], "s--", color="green", linewidth=2,
             label="Planned time")

    # the buffer that actually wins on realised time
    best = min(rows, key=lambda r: r["mean"])
    ax1.axvline(best["buffer"] * 100, color="red", linestyle=":", linewidth=1.5,
                label=f"Best: {best['buffer']:.0%} ({best['mean']:.1f}h)")

    ax1.set_xlabel("Planning buffer (% of true capacity)")
    ax1.set_ylabel("Survey time (hours)")
    ax1.set_title("Planned vs realised time")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # --- Right: what the buffer buys ---
    ax2.plot(x, [r["p_exceed"] * 100 for r in rows], "o-", color="#C44E52",
             linewidth=2, label="P(capacity exceeded)")
    ax2.set_xlabel("Planning buffer (% of true capacity)")
    ax2.set_ylabel("P(exceed) [%]", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.set_ylim(-2, 102)

    # per-trip, not absolute: a tighter buffer plans more trips, so the raw
    # count can fall just by spreading the same risk over more trips
    ax2b = ax2.twinx()
    ax2b.plot(x, [r.get("returns_per_trip",
                        r["e_returns"] / r["trips"] if r.get("trips") else 0.0)
                  for r in rows],
              "s--", color="#DD8452", linewidth=2,
              label="E[unscheduled returns] per planned trip")
    ax2b.set_ylabel("Unscheduled returns per planned trip", color="#DD8452")
    ax2b.tick_params(axis="y", labelcolor="#DD8452")

    lines = ax2.get_lines() + ax2b.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="best")
    ax2.set_title("Overflow risk")
    ax2.grid(alpha=0.3)

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
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
    # number trips within each boat, so the legend reads "Boat 2, Trip 1"
    # rather than continuing the global count
    trip_counter = {}
    for trip in solution_trips:
        trip_nodes = trip["nodes"]
        boat_id = trip["boat_id"]
        trip_idx = trip_counter.get(boat_id, 0)
        trip_counter[boat_id] = trip_idx + 1
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


def _draw_repaired_routes(ax, result, nodes):
    """Overlay the routes repair actually sailed, where they replaced the plan.

    Only the overflowed trip is re-planned, so the planned lines underneath are
    still what was sailed everywhere else. Drawn heavy and dark so the
    replacement reads as a correction to the plan rather than a fifth boat.
    """
    drew = False
    for entry in result.get("repaired_routes", []):
        for route in entry["nodes"]:
            lats = [nodes[n, 0] for n in route]
            lons = [nodes[n, 1] for n in route]
            ax.plot(lons, lats, color="#111111", linewidth=2.6, alpha=0.85,
                    zorder=8, solid_capstyle="round",
                    label="Repaired route" if not drew else None)
            drew = True
    return drew


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


def plot_solution(solution_trips, instance, save_path=None, title=None):
    """Map of a planned solution — no stochastic realisation needed.

    The deterministic counterpart to ``plot_realisation``: draws the routes a
    solver produced, without overflow markers or detour lines.

    Parameters
    ----------
    solution_trips : list[dict]
        Trip dicts from ``solution_to_trips`` (need 'nodes', 'boat_id').
    instance : ProblemInstance
    save_path : str, optional
        If given, save the figure here instead of showing it.
    title : str, optional
        Title text; defaults to a summary of trips and total time.
    """
    nodes, n_ports, n_stations = _load_nodes()
    island = _load_island()

    fig, ax = plt.subplots(figsize=(14, 10))
    _draw_base_map(ax, nodes, n_ports, island, instance)
    _draw_routes(ax, solution_trips, nodes)

    if title is None:
        total_time = sum(t["total_time"] for t in solution_trips)
        title = (f"Planned route — {instance.ns} stations, "
                 f"{instance.n_boats} boats, {len(solution_trips)} trips, "
                 f"total time {total_time:.1f} h")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


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
    drew_repair = _draw_repaired_routes(ax_sim, result, nodes)
    _draw_overflows(ax_sim, result, nodes)
    ax_sim.set_xlabel("Longitude")
    ax_sim.set_ylabel("Latitude")

    strategy_label = result.get("strategy", "backtrack")
    title = (f"Simulated ({strategy_label}) — "
             f"{result['n_unscheduled_returns']} overflow(s), "
             f"actual time: {result['total_time']:.1f} "
             f"({result['time_penalty']:+.1f})")
    if not drew_repair:
        # Without recorded routes this panel is the PLANNED route with markers
        # on it, not the route sailed. Say so rather than let it be misread.
        title += "\nroutes as planned; overflow points marked"
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
    ax1.set_ylabel("Time (hours)")
    ax1.set_title("Planned vs actual, per trip", fontsize=11)
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

    ax2.set_ylabel("Time (hours)")
    ax2.set_title("Season total", fontsize=11)
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
