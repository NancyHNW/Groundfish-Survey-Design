import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from util.distance import *
from data_loader import *
from itertools import permutations, combinations, product
from classes import Problem, Boat, Trip
from better_tsp import solve_tsp
import math
import random
import time

def tsp_move(Prob, boat_idx, trip_idx, current_obj, current_pen, k_fish, upper_bound):
    # improve trip using TSP solver and update penalty/objective values

    current_trip = Prob.boats[boat_idx].route[trip_idx]
    time_old = current_trip.total_time
    fish_old = current_trip.fish_time
    current_trip.improve_trip(Prob)
    time_new = current_trip.total_time
    fish_new = current_trip.fish_time
    current_obj += time_new - time_old

    # change in fish time penalty depends on if fish time was already exceeded before move
    if fish_old <= Prob.fish_time_limit:
        current_pen += max(0, fish_new - Prob.fish_time_limit) * k_fish
    else:
        current_pen += (max(fish_new, Prob.fish_time_limit) - fish_old) * k_fish
    current_cost = current_obj if current_pen < 1e-8 else current_obj + current_pen + upper_bound

    return current_obj, current_pen, current_cost

def trip_reversal(Prob, trip, current_obj, current_pen, k_fish, upper_bound):
    # reverse order of stations if it decreases fish time

    fish_old = trip.fish_time
    trip.stations = [trip.stations[len(trip.stations)-i-1] for i in range(len(trip.stations))]
    trip.evaluate_fish_time()
    fish_new = trip.fish_time

    # change in fish time penalty depends on if fish time was exceeded before
    if fish_old <= Prob.fish_time_limit:
        current_pen += max(0, fish_new - Prob.fish_time_limit) * k_fish
    else:
        current_pen += (max(fish_new, Prob.fish_time_limit) - fish_old) * k_fish
    current_cost = current_obj if current_pen < 1e-8 else current_obj + current_pen + upper_bound

    return current_obj, current_pen, current_cost

def find_best_insertion(Prob, full_trip, station):
    """
    Finds the best position to insert a station in a trip based on proximity,
    while keeping station pairs together (e.g., (x, x+1) or (x, x-1)).
    """
    if len(full_trip) <= 2:
        return 0

    # Find index of closest node (excluding the last node, which is typically the depot/end)
    closest_idx = np.argmin(Prob.time[full_trip[:-1], station])
    closest_station = full_trip[closest_idx]
    closest_tow = closest_station + 1 if closest_station % 2 == 1 else closest_station - 1

    # Keep station pairs together
    if full_trip[closest_idx + 1] == closest_tow:
        return closest_idx + 1
    else:
        return closest_idx

def evaluate_neighbour(Prob, old_seq1, new_seq1, old_seq2, new_seq2):
    """
    Evaluates change in fish time and total time from modifying two subpaths:
    - old_seq1 [a, x, y, b] -> new_seq1 [a, b]
    - old_seq2 [c, d]       -> new_seq2 [c, x, y, d]

    Returns:
        (delta_fish_1, delta_total_1, delta_fish_2, delta_total_2)
    """
    def seq_cost(seq, subtract=False):
        fish, total = 0, 0
        sign = -1 if subtract else 1
        for i in range(len(seq) - 1):
            t = Prob.time[seq[i], seq[i+1]]
            total += sign * t
            if seq[i] >= 13:
                fish += sign * t
        return fish, total

    # Changes: remove old, add new
    df1_old, dt1_old = seq_cost(old_seq1, subtract=True)
    df1_new, dt1_new = seq_cost(new_seq1)
    df2_old, dt2_old = seq_cost(old_seq2, subtract=True)
    df2_new, dt2_new = seq_cost(new_seq2)

    return (
        df1_old + df1_new,
        dt1_old + dt1_new,
        df2_old + df2_new,
        dt2_old + dt2_new
    )

def swap(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, st2, tow2, k_catch, k_fish, upper_bound):
    # determine the best position for insertion for an inputted swap, and calculate the change in objective and penalty
    def extract_sequence(trip, station, tow):
        # [x, a, b, y] -> [x, y] where a, b are the station & tow
        full_trip = trip.get_full_trip()
        i = full_trip.index(station) 
        j = i+1 if full_trip[i+1] == tow else i-1
        i, j = sorted([i, j])
        sequence = full_trip[i - 1:i + 3]
        changed = [sequence[0], sequence[-1]]
        full_trip.remove(station)
        full_trip.remove(tow)
        return full_trip, sequence, changed

    def evaluate_insertions(trip, station, tow):
        # check inserting station or tow first [a, b] or [b, a]
        p1 = find_best_insertion(Prob, trip, station)
        opt1 = [trip[p1], station, tow, trip[p1+1]]
        orig1 = [trip[p1], trip[p1+1]]

        p2 = find_best_insertion(Prob, trip, tow)
        opt2 = [trip[p2], tow, station, trip[p2+1]]
        orig2 = [trip[p2], trip[p2+1]]

        return (opt1, orig1, p1), (opt2, orig2, p2)

    def compute_penalty(fish_time_old, fish_time_new, limit):
        # need to calc change in penalty depending on whether limit was exceeded before swap
        if fish_time_old <= limit:
            return max(0, fish_time_new - limit) * k_fish
        return (max(fish_time_new, limit) - fish_time_old) * k_fish

    # get sequences from original trips
    full_trip1_st_rmvd, old_seq1, old_seq_chngd1 = extract_sequence(trip1, st1, tow1)
    full_trip2_st_rmvd, old_seq2, old_seq_chngd2 = extract_sequence(trip2, st2, tow2)

    # get the new sequences for each option (station or tow first)
    (s2opt1_full_seq, s2opt1_orig_seq, p_s2_1), (s2opt2_full_seq, s2opt2_orig_seq, p_s2_2) = evaluate_insertions(full_trip1_st_rmvd, st2, tow2)
    (s1opt1_full_seq, s1opt1_orig_seq, p_s1_1), (s1opt2_full_seq, s1opt2_orig_seq, p_s1_2) = evaluate_insertions(full_trip2_st_rmvd, st1, tow1)

    # calculate the new catch values
    d_fish = Prob.catch[(st1 - 13) // 2].item() - Prob.catch[(st2 - 13) // 2].item()
    new_catch1 = trip1.total_catch - d_fish
    new_catch2 = trip2.total_catch + d_fish

    # calculate the change in catch penalty
    catch_pen1 = max(0, new_catch1 - boat1.planning_capacity) * k_catch if trip1.total_catch <= boat1.planning_capacity else (max(new_catch1, boat1.planning_capacity) - trip1.total_catch) * k_catch
    catch_pen2 = max(0, new_catch2 - boat2.planning_capacity) * k_catch if trip2.total_catch <= boat2.planning_capacity else (max(new_catch2, boat2.planning_capacity) - trip2.total_catch) * k_catch

    def evaluate(s1opt, s1orig, s2opt, s2orig, node1, node2, p_s1, p_s2):
        # determine the change in objective and penalty for each option

        df1 = evaluate_neighbour(Prob, old_seq1, old_seq_chngd1, s2orig, s2opt)
        df2 = evaluate_neighbour(Prob, old_seq2, old_seq_chngd2, s1orig, s1opt)

        fish_time1 = trip1.fish_time + df1[0] + df1[2]
        fish_time2 = trip2.fish_time + df2[0] + df2[2]
        obj_new = old_obj + df1[1] + df1[3] + df2[1] + df2[3]

        pen1 = compute_penalty(trip1.fish_time, fish_time1, Prob.fish_time_limit)
        pen2 = compute_penalty(trip2.fish_time, fish_time2, Prob.fish_time_limit)

        penalty = old_pen + pen1 + pen2 + catch_pen1 + catch_pen2
        cost = obj_new if penalty < 1e-8 else obj_new + penalty + upper_bound
        return {
            "data": [p_s1, node1, p_s2, node2, boat_idx1, trip_idx1, boat_idx2, trip_idx2,
                     df1[0] + df1[2], df2[0] + df2[2], df1[1] + df1[3], df2[1] + df2[3], d_fish],
            "obj": obj_new, "pen": penalty, "cost": cost
        }

    # evaluate each option for the swap (station or tow first for each of the 2 trips)
    options = [
        evaluate(s1opt1_full_seq, s1opt1_orig_seq, s2opt1_full_seq, s2opt1_orig_seq, st1, st2, p_s1_1, p_s2_1), 
        evaluate(s1opt2_full_seq, s1opt2_orig_seq, s2opt1_full_seq, s2opt1_orig_seq, tow1, st2, p_s1_2, p_s2_1),
        evaluate(s1opt1_full_seq, s1opt1_orig_seq, s2opt2_full_seq, s2opt2_orig_seq, st1, tow2, p_s1_1, p_s2_2),
        evaluate(s1opt2_full_seq, s1opt2_orig_seq, s2opt2_full_seq, s2opt2_orig_seq, tow1, tow2, p_s1_2, p_s2_2)
    ]

    best = min(options, key=lambda o: o["cost"])
    return best["data"], best["obj"], best["pen"], best["cost"]

def random_swap(Prob, old_obj, old_pen, k_catch, k_fish, upper_bound, seed=None):
    # chooses a random pair of stations to swap (used by simulated annealing)
    if seed is not None:
        random.seed(seed)

    def select_valid_boat(prev_boat_idx = None):
        # choose a random boat to be involved in the swap
        valid_boats = [i for i, b in enumerate(Prob.boats) if len(b.stations) > 0] # and len(b.route) >= 2]
        while valid_boats:
            boat_index = random.choice(valid_boats)
            boat = Prob.boats[boat_index]
        
            if prev_boat_idx is None:
                return boat_index, boat
            
            if boat_index != prev_boat_idx:
                return boat_index, boat
            
            # if one of the boats has already been chosen, and the random choice is the same one, check that there are stations in at least 2 different trips
            if any(len(t.stations) == len(boat.stations) for t in boat.route):
                valid_boats.remove(boat_index)
                continue

            return boat_index, boat
        raise ValueError("No valid boat found for random swap.")

    def select_station_from_trip(trips, exclude_index=None):
        # choose a random station from a trip to be swapped
        indices = list(range(len(trips)))
        if exclude_index is not None:
            indices.remove(exclude_index)
        while indices:
            idx = random.choice(indices)
            if trips[idx].stations:
                station = random.choice(trips[idx].stations)
                tow = station + 1 if station % 2 == 1 else station - 1
                return idx, trips[idx], station, tow
            indices.remove(idx)
        raise ValueError("No valid station found in trips.")
    
    boat_idx1, boat1 = select_valid_boat()
    trip_idx1, trip1, st1, tow1 = select_station_from_trip(boat1.route)
    boat_idx2, boat2 = select_valid_boat(prev_boat_idx= boat_idx1)
    if boat_idx1 == boat_idx2:
        trip_idx2, trip2, st2, tow2 = select_station_from_trip(boat2.route, exclude_index=trip_idx1)
    else:
        trip_idx2, trip2, st2, tow2 = select_station_from_trip(boat2.route)
    
    return swap(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, st2, tow2, k_catch, k_fish, upper_bound)    

def perform_swap(Prob, data):
    # apply the necessary changes to the Problem object for the swap
    idx1 = data[0]
    node1 = data[1]
    node1_partner = node1+1 if node1 %2 == 1 else node1-1
    idx2 = data[2]
    node2 = data[3]
    node2_partner = node2+1 if node2 %2 == 1 else node2-1
    boat_idx1 = data[4]
    trip_idx1 = data[5]
    boat_idx2 = data[6]
    trip_idx2 = data[7]
    
    Prob.boats[boat_idx1].route[trip_idx1].stations.remove(node1)
    Prob.boats[boat_idx1].route[trip_idx1].stations.remove(node1_partner)
    Prob.boats[boat_idx2].route[trip_idx2].stations.remove(node2)
    Prob.boats[boat_idx2].route[trip_idx2].stations.remove(node2_partner)

    Prob.boats[boat_idx1].stations.remove(node1)
    Prob.boats[boat_idx1].stations.remove(node1_partner)
    Prob.boats[boat_idx2].stations.remove(node2)
    Prob.boats[boat_idx2].stations.remove(node2_partner)

    if boat_idx1 != boat_idx2:
        del Prob.boats[boat_idx1].station_dict[node1]
        del Prob.boats[boat_idx1].station_dict[node1_partner]
        del Prob.boats[boat_idx2].station_dict[node2]
        del Prob.boats[boat_idx2].station_dict[node2_partner]

    Prob.boats[boat_idx1].route[trip_idx1].stations.insert(idx2, node2)
    Prob.boats[boat_idx1].route[trip_idx1].stations.insert(idx2+1, node2_partner)
    Prob.boats[boat_idx2].route[trip_idx2].stations.insert(idx1, node1)
    Prob.boats[boat_idx2].route[trip_idx2].stations.insert(idx1+1, node1_partner)

    Prob.boats[boat_idx1].stations.append(node2)
    Prob.boats[boat_idx1].stations.append(node2_partner)
    Prob.boats[boat_idx2].stations.append(node1)
    Prob.boats[boat_idx2].stations.append(node1_partner)

    Prob.boats[boat_idx1].station_dict[node2] = trip_idx1
    Prob.boats[boat_idx1].station_dict[node2_partner] = trip_idx1
    Prob.boats[boat_idx2].station_dict[node1] = trip_idx2
    Prob.boats[boat_idx2].station_dict[node1_partner] = trip_idx2

    Prob.boat_dict[node1] = boat_idx2
    Prob.boat_dict[node1_partner] = boat_idx2
    Prob.boat_dict[node2] = boat_idx1
    Prob.boat_dict[node2_partner] = boat_idx1

    Prob.boats[boat_idx1].route[trip_idx1].fish_time += data[8]
    Prob.boats[boat_idx2].route[trip_idx2].fish_time += data[9]
    Prob.boats[boat_idx1].route[trip_idx1].total_time += data[10]
    Prob.boats[boat_idx2].route[trip_idx2].total_time += data[11]
    Prob.boats[boat_idx1].route[trip_idx1].total_catch -= data[12]
    Prob.boats[boat_idx2].route[trip_idx2].total_catch += data[12]

    Prob.boats[boat_idx1].total_time += data[10]
    Prob.boats[boat_idx2].total_time += data[11]

"""
def next_descent_swap(Prob, k_catch, k_fish, upper_bound, seed=None, test=False, tsp=None):
    # this is the old way of doing the next descent. Boat chosen, then trip in boat, then station from trip, etc.
    # better to select a random pair since you'll spend time checking every possible second station with a first even if the first wouldn't be good to swap
    if seed is not None:
        random.seed(seed)

    old_obj, old_pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
    old_cost = old_obj if old_pen < 1e-8 else old_obj + old_pen + upper_bound
    orig_cost = old_cost
    cost_list = [old_cost]
    time_list = [0]
    initial_time = time.time()
    boats_left1 = list(range(Prob.n_boats))
    counter = 0
    total_counter = 0
    skipped_counter = 0
    total_skipped_counter = 0
    stations_checked = [False] * 581
    while len(boats_left1) > 0:
        improvement = False
        boat_idx1 = random.choice(boats_left1)
        boat1 = Prob.boats[boat_idx1]
        if len(boat1.stations) == 0:
            boats_left1.remove(boat_idx1)
            continue

        trips_left1 = list(range(len(boat1.route)))
        while len(trips_left1) > 0 and not improvement:
            trip_idx1 = random.choice(trips_left1)
            trip1 = boat1.route[trip_idx1]
            if len(trip1.stations) == 0:
                trips_left1.remove(trip_idx1)
                continue

            stations_left1 = [station for station in trip1.stations.copy() if station % 2 == 1]
            while len(stations_left1) > 0 and not improvement:
                st1 = random.choice(stations_left1)
                tow1 = st1 + 1

                boats_left2 = list(range(Prob.n_boats))
                while len(boats_left2) > 0 and not improvement:
                    boat_idx2 = random.choice(boats_left2)
                    boat2 = Prob.boats[boat_idx2]
                    if len(boat2.stations) == 0 or (boat_idx1 == boat_idx2 and len(boat1.stations) == len(trip1.stations)):
                        boats_left2.remove(boat_idx2)
                        continue

                    trips_left2 = list(range(len(boat2.route)))
                    if boat_idx1 == boat_idx2:
                        trips_left2.remove(trip_idx1)
                    while len(trips_left2) > 0 and not improvement:
                        trip_idx2 = random.choice(trips_left2)
                        trip2 = boat2.route[trip_idx2]
                        if len(trip2.stations) == 0:
                            trips_left2.remove(trip_idx2)
                            continue

                        stations_left2 = [station for station in trip2.stations.copy() if station % 2 == 1]
                        while len(stations_left2) > 0 and not improvement:
                            
                            st2 = random.choice(stations_left2)
                            if stations_checked[(st2 - 13) // 2]:
                                stations_left2.remove(st2)
                                continue
                            tow2 = st2 + 1

                            if old_pen < 1e-6 and test:
                                change_in_catch = Prob.catch[int(math.floor((st1-13)/2))].item() - Prob.catch[int(math.floor((st2-13)/2))].item()
                                if trip1.total_catch - change_in_catch > boat1.planning_capacity or trip2.total_catch + change_in_catch > boat2.planning_capacity:
                                    stations_left2.remove(st2)
                                    skipped_counter += 1
                                    continue
                                trip1_singles = [station for station in trip1.stations if station % 2 == 1]
                                trip2_singles = [station for station in trip2.stations if station % 2 == 1]
                                time_to_trip1 = min(Prob.time[st2, [trip1.start_port] + trip1_singles + [trip1.end_port]])
                                time_to_trip2 = min(Prob.time[st1, [trip2.start_port] + trip2_singles + [trip2.end_port]])
                                time_perc1 = trip1.get_top_percentile_time_between_nodes(95)
                                time_perc2 = trip2.get_top_percentile_time_between_nodes(95)
                                if time_to_trip1 > time_perc1 and time_to_trip2 > time_perc2:
                                    stations_left2.remove(st2)
                                    skipped_counter += 1
                                    continue

                            data, new_obj, new_pen, new_cost = swap(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, st2, tow2, k_catch, k_fish, upper_bound)
                            counter += 1
                            cost_list.append(min(old_cost, new_cost))
                            time_list.append(time.time()-initial_time)
                            if new_cost < old_cost:
                                # print(f'swapped: {(int(st1), int(tow1))} <-> {(int(st2), int(tow2))}, objective: {old_obj} -> {new_obj}, counter: {counter} swaps checked, {skipped_counter} swaps skipped')
                                improvement = True
                                total_counter += counter
                                total_skipped_counter += skipped_counter
                                counter = 0
                                skipped_counter = 0
                                boats_left1 = list(range(Prob.n_boats))
                                stations_checked = [False] * 581
                                perform_swap(Prob, data)
                                
                                old_obj = new_obj
                                old_pen = new_pen
                                old_cost = new_cost

                                for trip in [trip1, trip2]:
                                    if trip.start_port == trip.end_port and len(trip.stations) > 0:
                                        if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                                            old_obj, old_pen, old_cost = trip_reversal(Prob, trip, old_obj, old_pen, k_fish, upper_bound)
                            else:
                                stations_left2.remove(st2)
                        if not improvement:
                            trips_left2.remove(trip_idx2)
                    if not improvement:
                        boats_left2.remove(boat_idx2)
                if not improvement:
                    stations_left1.remove(st1)
                    stations_checked[(st1 - 13) // 2] = True
            if not improvement:
                if tsp is not None:
                    if random.random() < tsp:
                        old_obj, old_pen, old_cost = tsp_move(Prob, boat_idx1, trip_idx1, old_obj, old_pen, k_fish, upper_bound)
                        old_obj, old_pen, old_cost = tsp_move(Prob, boat_idx2, trip_idx2, old_obj, old_pen, k_fish, upper_bound)
                trips_left1.remove(trip_idx1)
        if not improvement:
            boats_left1.remove(boat_idx1)
    # print(f'finished: {counter+skipped_counter} swaps checked/skipped at local min, {total_counter+total_skipped_counter} swaps checked/skipped in total, saving {orig_cost-old_cost} ({orig_cost} -> {old_cost})')

    # return time_list, cost_list
    return old_obj, old_pen, old_cost
"""

def next_descent_swap_improved(Prob, k_catch, k_fish, upper_bound, seed=None, test=False, tsp=None):
    # performs next descent by selecting a random pair of stations to swap

    if seed is not None:
        random.seed(seed)

    old_obj, old_pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
    old_cost = old_obj if old_pen < 1e-8 else old_obj + old_pen + upper_bound
    orig_cost = old_cost
    cost_list = [old_cost]
    time_list = [0]
    initial_time = time.time()
    counter = 0
    total_counter = 0
    skipped_counter = 0
    # only consider 1 station of a station pair for the swaps (since putting either first is checked in swap())
    station_singles = [station for station in Prob.stations.copy() if station % 2 == 1]
    all_swaps = set(combinations(station_singles, 2))
    possible_swaps = all_swaps.copy()
    catch_saved = 0
    time_saved = 0

    while len(possible_swaps) > 0: 
        possible_swaps_list = list(possible_swaps)
        random.shuffle(possible_swaps_list)
        for st1, st2 in possible_swaps_list:
            tow1 = st1+1
            tow2 = st2+1
            boat_idx1 = Prob.boat_dict[st1]
            boat1 = Prob.boats[boat_idx1]
            trip_idx1 = boat1.station_dict[st1]
            trip1 = boat1.route[trip_idx1]
            boat_idx2 = Prob.boat_dict[st2]
            boat2 = Prob.boats[boat_idx2]
            trip_idx2 = boat2.station_dict[st2]
            trip2 = boat2.route[trip_idx2]

            if boat_idx1 == boat_idx2 and trip_idx1 == trip_idx2:
                possible_swaps.remove((st1, st2))
                continue

            if old_pen < 1e-6 and test:
                # check if the swap is unlikely to be good without doing the calculation (testing showed it didn't really seem to help)
                change_in_catch = Prob.catch[int(math.floor((st1-13)/2))].item() - Prob.catch[int(math.floor((st2-13)/2))].item()
                if trip1.total_catch - change_in_catch > boat1.planning_capacity or trip2.total_catch + change_in_catch > boat2.planning_capacity:
                    possible_swaps.remove((st1, st2))
                    catch_saved += 1
                    skipped_counter += 1
                    continue
                trip1_singles = [station for station in trip1.stations if station % 2 == 1]
                trip2_singles = [station for station in trip2.stations if station % 2 == 1]
                time_to_trip1 = min(Prob.time[st2, [trip1.start_port] + trip1_singles + [trip1.end_port]])
                time_to_trip2 = min(Prob.time[st1, [trip2.start_port] + trip2_singles + [trip2.end_port]])
                time_perc1 = trip1.get_top_percentile_time_between_nodes(90)
                time_perc2 = trip2.get_top_percentile_time_between_nodes(90)
                # if the distance between each station to the trip they're moving too are longer than 90% of arcs in that trip then skip it
                if time_to_trip1 > time_perc1 and time_to_trip2 > time_perc2:
                    possible_swaps.remove((st1, st2))
                    time_saved += 1
                    skipped_counter += 1
                    continue

            data, new_obj, new_pen, new_cost = swap(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, st2, tow2, k_catch, k_fish, upper_bound)
            counter += 1
            cost_list.append(min(old_cost, new_cost))
            time_list.append(time.time()-initial_time)
            if new_cost < old_cost:
                # print(f'swapped: {(int(st1), int(tow1))} <-> {(int(st2), int(tow2))}, objective: {old_obj} -> {new_obj}, counter: {counter} swaps checked, {skipped_counter} swaps skipped')
                perform_swap(Prob, data)
                
                old_obj = new_obj
                old_pen = new_pen
                old_cost = new_cost

                for trip in [trip1, trip2]:
                    if trip.start_port == trip.end_port and len(trip.stations) > 0:
                        if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                            old_obj, old_pen, old_cost = trip_reversal(Prob, trip, old_obj, old_pen, k_fish, upper_bound)

                possible_swaps = all_swaps.copy()
                total_counter += counter
                counter = 0
                if tsp is not None:
                    # improve the trips using TSP solver with some probability
                    if random.random() < tsp:
                        old_obj, old_pen, old_cost = tsp_move(Prob, boat_idx1, trip_idx1, old_obj, old_pen, k_fish, upper_bound)
                        old_obj, old_pen, old_cost = tsp_move(Prob, boat_idx2, trip_idx2, old_obj, old_pen, k_fish, upper_bound)
            else:
                possible_swaps.remove((st1, st2))
                            
    # print(f'finished: {counter} swaps checked at local min, {total_counter+skipped_counter} swaps checked/skipped in total, saving {orig_cost-old_cost} ({orig_cost} -> {old_cost})')
    # if test:
    #     print(f'swaps saved from checking catch vs time: {catch_saved} vs {time_saved}')
    # return time_list, cost_list
    return old_obj, old_pen, old_cost

def steepest_descent_swap(Prob, k_catch, k_fish, upper_bound, tsp=None):
    # steepest descent by looping through all boats, then trips within each boat, then station within each trip, for 2 different stations
    # faster this way since don't need to recalculate boat/trip indices

    # Evaluate current solution once at the start
    old_obj, old_pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
    old_cost = old_obj if old_pen < 1e-8 else old_obj + old_pen + upper_bound
    orig_cost = old_cost
    cost_list = [old_cost]
    initial_time = time.time()
    time_list = [0]
    improvement = True
    best_data = None
    best_obj = None
    best_pen = None
    best_cost = orig_cost
    total_counter = 0
    while improvement:
        improvement = False
        counter = 0
        for boat_idx1 in range(Prob.n_boats):
            boat1 = Prob.boats[boat_idx1]
            if len(boat1.stations) == 0:
                continue
            for trip_idx1 in range(len(boat1.route)):
                trip1 = boat1.route[trip_idx1]
                if len(trip1.stations) == 0:
                    continue

                station1_options = [station for station in trip1.stations.copy() if station % 2 == 1]
                for st1 in station1_options:
                    tow1 = st1 + 1

                    for boat_idx2 in range(Prob.n_boats):
                        boat2 = Prob.boats[boat_idx2]
                        if len(boat2.route) == 0:
                            continue

                        trip2_options = list(range(len(boat2.route)))
                        if boat_idx1 == boat_idx2:
                            # cant swap stations that are in the same trip
                            trip2_options.remove(trip_idx1)
                        for trip_idx2 in trip2_options:
                            trip2 = boat2.route[trip_idx2]

                            station2_options = [station for station in trip2.stations.copy() if station % 2 == 1]
                            for st2 in station2_options:
                                # swapping a,b same as swapping b,a so can use symmetry to half number of pairs to check
                                if st2 < st1:
                                    continue
                                tow2 = st2 + 1

                                data, new_obj, new_pen, new_cost = swap(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, st2, tow2, k_catch, k_fish, upper_bound)
                                counter += 1
                                # cost_list.append(min(old_cost, new_cost))
                                # time_list.append(time.time()-initial_time)
                                if new_cost < best_cost:
                                    improvement = True
                                    best_data = data
                                    best_obj = new_obj
                                    best_pen = new_pen
                                    best_cost = new_cost
        total_counter += counter
        cost_list.append(best_cost)
        time_list.append(time.time()-initial_time)
        if improvement:
            # print(f'swap performed - objective: {old_cost} -> {best_cost}, counter: {counter} swaps checked')
            improvement = True
            perform_swap(Prob, best_data)
            
            old_obj = best_obj
            old_pen = best_pen
            old_cost = best_cost

            for trip in [Prob.boats[best_data[4]].route[best_data[5]], Prob.boats[best_data[6]].route[best_data[7]]]:
                if trip.start_port == trip.end_port and len(trip.stations) > 0:
                    if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                        old_obj, old_pen, old_cost = trip_reversal(Prob, trip, old_obj, old_pen, k_fish, upper_bound)

            counter = 0
            if tsp is not None:
                if random.random() < tsp:
                    old_obj, old_pen, old_cost = tsp_move(Prob, best_data[4], best_data[5], old_obj, old_pen, k_fish, upper_bound)
                    old_obj, old_pen, old_cost = tsp_move(Prob, best_data[6], best_data[7], old_obj, old_pen, k_fish, upper_bound)
        
    # print(f'finished: {total_counter} swaps checked in total, saving {orig_cost-old_cost} ({orig_cost} -> {best_cost})')
    
    return old_obj, old_pen, old_cost

def move(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, k_catch, k_fish, upper_bound):
    # evaluates moving a station to a new trip (change in objective and penalty)

    def compute_penalty(fish_time_old, fish_time_new, limit):
        # calculate change in fish time penalty depending on if it was violated before the swap
        if fish_time_old <= limit:
            return max(0, fish_time_new - limit) * k_fish
        return (max(fish_time_new, limit) - fish_time_old) * k_fish

    full_trip1 = trip1.get_full_trip()
    i = full_trip1.index(st1) 
    j = i+1 if full_trip1[i+1] == tow1 else i-1
    i, j = sorted([i, j])
    old_seq1 = full_trip1[i - 1:i + 3]
    old_seq_chngd1 = [old_seq1[0], old_seq1[-1]]
    full_trip2 = trip2.get_full_trip()

    # get the indices for eac option (station or tow visited first)
    p_s1_1 = find_best_insertion(Prob, full_trip2, st1)
    s1opt1_full_seq = [full_trip2[p_s1_1], st1, tow1, full_trip2[p_s1_1+1]]
    s1opt1_orig_seq = [full_trip2[p_s1_1], full_trip2[p_s1_1+1]]

    p_s1_2 = find_best_insertion(Prob, full_trip2, tow1)
    s1opt2_full_seq = [full_trip2[p_s1_2], tow1, st1, full_trip2[p_s1_2+1]]
    s1opt2_orig_seq = [full_trip2[p_s1_2], full_trip2[p_s1_2+1]]

    d_fish = Prob.catch[(st1 - 13) // 2].item()
    new_catch1 = trip1.total_catch - d_fish
    new_catch2 = trip2.total_catch + d_fish

    catch_pen1 = max(0, new_catch1 - boat1.planning_capacity) * k_catch if trip1.total_catch <= boat1.planning_capacity else (max(new_catch1, boat1.planning_capacity) - trip1.total_catch) * k_catch
    catch_pen2 = max(0, new_catch2 - boat2.planning_capacity) * k_catch if trip2.total_catch <= boat2.planning_capacity else (max(new_catch2, boat2.planning_capacity) - trip2.total_catch) * k_catch
    
    def evaluate(s1opt, s1orig, node1, p_s1):
        df = evaluate_neighbour(Prob, old_seq1, old_seq_chngd1, s1orig, s1opt)

        fish_time1 = trip1.fish_time + df[0]
        fish_time2 = trip2.fish_time + df[2]
        obj_new = old_obj + df[1] + df[3]

        pen1 = compute_penalty(trip1.fish_time, fish_time1, Prob.fish_time_limit)
        pen2 = compute_penalty(trip2.fish_time, fish_time2, Prob.fish_time_limit)

        penalty = old_pen + pen1 + pen2 + catch_pen1 + catch_pen2
        cost = obj_new if penalty < 1e-8 else obj_new + penalty + upper_bound
        return {
            "data": [p_s1, node1, boat_idx1, trip_idx1, boat_idx2, trip_idx2, df[0], df[2], df[1], df[3], d_fish],
            "obj": obj_new, "pen": penalty, "cost": cost
        }

    options = [
        evaluate(s1opt1_full_seq, s1opt1_orig_seq, st1, p_s1_1), 
        evaluate(s1opt2_full_seq, s1opt2_orig_seq, tow1, p_s1_2)
    ]
    
    best = min(options, key=lambda o: o["cost"])
    return best["data"], best["obj"], best["pen"], best["cost"]

def random_move(Prob, old_obj, old_pen, k_catch, k_fish, upper_bound, seed=None):
    # chooses a random station to move and the trip to which to move it (used by simulated annealing)
    if seed is not None:
        random.seed(seed)

    def select_valid_boat(prev_boat_idx = None):
        valid_boats = [i for i, b in enumerate(Prob.boats) if len(b.stations) > 0] # and len(b.route) >= 2]
        while valid_boats:
            boat_index = random.choice(valid_boats)
            boat = Prob.boats[boat_index]
            if prev_boat_idx is None:
                return boat_index, boat
            
            if boat_index != prev_boat_idx:
                return boat_index, boat
            
            # if random choice is the same as the previously chosen boat, make sure there are at least 2 trips for a move to happen
            if len(boat.route) < 2:
                valid_boats.remove(boat_index)
                continue

            return boat_index, boat
        raise ValueError("No valid boat found for random swap.")
    
    boat_idx1, boat1 = select_valid_boat()
    trip_indices1 = list(range(len(boat1.route)))
    while trip_indices1:
        trip_idx1 = random.choice(trip_indices1)
        trip1 = boat1.route[trip_idx1]
        if trip1.stations:
            st1 = random.choice(boat1.route[trip_idx1].stations)
            tow1 = st1 + 1 if st1 % 2 == 1 else st1 - 1
            break
        trip_indices1.remove(trip_idx1)
        
    boat_idx2, boat2 = select_valid_boat(prev_boat_idx=boat_idx1)
    trip_indices2 = list(range(len(boat2.route)))
    if boat_idx1 == boat_idx2:
        trip_indices2.remove(trip_idx1)
    trip_idx2 = random.choice(trip_indices2)
    trip2 = boat2.route[trip_idx2]

    return move(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, k_catch, k_fish, upper_bound)    

def perform_move(Prob, data):
    # apply a previously-evaluated move to the Problem object

    idx1 = data[0]
    node1 = data[1]
    node1_partner = node1+1 if node1 %2 == 1 else node1-1
    boat_idx1 = data[2]
    trip_idx1 = data[3]
    boat_idx2 = data[4]
    trip_idx2 = data[5]
    
    Prob.boats[boat_idx1].route[trip_idx1].stations.remove(node1)
    Prob.boats[boat_idx1].route[trip_idx1].stations.remove(node1_partner)

    Prob.boats[boat_idx1].stations.remove(node1)
    Prob.boats[boat_idx1].stations.remove(node1_partner)

    if boat_idx1 != boat_idx2:
        del Prob.boats[boat_idx1].station_dict[node1]
        del Prob.boats[boat_idx1].station_dict[node1_partner]

    Prob.boats[boat_idx2].route[trip_idx2].stations.insert(idx1, node1)
    Prob.boats[boat_idx2].route[trip_idx2].stations.insert(idx1+1, node1_partner)

    Prob.boats[boat_idx2].stations.append(node1)
    Prob.boats[boat_idx2].stations.append(node1_partner)

    Prob.boats[boat_idx2].station_dict[node1] = trip_idx2
    Prob.boats[boat_idx2].station_dict[node1_partner] = trip_idx2

    Prob.boat_dict[node1] = boat_idx2
    Prob.boat_dict[node1_partner] = boat_idx2

    Prob.boats[boat_idx1].route[trip_idx1].fish_time += data[6]
    Prob.boats[boat_idx2].route[trip_idx2].fish_time += data[7]
    Prob.boats[boat_idx1].route[trip_idx1].total_time += data[8]
    Prob.boats[boat_idx2].route[trip_idx2].total_time += data[9]
    Prob.boats[boat_idx1].route[trip_idx1].total_catch -= data[10]
    Prob.boats[boat_idx2].route[trip_idx2].total_catch += data[10]

    Prob.boats[boat_idx1].total_time += data[8]
    Prob.boats[boat_idx2].total_time += data[9]

def next_descent_move(Prob, k_catch, k_fish, upper_bound, work_limit, seed=None, tsp=None, update_tree_big=None):
    # performs next descent by looping through boats, then trips in that boat, then stations in that trip
    if seed is not None:
        random.seed(seed)

    old_obj, old_pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
    old_cost = old_obj if old_pen < 1e-8 else old_obj + old_pen + upper_bound
    orig_cost = old_cost
    cost_list = [old_cost]
    boats_left1 = list(range(Prob.n_boats))
    update_plot = 0 # for GUI
    counter = 0
    prev_counter = 0
    initial_time = time.time()
    time_list = [0]
    while len(boats_left1) > 0:
        improvement = False
        boat_idx1 = random.choice(boats_left1)
        boat1 = Prob.boats[boat_idx1]
        if len(boat1.stations) == 0:
            boats_left1.remove(boat_idx1)
            continue

        trips_left1 = list(range(len(boat1.route)))
        while len(trips_left1) > 0 and not improvement:
            trip_idx1 = random.choice(trips_left1)
            trip1 = boat1.route[trip_idx1]
            if len(trip1.stations) == 0:
                trips_left1.remove(trip_idx1)
                continue

            stations_left1 = [station for station in trip1.stations.copy() if station % 2 == 1]
            while len(stations_left1) > 0 and not improvement:
                st1 = random.choice(stations_left1)
                tow1 = st1 + 1

                boats_left2 = list(range(Prob.n_boats))
                while len(boats_left2) > 0 and not improvement:
                    boat_idx2 = random.choice(boats_left2)
                    boat2 = Prob.boats[boat_idx2]
                    if len(boat2.route) == 0:
                        boats_left2.remove(boat_idx2)
                        continue

                    trips_left2 = list(range(len(boat2.route)))
                    if boat_idx1 == boat_idx2:
                        trips_left2.remove(trip_idx1)
                    while len(trips_left2) > 0 and not improvement:
                        trip_idx2 = random.choice(trips_left2)
                        trip2 = boat2.route[trip_idx2]

                        data, new_obj, new_pen, new_cost = move(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, k_catch, k_fish, upper_bound)
                        counter += 1
                        cost_list.append(min(old_cost, new_cost))
                        time_list.append(time.time()-initial_time)
                        if new_cost < old_cost and (len(boat2.stations)/2 < work_limit or boat_idx1 == boat_idx2): # if same boat, not changing total number of stations visited so work_limit not affected
                            # print(f'moved: {(int(st1), int(tow1))} from boat {boat_idx1+1} trip {trip_idx1+1} -> boat {boat_idx2+1} trip {trip_idx2+1}, objective: {old_cost} -> {new_cost}, counter: {counter-prev_counter} moves checked')
                            improvement = True
                            prev_counter = counter
                            boats_left1 = list(range(Prob.n_boats))
                            perform_move(Prob, data)
                            # for GUI
                            if __name__ == '__main__':
                                pass
                            else: 
                                update_plot += 1
                                if update_plot % 20 == 0:
                                    update_tree_big()
                            # Prob.plot_all_routes(ax)
                            # plt.pause(2)
                            old_obj = new_obj
                            old_pen = new_pen
                            old_cost = new_cost

                            for trip in [trip1, trip2]:
                                if trip.start_port == trip.end_port and len(trip.stations) > 0:
                                    if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                                        old_obj, old_pen, old_cost = trip_reversal(Prob, trip, old_obj, old_pen, k_fish, upper_bound)

                            if tsp is not None:
                                if random.random() < tsp:
                                    old_obj, old_pen, old_cost = tsp_move(Prob, boat_idx1, trip_idx1, old_obj, old_pen, k_fish, upper_bound)
                                    old_obj, old_pen, old_cost = tsp_move(Prob, boat_idx2, trip_idx2, old_obj, old_pen, k_fish, upper_bound)
                        else:
                            trips_left2.remove(trip_idx2)
                    if not improvement:
                        boats_left2.remove(boat_idx2)
                if not improvement:
                    stations_left1.remove(st1)
            if not improvement:
                # if tsp is not None:
                #     if random.random() < tsp:
                #         old_obj, old_pen, old_cost = tsp_move(Prob, boat_idx1, trip_idx1, old_obj, old_pen, k_fish, upper_bound)
                trips_left1.remove(trip_idx1)
        if not improvement:
            boats_left1.remove(boat_idx1)
    # print(f'finished: {counter - prev_counter} moves checked at local min, {counter} moves checked in total, saving {orig_cost-old_cost} ({orig_cost} -> {old_cost})')
    # return time_list, cost_list
    return old_obj, old_pen, old_cost

def steepest_descent_combined(Prob, k_catch, k_fish, upper_bound, work_limit, tsp=None):
    # performs steepest descent on moves and then swaps if a boat is near capacitiy or the fish time limit

    old_obj, old_pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
    old_cost = old_obj if old_pen < 1e-8 else old_obj + old_pen + upper_bound
    orig_cost = old_cost
    cost_list = [old_cost]
    initial_time = time.time()
    time_list = [0]
    improvement = True
    best_data = None
    best_obj = None
    best_pen = None
    best_cost = orig_cost
    total_counter = 0
    while improvement:
        improvement = False
        counter = 0
        for boat_idx1 in range(Prob.n_boats):
            boat1 = Prob.boats[boat_idx1]
            if len(boat1.stations) == 0:
                continue
            for trip_idx1 in range(len(boat1.route)):
                trip1 = boat1.route[trip_idx1]
                if len(trip1.stations) == 0:
                    continue

                station1_options = [station for station in trip1.stations.copy() if station % 2 == 1]
                for st1 in station1_options:
                    tow1 = st1 + 1

                    for boat_idx2 in range(Prob.n_boats):
                        boat2 = Prob.boats[boat_idx2]
                        if len(boat2.route) == 0:
                            continue

                        trip2_options = list(range(len(boat2.route)))
                        if boat_idx1 == boat_idx2:
                            trip2_options.remove(trip_idx1)
                        for trip_idx2 in trip2_options:
                            trip2 = boat2.route[trip_idx2]

                            if trip2.total_catch / boat2.planning_capacity > 0.95 or trip2.fish_time / Prob.fish_time_limit > 0.95:
                                rule_type = 'swap'
                                station2_options = [station for station in trip2.stations.copy() if station % 2 == 1]
                                if len(trip2.stations) == 0:
                                    continue
                            else:
                                rule_type = 'move'
                                station2_options = [0]
                            did_rule = False
                            for st2 in station2_options:
                                if rule_type == 'swap':
                                    if st2 < st1:
                                        continue
                                    did_rule = True
                                    tow2 = st2 + 1
                                    data, new_obj, new_pen, new_cost = swap(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, st2, tow2, k_catch, k_fish, upper_bound)
                                    
                                else:
                                    did_rule = True
                                    data, new_obj, new_pen, new_cost = move(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, k_catch, k_fish, upper_bound)
                            counter += 1
                            # cost_list.append(min(old_cost, new_cost))
                            # time_list.append(time.time()-initial_time)
                            if did_rule:
                                if new_cost < old_cost and (rule_type == 'swap' or len(boat2.stations)/2 < work_limit or boat_idx1 == boat_idx2):
                                    improvement = True
                                    best_data = data
                                    best_obj = new_obj
                                    best_pen = new_pen
                                    best_cost = new_cost
                                    best_rule = rule_type
        total_counter += counter
        cost_list.append(best_cost)
        time_list.append(time.time()-initial_time)
        if improvement:
            # print(f'move performed - objective: {old_cost} -> {best_cost}, counter: {counter} moves checked')
            improvement = True
            if best_rule == 'swap':
                # print('swap')
                # print(best_data)
                perform_swap(Prob, best_data)
            else:
                perform_move(Prob, best_data)
            
            old_obj = best_obj
            old_pen = best_pen
            old_cost = best_cost

            if best_rule == 'move':
                trips = Prob.boats[best_data[2]].route[best_data[3]], Prob.boats[best_data[4]].route[best_data[5]]
            else:
                trips = [Prob.boats[best_data[4]].route[best_data[5]], Prob.boats[best_data[6]].route[best_data[7]]]
            for trip in trips:
                if trip.start_port == trip.end_port and len(trip.stations) > 0:
                    if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                        old_obj, old_pen, old_cost = trip_reversal(Prob, trip, old_obj, old_pen, k_fish, upper_bound)

            counter = 0
            if tsp is not None :
                if random.random() < tsp:
                    if best_rule == 'move':
                        old_obj, old_pen, old_cost = tsp_move(Prob, best_data[2], best_data[3], old_obj, old_pen, k_fish, upper_bound)
                        old_obj, old_pen, old_cost = tsp_move(Prob, best_data[4], best_data[5], old_obj, old_pen, k_fish, upper_bound)
                    else:
                        old_obj, old_pen, old_cost = tsp_move(Prob, best_data[4], best_data[5], old_obj, old_pen, k_fish, upper_bound)
                        old_obj, old_pen, old_cost = tsp_move(Prob, best_data[6], best_data[7], old_obj, old_pen, k_fish, upper_bound)
        
                    # print(f'trip {trip_idx1} improved with tsp - from {obj1} -> {current_trip.total_time}') 
    
    # print(f'finished: {total_counter} moves checked in total, saving {orig_cost-old_cost} ({orig_cost} -> {best_cost})')
    
    # return time_list, cost_list
    return old_obj, old_pen, old_cost

def steepest_descent_move(Prob, k_catch, k_fish, upper_bound, work_limit, tsp=None, update_tree_big=None):
    # performs steepest descent for moves, looping through boats, then trips then stations

    old_obj, old_pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
    old_cost = old_obj if old_pen < 1e-8 else old_obj + old_pen + upper_bound
    orig_cost = old_cost
    cost_list = [old_cost]
    initial_time = time.time()
    time_list = [0]
    improvement = True
    best_data = None
    best_obj = None
    best_pen = None
    best_cost = orig_cost
    total_counter = 0
    update_plot = 0
    while improvement:
        improvement = False
        counter = 0
        for boat_idx1 in range(Prob.n_boats):
            boat1 = Prob.boats[boat_idx1]
            if len(boat1.stations) == 0:
                continue
            for trip_idx1 in range(len(boat1.route)):
                trip1 = boat1.route[trip_idx1]
                if len(trip1.stations) == 0:
                    continue

                station1_options = [station for station in trip1.stations.copy() if station % 2 == 1]
                for st1 in station1_options:
                    tow1 = st1 + 1

                    for boat_idx2 in range(Prob.n_boats):
                        boat2 = Prob.boats[boat_idx2]
                        if len(boat2.route) == 0:
                            continue

                        trip2_options = list(range(len(boat2.route)))
                        if boat_idx1 == boat_idx2:
                            trip2_options.remove(trip_idx1)
                        for trip_idx2 in trip2_options:
                            trip2 = boat2.route[trip_idx2]

                            data, new_obj, new_pen, new_cost = move(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, k_catch, k_fish, upper_bound)
                            counter += 1
                            # cost_list.append(min(old_cost, new_cost))
                            # time_list.append(time.time()-initial_time)
                            if new_cost < old_cost and (len(boat2.stations)/2 < work_limit or boat_idx1 == boat_idx2):
                                improvement = True
                                best_data = data
                                best_obj = new_obj
                                best_pen = new_pen
                                best_cost = new_cost
        total_counter += counter
        cost_list.append(best_cost)
        time_list.append(time.time()-initial_time)
        if improvement:
            update_plot += 1
            # print(f'move performed - objective: {old_cost} -> {best_cost}, counter: {counter} moves checked')
            improvement = True
            perform_move(Prob, best_data)
            # for GUI
            if __name__ == '__main__':
                pass
            else: 
                if update_plot % 20 == 0:
                    update_tree_big()
                else:
                    pass
            old_obj = best_obj
            old_pen = best_pen
            old_cost = best_cost

            for trip in [Prob.boats[best_data[2]].route[best_data[3]], Prob.boats[best_data[4]].route[best_data[5]]]:
                if trip.start_port == trip.end_port and len(trip.stations) > 0:
                    if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                        old_obj, old_pen, old_cost = trip_reversal(Prob, trip, old_obj, old_pen, k_fish, upper_bound)

            counter = 0
            if tsp is not None :
                if random.random() < tsp:
                    old_obj, old_pen, old_cost = tsp_move(Prob, best_data[2], best_data[3], old_obj, old_pen, k_fish, upper_bound)
                    old_obj, old_pen, old_cost = tsp_move(Prob, best_data[4], best_data[5], old_obj, old_pen, k_fish, upper_bound)
                    # print(f'trip {trip_idx1} improved with tsp - from {obj1} -> {current_trip.total_time}') 
    
    # print(f'finished: {total_counter} moves checked in total, saving {orig_cost-old_cost} ({orig_cost} -> {best_cost})')
    
    # return time_list, cost_list
    return old_obj, old_pen, old_cost

def next_descent_combined(Prob, k_catch, k_fish, upper_bound, work_limit, seed=None, tsp=None, update_tree_big=None):
    """Randomized next-descent combining move and swap rules.
    Swaps are only considered when a boat is near capacity or near the fish time limit
    """
    if seed is not None:
        random.seed(seed)

    old_obj, old_pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
    old_cost = old_obj if old_pen < 1e-8 else old_obj + old_pen + upper_bound
    orig_cost = old_cost
    cost_list = [old_cost]
    boats_left1 = list(range(Prob.n_boats))
    counter = 0
    total_counter = 0
    initial_time = time.time()
    time_list = [0]
    update_plot = 0
    while len(boats_left1) > 0:
        improvement = False
        boat_idx1 = random.choice(boats_left1)
        boat1 = Prob.boats[boat_idx1]
        if len(boat1.stations) == 0:
            boats_left1.remove(boat_idx1)
            continue

        trips_left1 = list(range(len(boat1.route)))
        while len(trips_left1) > 0 and not improvement:
            trip_idx1 = random.choice(trips_left1)
            trip1 = boat1.route[trip_idx1]
            if len(trip1.stations) == 0:
                trips_left1.remove(trip_idx1)
                continue

            stations_left1 = [station for station in trip1.stations.copy() if station % 2 == 1]
            while len(stations_left1) > 0 and not improvement:
                st1 = random.choice(stations_left1)
                tow1 = st1 + 1

                boats_left2 = list(range(Prob.n_boats))
                while len(boats_left2) > 0 and not improvement:
                    boat_idx2 = random.choice(boats_left2)
                    boat2 = Prob.boats[boat_idx2]
                    if len(boat2.route) == 0:
                        boats_left2.remove(boat_idx2)
                        continue

                    trips_left2 = list(range(len(boat2.route)))
                    if boat_idx1 == boat_idx2:
                        trips_left2.remove(trip_idx1)
                    while len(trips_left2) > 0 and not improvement:
                        trip_idx2 = random.choice(trips_left2)
                        trip2 = boat2.route[trip_idx2]
                        # potentially change to amt of capacity left < average catch
                        if trip2.total_catch / boat2.planning_capacity > 0.95 or trip2.fish_time / Prob.fish_time_limit > 0.95:
                            rule_type = 'swap'
                            stations_left2 = [station for station in trip2.stations.copy() if station % 2 == 1]
                            if len(trip2.stations) == 0:
                                trips_left2.remove(trip_idx2)
                                continue
                        else:
                            rule_type = 'move'
                            stations_left2 = [0]

                        while len(stations_left2) > 0 and not improvement:
                            if rule_type == 'swap':
                                st2 = random.choice(stations_left2)
                                tow2 = st2 + 1

                                data, new_obj, new_pen, new_cost = swap(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, st2, tow2, k_catch, k_fish, upper_bound)
                            else:
                                data, new_obj, new_pen, new_cost = move(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, k_catch, k_fish, upper_bound)
                            counter += 1
                            cost_list.append(min(old_cost, new_cost))
                            time_list.append(time.time()-initial_time)
                            if new_cost < old_cost and (rule_type == 'swap' or len(boat2.stations)/2 < work_limit or boat_idx1 == boat_idx2):
                                improvement = True
                                boats_left1 = list(range(Prob.n_boats))
                                if rule_type == 'swap':
                                    perform_swap(Prob, data)
                                    # print(f'swapped: {(int(st1), int(tow1))} <-> {(int(st2), int(tow2))}, objective: {old_obj} -> {new_obj}, counter: {counter} swaps checked')
                                    # for GUI
                                    if __name__ == '__main__':
                                        pass
                                    else: 
                                        update_plot += 1
                                        if update_plot % 20 == 0:
                                            update_tree_big()
                                else:
                                    perform_move(Prob, data)
                                    # print(f'moved: {(int(st1), int(tow1))} from boat {boat_idx1+1} trip {trip_idx1+1} -> boat {boat_idx2+1} trip {trip_idx2+1}, objective: {old_cost} -> {new_cost}, counter: {counter} moves checked')
                                    if __name__ == '__main__':
                                        pass
                                    else: 
                                        update_plot += 1
                                        if update_plot % 20 == 0:
                                            update_tree_big()
                                total_counter += counter
                                counter = 0
                                old_obj = new_obj
                                old_pen = new_pen
                                old_cost = new_cost

                                for trip in [trip1, trip2]:
                                    if trip.start_port == trip.end_port and len(trip.stations) > 0:
                                        if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                                            old_obj, old_pen, old_cost = trip_reversal(Prob, trip, old_obj, old_pen, k_fish, upper_bound)

                                if tsp is not None:
                                    if random.random() < tsp:
                                        old_obj, old_pen, old_cost = tsp_move(Prob, boat_idx1, trip_idx1, old_obj, old_pen, k_fish, upper_bound)
                                        old_obj, old_pen, old_cost = tsp_move(Prob, boat_idx2, trip_idx2, old_obj, old_pen, k_fish, upper_bound)
                            else:
                                if rule_type == 'swap':
                                    stations_left2.remove(st2)
                                else:
                                    stations_left2 = []
                        if not improvement:
                            trips_left2.remove(trip_idx2)
                    if not improvement:
                        boats_left2.remove(boat_idx2)
                if not improvement:
                    stations_left1.remove(st1)
            if not improvement:
                # if tsp is not None:
                #     if random.random() < tsp:
                #         old_obj, old_pen, old_cost = tsp_move(Prob, boat_idx1, trip_idx1, old_obj, old_pen, k_fish, upper_bound)
                trips_left1.remove(trip_idx1)
        if not improvement:
            boats_left1.remove(boat_idx1)
    # print(f'finished: {counter} neighbours checked at local min, {total_counter} neighbours checked in total, saving {orig_cost-old_cost} ({orig_cost} -> {old_cost})')
    # return time_list, cost_list
    return old_obj, old_pen, old_cost

def tabu_search_swap(Prob, k_catch, k_fish, upper_bound, work_limit, history_size, max_iter, max_time, tsp=None):
    # performs tabu search with swaps (steepest descent with banned list)
    t_start = time.time()
    old_obj, old_pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
    old_cost = old_obj if old_pen < 1e-8 else old_obj + old_pen + upper_bound
    orig_cost = old_cost
    best_sol = Prob.save_solution_as_list()
    cost_list = [old_cost]
    initial_time = time.time()
    time_list = [0]    
    overall_best_cost = old_cost
    overall_best_obj = old_obj
    overall_best_pen = old_pen
    iteration = 0
    iters_since_improvement = 0
    counter = 0
    # gives the iteration when the station was most recently banned - if thats within the history size, the swap is invalid
    when_banned = {}
    for station in Prob.stations:
        if station % 2 == 1:
            when_banned[station] = -history_size
    while iters_since_improvement < max_iter and time.time() - t_start < max_time:
        best_iter_cost = np.inf
        iters_since_improvement += 1
        iteration += 1
        for boat_idx1 in range(Prob.n_boats):
            boat1 = Prob.boats[boat_idx1]
            if len(boat1.stations) == 0:
                continue
            for trip_idx1 in range(len(boat1.route)):
                trip1 = boat1.route[trip_idx1]
                if len(trip1.stations) == 0:
                    continue

                station1_options = [station for station in trip1.stations.copy() if station % 2 == 1]
                for st1 in station1_options:
                    if when_banned[st1] >= iteration - history_size:
                        continue
                    tow1 = st1 + 1

                    for boat_idx2 in range(Prob.n_boats):
                        boat2 = Prob.boats[boat_idx2]
                        if len(boat2.route) == 0:
                            continue

                        trip2_options = list(range(len(boat2.route)))
                        if boat_idx1 == boat_idx2:
                            trip2_options.remove(trip_idx1)
                        for trip_idx2 in trip2_options:
                            trip2 = boat2.route[trip_idx2]

                            station2_options = [station for station in trip2.stations.copy() if station % 2 == 1]
                            for st2 in station2_options:
                                if when_banned[st2] >= iteration - history_size or st2 > st1:
                                    continue
                                tow2 = st2 + 1

                                counter += 1
                                data, new_obj, new_pen, new_cost = swap(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, st2, tow2, k_catch, k_fish, upper_bound)
                                # cost_list.append(min(old_cost, new_cost))
                                # time_list.append(time.time()-initial_time)
                                if new_cost < best_iter_cost:
                                    best_iter_data = data
                                    best_iter_obj = new_obj
                                    best_iter_pen = new_pen
                                    best_iter_cost = new_cost
        if best_iter_cost > 1e10:
            continue

        # print(best_iter_cost)
        
        perform_swap(Prob, best_iter_data)
        # ban both stations
        when_banned[math.ceil(best_iter_data[1]/2)*2-1] = iteration
        when_banned[math.ceil(best_iter_data[3]/2)*2-1] = iteration

        if tsp is not None:
            if random.random() < tsp:
                best_iter_obj, best_iter_pen, best_iter_cost = tsp_move(Prob, best_iter_data[4], best_iter_data[5], best_iter_obj, best_iter_pen, k_fish, upper_bound)
                best_iter_obj, best_iter_pen, best_iter_cost = tsp_move(Prob, best_iter_data[6], best_iter_data[7], best_iter_obj, best_iter_pen, k_fish, upper_bound)
                # print('tsp done')

        cost_list.append(best_iter_cost)
        time_list.append(time.time()-initial_time)
        if best_iter_cost + 1e-6 < overall_best_cost:
            iters_since_improvement = 0
            overall_best_cost = best_iter_cost
            overall_best_obj = best_iter_obj
            overall_best_pen = best_iter_pen
            best_sol = Prob.save_solution_as_list()
        
        # print(f'swap stations {best_iter_data[1]} <-> {best_iter_data[3]} after {counter} checks - objective: {old_cost} -> {best_iter_cost}, iteration: {iteration}, {iters_since_improvement} iterations since improvement')
        counter = 0
        old_obj = best_iter_obj
        old_pen = best_iter_pen
        old_cost = best_iter_cost

        
        for trip in [Prob.boats[best_iter_data[4]].route[best_iter_data[5]], Prob.boats[best_iter_data[6]].route[best_iter_data[7]]]:
            if trip.start_port == trip.end_port and len(trip.stations) > 0:
                if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                    old_obj, old_pen, old_cost = trip_reversal(Prob, trip, old_obj, old_pen, k_fish, upper_bound)
        
    Prob.restore_solution_from_list(best_sol)
    # print(f'finished: {iteration} iterations completed, saving {orig_cost-overall_best_cost} ({orig_cost} -> {overall_best_cost})')
    
    return overall_best_obj, overall_best_pen, overall_best_cost

def tabu_search_move(Prob, k_catch, k_fish, upper_bound, work_limit, history_size, max_iter, max_time, tsp=None):
    # same as tabu_search_swap but for moves

    t_start = time.time()
    old_obj, old_pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
    old_cost = old_obj if old_pen < 1e-8 else old_obj + old_pen + upper_bound
    orig_cost = old_cost
    best_sol = Prob.save_solution_as_list()
    cost_list = [old_cost]
    initial_time = time.time()
    time_list = [0]
    overall_best_cost = old_cost
    overall_best_obj = old_obj
    overall_best_pen = old_pen
    iteration = 0
    iters_since_improvement = 0
    counter = 0
    when_banned = {}
    for station in Prob.stations:
        if station % 2 == 1:
            when_banned[station] = -history_size
    while iters_since_improvement < max_iter and time.time()-t_start < max_time:
        best_iter_cost = np.inf
        iters_since_improvement += 1
        iteration += 1
        for boat_idx1 in range(Prob.n_boats):
            boat1 = Prob.boats[boat_idx1]
            if len(boat1.stations) == 0:
                continue
            for trip_idx1 in range(len(boat1.route)):
                trip1 = boat1.route[trip_idx1]
                if len(trip1.stations) == 0:
                    continue

                station1_options = [station for station in trip1.stations.copy() if station % 2 == 1]
                for st1 in station1_options:
                    if when_banned[st1] >= iteration - history_size:
                        continue
                    tow1 = st1 + 1

                    for boat_idx2 in range(Prob.n_boats):
                        boat2 = Prob.boats[boat_idx2]
                        if len(boat2.route) == 0:
                            continue

                        trip2_options = list(range(len(boat2.route)))
                        if boat_idx1 == boat_idx2:
                            trip2_options.remove(trip_idx1)
                        for trip_idx2 in trip2_options:
                            trip2 = boat2.route[trip_idx2]

                            counter += 1
                            data, new_obj, new_pen, new_cost = move(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, k_catch, k_fish, upper_bound)
                            # cost_list.append(min(old_cost, new_cost))
                            # time_list.append(time.time()-initial_time)
                                                      
                            if new_cost < best_iter_cost:
                                if (len(boat2.stations)/2 < work_limit or boat_idx1 == boat_idx2):
                                    best_iter_data = data
                                    best_iter_obj = new_obj
                                    best_iter_pen = new_pen
                                    best_iter_cost = new_cost
        if best_iter_cost > 1e10:
            continue

        perform_move(Prob, best_iter_data)
        when_banned[math.ceil(best_iter_data[1]/2)*2-1] = iteration

        if tsp is not None:
            if random.random() < tsp:
                best_iter_obj, best_iter_pen, best_iter_cost = tsp_move(Prob, best_iter_data[2], best_iter_data[3], best_iter_obj, best_iter_pen, k_fish, upper_bound)
                best_iter_obj, best_iter_pen, best_iter_cost = tsp_move(Prob, best_iter_data[4], best_iter_data[5], best_iter_obj, best_iter_pen, k_fish, upper_bound)
        
        cost_list.append(best_iter_cost)
        time_list.append(time.time()-initial_time)
        if best_iter_cost + 1e-6 < overall_best_cost:
            iters_since_improvement = 0
            overall_best_cost = best_iter_cost
            overall_best_obj = best_iter_obj
            overall_best_pen = best_iter_pen
            best_sol = Prob.save_solution_as_list()
        
        # print(f'moved station {best_iter_data[1]} after {counter} checks - objective: {old_cost} -> {best_iter_cost}, iteration: {iteration}, {iters_since_improvement} iterations since improvement')
        counter = 0

        old_obj = best_iter_obj
        old_pen = best_iter_pen
        old_cost = best_iter_cost

        for trip in [Prob.boats[best_iter_data[2]].route[best_iter_data[3]], Prob.boats[best_iter_data[4]].route[best_iter_data[5]]]:
            if trip.start_port == trip.end_port and len(trip.stations) > 0:
                if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                    old_obj, old_pen, old_cost = trip_reversal(Prob, trip, old_obj, old_pen, k_fish, upper_bound)
        
    Prob.restore_solution_from_list(best_sol)
    # print(f'finished: {iteration} iterations completed, saving {orig_cost-overall_best_cost} ({orig_cost} -> {overall_best_cost})')
    
    return overall_best_obj, overall_best_pen, overall_best_cost

def tabu_search_combined(Prob, k_catch, k_fish, upper_bound, work_limit, history_size, max_iter, max_time, tsp=None):
    # same as the other tabu searches but uses the combined rule where swaps are only done if near a penalty
    t_start = time.time()
    old_obj, old_pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
    old_cost = old_obj if old_pen < 1e-8 else old_obj + old_pen + upper_bound
    orig_cost = old_cost
    best_sol = Prob.save_solution_as_list()
    cost_list = [old_cost]
    initial_time = time.time()
    time_list = [0]
    overall_best_cost = old_cost
    overall_best_obj = old_obj
    overall_best_pen = old_pen
    iteration = 0
    iters_since_improvement = 0
    counter = 0
    when_banned = {}
    for station in Prob.stations:
        if station % 2 == 1:
            when_banned[station] = -history_size
    while iters_since_improvement < max_iter and time.time()-t_start < max_time:
        # print(time.time()-t_start < max_time)
        best_iter_cost = np.inf
        iters_since_improvement += 1
        iteration += 1
        for boat_idx1 in range(Prob.n_boats):
            boat1 = Prob.boats[boat_idx1]
            if len(boat1.stations) == 0:
                continue
            for trip_idx1 in range(len(boat1.route)):
                trip1 = boat1.route[trip_idx1]
                if len(trip1.stations) == 0:
                    continue

                station1_options = [station for station in trip1.stations.copy() if station % 2 == 1]
                for st1 in station1_options:
                    if when_banned[st1] >= iteration - history_size:
                        continue
                    tow1 = st1 + 1

                    for boat_idx2 in range(Prob.n_boats):
                        boat2 = Prob.boats[boat_idx2]
                        if len(boat2.route) == 0:
                            continue

                        trip2_options = list(range(len(boat2.route)))
                        if boat_idx1 == boat_idx2:
                            trip2_options.remove(trip_idx1)
                        for trip_idx2 in trip2_options:
                            trip2 = boat2.route[trip_idx2]

                            if trip2.total_catch / boat2.planning_capacity > 0.95 or trip2.fish_time / Prob.fish_time_limit > 0.95:
                                rule_type = 'swap'
                                station2_options = [station for station in trip2.stations.copy() if station % 2 == 1]
                                
                            else:
                                rule_type = 'move'
                                station2_options = [-1]

                            for st2 in station2_options:
                                if rule_type == 'swap':
                                    if when_banned[st2] >= iteration - history_size or st2 > st1:
                                        continue
                                    tow2 = st2 + 1
                                    data, new_obj, new_pen, new_cost = swap(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, st2, tow2, k_catch, k_fish, upper_bound)
                                else:
                                    data, new_obj, new_pen, new_cost = move(Prob, old_obj, old_pen, boat_idx1, boat1, trip_idx1, trip1, st1, tow1, boat_idx2, boat2, trip_idx2, trip2, k_catch, k_fish, upper_bound)

                                counter += 1
                                
                                # cost_list.append(min(old_cost, new_cost))
                                # time_list.append(time.time()-initial_time)
                                                        
                                if new_cost < best_iter_cost:
                                    if rule_type == 'swap' or (len(boat2.stations)/2 < work_limit or boat_idx1 == boat_idx2):
                                        best_iter_data = data
                                        best_iter_obj = new_obj
                                        best_iter_pen = new_pen
                                        best_iter_cost = new_cost
                                        best_iter_rule = rule_type

        if best_iter_cost > 1e10:
            continue
        
        if best_iter_rule == 'move':
            perform_move(Prob, best_iter_data)
            when_banned[math.ceil(best_iter_data[1]/2)*2-1] = iteration
        else:
            perform_swap(Prob, best_iter_data)
            when_banned[math.ceil(best_iter_data[1]/2)*2-1] = iteration
            when_banned[math.ceil(best_iter_data[3]/2)*2-1] = iteration

        if tsp is not None:
            if random.random() < tsp:
                if best_iter_rule == 'move':
                    best_iter_obj, best_iter_pen, best_iter_cost = tsp_move(Prob, best_iter_data[2], best_iter_data[3], best_iter_obj, best_iter_pen, k_fish, upper_bound)
                    best_iter_obj, best_iter_pen, best_iter_cost = tsp_move(Prob, best_iter_data[4], best_iter_data[5], best_iter_obj, best_iter_pen, k_fish, upper_bound)
                else:
                    best_iter_obj, best_iter_pen, best_iter_cost = tsp_move(Prob, best_iter_data[4], best_iter_data[5], best_iter_obj, best_iter_pen, k_fish, upper_bound)
                    best_iter_obj, best_iter_pen, best_iter_cost = tsp_move(Prob, best_iter_data[6], best_iter_data[7], best_iter_obj, best_iter_pen, k_fish, upper_bound)
        
        cost_list.append(best_iter_cost)
        time_list.append(time.time()-initial_time)
        if best_iter_cost + 1e-6 < overall_best_cost:
            iters_since_improvement = 0
            overall_best_cost = best_iter_cost
            overall_best_obj = best_iter_obj
            overall_best_pen = best_iter_pen
            best_sol = Prob.save_solution_as_list()
        
        # if best_iter_rule == 'move':
        #     print(f'moved station {best_iter_data[1]} after {counter} checks - objective: {old_cost} -> {best_iter_cost}, iteration: {iteration}, {iters_since_improvement} iterations since improvement')
        # else:
        #     print(f'swap stations {best_iter_data[1]} <-> {best_iter_data[3]} after {counter} checks - objective: {old_cost} -> {best_iter_cost}, iteration: {iteration}, {iters_since_improvement} iterations since improvement')

        counter = 0

        old_obj = best_iter_obj
        old_pen = best_iter_pen
        old_cost = best_iter_cost

        if best_iter_rule == 'move':
            for trip in [Prob.boats[best_iter_data[2]].route[best_iter_data[3]], Prob.boats[best_iter_data[4]].route[best_iter_data[5]]]:
                if trip.start_port == trip.end_port and len(trip.stations) > 0:
                    if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                        old_obj, old_pen, old_cost = trip_reversal(Prob, trip, old_obj, old_pen, k_fish, upper_bound)
        else:
            for trip in [Prob.boats[best_iter_data[4]].route[best_iter_data[5]], Prob.boats[best_iter_data[6]].route[best_iter_data[7]]]:
                if trip.start_port == trip.end_port and len(trip.stations) > 0:
                    if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                        old_obj, old_pen, old_cost = trip_reversal(Prob, trip, old_obj, old_pen, k_fish, upper_bound)
        
        obj, pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
        if abs(obj - old_obj) > 1e-6 or abs(pen - old_pen) > 1e-6:
            print('ERROR')

    Prob.restore_solution_from_list(best_sol)
    # print(f'finished: {iteration} iterations completed, saving {orig_cost-overall_best_cost} ({orig_cost} -> {overall_best_cost})')
    
    return overall_best_obj, overall_best_pen, overall_best_cost

def simulated_annealing(Prob, k_catch, k_fish, upper_bound, temp_init=1000, alpha=0.95, stopping_temp=1e-8, max_iter=1000, tsp=None, seed=None):
    # performs simulated annealing using random moves and swaps
    if seed is not None:
        random.seed(seed)

    current_obj, current_pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
    # best = current[:]
    current_cost = current_obj if current_pen < 1e-8 else current_obj + current_pen + upper_bound
    best_cost = current_cost
    best_obj = current_obj
    best_pen = current_pen
    best_solution = Prob.save_solution_as_list()
    temp = temp_init

    improvement_list = [current_cost]

    initial_time = time.time()
    time_list = [0]
    # move_list = [0]

    for i in range(max_iter):
        # print(current_obj)
        if temp < stopping_temp:
            break

        # Choose one of four neighbourhood moves randomly
        if random.random() < 0.5:
            rule = 'move'
            neighbour_data, neighbour_obj, neighbour_pen, neighbour_cost = random_move(Prob, current_obj, current_pen, k_catch, k_fish, upper_bound, seed=i)
        else:
            rule = 'swap'
            neighbour_data, neighbour_obj, neighbour_pen, neighbour_cost = random_swap(Prob, current_obj, current_pen, k_catch, k_fish, upper_bound, seed=i)

        # Decide whether to accept the neighbour
        delta = neighbour_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            improvement_list.append(neighbour_cost)
            time_list.append(time.time()-initial_time)
            
            current_obj = neighbour_obj
            current_pen = neighbour_pen
            current_cost = neighbour_cost
            
            if rule == 'move':
                perform_move(Prob, neighbour_data)
                for trip in [Prob.boats[neighbour_data[2]].route[neighbour_data[3]], Prob.boats[neighbour_data[4]].route[neighbour_data[5]]]:
                    if trip.start_port == trip.end_port and len(trip.stations) > 0:
                        if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                            current_obj, current_pen, current_cost = trip_reversal(Prob, trip, current_obj, current_pen, k_fish, upper_bound)

                if tsp is not None :
                    if random.random() < tsp:
                        current_obj, current_pen, current_cost = tsp_move(Prob, neighbour_data[2], neighbour_data[3], current_obj, current_pen, k_fish, upper_bound)
                        current_obj, current_pen, current_cost = tsp_move(Prob, neighbour_data[4], neighbour_data[5], current_obj, current_pen, k_fish, upper_bound)
            else:
                perform_swap(Prob, neighbour_data)
                for trip in [Prob.boats[neighbour_data[4]].route[neighbour_data[5]], Prob.boats[neighbour_data[6]].route[neighbour_data[7]]]:
                    if trip.start_port == trip.end_port and len(trip.stations) > 0:
                        if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                            current_obj, current_pen, current_cost = trip_reversal(Prob, trip, current_obj, current_pen, k_fish, upper_bound)

                if tsp is not None :
                    if random.random() < tsp:
                        current_obj, current_pen, current_cost = tsp_move(Prob, neighbour_data[4], neighbour_data[5], current_obj, current_pen, k_fish, upper_bound)
                        current_obj, current_pen, current_cost = tsp_move(Prob, neighbour_data[6], neighbour_data[7], current_obj, current_pen, k_fish, upper_bound)

            obj, pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
            if abs(current_obj - obj) > 1e-6:
                print('obj error')
            if abs(current_pen - pen) > 1e-6:
                print('pen error')
                
            if current_cost < best_cost:
                best_cost = current_cost
                best_obj = current_obj
                best_pen = current_pen
                best_solution = Prob.save_solution_as_list()

            # shift this outside the loop for cooling every iteration regardless of if move/swap was successful
            temp *= alpha

    Prob.restore_solution_from_list(best_solution)
    obj, pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
    if abs(obj - best_obj) > 1e-6:
        print('error restoring solution')
        print(obj, best_obj)

    # return best_obj
    return best_obj, best_pen, best_cost

if __name__ == "__main__":
    # problem_id = -1
    # n_boats = 4
    # time_limit = 120
    # boat_capacities = [16000, 45000, 200000, 200000]
    # home_ports = [12, 12, 12, 12]
    problem_id = 2
    n_boats = 2
    time_limit = 30
    boat_capacities = [7000, 8000]
    home_ports = [12, 12]
    if problem_id == -1:
        port_idx = np.arange(13)
        stations_ids = np.arange(581)
    else:
        port_idx, stations_ids, _ = load_test_problem(problem_id)
    
    random.seed(0)
    # stations_ids = np.array(random.sample(list(stations_ids), 250))
    stations_idx = np.stack([stations_ids * 2, stations_ids * 2 + 1]).T.reshape(-1) + 13
    
    Prob = Problem(list(stations_idx), port_idx, time_limit, n_boats, boat_capacities, home_ports)
    # for i in range(n_boats):
    #     Prob.boats.append(Boat(boat_capacities[i], home_ports[i]))
    k_catch= 10
    k_fish = 2000
    upper_bound = 200
    iterations = 10
    times1 = []
    times2 = []
    costs1 = []
    costs2 = []
    best_obj = np.inf
    obj_list = []
    t0 = time.time()
    best_obj_list = [2000]
    times_list = [0]
    for i in range(iterations):
        t_start = time.time()
        print(i)
        Prob.reset()
        Prob.generate_initial_solution(seed=i)
        # Prob.GRASP(rcl_size=4, seed=i)
        # Prob.plot_all_routes()
        # time_list1, cost_list1 = 
        # next_descent_move(Prob, seed=i) #, test=True)
        # times1, costs1 = next_descent_combined(Prob, seed=i)
        obj, pen, cost = steepest_descent_move(Prob, k_catch, k_fish, upper_bound, work_limit=len(stations_idx)//Prob.n_boats*1.1, tsp=0.05)
        costs1.append(cost)

        