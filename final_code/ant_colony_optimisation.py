"""Ant Colony Optimization helpers for the Ground-Fish-P4P project.

This module implements constructive ACO-style solution builders and a
pheromone-driven optimization driver used to create initial and improved
solutions for the fishing-boat routing problem. It also includes some
visualisation helpers for pheromone intensities.
"""

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
import copy
import numpy as np

def construct_solution2(Prob, pheromone, alpha, beta, work_limit, seed=None):
    # use pheremones + travel times to get probabilities for closest 10 stations
    if seed is not None:
        random.seed(seed)
    
    while len(Prob.stations_left) > 0:
        # print(len(Prob.stations_left))
        boat = Prob.boats[random.randint(0, Prob.n_boats-1)]
        current = boat.current_node
        time_list = Prob.time[current, Prob.stations_left]
        num_to_choose = min(len(Prob.stations_left), 10)
        smallest_dist = np.argpartition(time_list, num_to_choose-1)[:num_to_choose]
        valid_stations = list(np.array(Prob.stations_left.copy())[smallest_dist])
        
        weightings = []
        for j in valid_stations:
            tau = pheromone[current][j] ** alpha
            eta = (1.0 / Prob.time[current][j]) ** beta
            weightings.append(tau * eta)

        probabilities = [p / sum(weightings) for p in weightings]

        if sum(probabilities) == 0:
            boat.return_to_closest_port()
            continue

        station_selected = False
        station_choices = list(np.random.choice(valid_stations, size=len(valid_stations), replace=False, p=probabilities))
        # keep choosing stations until a valid one is found
        while not station_selected and len(station_choices) > 0:
            station = station_choices[0]
            # also check work limit isn't violated
            if boat.check_node_valid(Prob, station) and (Prob.n_boats == 1 or len(boat.stations)/2 < work_limit):
                station_selected = True
            else:
                station_choices = station_choices[1:]
        
        # if didn't choose a station, return to port
        if len(station_choices) == 0:
            boat.return_to_closest_port(Prob)
            continue
        
        boat.add_to_route(Prob, station)

    for boat in Prob.boats:
        boat.return_to_closest_port(Prob)
        if boat.current_node != boat.home_port:
            boat.return_to_home_port(Prob)
        boat.route = boat.route[:-1]
    
    for boat in Prob.boats:
        boat.improve_route(Prob)

        for trip in boat.route:
            # if start/end ports are the same and if reversing trip can decrease fish time, do it (time from start port to first shorter than time from last to end port)
            if trip.start_port == trip.end_port and len(trip.stations) > 0:
                if Prob.time[trip.start_port, trip.stations[0]] < Prob.time[trip.stations[-1], trip.end_port]:
                    trip.stations = [trip.stations[len(trip.stations)-i-1] for i in range(len(trip.stations))]
                    trip.evaluate_fish_time()
    
    return sum([boat.total_time for boat in Prob.boats])

def ant_colony_optimization2(Prob, k_catch, k_fish, upper_bound, w, initial_iterations=10, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, rho=0.5, Q=100, seed=None):
    if seed is not None:
        random.seed(seed)
    
    pheromone = np.ones((1175, 1175))
    # generate initial solutions and use to initialise pheremone matrix
    for i in range(initial_iterations):
        # print(i)
        t0 = time.time()
        Prob.reset()
        if Prob.n_boats == 1:
            Prob.GRASP(rcl_size=1, seed=initial_iterations*seed+i)
        else:
            Prob.generate_initial_solution(seed=initial_iterations*seed+i)
        for boat in Prob.boats:
            boat.improve_route(Prob)
        obj, pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
        
        cost = obj if pen < 1e-6 else obj + pen + upper_bound
        print(cost)
        pheromone *= (1 - rho)
        # for boat in Prob.boats:
        #     for trip in boat.route:
        #         full_trip = trip.get_full_trip()
        #         for i in range(len(full_trip) - 1):
        #             a, b = full_trip[i], full_trip[i+1]
        #             pheromone[a][b] += rho * Q / cost
        
        update = rho * Q / cost
        for boat in Prob.boats:
            for trip in boat.route:
                full_trip = np.array(trip.get_full_trip())
                a = full_trip[:-1]
                b = full_trip[1:]
                pheromone[a, b] += update
        
    # plot_pheremones(Prob, pheromone)
    # plt.show()
    # return
    Prob.reset()
    best_solution = None
    best_cost = np.inf

    cost_list = []
    # print(np.max(pheromone))
    # plot_pheremones(Prob, pheromone)
    # plt.show()

    # move ants along edges to create solutions
    for i in range(num_iterations):
        solution_list = []
        for i in range(num_ants):
            cost = construct_solution2(Prob, pheromone, alpha, beta, w, seed=i)
            # print(cost)
            # plot_pheremones(Prob, pheromone)
            # plt.show()
            cost_list.append(cost)
            solution = Prob.save_solution_as_list()
            solution_list.append(solution)
            valid = True
            for boat in Prob.boats:
                if len(boat.stations)/2 < w:
                    valid = False
                    print('rejected')
                    break

            if (cost < best_cost and valid) or best_solution is None:
                best_solution = solution
                best_cost = cost
            Prob.reset()

        pheromone *= (1 - rho)

        # for s in range(len(solution_list)):
        #     sol = solution_list[s]
        #     for boat in sol:
        #         for trip in boat:
        #             for i in range(0, len(trip) - 1, 2):
        #                 a, b = trip[i], trip[i+1]
        #                 pheromone[a][b] += Q / cost_list[s]

        # update pheremone matrix based on ants' solutions
        for s, sol in enumerate(solution_list):
            update = Q / cost_list[s]
            edges = []

            for boat in sol:
                for trip in boat:
                    if len(trip) > 2:
                        trip = np.array(trip)
                        # Get every second edge
                        a = trip[::2]
                        b = trip[1::2]
                        edges.append(np.column_stack((a, b)))

            if edges:  # avoid concatenating empty list
                edges = np.vstack(edges)
                pheromone[edges[:, 0], edges[:, 1]] += update
    # plot_pheremones(Prob, pheromone)
    # print(np.max(pheromone))
    Prob.restore_solution_from_list(best_solution)
    best_obj, best_pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)

    return best_obj, best_pen, best_cost

def plot_pheremones(Prob, pheromone):
    # plot the pheremone matrix with arc thickness proportional to pheremone level
    island_data = load_island_data()
    nodes, total_ports, total_stations = load_nodes()
    fig,ax = plt.subplots(figsize=(20,15)) # figsize=(15,13)
    ax.plot(*degArr2point(island_data), color="k", linewidth=0.5)
    # plot ports
    for idx in Prob.ports:
        ax.scatter(*degArr2point(nodes[[idx]]), s=50)

    for i in range(0, len(Prob.stations), 2):
        station = nodes[[Prob.stations[i], Prob.stations[i]+1], :]
        ax.plot(*degArr2point(station), 'k')
        ax.annotate(int(math.floor((Prob.stations[i]-13)/2)), degArr2point(station[[0], :]).flatten())

    problem_nodes = list(Prob.ports) + list(Prob.stations)
    for i in range(len(problem_nodes)-1):
        for j in range(i+1, len(problem_nodes)):
            node_i = problem_nodes[i]
            node_j = problem_nodes[j]
            ax.plot(*degArr2point(nodes[[node_i,node_j]]),  'r-', linewidth=pheromone[node_i, node_j]/np.max(pheromone))
    # ax.set_xlim(0, 200)
    # ax.set_ylim(50, 200)
    ax.invert_yaxis()
    return ax

if __name__ == "__main__":
    problem_id = 5
    n_boats = 2
    time_limit = 80
    boat_capacities = [37000, 38000]
    home_ports = [12, 12]

    # problem_id = -1
    # n_boats = 2
    # time_limit = 120
    # boat_capacities = [15000, 45000, 200000, 200000]
    # home_ports = [0, 0, 0, 0]
    if problem_id == -1:
        port_idx = np.arange(13)
    
        stations_ids = np.arange(581)
    else:
        port_idx, stations_ids, _ = load_test_problem(problem_id)
    stations_idx = np.stack([stations_ids * 2, stations_ids * 2 + 1]).T.reshape(-1) + 13

    # penalty / objective parameters (problem-specific tuning)
    k_catch = 10
    k_fish = 10
    upper_bound = 10
    work_limit = len(stations_idx)//2+1

    # instantiate the Problem object (stations, ports, limits, boats)
    Prob = Problem(list(stations_idx), port_idx, time_limit, n_boats, boat_capacities, home_ports)

    iterations = 1
    best_cost = np.inf
    all_costs = []
    for iter in range(iterations):
        print(f'starting iteration {iter}')
        aco_sol, aco_cost, cost_list = ant_colony_optimization2(Prob, k_catch, k_fish, upper_bound, work_limit, initial_iterations=10, num_ants=10, num_iterations=10, alpha=2.0, beta=1.0, rho=0.75, Q=2000, seed=iter)

        all_costs += cost_list
        if aco_cost < best_cost:
            best_cost = aco_cost
            best_sol = aco_sol

    # plot collected cost samples across all constructed ants
    plt.scatter(list(range(len(all_costs))), all_costs)
    plt.show()
    port_idx, stations_ids, _ = load_test_problem(problem_id)
    stations_idx = np.stack([stations_ids * 2, stations_ids * 2 + 1]).T.reshape(-1) + 13

    k_catch = 10
    k_fish = 10
    upper_bound = 10
    work_limit = len(stations_idx)//2+1

    Prob = Problem(list(stations_idx), port_idx, time_limit, n_boats, boat_capacities, home_ports)

    iterations = 1
    best_cost = np.inf
    all_costs = []
    for iter in range(iterations):
        print(f'starting iteration {iter}')
        aco_sol, aco_cost, cost_list = ant_colony_optimization2(Prob, k_catch, k_fish, upper_bound, work_limit, initial_iterations=10, num_ants=10, num_iterations=10, alpha=2.0, beta=1.0, rho=0.75, Q=2000, seed=iter)

        all_costs += cost_list
        if aco_cost < best_cost:
            best_cost = aco_cost
            best_sol = aco_sol

    # print(best_cost)
    # Prob.restore_solution_from_list(best_sol)
    # print(Prob.evaluate_solution())
    # # Prob.plot_all_routes()
    plt.scatter(list(range(len(all_costs))), all_costs)
    plt.show()

