import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from util.distance import *
from data_loader import *
from itertools import permutations, combinations, product
from better_tsp import solve_tsp
import math
import random
import time

# needed to add this because had weird errors when using final testing scripts (not sure why)
try:
    time_matrix = np.load("final_code/local data/true_time.npy")
except FileNotFoundError:
    time_matrix = np.load("local data/true_time.npy")

class Problem:
    def __init__(self, stations, ports, fish_time_limit, n_boats, boat_capacities=None, home_ports=None, capacity_buffer=1.0):
        self.stations = stations.copy() # should be actual nodes
        self.stations_left = stations.copy()
        self.ports = ports.copy()
        self.fish_time_limit = fish_time_limit # in hours, time until fish starts rotting
        self.n_stations = len(stations)
        self.n_ports = len(ports)
        self.n_boats = n_boats
        self.boats = []
        # fraction of the true capacity the solver plans up to - leaves headroom for
        # catches that turn out heavier than planned (1.0 = plan to full capacity)
        self.capacity_buffer = capacity_buffer
        # stores which boat id each station is in for the current solution
        self.boat_dict = {}
        if boat_capacities is not None:
            for i in range(n_boats):
                self.boats.append(Boat(i+1, boat_capacities[i], home_ports[i], capacity_buffer=capacity_buffer))
        try:
            self.time = np.load("final_code/local data/true_time.npy")
        except (FileNotFoundError, ValueError):
            try:
                self.time = np.load("local data/true_time.npy")
            except (FileNotFoundError, ValueError):
                self.time = time_matrix
        self.catch = load_catch_fixed_random_assignment().to_numpy().reshape(-1)
    
    def __str__(self):
        return f'{self.n_stations} stations\n{self.n_ports} ports\n{self.n_boats} boats\n{self.fish_time_limit} hour time limit'

    def display_solution(self):
        # prints current problem state (boats + trips + catch + travel time)
        for b in range(self.n_boats):
            boat = self.boats[b]
            print(f'Boat {b+1}:')
            if len(boat.route) == 0:
                print("No trips")
            else:
                for t in range(len(boat.route)):
                    trip = boat.route[t]
                    print(f'Trip {t+1}: {trip.get_full_trip()} with load = {trip.total_catch}, fish time = {trip.fish_time} & total time = {trip.total_time}')

    def reset(self, gui = False):
        # return problem to initial state (no solution)
        self.stations_left = self.stations.copy()
        for boat in self.boats:
            if not gui:
                boat.route = [Trip(start_port=boat.home_port)]
            boat.current_node = boat.home_port
            boat.current_load = 0 # current load of boat (useful during construction of route)
            boat.current_time = 0 # current fish time (useful during construction of route)
            boat.total_time = 0 # sum of all trips' total time
            boat.stations = []
            boat.station_dict = {}
    
    def save_solution_as_list(self):
        # takes current state and stores it as [[], [], []] for each boat
        solution = []
        for boat in self.boats:
            boat_solution = []
            for trip in boat.route:
                boat_solution.append(trip.get_full_trip())
            solution.append(boat_solution)
        return solution
    
    def restore_solution_from_list(self, solution):
        # reverses save_solution_as_list
        self.reset()
        for b in range(len(solution)):
            boat_solution = solution[b]
            boat = self.boats[b]
            boat.route = []
            for t in range(len(boat_solution)):
                full_trip = boat_solution[t]
                if len(full_trip) > 2:
                    trip = Trip(full_trip[0], full_trip[1:-1], full_trip[-1], f'trip:{boat.id}:{t+1}')
                    trip.actual_trip = full_trip
                    boat.route.append(trip)
                else:
                    trip = Trip(full_trip[0], [], full_trip[-1], f'trip:{boat.id}:{t+1}')
                    trip.actual_trip = full_trip
                    boat.route.append(trip)

                current_trip = boat.route[t]
                current_trip.evaluate_fish_time()
                current_trip.evaluate_total_time()
                current_trip.evaluate_total_catch()
                boat.total_time += current_trip.total_time
                boat.stations += current_trip.stations
                for station in current_trip.stations:
                    boat.station_dict[station] = t
                    self.boat_dict[station] = b

            boat.current_node = boat.route[-1].end_port
    
    def resolve_boat_selection(self, boat_selection):
        # "auto" only uses locality-aware selection when the boats actually start
        # from different ports - with a shared home port every boat has the same
        # key on the first pick, so the rule degenerates into "always boat 1"
        if boat_selection != "auto":
            return boat_selection
        if len({boat.home_port for boat in self.boats}) == 1:
            return "random"
        return "nearest"

    def select_boat(self, boat_selection, balance_weight):
        # picks which boat gets the next station
        if boat_selection == "random":
            return self.boats[random.randint(0, self.n_boats-1)]

        # "nearest": prefer the boat with a close station, but push a boat away
        # once it is running ahead of the least-loaded one. Both terms are in
        # hours, so balance_weight is "hours of detour a boat is worth delaying
        # per hour it is ahead"; balance_weight=0 is the pure greedy rule, which
        # starves boats and must not be used as-is.
        min_total_time = min(boat.total_time for boat in self.boats)
        return min(
            self.boats,
            key=lambda b: (np.min(self.time[b.current_node, self.stations_left])
                           + balance_weight * (b.total_time - min_total_time))
        )

    def generate_initial_solution(self, seed=None):
        # creates greedy initial solution
        if seed is not None:
            random.seed(seed)
        # t0 = time.time()
        while len(self.stations_left) > 0:
            # if time.time()-t0 > 10:
            #     pass
            boat = self.boats[random.randint(0, self.n_boats-1)]
            next_node = boat.get_closest_station(self)[0]
            if boat.check_node_valid(self, next_node):
                boat.add_to_route(self, next_node)
            else:
                boat.return_to_closest_port(self)
                prev_trip = boat.route[-2]
                # if start/end ports are the same and if reversing trip can decrease fish time, do it (time from start port to first shorter than time from last to end port)
                if prev_trip.start_port == prev_trip.end_port and len(prev_trip.stations) > 0:
                    if self.time[prev_trip.start_port, prev_trip.stations[0]] < self.time[prev_trip.stations[-1], prev_trip.end_port]:
                        prev_trip.stations = [prev_trip.stations[len(prev_trip.stations)-i-1] for i in range(len(prev_trip.stations))]
                        prev_trip.evaluate_fish_time()

        for boat in self.boats:
            boat.return_to_closest_port(self)
            if boat.current_node != boat.home_port:
                boat.return_to_home_port(self)
            boat.route = boat.route[:-1] # when adding a port to a trip it automatically creates a new trip starting at that port so we need to delete the last trip

        return

    def GRASP(self, rcl_size=1, seed=None):
        # same as greedy algo but choosing between multiple of closest stations + chance to return to port based on current load/travel time
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        while len(self.stations_left) > 0:
            boat = self.boats[random.randint(0, self.n_boats-1)]
            rcl_list = boat.get_closest_station(self, rcl_size)
            # print(rcl_list)
            current_trip = boat.route[-1]
            # if either load or time limit at 0-50%, no port chance, if at 100%, guaranteed port chance, otherwise linear interpolation between them
            state = max(current_trip.total_catch / boat.planning_capacity, current_trip.fish_time / self.fish_time_limit)
            if state < 0.5:
                port_perc = 0
            elif state < 1:
                port_perc = state-0.5
            else:
                port_perc = 0.5

            perc_list = self.time[boat.current_node, rcl_list]
            perc_list /= sum(perc_list)/(1-port_perc)
            perc_list = np.append(perc_list, port_perc)
            
            choice = np.random.choice(range(len(perc_list)), 1, p=perc_list)[0]

            if choice != len(perc_list)-1:
                boat.add_to_route(self, rcl_list[choice])
            else:
                boat.return_to_closest_port(self)
                prev_trip = boat.route[-2]
                # if reversing trip can decrease fish time, do it (time from start port to first shorter than time from last to end port)
                if prev_trip.start_port == prev_trip.end_port and len(prev_trip.stations) > 0:
                    if self.time[prev_trip.start_port, prev_trip.stations[0]] < self.time[prev_trip.stations[-1], prev_trip.end_port]:
                        prev_trip.stations = [prev_trip.stations[len(prev_trip.stations)-i-1] for i in range(len(prev_trip.stations))]
                        prev_trip.evaluate_fish_time()

        for boat in self.boats:
            boat.return_to_closest_port(self)
            if boat.current_node != boat.home_port:
                boat.return_to_home_port(self)
            boat.route = boat.route[:-1] # when adding a port to a trip it automatically creates a new trip starting at that port so we need to delete the last trip

        return

    def evaluate_solution(self, k_catch, k_fish, upper_bound, check_stations=False, use_planning_capacity=True):
        # calculates the objective function and penalty values for the current solution
        # use_planning_capacity=True penalises catch above the buffered capacity (what the
        # solver plans to); False penalises only catch above the true boat capacity
        if check_stations:
            full_solution = []
            for boat in self.boats:
                first_trip = True
                for trip in boat.route:
                    full_trip = trip.get_full_trip()
                    if first_trip:
                        full_solution += full_trip
                        first_trip = False
                    else:
                        full_solution += full_trip[1:]

            full_solution_only_stations = [node for node in full_solution if node >= 13]
            num_stations_missing = len(set(self.stations) - set(full_solution_only_stations))
            num_stations_repeated = len(full_solution_only_stations) - len(set(full_solution_only_stations))
            num_stations_extra = len(set(full_solution_only_stations) - set(self.stations))

            num_not_start_home_port = 0
            num_not_end_home_port = 0

            for boat in self.boats:
                if boat.route[0].start_port != boat.home_port:
                    num_not_start_home_port += 1
                if boat.route[-1].end_port != boat.home_port:
                    num_not_end_home_port += 1

            num_stations_not_towed = 0
            for i, station in enumerate(full_solution):
                if station < 13:
                    continue

                expected_pair = station + 1 if station % 2 == 1 else station - 1

                prev_ok = i > 0 and full_solution[i - 1] == expected_pair
                next_ok = i < len(full_solution) - 1 and full_solution[i + 1] == expected_pair

                if not (prev_ok or next_ok):
                    num_stations_not_towed += 1
        else:
            num_stations_missing = 0
            num_stations_repeated = 0
            num_stations_extra = 0
            num_not_start_home_port = 0
            num_not_end_home_port = 0
            num_stations_not_towed = 0

        amt_over_capacity = 0
        amt_over_fish_time_limit = 0
        for boat in self.boats:
            cap = boat.planning_capacity if use_planning_capacity else boat.capacity
            for trip in boat.route:
                trip.evaluate_total_catch()
                trip.evaluate_fish_time()
                amt_over_capacity += max(0, trip.total_catch - cap)
                amt_over_fish_time_limit += max(0, trip.fish_time - self.fish_time_limit)

        for boat in self.boats:
            boat.evaluate_total_time()

        # print(num_stations_repeated)
        # print(num_stations_missing)
        # print(num_stations_extra)
        # print(num_not_start_home_port)
        # print(num_not_end_home_port)
        # print(num_stations_not_towed)
        # print(amt_over_capacity)
        # print(amt_over_fish_time_limit)
            
        objective_value = sum(boat.total_time for boat in self.boats)
        penalty_value = (num_stations_missing * 10000 +
                         num_stations_repeated * 10000 + 
                         num_stations_extra * 10000 +
                         num_not_start_home_port * 1000 +
                         num_not_end_home_port * 1000 +
                         num_stations_not_towed * 10000 +
                         amt_over_capacity * k_catch +
                         amt_over_fish_time_limit * k_fish)
        
        return objective_value, penalty_value

    def write_problem_to_txt(self, filename):
        # write current solution to a text file for use with UI
        sol_list = self.save_solution_as_list()
        boat_capacities = []
        home_ports = []
        for boat in self.boats:
            boat_capacities.append(boat.capacity)
            home_ports.append(boat.home_port)
        with open(filename, 'w') as f:
            f.write(str(self.stations) + '\n')
            f.write(str(list(self.ports)) + '\n')
            f.write(str(self.fish_time_limit) + '\n')
            f.write(str(self.n_boats) + '\n')
            f.write(str(boat_capacities) + '\n')
            f.write(str(home_ports) + '\n')
            f.write(str(sol_list) + '\n')
        
    def plot_all_routes(self, annotate=True, x_lim=None, y_lim=None):
        # plots current solution
        # ax.clear()
        island_data = load_island_data()
        nodes, total_ports, total_stations = load_nodes()
        solution = []
        for boat in self.boats:
            path = []
            for trip in boat.route:
                full_trip = trip.get_full_trip()
                for i in range(len(full_trip)-1):
                    path.append((full_trip[i], full_trip[i+1]))
            solution.append(path)

        fig,ax = plt.subplots(figsize=(20,15)) # figsize=(15,13)
        ax.plot(*degArr2point(island_data), color="k", linewidth=0.5)
        # plot ports
        for idx in self.ports:
            ax.scatter(*degArr2point(nodes[[idx]]), s=50)

        # plot stations
        for i in range(0, len(self.stations), 2):
            station = nodes[[self.stations[i], self.stations[i]+1], :]
            ax.plot(*degArr2point(station), 'k')
            if annotate:
                ax.annotate(int(math.floor((self.stations[i]-13)/2)), degArr2point(station[[0], :]).flatten())
        
        #plot routes
        colours = ['r-','b-','y-','c-','m-','g-','w-']
        k=0

        for path in solution:
            k += 1
            for i, j in path:
                ax.plot(*degArr2point(nodes[[i,j]]),  colours[k % 8], linewidth=1.5)
        
        if x_lim:
            ax.set_xlim(x_lim)   
        if y_lim:
            ax.set_ylim(y_lim)
                
        # plt.gca().invert_yaxis()
        ax.invert_yaxis()

        return ax
        

class Boat:
    def __init__(self, id, capacity, home_port, trip_w_port=True, capacity_buffer=1.0):
        self.id = id
        self.capacity = capacity # true physical capacity - used when evaluating a solution
        # capacity the solver constructs/searches against; planning below the true
        # capacity leaves room for catches heavier than the planning estimate
        self.planning_capacity = capacity * capacity_buffer
        self.home_port = home_port
        if trip_w_port:
            self.route = [Trip(start_port=home_port)]
        else:
            self.route = [Trip(start_port=home_port, id = f'trip:{id}:1')]
        # stores all the stations visited by this boat
        self.stations = []
        # stores which trip in the boat each station is in
        self.station_dict = {}
        self.current_node = home_port
        self.current_load = 0 # current load of boat (useful during construction of route)
        self.current_fish_time = 0 # current fish time (useful during construction of route)
        self.total_time = 0 # sum of all trips' total time
        try:
            self.time = np.load("final_code/local data/true_time.npy")
        except (FileNotFoundError, ValueError):
            try:
                self.time = np.load("local data/true_time.npy")
            except (FileNotFoundError, ValueError):
                self.time = time_matrix
        self.catch = load_catch_fixed_random_assignment().to_numpy().reshape(-1)
    
    def __str__(self):
        return f'Current load: {self.current_load:.2f}/{self.capacity}\nCurrent time: {self.current_fish_time:.2f}\nTotal time: {self.total_time:.2f}\nCurrent route: {self.route}'
    
    def evaluate_total_time(self):
        total_time = 0
        # for i in range(len(self.route)-1):
        #     total_time += Prob.time[self.route[i], self.route[i+1]]
        for trip in self.route:
            trip.evaluate_total_time()
            total_time += trip.total_time
        
        self.total_time = total_time
    
    def evaluate_fish_time(self):
        for trip in self.route:
            trip.evaluate_fish_time()

    def evaluate_total_catch(self):
        for trip in self.route:
            trip.evaluate_total_catch()

    def update_station_list(self):
        station_list = []
        for trip in self.route:
            station_list += trip.stations

        self.stations = station_list

    def update_station_dict(self):
        for t in range(len(self.route)):
            trip = self.route[t]
            for station in trip.stations:
                self.station_dict[station] = t
    
    def add_to_route(self, Prob, node):
        # used by initial solution constructions
        if node < 13:
            # next node is a port
            trip = self.route[-1]
            trip.end_port = node
            self.route.append(Trip(start_port=node))
            if len(trip.stations) > 0:
                time_added = self.time[trip.stations[-1], node]
            else:
                time_added = self.time[trip.start_port, trip.end_port]
            trip.total_time += time_added
            self.total_time += time_added
            self.current_node = node
            self.current_fish_time = 0
            self.current_load = 0
            
            return
        else:
            # next node is a station            
            trip = self.route[-1]
            trip.stations.append(node)
            self.stations.append(node)
            self.station_dict[node] = len(self.route)-1
            Prob.boat_dict[node] = self.id
            Prob.stations_left.remove(node)
            if node % 2 == 0:
                tow_node = node - 1
            else:
                tow_node = node + 1
            trip.stations.append(tow_node)
            self.stations.append(tow_node)
            self.station_dict[tow_node] = len(self.route)-1
            Prob.boat_dict[tow_node] = self.id
            self.current_node = tow_node
            Prob.stations_left.remove(tow_node)
            if len(trip.stations) == 2:
                # if this first stations in trip, don't count start node to station in fish time
                self.current_fish_time += self.time[node, tow_node]
                trip.fish_time += self.time[node, tow_node]
                self.total_time += self.time[trip.start_port, node] + self.time[node, tow_node]
                trip.total_time += self.time[trip.start_port, node] + self.time[node, tow_node]
            else:
                time_added = self.time[trip.stations[-3], node] + self.time[node, tow_node]
                self.current_fish_time += time_added
                trip.fish_time += time_added
                self.total_time += time_added
                trip.total_time += time_added

            self.current_load += float(self.catch[int(math.floor((node-13)/2))])
            trip.total_catch += float(self.catch[int(math.floor((node-13)/2))])
        
        return
    
    def get_closest_station(self, Prob, rcl_size=1):
        # get the closest rcl_size th closest stations to boats current node
        if rcl_size >= len(Prob.stations_left):
            return Prob.stations_left
        idx = np.argpartition(self.time[self.current_node, Prob.stations_left], rcl_size)
        stations = np.array(Prob.stations_left)[idx[:rcl_size]]
        # print(stations)
        return stations
    
    def get_closest_port(self, Prob):
        return Prob.ports[np.argmin(self.time[self.current_node, Prob.ports])]
    
    def return_to_closest_port(self, Prob):
        next_port = Prob.ports[np.argmin(self.time[self.current_node, Prob.ports])]
        self.add_to_route(Prob, next_port)
        return
    
    def return_to_home_port(self, Prob):
        self.add_to_route(Prob, self.home_port)
        return

    def check_node_valid(self, Prob, node):
        """
        used when constructing a solution - checks that adding a station to a current solution doesn't make it infeasible
        """
        if node < 13:
            # port always feasible
            return True
        else:
            if node not in Prob.stations_left:
                return False
            if self.current_load + float(self.catch[int(math.floor((node-13)/2))]) > self.planning_capacity:
                return False
            if node % 2 == 0:
                tow_node = node - 1
            else:
                tow_node = node + 1
            # check if going to station, then towing, then returning to nearest port exceeds fish time
            next_port = Prob.ports[np.argmin(self.time[tow_node, Prob.ports])]
            if self.current_node < 13:
                # start port to first station doesn't count towards fish time
                time_added = self.time[node, tow_node] + self.time[tow_node, next_port]
            else:
                time_added = self.time[self.current_node, node] + self.time[node, tow_node] + self.time[tow_node, next_port]
            if self.current_fish_time + time_added > Prob.fish_time_limit:
                return False
            
            return True
            
    def improve_route(self, Prob):
        """
        optimises the route (with TSP solver) by call improve_trip for each trip in the route
        """
        for trip in self.route:
            trip.improve_trip(Prob)
        self.evaluate_total_time()

        return
        

class Trip:
    def __init__(self, start_port, stations=None, end_port=None, id=0):
        self.start_port = start_port
        if end_port is None:
            self.end_port = start_port
        else:
            self.end_port = end_port
        self.stations = list(stations) if stations is not None else []
        self.nice_trip = []
        self.actual_trip = []
        self.total_dist = 0
        self.total_catch = 0
        self.total_time = 0
        self.fish_time = 0
        self.route_lines = []
        self.id = id
        try:
            self.time = np.load("final_code/local data/true_time.npy")
        except (FileNotFoundError, ValueError):
            try:
                self.time = np.load("local data/true_time.npy")
            except (FileNotFoundError, ValueError):
                self.time = time_matrix

        self.catch = load_catch_fixed_random_assignment().to_numpy().reshape(-1)

    def __repr__(self):
        # return f"nice trip: {self.nice_trip} \nactual trip: {self.actual_trip}"
        # return f'{self.id}'
        # return self.id if self.route_lines != [] else f'{self.id} (no lines)'
        return f"nice trip: {self.nice_trip}"
    
    def evaluate_total_catch(self):
        total_catch = 0
        if len(self.stations) > 0:
            for i in range(0, len(self.stations), 2):
                total_catch += float(self.catch[int(math.floor((self.stations[i]-13)/2))])
        elif len(self.actual_trip) >= 4:
            for i in range(1, len(self.actual_trip), 2):
                if self.actual_trip[i] <= 12:
                    continue
                total_catch += float(self.catch[int(math.floor((self.actual_trip[i]-13)/2))])
        self.total_catch = total_catch
    
    def evaluate_fish_time(self): 
        """
        Calculates time starting from the first station in the trip and finishing at a port.
        This is compared against fish_time_limit.
        Used to make sure fish does not start rotting before visiting a port.
        """
        if len(self.stations) == 0 and len(self.nice_trip) <= 1:
            self.fish_time = 0
            return
        fish_time = 0
        if self.stations:
            for i in range(len(self.stations)-1):
                fish_time += self.time[self.stations[i], self.stations[i+1]]
            # don't include start port because that doesn't count towards fish time
            if self.end_port is not None:
                fish_time += self.time[self.stations[-1], self.end_port]
        else:
            for i in range(1, len(self.nice_trip)-1):
                fish_time += self.time[self.nice_trip[i], self.nice_trip[i+1]]

        self.fish_time = fish_time
        # print('fish time:', self.fish_time)
        return
        
    def evaluate_total_time(self):
        """
        Calculates the total time of the trip, from leaving a port and returning to a port.
        """
        full_trip = self.get_full_trip()
        if len(full_trip) < 2 and len(self.nice_trip) <= 1:
            # print(len(full_trip), len(self.nice_trip))
            self.total_time = 0
            return
        
        total_time = 0
        if full_trip:
            for i in range(len(full_trip)-1):
                total_time += self.time[full_trip[i], full_trip[i+1]]
        else:
            for i in range(len(self.nice_trip)-1):
                total_time += self.time[self.nice_trip[i], self.nice_trip[i+1]]
        self.total_time = total_time

    def get_full_trip(self):
        """
        Concatenates the start port, stations, and end port into a full trip.
        """
        full_trip = []
        if self.start_port is not None:
            full_trip.append(self.start_port)
        if len(self.stations) > 0:
            full_trip += self.stations
        if self.end_port is not None:
            full_trip.append(self.end_port)
        return full_trip
    
    def improve_trip(self, Prob):
        """
        calls solve_tsp to optimise the station order
        """
        # solve TSP without fish time constraint first
        full_trip = self.get_full_trip()
        if len(full_trip) > 3:
            new_trip = solve_tsp(full_trip, Prob.time, initial_solution=False)
            self.stations = new_trip[1:-1].copy()

        # if improved solution violates fish time, re-solve with fish time constraints
        self.evaluate_fish_time()
        if self.fish_time > Prob.fish_time_limit:
            new_trip = solve_tsp(full_trip, Prob.time, time_limit=Prob.fish_time_limit, initial_solution=False)
            self.stations = new_trip[1:-1].copy()
        # if start/end ports are the same and if reversing trip can decrease fish time, do it (time from start port to first shorter than time from last to end port)
        if self.start_port == self.end_port and len(self.stations) > 0:
            if self.time[self.start_port, self.stations[0]] < self.time[self.stations[-1], self.end_port]:
                self.stations = [self.stations[len(self.stations)-i-1] for i in range(len(self.stations))]
                
        self.evaluate_fish_time()
        self.evaluate_total_time()
    
    def get_top_percentile_time_between_nodes(self, percentile):
        """
        returns the top x% time of the travel times between nodes in the trip
        (travel times along tows (betw station pairs) don't count)
        """
        full_trip = self.get_full_trip()
        time_list = []
        for i in range(0, len(full_trip)-1, 2):
            time_list.append(self.time[full_trip[i], full_trip[i+1]])

        return np.percentile(time_list, percentile)
    
    def get_nice_trip_from_actual_trip(self):
        # print(f"start nice to actual {self.nice_trip}")
        if len(self.actual_trip) <= 1:
            return # actual trip too short to convert
        actual_input = self.actual_trip.copy()
        nice_trip_from_actual = [int(actual_input[0])]
        for i in range(1, len(actual_input) - 2, 2):
            if actual_input[i] % 2 == 0:
                nice_trip_from_actual.append(int((actual_input[i] - 14)/2 + 13))
            if actual_input[i] % 2 == 1:
                nice_trip_from_actual.append(int((actual_input[i] - 13)/2 + 13))   
        nice_trip_from_actual.append(int(actual_input[-1]))
        self.nice_trip = nice_trip_from_actual.copy()
        # print(f"end nice to actual {self.nice_trip}")

    def get_actual_trip_from_nice_trip(self):
        if len(self.nice_trip) <= 1:
            return # nice trip too short to convert
        if self.nice_trip[0] >= 13:
            return # first node must be a port
        
        # print(f"pre {self.nice_trip}")
        nice_input = list(np.array(self.nice_trip.copy()) - 13)
        actual_trip = [int(nice_input[0] + 13)]

        if self.nice_trip[-1] >= 13:
            for i in range(1, len(nice_input)):
                # print(f"nice_input[i]: {nice_input[i]}")
                actual_trip.extend([int(nice_input[i] * 2 + 13), int(nice_input[i] * 2 + 14)])
            self.actual_trip = actual_trip.copy()
            # print(f"post {self.nice_trip}")
        else:
            for i in range(1, len(nice_input) - 1):
                actual_trip.extend([int(nice_input[i] * 2 + 13), int(nice_input[i] * 2 + 14)])
            actual_trip.append(int(nice_input[-1] + 13))
            # print(actual_trip, nice_input)
            actual_trip = solve_tsp(actual_trip, self.time)
            self.actual_trip = actual_trip.copy()
            self.get_nice_trip_from_actual_trip()
