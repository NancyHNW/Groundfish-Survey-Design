import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from util.distance import *
from data_loader import *
from itertools import permutations, combinations, product
from classes import Problem, Boat, Trip
from better_tsp import solve_tsp
import random
import time

# using the write_to_txt method

# problem_id = 1
# n_boats = 2
# time_limit = 30
# boat_capacities = [7000, 8000]
# home_ports = [12, 12]
# if problem_id == -1:
#     port_idx = np.arange(13)
#     stations_ids = np.arange(581)
# else:
#     port_idx, stations_ids, _ = load_test_problem(problem_id)

# random.seed(0)
# # stations_ids = np.array(random.sample(list(stations_ids), 250))
# stations_idx = np.stack([stations_ids * 2, stations_ids * 2 + 1]).T.reshape(-1) + 13

# Prob = Problem(list(stations_idx), port_idx, time_limit, n_boats, boat_capacities, home_ports)

# Prob.generate_initial_solution(seed=1)

# Prob.write_problem_to_txt('problem1.txt')

with open('problem1.txt', 'r') as f:
        lines = f.readlines()

lines = [line.strip() for line in lines]


stations = eval(lines[0])
ports = eval(lines[1])
fish_time_limit = eval(lines[2])
n_boats = eval(lines[3])
boat_capacities = eval(lines[4])
home_ports = eval(lines[5])
sol_list = eval(lines[6])

Prob = Problem(list(stations), ports, fish_time_limit, n_boats, boat_capacities, home_ports)
Prob.restore_solution_from_list(sol_list)
Prob.plot_all_routes()
plt.show()