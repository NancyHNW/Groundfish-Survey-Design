import numpy as np
from util.distance import *
from data_loader import *
from classes import Trip, Problem
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

boat_capacities = [15000, 40000, 200000, 200000]
# boat_capacities = []
# boat_capacities = [15000]
n_boats = len(boat_capacities)
dist_limit = 700
fish_time_limit = 120

# port_idx, stations_ids, _ = load_test_problem(test_problem)
# port_idx = np.append(port_idx, 10)
# port_idx = np.array(port_idx)

home_ports = [12, 12, 12, 12]
# home_ports = [12]
port_idx = np.arange(13)
station_ids = np.arange(581)
station_ids_shifted = station_ids + 13

island_data = load_island_data()
nodes, n_ports, n_stations = load_nodes()

dict_station_num_annotations = {}
previous_selection = [None]
try: 
    time = np.load("local data/true_time.npy")
except FileNotFoundError:
    time = np.load("my_code/local data/true_time.npy")

distance = load_true_dist()
catch = load_catch_fixed_random_assignment().to_numpy()

# index of nodes
index = np.arange(nodes.shape[0])

ports_idx = port_idx

station_idx = station_ids * 2 + n_ports
stations_idx = np.stack([station_ids * 2, station_ids * 2 + 1]).T.reshape(-1) + n_ports


node_idx = np.concatenate([port_idx, station_idx])
nodes_idx = np.concatenate([ports_idx, stations_idx])

current_n_ports = len(ports_idx)
current_n_stations = len(station_ids)

island_data_converted_to_points = np.array(degArr2point(island_data))
nodes_converted_to_points = np.array(degArr2point(nodes)).T

try:
    time_matrix = np.load("local data/true_time.npy")
except FileNotFoundError:
    time_matrix = np.load("my_code/local data/true_time.npy")

n_nodes = len(nodes)
trip_objs = []
boat_objs = []

# heuristic parameters
k_catch= 10
k_fish = 2000 
upper_bound = 200
iterations = 10
work_limit = 300
tsp = 0.01

Prob = Problem(list(stations_idx), ports_idx, fish_time_limit, n_boats)
