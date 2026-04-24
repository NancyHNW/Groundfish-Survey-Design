import subprocess
import numpy as np
import matplotlib.pyplot as plt
from ..util.distance import *
from ..util.data_loader import *
from ..util.visualise import (vis_ports_stations, vis_sol,
  vis_ports_stations_simple, vis_sol_simple)
from itertools import permutations, combinations, product
import math

island_data = load_island_data()
nodes, n_ports, n_stations, station_labels = load_nodes()
dist = load_true_dist()
# index of nodes
index = np.arange(nodes.shape[0])

bounds = [0, 350, -50, np.inf]
ports = [11, 12]
# ports = np.arange(13)
n_instances = 30
n_prob_stations = [20, 40, 60, 80, 100]
vessels = [2, 3, 4]
capacity_factor = [62.5, 125, 250]
capacities = [n_prob_stations * 250, n_prob_stations * 275]
home_port_ind = 0

dist_limit = 500000

for ns in n_prob_stations:
  print('Instance size: {0}'.format(ns))
  for nv in vessels:
    print('Vessels: {0}'.format(nv))
    for cf in capacity_factor:
      print('Capacity Factor: {0}'.format(cf))
      base_capacity = int(cf * ns)

      n_boats = nv
  

      # home_ind = 3
      # home_ind = 0

      for t_i in range(1, n_instances + 1):
        print('Instance: {0}'.format(t_i))
        # t_i = 9

        port_idx, stations_ids, home_ind, catch_capacity = (
          load_test_problem(ns, nv, base_capacity, t_i))
        port_idx, len(stations_ids), stations_ids
        port_idx = np.array(port_idx)

        catch = load_catch_fixed_random_assignment().to_numpy()

        int_catch = np.ceil(catch)
        total_catch = int_catch[stations_ids].sum()

        n_dummy_port = int(np.ceil(total_catch / catch_capacity[-1]))
        dummy_start = n_ports + 2 * n_stations
        ports_idx = port_idx

        station_idx = stations_ids * 2 + n_ports
        stations_idx = np.stack([stations_ids * 2, stations_ids * 2 + 1]).T.reshape(-1) + n_ports

        dummy_idx = []

        actual_port_dict = {p: p for p in ports_idx}

        counter=0
        for port in range(len(ports_idx)):
          for i in range(n_dummy_port):
            dummy_idx.append(dummy_start + counter)
            actual_port_dict[dummy_start + counter] = ports_idx[port]
            counter = counter + 1

        dummy_idx = np.array(dummy_idx)

        all_idx = np.concatenate((stations_idx, port_idx, dummy_idx))
        n_station_nodes = len(stations_idx)

        distance_working = np.zeros([len(all_idx),len(all_idx)])

        distance_working[:n_station_nodes, :n_station_nodes] = (dist[np.ix_(stations_idx,stations_idx)] * 1000).round()

        port_dist_inds = n_station_nodes + np.arange(len(port_idx))
        station_port_dists = (dist[np.ix_(stations_idx, ports_idx)] * 1000).round()

        distance_working[:n_station_nodes, port_dist_inds] = station_port_dists
        distance_working[port_dist_inds, :n_station_nodes] = station_port_dists.T

        dummy_dist_inds = n_station_nodes + len(port_idx) + np.arange(len(port_idx) * n_dummy_port)
        station_dummy_dists = np.repeat(station_port_dists, n_dummy_port, axis=1)
          
        distance_working[:n_station_nodes, dummy_dist_inds] = station_dummy_dists
        distance_working[dummy_dist_inds, :n_station_nodes] = station_dummy_dists.T

        port_port_inds = np.hstack((port_dist_inds, dummy_dist_inds))

        port_port_dist = (dist[np.ix_(ports_idx, ports_idx)] * 1000).round()

        port_port_dist = np.hstack((port_port_dist, np.repeat(port_port_dist, n_dummy_port, axis=1)))

        port_port_dist = np.vstack((port_port_dist, np.repeat(port_port_dist, n_dummy_port, axis=0)))

        distance_working[np.ix_(port_port_inds, port_port_inds)] = port_port_dist

        os.makedirs(os.path.join(OUT_DIR, 'lkh', 'input_files',
          str(ns), str(nv), str(base_capacity)), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'lkh', 'solutions',
          str(ns), str(nv), str(base_capacity)), exist_ok=True)

        file_base = os.path.abspath(os.path.join(OUT_DIR, 'lkh', 'input_files',
          str(ns), str(nv), str(base_capacity),
          'ins_{0}_{1}_{2}_{3}')).format(ns, nv, base_capacity, t_i)
        
        result = subprocess.run(['./LKH', file_base + '.par'],
          capture_output=True, text=True)

        out_file = os.path.abspath(os.path.join(OUT_DIR, 'lkh', 'solutions',
          str(ns), str(nv), str(base_capacity),
          'ins_{0}_{1}_{2}_{3}.txt')).format(ns, nv, base_capacity, t_i)

        with open(out_file, "w") as text_file:
          text_file.write(result.stdout)

        tour_file = out_file.rstrip('txt') + 'tour'

        trips = read_lkh(stations_idx, ports_idx, home_ind, n_dummy_port, catch[((station_idx - n_ports) / 2).astype('int32')], dist, distance_working, tour_file)

        sol_file = out_file.rstrip('txt') + 'sol'

        with open(sol_file, "w+") as f:
          for t in trips:
            f.write(f"{', '.join(map(str, t))}\n")

        save_file = os.path.abspath(os.path.join(OUT_DIR, 'lkh', 'solutions',
          str(ns), str(nv), str(base_capacity),
          'ins_{0}_{1}_{2}_{3}_solution.pdf')).format(ns, nv, base_capacity,
          t_i)

        vis_sol_simple(trips, nodes, n_ports, ports_idx, stations_ids,
          station_labels, save_file)

