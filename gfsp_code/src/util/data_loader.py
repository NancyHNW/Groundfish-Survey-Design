import numpy as np
import pandas as pd
import os
import re
import math

from .distance import degmin2deg
from ..configs import DATA_DIR, OUT_DIR

def load_island_data():
    """Return the island boundary in degrees."""

    island_path = os.path.abspath(os.path.join(DATA_DIR, 'island.bin'))

    with open(island_path, 'rb') as f:
        landata = np.fromfile(f, dtype=np.float32)
    LandDeg = np.vstack((landata[:int(landata.size/2)],-landata[int(landata.size/2):])).T
    return LandDeg

def load_ports_data():
    """Load ports data in degrees"""
    Ports = pd.read_csv('data/ports.csv')
    Ports["longitude"] = -Ports["longitude"]
    return Ports

def load_station_data():
    """Load stations data in degrees"""
    # get station data

    station_path = os.path.abspath(os.path.join(DATA_DIR, 'smb.2019.dat'))

    with open(station_path) as f:
        lines = [line.split('\t') for line in f.readlines()]
    # convert station data from list to dataframe
    stations=pd.DataFrame(
        np.array(lines)[:,[3, 4, 5, 6]],
        columns=['i_latitude', 'i_longitude', 'f_latitude','f_longitude']
    )
    columns=stations.columns
    # convert all the columns to integer
    for column in columns:
        stations[column]=stations[column].astype(int)
    stations = degmin2deg(stations)

    station_labels = np.array(lines)[:,[0,1]]
    station_labels = np.char.add(station_labels[:, 0], np.char.zfill(station_labels[:,1], 2))

    return stations, station_labels


def load_catch_fixed_random_assignment():
    """Load catch data.
    Note the catch data is not the actual catch data of those stations but a randomly selectd
    set of real catch data from 2019 intended to be more similar to the actual data for testing
    """

    # catch_path = os.path.abspath(os.path.join(DATA_DIR,
    #     'RandomMatchingCatches.dat'))

    catch_path = os.path.abspath(os.path.join(DATA_DIR,
        'LimitedCatch.csv'))

    # get station data
    with open(catch_path) as f:
        lines = [line.split(',') for line in f.readlines()]
    # convert station data from list to dataframe
    stations = pd.DataFrame(
        np.array(lines)[1:, [1]],
        columns=['catch']
    )
    columns = stations.columns
    # convert all the columns to floats
    for column in columns:
        stations[column] = stations[column].astype(float)
    return stations

def load_nodes():
    """Load port and station data in degrees.

    Return:
        nodes: (Numpy 2d array) lat and lon in degrees of all the nodes
            The dimension is (np + ns * 2, 2)
            2 nodes from each stations are touching in the array.
        Np: (Integer) number of ports
        Ns: (Integer) number of stations
    
    Future:
        nw: number of way points in the problem. 
            The dimension of nodes would be (np + nw + ns * 2, 2)
            Present in the order of ports, waypoints, stations.
    """
    ports = load_ports_data()
    ports = ports[["latitude", "longitude"]].to_numpy()
    stations, station_labels = load_station_data()
    station_nodes = stations.to_numpy().reshape((-1,2))
    nodes = np.concatenate((ports, station_nodes), axis=0)
    return nodes, ports.shape[0], stations.shape[0], station_labels

def load_true_dist():
    """Load true distance between ports and stations."""
    # dist = np.load("local data/updated_true_dist.npy")

    dist_path = os.path.abspath(os.path.join(DATA_DIR,
        'true_dist.npy'))

    dist = np.load(dist_path)
    # make the array symmetrical
    dist[dist == -1] = 0
    dist += dist.T
    # if a distance is broken, set to inf
    dist[dist == 0] = np.inf
    # dist from a to a is 0
    np.fill_diagonal(dist, 0)
    return dist

def load_test_problem(n_stations, n_vessels, capacity, ins=None):
    """Load test problem from /test_problems and return port, stations and optimal tours if computed.
    Added sub-instances of each problem size. If no ins, then use old format, otherwise new one.    
    """
    import glob
    import json

    if ins == None:
        prob_path = os.path.abspath(os.path.join(DATA_DIR, 'test_problems',
            'ins{0:02}*')).format(problem_id)

        with open(glob.glob(prob_path)[0]) as f:
            lines = f.readlines()
        ports = np.array(lines[0].split(",")).astype(int)
        stations = np.array(lines[1].split(",")).astype(int)
        tours = None
        if len(lines) > 2:
            tours = json.loads(lines[2])
        return ports, stations, tours

    else:
        prob_path = os.path.abspath(os.path.join(DATA_DIR, 'test_problems',
          str(n_stations), str(n_vessels), str(capacity),
          'ins_{0}_{1}_{2}_{3}.txt')).format(n_stations, n_vessels,
          capacity, ins)

        with open(glob.glob(prob_path)[0]) as f:
            lines = f.readlines()
        ports = np.array(lines[1].split(",")).astype(int)
        stations = np.array(lines[3].split(",")).astype(int)
        home_ind = int(lines[5][0])
        capacities = np.array(lines[7].split(",")).astype(float)

        return ports, stations, home_ind, capacities

def load_solution(problem_id, ins):
    """Load solution to test problem and instance.    
    """
    import glob
    import json

    prob_path = os.path.abspath(os.path.join(OUT_DIR, 'solutions',
        'ins_{0}_{1}_solution.txt')).format(problem_id, ins)
    
    with open(prob_path) as f:
        lines = f.readlines()

    sol = []

    for l in lines:
        l1 = re.split(r'\[|\]', l)
        l2 = '[' + l1[1] + ']'
        trip = json.loads(l2)
        l3 = l1[0].split(',')

        sol.append((int(l3[0]), int(l3[1]), trip))

    return sol

def load_station_solution(problem_id, ins):
    """Load solution to test problem and instance.    
    """
    import glob
    import json

    prob_path = os.path.abspath(os.path.join(OUT_DIR, 'solutions',
        'ins_{0}_{1}_station_solution.txt')).format(problem_id, ins)
    
    with open(prob_path) as f:
        lines = f.readlines()

    sol = []

    for l in lines:
        l1 = re.split(r'\[|\]', l)
        l2 = '[' + l1[1] + ']'
        trip = json.loads(l2)
        l3 = l1[0].split(',')

        sol.append((int(l3[0]), int(l3[1]), trip))

    return sol

def write_lkh(stations, ports, home_ind, dummy_ports, n_vessels, capacity,
    dist, catch, file_name):

  dim = len(stations) + len(ports) + len(dummy_ports)
  n_stations = len(stations) / 2
  station_limit = 2 * np.ceil(n_stations * ((1 / n_vessels) +
    (1 / (3 * n_vessels ** 2))))
  trip_lim = (n_stations * 7.5 + 450) * 1000
  total_lim = 4200000 / n_vessels

  vrp_file = file_name + '.vrp'

  with open(vrp_file, "w+") as f:
    f.write("NAME: GFS_20_1\n")
    f.write("TYPE: GFSP\n")
    f.write("DIMENSION: {0}\n".format(dim))
    f.write("VEHICLES: {0}\n".format(n_vessels))
    f.write("CAPACITY: {0}\n".format(capacity[0]))
    f.write("TRIP_DIST_LIMIT: {0}\n".format(trip_lim))
    f.write("TOTAL_DIST_LIMIT: {0}\n".format(total_lim))
    # f.write("STATION_LIMIT: {0}\n".format(station_limit))
    f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
    f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
    f.write("EDGE_WEIGHT_SECTION\n")
    for s in range(dim):
      f.write(' '.join(map(str, dist[s])) + "\n")
    f.write("FIXED_EDGES_SECTION\n")
    for s in range(int(len(stations) / 2)):
      f.write('{0} {1}\n'.format(s * 2 + 1, s * 2 + 2))
    f.write("-1 \n")
    f.write("DEMAND_SECTION\n")
    for s in range(len(stations)):
      f.write('{0} {1}\n'.format(s + 1, round(catch[math.floor(s/2)][0] / 2)))
    for s in range(len(ports) + len(dummy_ports)):
      f.write('{0} 0\n'.format(len(stations) + s + 1))
    f.write("DEPOT_SECTION\n")
    f.write("{0} \n".format(len(stations) + home_ind + 1))
    f.write("-1 \n")
    f.write("CAPACITIES_SECTION\n")
    for c in capacity:
      f.write("{0} \n".format(int(c)))
    f.write("-1 \n")
    f.write("EOF")

  par_file = file_name + '.par'
  sol_file = file_name.replace('input_files', 'solutions') + '.tour'

  with open(par_file, "w+") as f:
    f.write("SPECIAL\n")
    f.write("PROBLEM_FILE = {0}\n".format(vrp_file))
    f.write("TOUR_FILE = {0}\n".format(sol_file))
    f.write("MAX_CANDIDATES = 6\n")
    f.write("MAX_TRIALS = 10000\n")
    f.write("RUNS = 1\n")
    f.write("SEED = 1\n")
    f.write("TRACE_LEVEL = 1\n")
    f.write("TIME_LIMIT = 180\n")
    # f.write("VEHICLES = {0}\n".format(np.ceil(catch.sum() / capacity) + 2))

  return 0

def read_lkh(stations, ports, home_ind, n_dummy_ports, catch, dist, dist_w, sol_file):

  with open(sol_file) as f:
    lines = f.readlines()

  mission = []
  
  for l in lines[6:-2]:
    node_num = int(l.split('\n')[0])
    mission.append(node_num)

  start_ind = mission.index(len(stations) + home_ind + 1)
  mission = mission[start_ind:] + mission[:start_ind]

  stations_visited = np.zeros(len(stations))

  dummy_start = len(stations) + len(ports) + 1
  boat_start_inds = [-1]

  for i, n in enumerate(mission):
    if n >= dummy_start:
      if n >= dummy_start + len(ports) * n_dummy_ports: # changed from len(dummy_ports)
        # print('Found dummy depot')
        n = len(stations) + home_ind + 1
        boat_start_inds.append(i + 1)
      else:
        n = ((n - dummy_start) // n_dummy_ports) + len(stations) + 1
    mission[i] = n

  # for i, n in enumerate(mission[:-1]):
  #   if mission[i] == mission[i + 1]:

  # i = 0
  # while i < len(mission)-1:
  #   if mission[i] == mission[i+1]:
  #       del mission[i]
  #   else:
  #       i = i+1


  trips = []
  trip_num = 1
  boat_num = 1

  trip_start = 0

  for i, n in enumerate(mission):
    if i in boat_start_inds:
      boat_num += 1
    if (n > len(stations) and i > 0) or (i == len(mission) - 1):
      trip = mission[trip_start:i + 1]
      trip_start = i

      # print(trip, i, boat_num)

      if len(trip) > 2:
        if i == len(mission) - 1:
          if (n <= len(stations)):
            trip.append(len(stations) + home_ind + 1)
        trips.append([boat_num, trip_num, trip, 0, 0, 0])
        trip_num += 1

  for t in trips:
    trip_catch = 0
    trip_dist = 0
    for i, n in enumerate(t[2]):
      if n > len(stations):
        t[2][i] = ports[n - len(stations) - 1]
      else:
        t[2][i] = stations[n - 1]
        trip_catch += round(catch[math.floor((n - 1)/2)][0] / 2)

      if i < len(t[2]) - 1:
        trip_dist += dist_w[n - 1][t[2][i + 1] - 1]

    t[3] = trip_catch
    t[4] = trip_dist


  for t in trips:
    trip_dist = 0
    for i, n in enumerate(t[2][:-1]):
      trip_dist += dist[n][t[2][i + 1]]

    t[5] = trip_dist

  return trips
