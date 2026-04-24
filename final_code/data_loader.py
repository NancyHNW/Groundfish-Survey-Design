import numpy as np
import pandas as pd

from util.distance import degmin2deg

def load_island_data():
    """Return the island boundary in degrees."""
    try:
        with open('my_code/data/island.bin', 'rb') as f:
            landata = np.fromfile(f, dtype=np.float32)
    except FileNotFoundError: 
        with open('data/island.bin', 'rb') as f:
            landata = np.fromfile(f, dtype=np.float32)
    LandDeg = np.vstack((landata[:int(landata.size/2)],-landata[int(landata.size/2):])).T
    return LandDeg

def load_ports_data():
    """Load ports data in degrees"""
    try:
        Ports = pd.read_csv('my_code/data/ports.csv')
        Ports["longitude"] = -Ports["longitude"]
    except FileNotFoundError:
        Ports = pd.read_csv('data/ports.csv')
        Ports["longitude"] = -Ports["longitude"]
    return Ports

def load_station_data():
    """Load stations data in degrees"""
    # get station data
    try:
        with open('my_code/sample data/smb.2019.dat') as f:
            lines = [line.split('\t') for line in f.readlines()]
    except FileNotFoundError: 
        with open('sample data/smb.2019.dat') as f:
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
    return stations


def load_catch_fixed_random_assignment():
    """Load catch data.
    Note the catch data is not the actual catch data of those stations but a randomly selectd
    set of real catch data from 2019 intended to be more similar to the actual data for testing
    """
    # get station data

    try:
        with open('my_code/sample data/RandomMatchingCatches.dat') as f:
            lines = [line.split(',') for line in f.readlines()]
        # convert station data from list to dataframe
        stations = pd.DataFrame(
            np.array(lines)[1:, [1]],
            columns=['catch']
        )
    except FileNotFoundError:
        with open('sample data/RandomMatchingCatches.dat') as f:
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
    stations = load_station_data()
    station_nodes = stations.to_numpy().reshape((-1,2))
    nodes = np.concatenate((ports, station_nodes), axis=0)
    return nodes, ports.shape[0], stations.shape[0]

def load_true_dist():
    """Load true distance between ports and stations."""
    # dist = np.load("local data/updated_true_dist.npy")
    try:
        dist = np.load("my_code/local data/true_dist(temporary).npy")
    except FileNotFoundError:
        dist = np.load("local data/true_dist(temporary).npy")
    # make the array symmetrical
    dist[dist == -1] = 0
    dist += dist.T
    # if a distance is broken, set to inf
    dist[dist == 0] = np.inf
    # dist from a to a is 0
    np.fill_diagonal(dist, 0)
    return dist

def load_test_problem(problem_id):
    """Load test problem from /test_problems and return port, stations and optimal tours if computed."""
    import glob
    import json
    import os
    # print("Current working directory:", os.getcwd())

    file_paths = glob.glob(f"./test_problems/ins{problem_id:02}*")

    if file_paths:
        with open(glob.glob(f"./test_problems/ins{problem_id:02}*")[0]) as f:
            lines = f.readlines()
    else:
        with open(glob.glob(f"./my_code/test_problems/ins{problem_id:02}*")[0]) as f:
            lines = f.readlines()

    ports = np.array(lines[0].split(",")).astype(int)
    stations = np.array(lines[1].split(",")).astype(int)
    tours = None
    if len(lines) > 2:
        tours = json.loads(lines[2])
    return ports, stations, tours
