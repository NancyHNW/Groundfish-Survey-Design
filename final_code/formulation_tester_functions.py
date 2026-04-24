import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from util.distance import *
#from util.tour import *
from data_loader import *
from itertools import permutations, combinations, product
import math
import random
import glob

# taken from The_Multiboat_GSP_Formulation

def get_test_problem(prob_id):
    filename = glob.glob(f"better_test_problems/test_problem{prob_id:02d}*")[0]
    with open(filename, "r") as f:
        result = f.read()
    split = result.split('; ')
    s = np.array(eval(split[0]))
    p = np.array(eval(split[1]))
    b = int(split[2])
    c = np.array(eval(split[3]))
    t = int(split[4])
    w = int(split[5])
    u = int(split[6])

    return s, p, b, c, t, w, u

def get_problem_data():
    island_data = load_island_data()
    nodes, n_ports, n_stations = load_nodes()
    # dist = load_true_dist()
    try:
        dist = np.load("my_code/local data/true_time.npy")
    except FileNotFoundError:
        dist = np.load("local data/true_time.npy")
    dummy_start = n_ports + 2 * n_stations
    catch = load_catch_fixed_random_assignment().to_numpy()
    # index of nodes
    index = np.arange(nodes.shape[0])

    return island_data, nodes, n_ports, n_stations, dist, dummy_start, catch, index

def get_distance_matrix(dist, port_idx, stations_ids, n_ports, n_dummy_port, dummy_start):
    
    ports_idx = port_idx

    station_idx = stations_ids * 2 + n_ports
    stations_idx = np.stack([stations_ids * 2, stations_ids * 2 + 1]).T.reshape(-1) + n_ports

    dummy_idx = []

    counter=0
    for port in range(len(ports_idx)):
        for i in range(n_dummy_port):
            dummy_idx.append(dummy_start + counter)
            counter = counter + 1

    dummy_idx = np.array(dummy_idx)

    # indices go [ports, stations, dummies]

    distance_working = np.zeros([dummy_start + counter,dummy_start + counter])
    distance_working[0:dummy_start,0:dummy_start] = dist # copy true dist matrix except for dummies

    for port in range(len(ports_idx)):
        for i in range(n_dummy_port):
            distance_working[dummy_start + i + port * n_dummy_port,0:dummy_start]\
                = dist[ports_idx[port], 0:dummy_start]
            distance_working[0:dummy_start,dummy_start + i + port * n_dummy_port]\
                = dist[0:dummy_start,ports_idx[port]]

    for i in range(counter):
        for j in range(counter):
            distance_working[dummy_start + i, dummy_start + j] = \
                dist[ports_idx[math.floor(i/n_dummy_port)], ports_idx[math.floor(j/n_dummy_port)]]

    dist = distance_working # true distance matrix with distances for dummy nodes (i think they're the same as the ports??)

    # display(distance_working)

    if n_dummy_port == 0:
        node_idx = np.concatenate([port_idx, station_idx]) #
        nodes_idx = np.concatenate([ports_idx, stations_idx])#
    else:
        node_idx = np.concatenate([port_idx, station_idx, dummy_idx]) #
        nodes_idx = np.concatenate([ports_idx, stations_idx, dummy_idx])#

    return dist, nodes_idx, ports_idx, stations_idx, dummy_idx, station_idx, node_idx
   
def create_model3(m, valid_pairs_idx, dist, nodes_idx, catch, ports_idx, stations_idx, n_boats, dummy_idx, station_idx, node_idx, max_work, n_dummy_port, node_compress, catch_capacity, dist_limit):
    # same as from The_Multiboat_GSP_Formulation except use valid pairs to exclude some variables / constraints

    dist_dict = {}
    catch_dict = {}
    for (i, j) in permutations(nodes_idx, 2):
        if (i, j) not in valid_pairs_idx:
            continue
        dist_dict[(i, j)] = dist[i][j]
        if abs(i-j) == 1 and i <= 1174 and j <= 1174 and i >=13 and j>=13 and ((i<j and i%2 == 1) or (i>j and j%2 == 1)):
            catch_dict[(i, j)] = float(catch[int (math.floor((i-13)/2))])

    home_port = ports_idx[0]

    Xs = m.addVars(dist_dict.keys(), obj=dist_dict, vtype=GRB.BINARY, name="x")

    #forced visits

    # Each station is visited once (one leaving arc)
    m.addConstrs((Xs.sum(i, '*') == 1 for i in stations_idx), name="once")
    # The tow has to be selected
    m.addConstrs((Xs.sum(i, j) + Xs.sum(j, i) == 1 for (i, j) in stations_idx.reshape(-1, 2)), name="tow") #don't need valid_pairs cos just station pairs

    # All routes start at designated port
    m.addConstr(Xs.sum(home_port, '*') <= n_boats, name="homeout")
    # All routes end at designated port
    m.addConstr(Xs.sum('*', home_port) <= n_boats, name="homein")

    port_dummy_no_home_idx = np.concatenate([ports_idx[1:], dummy_idx])
    # Each dummy node is visited at most once (one leaving arc)
    m.addConstrs((Xs.sum(i, '*') <= 1 for i in port_dummy_no_home_idx), name="onceDummy")

    # All routes that go to node must leave node
    m.addConstrs((Xs.sum('*', i) - Xs.sum(i, '*') == 0 for i in nodes_idx), name="flow")

    node_idx_no_home = np.concatenate([ports_idx[1:], station_idx, dummy_idx])

    port_dummy_idx = np.concatenate([ports_idx, dummy_idx])


    # subtour elimination
    station_pairs = [
        (i,j) for i, j in product(node_idx_no_home, node_idx) if (i != j and (i, j) in valid_pairs_idx)
    ]

    valid_yze_tuples = []
    for i in node_idx_no_home:
        for j in node_idx:
            if i == j:
                continue
            if i in port_dummy_idx and j in port_dummy_idx:
                if (i, j) in valid_pairs_idx:
                    valid_yze_tuples.append((i, j))
            elif i in port_dummy_idx:
                if (i, j) in valid_pairs_idx or (i, j+1) in valid_pairs_idx:
                    valid_yze_tuples.append((i, j))
            elif j in port_dummy_idx:
                if (i, j) in valid_pairs_idx or (i+1, j) in valid_pairs_idx:
                    valid_yze_tuples.append((i, j))
            else:
                if (i, j) in valid_pairs_idx or (i+1, j) in valid_pairs_idx or (i, j+1) in valid_pairs_idx or (i+1, j+1) in valid_pairs_idx:
                    valid_yze_tuples.append((i, j))

    # Ys = m.addVars(node_idx_no_home, node_idx, name="y")
    Ys = m.addVars(valid_yze_tuples, name="y")
    m.addConstrs((
            Xs.sum(i, j) == Ys[i,j] \
            if np.isin([j],port_dummy_idx) and np.isin([i],port_dummy_idx) else \
            Xs.sum(i, j) + Xs.sum(i + 1, j) == Ys[i,j] \
            if np.isin([j],port_dummy_idx) else \
            Xs.sum(i, j) + Xs.sum(i, j + 1) == Ys[i,j]\
            if np.isin([i],port_dummy_idx) else \
            Xs.sum(i, j) + Xs.sum(i + 1, j) +\
            Xs.sum(i, j + 1) + Xs.sum(i + 1, j + 1) == Ys[i,j] \
            for i, j in valid_yze_tuples ), name="compressGG" #for i, j in station_pairs ), name="compressGG"
        )

    # Zs = m.addVars(node_idx_no_home, node_idx, name="z")
    Zs = m.addVars(valid_yze_tuples, name="z")

    m.addConstrs(
        (Zs.sum(i, '*') - Zs.sum('*', i) == 1 for i in station_idx) , name="GGplusoneStation"
    )
    m.addConstrs(
        (Zs.sum(i, '*') - Zs.sum('*', i) == Ys.sum(i, '*')  for i in port_dummy_no_home_idx) , name="GGplusoneDummyports"
    )
    m.addConstrs(
        # (Zs[i,j] <=(max_work + n_dummy_port/n_boats) * Ys[i,j] for i in node_idx_no_home for j in node_idx) , name="GGbigM"
        (Zs[i,j] <=(max_work + n_dummy_port/n_boats) * Ys[i,j] for i, j in valid_yze_tuples) , name="GGbigM"
    )

    if node_compress:
        cap_dict = {}
        valid_b_tuples = []
        for i in node_idx_no_home:
            if (home_port, i) in valid_pairs_idx or (home_port, i+1) in valid_pairs_idx:
                for k in range(n_boats):
                    cap_dict[(home_port, i, k)] = catch_capacity[k]
                    valid_b_tuples.append((home_port, i, k))

        #def bin vars
        # Bs = m.addVars([home_port],node_idx_no_home,range(n_boats), vtype=GRB.BINARY, name="b") # 1 if arc from hp to node j for boat k
        Bs = m.addVars(valid_b_tuples, vtype=GRB.BINARY, name="b") # 1 if arc from hp to node j for boat k
        #bin constraints
        #max one boat per arc
        m.addConstrs((Bs.sum(home_port, j, '*') <= 1 for j in node_idx_no_home), name="oneboatperarc")
        #max one arc per boat
        m.addConstrs((Bs.sum(home_port, '*', k) <= 1 for k in range(n_boats)), name="onearcperboat")
        # boat can only travel if path is selected
        m.addConstrs((Bs.sum(home_port,j,k) <= Xs.sum(home_port,j) + Xs.sum(home_port,j+1) for j in station_idx for k in range(n_boats)) , name="boatifpathselected_compress_station")
        m.addConstrs((Bs.sum(home_port,j,k) <= Xs.sum(home_port,j) for j in port_dummy_no_home_idx for k in range(n_boats)), name="Boatifpathselected_compress_port")

        #def flow
        # Es = m.addVars(node_idx_no_home,node_idx, name = 'e')
        Es = m.addVars(valid_yze_tuples, name = 'e')
        #flow constraints
        # pass the flow on
        m.addConstrs((Es.sum(i,'*') - (Es.sum('*', i) + Bs.prod(cap_dict,home_port,i,'*')) == 0 for i in node_idx_no_home), name="Capacitytransfer")
        # limited by Xs
        maxCap = max(catch_capacity)
        m.addConstrs(
            (Es[i,j] <= maxCap * Xs[i,j] \
                if np.isin([j],port_dummy_idx) and np.isin([i],port_dummy_idx) else \
                    Es[i,j] <= maxCap * Xs.sum(i,j) + maxCap * Xs.sum(i,j + 1) \
                if np.isin([i],port_dummy_idx) else\
                    Es[i,j] <= maxCap * Xs.sum(i,j) + maxCap * Xs.sum(i + 1,j) \
                if np.isin([j],port_dummy_idx) else\
                    Es[i,j] <= maxCap * Xs.sum(i,j) + maxCap * Xs.sum(i + 1,j) +\
                    maxCap * Xs.sum(i,j + 1) + maxCap * Xs.sum(i + 1,j + 1)  \
                # for i, j in station_pairs), name="Capacitytransferif_x_compress"
                for i, j in valid_yze_tuples), name="Capacitytransferif_x_compress"
        )
    else:
        nodes_idx_no_home = np.concatenate([ports_idx[1:], stations_idx, dummy_idx])
        stations_pairs = [
        (i,j) for i, j in product(nodes_idx_no_home, nodes_idx) if i != j
    ]

        cap_dict = {}
        for i in nodes_idx_no_home:
            for k in range(n_boats):
                cap_dict[(home_port, i, k)] = catch_capacity[k]


        #def bin vars
        Bs = m.addVars([home_port],nodes_idx_no_home,range(n_boats), vtype=GRB.BINARY, name="b")
        #bin constraints
        #max one boat per arc
        m.addConstrs((Bs.sum(home_port, j, '*') <= 1 for j in nodes_idx_no_home), name="oneboatperarc")
        #max one arc per boat
        m.addConstrs((Bs.sum(home_port, '*', k) <= 1 for k in range(n_boats)), name="onearcperboat")
        # boat can only travel if path is selected
        m.addConstrs((Bs[home_port,j,k] <= Xs[home_port,j] for j in nodes_idx_no_home for k in range(n_boats)), name="boatifpathselected_uncompress")
        #m.addConstr(Bs[home_port,301,0] == 1)
        #m.addConstr(Bs[home_port,149,1] == 1)

        #def flow
        Es = m.addVars(nodes_idx_no_home,nodes_idx, name = 'e')
        #flow constraints
        # pass the flow on
        m.addConstrs((Es.sum(i,'*') - (Es.sum('*', i) + Bs.prod(cap_dict,home_port,i,'*')) == 0 for i in nodes_idx_no_home), name="capacitytrasfer")
        # limited by Xs
        maxCap = max(catch_capacity)
        m.addConstrs((Es[i,j] <= maxCap * Xs[i,j] for i, j in stations_pairs), name="capacityflowif_x")

    #catch cap
    station_pairs2 = [
    (i,j) for i, j in product(station_idx, node_idx) if i != j
    ]
    valid_cpdt_tuples = []
    for i in station_idx:
        for j in node_idx:
            if i == j:
                continue
            if j in port_dummy_idx:
                if (i, j) in valid_pairs_idx or (i+1, j) in valid_pairs_idx:
                    valid_cpdt_tuples.append((i, j))
            else:
                if (i, j) in valid_pairs_idx or (i+1, j) in valid_pairs_idx or (i, j+1) in valid_pairs_idx or (i+1, j+1) in valid_pairs_idx:
                    valid_cpdt_tuples.append((i, j))

    # Cs = m.addVars(station_idx, node_idx, name="c") # cs are ws
    Cs = m.addVars(valid_cpdt_tuples, name="c") # cs are ws
    m.addConstrs(
            Xs.sum(i, j) + Xs.sum(i + 1, j) == Cs[i,j] \
            if np.isin([j],port_dummy_idx) else \
            Xs.sum(i, j) + Xs.sum(i + 1, j) +\
            Xs.sum(i, j + 1) + Xs.sum(i + 1, j + 1) == Cs[i,j] \
            # for i, j in station_pairs2
            for i, j in valid_cpdt_tuples
        )

    # Ps = m.addVars(station_idx, node_idx, name="p")
    Ps = m.addVars(valid_cpdt_tuples, name="p")

    m.addConstrs(
        Ps.sum(i, '*') - Ps.sum('*', i) == catch[int((i-13)/2)] for i in station_idx #need to change to catch at i
    )
    if node_compress:
        m.addConstrs(
            # Ps[i,j] <= Es[i,j] for i in station_idx for j in node_idx
            Ps[i,j] <= Es[i,j] for i, j in valid_cpdt_tuples
        )
    else:
        m.addConstrs(
            Ps[i,j] <= Es[i,j] + Es[i + 1,j]  \
            if np.isin([j],port_dummy_idx) else
            Ps[i,j] <= Es[i,j] + Es[i + 1,j] + Es[i,j + 1] + Es[i + 1,j + 1]
            for i in station_idx for j in node_idx
        )

    #dist cap
    #note: this doesn't start counting distance until the first station is started
    #      this is intentional as it doesn't need to consider the fish rotting if there
    #      are no fish on board. It does start at the start of the first tow, so a 4
    #      nautical mile correction factor is included later
    # Ds = m.addVars(station_idx, node_idx, name="d") #ds are es
    Ds = m.addVars(valid_cpdt_tuples, name="d") #ds are es
    m.addConstrs(
            Xs.sum(i, j) * dist[i,j] + Xs.sum(i + 1, j)  * dist[i+1,j] + \
            Xs.sum(i,j) * dist[i+1,i] + Xs.sum(i+1,j) * dist[i,i+1] == Ds[i,j] \
            if np.isin([j],port_dummy_idx) else \
            Xs.sum(i, j)  * dist[i,j] + Xs.sum(i + 1, j) * dist[i+1,j] +\
            Xs.sum(i, j + 1) * dist[i,j+1] + Xs.sum(i + 1, j + 1) * dist[i+1,j+1] + \
            (Xs.sum(i,j) + Xs.sum(i,j+1)) * dist[i+1,i] + \
            (Xs.sum(i+1,j) + Xs.sum(i+1,j+1)) * dist[i,i+1]== Ds[i,j] \
            # for i, j in station_pairs2
            for i, j in valid_cpdt_tuples
        )

    # Ts = m.addVars(station_idx, node_idx, name="t")
    Ts = m.addVars(valid_cpdt_tuples, name="t")

    m.addConstrs(
        Ts.sum(i, '*') - Ts.sum('*', i) == Ds.sum(i,'*') for i in station_idx
    )
    m.addConstrs(
        # Ts[i,j] <=(dist_limit) * Cs[i,j] for i in station_idx for j in node_idx
        Ts[i,j] <=(dist_limit) * Cs[i,j] for i, j in valid_cpdt_tuples
    )

    # return m, Xs, Ys, Zs, Bs, Cs, Ds, Es, Ps, Ts, dist_dict
    return m, Xs

def get_valid_pairs(dist, perc_to_keep, stations_idx, ports_idx, dummy_idx, n_closest_ports, n_dummy_ports):

    nodes_idx = list(stations_idx) + list(ports_idx) + list(dummy_idx)

    # Start with all directed pairs (i, j), excluding self-loops
    valid_pairs = {(i, j) for i in nodes_idx for j in nodes_idx if i != j}

    if perc_to_keep is not None and perc_to_keep < 100:
        n_closest = math.ceil(perc_to_keep / 100 * (len(stations_idx) - 1))
        n_closest = max(n_closest, 1)

        for i in stations_idx:
            # Candidate arcs (exclude self and enforce custom skip rule)
            outgoing = [
                (j, dist[i, j])
                for j in stations_idx
                if i != j and (abs(i - j) > 1 or (i > j and i % 2 == 1) or (i < j and i % 2 == 0))
            ]

            # Sort by distance ascending
            outgoing_sorted = sorted(outgoing, key=lambda x: x[1])

            # Remove arcs not in keep_set
            for j, _ in outgoing_sorted[n_closest:]:
                valid_pairs.discard((i, j))
                # valid_pairs.discard((j, i))  # remove reverse as well
    
    all_port_no_home = list(ports_idx[1:]) + list(dummy_idx[n_dummy_ports:])
    for i in all_port_no_home:
        for j in all_port_no_home:
            valid_pairs.discard((i, j))
            # print(i, j)
    
    n_ports = len(ports_idx)
    # keep only n_closest_ports for each station (need to include dummy ports)
    if n_closest_ports < n_ports:
        for st in stations_idx:
            closest_idx = np.argpartition(dist[st, ports_idx], n_closest_ports)
            idx_to_keep = closest_idx[:n_closest_ports]
            for p in range(len(ports_idx)):
                if p not in idx_to_keep:
                    valid_pairs.discard((st, ports_idx[p]))
                    valid_pairs.discard((ports_idx[p], st))
                    # print((st, ports_idx[p]))
                    # print((ports_idx[p], st))
                    for d in dummy_idx[(n_dummy_ports*(p)):(n_dummy_ports*(p+1))]:
                        valid_pairs.discard((st, d))
                        valid_pairs.discard((d, st))
                    
                        # print((st, d))
                        # print((d, st))


    return valid_pairs
