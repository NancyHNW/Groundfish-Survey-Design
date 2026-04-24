import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from util.distance import *
from data_loader import *
from itertools import permutations, combinations, product
from better_tsp import solve_tsp
from classes import *
import math
import random
import glob

# entire problem
# entire problem with random 50% of stations
# 1 port with 20, 50, 100 stations
# 2 ports with 20, 50, 100 stations

# 581 stations, 13 ports, 4 boats, [15000, 45000, 200000, 200000], 120, 150
# 581 stations, 13 ports, 4 boats, [100000, 100000, 100000, 100000], 120, 150
# 581 stations, 13 ports, 2 boats, [45000, 100000], 120, 300
# 581 stations, 13 ports, 2 boats, [75000, 75000], 120, 300
# 581 stations, 13 ports, 1 boat, [200000], 120, 581
# 581 stations, 13 ports, 1 boat, [50000], 120, 581
# 290 stations, 13 ports, 4 boats, [15000, 45000, 200000, 200000], 120, 75
# 290 stations, 13 ports, 4 boats, [100000, 100000, 100000, 100000], 120, 75
# 290 stations, 13 ports, 2 boats, [45000, 100000], 120, 150
# 290 stations, 13 ports, 2 boats, [75000, 75000], 120, 150
# 290 stations, 13 ports, 1 boat, [200000], 120, 290
# 290 stations, 13 ports, 1 boat, [50000], 120, 290
# 100 stations, 2 ports, 2 boats, [5000, 30000], 80, 55
# 100 stations, 2 ports, 2 boats, [20000, 20000], 80, 55
# 100 stations, 2 ports, 1 boat, [30000], 80, 100
# 100 stations, 2 ports, 1 boat, [10000], 80, 100
# 100 stations, 1 port, 2 boats, [5000, 30000], 80, 55
# 100 stations, 1 port, 2 boats, [20000, 20000], 80, 55
# 100 stations, 1 port, 1 boat, [30000], 80, 100
# 100 stations, 1 port, 1 boat, [10000], 80, 100
# 50 stations, 2 ports, 2 boats, [5000, 15000], 60, 28
# 50 stations, 2 ports, 2 boats, [10000, 10000], 60, 28
# 50 stations, 2 ports, 1 boat, [15000], 60, 50
# 50 stations, 2 ports, 1 boat, [50000], 60, 50
# 50 stations, 1 port, 2 boats, [5000, 15000], 60, 28
# 50 stations, 1 port, 2 boats, [10000, 10000], 60, 28
# 50 stations, 1 port, 1 boat, [15000], 60, 50
# 50 stations, 1 port, 1 boat, [50000], 60, 50
# 20 stations, 2 ports, 2 boats, [5000, 10000], 30, 12
# 20 stations, 2 ports, 2 boats, [7000, 7000], 30, 12
# 20 stations, 2 ports, 1 boat, [30000], 30, 20
# 20 stations, 2 ports, 1 boat, [10000], 30, 20
# 20 stations, 1 port, 2 boats, [5000, 10000], 30, 12
# 20 stations, 1 port, 2 boats, [7000, 7000], 30, 12
# 20 stations, 1 port, 1 boat, [30000], 30, 20
# 20 stations, 1 port, 1 boat, [10000], 30, 20

import re

def values_only(s):
    # Extract numbers and lists
    matches = re.findall(r'\[.*?\]|\d+', s)
    # Join them with semicolons
    return "; ".join(matches)

# Example usage
strings = ['581 stations, 13 ports, 4 boats, [15000, 45000, 200000, 200000], 120, 150',
'581 stations, 13 ports, 4 boats, [100000, 100000, 100000, 100000], 120, 150',
'581 stations, 13 ports, 2 boats, [45000, 100000], 120, 300',
'581 stations, 13 ports, 2 boats, [75000, 75000], 120, 300',
'581 stations, 13 ports, 1 boat, [200000], 120, 581',
'581 stations, 13 ports, 1 boat, [50000], 120, 581',
'200 stations, 13 ports, 4 boats, [15000, 45000, 200000, 200000], 120, 75',
'200 stations, 13 ports, 4 boats, [100000, 100000, 100000, 100000], 120, 75',
'200 stations, 13 ports, 2 boats, [45000, 100000], 120, 150',
'200 stations, 13 ports, 2 boats, [75000, 75000], 120, 150',
'200 stations, 13 ports, 1 boat, [200000], 120, 290',
'200 stations, 13 ports, 1 boat, [50000], 120, 290',
'100 stations, 2 ports, 2 boats, [5000, 30000], 80, 55',
'100 stations, 2 ports, 2 boats, [20000, 20000], 80, 55',
'100 stations, 2 ports, 1 boat, [30000], 80, 100',
'100 stations, 2 ports, 1 boat, [10000], 80, 100',
'100 stations, 1 port, 2 boats, [5000, 30000], 80, 55',
'100 stations, 1 port, 2 boats, [20000, 20000], 80, 55',
'100 stations, 1 port, 1 boat, [30000], 80, 100',
'100 stations, 1 port, 1 boat, [10000], 80, 100',
'50 stations, 2 ports, 2 boats, [5000, 15000], 60, 28',
'50 stations, 2 ports, 2 boats, [10000, 10000], 60, 28',
'50 stations, 2 ports, 1 boat, [15000], 60, 50',
'50 stations, 2 ports, 1 boat, [50000], 60, 50',
'50 stations, 1 port, 2 boats, [5000, 15000], 60, 28',
'50 stations, 1 port, 2 boats, [10000, 10000], 60, 28',
'50 stations, 1 port, 1 boat, [15000], 60, 50',
'50 stations, 1 port, 1 boat, [50000], 60, 50',
'20 stations, 2 ports, 2 boats, [5000, 10000], 30, 12',
'20 stations, 2 ports, 2 boats, [7000, 7000], 30, 12',
'20 stations, 2 ports, 1 boat, [30000], 30, 20',
'20 stations, 2 ports, 1 boat, [10000], 30, 20',
'20 stations, 1 port, 2 boats, [5000, 10000], 30, 12',
'20 stations, 1 port, 2 boats, [7000, 7000], 30, 12',
'20 stations, 1 port, 1 boat, [30000], 30, 20',
'20 stations, 1 port, 1 boat, [10000], 30, 20']

strings2 = ['581 stations, 13 ports, 4 boats, [15000, 45000, 200000, 200000], 120, 160',
'200 stations, 13 ports, 4 boats, [15000, 45000, 200000, 200000], 120, 80',
'100 stations, 2 ports, 2 boats, [5000, 30000], 80, 55',
'50 stations, 1 port, 2 boats, [5000, 15000], 60, 28',
'20 stations, 1 port, 1 boat, [30000], 30, 20']

ids = [0, 6, 12, 24, 34]

results = []
for s in strings:
    result = values_only(s)
    results.append(result)

# for i in range(len(results)):
#     result = results[i]
#     split = result.split('; ')
#     s = int(split[0])
#     filename = f'problem_{i:02d}_{s}stations.txt'
#     with open(filename, "w") as f:
#         f.write(result)

port_idx, stations_ids, _ = load_test_problem(5)
# Prob = Problem(np.stack([stations_ids * 2, stations_ids * 2 + 1]).T.reshape(-1) + 13, [12], 30, 1, [1], [12])
# Prob.plot_all_routes()
# plt.show()

for i in ids: # range(36):
    for count in range(10):
        random.seed(count*100)
        # filename = glob.glob(f"my_code/new_test_problems/problem_{i:02d}*")[0]
        # with open(filename, "r") as f:
        #     result = f.read()

        split = values_only(strings[i]).split('; ')

        s = int(split[0])
        p = int(split[1])

        # Prob = Problem([19, 20], [1], 30, 1, [1], [1])

        if s > 300:
            stations_ids = list(range(581))
            ports_ids = [4, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]

            stations_idx = np.stack([np.array(stations_ids) * 2, np.array(stations_ids) * 2 + 1]).T.reshape(-1) + 13
            # print(max(stations_idx))
            # max1 = 0
            # for p in ports_ids:
            #     max1 = max(np.max(Prob.time[stations_idx, p]), max1)
            # if max1 > int(split[4])*0.95:
            #     print('error')
        elif s > 100:
            stations_ids = list(range(581))
            ports_ids = [4, 0, 6, 7, 9, 11, 12]

            stations_idx = np.stack([np.array(stations_ids) * 2, np.array(stations_ids) * 2 + 1]).T.reshape(-1) + 13
            # max1 = 0
            # for p in ports_ids:
            #     max1 = max(np.max(Prob.time[stations_idx, p]), max1)
            # if max1 > int(split[4])*0.95:
            #     print('error')
        elif s > 20 and p == 2:
            port_idx, stations_ids1, _ = load_test_problem(5)
            port_idx, stations_ids2, _ = load_test_problem(2)
            stations_ids = list(set(stations_ids1).union(set(stations_ids2)))
            ports_ids = [11, 12]

            # stations_idx = np.stack([np.array(stations_ids) * 2, np.array(stations_ids) * 2 + 1]).T.reshape(-1) + 13
            # max1 = 0
            # for p in ports_ids:
            #     max1 = max(np.max(Prob.time[stations_idx, p]), max1)
            # if max1 > int(split[4])*0.95:
            #     print('error')
            
        elif s > 20 and p == 1:
            port_idx, stations_ids, _ = load_test_problem(5)
            ports_ids = [11]

            # stations_idx = np.stack([np.array(stations_ids) * 2, np.array(stations_ids) * 2 + 1]).T.reshape(-1) + 13
            # max1 = 0
            # for p in ports_ids:
            #     max1 = max(np.max(Prob.time[stations_idx, p]), max1)
            # if max1 > int(split[4])*0.95:
            #     print('error')
        elif p == 2:
            port_idx, stations_ids, _ = load_test_problem(4)
            ports_ids = [12, 11]

            stations_idx = np.stack([np.array(stations_ids) * 2, np.array(stations_ids) * 2 + 1]).T.reshape(-1) + 13
            # max1 = 0
            # for p in ports_ids:
            #     max1 = max(np.max(Prob.time[stations_idx, p]), max1)
            # # print(int(split[4]))
            # print(Prob.time[12, 121] + Prob.time[121, 122])
            
            # Prob2 = Problem(list(stations_idx), ports_ids, 30, 2, [5000, 10000], [11, 11])
            # Prob2.plot_all_routes()
            # plt.show()
            # if max1 > int(split[4])*0.75:
            #     print('error')
        else:
            port_idx, stations_ids, _ = load_test_problem(6)
            ports_ids = [12]

            stations_idx = np.stack([np.array(stations_ids) * 2, np.array(stations_ids) * 2 + 1]).T.reshape(-1) + 13
            # max1 = 0
            # for p in ports_ids:
            #     max1 = max(np.max(Prob.time[stations_idx, p]), max1)
            # if max1 > int(split[4])*0.95:
            #     print('error')
        
        station_sample = np.random.choice(stations_ids, s, replace=False)
        # stations_idx = np.stack([station_sample * 2, station_sample * 2 + 1]).T.reshape(-1) + 13
        # if i == 35:
        #     print(station_sample)
        #     Prob = Problem(np.stack([station_sample * 2, station_sample * 2 + 1]).T.reshape(-1) + 13, [12], 30, 1, [1], [12])
        #     Prob.plot_all_routes()
        #     plt.show()

        split[0] = "[" + ", ".join(map(str, station_sample)) + "]"
        # print(split)
        split[1] = str(ports_ids)

        with open(f"my_code/final_test_problems/test_problem{i:02d}_{count}.txt", "w") as f:
            f.write("; ".join(split))


    
    
    

