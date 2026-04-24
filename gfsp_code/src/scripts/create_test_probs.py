import numpy as np
import matplotlib.pyplot as plt
from ..util.distance import *
from ..util.data_loader import *
from ..util.visualise import vis_ports_stations_simple

island_data = load_island_data()
nodes, n_ports, n_stations, station_labels = load_nodes()
dist = load_true_dist()
# index of nodes
index = np.arange(nodes.shape[0])

bounds = [0, 350, -50, np.inf]
ports = [11, 12]
# ports = np.arange(13)
n_instances = 30
n_prob_stations = [20]  #10, 20, 40, 60, 80, 100]
vessels = [2, 3, 4]
capacity_factor = [62.5, 125, 250]
capacities = [n_prob_stations * 250, n_prob_stations * 275]
home_port_ind = 0


filtered_nodes = degArr2point(nodes[n_ports::2, ]).T
station_list = np.arange(n_stations)
station_list = station_list[(filtered_nodes[:, 0] >= bounds[0]) & (filtered_nodes[:, 0] <= bounds[1]) & (filtered_nodes[:, 1] >= bounds[2]) & (filtered_nodes[:, 1] <= bounds[3])]
station_list, len(station_list)



rng = np.random.default_rng(789123)

for ns in n_prob_stations:
  for nv in vessels:
    for cf in capacity_factor:
      capacities = np.zeros(nv)
      for i in range(nv):
        capacities[i] = (1 + i / 10) * cf * ns

      os.makedirs(os.path.join(DATA_DIR, 'test_problems', str(ns), str(nv),
        str(int(cf * ns))), exist_ok=True)

      for j in np.arange(n_instances):
        station_list_ins = rng.choice(station_list, ns, False)

        vis_ports_stations_simple(nodes, n_ports, ports, station_list_ins, station_labels)


        prob_path = os.path.abspath(os.path.join(DATA_DIR, 'test_problems',
          str(ns), str(nv), str(int(cf * ns)),
          'ins_{0}_{1}_{2}_{3}.txt')).format(ns, nv, int(cf * ns), j + 1)

        with open(prob_path, "w+") as f:
          f.write("PORT_SECTION\n")
          f.write(f"{','.join(map(str, ports))}\n")
          f.write("STATION_SECTION\n")
          f.write(f"{','.join(map(str, station_list_ins))}\n")
          f.write("HOME_PORT_INDEX_SECTION\n")
          f.write(f"{'{0}'.format(home_port_ind)}\n")
          f.write("CAPACITIES_SECTION\n")
          f.write(f"{','.join(map(str, capacities))}")

  # fig, ax = plt.subplots(figsize=[20, 15])

  # plt.plot(*degArr2point(island_data))
  # # plot ports
  # # for idx in range(n_ports):
  # for idx in ports:
  #     plt.scatter(*degArr2point(nodes[[idx]]), c='red')
  #     ax.annotate(idx, degArr2point(nodes[[idx]]))
  # # plot stations
  # for id in station_list: # range(n_stations):
  #     station = nodes[[id * 2 + n_ports, id * 2 + 1 + n_ports], :]
  #     # print(station)
  #     plt.plot(*degArr2point(station), c='blue')
  #     ax.annotate(id, degArr2point(station[[0], :]).flatten())

  # plt.gca().invert_yaxis()
  # plt.show()

