import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ..configs import OUT_DIR
from .data_loader import *
from .distance import *

try:
  from mpl_toolkits.basemap import Basemap
except ImportError:
  Basemap = None

def vis_ports_stations(nodes, n_ports, ports_idx, station_idx):

  if Basemap is None:
    raise ImportError(
      "Basemap is not installed. Install it or use vis_ports_stations_simple()."
    )

  # m = Basemap(projection='cyl', resolution='c', llcrnrlat=62.3,urcrnrlat=66.7,
  #           llcrnrlon=-18.2,urcrnrlon=-10.4)

  m = Basemap(projection='cyl', resolution='c', llcrnrlat=62.3,urcrnrlat=68,
            llcrnrlon=-27.8,urcrnrlon=-10)

  m.arcgisimage(dpi=1000)

  for p in ports_idx:
    xpt, ypt = m(-nodes[p][1], nodes[p][0])
    if p == 11:
      m.plot(xpt,ypt,c='#4b4be5', marker='o')
    else:
      m.plot(xpt,ypt,c='#22A884', marker='o')
    # plt.text(xpt, ypt, p)

  for s in station_idx:
    xpt1, ypt1 = m(-nodes[s * 2 + n_ports][1], nodes[s * 2 + n_ports][0])
    xpt2, ypt2 = m(-nodes[s * 2 + n_ports + 1][1], nodes[s * 2 + n_ports + 1][0])
    m.plot([xpt1, xpt2], [ypt1, ypt2], c='black', linewidth=5, alpha=0.5, 
      solid_capstyle='round')
    # plt.text(xpt1, ypt1, s * 2 + n_ports)

  # save_file = os.path.abspath(os.path.join(OUT_DIR, 'example_20.pdf'))

  # plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
  # plt.savefig(save_file,
  #   orientation='landscape', transparent=True, dpi=1000, format='pdf')

  # plt.show()

  return m

def vis_ports_stations_simple(nodes, n_ports, ports_idx, station_idx, station_labels):

  # Load island data
  island_data = load_island_data()

  fig, ax = plt.subplots(figsize=(20, 15))
  # Plot outline of Iceland
  ax.plot(*degArr2point(island_data), color="k", linewidth=0.25)

  for p in ports_idx:
    xpt, ypt = degArr2point(nodes[[p]])
    if p == 3:
      ax.scatter(xpt,ypt,c='#4b4be5', marker='o', s=500)
    else:
      ax.scatter(xpt,ypt,c='#22A884', marker='o', s=500)
    # plt.text(xpt, ypt, p)

  for i, s in enumerate(station_idx):
    xpt1, ypt1 = degArr2point(nodes[[s * 2 + n_ports]])
    xpt2, ypt2 = degArr2point(nodes[[s * 2 + n_ports + 1]])
    ax.plot([xpt1, xpt2], [ypt1, ypt2], c='red', linewidth=5, alpha=0.5, 
      solid_capstyle='round')
    # plt.text(xpt1, ypt1, station_labels[i])

  plt.gca().invert_yaxis()

  save_file = os.path.abspath(os.path.join(OUT_DIR, 'example_80.pdf'))

  plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
  plt.savefig(save_file,
    orientation='landscape', transparent=True, dpi=1000, format='pdf')

  
  # plt.show()
  
  return fig, ax

def vis_sol(sol, nodes, n_ports, port_idx, stations_ids, save_file):

  m = vis_ports_stations(nodes, n_ports, port_idx, stations_ids)

  # boat_colors = ["#E54B4B", "#E5984B"]
  boat_colors = list(mcolors.TABLEAU_COLORS.values())
  trip_styles = ['solid', 'dotted', 'dashed', 'dashdot']
  trip_styles = [(0, ()), (0, (1, 20)), (0, (5, 30)), (0, (3, 40, 1, 50))]

  for t in sol:
    tripX = []
    tripY = []
    for trip_node in t[2]:
      xpt, ypt = m(-nodes[trip_node][1], nodes[trip_node][0])
      tripX.append(xpt)
      tripY.append(ypt)
      
      c_ind = t[1] % 10
      l_ind = (t[1] // 10) % 4

      m.plot(tripX, tripY, c=boat_colors[c_ind],
        linestyle=trip_styles[l_ind])

  plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
  plt.savefig(save_file, orientation='landscape', transparent=True, dpi=1000,
    format='pdf')

  plt.close()

def vis_sol_simple(sol, nodes, n_ports, port_idx, stations_ids, station_labels, save_file):

  fig, ax = vis_ports_stations_simple(nodes, n_ports, port_idx, stations_ids, station_labels)

  # boat_colors = ["#E54B4B", "#E5984B"]
  boat_colors = list(mcolors.TABLEAU_COLORS.values())
  trip_styles = ['solid', 'dotted', 'dashed', 'dashdot']
  trip_styles = [(0, ()), (0, (4, 10)), (0, (10, 10)), (0, (4, 10, 10, 10))]

  for t in sol:
    tripX = []
    tripY = []
    for trip_node in t[2]:
      xpt, ypt = degArr2point(nodes[[trip_node]])
      tripX.append(xpt[0])
      tripY.append(ypt[0])
      
      c_ind = t[0]
      l_ind = t[1] % 4 # (t[1] // 10) % 4

      ax.plot(tripX, tripY, c=boat_colors[c_ind],
        linestyle=trip_styles[l_ind], linewidth=0.15)

  plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
  fig.set_size_inches(4, 3)

  ax.set_xlim(-25, 339.1)
  ax.set_ylim(-75, 198.9)

  plt.gca().invert_yaxis()
  plt.savefig(save_file, orientation='landscape', transparent=False, dpi=400,
    format='png')

  plt.close()


# plt.plot(*degArr2point(island_data))
# # plot ports
# for idx in ports_idx:
#     plt.scatter(*degArr2point(nodes[[idx]]), label=str(idx))
# # plot stations
# for id in stations_ids:
#     station = nodes[[id * 2 + n_ports, id * 2 + 1 + n_ports], :]
#     plt.plot(*degArr2point(station))
#     # plt.annotate(id, degArr2point(station[[0], :]).flatten())
# plt.legend()
# plt.gca().invert_yaxis()
# plt.show()

# print(stations_ids)
# vis_ports_stations(island_data, nodes, n_ports, ports_idx, stations_ids, True,
#     bounds=(0, 300, -50, 200))
# plt.show()