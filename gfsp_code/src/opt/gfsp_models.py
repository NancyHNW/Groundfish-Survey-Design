import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import math
from itertools import permutations, product
from ..util.distance import *
from ..util.data_loader import *
from ..configs import OUT_DIR

class GroundFishSurveyProblem:
    def __init__(self, ns, nv, cf, t_i):
        self.ns = ns  # number of stations
        self.nv = nv  # number of vessels
        self.cf = cf  # capacity factor
        self.t_i = t_i  # instance index
        self.base_capacity = int(cf * ns)
        self.trip_lim = ns * 7.5 + 450
        self.total_lim = 4200 / nv
        self.port_idx = None
        self.stations_ids = None
        self.home_ind = None
        self.catch_capacity = None
        self.nodes = None
        self.n_ports = None
        self.n_stations = None
        self.dist = None
        self.catch = None
        self.int_catch = None
        self.total_catch = None
        self.n_dummy_port = None
        self.vehicles = None
        self.stations = None
        self.customers = None
        self.ifs = None# intermediate facilities i.e. ports
        self.vrp_nodes = None
        self.vrp_dist = None
        self.trips = []

        self._load_data()

    def _load_data(self):
        self.port_idx, self.stations_ids, self.home_ind, self.catch_capacity = load_test_problem(
            self.ns, self.nv, self.base_capacity, self.t_i)
        self.port_idx = np.array(self.port_idx)
        self.nodes, self.n_ports, self.n_stations, station_labels = load_nodes()
        self.dist = load_true_dist()
        self.catch = load_catch_fixed_random_assignment().to_numpy()
        if self.ns == 10:
            self.catch = self.catch / 2
            self.trip_lim *= 1.2
        self.int_catch = np.ceil(self.catch)
        self.total_catch = self.int_catch[self.stations_ids].sum()
        self.n_dummy_port = int(np.ceil(self.total_catch / self.catch_capacity[-1]))

    def optimize(self):
        self.model.update()
        self.model.Params.WorkLimit = 300
        self.model.reset()
        self.model.optimize()
                 
        # self._process_solution()

class ThreeIndexUncompressedModel(GroundFishSurveyProblem):
    def __init__(self, ns, nv, cf, t_i):
        super().__init__(ns, nv, cf, t_i)
        self.model = None

        self.vehicles = np.arange(1, nv + 1)

        self.stations = np.arange(1, ns + 1)
        self.customers = np.arange(1, 2 * ns + 1)
        self.ifs = np.arange(2 * ns + 1, 2 * ns + len(self.port_idx) + 1)
        self.vrp_nodes = np.concatenate(([0], self.customers, self.ifs))

        dist_inds = np.concatenate(([self.port_idx[self.home_ind]],
          np.stack([self.stations_ids * 2,
          self.stations_ids * 2 + 1]).T.reshape(-1) + self.n_ports,
          self.port_idx[self.home_ind:], self.port_idx[:self.home_ind]))

        self.vrp_dist = self.dist[np.ix_(dist_inds,dist_inds)]

        self.arc_catch = {}
        for i in self.vrp_nodes:
          for j in self.vrp_nodes:
            if i in (self.stations * 2 - 1) and j == i + 1:
              self.arc_catch[i, j] = self.int_catch[self.stations_ids[
                int((i - 1)/2)]][0]
            elif i in (self.stations * 2) and j == i - 1:
              self.arc_catch[i, j] = self.int_catch[self.stations_ids[
                int((j - 1)/2)]][0]
            else:
              self.arc_catch[i, j] = 0

        self.Xs = None
        self.ys = None
        self.Ys = None
        self.Fs = None

    def build_model(self):
        self.model = gp.Model("ThreeIndexUncompressed")

        arc_inds = list(product(permutations(self.vrp_nodes, 2), self.vehicles))
        arc_inds = [a + (b,) for a, b in arc_inds]

        self.Xs = self.model.addVars(arc_inds, vtype=GRB.BINARY, name="x")
        self.ys = self.model.addVars(arc_inds, name="y")
        self.Ys = self.model.addVars(arc_inds, name="Y")
        self.fs = self.model.addVars(arc_inds, name="f")

        # Objective
        self.model.setObjective(quicksum(self.vrp_dist[i, j] *
          quicksum(self.Xs[i, j, k] for k in self.vehicles)
          for i in self.vrp_nodes for j in self.vrp_nodes if i != j))

        # Each station must be traversed
        self.model.addConstrs((quicksum(self.Xs[2*i-1, 2*i, k] +
          self.Xs[2*i, 2*i-1, k] for k in self.vehicles) == 1
          for i in self.stations), name="TraverseStation")

        # Each station must be visited once
        self.model.addConstrs((quicksum(self.Xs[i, j, k]
          for i in self.vrp_nodes for k in self.vehicles if i != j) == 1
          for j in self.customers), name="OneVisit")

        # Each vehicle can leave the home port at most once
        self.model.addConstrs((quicksum(self.Xs[0, j, k]
          for j in self.vrp_nodes if j != 0) <= 1
          for k in self.vehicles), name="LeaveHome")

        # Each vehicle that enters a node must leave the node
        self.model.addConstrs((quicksum(self.Xs[i, h, k]
          for i in self.vrp_nodes if i != h) -
          quicksum(self.Xs[h, j, k] for j in self.vrp_nodes if j != h) == 0
          for h in self.vrp_nodes for k in self.vehicles), name="NodeFlow")

        # Trip dist, total dist, and catch are limited by maximums/capacity, 
        # and whether the arc is traversed
        self.model.addConstrs((self.ys[i, j, k] <=
          self.trip_lim * self.Xs[i, j, k] for i in self.vrp_nodes
          for j in self.vrp_nodes if i != j for k in self.vehicles),
          name="TripLimit")
        self.model.addConstrs((self.Ys[i, j, k] <=
          self.total_lim * self.Xs[i, j, k] for i in self.vrp_nodes
          for j in self.vrp_nodes if i != j for k in self.vehicles),
          name="TotalLimit")
        self.model.addConstrs((self.fs[i, j, k] <=
          self.catch_capacity[k - 1] * self.Xs[i, j, k] for i in self.vrp_nodes
          for j in self.vrp_nodes if i != j for k in self.vehicles),
          name="CatchLimit")

        # Trip dist, total dist, and catch start at 0
        self.model.addConstrs((self.ys[i, j, k] == self.vrp_dist[i, j] *
          self.Xs[i, j, k]
          for i in np.concatenate(([0], self.ifs))
          for j in self.vrp_nodes if i != j
          for k in self.vehicles),
          name="TripStart")
        self.model.addConstrs((self.Ys[0, j, k] == self.vrp_dist[0, j] *
          self.Xs[0, j, k]
          for j in self.vrp_nodes if j != 0
          for k in self.vehicles),
          name="TotalStart")
        self.model.addConstrs((self.fs[i, j, k] == 0
          for i in np.concatenate(([0], self.ifs))
          for j in self.vrp_nodes if i != j
          for k in self.vehicles),
          name="CatchStart")

        # Trip dist, total dist, and catch are conserved at the nodes
        self.model.addConstrs((quicksum(self.ys[h, j, k]
          for j in self.vrp_nodes if j != h) -
          quicksum(self.ys[i, h, k] + self.vrp_dist[h, i] *
          self.Xs[h, i, k] for i in self.vrp_nodes if i != h) == 0
          for h in self.customers for k in self.vehicles), name="TripFlow")
        self.model.addConstrs((quicksum(self.Ys[h, j, k]
          for j in self.vrp_nodes if j != h) -
          quicksum(self.Ys[i, h, k] + self.vrp_dist[i, h] *
          self.Xs[i, h, k] for i in self.vrp_nodes if i != h) == 0
          for h in np.concatenate((self.customers, self.ifs))
          for k in self.vehicles), name="TotalFlow")
        self.model.addConstrs((quicksum(self.fs[h, j, k]
          for j in self.vrp_nodes if j != h) -
          quicksum(self.fs[i, h, k] + self.arc_catch[i, h] *
          self.Xs[i, h, k] for i in self.vrp_nodes if i != h) == 0
          for h in self.customers for k in self.vehicles), name="CatchFlow")

    def _process_solution(self):

      xVals = self.model.getAttr('X', self.Xs)
      xVals = {k for (k,v) in xVals.items() if v > .5}

      leave_port = [s for s in xVals if s[0] in
        np.concatenate(([0], self.ifs))]

      for i, boat in enumerate(leave_port):

        next_node_id = boat[1]
        trip = [boat[0]]

        not_home = 1
        trip_num = 1

        while not_home:
          if next_node_id in np.concatenate(([0], self.ifs)):
            not_home = 0
          trip.append(next_node_id)

          next_node_id = [s for s in xVals if s[0] == next_node_id][0][1]

        self.trips.append([boat[2], trip, 0, 0])

      for t in self.trips:
        trip_catch = 0
        trip_dist = 0
        for i, n in enumerate(t[1]):
          if i < len(t[1]) - 1:
            trip_catch += self.arc_catch[n, t[1][i + 1]]
            trip_dist += self.vrp_dist[n, t[1][i + 1]]

        t[2] = trip_catch
        t[3] = trip_dist

      for t in self.trips:
        for i, n in enumerate(t[1]):
          if n == 0:
            t[1][i] = self.port_idx[self.home_ind]
          elif n in self.customers:
            if n % 2 == 1:
              # Start of a station
              t[1][i] = self.stations_ids[int((n - 1) / 2)] * 2 + self.n_ports
              # print('Test')
            else:
              # End of a station
              t[1][i] = self.stations_ids[int((n - 2) / 2)] * 2 + self.n_ports + 1
              # print('Test')
          else:
            # Must be a port
            t[1][i] = self.port_idx[n - self.ns * 2 - 1]

      print('Test')

      #   while not_home:
      #     if next_node_id in port_dummy_idx:
      #       if next_node_id < dummy_start:
      #         actual_port = next_node_id

      #         if next_node_id == home_port:
      #           not_home = 0
      #       else:
      #         actual_port = port_dummy_idx[math.floor((next_node_id - 
      #           dummy_start)/n_dummy_port)]

      #       trip.append(actual_port)

      #       if len(trip) > 2 or trip[0] != trip[1]:
      #         trips.append([i + 1, trip_num, trip, 0])
      #         trip_num += 1
              
      #       trip = [actual_port]

      #     else:
      #       trip.append(next_node_id)

      #     next_node_id = [s for s in xVals if s[0] == next_node_id][0][1]

      # for t in trips:
      #   trip_catch = 0
      #   for i, n in enumerate(t[2]):
      #     if n not in port_dummy_idx:
      #       trip_catch += (catch[int((n-13)/2)] / 2)[0]

      #   t[3] = trip_catch

      return trips

