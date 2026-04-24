import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import math
from itertools import permutations, product
from .gfsp_models import GroundFishSurveyProblem

class TwoIndexCompressedModel(GroundFishSurveyProblem):
    def __init__(self, ns, nv, cf, t_i):
        super().__init__(ns, nv, cf, t_i)
        self.model = None

        self.vehicles = np.arange(1, nv + 1)

        self.stations = np.arange(1, ns + 1)
        self.customers = np.arange(1, 2 * ns + 1)
        self.ifs = np.arange(2 * ns + 1, 2 * ns + len(self.port_idx) *
          self.n_dummy_port + 1)
        self.vrp_nodes = np.concatenate(([0], self.customers, self.ifs))
        self.port_stations = np.concatenate(([0], self.stations, self.ifs))

        dist_inds = np.concatenate(([self.port_idx[self.home_ind]],
          np.stack([self.stations_ids * 2,
          self.stations_ids * 2 + 1]).T.reshape(-1) + self.n_ports,
          np.repeat(np.concatenate((self.port_idx[self.home_ind:], self.port_idx[:self.home_ind])), self.n_dummy_port)))

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

        self.node_catch = {}
        for i in self.vrp_nodes:
          if i in (self.stations * 2 - 1):
            self.node_catch[i] = self.int_catch[self.stations_ids[
                int((i - 1)/2)]][0]
          else:
            self.node_catch[i] = 0

        self.station_catch = {}
        for i in self.stations:
          self.station_catch[i] = self.int_catch[self.stations_ids[i - 1]][0]

        self.Xs = None
        self.Zs = None
        self.ds = None
        self.ys = None
        self.Ys = None
        self.Fs = None

    def build_model(self):
      self.model = gp.Model("TwoIndexCompressed")

      node_arc_inds = list(permutations(self.vrp_nodes, 2))

      station_arc_inds = list(permutations(self.port_stations, 2))

      self.Xs = self.model.addVars(node_arc_inds, vtype=GRB.BINARY, name="x")
      self.Zs = self.model.addVars(station_arc_inds, vtype=GRB.BINARY,
        name="z")

      self.ds = self.model.addVars(station_arc_inds, name="d")  
      self.ys = self.model.addVars(station_arc_inds, name="y")
      self.Ys = self.model.addVars(station_arc_inds, name="Y")
      self.fs = self.model.addVars(station_arc_inds, name="f")

      v_inds = [(k, i) for k in self.vehicles for i in self.port_stations
        if i != 0]
      self.Vs = self.model.addVars(v_inds, vtype=GRB.BINARY, name="v")
      self.es = self.model.addVars(station_arc_inds, name="e")

      # Objective
      self.model.setObjective(quicksum(self.vrp_dist[i, j] * self.Xs[i, j]
        for i in self.vrp_nodes for j in self.vrp_nodes if i != j))

      # Each station must be visited once
      self.model.addConstrs((quicksum(self.Zs[i, j]
        for j in self.port_stations if i != j) == 1
        for i in self.stations), name="OneVisitStation")

      # Each dummy port can be visited once
      self.model.addConstrs((quicksum(self.Zs[i, j]
        for j in self.port_stations if i != j) <= 1
        for i in self.ifs if i != 0), name="OneVisitPort")

      # Each station must be traversed
      self.model.addConstrs((quicksum(self.Zs[i, j]
        for j in self.port_stations if i != j) == self.Xs[2*i-1, 2*i] +
        self.Xs[2*i, 2*i-1]
        for i in self.stations), name="TraverseStation")

      # Each vehicle can leave the home port at most once
      self.model.addConstrs((quicksum(self.Xs[0, j]
        for j in self.vrp_nodes if j != 0) <= self.nv
        for k in self.vehicles), name="LeaveHomeX")
      self.model.addConstrs((quicksum(self.Zs[0, j]
        for j in self.port_stations if j != 0) <= self.nv
        for k in self.vehicles), name="LeaveHomeZ")

      # Each vehicle can leave to at most one station
      self.model.addConstrs((quicksum(self.Vs[k, j]
        for j in self.port_stations if j != 0) <= 1
        for k in self.vehicles), name="VehicleLeave")

      # Each node is left to by at most one vehicle
      self.model.addConstrs((quicksum(self.Vs[k, j]
        for k in self.vehicles) <= 1
        for j in self.port_stations if j != 0), name="NodeLeave")

      # Each vehicle that enters a node must leave the node
      self.model.addConstrs((quicksum(self.Xs[i, h]
        for i in self.vrp_nodes if i != h) -
        quicksum(self.Xs[h, j] for j in self.vrp_nodes if j != h) == 0
        for h in self.vrp_nodes), name="NodeFlowX")
      self.model.addConstrs((quicksum(self.Zs[i, h]
        for i in self.port_stations if i != h) -
        quicksum(self.Zs[h, j] for j in self.port_stations if j != h) == 0
        for h in self.port_stations), name="NodeFlowZ")

      # Define the z variables
      self.model.addConstrs((self.Zs[i, j] == self.Xs[i, j]
        for i in np.concatenate(([0], self.ifs))
        for j in np.concatenate(([0], self.ifs))
        if i != j), name="Compress_pp")

      self.model.addConstrs((self.Zs[i, j] == 
        self.Xs[2*i-1,j] + self.Xs[2*i,j]
        for i in self.stations for j in np.concatenate(([0], self.ifs))),
        name="Compress_sp")

      self.model.addConstrs((self.Zs[i, j] ==
        self.Xs[i, 2*j-1] + self.Xs[i , 2*j]
        for i in np.concatenate(([0], self.ifs)) for j in self.stations),
        name="Compress_ps")

      self.model.addConstrs((self.Zs[i, j] ==
        self.Xs[2*i-1, 2*j-1] +
        self.Xs[2*i, 2*j-1] +
        self.Xs[2*i-1 , 2*j] +
        self.Xs[2*i , 2*j]        
        for i in self.stations for j in self.stations
        if i != j), name="Compress_ss")

      # Trip dist, total dist, and catch are limited by maximums/capacity, 
      # and whether the arc is traversed
      self.model.addConstrs((self.ys[i, j] <=
        self.trip_lim * self.Zs[i, j] for i in self.port_stations
        for j in self.port_stations if i != j),
        name="TripLimit")
      self.model.addConstrs((self.Ys[i, j] <=
        self.total_lim * self.Zs[i, j] for i in self.port_stations
        for j in self.port_stations if i != j),
        name="TotalLimit")
      self.model.addConstrs((self.fs[i, j] <= self.es[i, j]
        for i in self.port_stations
        for j in self.port_stations if i != j),
        name="CatchLimit")
      self.model.addConstrs((self.ds[i, j] <=
        self.total_lim * self.Zs[i, j] for i in self.port_stations
        for j in self.port_stations if i != j),
        name="DistLimit")
      maxCap = max(self.catch_capacity)
      self.model.addConstrs((self.es[i, j] <=
        maxCap * self.Zs[i, j] for i in self.port_stations
        for j in self.port_stations if i != j),
        name="CapacityLimit")

      # Trip dist, total dist, and catch start at 0
      self.model.addConstrs((self.ys[i, j] == self.ds[i, j]
        for i in np.concatenate(([0], self.ifs))
        for j in self.port_stations if i != j),
        name="TripStart")
      self.model.addConstrs((self.Ys[0, j] == self.ds[0, j]
        for j in self.port_stations if j != 0),
        name="TotalStart")
      self.model.addConstrs((self.fs[i, j] == 0
        for i in np.concatenate(([0], self.ifs))
        for j in self.port_stations if i != j),
        name="CatchStart")

      # Trip dist, total dist, and catch are conserved at the nodes
      self.model.addConstrs((quicksum(self.ys[h, j]
        for j in self.port_stations if j != h) -
        quicksum(self.ys[i, h] + self.ds[h, i] for i in self.port_stations if i != h) == 0
        for h in self.stations), name="TripFlow")
      self.model.addConstrs((quicksum(self.Ys[h, j]
        for j in self.port_stations if j != h) -
        quicksum(self.Ys[i, h] + self.ds[h, i] for i in self.port_stations if i != h) == 0
        for h in np.concatenate((self.stations, self.ifs))), name="TotalFlow")

      # self.model.addConstrs((quicksum(self.fs[h, j, k]
      #   for j in self.port_stations if j != h) -
      #   quicksum(self.fs[i, h, k] + self.arc_catch[i, h] *
      #   self.Xs[i, h, k] for i in self.port_stations if i != h) == 0
      #   for h in self.customers for k in self.vehicles), name="CatchFlow")

      self.model.addConstrs((quicksum(self.fs[h, j]
        for j in self.port_stations if j != h) -
        quicksum(self.fs[i, h] for i in self.port_stations if i != h) ==
        self.station_catch[h]
        for h in self.stations), name="CatchFlow")

      self.model.addConstrs((quicksum(self.es[h, j]
        for j in self.port_stations if j != h) -
        quicksum(self.es[i, h] for i in self.port_stations if i != h) +
        quicksum(self.catch_capacity[k - 1] * self.Vs[k, h]
        for k in self.vehicles) == 0
        for h in self.stations ), name="CapacityFlow")

      # Define the d variables
      self.model.addConstrs((self.ds[i, j] == self.vrp_dist[i, j] *
        self.Xs[i, j]
        for i in np.concatenate(([0], self.ifs))
        for j in np.concatenate(([0], self.ifs))
        if i != j), name="Dist_pp")

      self.model.addConstrs((self.ds[i, j] == 
        (self.vrp_dist[2*i-1,j] + self.vrp_dist[2*i,2*i-1]) *
        self.Xs[2*i-1,j] +
        (self.vrp_dist[2*i,j] + self.vrp_dist[2*i-1,2*i]) * self.Xs[2*i,j]
        for i in self.stations for j in np.concatenate(([0], self.ifs))),
        name="Dist_sp")

      self.model.addConstrs((self.ds[i, j] ==
        self.vrp_dist[i, 2*j-1] * self.Xs[i, 2*j-1] +
        self.vrp_dist[i, 2*j] * self.Xs[i , 2*j]
        for i in np.concatenate(([0], self.ifs)) for j in self.stations), 
        name="Dist_ps")

      self.model.addConstrs((self.ds[i, j] ==
        self.vrp_dist[2*i-1, 2*j-1] * self.Xs[2*i-1, 2*j-1] +
        self.vrp_dist[2*i, 2*j-1] * self.Xs[2*i, 2*j-1] +
        self.vrp_dist[2*i-1, 2*j] * self.Xs[2*i-1 , 2*j] +
        self.vrp_dist[2*i, 2*j] * self.Xs[2*i , 2*j] +
        self.vrp_dist[2*i-1, 2*i] * (self.Xs[2*i, 2*j-1] +
        self.Xs[2*i, 2*j]) +
        self.vrp_dist[2*i, 2*i-1] * (self.Xs[2*i-1, 2*j-1] +
        self.Xs[2*i-1, 2*j])
        for i in self.stations for j in self.stations
        if i != j), name="Dist_ss")
    

    def _process_solution(self):

      xVals = self.model.getAttr('X', self.Xs)
      xVals = {k for (k,v) in xVals.items() if v > .5}

      for k in xVals:
        print(k)

      dVals = self.model.getAttr('X', self.ds)
      dVals = {k:v for (k,v) in dVals.items() if v > 0.005}

      for k in dVals.keys():
        print(k, dVals[k])

      YVals = self.model.getAttr('X', self.Ys)
      YVals = {k:v for (k,v) in YVals.items() if v > 0.005}

      for k in YVals.keys():
        print(k, YVals[k])

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