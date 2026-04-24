"""Improved TSP helper backed by Gurobi.

This module contains a single function `solve_tsp` that builds and solves
a (variant of) Traveling Salesman Problem using Gurobi. The function uses
lazy subtour-elimination constraints and supports an optional time limit
variant used by the calling project.

Notes:
- The code contains some project-specific thresholds (e.g. the value 13)
  and assumptions about node indexing. Those were left as-is but are
  annotated below where they appear.
"""

import gurobipy as gp
from gurobipy import GRB
from itertools import combinations, permutations
import numpy as np


def solve_tsp(path, travel_times, time_limit=None, initial_solution=False):
    """Solve a TSP-like routing problem and return a route.

    Parameters
    - path: list[int]
        A list of node indices representing a (possibly partial) path. The
        function uses the nodes appearing in `path` as the problem nodes.
        NOTE: `nodes = list(set(path))` is used below which discards ordering
        — the original `path` order is only kept for start/end hints.
    - travel_times: array-like or dict-like accessed as travel_times[i, j]
        Matrix or mapping giving travel time / cost between nodes i and j.
    - time_limit: float or None
        if no fish time limit to be used then None, otherwise the fish time limit
    - initial_solution: bool
        If True, the provided `path` is used to seed the MIP start.

    Returns
    - route: list[int]
        A route (sequence of node indices). If the model is infeasible the
        original `path` is returned.
    """

    # Use unique nodes from the provided path. This loses ordering but keeps
    # the set of nodes to solve over. The original path's first/last are
    # used later as hints/terminators.
    nodes = list(set(path))

    # Two formulation branches are used in the codebase:
    # - time_limit is None: compact/undirected version (only i<j stored)
    # - time_limit provided: directed version (both (i,j) pairs used)
    # The difference changes variable creation and degree constraints.
    if time_limit is None:

        # Define pairwise costs only for i<j to avoid duplicate symmetric
        # entries; later we create symmetric aliases for the variables.
        time = {(i, j): travel_times[i, j] for i in nodes for j in nodes if i != j and i < j}

        m = gp.Model()

        # Binary variable x[i,j] = 1 if edge (i,j) appears in solution.
        vars = m.addVars(time.keys(), obj=time, vtype=GRB.BINARY, name='x')

        # Create symmetric mapping so vars[(j,i)] references the same var
        # as vars[(i,j)]. This keeps the model representing undirected edges.
        vars.update({(j, i): vars[i, j] for i, j in vars.keys()})

        # Degree constraint: in undirected formulation each node has degree 2
        m.addConstrs(vars.sum(c, '*') == 2 for c in nodes)

        # ensure station pairs are kept together
        m.addConstrs(vars[i, j] == 1 for i, j in vars.keys() if (i % 2 == 0 and j == i - 1) or (i % 2 == 1 and j == i + 1))

        # If the provided path isn't already a closed loop, add a constraint
        # that links the end back to the start.
        if path[0] != path[-1]:
            m.addConstr(vars[path[-1], path[0]] == 1)

    else:
        # Directed-style formulation: include both (i,j) and (j,i) entries.
        time = {(i, j): travel_times[i, j] for i in nodes for j in nodes if i != j}

        m = gp.Model()

        # Binary x[i,j] in the directed model
        vars = m.addVars(time.keys(), obj=time, vtype=GRB.BINARY, name='x')

        # Degree constraints in directed form: one outgoing and one incoming
        # arc for each node (i.e., sum of row and column equals 1).
        m.addConstrs(vars.sum(c, '*') == 1 for c in nodes)
        m.addConstrs(vars.sum('*', c) == 1 for c in nodes)

        # Preserve the project-specific adjacency constraints here as well
        m.addConstrs(vars[i, j] + vars[j, i] == 1 for i, j in vars.keys() if (i % 2 == 0 and j == i - 1) or (i % 2 == 1 and j == i + 1))

        if path[0] != path[-1]:
            m.addConstr(vars[path[-1], path[0]] + vars[path[0], path[-1]] == 1)

        # fish time limit constraint
        m.addConstr(gp.quicksum(time[i, j] * vars[i, j] for i, j in vars.keys() if i > 12) <= time_limit, name="DistanceLimit")

    # Subtour elimination callback
    def subtourelim(model, where):
        """Lazy-constraint callback to cut off subtours found in integer sols.

        When Gurobi finds a new MIP solution (MIPSOL) we inspect the selected
        edges and, if they form a subtour smaller than the full set of
        nodes, add a lazy constraint to forbid that subtour.
        """
        if where == GRB.Callback.MIPSOL:
            # cbGetSolution expects a dict-like of variables
            vals = model.cbGetSolution(model._vars)
            # Build a tuplelist of selected edges (those with value ~ 1).
            selected = gp.tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)

            tour = subtour(selected)
            # If the tour doesn't visit all nodes add a lazy cut. For the
            # undirected/compact form we sum over combinations, for the
            # directed form we use permutations. This mirrors the two
            # formulation styles used above.
            if len(tour) < len(nodes):
                if time_limit is None:
                    model.cbLazy(gp.quicksum(model._vars[i, j]
                                            for i, j in combinations(tour, 2))
                                <= len(tour) - 1)
                else:
                    model.cbLazy(gp.quicksum(model._vars[i, j]
                                        for i, j in permutations(tour, 2))
                            <= len(tour) - 1)


    # Find the shortest subtour in the solution
    def subtour(edges):
        """Return the shortest cycle (list of nodes) found in 'edges'.

        The implementation does a simple traversal of unvisited nodes to find
        cycles. It's not the most optimized routine, but it is clear and
        sufficient for the (typically small) subtours that appear during
        callback checks.
        """
        unvisited = nodes[:]
        cycle = nodes[:]
        while unvisited:
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(thiscycle) <= len(cycle):
                cycle = thiscycle
        return cycle

    # Wire model variables into callback and set lazy-constraint mode
    m._vars = vars
    m.setParam('OutputFlag', 0)
    m.Params.LazyConstraints = 1

    # Seed an initial solution if requested (simple MIP start using provided path)
    if initial_solution:
        for i in range(len(path)-1):
            vars[path[i], path[i+1]].start = 1

    # Optimize with our callback that will add subtour-elimination cuts lazily
    m.optimize(subtourelim)

    # If infeasible, return the given path as a fallback
    if m.Status == GRB.INFEASIBLE:
        return path

    # Retrieve selected edges from the solved model
    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

    def create_route(selected):
        """Construct a route list from selected edges.

        The routing logic here follows the project's original behavior: it
        starts at path[0] and walks the selected edges forward, preferring
        nodes that satisfy a project-specific index threshold (>=13).
        """
        route = [path[0]]
        # Walk until we've collected all nodes
        while len(route) < len(nodes):
            for i, j in selected:
                # Prefer edges that originate at the last visited node and
                # lead to a not-yet-visited node. The '(i >= 13 or j >= 13)'
                # guard is project-specific and preserved.
                if i == route[-1] and j not in route and (i >= 13 or j >= 13):
                    route.append(j)
                    break
                elif j == route[-1] and i not in route and (i >= 13 or j >= 13):
                    route.append(i)
                    break
                else:
                    continue
        # If the route ends with a node in the project-specific range,
        # append the original path's last node as a terminator.
        if route[-1] >= 13:
            route.append(path[-1])

        return route

    new_route = create_route(selected)

    return new_route

