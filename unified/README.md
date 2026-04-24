# Unified Adapter Layer

Run heuristic solvers or Kun Er's MIP formulations on the same test problems.

## Quick Start

All commands run from the `StartingMaterials/` directory.

```bash
cd "StartingMaterials"
```

### Run a single heuristic on a gfsp test problem

```bash
python -m unified.run_example --ns 20 --nv 2 --cf 62.5 --instance 1 --method tabu_move --init greedy --time-limit 15
```

### Compare multiple methods

```bash
python -m unified.run_example --ns 20 --nv 2 --cf 62.5 --instance 1 --compare --time-limit 10
```

---

## CLI Arguments

| Argument | Values | Default | Description |
|---|---|---|---|
| `--ns` | 10, 20, 40, 60, 80, 100 | 20 | Number of stations in the test problem |
| `--nv` | 2, 3, 4 | 2 | Number of vessels (boats) |
| `--cf` | 62.5, 125, 250 | 62.5 | Capacity factor (capacity = cf * ns) |
| `--instance` | 1-30 | 1 | Test problem instance number |
| `--init` | `greedy`, `grasp` | `greedy` | Initial solution construction method |
| `--method` | See table below | `grasp` | Neighbourhood search method |
| `--time-limit` | any float (seconds) | 10 | How long to run the improvement loop |
| `--compare` | flag | off | Run grasp_only, grasp, and sa side by side |

---

## Available Methods

### Initial Solution (`--init`)

| Value | Description | Feasibility |
|---|---|---|
| `greedy` | Nearest-station greedy with strict capacity/time checks | Always feasible |
| `grasp` | GRASP (probabilistic RCL + soft capacity via return-to-port probability) | Often infeasible, relies on improvement phase |

### Neighbourhood Search (`--method`)

| Value | Full Name | What it Does | Best For |
|---|---|---|---|
| `grasp_only` | No improvement | Only runs initial construction, no neighbourhood search | Baseline / speed |
| `grasp` | Next-descent swap | Randomly picks station pairs and swaps between trips; accepts first improvement | Fast improvement |
| `tabu_swap` | Tabu search (swaps) | Steepest-descent swaps with a tabu list to escape local optima | Better quality, slower |
| `tabu_move` | Tabu search (moves) | Moves stations between trips/boats with tabu list; can fix capacity violations | Best for capacity-tight problems |
| `sa` | Simulated annealing | Random swaps/moves with Metropolis acceptance; explores broadly | Escaping local optima |

All methods except `grasp_only` also run TSP optimisation (via Gurobi) on each trip to improve station ordering within trips.

---

## Available Test Problems

### gfsp_code problems (1,620 total)

Organised by `ns/nv/capacity/instance`:

| Stations (ns) | Vessels (nv) | Capacity factors (cf) | Instances |
|---|---|---|---|
| 10 | 2, 3, 4 | 62.5, 125, 250 | 1-30 each |
| 20 | 2, 3, 4 | 62.5, 125, 250 | 1-30 each |
| 40 | 2, 3, 4 | 62.5, 125, 250 | 1-30 each |
| 60 | 2, 3, 4 | 62.5, 125, 250 | 1-30 each |
| 80 | 2, 3, 4 | 62.5, 125, 250 | 1-30 each |
| 100 | 2, 3, 4 | 62.5, 125, 250 | 1-30 each |

Plus a full 581-station problem (`full_problem`).

### the heuristic code's problems (36 total)

Problem IDs 0-35 in `final_code/better_test_problems/`, with varying station counts (3-200), boat counts (1-4), and capacities.

---

## Example Commands

```bash
# Small problem, fast
python -m unified.run_example --ns 10 --nv 2 --cf 62.5 --instance 1 --method grasp --time-limit 5

# Medium problem, tabu search
python -m unified.run_example --ns 40 --nv 3 --cf 125 --instance 5 --method tabu_move --time-limit 30

# Large problem, simulated annealing
python -m unified.run_example --ns 100 --nv 4 --cf 250 --instance 1 --method sa --time-limit 60

# Compare methods on a 20-station problem
python -m unified.run_example --ns 20 --nv 2 --cf 62.5 --instance 1 --compare --time-limit 10

# Use GRASP initialisation instead of greedy
python -m unified.run_example --ns 20 --nv 2 --cf 62.5 --instance 1 --method tabu_move --init grasp --time-limit 15
```

---

## Using the API Directly

```python
from unified import load_gfsp_problem, heuristic_context, to_heuristic_problem

# Load any gfsp test problem
inst = load_gfsp_problem(ns=20, nv=2, cf=62.5, instance=1)

# Run heuristics
with heuristic_context():
    from classes import Problem, Trip
    from neighbourhood_rules import tabu_search_move

    prob = to_heuristic_problem(inst, catch_source="gfsp")
    prob.generate_initial_solution(seed=0)

    # Save/restore cycle (required before neighbourhood search)
    sol = prob.save_solution_as_list()
    prob.restore_solution_from_list(sol)

    # Improve
    for boat in prob.boats:
        boat.improve_route(prob)
    for boat in prob.boats:
        if len(boat.route) == 1:
            boat.route.append(Trip(boat.home_port, [], boat.home_port))

    obj, pen, cost = tabu_search_move(prob, k_catch=100, k_fish=1,
        upper_bound=10000, work_limit=inst.work_limit,
        history_size=4, max_iter=10, max_time=60, tsp=0.02)

    prob.display_solution()
```

### Loading the heuristic code's own test problems

```python
from unified import load_heuristic_problem, heuristic_context, to_heuristic_problem

inst = load_heuristic_problem(12)  # problem ID 12
with heuristic_context():
    prob = to_heuristic_problem(inst)  # uses the heuristic code's catch data by default
    prob.generate_initial_solution(seed=0)
    prob.display_solution()
```

---

## Module Reference

| Module | Key Functions |
|---|---|
| `unified.problem` | `ProblemInstance` dataclass, `station_ids_to_node_indices()`, `node_indices_to_station_ids()` |
| `unified.loaders` | `load_gfsp_problem()`, `load_heuristic_problem()`, `load_gfsp_full_problem()`, `list_gfsp_problems()`, `list_heuristic_problems()` |
| `unified.adapters` | `heuristic_context()`, `gfsp_context()`, `to_heuristic_problem()`, `to_gfsp_model()`, `override_catch_data()` |
| `unified.evaluate` | `evaluate_heuristic_solution()`, `solution_to_trips()`, `compute_total_distance()`, `compute_total_time()`, `compare_results()` |

---

## Notes

- **Distance vs Time**: heuristic solvers optimise travel *time* (`true_time.npy`); Kun Er's MIPs optimise *distance* (`true_dist.npy`). Objectives from different solvers are not directly comparable.
- **Catch data**: Use `catch_source="gfsp"` when running heuristics on gfsp test problems (capacities were designed around `LimitedCatch.csv`). Use `catch_source="heuristic"` (default) for the heuristic code's own test problems.
- **Gurobi license**: The TSP improvement step and MIP formulations require a Gurobi license.
- **Save/restore before neighbourhood search**: Always do `prob.save_solution_as_list()` then `prob.restore_solution_from_list(sol)` after initial construction and before running neighbourhood rules. This fixes an internal indexing inconsistency.
