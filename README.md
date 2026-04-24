# Groundfish Survey Design Optimization

This repository contains code and data for optimising the design of groundfish surveys. 

---

## Project Structure

The repository is organized into three main directories:

- **`final_code/`**: A Python-based heuristic solver. It includes modules for:
    - Ant colony optimization
    - TSP solving
    - Neighbourhood search rules
    - Data loading and problem generation
    - GUI functions for visualization

- **`gfsp_code/`**: A collection of Mixed-Integer Programming (MIP) formulations and the LKH (LKH-3) solver for the Groundfish Survey Problem (GFSP). It contains:
    - The LKH executable
    - Data files for ports, catch, and station locations
    - A large set of test problems organized by size, number of vessels, and capacity.

- **`unified/`**: A unified adapter layer that allows running both the heuristic solvers from `final_code` and the MIP/LKH solvers from `gfsp_code` through a single command-line interface. This is the recommended way to interact with the solvers.

---

## Dependencies

- Python 3
- Gurobi (with a valid license)
- `numpy`
- `matplotlib`

---

## Test Problems

The repository includes two sets of test problems:

1.  **`gfsp_code` problems**: 1,620 instances, organized by `ns/nv/capacity/instance`.
2.  **`final_code` problems**: 36 instances with varying parameters, located in `final_code/better_test_problems/`.