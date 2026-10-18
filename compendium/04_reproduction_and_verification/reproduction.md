# Reproduction guide

## Prerequisites

Use Python 3.12, the package versions listed in the repository
[`README.md`](../../README.md), and a licensed Gurobi installation. Run
commands from the repository root.

## Basic verification

```powershell
python -m pytest
```

## Small demonstration

Use the saved 100-station plan for a quick route-scoring demonstration:

```powershell
python -m experiments.monte_carlo `
    --ns 100 --nv 2 --cf 125 `
    --plan tests/plans/ns100_default.json
```

## Full-survey example

```powershell
python -m experiments.overflow_map `
    --full `
    --home-ports 0 6 10 12 `
    --time-limit 300 `
    --n-scenarios 1000 `
    --plan tests/plans/full_default.json
```

## Travel-matrix rebuild

```powershell
python -m unified.rebuild_matrix --write
```

This regenerates derived matrix output. Read the matrix documentation before
replacing active files. Full heuristic solves can vary with machine load
because their restart loop is wall-clock limited; saved plans should be used
when a fixed route is required.

