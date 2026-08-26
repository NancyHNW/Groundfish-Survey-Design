"""Experiment drivers for the groundfish survey routing problem.

Each module is one research question and is run directly:

    python -m experiments.buffers      --ns 100 --nv 2 ...
    python -m experiments.monte_carlo  --ns 100 --nv 2 ...
    python -m experiments.strategies   --ns 100 --nv 2 ...
    python -m experiments.realisation  --ns 20  --seed 7

Shared machinery lives in ``experiments.common``. The library code these
drivers call (solving, evaluating, plotting) lives in ``unified``.
"""
