# Experimental design

The report uses four main experiment families and a supporting re-planning
experiment.

| Family | Purpose | Main runner |
|---|---|---|
| Cost of uncertainty | Score the default full-survey route under historical catch | `experiments.overflow_map` |
| Planner variability | Separate solver-seed variation from catch variation | `experiments.variance` |
| Planner comparison | Compare route planners under common stochastic scoring | `experiments.monte_carlo`, `experiments.paired` |
| Buffer and strategy comparison | Test capacity buffers and overflow responses | `experiments.full_sweep`, `experiments.strategies`, `experiments.paired` |
| Re-planning support | Compare route time after a return or repair | `experiments.strategies` and archived report workflow |

The full problem uses four home ports: `0 6 10 12`. The default scenario seed
is 123, the default planner seed is 42, and the standard recourse policy is
Forward unless a command states otherwise.

