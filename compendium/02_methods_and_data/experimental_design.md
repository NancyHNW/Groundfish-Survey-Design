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

# Validation design
The main validation uses historical data from 2002–2022 and repeated random splits, with 10 years used for training and 10 years for testing in each split. Results are assessed at station level and, where applicable, for total survey catch. Evaluation measures include mean absolute error (MAE), and 95% prediction-interval coverage; additional station-level metrics include RMSE, MAPE, bias, original-scale R-squared, and interval width.

The standard-distribution experiments use 100 repetitions and 1,000 simulations per split, with seeds 566 and 818 used for reproducibility. The total-proportion Dirichlet experiment uses 100 simulations per test year. The station-by-station Dirichlet model uses alpha = 1, giving equal weight to historical years in the Dirichlet parameterisation.

# Interpretation and intended use
The total-proportion Dirichlet model provides station-level uncertainty intervals but requires the total catch for the test year, so it is most appropriate for allocating a known total rather than forecasting the total independently. The simplified regression model offers a more compact predictor set for station-level forecasts. The standard distributions are straightforward baselines, but their total-catch predictions and interval coverage are less reliable. The station-by-station Dirichlet model provides a way to represent historical variability at individual stations, although its performance depends on whether future catches are represented by the historical values available for training.

Predictions can support survey planning by highlighting expected catch and uncertainty across stations. They should be used alongside their validation results and limitations, rather than as exact forecasts, and can inform route planning or sampling priorities when combined with operational constraints.