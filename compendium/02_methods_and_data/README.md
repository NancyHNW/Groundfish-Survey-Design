# Research method -- Route Assessment

This section explains the research independently of the implementation.

The supporting system schematics are in
[`diagrams/`](diagrams/), including the pipeline architecture, project
structure, and recourse-policy diagrams.

## Research question

How much additional vessel time is caused by uncertain catch, and how can
planning buffers or route-repair policies reduce the realised cost of
capacity overflows?

## System

The survey contains 581 stations and four vessels. A station is represented by
two tow-end nodes. A planned route is evaluated under catch scenarios sampled
from historical spring-survey data. If accumulated catch exceeds a vessel's
capacity, the vessel makes an unscheduled return to the nearest port, incurs a
round-trip detour, offloads, and continues according to the selected recourse
policy.

## Method components

- Deterministic route planning uses inherited heuristic planners and saved
  plans.
- Historical catch distributions are fitted per station where data exists;
  stations without records inherit the nearest station's fitted distribution.
- The stochastic evaluator scores a fixed route against repeated catch
  scenarios.
- Planning buffers reduce the capacity used during route construction.
- Recourse strategies include Forward, Backtrack, Preemptive, and Repair
  variants.
- Paired experiments use shared blocks or scenarios so comparisons are not
  dominated by instance or catch noise.

## Main measures

- planned time;
- realised time;
- cost of uncertainty, realised time minus planned time;
- probability of at least one true capacity overflow;
- Wilson 95% interval for overflow probability;
- expected unscheduled returns;
- trip-level overflow position;
- repair and diversion penalties.

## Scope and limitations

The report's results were generated on one Windows 11 machine with Gurobi
limited to one thread. Heuristic restart loops stop on wall-clock time, so
planned routes can differ between machines even with the same seed. Scenario
sets with different sizes are separate random draws and must not be
differenced as if one were nested inside the other.

## Catch modelling

This section describes the catch modelling and prediction methodology independently of the implementation.

## Research question

How accurately can historical survey data be used to predict total and station-level catch for future Icelandic Groundfish Surveys, and which modelling approaches are most suitable for supporting efficient survey planning?

## Data

The analysis uses historical spring-survey data from 1985–2022, with the main modelling and validation experiments focusing on 2002–2022 to better reflect recent catch and environmental conditions. The dataset records catch at individual tow stations alongside environmental, geographical, and operational variables. Catch is highly right-skewed, motivating the investigation of probability distributions, log-transformed regression, and Dirichlet-based approaches.

## Method components

- **Standard distributions** fit Normal, Gamma, and Lognormal distributions to historical catch at each station. Simulated station catches are summed to estimate total survey catch.
- **Multivariate regression** predicts station-level catch using survey year and selected geographical, environmental, and operational predictors. Preliminary and simplified models are compared.
- **Total-proportion Dirichlet** models the allocation of a known total catch across stations using historical station-level catch proportions.
- **Station-by-station Dirichlet** uses each station's historical catch values to simulate future catch and represent variability between years.
- **Repeated validation** evaluates predictions across random training and testing year splits, using 10 years for training and 10 years for testing.
- **Model comparison** assesses prediction accuracy and uncertainty to identify suitable approaches for station-level forecasting and total-catch estimation.

## Main measures

- mean absolute error (MAE);
- root mean squared error (RMSE);
- prediction-interval coverage;
- prediction-interval width.

## Scope and limitations

Model performance depends on the historical data used for training and how representative these years are of future survey conditions. Catch variability between years and stations limits the accuracy of individual predictions. The total-proportion Dirichlet model requires the total catch to be known, so it is primarily suitable for allocating catch across stations rather than independently forecasting total catch. Standard distributions may not adequately capture the variability of total survey catch, while regression predictions depend on the relationships between catch and the selected predictors remaining reasonably stable. Consequently, predictions should be interpreted alongside their uncertainty and validation results before being used to support survey planning.
