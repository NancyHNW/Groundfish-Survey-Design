# Simulation design

The report compares four model families for predicting catch in the Icelandic Groundfish Survey.
Family	Purpose	Main implementation
| Model | Purpose | Main Implementation |
|---|---|---|
| Standard station distributions | 	Fit Normal, Gamma, and Lognormal distributions to historical catch at each station, then simulate station catches and sum them to estimate total catch | `create_distributions.py` |
| Multivariate regression | Predict station-level catch using survey year and environmental or operational variables; compare the preliminary model with the simplified model_2 | `simulations\data\DA and Multivariate Linear Regression Model.Rmd`|
| Total-proportion Dirichlet | Model the distribution of catch proportions across stations, then allocate a known survey total across stations |`simulations\Total Proportion Dirichlet Model.Rmd`|
|Station-by-station Dirichlet | Use each station’s historical catch values to simulate its future catch and represent station-level variability |`simulations\Station by Station Dirichlet Model.Rmd`|