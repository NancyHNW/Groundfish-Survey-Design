# Catch models and simulations

This folder contains the data analysis, catch prediction models, and validation experiments discussed in Jessica Zeng's research report on the Icelandic Groundfish Survey. The models use historical spring-survey data to predict total and station-level catch, with the aim of supporting more efficient survey planning.

## Folders

**`data/`** contains the historical survey dataset, exploratory data analysis, discriminant analysis (DA), multivariate linear regression models, and a Power BI dashboard for interactive data exploration.

## Files

**`create_distributions.py`** contains functions to fit Normal, Gamma, and Lognormal distributions to historical catch data at individual stations and generate simulated catches.

**`datavisualisations.py`** contains functions and code for visualising and exploring the raw survey data, including catch distributions and relationships between variables.

**`model_testing.py`** tests the Normal, Gamma, and Lognormal models by simulating catches and evaluating their prediction performance.

**`Station by Station Dirichlet Model.Rmd`** implements the station-by-station Dirichlet model, which uses historical catch values at each station to simulate future catches and quantify uncertainty.

**`Total Proportion Dirichlet Model.Rmd`** implements the total-proportion Dirichlet model, which allocates a known total catch across stations according to their historical catch proportions.

**`simulated_dirichlet_proportions.csv`** contains simulated Dirichlet catch proportions. The first column identifies the simulation, and each subsequent column represents a station.

**`stationbystation_station_summary.csv`** summarises station-by-station Dirichlet validation results, including the number and percentage of predictions inside and outside the prediction intervals.

## Files in `data/`

**`DA and Multivariate Linear Regression Model.Rmd`** contains the discriminant analysis and multivariate regression modelling workflows, including the preliminary and simplified regression models.

**`spring_historical_english.xlsx`** is the historical spring-survey dataset used for modelling and analysis. It includes an additional `quadrant` variable derived from station coordinates.

**`Historical Spring Dashboard.pbix`** is a Power BI dashboard for interactively exploring historical catch, station characteristics, and other survey variables.
