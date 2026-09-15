# ============================================================
# Dirichlet Catch Model
# How it works:
#   1. Each 'dimension' of the model is a year's catch for a single station
#   2. Each historical year is equally likely to be chosen/sampled. 
#   3. The generated proportions will then be the % of a sampled catch made of up to a year historical data.
# ============================================================

import numpy as np
import pandas as pd

df = pd.read_excel('data/spring_historical_english.xlsx')

# GENERATED CODE, HAVENT LOOKED THRU YET:
def simulate_station_catch(historical_catches, alpha=1.0, n_simulations=1):
    """
    Simulate catch for one station using a symmetric Dirichlet model.

    Parameters
    ----------
    historical_catches : array-like
        Historical catches for the station.
    alpha : float
        Dirichlet concentration parameter.
    n_simulations : int
        Number of simulated catches.

    Returns
    -------
    simulated_catches : numpy array
        Simulated catches.
    proportions : numpy array
        Dirichlet proportions used for each simulation.
    """

    historical_catches = np.asarray(historical_catches)

    n_years = len(historical_catches)

    # Equal likelihood for every historical year
    alpha_vector = np.full(n_years, alpha)

    # Generate Dirichlet proportions
    proportions = np.random.dirichlet(
        alpha_vector,
        size=n_simulations
    )

    # Weighted combination of historical catches
    simulated_catches = proportions @ historical_catches

    return simulated_catches, proportions