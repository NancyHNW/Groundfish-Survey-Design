# Functions to create a set of distriutions for stochastic simulations
import numpy as np
import pandas as pd

# =============
# Function 1: Call the distribution function to create a set of scenarios, and write it to a file
# =============
def create_scenarios(num_scenarios, distribution_func, output_file):
    """
    Create a set of scenarios based on a specified distribution function and write them to a file.

    Parameters:
        num_scenarios (int): The number of scenarios to generate.
        distribution_func (function): A function that generates random numbers based on a specific distribution.
        output_file (str): The path to the output file where the scenarios will be saved.
    """
    # # Generate scenarios using the provided distribution function
    # scenarios = distribution_func(num_scenarios)

    # # Save the scenarios to a file
    # np.savetxt(output_file, scenarios, delimiter=',')


# =============
# Function 2: Create distribution for per station catch amount
# =============
def create_station_distributions(df,
                                 station_col='station_number',
                                 catch_col='catch'):
    """
    Creates a normal distribution for each station.

    Inputs: 
        df (pd.DataFrame): The input DataFrame containing the catch data.
        station_col (str): The name of the column in the DataFrame that contains station identifiers.
        catch_col (str): The name of the column in the DataFrame that contains catch amounts.
    
    Returns:
        distribution (dict): A dictionary where each key is a station number and the value is another dictionary 
                                containing the mean and standard deviation of the catch amounts for that station. 
                    dict: {station_number: {'mean': μ, 'std': σ}}
    """

    distributions = {}

    # Group the DataFrame by the station column
    grouped = df.groupby(station_col)

    # For each station, calculate the mean and standard deviation of the catch amounts and store them in the distributions dictionary
    for station, group in grouped:
        catches = group[catch_col].dropna()
        mean = catches.mean()
        std = catches.std(ddof=1)

        distributions[station] = {
            'mean': mean,
            'std': std
        }

    return distributions


# =============
# Function 3: Create distribution for per region catch amount
# =============
def create_region_distributions(df,
                         group_col='region',
                         station_col='station_number',
                         catch_col='catch'):
    """
    Creates a normal distribution for each region and
    records the station numbers belonging to that region.

    Returns:
        dict:
            {
                region_name: {
                    'stations': [...],
                    'mean': mean,
                    'std': std
                }
            }
    """

    distributions = {}

    grouped = df.groupby(group_col)

    for group_name, group in grouped:

        catches = group[catch_col].dropna()
        mean = catches.mean()
        std = catches.std(ddof=1)

        # Get the unique station numbers for the current region and sort them
        stations = sorted(group[station_col].unique())

        # Store the station numbers, mean, and standard deviation in the distributions dictionary
        distributions[group_name] = {
            'stations': stations,
            'mean': mean,
            'std': std
        }

    return distributions