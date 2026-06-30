# Functions to create a set of distriutions for stochastic simulations
import numpy as np
import pandas as pd

df = pd.read_excel('data/spring_historical_english.xlsx')

# =============
# Function 1: Call the distribution function to create a set of scenarios, and write it to a file
# =============
def create_scenarios(station_ids, num_scenarios, distributions, output_file):
    """
    Create a set of scenarios by sampling from each station's normal distribution.

    Parameters:
        station_ids (list): A list of station IDs for which to create scenarios.
        num_scenarios (int): The number of scenarios to generate for each station.
        distributions (dict): The output of create_distributions(), containing each
                              station's mean and std.
        output_file (str): The path to the output CSV where the scenarios will be saved.

    Returns:
        scenarios (pd.DataFrame): A DataFrame of shape (num_scenarios, len(station_ids))
                                  where each column is a station and each row is a scenario.
    """
    
    scenarios = {}

    for station in station_ids:
        mean = distributions[station]['mean']
        std  = distributions[station]['std']

        # Sample num_scenarios values from this station's normal distribution
        scenarios[station] = np.random.normal(loc=mean, scale=std, size=num_scenarios)

    # Convert to DataFrame; rows = scenarios, columns = stations
    scenarios = pd.DataFrame(scenarios)
    scenarios.to_csv(output_file, index=False)

    return scenarios
        

# =============
# Function 2: Create distributions for each group in the DataFrame
# =============
def create_distributions(df,
                         group_col='station_number',
                         catch_col='catch',
                         station_col=None):
    """
    Creates a normal distribution for each group in the DataFrame.

    Inputs:
        df (pd.DataFrame): The input DataFrame containing the catch data.
        group_col (str): The column to group by (typically 'station_number' or 'region').
        catch_col (str): The column containing catch amounts.
        station_col (str | None): If provided, each group's entry will include a sorted
                                  list of unique station numbers under the 'stations' key.

    Returns:
        dict: {
            group_name: {
                'mean': float,
                'std': float,
                ['stations': list]  # only present if station_col is provided
            }
        }
    """

    distributions = {}

    # group by the specified column and calculate mean and std for each group
    for group_name, group in df.groupby(group_col):
        catches = group[catch_col].dropna()
        entry = {
            'mean': catches.mean(),
            'std': catches.std(ddof=1)
        }

        # If station_col is provided, add a sorted list of unique station numbers to the entry
        if station_col is not None:
            entry['stations'] = sorted(group[station_col].unique())

        distributions[group_name] = entry

    return distributions

# Calling it:
# # If we want to group by station number:
# station_dists = create_distributions(df, group_col='station_number')
# # If we want to group by region, and also include the station numbers in each region:
# region_dists = create_distributions(df, group_col='region', station_col='station_number')

def main():
    # Example usage
    distributions = create_distributions(df, group_col='tow_station')

    # Then generate scenarios
    scenarios = create_scenarios(
        station_ids   = [1, 2, 3],
        num_scenarios = 1000,
        distributions = distributions,
        output_file   = 'scenarios.csv'
    )
    