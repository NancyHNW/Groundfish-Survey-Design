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
                         group_col='tow_station',
                         catch_col='catch',
                         station_col=None):
    """
    Creates a normal distribution for each group in the DataFrame.

    Inputs:
        df (pd.DataFrame): The input DataFrame containing the catch data.
        group_col (str): The column to group by (typically 'tow_station' or 'quadrant').
            i.e. if 'tow_station' is provided, the function will create a distribution for each unique station number.
                 if 'quadrant' is provided, the function will create a distribution for each unique quadrant.
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
# station_dists = create_distributions(df, group_col='tow_station')
# # If we want to group by region, and also include the station numbers in each region:
# region_dists = create_distributions(df, group_col='region', station_col='tow_station')


def create_gamma_distributions(df,
                               group_col='tow_station',
                               catch_col='catch',
                               station_col=None):
    """
    Creates a Gamma distribution for each group in the DataFrame.

    Inputs:
        df (pd.DataFrame): The input DataFrame containing the catch data.
        group_col (str): The column to group by (e.g. 'tow_station' or 'quadrant').
        catch_col (str): The column containing catch amounts.
        station_col (str | None): If provided, each group's entry will include a
                                  sorted list of unique station numbers.

    Returns:
        dict: {
            group_name: {
                'mean': float,
                'std': float,
                'shape': float,
                'scale': float,
                ['stations': list]
            }
        }
    """

    distributions = {}

    for group_name, group in df.groupby(group_col):

        # Remove missing values
        catches = group[catch_col].dropna()

        # Calculate sample mean and standard deviation
        mean = catches.mean()
        std = catches.std(ddof=1)

        # Calculate Gamma parameters using method of moments
        shape = (mean ** 2) / (std ** 2)
        scale = (std ** 2) / mean

        entry = {
            'mean': mean,
            'std': std,
            'shape': shape,
            'scale': scale
        }

        # Add station list if requested
        if station_col is not None:
            entry['stations'] = sorted(group[station_col].unique())

        distributions[group_name] = entry

    return distributions

# =========== Example to call:
# gamma_distributions = create_gamma_distributions(
#     df,
#     group_col='quadrant',
#     catch_col='catch',
#     station_col='tow_station'
# )

# =========== Will get something like:
# Quadrant 1:
#     mean: 692.89
#     std: 850.21
#     shape: 0.664
#     scale: 1043.12
#     stations: [...]

# =========== To generate distributions:
# np.random.gamma(
#     shape=gamma_distributions[1]['shape'],
#     scale=gamma_distributions[1]['scale']
# )

def create_lognormal_distributions(df,
                                   group_col='tow_station',
                                   catch_col='catch',
                                   station_col=None):
    """
    Creates a log-normal distribution for each group in the DataFrame.

    Inputs:
        df (pd.DataFrame): The input DataFrame containing the catch data.
        group_col (str): The column to group by (e.g. 'tow_station' or 'quadrant').
        catch_col (str): The column containing catch amounts.
        station_col (str | None): If provided, each group's entry will include a
                                  sorted list of unique station numbers.

    Returns:
        dict: {
            group_name: {
                'mean': float,
                'std': float,
                'mu': float,
                'sigma': float,
                ['stations': list]
            }
        }
    """

    distributions = {}

    for group_name, group in df.groupby(group_col):

        # Remove missing values
        catches = group[catch_col].dropna()

        # Log-normal requires strictly positive values
        catches = catches[catches > 0]

        # Calculate mean and std of the original catch data
        mean = catches.mean()
        std = catches.std(ddof=1)

        # Take natural logarithm of the catch values
        log_catches = np.log(catches)

        # Parameters of the underlying normal distribution
        mu = log_catches.mean()
        sigma = log_catches.std(ddof=1)

        entry = {
            'mean': mean,
            'std': std,
            'mu': mu,
            'sigma': sigma
        }

        # Add station list if requested
        if station_col is not None:
            entry['stations'] = sorted(group[station_col].unique())

        distributions[group_name] = entry

    return distributions

# =========== Example to call:
# lognormal_distributions = create_lognormal_distributions(
#     df,
#     group_col='quadrant',
#     catch_col='catch',
#     station_col='tow_station'
# )

# =========== To generate distributions:
# np.random.lognormal(
#     mean=lognormal_distributions[1]['mu'],
#     sigma=lognormal_distributions[1]['sigma']
# )

def main():
    # Example usage
    distributions = create_distributions(df, group_col='tow_station')

    # print("Distributions created for each station:")
    # for station, dist in distributions.items():
    #     print(f"Station {station}: Mean = {dist['mean']:.2f}, Std = {dist['std']:.2f}")

    # Then generate scenarios
    # scenarios = create_scenarios(
    #     station_ids   = [1, 2, 3],
    #     num_scenarios = 1000,
    #     distributions = distributions,
    #     output_file   = 'scenarios.csv'
    # )

if __name__ == '__main__':
    main()