# import limitedcatch.csv from ../gfsp_code/data/limitedcatch.csv
# can probably delete this entire file 

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data
# lim_data = pd.read_csv('gfsp_code/data/LimitedCatch.csv')
# spring_data = pd.read_csv('final_code/data/Spring.csv')

# # Display the last 5 rows of the data
# print(lim_data.tail())
# print(spring_data.tail())

# # calculate mean catch amount for each dataset
# lim_mean = lim_data.mean()
# spring_mean = spring_data.mean()

# #plot the mean as a solid red line for each dataset
# # print mean
# print(f"Mean catch amount for Limited Catch dataset: {lim_mean}")
# print(f"Mean catch amount for Spring dataset: {spring_mean}")

# # Plotting only the second column, scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(lim_data)), lim_data.iloc[:, 1], marker='o')
# plt.axhline(y=lim_mean.iloc[1], color='r', linestyle='-', label=f'Mean: {lim_mean.iloc[1]:.2f}')
# plt.title('Limited Catch Data')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.grid()
# plt.show()

# # Plot the last column (or 11th column) of the spring data
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(spring_data)), spring_data.iloc[:, -1], marker='o')
# plt.axhline(y=spring_mean.iloc[-1], color='r', linestyle='-', label=f'Mean: {spring_mean.iloc[-1]:.2f}')
# plt.title('Spring Data - Last Column')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.grid()
# plt.show()

"""
Plotting utilities for the station-level normal catch distributions
produced by create_distributions() in the existing distributions module.

Assumes create_distributions / create_scenarios live in a module you
already have (e.g. `distributions.py`) alongside this script. Update
the import below to match your actual filename.
"""
"""
Plotting utilities for the station-level normal catch distributions
produced by create_distributions() in create_distributions.py.

Lives in datavisualisations.py, alongside create_distributions.py, inside
the `simulations/` folder:

    simulations/
        data/
            spring_historical_english.xlsx
        create_distributions.py
        datavisualisations.py   <- this file

Run this file with `simulations/` as the working directory (e.g. run it
from inside that folder, or run it from your IDE with that folder set as
the project root) since create_distributions.py loads the Excel file
using the relative path 'data/spring_historical_english.xlsx'.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# create_distributions.py loads the Excel file using the relative path
# 'data/spring_historical_english.xlsx', which only resolves correctly if
# the working directory is `simulations/`. Depending on how this script is
# run (VS Code's Run button, debugger, terminal from a different folder,
# etc.), that's not always the case -- so we force it here, before the
# import, based on where THIS file lives on disk.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from create_distributions import create_distributions, df  # df is already loaded in create_distributions.py


# =============
# Plot 1: single station's normal distribution
# =============
def plot_single_station(distributions, station_id, ax=None, n_points=500,
                         truncate_at_zero=True, show_hist_data=None,
                         catch_col='catch'):
    """
    Plot the fitted normal distribution (PDF) for a single station.

    Parameters:
        distributions (dict): Output of create_distributions().
        station_id: Key into `distributions` (e.g. a tow_station number).
        ax (plt.Axes | None): Existing axes to plot on; creates a new figure if None.
        n_points (int): Resolution of the PDF curve.
        truncate_at_zero (bool): If True, don't draw the curve into negative catch values,
                                  since catch cannot physically be negative.
        show_hist_data (pd.DataFrame | None): Optionally pass the raw df to overlay a
                                  histogram of that station's actual historical catches.
        catch_col (str): Catch column name, used only if show_hist_data is provided.

    Returns:
        ax (plt.Axes)
    """
    if station_id not in distributions:
        raise KeyError(f"Station {station_id} not found in distributions dict.")

    mean = distributions[station_id]['mean']
    std = distributions[station_id]['std']

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    lower = 0 if truncate_at_zero else mean - 4 * std
    upper = mean + 4 * std
    x = np.linspace(lower, upper, n_points)
    y = norm.pdf(x, loc=mean, scale=std)

    ax.plot(x, y, color='steelblue', lw=2, label=f'Station {station_id} (μ={mean:.0f}, σ={std:.0f})')
    ax.fill_between(x, y, alpha=0.15, color='steelblue')

    # optional: overlay actual historical catch values for this station
    if show_hist_data is not None:
        station_catches = show_hist_data.loc[
            show_hist_data['tow_station'] == station_id, catch_col
        ].dropna()
        if len(station_catches) > 0:
            ax.hist(station_catches, bins=15, density=True, alpha=0.4,
                     color='darkorange', label='Historical catch (observed)')

    ax.axvline(mean, color='steelblue', linestyle='--', lw=1)
    ax.set_xlabel('Catch (kg)')
    ax.set_ylabel('Density')
    ax.set_title(f'Fitted Catch Distribution — Station {station_id}')
    ax.legend()
    ax.set_xlim(left=0 if truncate_at_zero else None)

    return ax


# =============
# Plot 2: all stations' normal distributions overlaid
# =============
def plot_all_stations(distributions, ax=None, n_points=300, truncate_at_zero=True,
                       colormap='viridis', alpha=0.5, legend=False, max_x=None):
    """
    Overlay the fitted normal distribution (PDF) for every station/group in
    `distributions` on a single figure.

    Parameters:
        distributions (dict): Output of create_distributions().
        ax (plt.Axes | None): Existing axes to plot on; creates a new figure if None.
        n_points (int): Resolution of each PDF curve.
        truncate_at_zero (bool): If True, curves are only drawn for catch >= 0.
        colormap (str): Matplotlib colormap name used to distinguish stations.
        alpha (float): Line transparency (useful with many overlapping curves).
        legend (bool): Whether to show a legend. With 518 stations this gets
                        unreadable, so it defaults to off.
        max_x (float | None): Fixed upper x-limit across all curves. If None,
                        it is set to the 99th percentile of (mean + 4*std)
                        across all stations, so a few extreme stations don't
                        squash the rest of the plot.

    Returns:
        ax (plt.Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    means = np.array([d['mean'] for d in distributions.values()])
    stds = np.array([d['std'] for d in distributions.values()])
    upper_bounds = means + 4 * stds

    if max_x is None:
        max_x = np.percentile(upper_bounds, 99)

    cmap = plt.get_cmap(colormap)
    n = len(distributions)

    for i, (group_name, dist) in enumerate(distributions.items()):
        mean, std = dist['mean'], dist['std']
        lower = 0 if truncate_at_zero else mean - 4 * std
        x = np.linspace(lower, max_x, n_points)
        y = norm.pdf(x, loc=mean, scale=std)
        color = cmap(i / max(n - 1, 1))
        ax.plot(x, y, color=color, alpha=alpha, lw=1, label=str(group_name))

    ax.set_xlabel('Catch (kg)')
    ax.set_ylabel('Density')
    ax.set_title(f'Fitted Catch Distributions — All {n} Groups Overlaid')
    ax.set_xlim(0 if truncate_at_zero else None, max_x)

    if legend:
        ax.legend(fontsize='xx-small', ncol=4)

    return ax


if __name__ == '__main__':
    # df is imported directly from create_distributions.py above,
    # no need to reload the Excel file here.

    # Per-station distributions
    station_dists = create_distributions(df, group_col='tow_station')

    # --- Plot 1: a single station ---
    first_station = next(iter(station_dists))
    plot_single_station(station_dists, first_station, show_hist_data=df)
    plt.tight_layout()
    plt.savefig('single_station_distribution.png', dpi=150)
    plt.show()

    # --- Plot 2: all stations overlaid ---
    plot_all_stations(station_dists)
    plt.tight_layout()
    plt.savefig('all_stations_overlaid.png', dpi=150)
    plt.show()