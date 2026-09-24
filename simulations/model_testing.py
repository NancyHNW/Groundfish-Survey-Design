import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from create_distributions import (
    create_distributions,
    create_gamma_distributions,
    create_lognormal_distributions
)

# ============================================================
# SETTINGS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'data' / 'spring_historical_english.xlsx'

START_YEAR = 2002
END_YEAR = 2021       # 20 years: 2002-2021 inclusive

N_TRAIN = 10
N_SIMULATIONS = 500
RANDOM_SEED = 247012216

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_excel(DATA_FILE)

# Keep only the desired 20-year period
df = df[
    (df['year'] >= START_YEAR) &
    (df['year'] <= END_YEAR)
].copy()

print(f"Years included: {START_YEAR}-{END_YEAR}")
print(f"Number of years: {df['year'].nunique()}")
print()


# ============================================================
# RANDOMLY SPLIT YEARS INTO TRAINING AND TESTING
# ============================================================
rng = np.random.default_rng(RANDOM_SEED)

all_years = sorted(df['year'].unique())

training_years = sorted(
    rng.choice(
        all_years,
        size=N_TRAIN,
        replace=False
    )
)

testing_years = sorted(
    [year for year in all_years if year not in training_years]
)

print("Training years:")
print(training_years)

print("\nTesting years:")
print(testing_years)

print()


# ============================================================
# CREATE TRAINING AND TESTING DATASETS
# ============================================================

train_df = df[df['year'].isin(training_years)].copy()
test_df = df[df['year'].isin(testing_years)].copy()


# ============================================================
# ACTUAL TESTING MEAN
# ============================================================
# Calculate total catch for each testing year
test_year_totals = (
    test_df
    .groupby('year')['catch']
    .sum()
)

actual_test_mean = test_year_totals.mean()

print("Actual testing-year total catches:")
print(test_year_totals)

print(f"\nActual mean total catch over testing years: "
      f"{actual_test_mean:.2f}")

print()


# ============================================================
# FUNCTION TO SIMULATE NORMAL MODEL
# ============================================================

def simulate_normal(distributions, station_ids, n_simulations, rng):

    simulated_totals = []

    for _ in range(n_simulations):

        total_catch = 0

        for station in station_ids:

            mean = distributions[station]['mean']
            std = distributions[station]['std']

            catch = rng.normal(
                loc=mean,
                scale=std
            )

            # Catch cannot be negative
            catch = max(0, catch)

            total_catch += catch

        simulated_totals.append(total_catch)

    return np.array(simulated_totals)


# ============================================================
# FUNCTION TO SIMULATE GAMMA MODEL
# ============================================================

def simulate_gamma(distributions, station_ids, n_simulations, rng):

    simulated_totals = []

    for _ in range(n_simulations):

        total_catch = 0

        for station in station_ids:

            shape = distributions[station]['shape']
            scale = distributions[station]['scale']

            catch = rng.gamma(
                shape=shape,
                scale=scale
            )

            total_catch += catch

        simulated_totals.append(total_catch)

    return np.array(simulated_totals)


# ============================================================
# FUNCTION TO SIMULATE LOGNORMAL MODEL
# ============================================================

def simulate_lognormal(distributions, station_ids, n_simulations, rng):

    simulated_totals = []

    for _ in range(n_simulations):

        total_catch = 0

        for station in station_ids:

            mu = distributions[station]['mu']
            sigma = distributions[station]['sigma']

            catch = rng.lognormal(
                mean=mu,
                sigma=sigma
            )

            total_catch += catch

        simulated_totals.append(total_catch)

    return np.array(simulated_totals)


# ============================================================
# FIT DISTRIBUTIONS USING TRAINING DATA
# ============================================================

normal_distributions = create_distributions(
    train_df,
    group_col='tow_station',
    catch_col='catch'
)

gamma_distributions = create_gamma_distributions(
    train_df,
    group_col='tow_station',
    catch_col='catch'
)

lognormal_distributions = create_lognormal_distributions(
    train_df,
    group_col='tow_station',
    catch_col='catch'
)


# ============================================================
# GET STATIONS
# ============================================================

station_ids = sorted(train_df['tow_station'].dropna().unique())

print(f"Number of stations: {len(station_ids)}")
print()


# ============================================================
# SIMULATE EACH MODEL
# ============================================================

normal_simulations = simulate_normal(
    normal_distributions,
    station_ids,
    N_SIMULATIONS,
    rng
)

gamma_simulations = simulate_gamma(
    gamma_distributions,
    station_ids,
    N_SIMULATIONS,
    rng
)

lognormal_simulations = simulate_lognormal(
    lognormal_distributions,
    station_ids,
    N_SIMULATIONS,
    rng
)


# ============================================================
# CALCULATE PERFORMANCE
# ============================================================

def calculate_performance(simulations, actual_mean):

    simulated_mean = np.mean(simulations)

    lower_ci = np.percentile(simulations, 2.5)
    upper_ci = np.percentile(simulations, 97.5)

    inside_ci = (
        lower_ci <= actual_mean <= upper_ci
    )

    error = simulated_mean - actual_mean

    absolute_error = abs(error)

    percentage_error = (
        absolute_error / actual_mean
    ) * 100

    return {
        'Simulated Mean': simulated_mean,
        '95% CI Lower': lower_ci,
        '95% CI Upper': upper_ci,
        'Actual Test Mean': actual_mean,
        'Error': error,
        'Absolute Error': absolute_error,
        'Percentage Error (%)': percentage_error,
        'Actual Mean Inside CI': inside_ci
    }


normal_results = calculate_performance(
    normal_simulations,
    actual_test_mean
)

gamma_results = calculate_performance(
    gamma_simulations,
    actual_test_mean
)

lognormal_results = calculate_performance(
    lognormal_simulations,
    actual_test_mean
)


# ============================================================
# PRINT RESULTS
# ============================================================

results = pd.DataFrame({
    'Normal': normal_results,
    'Gamma': gamma_results,
    'Lognormal': lognormal_results
}).T

print("=" * 80)
print("MODEL PERFORMANCE")
print("=" * 80)

print(results.round(2))

print()


# ============================================================
# MORE READABLE OUTPUT
# ============================================================

for model_name, result in [
    ('Normal', normal_results),
    ('Gamma', gamma_results),
    ('Lognormal', lognormal_results)
]:

    print("-" * 60)
    print(model_name)

    print(
        f"Simulated mean:       "
        f"{result['Simulated Mean']:.2f}"
    )

    print(
        f"95% CI:               "
        f"({result['95% CI Lower']:.2f}, "
        f"{result['95% CI Upper']:.2f})"
    )

    print(
        f"Actual test mean:     "
        f"{result['Actual Test Mean']:.2f}"
    )

    print(
        f"Absolute error:       "
        f"{result['Absolute Error']:.2f}"
    )

    print(
        f"Percentage error:     "
        f"{result['Percentage Error (%)']:.2f}%"
    )

    print(
        f"Actual mean in CI:    "
        f"{result['Actual Mean Inside CI']}"
    )

print("-" * 60)

# ============================================================
# PLOTTING RESULTS
# ============================================================

# 1) Estimated mean vs actual testing mean
model_names = ['Normal', 'Gamma', 'Lognormal']

simulated_means = [
    normal_results['Simulated Mean'],
    gamma_results['Simulated Mean'],
    lognormal_results['Simulated Mean']
]

actual_mean = actual_test_mean

plt.figure(figsize=(8, 6))

plt.bar(model_names, simulated_means, alpha=0.8)

# Actual testing mean
plt.axhline(
    actual_mean,
    linestyle='--',
    label='Actual testing mean'
)

plt.ylabel('Total catch')
plt.title('Model Estimated Mean vs Actual Testing Mean')
plt.legend()

plt.tight_layout()
plt.show()


# 2) Percentage error
percentage_errors = [
    normal_results['Percentage Error (%)'],
    gamma_results['Percentage Error (%)'],
    lognormal_results['Percentage Error (%)']
]

plt.figure(figsize=(8, 6))

plt.bar(model_names, percentage_errors, alpha=0.8)

plt.ylabel('Absolute percentage error (%)')
plt.title('Model Performance: Percentage Error')

plt.tight_layout()
plt.show()


# 3) 95% CI compared with testing mean
ci_lower = [
    normal_results['95% CI Lower'],
    gamma_results['95% CI Lower'],
    lognormal_results['95% CI Lower']
]

ci_upper = [
    normal_results['95% CI Upper'],
    gamma_results['95% CI Upper'],
    lognormal_results['95% CI Upper']
]

plt.figure(figsize=(8, 6))

x = np.arange(len(model_names))

# Estimated means
plt.scatter(
    x,
    simulated_means,
    marker='o',
    label='Simulated mean'
)

# 95% CI
plt.vlines(
    x,
    ci_lower,
    ci_upper,
    linewidth=3,
    label='95% CI'
)

# Actual testing mean
plt.axhline(
    actual_mean,
    linestyle='--',
    label='Actual testing mean'
)

plt.xticks(x, model_names)
plt.ylabel('Total catch')
plt.title('Model Simulations and 95% Confidence Intervals')
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# SECOND TESTS
# ============================================================

# ============================================================
# TEST MODELS AGAINST RANDOM HISTORICAL CATCHES
# ============================================================

# Random number generator
rng = np.random.default_rng(42)

# Historical years to sample from
historical_years = list(range(2002, 2013))


def test_model_against_historical(
    distributions,
    station_ids,
    model,
    df,
    n_simulations=100,
    rng=None
):
    """
    Test a fitted distribution against randomly selected
    historical catches for each station.

    For each station:
        1. Randomly select a year from 2002-2012
        2. Get the actual catch in that year
        3. Simulate 100 catches from the fitted distribution
        4. Calculate the 95% CI
        5. Check whether the actual catch falls inside the CI
        6. Calculate error between simulated mean and actual catch

    Returns:
        DataFrame containing results for every station.
    """

    results = []

    for station in station_ids:

        # ----------------------------------------------------
        # Get historical observations for this station
        # ----------------------------------------------------

        station_data = df[
            (df['tow_station'] == station) &
            (df['year'].isin(historical_years))
        ].dropna(subset=['catch'])

        # Skip station if there is no historical data
        if len(station_data) == 0:
            continue

        # ----------------------------------------------------
        # Randomly select one historical year
        # ----------------------------------------------------

        selected_row = station_data.iloc[
            rng.integers(0, len(station_data))
        ]

        selected_year = selected_row['year']
        actual_catch = selected_row['catch']

        # ----------------------------------------------------
        # Simulate from the fitted model
        # ----------------------------------------------------

        if model == 'normal':

            mean = distributions[station]['mean']
            std = distributions[station]['std']

            simulations = rng.normal(
                loc=mean,
                scale=std,
                size=n_simulations
            )

            # Catch cannot be negative
            simulations = np.maximum(simulations, 0)

        elif model == 'gamma':

            shape = distributions[station]['shape']
            scale = distributions[station]['scale']

            simulations = rng.gamma(
                shape=shape,
                scale=scale,
                size=n_simulations
            )

        elif model == 'lognormal':

            mu = distributions[station]['mu']
            sigma = distributions[station]['sigma']

            simulations = rng.lognormal(
                mean=mu,
                sigma=sigma,
                size=n_simulations
            )

        else:
            raise ValueError(
                "model must be 'normal', 'gamma', or 'lognormal'"
            )

        # ----------------------------------------------------
        # Calculate simulated statistics
        # ----------------------------------------------------

        simulated_mean = np.mean(simulations)

        lower_ci = np.percentile(
            simulations,
            2.5
        )

        upper_ci = np.percentile(
            simulations,
            97.5
        )

        # ----------------------------------------------------
        # Check whether actual catch is inside CI
        # ----------------------------------------------------

        inside_ci = (
            lower_ci <= actual_catch <= upper_ci
        )

        # ----------------------------------------------------
        # Calculate error
        # ----------------------------------------------------

        absolute_error = abs(
            simulated_mean - actual_catch
        )

        percentage_error = (
            absolute_error / actual_catch
        ) * 100 if actual_catch != 0 else np.nan

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({
            'station': station,
            'year': selected_year,
            'actual_catch': actual_catch,
            'simulated_mean': simulated_mean,
            'ci_lower': lower_ci,
            'ci_upper': upper_ci,
            'inside_ci': inside_ci,
            'absolute_error': absolute_error,
            'percentage_error': percentage_error
        })

    return pd.DataFrame(results)


# ============================================================
# RUN HISTORICAL CATCH TEST
# ============================================================

normal_historical_results = test_model_against_historical(
    normal_distributions,
    station_ids,
    'normal',
    df,
    n_simulations=100,
    rng=rng
)

gamma_historical_results = test_model_against_historical(
    gamma_distributions,
    station_ids,
    'gamma',
    df,
    n_simulations=100,
    rng=rng
)

lognormal_historical_results = test_model_against_historical(
    lognormal_distributions,
    station_ids,
    'lognormal',
    df,
    n_simulations=100,
    rng=rng
)

print("\nNormal model:")
print(normal_historical_results)

print("\nGamma model:")
print(gamma_historical_results)

print("\nLognormal model:")
print(lognormal_historical_results)

def summarise_historical_performance(results):

    return {
        'Number of stations': len(results),

        'Inside 95% CI': results['inside_ci'].sum(),

        'Outside 95% CI': (~results['inside_ci']).sum(),

        'Percentage inside CI (%)':
            results['inside_ci'].mean() * 100,

        'Percentage outside CI (%)':
            (~results['inside_ci']).mean() * 100,

        'Mean absolute error':
            results['absolute_error'].mean(),

        'Median absolute error':
            results['absolute_error'].median(),

        'Mean percentage error (%)':
            results['percentage_error'].mean(),

        'Median percentage error (%)':
            results['percentage_error'].median()
    }


normal_summary = summarise_historical_performance(
    normal_historical_results
)

gamma_summary = summarise_historical_performance(
    gamma_historical_results
)

lognormal_summary = summarise_historical_performance(
    lognormal_historical_results
)


summary = pd.DataFrame({
    'Normal': normal_summary,
    'Gamma': gamma_summary,
    'Lognormal': lognormal_summary
}).T


print("\n")
print("=" * 80)
print("HISTORICAL CATCH PERFORMANCE")
print("=" * 80)
print(summary.round(2))