"""Generate 581-station catch scenarios from historical survey data.

Stations with no records use the nearest station with records. Samples are
normal draws clipped at zero.
"""

import os
import sys
import numpy as np
import pandas as pd

# Make simulations/ importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SIM_DIR = os.path.join(_ROOT, "simulations")
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

from create_distributions import create_distributions

N_STATIONS = 581
# Coordinate helpers
def _degmin2deg(val):
    """Convert degree-minute encoded integers to decimal degrees."""
    val = np.asarray(val, dtype=float)
    min_ = (val / 100) - np.floor(val / 10000.0) * 100.0
    return (val + (200.0 / 3.0) * min_) / 10000.0
# Station mapping
def _load_station_map():
    """Parse smb.2019.dat and return mapping tables + midpoint coordinates.

    Returns
    -------
    ordinal_to_tow : dict  {0: 1, 1: 31001, ...}
    tow_to_ordinal : dict  {1: 0, 31001: 1, ...}
    midpoints : np.ndarray, shape (581, 2)
        Midpoint (lat, lon) in decimal degrees for each station ordinal.
        Longitude is positive-west (matching the survey convention).
    """
    smb_path = os.path.join(_ROOT, "gfsp_code", "data", "smb.2019.dat")
    with open(smb_path) as f:
        lines = [line.split('\t') for line in f]

    ordinal_to_tow = {}
    tow_to_ordinal = {}
    raw_coords = np.zeros((len(lines), 4), dtype=int)

    for ordinal, cols in enumerate(lines):
        tow = int(cols[0]) * 100 + int(cols[1])
        ordinal_to_tow[ordinal] = tow
        tow_to_ordinal[tow] = ordinal
        raw_coords[ordinal] = [int(cols[3]), int(cols[4]),
                               int(cols[5]), int(cols[6])]

    deg = _degmin2deg(raw_coords)  # (581, 4): i_lat, i_lon, f_lat, f_lon
    midpoints = np.column_stack([
        (deg[:, 0] + deg[:, 2]) / 2,  # mid lat
        (deg[:, 1] + deg[:, 3]) / 2,  # mid lon
    ])
    return ordinal_to_tow, tow_to_ordinal, midpoints
# Fit cache
# Cached because fitting takes ~138s and the inputs never change.

_SMB_PATH = os.path.join(_ROOT, "gfsp_code", "data", "smb.2019.dat")
_CACHE_PATH = os.path.join(_ROOT, ".cache", "catch_fit.npz")

# Bump when anything stored changes, so old caches are rejected.
# 2 = dropped the regional tier; the 35 stations it caught now take the
# nearest station's fit like the other 28.
_CACHE_VERSION = 2


def _source_fingerprint(excel_path):
    """Size and mtime of the two source files, so a stale cache is rejected."""
    parts = [_CACHE_VERSION]
    for path in (_SMB_PATH, excel_path):
        stat = os.stat(path)
        parts += [stat.st_size, stat.st_mtime_ns]
    return np.array(parts, dtype=np.int64)


def _load_fit_cache(excel_path):
    """Return the cached fit, or None if absent, stale or unreadable."""
    if not os.path.isfile(_CACHE_PATH):
        return None
    try:
        with np.load(_CACHE_PATH) as data:
            if not np.array_equal(data["fingerprint"],
                                  _source_fingerprint(excel_path)):
                return None
            return {k: data[k] for k in
                    ("means", "stds", "tiers", "midpoints", "tows")}
    except (OSError, ValueError, KeyError):
        # Truncated or from an older version -- just refit.
        return None


def _save_fit_cache(excel_path, means, stds, tiers, midpoints, tows):
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    np.savez(_CACHE_PATH,
             fingerprint=_source_fingerprint(excel_path),
             means=means, stds=stds, tiers=tiers,
             midpoints=midpoints, tows=tows)


def _tows_to_maps(tows):
    """Rebuild both dicts from the cached array -- they are exact inverses."""
    ordinal_to_tow = {int(o): int(t) for o, t in enumerate(tows)}
    tow_to_ordinal = {int(t): int(o) for o, t in enumerate(tows)}
    return ordinal_to_tow, tow_to_ordinal
# Distribution fitting
def _fit_distributions(df, ordinal_to_tow, tow_to_ordinal, midpoints):
    """Fit catch distributions for all 581 stations, with one fallback.

    Stations with records use their own fitted distribution. The rest use the
    nearest station with records.

    Parameters
    ----------
    df : pd.DataFrame
        Historical catch data with columns 'tow_station' and 'catch'.
    ordinal_to_tow : dict
    tow_to_ordinal : dict
    midpoints : np.ndarray, shape (581, 2)

    Returns
    -------
    means : np.ndarray, shape (581,)
    stds  : np.ndarray, shape (581,)
    tiers : np.ndarray, shape (581,), dtype=int
        1 = own records, 2 = nearest station's.
    """
    means = np.full(N_STATIONS, np.nan)
    stds = np.full(N_STATIONS, np.nan)
    tiers = np.zeros(N_STATIONS, dtype=int)

    # Clean the data
    catch_df = df[['tow_station', 'catch']].dropna().copy()
    catch_df['tow_station'] = catch_df['tow_station'].astype(int)

    # Tier 1: per-station distributions (Jess's function)
    station_dists = create_distributions(catch_df, group_col='tow_station',
                                         catch_col='catch')

    for tow, params in station_dists.items():
        if tow in tow_to_ordinal:
            ordinal = tow_to_ordinal[tow]
            means[ordinal] = params['mean']
            std = params['std']
            # If only one observation, std is NaN — use 30% of mean
            stds[ordinal] = std if pd.notna(std) else params['mean'] * 0.3
            tiers[ordinal] = 1

    # Tier 2: nearest station that has records
    # Candidates are tier 1 only, so no station inherits a fit that was itself
    # inherited -- otherwise the choice would depend on fitting order.
    has_dist = np.where(tiers == 1)[0]
    needs_dist = np.where(tiers == 0)[0]

    if len(needs_dist) > 0 and len(has_dist) > 0:
        for ordinal in needs_dist:
            dists = np.hypot(midpoints[has_dist, 0] - midpoints[ordinal, 0],
                             midpoints[has_dist, 1] - midpoints[ordinal, 1])
            nearest = has_dist[np.argmin(dists)]
            means[ordinal] = means[nearest]
            stds[ordinal] = stds[nearest]
            tiers[ordinal] = 2

    return means, stds, tiers
# Main class
class CatchSimulator:
    """Generate stochastic catch scenarios for the 581-station survey network.

    Fits normal distributions from historical spring survey data, falling back
    to the nearest station with records where a station has none.

    Parameters
    ----------
    excel_path : str, optional
        Path to spring_historical_english.xlsx.
        Defaults to simulations/data/spring_historical_english.xlsx.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> sim = CatchSimulator(seed=42)
    >>> sim.fit()
    >>> scenarios = sim.sample(n_scenarios=1000)  # shape (1000, 581)
    """

    def __init__(self, excel_path=None, seed=None):
        if excel_path is None:
            excel_path = os.path.join(
                _ROOT, "simulations", "data", "spring_historical_english.xlsx"
            )
        self.excel_path = excel_path
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Loaded on fit()
        self.means = None
        self.stds = None
        self.tiers = None
        self.ordinal_to_tow = None
        self.tow_to_ordinal = None
        self.midpoints = None
        self._fitted = False

    def fit(self, cache=True):
        """Fit distributions from the historical data.

        Loads smb.2019.dat for station mapping and the Excel file for
        historical catch observations. Applies the nearest-station fallback.

        Reading the workbook takes about 138 seconds and gives the same answer
        every time, so the result is cached to disk and reused while the source
        files are unchanged. Pass cache=False to force a real fit.
        """
        cached = _load_fit_cache(self.excel_path) if cache else None

        if cached is not None:
            self.means = cached["means"]
            self.stds = cached["stds"]
            self.tiers = cached["tiers"]
            self.midpoints = cached["midpoints"]
            self.ordinal_to_tow, self.tow_to_ordinal = _tows_to_maps(
                cached["tows"])
            source = "cached"
        else:
            self.ordinal_to_tow, self.tow_to_ordinal, self.midpoints = (
                _load_station_map()
            )
            df = pd.read_excel(self.excel_path)
            self.means, self.stds, self.tiers = _fit_distributions(
                df, self.ordinal_to_tow, self.tow_to_ordinal, self.midpoints
            )
            source = "fitted"
            if cache:
                tows = np.array([self.ordinal_to_tow[o]
                                 for o in range(len(self.ordinal_to_tow))],
                                dtype=np.int64)
                _save_fit_cache(self.excel_path, self.means, self.stds,
                                self.tiers, self.midpoints, tows)

        self._fitted = True

        print(f"CatchSimulator {source}: "
              f"{int((self.tiers == 1).sum())} direct, "
              f"{int((self.tiers == 2).sum())} nearest-neighbour")

    def sample(self, n_scenarios=1000, station_ids=None):
        """Generate catch scenario matrix.

        Parameters
        ----------
        n_scenarios : int
            Number of scenarios to generate.
        station_ids : array-like, optional
            Subset of station ordinals (0-580) to sample for.
            If None, samples for all 581 stations.

        Returns
        -------
        np.ndarray, shape (n_scenarios, 581)
            Catch values per scenario per station ordinal.
            Stations not in station_ids (if specified) are set to 0.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .sample()")

        scenarios = np.zeros((n_scenarios, N_STATIONS))

        if station_ids is None:
            ids = np.arange(N_STATIONS)
        else:
            ids = np.asarray(station_ids)

        for sid in ids:
            raw = self.rng.normal(self.means[sid], self.stds[sid], n_scenarios)
            scenarios[:, sid] = np.maximum(raw, 0.0)  # no negative catch

        return scenarios

    def get_params(self):
        """Return fitted parameters for inspection/reproducibility.

        Returns
        -------
        pd.DataFrame with columns:
            ordinal, tow_station, mean, std, tier
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .get_params()")
        return pd.DataFrame({
            "ordinal": range(N_STATIONS),
            "tow_station": [self.ordinal_to_tow[i] for i in range(N_STATIONS)],
            "mean": self.means,
            "std": self.stds,
            "tier": self.tiers,
        })

    def summary(self):
        """Print a summary of fitted distributions by tier."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .summary()")
        params = self.get_params()
        print(f"\nCatchSimulator — {N_STATIONS} stations")
        print(f"  Seed: {self.seed}")
        for tier in [1, 2, 3]:
            subset = params[params["tier"] == tier]
            label = {1: "Direct historical", 2: "Regional fallback",
                     3: "Nearest neighbour"}[tier]
            print(f"\n  Tier {tier} — {label} ({len(subset)} stations):")
            print(f"    Mean catch: {subset['mean'].mean():.1f} "
                  f"(range {subset['mean'].min():.1f}–{subset['mean'].max():.1f})")
            print(f"    Mean std:   {subset['std'].mean():.1f}")
