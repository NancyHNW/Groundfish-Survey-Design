import numpy as np
from dataclasses import dataclass, field


N_PORTS = 13  # fixed for this dataset


@dataclass
class ProblemInstance:
    """Standardised representation of a Ground Fish Survey Problem instance.

    Station IDs are stored as 0-based ordinals (0-580).  Use the helper
    methods to convert to/from the raw node indices used by the distance
    and time matrices (ports 0-12, then station pairs at 13+).
    """

    station_ids: np.ndarray       # station ordinals, shape (ns,)
    port_ids: np.ndarray          # port node indices (0-12)
    n_boats: int
    capacities: np.ndarray        # per-vessel capacities, shape (n_boats,)
    home_port_index: int          # index *into* port_ids (e.g. 0 means port_ids[0])
    fish_time_limit: float        # hours — Saahil's spoilage constraint
    work_limit: int               # max stations per boat
    source: str = ""              # "gfsp" or "saahil"
    upper_bound: float = np.inf   # optional upper bound on objective

    # ---- conversion helpers ------------------------------------------------

    def station_ids_to_node_indices(self) -> np.ndarray:
        """Convert station ordinals to raw node indices (pairs)."""
        s = self.station_ids
        return np.stack([s * 2, s * 2 + 1]).T.reshape(-1) + N_PORTS

    @staticmethod
    def node_indices_to_station_ids(node_indices: np.ndarray) -> np.ndarray:
        """Convert raw node indices back to unique station ordinals."""
        return np.unique((node_indices - N_PORTS) // 2)

    @property
    def home_port(self) -> int:
        """Actual port node index used as home."""
        return int(self.port_ids[self.home_port_index])

    @property
    def ns(self) -> int:
        return len(self.station_ids)

    def __str__(self) -> str:
        return (
            f"ProblemInstance(source={self.source}, "
            f"stations={self.ns}, boats={self.n_boats}, "
            f"home_port={self.home_port}, "
            f"fish_time_limit={self.fish_time_limit})"
        )
