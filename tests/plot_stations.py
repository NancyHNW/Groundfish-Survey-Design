"""Plot all 581 stations and 13 ports on a map of Iceland."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _ROOT)

from unified.stochastic_eval import _load_nodes, _load_island

PORT_NAMES = [
    "Vestmannaeyjar", "Grindavík", "Sandgerði", "Hafnarfjörður",
    "Reykjavík", "Akranes", "Ísafjörður", "Sauðárkrókur",
    "Akureyri", "Húsavík", "Seyðisfjörður", "Neskaupstaður",
    "Hornafjörður",
]


def plot_map(highlight_ports=None):
    nodes, n_ports, n_stations = _load_nodes()
    island = _load_island()

    fig, ax = plt.subplots(figsize=(14, 10))

    # Coastline
    ax.plot(island[:, 1], island[:, 0], color="grey", linewidth=0.5)

    # Stations — plot tow-start points as small dots
    for s in range(n_stations):
        n1 = s * 2 + n_ports  # tow start
        ax.scatter(nodes[n1, 1], nodes[n1, 0],
                   c="#aaaaaa", s=8, zorder=2, alpha=0.6)

    # Ports — larger markers
    for p in range(n_ports):
        if highlight_ports and p in highlight_ports:
            ax.scatter(nodes[p, 1], nodes[p, 0],
                       c="red", marker="*", s=300, zorder=10,
                       edgecolors="black", linewidths=0.5)
        else:
            ax.scatter(nodes[p, 1], nodes[p, 0],
                       c="#22A884", marker="s", s=80, zorder=5,
                       edgecolors="black", linewidths=0.5)
        ax.annotate(f"  {p}: {PORT_NAMES[p]}", (nodes[p, 1], nodes[p, 0]),
                    fontsize=7, zorder=11)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = "Iceland Groundfish Survey — 581 Stations, 13 Ports"
    if highlight_ports:
        names = [f"{p} ({PORT_NAMES[p]})" for p in highlight_ports]
        title += f"\nHome ports: {', '.join(names)}"
    ax.set_title(title)
    ax.set_aspect("equal")

    save_path = os.path.join(_ROOT, "tests", "outputs", "station_map.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--home-ports", type=int, nargs="+", default=None,
                        help="Highlight these port indices (e.g. --home-ports 4 8)")
    args = parser.parse_args()
    plot_map(highlight_ports=args.home_ports)
