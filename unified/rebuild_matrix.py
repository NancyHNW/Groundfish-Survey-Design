"""Rebuild true_time.npy from a clean distance source.

Reads `updated_true_dist.npy`, closes it under shortest paths, then applies the
two speeds. See "Defects found and corrected" in the repository README for what
was wrong with the shipped matrix and why.

Closure is applied to **distance, not time**. Distance is pure geometry, so a
shorter path past a station node is a legitimate correction. Time is not
speed-homogeneous -- tows run at 7 kn, transit at 14 -- so closing time would
let a route shortcut through another station's tow. The residual violations in
the finished time matrix are all tow legs, which is correct: a tow has to be
towed, so being slower than sailing around it is a real property.

    python -m unified.rebuild_matrix              # report, write nothing
    python -m unified.rebuild_matrix --write      # back up and replace

Writing backs the current matrix up first. Afterwards every result in the
project is stale, so re-run the suite.
"""

import argparse
import os
import shutil
from datetime import datetime

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_LOCAL = os.path.join(_ROOT, "final_code", "local data")

SOURCE = os.path.join(_LOCAL, "updated_true_dist.npy")
TIME_OUT = os.path.join(_LOCAL, "true_time.npy")
DIST_OUT = os.path.join(_LOCAL, "true_dist_fixed.npy")

# The replaced matrix is kept here with a note saying what it was and why it
# went. true_time.npy is tracked in git, so history already holds it -- but a
# binary in a commit is not something anyone will find in six months, and every
# number in the project up to the rebuild date was produced with it.
ARCHIVE = os.path.join(_LOCAL, "matrix_archive")

# Dana's speeds, in knots. Distances are nautical miles (arcdist uses
# r = 3437.905, the Earth's radius in nm -- its "miles" docstring is wrong), so
# nm / kn gives hours.
SPEED_TRAVEL = 14.0
SPEED_TOW = 7.0

N_PORTS = 13


def tow_mask(n, n_ports=N_PORTS):
    """True on the two nodes of one station, which is a tow and not a transit.

    Stations occupy nodes n_ports.. in pairs, so the tow legs are
    (n_ports, n_ports+1), (n_ports+2, n_ports+3), ... in both directions. Same
    index arithmetic as convert_dist_to_time.ipynb, kept identical so the
    rebuilt matrix is comparable with the shipped one.
    """
    i, j = np.indices((n, n))
    off_i, off_j = i - n_ports, j - n_ports
    return ((i >= n_ports) & (j >= n_ports)
            & (((off_i % 2 == 0) & (j == i + 1))
               | ((off_i % 2 == 1) & (j == i - 1))))


def symmetrise(path):
    """Mirror the upper-triangular source. -1 means "not stored"."""
    d = np.load(path).astype(float)
    tri = d[np.tril_indices_from(d, -1)]
    if not (tri == -1).all():
        raise ValueError(
            f"{os.path.basename(path)} is not upper-triangular -- its lower "
            f"triangle holds real values, so mirroring it would DOUBLE every "
            f"distance. That is the bug this script exists to fix; do not "
            f"point it at true_dist(temporary).npy.")
    d[d == -1] = 0.0
    d += d.T
    np.fill_diagonal(d, 0.0)
    # A genuinely missing leg reads as 0 after mirroring. inf lets the closure
    # route around it instead of treating it as free.
    off_diag = ~np.eye(len(d), dtype=bool)
    d[(d == 0) & off_diag] = np.inf
    return d


def close(d):
    """Floyd-Warshall. In place, O(n^3) but vectorised over the inner two."""
    for k in range(len(d)):
        np.minimum(d, d[:, k:k + 1] + d[k:k + 1, :], out=d)
    return d


def to_time(d):
    """Apply the two speeds. Tow legs are slow; everything else is transit."""
    return np.where(tow_mask(len(d)), d / SPEED_TOW, d / SPEED_TRAVEL)


def build():
    """Return (dist, time) rebuilt from SOURCE, plus a report dict."""
    d = symmetrise(SOURCE)
    before = float(d[np.isfinite(d)].max())
    missing = int(np.isinf(d).sum())
    close(d)
    t = to_time(d)

    tow = tow_mask(len(d))
    m = t.copy()
    close(m)
    viol = m < t - 1e-9
    return d, t, {
        "max_nm_before_closure": before,
        "max_nm_after_closure": float(d.max()),
        "missing_legs": missing,
        "max_hours": float(t.max()),
        "residual_violations": int(viol.sum()),
        "residual_non_tow": int((viol & ~tow).sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--write", action="store_true",
                   help="Replace true_time.npy (backs the current one up "
                        "first). Without this, nothing is written.")
    args = p.parse_args()

    old = np.load(TIME_OUT) if os.path.exists(TIME_OUT) else None
    dist, time, rep = build()

    print(f"source: {os.path.basename(SOURCE)}")
    print(f"  max distance   {rep['max_nm_before_closure']:8.2f} nm before "
          f"closure, {rep['max_nm_after_closure']:8.2f} nm after")
    print(f"  missing legs   {rep['missing_legs']}")
    print(f"  max time       {rep['max_hours']:8.2f} h")
    print(f"  residual triangle violations: {rep['residual_violations']} "
          f"({rep['residual_non_tow']} of them NOT tow legs)")
    if rep["residual_non_tow"]:
        print("  !! non-tow violations should be zero after closure. "
              "Something is wrong -- do not --write.")
    if old is not None:
        print(f"\ncurrent true_time.npy max {old.max():.2f} h  ->  rebuilt "
              f"{time.max():.2f} h  (factor {old.max() / time.max():.2f})")

    if not args.write:
        print("\nDry run. Pass --write to replace true_time.npy.")
        return

    if rep["residual_non_tow"]:
        raise SystemExit("refusing to write: non-tow violations remain")

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    os.makedirs(ARCHIVE, exist_ok=True)
    backup = os.path.join(ARCHIVE, f"true_time_PRE-REBUILD-{stamp}.npy")
    if os.path.exists(backup):
        raise SystemExit(f"{backup} already exists; refusing to overwrite an "
                         f"archived matrix")
    shutil.copy(TIME_OUT, backup)
    _write_provenance(stamp, old, time, rep)

    np.save(TIME_OUT, time)
    np.save(DIST_OUT, dist)
    print(f"\narchived   matrix_archive/{os.path.basename(backup)}")
    print(f"           matrix_archive/README.md  (provenance note)")
    print(f"wrote      {os.path.basename(TIME_OUT)}")
    print(f"wrote      {os.path.basename(DIST_OUT)}")
    print("\n!! Every stored result is now stale. Planned time roughly halves. "
          "Re-run the suite.")


def _write_provenance(stamp, old, new, rep):
    """Append a note recording what was replaced and why.

    Appends rather than overwrites, so a second rebuild does not erase the
    record of the first.
    """
    path = os.path.join(ARCHIVE, "README.md")
    new_file = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new_file:
            f.write("# Archived travel-time matrices\n\n"
                    "Superseded versions of `../true_time.npy`, kept because "
                    "every result produced before each rebuild date used the "
                    "matrix archived here. `true_time.npy` is also tracked in "
                    "git, so `git show <rev>:'final_code/local data/"
                    "true_time.npy'` is an alternative route to the same "
                    "bytes.\n")
        f.write(f"\n## `true_time_PRE-REBUILD-{stamp}.npy`\n\n")
        f.write(f"Replaced {datetime.now():%Y-%m-%d %H:%M} by "
                f"`python -m unified.rebuild_matrix --write`.\n\n")
        if old is not None:
            f.write(f"- max leg **{old.max():.2f} h** -> "
                    f"**{new.max():.2f} h** (factor "
                    f"{old.max() / new.max():.2f})\n")
        f.write(f"- rebuilt from `{os.path.basename(SOURCE)}` "
                f"(upper-triangular, complete, no corrupted legs)\n")
        f.write(f"- max distance {rep['max_nm_before_closure']:.2f} nm before "
                f"metric closure, {rep['max_nm_after_closure']:.2f} nm after\n")
        f.write(f"- residual triangle violations {rep['residual_violations']}, "
                f"all tow legs ({rep['residual_non_tow']} non-tow)\n\n")
        f.write("Why it was replaced -- three independent defects:\n\n"
                "1. **Every distance doubled.** `final_code/data_loader.py` "
                "mirrors an upper-triangular matrix with `dist += dist.T`, but "
                "was reading `true_dist(temporary).npy`, which is already "
                "symmetric. Implied steaming speed was 7 kn against Dana's "
                "14.\n"
                "2. **Port 0 (Vestmannaeyjar) corrupted.** 135 station legs "
                "under 1 nm against a true 170-185 nm; 270 station-port legs "
                "shorter than the great circle, all of them port 0.\n"
                "3. **Not a metric.** 52% of cells failed the triangle "
                "inequality (mean 12.8 h, max 45.9 h), so diverting via a port "
                "often scored faster than sailing direct. Half of all "
                "simulated overflow diversions cost nothing and a third "
                "shortened the route.\n")


if __name__ == "__main__":
    main()
