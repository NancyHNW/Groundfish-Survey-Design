"""Combine per-seed paired runs into one paired comparison.

A 20-solve run is a single point of failure; five 4-solve runs are not. Run
`paired` one seed at a time, then combine the `*_blocks.csv` files here. The
result is identical to what one five-seed run would have produced.

    python -m experiments.paired --compare buffer --full \
        --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 1000 \
        --solver-seeds 42        # repeat for 52 62 72 82

    python -m experiments.combine_blocks \
        "tests/outputs/paired_buffer-tabu_move_full_s*_blocks.csv"
"""

import csv
import glob
import math
import sys

from experiments.common import PAIRED_COLUMNS, _t_critical, print_table


def load(pattern):
    """Read every matching blocks file into {setting: {block: mean}}."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files matched {pattern!r}")
    by_setting = {}
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # A block is (instance, seed); keep both so two files cannot
                # collide on instance alone.
                block = (row.get("instance", "1"), row.get("seed", "?"))
                by_setting.setdefault(row["setting"], {})[block] = \
                    float(row["mean"])
    print(f"combined {len(files)} file(s): "
          + ", ".join(p.rsplit("\\", 1)[-1] for p in files))
    return by_setting


def compare(by_setting, baseline):
    """Paired differences against the baseline, over the shared blocks."""
    if baseline not in by_setting:
        raise SystemExit(f"baseline {baseline!r} not among "
                         f"{sorted(by_setting)}")
    base = by_setting[baseline]
    rows = []
    for setting, vals in by_setting.items():
        blocks = sorted(set(vals) & set(base))
        diffs = [vals[b] - base[b] for b in blocks]
        n = len(diffs)
        mean = sum(diffs) / n if n else 0.0
        if n > 1:
            sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
            half = _t_critical(n - 1) * sd / math.sqrt(n)
        else:
            sd = half = 0.0
        rows.append({
            "setting": setting, "n": n,
            "mean": sum(vals[b] for b in blocks) / n if n else 0.0,
            "diff": mean, "sd": sd,
            "ci_lo": mean - half, "ci_hi": mean + half,
            "significant": bool(n > 1 and (mean - half) * (mean + half) > 0),
            "infeasible_blocks": 0,
        })
    rows.sort(key=lambda r: r["diff"])
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    pattern = sys.argv[1]
    baseline = sys.argv[2] if len(sys.argv) > 2 else None
    data = load(pattern)
    if baseline is None:
        # buffer_1.0 for a buffer sweep, backtrack for a strategy sweep
        for guess in ("buffer_1.0", "backtrack", "grasp_only"):
            if guess in data:
                baseline = guess
                break
        else:
            baseline = sorted(data)[0]
    rows = compare(data, baseline)
    print_table(rows, PAIRED_COLUMNS,
                title=f"COMBINED PAIRED COMPARISON  (baseline {baseline}, "
                      f"{rows[0]['n']} blocks)")
