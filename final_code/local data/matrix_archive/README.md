# Archived travel-time matrices

Superseded versions of `../true_time.npy`, kept because results produced
before the rebuild used the matrix archived here. `true_time.npy` is also
tracked in git, so `git show <rev>:'final_code/local data/true_time.npy'` is an
alternative route to the same bytes.

## `true_time_PRE-REBUILD-20261008-1814.npy`

Replaced 2026-10-08 18:14 by `python -m unified.rebuild_matrix --write`.

- max leg **72.46 h** -> **32.52 h** (factor 2.23)
- rebuilt from `updated_true_dist.npy` (upper-triangular, complete, no corrupted legs)
- max distance 507.25 nm before metric closure, 455.32 nm after
- residual triangle violations 448, all tow legs (0 non-tow)

Why it was replaced -- three independent defects:

1. **Every distance doubled.** `final_code/data_loader.py` mirrors an upper-triangular matrix with `dist += dist.T`, but was reading `true_dist(temporary).npy`, which is already symmetric. Implied steaming speed was 7 kn against Dana's 14.
2. **Port 0 (Vestmannaeyjar) corrupted.** 135 station legs under 1 nm against a true 170-185 nm; 270 station-port legs shorter than the great circle, all of them port 0.
3. **Not a metric.** 52% of cells failed the triangle inequality (mean 12.8 h, max 45.9 h), so diverting via a port often scored faster than sailing direct. Half of all simulated overflow diversions cost nothing and a third shortened the route.
