# Data catalogue

The repository keeps canonical data in the existing source folders. This
section describes its role and provenance rather than copying large binary
files.

## Required research inputs

| File | Location | Status | Use |
|---|---|---|---|
| `spring_historical_english.xlsx` | `simulations/data/` | Raw historical input | Fit station catch distributions and draw scenarios |
| `smb.2019.dat` | `gfsp_code/data/` | Inherited survey input | Station/tow geometry and labels |
| `ports.csv` | `final_code/data/` and inherited solver data | Input | Port positions |
| `island.bin` | `final_code/data/` | Input | Iceland coastline plots |
| `updated_true_dist.npy` | `final_code/local data/` | Rebuild input | Clean upper-triangular distance source |
| `true_time.npy` | `final_code/local data/` | Active derived input | Travel-time scoring and heuristic routing |
| `true_dist.npy` | `gfsp_code/data/` | Inherited solver input | GFSP distance-based workflows |
| `full_default.json` | `tests/plans/` | Saved plan | Main full-survey route |
| `ns100_default.json` | `tests/plans/` | Saved plan | Small demonstration route |

## Catch provenance

There are records for 581 stations. 518 stations use their own historical
records; 63 stations inherit the nearest station with available data. Catch
scenarios use the historical fitted distributions. Inherited catch files are
retained for compatibility with the original solvers, not as the source of
the report's stochastic results.

## Travel-matrix provenance

`true_time.npy` was rebuilt on 2026-10-08 from the clean distance source using
`unified.rebuild_matrix`. The superseded matrix had doubled distances, a
corrupted port-0 connection, and triangle-inequality violations. Results
produced before the rebuild are not directly comparable with the report's
post-fix results.

## Data handling requirements

For each new derived file, record the source file, processing script, date,
units, dimensions, and whether it is required for normal execution. Large
generated outputs should be referenced by the experiment register rather than
duplicated here.

