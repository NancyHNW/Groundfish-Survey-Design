# Archived report reproduction files

These files were used to generate and refresh the report's Chapter 4 results.
They are retained for research provenance but are not part of the normal
application workflow.

## Contents

- `scripts/` contains the long-running batch runners:
  - `run_chapter4.sh` runs the complete Chapter 4 chain.
  - `run_chapter4b.sh` resumes the later part of that chain.
  - `run_refresh.sh` refreshes results affected by the tow-end correction.
  - `run_refresh_all.sh` refreshes the remaining report results.
- `plans/` contains `repro_a.json` and `repro_b.json`, the duplicate plans
  used to verify that the full-survey solve was reproducible.

The active saved plans remain in `tests/plans/`:

- `full_default.json` is the reproducible full 581-station plan used by the
  report.
- `ns100_default.json` is a smaller cached plan for demonstrations and quick
  experiments.

The archived shell scripts assume they are run from the repository root. If
the report needs to be regenerated, copy the required script back to the root
or update its `cd` and path handling before running it.
