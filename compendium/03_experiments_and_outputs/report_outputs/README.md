# Report outputs

This folder contains curated copies of the final post-fix outputs used by the
project report. The working copies are generated under
`tests/outputs/`, which remains gitignored because it contains regenerated
outputs, logs, duplicates, and exploratory runs.

The timestamp in an output filename identifies the specific execution that
produced it. The stable experiment ID and report mapping are recorded in
[`../report_output_register.csv`](../report_output_register.csv).

## Contents

- `tables/` contains the CSV artifacts supporting report tables.
- `figures/` contains the selected report figures.
- `simulations/` contains the outputs of simulations. 

These files are evidence artifacts, not normal experiment inputs. To generate
new results, run the documented experiment modules from the repository root.
New runs will normally be written to `tests/outputs/` with a new timestamp.

## Deliberate exclusions

- Old, duplicate, exploratory, and superseded outputs are not copied here.
- Table 4.1 has no dedicated saved CSV in the current repository.
- Table 4.8 came from a missing scratch `ceiling300.py` analysis and has no
  reproducible artifact here.
