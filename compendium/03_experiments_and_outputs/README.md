# Experiment register

Experiment IDs are stable descriptions of research questions. Output
filenames may include timestamps; those timestamps identify a particular run
and are not part of the experiment definition.

The final report's post-fix results are listed in
[`report_output_register.csv`](report_output_register.csv). Exact report
artifacts should be retained even when a future rerun produces a new
timestamped filename.

## Reproduction conventions

- Run commands from the repository root.
- Use `--plan tests/plans/full_default.json` when the report used the saved
  full-survey route.
- Record scenario count, scenario seed, planner seed, method, policy, buffer,
  and home ports.
- Do not compare results scored with different scenario counts as paired
  observations.

