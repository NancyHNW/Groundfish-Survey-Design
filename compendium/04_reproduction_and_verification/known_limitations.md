# Known limitations

- Heuristic restart counts depend on wall-clock time and machine load.
- Scenario sets with different scenario counts are not nested.
- Table 4.1 was reported from a direct scoring run but does not currently have
  a dedicated CSV artifact.
- Table 4.8 came from a scratch `ceiling300.py` analysis that is not currently
  present in the repository.
- Historical pre-2026-10-08 outputs used a superseded travel-time matrix and
  should not be mixed with post-fix results.
- Inherited loaders and catch files remain for compatibility and are not all
  part of the current stochastic workflow.

