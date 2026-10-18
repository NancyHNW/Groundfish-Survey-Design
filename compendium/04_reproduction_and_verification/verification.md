# Verification

The repository contains automated tests in [`../../tests/`](../../tests/).
They cover problem loading, route conversion, deterministic evaluation,
stochastic catch, overflow handling, repair policies, confidence intervals,
and experiment helpers.

## Test commands

```powershell
python -m pytest -m "not full" -q
python -m pytest -q
```

The report appendix records 227 tests for the everyday non-full suite and 236
tests for the complete suite at the time of reporting. 
