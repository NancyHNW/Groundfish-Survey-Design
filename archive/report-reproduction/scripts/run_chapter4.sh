#!/usr/bin/env bash
# Every run in RUN_PLAN_COMMANDS.md, in order, one at a time.
#
# Rule 5: no parallelism. Restart counts depend on machine load, so two solves
# at once would corrupt every planned time in Chapter 4.
#
#   bash run_chapter4.sh
#
# Per-step logs in tests/logs/ch4/, status line appended to ch4_status.log.

cd "$(dirname "$0")" || exit 1
LOG=tests/logs/ch4
mkdir -p "$LOG" tests/plans
STATUS="$LOG/_status.log"

# Exit code must be captured on its own line. $(date) in the same echo
# overwrites $? and every step silently reports success.
step () {
    local id="$1"; shift
    local started
    started=$(date '+%H:%M:%S')
    echo "=== $id  start $started ===" | tee -a "$STATUS"
    "$@" > "$LOG/$id.log" 2>&1
    local rc=$?
    local ended
    ended=$(date '+%H:%M:%S')
    echo "    $id  rc=$rc  $started -> $ended" | tee -a "$STATUS"
    return 0          # never abort the chain; a dead step is one table, not twelve
}

echo "######## chapter 4 chain, started $(date) ########" | tee -a "$STATUS"

# ---- R0 verification ------------------------------------------------------
step 01-pytest python -m pytest -m "not full" -q
step 02-environment python -c "
import sys, platform, numpy, scipy, pandas, matplotlib
print('python  ', sys.version.split()[0])
print('platform', platform.platform())
print('machine ', platform.processor())
print('numpy   ', numpy.__version__)
print('scipy   ', scipy.__version__)
print('pandas  ', pandas.__version__)
print('mpl     ', matplotlib.__version__)
import gurobipy; print('gurobi  ', gurobipy.gurobi.version())
"

# ---- R0 reproducibility: two identical full solves -------------------------
step 03-repro-a python -u -m experiments.overflow_map --full \
    --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 50 \
    --plan tests/plans/repro_a.json
step 04-repro-b python -u -m experiments.overflow_map --full \
    --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 50 \
    --plan tests/plans/repro_b.json
step 05-repro-compare python -c "
import json
a = json.load(open('tests/plans/repro_a.json'))
b = json.load(open('tests/plans/repro_b.json'))
print('planned ', a['planned_time'], b['planned_time'])
print('trips   ', len(a['trips']), len(b['trips']))
print('restarts', a['restarts'], b['restarts'])
print('ROUTES IDENTICAL' if a['trips'] == b['trips'] else 'ROUTES DIFFER')
"

# The default plan is that same solve. Copying rather than re-solving keeps it
# provably the one the reproducibility check passed on.
cp tests/plans/repro_a.json tests/plans/full_default.json
echo "    default plan = repro_a" | tee -a "$STATUS"

# ---- R1 planner variability, Table 3.3 ------------------------------------
step 06-R1-variance python -u -m experiments.variance --full \
    --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 1000 \
    --solver-seeds 42 52 62 72 82 92 102 112 122 132

# ---- R3 cost of uncertainty, Table 4.1 + Figures 4.2, 4.3 ------------------
step 07-R3-overflow-map python -u -m experiments.overflow_map --full \
    --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 1000 \
    --plan tests/plans/full_default.json

# ---- R3 Figure 4.1, one season --------------------------------------------
step 08-R3-realisation python -u -m experiments.realisation --full \
    --home-ports 0 6 10 12 --time-limit 300 --seed 7 --no-catch-table \
    --plan tests/plans/full_default.json

# ---- R6b + R6c recourse on the full problem, Tables 4.6 right, 4.7 ---------
step 09-R6b-strategies python -u -m experiments.strategies --full \
    --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 500 \
    --plan tests/plans/full_default.json

# ---- R4b planner comparison, full problem, Table 4.3 ----------------------
step 10-R4b-monte-carlo python -u -m experiments.monte_carlo --all-methods \
    --full --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 1000 \
    --no-histograms

# ---- R5b buffer sweep, full problem, Tables 4.4 row 2 and 4.5 -------------
step 11-R5b-full-sweep python -u -m experiments.full_sweep --full \
    --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 1000 \
    --buffer-ref-strategy forward

# ---- R4a planner comparison, test problems, Table 4.2 ---------------------
step 12-R4a-paired-method python -u -m experiments.paired --compare method \
    --instances 1-30 --time-limit 60 --n-scenarios 500

# ---- R5a buffer, test problems, Table 4.4 row 1 ---------------------------
step 13-R5a-paired-buffer python -u -m experiments.paired --compare buffer \
    --instances 1-30 --time-limit 60 --n-scenarios 500

# ---- R5c buffer, full problem, five seeds, Table 4.4 row 3 ----------------
step 14-R5c-paired-buffer-full python -u -m experiments.paired --compare buffer \
    --full --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 1000 \
    --solver-seeds 42 52 62 72 82

# ---- R6a recourse, test problems, Table 4.6 left --------------------------
# Timing probe first. repair_boat_tabu is the slow one; one instance x30 is
# the projection. The full run follows regardless -- kill it if the probe
# says it will not finish.
step 15-R6a-probe python -u -m experiments.paired --compare strategy \
    --instances 1 --time-limit 60 --n-scenarios 500
step 16-R6a-paired-strategy python -u -m experiments.paired --compare strategy \
    --instances 1-30 --time-limit 60 --n-scenarios 500

echo "######## chain finished $(date) ########" | tee -a "$STATUS"
