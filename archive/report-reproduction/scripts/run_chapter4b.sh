#!/usr/bin/env bash
# Steps 11-16 of RUN_PLAN_COMMANDS.md. Steps 1-10 completed clean on 10-14
# and their outputs are already in tests/outputs/ -- do not redo them.
#
#   bash run_chapter4b.sh
#
# Appends to the same status log as the first half.

cd "$(dirname "$0")" || exit 1
LOG=tests/logs/ch4
mkdir -p "$LOG"
STATUS="$LOG/_status.log"

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
    return 0
}

echo "######## resuming at step 11, $(date) ########" | tee -a "$STATUS"

# ---- R5b buffer sweep, full problem, Tables 4.4 row 2 and 4.5 -------------
step 11-R5b-full-sweep python -u -m experiments.full_sweep --full \
    --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 1000 \
    --buffer-ref-strategy forward

# ---- R4a planner comparison, test problems, Table 4.2 ---------------------
# The one that carries the planner ranking: R4b's -31.67 h sits inside the
# 49.9 h single-solve margin, so the 30-instance pairing is what settles it.
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
# Probe first so the projection is on record, then the full run. Kill 16 if
# the probe says it cannot finish; the other rows write out as it goes.
step 15-R6a-probe python -u -m experiments.paired --compare strategy \
    --instances 1 --time-limit 60 --n-scenarios 500
step 16-R6a-paired-strategy python -u -m experiments.paired --compare strategy \
    --instances 1-30 --time-limit 60 --n-scenarios 500

echo "######## chain finished $(date) ########" | tee -a "$STATUS"
