#!/usr/bin/env bash
# Steps 07, 09, 16 again, after the tow-end fix of 2026-10-16.
#
# Refreshes Table 4.1, Table 4.6 (both halves) and Table 4.7. Everything else
# from the 10-14/10-15 chain is unaffected: the fix only touches how a
# diversion is priced, and steps 06, 10-14 do not score a diversion-sensitive
# quantity differently.
#
#   bash run_refresh.sh
#
# 07 and 09 load tests/plans/full_default.json, so only step 16 solves.

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

echo "######## refresh after the tow-end fix, $(date) ########" | tee -a "$STATUS"

# Guard: the fix must be in before anything is re-measured.
python -m pytest tests/test_strategies.py -q -k "diverts or prices_the_diversion" \
    > "$LOG/00-fix-guard.log" 2>&1
if [ $? -ne 0 ]; then
    echo "!! tow-end tests FAIL -- the fix is not in place. Aborting." \
        | tee -a "$STATUS"
    exit 1
fi
echo "    tow-end tests pass, fix confirmed in place" | tee -a "$STATUS"

# ---- R3 cost of uncertainty, Table 4.1 + Figures 4.2, 4.3 -----------------
step 07b-R3-overflow-map python -u -m experiments.overflow_map --full \
    --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 1000 \
    --plan tests/plans/full_default.json

# ---- R3 Figure 4.1 --------------------------------------------------------
step 08b-R3-realisation python -u -m experiments.realisation --full \
    --home-ports 0 6 10 12 --time-limit 300 --seed 7 --no-catch-table \
    --plan tests/plans/full_default.json

# ---- R6b + R6c, Tables 4.6 right and 4.7 ---------------------------------
step 09b-R6b-strategies python -u -m experiments.strategies --full \
    --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 500 \
    --plan tests/plans/full_default.json

# ---- R6a, Table 4.6 left (the long one) ----------------------------------
step 16b-R6a-paired-strategy python -u -m experiments.paired --compare strategy \
    --instances 1-30 --time-limit 60 --n-scenarios 500

echo "######## refresh finished $(date) ########" | tee -a "$STATUS"
