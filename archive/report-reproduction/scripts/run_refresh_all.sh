#!/usr/bin/env bash
# Everything still scored by pre-fix code: steps 06, 10, 11, 12, 13, 14.
#
# Tables 4.1, 4.6 and 4.7 were refreshed on 2026-10-16 (run_refresh.sh), and
# the ns100 variance row was measured after the fix. These six are the rest.
#
# Ordered cheapest-and-most-diagnostic first, so an interrupted run still
# answers the question: step 10 re-solves the same six plans deterministically
# and only the scoring differs, which tests whether the bias cancels at all.
#
#   bash run_refresh_all.sh          # about 10 hours

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

echo "######## full refresh after the tow-end fix, $(date) ########" \
    | tee -a "$STATUS"

# The fix must be in before anything is re-measured.
python -m pytest tests/test_strategies.py -q \
    -k "diverts or prices_the_diversion or charges_exactly" \
    > "$LOG/00-fix-guard-all.log" 2>&1
if [ $? -ne 0 ]; then
    echo "!! tow-end tests FAIL -- fix not in place. Aborting." | tee -a "$STATUS"
    exit 1
fi
echo "    6 tow-end tests pass, fix confirmed" | tee -a "$STATUS"

# Keep the pre-fix tables so every number can be diffed old against new.
mkdir -p tests/outputs/prefix_20261015
cp tests/outputs/variance_tabu_move_full.csv \
   tests/outputs/method-comparison_forward_full_sc1000_*.csv \
   tests/outputs/full-sweep_tabu_move_full_*.csv \
   tests/outputs/paired_method_ns100-nv2_s42*.csv \
   tests/outputs/paired_buffer-tabu_move_ns100-nv2*.csv \
   tests/outputs/paired_buffer-tabu_move_full*.csv \
   tests/outputs/prefix_20261015/ 2>/dev/null
echo "    pre-fix tables archived to tests/outputs/prefix_20261015/" \
    | tee -a "$STATUS"

# --- 10: Table 4.3. Same six plans, only the scoring differs, so this is the
#         controlled test of whether the bias cancels. ~32 min.
step 10c-R4b-monte-carlo python -u -m experiments.monte_carlo --all-methods \
    --full --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 1000 \
    --no-histograms

# --- 06: Table 3.3 and the planner-variability margin. ~54 min.
step 06c-R1-variance python -u -m experiments.variance --full \
    --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 1000 \
    --solver-seeds 42 52 62 72 82 92 102 112 122 132

# --- 11: Tables 4.5 and 4.4 row 2. ~1h36m.
step 11c-R5b-full-sweep python -u -m experiments.full_sweep --full \
    --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 1000 \
    --buffer-ref-strategy forward

# --- 12: Table 4.2. ~3h03m.
step 12c-R4a-paired-method python -u -m experiments.paired --compare method \
    --instances 1-30 --time-limit 60 --n-scenarios 500

# --- 13: Table 4.4 row 1. ~2h02m.
step 13c-R5a-paired-buffer python -u -m experiments.paired --compare buffer \
    --instances 1-30 --time-limit 60 --n-scenarios 500

# --- 14: Table 4.4 row 3. ~1h53m.
step 14c-R5c-paired-buffer-full python -u -m experiments.paired --compare buffer \
    --full --home-ports 0 6 10 12 --time-limit 300 --n-scenarios 1000 \
    --solver-seeds 42 52 62 72 82

echo "######## full refresh finished $(date) ########" | tee -a "$STATUS"
