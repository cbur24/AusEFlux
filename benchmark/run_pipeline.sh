#!/usr/bin/env bash
# Run the new AusEFlux pipeline end-to-end on the benchmark bbox.
#   step1 (harmonisation) for every climatology year, in parallel
#   step2 (feature datasets), once
#   step3 (ensemble predict + combine) for the target year
#
# Usage: bash run_pipeline.sh [TARGET_YEAR] [CLIM_START] [CLIM_END] [N_PARALLEL]
set -euo pipefail

PIPE=/home/prl900/Downloads/auseflux_pipeline
TARGET_YEAR="${1:-2020}"
CLIM_START="${2:-2002}"
CLIM_END="${3:-2022}"
NP="${4:-4}"
LOG=/home/prl900/Downloads/wald_eflux/logs
mkdir -p "$LOG"

cd "$PIPE"

echo "### STEP 1: harmonisation ${CLIM_START}-${CLIM_END} (${NP} parallel) ###"
seq "$CLIM_START" "$CLIM_END" | xargs -P "$NP" -I {} bash -c \
  'echo "  start step1 {}"; uv run step1_harmonisation.py --year {} \
     > '"$LOG"'/step1_{}.log 2>&1 && echo "  done step1 {}" || echo "  FAIL step1 {}"'

echo "### STEP 2: feature datasets ###"
uv run step2_feature_datasets.py > "$LOG/step2.log" 2>&1 && echo "  done step2" || { echo "  FAIL step2"; tail -20 "$LOG/step2.log"; exit 1; }

echo "### STEP 3: ensemble prediction + combination (${TARGET_YEAR}) ###"
uv run step3_ensemble_prediction.py --year "$TARGET_YEAR" --verbose \
  > "$LOG/step3_${TARGET_YEAR}.log" 2>&1 && echo "  done step3" || { echo "  FAIL step3"; tail -30 "$LOG/step3_${TARGET_YEAR}.log"; exit 1; }

echo "### PIPELINE COMPLETE ###"
ls -la /home/prl900/Downloads/wald_eflux/results/AusEFlux/*/ 2>/dev/null
