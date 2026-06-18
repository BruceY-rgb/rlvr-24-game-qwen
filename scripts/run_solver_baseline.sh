#!/usr/bin/env bash
# Exact-search solver baseline across all splits (CPU, deterministic, k=1).
set -uo pipefail

OUTDIR="${1:-outputs/eval_v2/solver}"
mkdir -p "$OUTDIR"

for split in dev ood_test hard_test unsolvable countdown_ood countdown_unique; do
  echo "[$(date +%H:%M:%S)] solver ${split}"
  python -m twentyfour_rl.eval \
    --dataset "data/processed/${split}.jsonl" \
    --output "${OUTDIR}/${split}.jsonl" \
    --metrics-output "${OUTDIR}/${split}_metrics.json" \
    --model-name exact-search --solver-baseline --k 1
done

echo "[$(date +%H:%M:%S)] SOLVER_DONE"
