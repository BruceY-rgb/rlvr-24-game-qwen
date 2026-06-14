#!/usr/bin/env bash
# Evaluate ONE model across all splits with per-split k/temperature.
# Usage: run_full_eval.sh <model_name> <adapter_arg> <out_dir>
#   adapter_arg: "" for base model, or "--adapter-path checkpoint-400" for trained.
# Env: MODEL_PATH (default Qwen base), BS (batch size, default 24).
set -uo pipefail

MODEL_NAME="$1"
ADAPTER_ARG="$2"
OUTDIR="$3"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
BS="${BS:-24}"
mkdir -p "$OUTDIR"

run() {  # split k temperature
  local split="$1" k="$2" temp="$3"
  echo "[$(date +%H:%M:%S)] >>> ${MODEL_NAME} ${split} k=${k} temp=${temp}"
  python -m twentyfour_rl.eval \
    --dataset "data/processed/${split}.jsonl" \
    --output "${OUTDIR}/${split}.jsonl" \
    --metrics-output "${OUTDIR}/${split}_metrics.json" \
    --model-name "${MODEL_NAME}" --model-path "${MODEL_PATH}" ${ADAPTER_ARG} \
    --k "${k}" --temperature "${temp}" --batch-size "${BS}" --max-new-tokens 256
  echo "[$(date +%H:%M:%S)] <<< done ${split}"
}

# Small splits: sampling, pass@4
run dev         4 0.8
run ood_test    4 0.8
run hard_test   4 0.8
run unsolvable  4 0.8
# Countdown OOD: full, greedy, pass@1 (large, so single deterministic pass)
run countdown_ood    1 0
run countdown_unique 1 0

echo "[$(date +%H:%M:%S)] ALL_DONE ${MODEL_NAME}"
