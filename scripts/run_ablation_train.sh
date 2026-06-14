#!/usr/bin/env bash
# Train ONE ablation model (TRL GRPO), aligned with the original checkpoint-400
# hyperparameters, optionally zeroing a reward component.
# Usage: run_ablation_train.sh <name> <disable_arg>
#   run_ablation_train.sh full         ""
#   run_ablation_train.sh no_legal     "--disable-rewards legal"
#   run_ablation_train.sh no_format    "--disable-rewards format"
#   run_ablation_train.sh no_closeness "--disable-rewards closeness"
# Env: STEPS (default 400), OUT_ROOT (default outputs/ablation).
set -uo pipefail

NAME="$1"
DISABLE="$2"
STEPS="${STEPS:-400}"
OUT_ROOT="${OUT_ROOT:-outputs/ablation}"
OUT="${OUT_ROOT}/${NAME}"
mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] TRAIN ablation=${NAME} disable='${DISABLE}' steps=${STEPS} -> ${OUT}"
python -m twentyfour_rl.train_grpo \
  --trainer trl \
  --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --train-file data/processed/train.jsonl \
  --dev-file data/processed/dev.jsonl \
  --output-dir "$OUT" \
  --max-steps "$STEPS" \
  --num-generations 4 \
  --gradient-checkpointing \
  ${DISABLE}
echo "[$(date +%H:%M:%S)] TRAIN_DONE ${NAME}"
