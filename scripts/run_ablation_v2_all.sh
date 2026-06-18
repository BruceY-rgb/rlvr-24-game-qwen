#!/usr/bin/env bash
# Round-2 ablation: a full-reward baseline plus dropping each of the 5 ablatable
# components, TRL GRPO, single NPU, run serially. All runs share one config so
# the comparison is internally controlled.
# Env: STEPS (default 400), NUMGEN (default 4).
set -uo pipefail

cd /home/ma-user/work/24game
export PYTHONPATH=$PWD/src HF_HOME=/home/ma-user/work/.cache/huggingface
STEPS="${STEPS:-400}"
NUMGEN="${NUMGEN:-4}"
mkdir -p logs outputs/ablation_v2

echo "[$(date +%H:%M:%S)] ABLATION_V2 START steps=${STEPS} numgen=${NUMGEN}"

train() {  # name disable_arg
  local name="$1" dis="$2"
  echo "[$(date +%H:%M:%S)] >>> TRAIN ${name} ${dis}"
  python -m twentyfour_rl.train_grpo_ascend \
    --trainer trl \
    --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
    --train-file data/processed/train.jsonl \
    --dev-file data/processed/dev.jsonl \
    --output-dir "outputs/ablation_v2/${name}" \
    --max-steps "${STEPS}" --num-generations "${NUMGEN}" --gradient-checkpointing \
    ${dis} > "logs/ab_v2_${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] <<< DONE ${name}"
}

train full        ""
train no_legal     "--disable-rewards legal"
train no_format    "--disable-rewards format"
train no_closeness "--disable-rewards closeness"
train no_refusal   "--disable-rewards refusal"
train no_penalty   "--disable-rewards penalty"

echo "[$(date +%H:%M:%S)] ALL_ABLATION_V2_DONE"
