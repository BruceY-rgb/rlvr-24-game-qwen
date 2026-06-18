#!/usr/bin/env bash
# Round-2 reduced ablation after the idle-stop interruption: full baseline plus
# the three most informative drops (legal, refusal, penalty). Shorter runs so a
# single segment fits within the Notebook auto-stop window.
# Env: STEPS (default 200), NUMGEN (default 4).
set -uo pipefail

cd /home/ma-user/work/24game
export PYTHONPATH=$PWD/src HF_HOME=/home/ma-user/work/.cache/huggingface
STEPS="${STEPS:-200}"
NUMGEN="${NUMGEN:-4}"
mkdir -p logs outputs/ablation_v2_key

echo "[$(date +%H:%M:%S)] ABLATION_V2_KEY START steps=${STEPS} numgen=${NUMGEN}"

train() {  # name disable_arg
  local name="$1" dis="$2"
  echo "[$(date +%H:%M:%S)] >>> TRAIN ${name} ${dis}"
  python -m twentyfour_rl.train_grpo_ascend \
    --trainer trl \
    --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
    --train-file data/processed/train.jsonl \
    --dev-file data/processed/dev.jsonl \
    --output-dir "outputs/ablation_v2_key/${name}" \
    --max-steps "${STEPS}" --num-generations "${NUMGEN}" --gradient-checkpointing \
    ${dis} > "logs/abk_${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] <<< DONE ${name}"
}

train full      ""
train no_legal   "--disable-rewards legal"
train no_refusal "--disable-rewards refusal"
train no_penalty "--disable-rewards penalty"

echo "[$(date +%H:%M:%S)] ALL_ABLATION_V2_KEY_DONE"
