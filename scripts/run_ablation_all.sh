#!/usr/bin/env bash
# Orchestrate all ablation trainings on 2 NPUs in 2 batches.
#   batch 1: full (baseline)         on card0 | no_legal     on card1
#   batch 2: no_format               on card0 | no_closeness on card1
# Each model trains ~STEPS steps (default 400, aligned with checkpoint-400).
set -uo pipefail

cd /home/ma-user/work/24game
export PYTHONPATH=$PWD/src HF_HOME=/home/ma-user/work/.cache/huggingface
STEPS="${STEPS:-400}"
mkdir -p logs outputs/ablation

echo "[$(date +%H:%M:%S)] ABLATION START steps=${STEPS}"

# ---- batch 1 ----
ASCEND_RT_VISIBLE_DEVICES=0 STEPS=$STEPS nohup bash scripts/run_ablation_train.sh full "" \
  > logs/ab_full.log 2>&1 &
P1=$!
ASCEND_RT_VISIBLE_DEVICES=1 STEPS=$STEPS nohup bash scripts/run_ablation_train.sh no_legal "--disable-rewards legal" \
  > logs/ab_no_legal.log 2>&1 &
P2=$!
wait $P1; echo "[$(date +%H:%M:%S)] batch1: full done"
wait $P2; echo "[$(date +%H:%M:%S)] batch1: no_legal done"

# ---- batch 2 ----
ASCEND_RT_VISIBLE_DEVICES=0 STEPS=$STEPS nohup bash scripts/run_ablation_train.sh no_format "--disable-rewards format" \
  > logs/ab_no_format.log 2>&1 &
P3=$!
ASCEND_RT_VISIBLE_DEVICES=1 STEPS=$STEPS nohup bash scripts/run_ablation_train.sh no_closeness "--disable-rewards closeness" \
  > logs/ab_no_closeness.log 2>&1 &
P4=$!
wait $P3; echo "[$(date +%H:%M:%S)] batch2: no_format done"
wait $P4; echo "[$(date +%H:%M:%S)] batch2: no_closeness done"

echo "[$(date +%H:%M:%S)] ALL_ABLATION_TRAIN_DONE"
