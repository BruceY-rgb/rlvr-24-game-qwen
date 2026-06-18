#!/usr/bin/env bash
# Round-2 full evaluation orchestration on a single NPU: trained then base.
set -uo pipefail

cd /home/ma-user/work/24game
export PYTHONPATH=$PWD/src HF_HOME=/home/ma-user/work/.cache/huggingface
mkdir -p logs outputs/eval_v2

echo "[$(date +%H:%M:%S)] EVAL_V2 START"
bash scripts/run_full_eval.sh grpo1400 "--adapter-path outputs/final_grpo/checkpoint-1400" outputs/eval_v2/trained
bash scripts/run_full_eval.sh base "" outputs/eval_v2/base
echo "[$(date +%H:%M:%S)] EVAL_V2_MODELS_DONE"
