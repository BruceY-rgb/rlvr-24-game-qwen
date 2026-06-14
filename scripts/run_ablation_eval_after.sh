#!/usr/bin/env bash
# Watcher: wait until ablation TRAINING finishes, then evaluate every ablation
# model on dev + unsolvable (2 NPUs, 2 batches), then aggregate everything.
set -uo pipefail

cd /home/ma-user/work/24game
export PYTHONPATH=$PWD/src HF_HOME=/home/ma-user/work/.cache/huggingface

echo "[$(date +%H:%M:%S)] watcher: waiting for ALL_ABLATION_TRAIN_DONE ..."
while ! grep -q "ALL_ABLATION_TRAIN_DONE" logs/ablation_all.log 2>/dev/null; do
  sleep 120
done
echo "[$(date +%H:%M:%S)] training done -> evaluating ablation models"

eval_model() {  # name device
  local name="$1" dev="$2"
  local adapter="outputs/ablation/${name}"
  # fall back to the last step checkpoint if the root adapter wasn't written
  if [ ! -f "${adapter}/adapter_config.json" ]; then
    local cand
    cand=$(ls -d outputs/ablation/${name}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    [ -n "$cand" ] && adapter="$cand"
  fi
  echo "[$(date +%H:%M:%S)] eval ${name} (adapter=${adapter}) on card ${dev}"
  for split in dev unsolvable; do
    ASCEND_RT_VISIBLE_DEVICES=$dev python -m twentyfour_rl.eval \
      --dataset "data/processed/${split}.jsonl" \
      --output "outputs/eval/ab_${name}/${split}.jsonl" \
      --metrics-output "outputs/eval/ab_${name}/${split}_metrics.json" \
      --model-name "ab_${name}" --model-path Qwen/Qwen2.5-1.5B-Instruct \
      --adapter-path "$adapter" --k 4 --temperature 0.8 --batch-size 96 --max-new-tokens 256
  done
}

# batch 1
eval_model full     0 > logs/abeval_full.log     2>&1 &
E1=$!
eval_model no_legal 1 > logs/abeval_no_legal.log 2>&1 &
E2=$!
wait $E1 $E2
# batch 2
eval_model no_format    0 > logs/abeval_no_format.log    2>&1 &
E3=$!
eval_model no_closeness 1 > logs/abeval_no_closeness.log 2>&1 &
E4=$!
wait $E3 $E4

echo "[$(date +%H:%M:%S)] aggregating final summary"
python3 scripts/aggregate_results.py --eval-root outputs/eval \
  --out-md outputs/eval/summary_final.md --out-csv outputs/eval/summary_final.csv
echo "[$(date +%H:%M:%S)] ALL_DONE_ABLATION_PIPELINE"
