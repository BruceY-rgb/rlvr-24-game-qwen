#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-grpo}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/qwen24-grpo}"
OUT_DIR="${OUT_DIR:-outputs/eval}"
mkdir -p "${OUT_DIR}"

python -m twentyfour_rl.eval \
  --dataset data/processed/dev.jsonl \
  --output "${OUT_DIR}/dev.jsonl" \
  --metrics-output "${OUT_DIR}/dev_metrics.json" \
  --model-name "${MODEL_NAME}" \
  --model-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --k "${K:-1}"

python -m twentyfour_rl.eval \
  --dataset data/processed/ood_test.jsonl \
  --output "${OUT_DIR}/ood_test.jsonl" \
  --metrics-output "${OUT_DIR}/ood_metrics.json" \
  --model-name "${MODEL_NAME}" \
  --model-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --k "${K:-1}"

python -m twentyfour_rl.eval \
  --dataset data/processed/hard_test.jsonl \
  --output "${OUT_DIR}/hard_test.jsonl" \
  --metrics-output "${OUT_DIR}/hard_metrics.json" \
  --model-name "${MODEL_NAME}" \
  --model-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --k "${K:-1}"

python -m twentyfour_rl.eval \
  --dataset data/processed/unsolvable.jsonl \
  --output "${OUT_DIR}/unsolvable.jsonl" \
  --metrics-output "${OUT_DIR}/unsolvable_metrics.json" \
  --model-name "${MODEL_NAME}" \
  --model-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --k "${K:-1}"

if [ -f data/processed/countdown_ood.jsonl ]; then
  python -m twentyfour_rl.eval \
    --dataset data/processed/countdown_ood.jsonl \
    --output "${OUT_DIR}/countdown_ood.jsonl" \
    --metrics-output "${OUT_DIR}/countdown_ood_metrics.json" \
    --model-name "${MODEL_NAME}" \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${ADAPTER_PATH}" \
    --k "${K:-1}"
fi

if [ -f data/processed/countdown_unique.jsonl ]; then
  python -m twentyfour_rl.eval \
    --dataset data/processed/countdown_unique.jsonl \
    --output "${OUT_DIR}/countdown_unique.jsonl" \
    --metrics-output "${OUT_DIR}/countdown_unique_metrics.json" \
    --model-name "${MODEL_NAME}" \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${ADAPTER_PATH}" \
    --k "${K:-1}"
fi
