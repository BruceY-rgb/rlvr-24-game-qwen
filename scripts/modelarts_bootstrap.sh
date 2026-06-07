#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_DIR}"
echo "[ModelArts] Python: $(${PYTHON_BIN} --version)"
echo "[ModelArts] Project: ${PROJECT_DIR}"

${PYTHON_BIN} -m pip install -r requirements-modelarts.txt
${PYTHON_BIN} - <<'PY'
import torch
print("[ModelArts] torch", torch.__version__)
try:
    import torch_npu
    print("[ModelArts] torch_npu imported")
    print("[ModelArts] npu available", torch.npu.is_available())
except Exception as exc:
    print("[ModelArts] torch_npu import failed:", exc)
PY

DATA_ARGS=(--output-dir "${DATA_DIR:-data/processed}")
if [ "${INCLUDE_COUNTDOWN:-1}" = "1" ]; then
  DATA_ARGS+=(--include-countdown --countdown-max-samples "${COUNTDOWN_MAX_SAMPLES:-2000}")
fi
${PYTHON_BIN} -m twentyfour_rl.data "${DATA_ARGS[@]}"

${PYTHON_BIN} -m twentyfour_rl.train_grpo \
  --trainer "${TRAINER:-trl}" \
  --model-name-or-path "${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-1.5B-Instruct}" \
  --train-file "${TRAIN_FILE:-data/processed/train.jsonl}" \
  --dev-file "${DEV_FILE:-data/processed/dev.jsonl}" \
  --output-dir "${OUTPUT_DIR:-outputs/qwen24-grpo}" \
  --gradient-checkpointing \
  --num-generations "${NUM_GENERATIONS:-4}" \
  --max-steps "${MAX_STEPS:--1}"
