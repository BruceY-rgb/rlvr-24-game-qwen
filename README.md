# 24 Game RLVR on ModelArts Ascend

This repo implements the course project scaffold for training a small open LLM
to solve the 24 game with verifiable rewards.

## 大文件 / 完整产物（浙大云盘）

模型权重、消融 adapter、完整评估原始结果等大文件不纳入 git 仓库，存放在浙大云盘。

第二轮（checkpoint-1400 与缩减消融）
- 链接：<https://pan.zju.edu.cn/apps/files/desktop/files/folder/455006616665?isopen=1>
- 上传日期：2026-06-19
- 内容：4 个消融模型 adapter（full、no_legal、no_refusal、no_penalty，各 200 步）。评估结果与分析见 docs/round2_report.md。

第一轮（checkpoint-400）
- 链接：<https://pan.zju.edu.cn/apps/files/desktop/files/folder/455006606692?isopen=1>
- 上传日期：2026-06-14
- 内容：第一轮 LoRA adapter 与 optimizer、评估结果、第一轮报告。

仓库内只保留代码、脚本、文档与小评估指标。

## What is included

- Safe expression verifier: checks format, allowed operators, card usage,
  division by zero, and whether the expression equals 24.
- Reward function for GRPO/RLVR: exact correctness, legality, R1 format, and
  numeric closeness.
- Data preparation for `nlile/24-game`, `test-time-compute/game-of-24`,
  and optional Countdown 3-to-4 number OOD extensions.
- Training entrypoint with TRL `GRPOTrainer` and a manual grouped RL fallback
  for Ascend compatibility.
- Evaluation script that writes `eval_results.jsonl/csv` and metrics JSON.
- Streamlit dashboard for single-case solving, training curves, model
  comparison, and failure analysis.

## Datasets

Main experiment:

- `nlile/24-game`: train/dev source for solvable 24-game puzzles.
- `test-time-compute/game-of-24`: Tree-of-Thoughts style held-out and hard
  24-game evaluation.
- Generated unsolvable 24-game split: if the current Hugging Face copy of
  `nlile/24-game` has no `solvable=False` rows, the data script enumerates
  1-13 card multisets and uses exact search to create `unsolvable.jsonl`.

Extension / bonus experiment:

- `Jiayi-Pan/Countdown-Tasks-3to4`: 3-to-4 numbers with arbitrary target.
- `Jiayi-Pan/Countdown-Tasks-3to4-Unique`: unique Countdown variant for a
  stricter OOD check.

Prepared files:

```text
data/processed/train.jsonl
data/processed/dev.jsonl
data/processed/ood_test.jsonl
data/processed/hard_test.jsonl
data/processed/unsolvable.jsonl
data/processed/countdown_ood.jsonl
data/processed/countdown_unique.jsonl
```

## Local smoke test

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m twentyfour_rl.data \
  --output-dir data/processed \
  --include-countdown \
  --allow-synthetic
PYTHONPATH=src python -m twentyfour_rl.eval \
  --dataset data/processed/dev.jsonl \
  --output outputs/eval_results.jsonl \
  --metrics-output outputs/eval_metrics.json \
  --model-name exact-search \
  --solver-baseline \
  --k 1
streamlit run streamlit_app.py
```

## ModelArts Ascend workflow

Use a ModelArts Ascend PyTorch image where PyTorch, `torch-npu`, and CANN are
already matched. Do not reinstall `torch` or `torch-npu` unless your platform
image documentation explicitly requires it.

Suggested OBS layout:

```text
obs://bucket/24game/code/
obs://bucket/24game/data/
obs://bucket/24game/models/
obs://bucket/24game/logs/
obs://bucket/24game/outputs/
```

Notebook phase:

```bash
pip install -r requirements-modelarts.txt
export PYTHONPATH=$PWD/src
python -m unittest discover -s tests
python -m twentyfour_rl.data --output-dir data/processed --include-countdown
python -m twentyfour_rl.train_grpo \
  --trainer manual \
  --train-file data/processed/train.jsonl \
  --dev-file data/processed/dev.jsonl \
  --output-dir outputs/sanity \
  --max-steps 20 \
  --num-generations 4 \
  --gradient-checkpointing
```

Formal training job boot command:

```bash
export PYTHONPATH=$PWD/src
bash scripts/modelarts_bootstrap.sh
```

If TRL is compatible in the selected image, keep `TRAINER=trl`. If it fails on
Ascend, run with `TRAINER=manual`; the fallback uses grouped sampled completions
and normalized group advantages with the same verifier reward.

## Evaluation

Run base and trained adapters with the same sampling setup:

```bash
export PYTHONPATH=$PWD/src
python -m twentyfour_rl.eval \
  --dataset data/processed/ood_test.jsonl \
  --output outputs/eval_results.jsonl \
  --metrics-output outputs/eval_metrics.json \
  --model-name grpo \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path outputs/qwen24-grpo \
  --k 4
```

Optional Countdown OOD evaluation after preparing data with
`--include-countdown`:

```bash
export PYTHONPATH=$PWD/src
python -m twentyfour_rl.eval \
  --dataset data/processed/countdown_ood.jsonl \
  --output outputs/countdown_ood_results.jsonl \
  --metrics-output outputs/countdown_ood_metrics.json \
  --model-name grpo-countdown-ood \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path outputs/qwen24-grpo \
  --k 4
```

For the dashboard, point the sidebar to:

- `outputs/qwen24-grpo/train_metrics.jsonl`
- `outputs/eval_results.jsonl`
- `outputs/eval_metrics.json`
