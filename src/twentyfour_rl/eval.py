from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import read_jsonl, write_jsonl
from .prompts import build_prompt_text
from .rewards import score_output
from .solver import solve_24


def _import_torch():
    import torch

    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass
    return torch


def choose_device():
    torch = _import_torch()
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_name_or_path: str, adapter_path: str | None = None, dtype: str = "bf16"):
    torch = _import_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # decoder-only models need left padding so the completion tokens line up
    # across a padded batch in generate().
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    device = choose_device()
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_completion(
    model,
    tokenizer,
    device,
    numbers: list[int],
    target: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_text: str | None = None,
) -> str:
    torch = _import_torch()
    # Use the exact same plain-text prompt the model was trained on (TRL GRPO
    # feeds the dataset's "prompt" column verbatim). Wrapping it in the chat
    # template here would shift the input distribution and collapse the
    # learned <think>/<answer> format, badly under-reporting the model.
    prompt = prompt_text if prompt_text is not None else build_prompt_text(numbers, target)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def generate_completions_batch(
    model,
    tokenizer,
    device,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate completions for a batch of prompts at once (left-padded)."""
    torch = _import_torch()
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = encoded["input_ids"].shape[1]
    completions = generated[:, prompt_len:]
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in completions]


def solver_completion(numbers: list[int], target: int) -> str:
    expr = solve_24(numbers, target=target)
    if expr is None:
        return "<think>Exact search did not find a valid expression.</think><answer></answer>"
    return f"<think>Use exact arithmetic search to combine the four cards.</think><answer>{expr}</answer>"


def evaluate_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    model_name: str,
    model_path: str | None = None,
    adapter_path: str | None = None,
    solver_baseline: bool = False,
    k: int = 1,
    max_samples: int | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    dtype: str = "bf16",
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    if max_samples is not None:
        rows = rows[:max_samples]

    model = tokenizer = device = None
    if not solver_baseline:
        if model_path is None:
            raise ValueError("--model-path is required unless --solver-baseline is set.")
        model, tokenizer, device = load_model_and_tokenizer(model_path, adapter_path, dtype=dtype)

    # Flatten (sample x k attempts) into a task list so generation can batch
    # across both samples and attempts.
    tasks: list[tuple[dict[str, Any], list[int], int, str, int]] = []
    for sample in rows:
        numbers = [int(n) for n in sample["numbers"]]
        target = int(sample.get("target", 24))
        prompt_text = sample.get("prompt") or build_prompt_text(numbers, target)
        for attempt_idx in range(k):
            tasks.append((sample, numbers, target, prompt_text, attempt_idx))

    def build_row(sample, numbers, target, attempt_idx, raw_output):
        scored = score_output(numbers, raw_output, target)
        return {
            "sample_id": sample.get("id"),
            "numbers": numbers,
            "numbers_key": sample.get("numbers_key"),
            "target": target,
            "split": sample.get("split", Path(dataset_path).stem),
            "source": sample.get("source", ""),
            "solvable": sample.get("solvable", True),
            "model_name": model_name,
            "attempt_index": attempt_idx,
            "raw_output": raw_output,
            **scored,
        }

    outputs: list[dict[str, Any]] = []
    if solver_baseline:
        for sample, numbers, target, _prompt, attempt_idx in tasks:
            raw_output = solver_completion(numbers, target)
            outputs.append(build_row(sample, numbers, target, attempt_idx, raw_output))
    else:
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            prompts = [task[3] for task in batch]
            raws = generate_completions_batch(
                model, tokenizer, device, prompts, max_new_tokens, temperature, top_p
            )
            for (sample, numbers, target, _prompt, attempt_idx), raw_output in zip(batch, raws):
                outputs.append(build_row(sample, numbers, target, attempt_idx, raw_output))
            print(f"[eval] {min(start + batch_size, len(tasks))}/{len(tasks)} generated", flush=True)
    write_jsonl(output_path, outputs)
    write_csv(Path(output_path).with_suffix(".csv"), outputs)
    return outputs


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("model_name", "model")), str(row.get("split", "eval")))].append(row)

    metrics = []
    for (model_name, split), group in sorted(groups.items()):
        first_attempts = [r for r in group if int(r.get("attempt_index", 0)) == 0]
        by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_sample[str(row.get("sample_id", row.get("numbers_key")))].append(row)
        pass_at_k = sum(any(r["is_correct"] for r in attempts) for attempts in by_sample.values()) / max(len(by_sample), 1)
        pass_at_1 = sum(bool(r["is_correct"]) for r in first_attempts) / max(len(first_attempts), 1)
        legal_rate = sum(bool(r["is_valid"]) for r in first_attempts) / max(len(first_attempts), 1)
        format_rate = sum(float(r.get("format", 0.0)) for r in first_attempts) / max(len(first_attempts), 1)
        invalid_rate = 1.0 - legal_rate
        hallucination_rate = 0.0
        unsolvable = [r for r in first_attempts if not bool(r.get("solvable", True))]
        if unsolvable:
            hallucination_rate = sum(bool(r.get("answer")) for r in unsolvable) / len(unsolvable)
        metrics.append(
            {
                "model_name": model_name,
                "split": split,
                "samples": len(by_sample),
                "attempts": len(group),
                "pass_at_1": pass_at_1,
                "pass_at_k": pass_at_k,
                "legal_rate": legal_rate,
                "format_rate": format_rate,
                "invalid_rate": invalid_rate,
                "hallucination_rate": hallucination_rate,
            }
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a 24-game model or exact-search baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="outputs/eval_results.jsonl")
    parser.add_argument("--metrics-output", default="outputs/eval_metrics.json")
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--model-path")
    parser.add_argument("--adapter-path")
    parser.add_argument("--solver-baseline", action="store_true")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    rows = evaluate_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        model_name=args.model_name,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        solver_baseline=args.solver_baseline,
        k=args.k,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dtype=args.dtype,
        batch_size=args.batch_size,
    )
    metrics = aggregate_metrics(rows)
    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

