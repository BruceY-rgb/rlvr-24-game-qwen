from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any

# 添加包路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twentyfour_rl.data import read_jsonl
from twentyfour_rl.eval import choose_device
from twentyfour_rl.prompts import render_prompt
from twentyfour_rl.rewards import grpo_reward_func, score_output, compute_detailed_metrics


def _import_torch():
    import torch

    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass
    return torch


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def dtype_from_arg(torch, dtype: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(dtype, torch.bfloat16)


def build_lora_config(args):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
        bias="none",
    )


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def get_stage_metrics(epoch, total_epochs):
    """获取当前阶段的监控指标"""
    if epoch < total_epochs * 0.3:
        return {
            "stage": "format_learning",
            "focus": "Format and Legal Compliance",
            "expected_format_rate": 0.9,
            "expected_legal_rate": 0.9,
            "expected_exact_rate": 0.1,
        }
    elif epoch < total_epochs * 0.7:
        return {
            "stage": "problem_solving",
            "focus": "Exact Problem Solving",
            "expected_format_rate": 0.95,
            "expected_legal_rate": 0.95,
            "expected_exact_rate": 0.3,
        }
    else:
        return {
            "stage": "refinement",
            "focus": "Refusal and Hallucination Control",
            "expected_format_rate": 0.95,
            "expected_legal_rate": 0.95,
            "expected_exact_rate": 0.5,
            "expected_refusal_rate": 0.1,
        }


def run_manual_grouped_rl_enhanced(args) -> None:
    torch = _import_torch()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics_enhanced.jsonl"

    train_rows = read_jsonl(args.train_file)
    dev_rows = read_jsonl(args.dev_file) if args.dev_file else []
    model, ref_model, tokenizer, device = load_manual_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    # 记录每个阶段的指标
    stage_metrics_history = []

    for epoch in range(int(args.num_train_epochs)):
        stage_info = get_stage_metrics(epoch, args.num_train_epochs)
        print(f"\n=== Epoch {epoch + 1}/{args.num_train_epochs} - Stage: {stage_info['stage']} ===")
        print(f"Focus: {stage_info['focus']}")
        print(f"Expected - Format: {stage_info['expected_format_rate']}, Legal: {stage_info['expected_legal_rate']}, Exact: {stage_info['expected_exact_rate']}")

        random.shuffle(train_rows)
        for row_idx, row in enumerate(train_rows):
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            numbers = [int(n) for n in row["numbers"]]
            target = int(row.get("target", 24))
            prompt = render_prompt(tokenizer, numbers, target)

            # 根据训练阶段调整权重
            stage_weights = get_stage_weights(epoch, args.num_train_epochs)

            completions = generate_group(model, tokenizer, device, numbers, target, args)
            scored = [score_output(numbers, completion, target, weights=stage_weights) for completion in completions]

            # 计算详细指标
            if row_idx % 50 == 0:  # 每50个样本计算一次详细指标
                detailed_metrics = compute_detailed_metrics(scored)
                stage_metrics_history.append({
                    "epoch": epoch,
                    "step_in_epoch": row_idx,
                    "global_step": global_step,
                    **stage_info,
                    **detailed_metrics
                })

            rewards = torch.tensor([s["reward"] for s in scored], dtype=torch.float32, device=device)
            std = rewards.std(unbiased=False)
            advantages = rewards - rewards.mean()
            if float(std.detach().cpu()) > 1e-6:
                advantages = advantages / (std + 1e-6)

            policy_losses = []
            kl_penalties = []
            for completion, advantage in zip(completions, advantages):
                logprob = completion_logprob(model, tokenizer, device, prompt, completion)
                policy_losses.append(-advantage.detach() * logprob)
                if args.kl_coef > 0.0 and ref_model is not None:
                    kl_penalties.append(compute_kl_penalty(model, ref_model, tokenizer, device, prompt, completion))

            policy_loss = torch.stack(policy_losses).mean()
            kl_loss = torch.stack(kl_penalties).mean() if kl_penalties else torch.tensor(0.0, device=device)
            loss = (policy_loss + args.kl_coef * kl_loss) / args.gradient_accumulation_steps
            loss.backward()

            if (row_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # 基础指标
                exact = sum(s["exact"] for s in scored) / len(scored)
                legal = sum(s["legal"] for s in scored) / len(scored)
                fmt = sum(s["format"] for s in scored) / len(scored)
                refusal = sum(1 for s in scored if s.get("is_refusal", False)) / len(scored)
                avg_reward = sum(s["reward"] for s in scored) / len(scored)

                metrics = {
                    "step": global_step,
                    "epoch": epoch,
                    "stage": stage_info["stage"],
                    "train_loss": float(loss.detach().cpu()) * args.gradient_accumulation_steps,
                    "train_reward": float(avg_reward),
                    "train_exact_rate": float(exact),
                    "train_legal_rate": float(legal),
                    "train_format_rate": float(fmt),
                    "train_refusal_rate": float(refusal),
                    "train_invalid_rate": sum(0.0 if s["is_valid"] else 1.0 for s in scored) / len(scored),
                    "completion_length": sum(len(c) for c in completions) / len(completions),
                    "kl_penalty": float(kl_loss.detach().cpu()) if kl_penalties else 0.0,
                    "stage_focus": stage_info["focus"],
                }

                # 评估
                if global_step % args.eval_steps == 0 and dev_rows:
                    metrics.update(evaluate_dev_subset(model, tokenizer, device, dev_rows, args, global_step, stage_weights))

                append_jsonl(metrics_path, metrics)

                if global_step % args.logging_steps == 0:
                    print(json.dumps(metrics, ensure_ascii=False, indent=2))

                if global_step % args.save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

                    # 保存阶段指标历史
                    with open(output_dir / "stage_metrics_history.json", "w") as f:
                        json.dump(stage_metrics_history, f, indent=2, ensure_ascii=False)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # 保存最终模型和指标
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 保存最终阶段指标
    with open(output_dir / "final_stage_metrics.json", "w") as f:
        json.dump(stage_metrics_history, f, indent=2, ensure_ascii=False)

    print(f"\n=== Training Complete ===")
    print(f"Total steps: {global_step}")
    print(f"Final stage metrics saved to: {output_dir / 'final_stage_metrics.json'}")


# 复制必要的辅助函数
def load_manual_model(args):
    torch = _import_torch()
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype_from_arg(torch, args.dtype),
        low_cpu_mem_usage=True,
    )
    model = get_peft_model(model, build_lora_config(args))
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    device = choose_device()
    model.to(device)
    model.train()

    # Load reference model WITHOUT LoRA for KL divergence
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype_from_arg(torch, args.dtype),
        low_cpu_mem_usage=True,
    )
    ref_model.config.use_cache = False
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    return model, ref_model, tokenizer, device


def generate_group(model, tokenizer, device, numbers: list[int], target: int, args) -> list[str]:
    torch = _import_torch()
    prompt = render_prompt(tokenizer, numbers, target)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_generations,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = encoded["input_ids"].shape[1]
    return [
        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        for seq in generated
    ]


def completion_logprob(model, tokenizer, device, prompt: str, completion: str):
    torch = _import_torch()
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    prompt_len = encoded_prompt["input_ids"].shape[1]
    encoded = tokenizer(prompt + completion, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    denom = mask.sum().clamp_min(1)
    return (token_log_probs * mask).sum() / denom


def evaluate_dev_subset(model, tokenizer, device, rows: list[dict[str, Any]], args, step: int, stage_weights: dict) -> dict[str, Any]:
    if not rows:
        return {}
    model.eval()
    sample_rows = rows[: args.dev_eval_samples]
    exact = legal = fmt = refusal = invalid = total_reward = 0.0
    for row in sample_rows:
        numbers = [int(n) for n in row["numbers"]]
        target = int(row.get("target", 24))
        completion = generate_group(model, tokenizer, device, numbers, target, args)[0]
        scored = score_output(numbers, completion, target, weights=stage_weights)
        exact += scored["exact"]
        legal += scored["legal"]
        fmt += scored["format"]
        refusal += 1.0 if scored.get("is_refusal", False) else 0.0
        invalid += 0.0 if scored["is_valid"] else 1.0
        total_reward += scored["reward"]
    denom = max(len(sample_rows), 1)
    model.train()
    return {
        "step": step,
        "dev_reward": total_reward / denom,
        "dev_exact_rate": exact / denom,
        "dev_legal_rate": legal / denom,
        "dev_format_rate": fmt / denom,
        "dev_refusal_rate": refusal / denom,
        "dev_invalid_rate": invalid / denom,
    }


def completion_logprob_per_token(model, tokenizer, device, prompt: str, completion: str):
    """Return per-token logprobs and a mask for the completion tokens."""
    torch = _import_torch()
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    prompt_len = encoded_prompt["input_ids"].shape[1]
    encoded = tokenizer(prompt + completion, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1) * mask
    return token_log_probs, mask


def compute_kl_penalty(
    policy_model, ref_model, tokenizer, device, prompt: str, completion: str
):
    """Compute approximate KL divergence between policy and reference model."""
    torch = _import_torch()
    with torch.no_grad():
        ref_log_probs, mask = completion_logprob_per_token(ref_model, tokenizer, device, prompt, completion)
    policy_log_probs, _ = completion_logprob_per_token(policy_model, tokenizer, device, prompt, completion)
    # Only compute KL on tokens where we have both ref and policy logprobs
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    diff = ref_log_probs - policy_log_probs
    kl = torch.exp(diff) - diff - 1.0
    # Weight by the mask and normalize by number of completion tokens
    return (kl * mask).sum() / mask.sum()


def get_stage_weights(epoch, total_epochs):
    """根据训练阶段调整权重"""
    if epoch < total_epochs * 0.3:
        # 阶段1：学习格式和合法性
        return {
            "exact": 0.1,
            "legal": 0.3,
            "format": 2.0,
            "closeness": 0.1,
            "refusal": 0.0,
            "penalty": 0.0,
        }
    elif epoch < total_epochs * 0.7:
        # 阶段2：学习精确解题
        return {
            "exact": 2.0,
            "legal": 0.2,
            "format": 0.1,
            "closeness": 0.1,
            "refusal": 0.5,
            "penalty": -0.3,
        }
    else:
        # 阶段3：学习拒答和抑制幻觉
        return {
            "exact": 3.0,
            "legal": 0.1,
            "format": 0.05,
            "closeness": 0.1,
            "refusal": 1.0,
            "penalty": -0.5,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhanced training for 24 game with staged training and better monitoring.")
    parser.add_argument("--trainer", choices=["trl", "manual"], default="manual")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-file", default="data/processed/train.jsonl")
    parser.add_argument("--dev-file", default="data/processed/dev.jsonl")
    parser.add_argument("--output-dir", default="outputs/qwen24-grpo-enhanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--dev-eval-samples", type=int, default=64)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--kl-coef", type=float, default=0.04, help="KL divergence penalty coefficient (0 to disable)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.trainer == "trl":
        print("TRL trainer not implemented in enhanced version, using manual")
        args.trainer = "manual"
    run_manual_grouped_rl_enhanced(args)


if __name__ == "__main__":
    main()