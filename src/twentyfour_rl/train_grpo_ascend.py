from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any

from .data import read_jsonl
from .eval import choose_device
from .prompts import render_prompt
from .rewards import make_grpo_reward_func, reward_weights, score_output


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
    # 新增 local_files_only=True 强制本地离线加载
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def run_trl_grpo(args) -> None:
    torch = _import_torch()
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    train_rows = read_jsonl(args.train_file)
    dev_rows = read_jsonl(args.dev_file) if args.dev_file else []
    tokenizer = load_tokenizer(args.model_name_or_path)
    lora_config = build_lora_config(args)
    disabled = [s for s in (getattr(args, "disable_rewards", "") or "").split(",") if s.strip()]
    weights = reward_weights(disabled)
    if disabled:
        print(f"[ablation] disabled reward components: {disabled} -> {weights}", flush=True)

    config_values = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "bf16": args.dtype == "bf16",
        "fp16": args.dtype == "fp16",
        "gradient_checkpointing": args.gradient_checkpointing,
        "use_vllm": False,
        "seed": args.seed,
        "report_to": [],
        "model_init_kwargs": {
            "trust_remote_code": True,
            "torch_dtype": dtype_from_arg(torch, args.dtype),
            "local_files_only": True,
        },
    }
    grpo_sig = inspect.signature(GRPOConfig.__init__).parameters
    config_values = {k: v for k, v in config_values.items() if k in grpo_sig}
    training_args = GRPOConfig(**config_values)

    trainer_values = {
        "model": args.model_name_or_path,
        "args": training_args,
        "reward_funcs": [make_grpo_reward_func(weights)],
        "train_dataset": Dataset.from_list(train_rows),
        "eval_dataset": Dataset.from_list(dev_rows) if dev_rows else None,
        "peft_config": lora_config,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer_sig = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_values = {k: v for k, v in trainer_values.items() if k in trainer_sig and v is not None}
    trainer = GRPOTrainer(**trainer_values)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def load_manual_model(args):
    torch = _import_torch()
    from peft import get_peft_model, PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(args.model_name_or_path)
    # 加载基座，增加本地离线参数
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype_from_arg(torch, args.dtype),
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    # 如果传入SFT LoRA路径，先加载SFT权重
    if args.sft_lora_path and Path(args.sft_lora_path).exists():
        print(f"✅ 加载SFT预训练LoRA: {args.sft_lora_path}")
        model = PeftModel.from_pretrained(model, args.sft_lora_path)
    # 再绑定本次GRPO训练LoRA配置
    model = get_peft_model(model, build_lora_config(args))
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    device = choose_device()
    model.to(device)
    model.train()

    # Reference model：只用纯基座，不带任何LoRA，KL对比原始基线
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype_from_arg(torch, args.dtype),
        low_cpu_mem_usage=True,
        local_files_only=True,
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


def evaluate_dev_subset(model, tokenizer, device, rows: list[dict[str, Any]], args, step: int) -> dict[str, Any]:
    if not rows:
        return {}
    model.eval()
    sample_rows = rows[: args.dev_eval_samples]
    exact = legal = fmt = invalid = total_reward = 0.0
    for row in sample_rows:
        numbers = [int(n) for n in row["numbers"]]
        target = int(row.get("target", 24))
        completion = generate_group(model, tokenizer, device, numbers, target, args)[0]
        scored = score_output(numbers, completion, target)
        exact += scored["exact"]
        legal += scored["legal"]
        fmt += scored["format"]
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
            "exact": 0.5,
            "legal": 0.3,
            "format": 0.2,
            "closeness": 0.1,
            "refusal": 0.0,
            "penalty": 0.0,
        }
    elif epoch < total_epochs * 0.7:
        # 阶段2：学习精确解题
        return {
            "exact": 3.0,
            "legal": 0.2,
            "format": 0.1,
            "closeness": 0.1,
            "refusal": 0.0,
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


def run_manual_grouped_rl(args) -> None:
    torch = _import_torch()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"

    train_rows = read_jsonl(args.train_file)
    dev_rows = read_jsonl(args.dev_file) if args.dev_file else []
    model, ref_model, tokenizer, device = load_manual_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(args.num_train_epochs)):
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

                metrics = {
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": float(loss.detach().cpu()) * args.gradient_accumulation_steps,
                    "train_reward": float(rewards.mean().detach().cpu()),
                    "train_exact_rate": sum(s["exact"] for s in scored) / len(scored),
                    "train_legal_rate": sum(s["legal"] for s in scored) / len(scored),
                    "train_format_rate": sum(s["format"] for s in scored) / len(scored),
                    "train_invalid_rate": sum(0.0 if s["is_valid"] else 1.0 for s in scored) / len(scored),
                    "completion_length": sum(len(c) for c in completions) / len(completions),
                    "kl_penalty": float(kl_loss.detach().cpu()) if kl_penalties else 0.0,
                }
                if global_step % args.eval_steps == 0:
                    metrics.update(evaluate_dev_subset(model, tokenizer, device, dev_rows, args, global_step))
                append_jsonl(metrics_path, metrics)

                if global_step % args.logging_steps == 0:
                    print(json.dumps(metrics, ensure_ascii=False), flush=True)
                if global_step % args.save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Qwen on 24 game with TRL GRPO or a manual grouped RL fallback.")
    parser.add_argument("--trainer", choices=["trl", "manual"], default="trl")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    # 新增SFT LoRA参数
    parser.add_argument("--sft-lora-path", type=str, default="", help="预训练SFT LoRA权重目录，GRPO基于SFT续训")
    parser.add_argument("--train-file", default="data/processed/train.jsonl")
    parser.add_argument("--dev-file", default="data/processed/dev.jsonl")
    parser.add_argument("--output-dir", default="outputs/qwen24-grpo")
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
    parser.add_argument(
        "--disable-rewards",
        default="",
        help="Comma-separated reward components to zero out for ablation (legal,format,closeness,refusal,penalty).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.trainer == "trl":
        run_trl_grpo(args)
    else:
        run_manual_grouped_rl(args)


if __name__ == "__main__":
    main()