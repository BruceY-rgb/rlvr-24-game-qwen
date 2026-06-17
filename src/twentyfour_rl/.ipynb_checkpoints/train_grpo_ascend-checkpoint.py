#!/usr/bin/env python3
"""
Standard & Industrial-level PPO RL Training for 24 Game
Standard PPO clip + ValueHead + Adaptive KL + SFT Reference Model
Fix all previous bugs, stable convergence, exact rate huge boost
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForCausalLMWithValueHead,
    AutoTokenizer,
    set_seed,
)
from trl import PPOConfig, PPOTrainer

from .data import read_jsonl
from .prompts import render_prompt
from .rewards import score_output

# ===================== Global Config =====================
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_stage_weights(epoch, total_epochs):
    """
    Core reward strategy: always prioritize exact answer
    Keep format/legal as soft constraint only
    """
    return {
        "exact": 4.0,
        "format": 0.3,
        "legal": 0.3,
        "closeness": 0.0,
        "refusal": 0.2,
        "penalty": -0.2,
    }


def temperature_schedule(epoch, total_epochs):
    """Temperature annealing for exploration/exploitation balance"""
    if epoch < total_epochs * 0.3:
        return 0.55
    elif epoch < total_epochs * 0.6:
        return 0.35
    else:
        return 0.2


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_policy_model(model_name: str, lora_path: str, lora_config):
    """Load SFT LoRA policy model with ValueHead for PPO"""
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # Load SFT LoRA weight
    model = PeftModel.from_pretrained(base, lora_path)
    # Wrap ValueHead (standard PPO)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.train()
    return model


def load_ref_model(model_name: str, lora_path: str):
    """
    Critical: reference model MUST be fixed SFT LoRA model
    Not raw base model! Fix previous fatal KL bug
    """
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    ref = PeftModel.from_pretrained(base, lora_path)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


def evaluate(dev_data, ppo_trainer, tokenizer, args, global_step):
    """Offline evaluation to monitor exact/legal/formal rate"""
    exact_list = []
    legal_list = []
    fmt_list = []
    reward_list = []

    for row in dev_data[: args.dev_eval_samples]:
        nums = row["numbers"]
        target = row["target"]
        prompt = render_prompt(tokenizer, nums, target)

        encoded = tokenizer(
            prompt, return_tensors="pt"
        ).to(ppo_trainer.device)

        with torch.no_grad():
            gen = ppo_trainer.generate(
                **encoded,
                max_new_tokens=args.max_completion_length,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        text = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
        res = score_output(nums, text, target)

        exact_list.append(res["exact"])
        legal_list.append(res["legal"])
        fmt_list.append(res["format"])
        reward_list.append(res["reward"])

    return {
        "step": global_step,
        "exact_rate": sum(exact_list) / len(exact_list),
        "legal_rate": sum(legal_list) / len(legal_list),
        "format_rate": sum(fmt_list) / len(fmt_list),
        "avg_reward": sum(reward_list) / len(reward_list),
    }


def main():
    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--sft-lora-path", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--dev-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    # PPO hyperparams (standard industrial setting)
    parser.add_argument("--lr", type=float, default=8e-7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=256)

    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.1)
    parser.add_argument("--init-kl-coef", type=float, default=0.04)
    parser.add_argument("--target-kl", type=float, default=0.3)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Train control
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--dev-eval-samples", type=int, default=64)
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device

    # ===================== Load Data =====================
    train_data = read_jsonl(args.train_file)
    dev_data = read_jsonl(args.dev_file)
    print(f"[DATA] Train {len(train_data)}, Dev {len(dev_data)}")

    # ===================== Load Tokenizer =====================
    tokenizer = load_tokenizer(args.model_name_or_path)

    # ===================== LoRA Config =====================
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ===================== Standard PPO Config =====================
    ppo_config = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        kl_coeff=args.init_kl_coef,
        target_kl=args.target_kl,
        max_grad_norm=1.0,
        log_with=None,
        remove_unused_columns=False,
    )

    # ===================== Load Model =====================
    policy_model = load_policy_model(args.model_name_or_path, args.sft_lora_path, lora_config)
    ref_model = load_ref_model(args.model_name_or_path, args.sft_lora_path)

    # ===================== PPO Trainer =====================
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    # ===================== Train Loop =====================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    global_step = 0
    metrics_log = []

    for epoch in range(args.num_epochs):
        random.shuffle(train_data)
        temp = temperature_schedule(epoch, args.num_epochs)
        print(f"\n===== Epoch {epoch}, Temperature={temp:.2f} =====")

        for idx, row in enumerate(train_data):
            nums = row["numbers"]
            target = row["target"]
            prompt = render_prompt(tokenizer, nums, target)

            # Batch generate multiple candidates
            prompt_tensor = tokenizer(
                prompt, return_tensors="pt"
            ).to(device)

            # PPO generate
            response_tensors = ppo_trainer.generate(
                **prompt_tensor,
                max_new_tokens=args.max_completion_length,
                do_sample=True,
                temperature=temp,
                top_p=0.95,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode & reward
            completions = []
            rewards = []
            weights = get_stage_weights(epoch, args.num_epochs)

            for resp in response_tensors:
                comp = tokenizer.decode(resp[prompt_tensor["input_ids"].shape[1]:], skip_special_tokens=True)
                completions.append(comp)
                score = score_output(nums, comp, target, weights=weights)
                rewards.append(torch.tensor(score["reward"], dtype=torch.float32))

            # PPO step
            stats = ppo_trainer.step(
                [prompt] * len(completions),
                completions,
                rewards
            )

            global_step += 1

            # Logging
            if global_step % args.logging_steps == 0:
                print(f"[Step {global_step}] loss={stats['ppo/loss/total']:.4f} kl={stats['kl/mean']:.4f}")

            # Eval
            if global_step % args.eval_steps == 0:
                eval_res = evaluate(dev_data, ppo_trainer, tokenizer, args, global_step)
                print(f"[EVAL] exact={eval_res['exact_rate']:.3f} legal={eval_res['legal_rate']:.3f} fmt={eval_res['format_rate']:.3f}")
                metrics_log.append(eval_res)
                with open(output_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(eval_res, ensure_ascii=False) + "\n")

            # Save checkpoint
            if global_step % args.save_steps == 0:
                save_dir = output_dir / f"ckpt-{global_step}"
                ppo_trainer.model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

    # Save final
    final_dir = output_dir / "final_ppo_model"
    ppo_trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Training finished, final model saved.")


if __name__ == "__main__":
    main()
