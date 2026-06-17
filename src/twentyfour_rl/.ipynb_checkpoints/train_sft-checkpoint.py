#!/usr/bin/env python3
"""SFT 预热训练 - 24点（手写循环版）"""
import argparse, json, random
from pathlib import Path
import torch, torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_file", default="data/sft_train.jsonl")
    parser.add_argument("--output_dir", default="outputs/sft_warmup")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    device = torch.device('npu') if torch.npu.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    lora_config = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none")
    model = get_peft_model(model, lora_config)
    model = model.to(device)

    # 加载数据
    data = []
    with open(args.train_file) as f:
        for line in f:
            item = json.loads(line.strip())
            text = item['prompt'] + item['answer']
            data.append(text)
    print(f"Samples: {len(data)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.num_epochs):
        random.shuffle(data)
        total_loss = 0
        for i in range(0, len(data), args.batch_size):
            batch = data[i:i+args.batch_size]
            encoded = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            total_loss += loss.item()

            if global_step % 10 == 0:
                avg = total_loss / min(10, (i//args.batch_size)+1)
                print(f"Step {global_step}: loss={avg:.4f}")
                total_loss = 0
            
            if global_step % 200 == 0:
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done! Model saved to {output_dir}")

if __name__ == "__main__":
    main()
