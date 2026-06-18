"""Round-2 smoke probe: verify adapter loads and pick the prompt format.

Loads checkpoint-1400 on NPU and, for a few dev items, generates under three
prompt variants (chat template / new plain text / old data prompt field) so we
can see which one the round-2 model actually expects.
"""
import json
import torch
import torch_npu  # noqa: F401

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from twentyfour_rl.prompts import render_prompt, build_prompt_text
from twentyfour_rl.verifier import extract_answer, verify

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "outputs/final_grpo/checkpoint-1400"

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
print("TOKENIZER_OK", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(model, ADAPTER)
device = torch.device("npu:0")
model.to(device)
model.eval()
print("ADAPTER_LOADED_OK", flush=True)

rows = [json.loads(l) for l in open("data/processed/dev.jsonl")][:3]


def gen(prompt: str) -> str:
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=200, do_sample=False,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


for r in rows:
    n, t = r["numbers"], r["target"]
    print("== numbers", n, "target", t, flush=True)
    variants = [
        ("chat", render_prompt(tok, n, t)),
        ("plain", build_prompt_text(n, t)),
        ("olddata", r["prompt"]),
    ]
    for name, p in variants:
        out = gen(p)
        a = extract_answer(out)
        v = verify(n, a, t)
        print("  [%-7s] ok=%s legal=%s ans=%r :: out=%r"
              % (name, v["is_correct"], v["is_valid"], a[:40], out[:100]), flush=True)

print("PROBE_DONE", flush=True)
