#!/usr/bin/env python3
"""Aggregate eval metrics across conditions/splits into a comparison table.

Scans <eval-root>/<condition>/<split>_metrics.json (condition = trained, base,
solver, ablation_*), and writes a long CSV plus a Markdown report with one
pivot table per key metric (rows = split, columns = condition).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

SPLIT_ORDER = ["dev", "ood_test", "hard_test", "unsolvable", "countdown_ood", "countdown_unique"]
METRICS = [
    ("pass_at_1", "pass@1"),
    ("pass_at_k", "pass@k"),
    ("legal_rate", "legal_rate"),
    ("format_rate", "format_rate"),
    ("hallucination_rate", "hallucination_rate"),
]


def load_all(eval_root: str) -> list[dict]:
    rows = []
    for mdir in sorted(Path(eval_root).glob("*")):
        if not mdir.is_dir():
            continue
        for f in sorted(mdir.glob("*_metrics.json")):
            try:
                payload = json.load(open(f))
            except Exception:
                continue
            for m in payload:
                m = dict(m)
                m["condition"] = mdir.name
                rows.append(m)
    return rows


def split_key(s: str) -> tuple[int, str]:
    return (SPLIT_ORDER.index(s), s) if s in SPLIT_ORDER else (len(SPLIT_ORDER), s)


def pivot_md(rows: list[dict], metric: str, conditions: list[str]) -> str:
    splits = sorted({r["split"] for r in rows}, key=split_key)
    table = {(r["condition"], r["split"]): r.get(metric) for r in rows}
    head = "| split | " + " | ".join(conditions) + " |"
    sep = "|" + "---|" * (len(conditions) + 1)
    lines = [head, sep]
    for s in splits:
        cells = []
        for c in conditions:
            v = table.get((c, s))
            cells.append("—" if v is None else f"{v:.3f}")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="outputs/eval")
    ap.add_argument("--out-md", default="outputs/eval/summary.md")
    ap.add_argument("--out-csv", default="outputs/eval/summary.csv")
    args = ap.parse_args()

    rows = load_all(args.eval_root)
    if not rows:
        print(f"No metrics found under {args.eval_root}")
        return

    # preferred condition ordering
    pref = ["solver", "base", "trained", "full", "no_legal", "no_format", "no_closeness"]
    conditions = sorted({r["condition"] for r in rows}, key=lambda c: (pref.index(c) if c in pref else len(pref), c))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "split", "samples", "attempts", "pass_at_1", "pass_at_k",
              "legal_rate", "format_rate", "invalid_rate", "hallucination_rate"]
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (split_key(r["split"]), r["condition"])):
            w.writerow(r)

    md = ["# 24-game evaluation summary", "", f"Conditions: {', '.join(conditions)}", ""]
    for key, label in METRICS:
        md.append(f"## {label}")
        md.append("")
        md.append(pivot_md(rows, key, conditions))
        md.append("")
    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {args.out_csv} and {args.out_md}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
