from __future__ import annotations

import argparse
import json
import random
import re
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Iterable

from .prompts import build_prompt_text
from .solver import solve_24


NUMBER_KEYS = (
    "numbers",
    "nums",
    "cards",
    "digits",
    "input_numbers",
    "operands",
)
TEXT_KEYS = (
    "input",
    "question",
    "problem",
    "puzzle",
    "Puzzles",
    "prompt",
    "text",
    "target",
)
TARGET_KEYS = (
    "target",
    "answer",
    "goal",
    "result",
)


def canonical_numbers(numbers: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(int(n) for n in numbers))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_numbers(example: dict[str, Any], min_numbers: int = 4, max_numbers: int = 4) -> list[int] | None:
    for key in NUMBER_KEYS:
        if key in example:
            value = example[key]
            if isinstance(value, (list, tuple)):
                nums = [int(v) for v in value]
                if len(nums) >= min_numbers:
                    return nums[:max_numbers]
            if isinstance(value, str):
                nums = _numbers_from_text(value)
                if len(nums) >= min_numbers:
                    return nums[:max_numbers]
    for key in TEXT_KEYS:
        if key in example and isinstance(example[key], str):
            nums = _numbers_from_text(example[key])
            if len(nums) >= min_numbers:
                return nums[:max_numbers]
    return None


def _numbers_from_text(text: str) -> list[int]:
    nums = [int(x) for x in re.findall(r"(?<!\d)\d+(?!\d)", text)]
    return nums


def extract_target(example: dict[str, Any], default: int = 24) -> int:
    for key in TARGET_KEYS:
        if key not in example:
            continue
        value = example[key]
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            match = re.search(r"(?<!\d)-?\d+(?!\d)", value)
            if match:
                return int(match.group(0))
    return int(default)


def extract_solvable(example: dict[str, Any], default: bool = True) -> bool:
    for key in ("solvable", "is_solvable", "can_solve"):
        if key in example:
            value = example[key]
            if isinstance(value, str):
                return value.strip().lower() in {"true", "1", "yes", "y"}
            return bool(value)
    return default


def normalize_row(
    example: dict[str, Any],
    split: str,
    source: str,
    idx: int,
    target: int = 24,
    solvable_default: bool = True,
    min_numbers: int = 4,
    max_numbers: int = 4,
    min_card_value: int = 1,
    max_card_value: int = 13,
    target_from_example: bool = False,
) -> dict[str, Any] | None:
    numbers = extract_numbers(example, min_numbers=min_numbers, max_numbers=max_numbers)
    if numbers is None or not (min_numbers <= len(numbers) <= max_numbers):
        return None
    if any(int(n) < min_card_value or int(n) > max_card_value for n in numbers):
        return None
    target_value = extract_target(example, default=target) if target_from_example else int(target)
    solvable = extract_solvable(example, default=solvable_default)
    return {
        "id": f"{source}:{split}:{idx}",
        "source": source,
        "split": split,
        "numbers": [int(n) for n in numbers],
        "numbers_key": "-".join(str(n) for n in canonical_numbers(numbers)),
        "target": int(target_value),
        "solvable": bool(solvable),
        "prompt": build_prompt_text(numbers, target_value),
    }


def load_hf_rows(
    dataset_name: str,
    split: str,
    source: str,
    solvable_default: bool,
    max_samples: int | None = None,
    streaming: bool = False,
    **normalize_kwargs,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    rows = []
    for idx, example in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        row = normalize_row(
            example,
            split=split,
            source=source,
            idx=idx,
            solvable_default=solvable_default,
            **normalize_kwargs,
        )
        if row is not None:
            rows.append(row)
    return rows


def make_synthetic_rows(task: str = "24") -> list[dict[str, Any]]:
    if task == "countdown":
        examples = [
            ([44, 19, 35], 98),
            ([41, 70, 18, 35], 57),
            ([7, 24, 4], 35),
            ([55, 56, 38], 38),
        ]
        return [
            {
                "id": f"synthetic_countdown:train:{idx}",
                "source": "synthetic_countdown",
                "split": "train",
                "numbers": nums,
                "numbers_key": "-".join(str(n) for n in canonical_numbers(nums)),
                "target": target,
                "solvable": solve_24(nums, target=target) is not None,
                "prompt": build_prompt_text(nums, target),
            }
            for idx, (nums, target) in enumerate(examples)
        ]

    examples = [
        [1, 3, 4, 6],
        [3, 3, 8, 8],
        [1, 5, 5, 5],
        [2, 2, 6, 6],
        [4, 4, 10, 10],
        [1, 1, 1, 1],
        [1, 1, 1, 2],
        [2, 3, 4, 5],
        [7, 7, 3, 3],
        [13, 13, 4, 7],
    ]
    rows = []
    for idx, nums in enumerate(examples):
        rows.append(
            {
                "id": f"synthetic:train:{idx}",
                "source": "synthetic",
                "split": "train",
                "numbers": nums,
                "numbers_key": "-".join(str(n) for n in canonical_numbers(nums)),
                "target": 24,
                "solvable": solve_24(nums) is not None,
                "prompt": build_prompt_text(nums, 24),
            }
        )
    return rows


def generate_unsolvable_24_rows(max_samples: int = 100) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for nums_tuple in combinations_with_replacement(range(1, 14), 4):
        nums = list(nums_tuple)
        if solve_24(nums, target=24) is not None:
            continue
        rows.append(
            {
                "id": f"generated_unsolvable:unsolvable:{len(rows)}",
                "source": "generated_unsolvable_24",
                "split": "unsolvable",
                "numbers": nums,
                "numbers_key": "-".join(str(n) for n in canonical_numbers(nums)),
                "target": 24,
                "solvable": False,
                "prompt": build_prompt_text(nums, 24),
            }
        )
        if len(rows) >= max_samples:
            break
    return rows


def dedupe_by_key(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = f"{row.get('target', 24)}|{row['numbers_key']}"
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def with_output_split(rows: Iterable[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item["split"] = split
        item["id"] = f"{item.get('source', 'data')}:{split}:{idx}"
        out.append(item)
    return out


def mix_with_unsolvable(train_rows: list[dict[str, Any]], unsolvable_rows: list[dict[str, Any]], ratio: float = 0.1, seed: int = 42) -> list[dict[str, Any]]:
    """混合可解和无解数据"""
    rng = random.Random(seed)

    # 根据比例计算无解样本数量
    num_unsolvable = min(int(len(train_rows) * ratio), len(unsolvable_rows))

    # 随机选择无解样本
    unsampled = unsolvable_rows[:num_unsolvable]

    # 混合数据
    mixed = train_rows + unsampled
    rng.shuffle(mixed)

    print(f"混合后的数据: {len(train_rows)} 可解 + {num_unsolvable} 无解 = {len(mixed)} 总样本")
    return mixed


def split_train_dev(rows: list[dict[str, Any]], dev_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    dev_size = max(1, int(len(rows) * dev_ratio)) if len(rows) > 1 else 0
    return rows[dev_size:], rows[:dev_size]


def build_prepared_data(
    output_dir: str | Path,
    seed: int = 42,
    dev_ratio: float = 0.1,
    allow_synthetic: bool = False,
    include_countdown: bool = False,
    countdown_max_samples: int = 2000,
    unsolvable_max_samples: int = 100,
) -> dict[str, int]:
    output_dir = Path(output_dir)
    try:
        train_source = load_hf_rows(
            "nlile/24-game",
            "train",
            "nlile_24_game",
            solvable_default=True,
            min_numbers=4,
            max_numbers=4,
            min_card_value=1,
            max_card_value=13,
        )
    except Exception:
        if not allow_synthetic:
            raise
        train_source = make_synthetic_rows("24")

    solvable = [r for r in train_source if r["solvable"]]
    unsolvable_from_source = [r for r in train_source if not r["solvable"]]
    solvable = dedupe_by_key(solvable)
    train_rows, dev_rows = split_train_dev(solvable, dev_ratio=dev_ratio, seed=seed)
    train_rows = with_output_split(train_rows, "train")
    dev_rows = with_output_split(dev_rows, "dev")

    # 生成额外的无解数据用于训练
    if unsolvable_max_samples > 0 and allow_synthetic:
        generated_unsolvable = generate_unsolvable_24_rows(unsolvable_max_samples)
        # 混合10%的无解数据到训练集
        train_rows = mix_with_unsolvable(train_rows, generated_unsolvable, ratio=0.1, seed=seed)

    try:
        ood_rows = load_hf_rows(
            "test-time-compute/game-of-24",
            "train",
            "game_of_24",
            solvable_default=True,
            min_numbers=4,
            max_numbers=4,
            min_card_value=1,
            max_card_value=13,
        )
    except Exception:
        if not allow_synthetic:
            raise
        ood_rows = make_synthetic_rows("24")

    full_tot_rows = dedupe_by_key(ood_rows)
    train_keys = {r["numbers_key"] for r in train_rows}
    ood_rows = [r for r in full_tot_rows if r["numbers_key"] not in train_keys]
    hard_rows = [r for idx, r in enumerate(full_tot_rows) if 900 <= idx < 1000]
    ood_rows = with_output_split(ood_rows, "ood_test")
    hard_rows = with_output_split(hard_rows, "hard_test")
    unsolvable = with_output_split(unsolvable_from_source, "unsolvable")
    if not unsolvable and unsolvable_max_samples > 0:
        unsolvable = generate_unsolvable_24_rows(unsolvable_max_samples)

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "dev.jsonl", dev_rows)
    write_jsonl(output_dir / "ood_test.jsonl", ood_rows)
    write_jsonl(output_dir / "hard_test.jsonl", hard_rows)
    write_jsonl(output_dir / "unsolvable.jsonl", unsolvable)

    stats = {
        "train": len(train_rows),
        "dev": len(dev_rows),
        "ood_test": len(ood_rows),
        "hard_test": len(hard_rows),
        "unsolvable": len(unsolvable),
    }

    if include_countdown:
        try:
            countdown_rows = load_hf_rows(
                "Jiayi-Pan/Countdown-Tasks-3to4",
                "train",
                "countdown_3to4",
                solvable_default=True,
                max_samples=countdown_max_samples,
                streaming=True,
                min_numbers=3,
                max_numbers=4,
                min_card_value=1,
                max_card_value=100,
                target_from_example=True,
            )
        except Exception:
            if not allow_synthetic:
                raise
            countdown_rows = make_synthetic_rows("countdown")

        try:
            countdown_unique_rows = load_hf_rows(
                "Jiayi-Pan/Countdown-Tasks-3to4-Unique",
                "train",
                "countdown_3to4_unique",
                solvable_default=True,
                max_samples=countdown_max_samples,
                streaming=True,
                min_numbers=3,
                max_numbers=4,
                min_card_value=1,
                max_card_value=100,
                target_from_example=True,
            )
        except Exception:
            if not allow_synthetic:
                raise
            countdown_unique_rows = make_synthetic_rows("countdown")

        countdown_rows = with_output_split(dedupe_by_key(countdown_rows), "countdown_ood")
        countdown_unique_rows = with_output_split(dedupe_by_key(countdown_unique_rows), "countdown_unique")
        write_jsonl(output_dir / "countdown_ood.jsonl", countdown_rows)
        write_jsonl(output_dir / "countdown_unique.jsonl", countdown_unique_rows)
        stats["countdown_ood"] = len(countdown_rows)
        stats["countdown_unique"] = len(countdown_unique_rows)

    write_jsonl(output_dir / "data_stats.jsonl", [stats])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 24-game JSONL splits.")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--include-countdown", action="store_true")
    parser.add_argument("--countdown-max-samples", type=int, default=2000)
    parser.add_argument("--unsolvable-max-samples", type=int, default=100)
    args = parser.parse_args()
    stats = build_prepared_data(
        args.output_dir,
        args.seed,
        args.dev_ratio,
        args.allow_synthetic,
        include_countdown=args.include_countdown,
        countdown_max_samples=args.countdown_max_samples,
        unsolvable_max_samples=args.unsolvable_max_samples,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()