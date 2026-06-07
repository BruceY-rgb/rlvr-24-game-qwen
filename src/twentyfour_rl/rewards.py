from __future__ import annotations

import math
from typing import Any

from .verifier import extract_answer, has_r1_format, verify


DEFAULT_REWARD_WEIGHTS = {
    "exact": 1.0,
    "legal": 0.3,
    "format": 0.2,
    "closeness": 0.2,
}


def _target_from_value(value: Any, default: float = 24.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def closeness_score(value: float | None, target: float = 24.0) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    denom = max(abs(target), 1.0)
    return max(0.0, 1.0 - min(abs(value - target) / denom, 1.0))


def score_output(
    numbers: list[int],
    raw_output: str,
    target: float = 24.0,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    weights = weights or DEFAULT_REWARD_WEIGHTS
    answer = extract_answer(raw_output)
    verdict = verify(numbers, answer, target=target)
    format_ok = has_r1_format(raw_output)
    exact = 1.0 if verdict["is_correct"] else 0.0
    legal = 1.0 if verdict["is_valid"] else 0.0
    fmt = 1.0 if format_ok else 0.0
    close = closeness_score(verdict["value"], target)
    reward = (
        weights.get("exact", 1.0) * exact
        + weights.get("legal", 0.3) * legal
        + weights.get("format", 0.2) * fmt
        + weights.get("closeness", 0.2) * close
    )
    return {
        "reward": float(reward),
        "exact": exact,
        "legal": legal,
        "format": fmt,
        "closeness": close,
        "answer": answer,
        **verdict,
    }


def normalize_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def grpo_reward_func(completions, numbers=None, target=None, **kwargs):
    numbers = numbers or kwargs.get("nums") or kwargs.get("cards")
    if numbers is None:
        raise ValueError("GRPO reward requires a 'numbers' column.")
    if target is None:
        target = kwargs.get("targets") or [24.0] * len(completions)
    rewards: list[float] = []
    for completion, nums, tgt in zip(completions, numbers, target):
        text = normalize_completion_text(completion)
        rewards.append(score_output(list(nums), text, _target_from_value(tgt))["reward"])
    return rewards

