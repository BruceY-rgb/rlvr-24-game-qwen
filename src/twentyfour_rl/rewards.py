from __future__ import annotations

import math
from typing import Any, Iterable

from .verifier import extract_answer, has_r1_format, verify


DEFAULT_REWARD_WEIGHTS = {
    "exact": 1.0,
    "legal": 0.3,
    "format": 0.2,
    "closeness": 0.2,
}

# Reward components that may be ablated (zeroed out). "exact" is the core
# verifiable signal and is never ablatable.
ABLATABLE_REWARDS = ("legal", "format", "closeness")


def reward_weights(disabled: Iterable[str] | None = None) -> dict[str, float]:
    """Return a copy of the default weights with the named components zeroed.

    Used for ablation runs (e.g. disabled=["format"]). "exact" is protected.
    """
    weights = dict(DEFAULT_REWARD_WEIGHTS)
    for name in disabled or []:
        name = name.strip()
        if not name:
            continue
        if name == "exact":
            raise ValueError("Refusing to ablate the 'exact' reward component.")
        if name not in weights:
            raise ValueError(f"Unknown reward component '{name}'. Choices: {ABLATABLE_REWARDS}.")
        weights[name] = 0.0
    return weights


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


def make_grpo_reward_func(weights: dict[str, float] | None = None):
    """Build a TRL-compatible reward function bound to a weight dict.

    Use ``reward_weights(disabled=[...])`` to construct ablation weights.
    """
    weights = weights or DEFAULT_REWARD_WEIGHTS

    def grpo_reward_func(completions, numbers=None, target=None, **kwargs):
        numbers = numbers or kwargs.get("nums") or kwargs.get("cards")
        if numbers is None:
            raise ValueError("GRPO reward requires a 'numbers' column.")
        if target is None:
            target = kwargs.get("targets") or [24.0] * len(completions)
        rewards: list[float] = []
        for completion, nums, tgt in zip(completions, numbers, target):
            text = normalize_completion_text(completion)
            rewards.append(
                score_output(list(nums), text, _target_from_value(tgt), weights=weights)["reward"]
            )
        return rewards

    # TRL keys training metrics by the reward function's __name__; keep it
    # stable so logs stay comparable across ablation runs.
    grpo_reward_func.__name__ = "grpo_reward_func"
    return grpo_reward_func


# Default full-reward instance (kept for backwards compatibility / imports).
grpo_reward_func = make_grpo_reward_func()

