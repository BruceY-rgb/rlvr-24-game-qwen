from __future__ import annotations

import math
from typing import Any, Iterable

from .verifier import extract_answer, has_r1_format, verify


DEFAULT_REWARD_WEIGHTS = {
    "exact": 20.0,        # 🔥 主目标
    "legal": 0.05,        # 只做约束
    "format": 0.02,       # 只做约束
    "closeness": 1.0,     # 🔥 shaping signal
    "refusal": 1.0,
    "penalty": -0.5,
}


ABLATABLE_REWARDS = ("legal", "format", "closeness", "refusal", "penalty")


def reward_weights(disabled: Iterable[str] | None = None) -> dict[str, float]:
    """Copy default weights, zeroing the named components for ablation.

    exact is the core verifiable signal and is never ablatable.
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


def should_refuse_output(raw_output: str, numbers: list[int]) -> bool:
    """判断是否应该拒绝回答"""
    if raw_output is None:
        return True

    refuse_keywords = ["no solution", "impossible", "cannot solve", "no solution found", "no answer"]
    text = raw_output.lower()
    if any(keyword in text for keyword in refuse_keywords):
        return True

    if len(raw_output.strip()) < 10:
        return True

    answer = extract_answer(raw_output)
    if not answer.strip():
        return True

    return False


def score_output(
    numbers: list[int],
    raw_output: str,
    target: float = 24.0,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    # 兼容外部动态权重（适配PPO脚本的epoch权重策略）
    weights = weights or DEFAULT_REWARD_WEIGHTS

    if should_refuse_output(raw_output, numbers):
        return {
            "reward": float(weights.get("refusal", 1.0)),
            "exact": 0.0,
            "legal": 0.0,
            "format": 0.0,
            "closeness": 0.0,
            "refusal": 1.0,
            "penalty": 0.0,
            "is_refusal": True,
            "is_correct": False,
            "answer": "",
            "expression": "",
            "normalized_expression": "",
            "value": None,
            "is_valid": False,
            "used_numbers": [],
            "error_type": "refusal",
            "message": "Model correctly refused to answer.",
            "calculation_steps": [],
        }

    answer = extract_answer(raw_output)
    verdict = verify(numbers, answer, target=target)
    format_ok = has_r1_format(raw_output)
    exact = 1.0 if verdict["is_correct"] else 0.0
    legal = 1.0 if verdict["is_valid"] else 0.0
    fmt = 1.0 if format_ok else 0.0
    close = closeness_score(verdict["value"], target)

    penalty = 0.0
    if not verdict["is_valid"]:
        penalty = weights.get("penalty", -0.5)

    reward = (
        weights.get("exact", 20.0) * exact
        + weights.get("legal", 0.05) * legal
        + weights.get("format", 0.02) * fmt
        + weights.get("closeness", 1.0) * close
        + weights.get("penalty", -0.5) * penalty
    )

    return {
        "reward": float(reward),
        "exact": exact,
        "legal": legal,
        "format": fmt,
        "closeness": close,
        "refusal": 0.0,
        "penalty": penalty,
        "is_refusal": False,
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
    """Build a TRL reward function bound to a weight dict (for ablation)."""
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
            rewards.append(score_output(list(nums), text, _target_from_value(tgt), weights=weights)["reward"])
        return rewards

    grpo_reward_func.__name__ = "grpo_reward_func"
    return grpo_reward_func


grpo_reward_func = make_grpo_reward_func()


def compute_detailed_metrics(results):
    """计算更详细的评估指标"""
    if not results:
        return {}

    total = len(results)
    metrics = {
        "pass_at_1": sum(r["exact"] for r in results) / total,
        "legal_rate": sum(r["legal"] for r in results) / total,
        "format_rate": sum(r["format"] for r in results) / total,
        "closeness_rate": sum(r["closeness"] for r in results) / total,

        "refusal_rate": sum(1 for r in results if r.get("is_refusal", False)) / total,
        "hallucination_rate": sum(1 for r in results if not r["is_valid"]) / total,
        "penalty_rate": sum(1 for r in results if r.get("penalty", 0) < 0) / total,

        "smart_hallucination": sum(1 for r in results
                                 if not r["is_valid"] and r["format"]) / total,
        "valid_but_wrong": sum(1 for r in results
                             if r["is_valid"] and not r["is_correct"]) / total,
        "close_calls": sum(1 for r in results
                    if r.get("value") is not None and 20 <= r.get("value", 0) <= 28) / total,
        "exact_but_invalid": sum(1 for r in results
                               if r["exact"] and not r["is_valid"]) / total,

        "avg_reward": sum(r["reward"] for r in results) / total,
        "max_reward": max(r["reward"] for r in results),
        "min_reward": min(r["reward"] for r in results),
    }
    return metrics