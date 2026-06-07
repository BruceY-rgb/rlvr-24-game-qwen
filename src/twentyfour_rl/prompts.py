from __future__ import annotations

from typing import Iterable


SYSTEM_PROMPT = (
    "You are a precise arithmetic reasoning model. Solve arithmetic card puzzles exactly. "
    "Use each given number once and only once. Use only +, -, *, /, and parentheses."
)


def format_numbers(numbers: Iterable[int]) -> str:
    return " ".join(str(int(n)) for n in numbers)


def build_user_prompt(numbers: Iterable[int], target: int = 24) -> str:
    nums = format_numbers(numbers)
    return (
        f"Given these integers: {nums}. Build one expression that equals {target}. "
        "Every input number must be used exactly once. "
        "Return exactly this format:\n"
        "<think>brief reasoning</think><answer>expression</answer>"
    )


def build_messages(numbers: Iterable[int], target: int = 24) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(numbers, target)},
    ]


def build_prompt_text(numbers: Iterable[int], target: int = 24) -> str:
    return f"{SYSTEM_PROMPT}\n\n{build_user_prompt(numbers, target)}\n"


def render_prompt(tokenizer, numbers: Iterable[int], target: int = 24) -> str:
    messages = build_messages(numbers, target)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return build_prompt_text(numbers, target)
