from __future__ import annotations

from typing import Iterable


SYSTEM_PROMPT = (
    "You are a precise arithmetic reasoning model. Solve arithmetic card puzzles exactly. "
    "Use each given number once and only once. Use only +, -, *, /, and parentheses. "
    "If no solution exists, respond with 'no solution'. "
    "Always provide a clear arithmetic expression or indicate impossibility."
    "Every input number must be used exactly once. "
)


def format_numbers(numbers: Iterable[int]) -> str:
    """格式化数字列表为字符串，统一输入格式"""
    return " ".join(str(int(n)) for n in numbers)


def build_user_prompt(numbers: Iterable[int], target: int = 24) -> str:
    """构建用户侧问答Prompt，固定输出格式与约束"""
    nums = format_numbers(numbers)
    template = (
        "Given these integers: {nums}. Build one expression that equals {target}. "
        "Every input number must be used exactly once. "
        "Do NOT use any other numbers. Do NOT include '=' in your answer. "
        "Return exactly this format:\n"
        "brief reasoning\n"
        "expression\n\n"
        "Example:\n"
        "Numbers: 4 5 7 9\n"
        "Target: 24\n"
        "Try (9-7)=2, then 2*5=10, then 10+4=14 not 24. Try (9-7)*5+4=14. Try (7-5)*9+4=22. Try (7-5)*9+4? Try (9+7)*? \n"
        "(9-7)*5+4\n\n"
        "Now solve:\n"
        "Numbers: {nums}\n"
        "Target: {target}\n"
    )
    return template.format(nums=nums, target=target)


def build_messages(numbers: Iterable[int], target: int = 24) -> list[dict[str, str]]:
    """构建对话消息列表，适配模型chat模板"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(numbers, target)},
    ]


def build_prompt_text(numbers: Iterable[int], target: int = 24) -> str:
    """拼接完整Prompt文本，兜底无chat模板场景"""
    return f"{SYSTEM_PROMPT}\n\n{build_user_prompt(numbers, target)}\n"


def render_prompt(tokenizer, numbers: Iterable[int], target: int = 24) -> str:
    """
    统一Prompt渲染入口（适配PPO/SFT训练）
    优先使用tokenizer对话模板，兜底纯文本拼接
    与主训练脚本完全对齐
    """
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