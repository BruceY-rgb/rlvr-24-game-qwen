from __future__ import annotations

import ast
import math
import operator
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
ALLOWED_EXPR_RE = re.compile(r"^[0-9+\-*/().\s]+$")


@dataclass
class VerifyResult:
    expression: str
    normalized_expression: str
    numbers: list[int]
    target: float = 24.0
    value: float | None = None
    is_valid: bool = False
    is_correct: bool = False
    used_numbers: list[int] = field(default_factory=list)
    error_type: str = "unknown"
    message: str = ""
    calculation_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VerificationError(ValueError):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class SafeArithmeticEvaluator(ast.NodeVisitor):
    OPS = {
        ast.Add: ("+", operator.add),
        ast.Sub: ("-", operator.sub),
        ast.Mult: ("*", operator.mul),
        ast.Div: ("/", operator.truediv),
    }

    def __init__(self, allow_float_literals: bool = False):
        self.used_numbers: list[int] = []
        self.steps: list[str] = []
        self.allow_float_literals = allow_float_literals

    def visit_Expression(self, node: ast.Expression) -> tuple[float, str]:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> tuple[float, str]:
        left_value, left_expr = self.visit(node.left)
        right_value, right_expr = self.visit(node.right)
        op_info = self.OPS.get(type(node.op))
        if op_info is None:
            raise VerificationError("unsupported_operator", "Only +, -, *, and / are allowed.")
        symbol, fn = op_info
        if symbol == "/" and math.isclose(right_value, 0.0, abs_tol=1e-12):
            raise VerificationError("division_by_zero", "Expression divides by zero.")
        value = fn(left_value, right_value)
        expr = f"({left_expr}{symbol}{right_expr})"
        self.steps.append(f"{expr} = {value:g}")
        return value, expr

    def visit_Constant(self, node: ast.Constant) -> tuple[float, str]:
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VerificationError("invalid_literal", "Only numeric literals are allowed.")
        if isinstance(value, float):
            if not self.allow_float_literals:
                raise VerificationError("float_literal", "Only the original integer cards are allowed.")
            if not value.is_integer():
                raise VerificationError("float_literal", "Float literals must be integer-valued.")
            int_value = int(value)
        else:
            int_value = int(value)
        self.used_numbers.append(int_value)
        return float(int_value), str(int_value)

    def generic_visit(self, node: ast.AST):
        raise VerificationError(
            "unsupported_syntax",
            f"Unsupported expression syntax: {type(node).__name__}.",
        )


def extract_answer(text: str) -> str:
    import re

    if text is None:
        return ""

    text = str(text).strip()

    # 1. 优先提取 <answer> 标签内表达式
    match = re.search(
        r"<answer>(.*?)</answer>",
        text,
        re.DOTALL | re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    # 2. 提取「表达式 = 24」格式的表达式
    for line in text.splitlines():
        line = line.strip()
        if "=" in line:
            expr = line.split("=")[0].strip()
            if re.fullmatch(r"[0-9+\-*/().\s]+", expr):
                return expr

    # 3. 提取纯运算表达式行
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"[0-9+\-*/().\s]+", line):
            return line

    return ""


def has_r1_format(raw_output: str) -> bool:
    """校验输出是否符合R1标准格式，适配训练打分"""
    if raw_output is None:
        return False

    text = str(raw_output).strip()
    # 匹配标准标签格式
    if ANSWER_RE.search(text):
        return True

    # 兼容宽松合法表达式格式
    answer = extract_answer(text)
    if answer:
        return True

    return False


def normalize_expression(expression: str) -> str:
    """标准化表达式，去除所有空格"""
    return re.sub(r"\s+", "", expression or "")


def verify(
    numbers: list[int] | tuple[int, ...],
    expression: str,
    target: float = 24.0,
    tolerance: float = 1e-6,
    allow_float_literals: bool = False,
) -> dict[str, Any]:
    """
    核心24点校验函数
    校验：数字复用、语法合法、运算合法、结果精准、格式合规
    完全适配PPO奖励函数打分逻辑
    """
    numbers_list = [int(n) for n in numbers]
    raw_expression = "" if expression is None else str(expression).strip()
    normalized = normalize_expression(raw_expression)
    result = VerifyResult(
        expression=raw_expression,
        normalized_expression=normalized,
        numbers=numbers_list,
        target=float(target),
    )

    # 空表达式校验
    if not normalized:
        result.error_type = "empty"
        result.message = "No expression was provided."
        return result.to_dict()

    # 非法字符校验
    if not ALLOWED_EXPR_RE.match(raw_expression):
        result.error_type = "illegal_character"
        result.message = "Expression contains characters outside digits, operators, dots, spaces, and parentheses."
        return result.to_dict()

    # 语法解析校验
    try:
        tree = ast.parse(raw_expression, mode="eval")
    except SyntaxError as exc:
        result.error_type = "syntax_error"
        result.message = str(exc)
        return result.to_dict()

    # 安全运算求值
    evaluator = SafeArithmeticEvaluator(allow_float_literals=allow_float_literals)
    try:
        value, _ = evaluator.visit(tree)
    except VerificationError as exc:
        result.error_type = exc.error_type
        result.message = exc.message
        result.used_numbers = evaluator.used_numbers
        result.calculation_steps = evaluator.steps
        return result.to_dict()
    except Exception as exc:
        result.error_type = "evaluation_error"
        result.message = str(exc)
        result.used_numbers = evaluator.used_numbers
        result.calculation_steps = evaluator.steps
        return result.to_dict()

    result.value = float(value)
    result.used_numbers = evaluator.used_numbers
    result.calculation_steps = evaluator.steps

    # 数字全集匹配校验（严格4数字复用）
    if Counter(result.used_numbers) != Counter(numbers_list):
        result.error_type = "number_mismatch"
        result.message = (
            f"Used numbers {result.used_numbers}, expected multiset {numbers_list}."
        )
        return result.to_dict()

    # 合法表达式判定
    result.is_valid = True
    # 结果精准判定
    if math.isclose(value, float(target), abs_tol=tolerance):
        result.is_correct = True
        result.error_type = "ok"
        result.message = "Expression is valid and equals the target."
    else:
        result.error_type = "not_target"
        result.message = f"Expression value is {value:g}, not {target:g}."
    return result.to_dict()
