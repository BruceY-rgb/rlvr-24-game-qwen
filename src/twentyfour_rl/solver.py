from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations


@dataclass(frozen=True)
class ExprState:
    value: Fraction
    expr: str


def solve_24(numbers: list[int] | tuple[int, ...], target: int = 24) -> str | None:
    states = [ExprState(Fraction(int(n), 1), str(int(n))) for n in numbers]
    solution = _search(states, Fraction(target, 1))
    return solution.expr if solution else None


def _search(states: list[ExprState], target: Fraction) -> ExprState | None:
    if len(states) == 1:
        return states[0] if states[0].value == target else None

    for i, j in combinations(range(len(states)), 2):
        left = states[i]
        right = states[j]
        rest = [s for idx, s in enumerate(states) if idx not in (i, j)]
        for combined in _combine(left, right):
            result = _search(rest + [combined], target)
            if result is not None:
                return result
    return None


def _combine(a: ExprState, b: ExprState) -> list[ExprState]:
    out = [
        ExprState(a.value + b.value, f"({a.expr}+{b.expr})"),
        ExprState(a.value - b.value, f"({a.expr}-{b.expr})"),
        ExprState(b.value - a.value, f"({b.expr}-{a.expr})"),
        ExprState(a.value * b.value, f"({a.expr}*{b.expr})"),
    ]
    if b.value != 0:
        out.append(ExprState(a.value / b.value, f"({a.expr}/{b.expr})"))
    if a.value != 0:
        out.append(ExprState(b.value / a.value, f"({b.expr}/{a.expr})"))
    return out

