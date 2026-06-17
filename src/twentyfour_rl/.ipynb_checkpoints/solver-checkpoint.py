from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations


@dataclass(frozen=True)
class ExprState:
    value: Fraction
    expr: str


def solve_24(numbers: list[int] | tuple[int, ...], target: int = 24) -> str | None:
    """
    24点穷举求解器（分数精确计算，无浮点误差）
    用于：训练集真值生成、样本筛选、标准答案校验
    返回合法表达式，无解返回 None
    """
    states = [ExprState(Fraction(int(n), 1), str(int(n))) for n in numbers]
    solution = _search(states, Fraction(target, 1))
    return solution.expr if solution else None


def _search(states: list[ExprState], target: Fraction) -> ExprState | None:
    # 递归终止：只剩一个表达式，判断是否等于目标值
    if len(states) == 1:
        return states[0] if states[0].value == target else None

    # 两两组合遍历所有数字配对
    for i, j in combinations(range(len(states)), 2):
        left = states[i]
        right = states[j]
        # 剩余未参与组合的数字/表达式
        rest = [s for idx, s in enumerate(states) if idx not in (i, j)]
        # 遍历当前两个数字的所有运算组合
        for combined in _combine(left, right):
            result = _search(rest + [combined], target)
            if result is not None:
                return result
    return None


def _combine(a: ExprState, b: ExprState) -> list[ExprState]:
    """
    两个表达式所有合法运算组合
    包含：加减、互换减、乘、双向除（除零保护）
    全程分数运算，杜绝浮点精度误差
    """
    out = [
        ExprState(a.value + b.value, f"({a.expr}+{b.expr})"),
        ExprState(a.value - b.value, f"({a.expr}-{b.expr})"),
        ExprState(b.value - a.value, f"({b.expr}-{a.expr})"),
        ExprState(a.value * b.value, f"({a.expr}*{b.expr})"),
    ]
    # 防止除零错误，双向除法
    if b.value != 0:
        out.append(ExprState(a.value / b.value, f"({a.expr}/{b.expr})"))
    if a.value != 0:
        out.append(ExprState(b.value / a.value, f"({b.expr}/{a.expr})"))
    return out


def has_solution(numbers: list[int] | tuple[int, ...], target: int = 24) -> bool:
    """快速判断一组数字是否存在24点解，用于数据过滤"""
    return solve_24(numbers, target) is not None


if __name__ == "__main__":
    # 测试示例
    test_case = [1, 2, 3, 4]
    ans = solve_24(test_case)
    print(f"测试用例 {test_case} 求解结果：{ans}")
