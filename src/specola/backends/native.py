"""
native.py — native temporal backend for the declared STL fragment.

Supported fragment (see backends.selection): atomic predicates, Boolean
combinations (not/and/or/->/<->/xor), bounded and unbounded past
historically/once/since, previous, rise, and fall, nested arbitrarily.

Semantics match the pipeline's pinned conventions exactly (validated by
the backend-parity tests): and=min, or=max, not=negate, derived forms
for ->/<->/xor; an authored [0,N] window covers max(N,1) samples; while
a window is still filling, the value is the robustness over the
available prefix (provisional warmup semantics).

Atomic predicates are named mu_0, mu_1, … in depth-first left-to-right
order over the AST — the same order the semantic compiler assigns when
compiling predicate functions.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Mapping

from ..spec import ast
from ..spec.intervals import convert_interval


class _Atom:
    def __init__(self, name: str) -> None:
        self.name = name

    def evaluate(self, signals: Mapping[str, float]) -> float:
        return signals[self.name]

    def reset(self) -> None:
        pass


class _Unary:
    def __init__(self, child) -> None:
        self.child = child

    def reset(self) -> None:
        self.child.reset()


class _Not(_Unary):
    def evaluate(self, signals):
        return -self.child.evaluate(signals)


class _Binary:
    def __init__(self, left, right) -> None:
        self.left = left
        self.right = right

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()


class _And(_Binary):
    def evaluate(self, signals):
        return min(
            self.left.evaluate(signals), self.right.evaluate(signals)
        )


class _Or(_Binary):
    def evaluate(self, signals):
        return max(
            self.left.evaluate(signals), self.right.evaluate(signals)
        )


class _Implies(_Binary):
    def evaluate(self, signals):
        return max(
            -self.left.evaluate(signals), self.right.evaluate(signals)
        )


class _Iff(_Binary):
    def evaluate(self, signals):
        left = self.left.evaluate(signals)
        right = self.right.evaluate(signals)
        return min(max(-left, right), max(-right, left))


class _Xor(_Binary):
    def evaluate(self, signals):
        left = self.left.evaluate(signals)
        right = self.right.evaluate(signals)
        return -min(max(-left, right), max(-right, left))


class _Window(_Unary):
    """Bounded past min/max with identity padding before ``lower``."""

    def __init__(self, child, lower: int, upper: int, is_min: bool) -> None:
        super().__init__(child)
        self._buffer = deque(maxlen=upper + 1)
        self._lower = lower
        self._is_min = is_min

    def evaluate(self, signals):
        self._buffer.append(self.child.evaluate(signals))
        if len(self._buffer) <= self._lower:
            return math.inf if self._is_min else -math.inf

        values = list(self._buffer)
        eligible = values[: len(values) - self._lower]
        return min(eligible) if self._is_min else max(eligible)

    def reset(self) -> None:
        self._buffer.clear()
        super().reset()


class _RunningWindow(_Unary):
    """Unbounded running min/max since the contiguous segment origin."""

    def __init__(self, child, is_min: bool) -> None:
        super().__init__(child)
        self._is_min = is_min
        self._value = None

    def evaluate(self, signals):
        current = self.child.evaluate(signals)
        if self._value is None:
            self._value = current
        elif self._is_min:
            self._value = min(self._value, current)
        else:
            self._value = max(self._value, current)
        return self._value

    def reset(self) -> None:
        self._value = None
        super().reset()


class _Previous(_Unary):
    def __init__(self, child) -> None:
        super().__init__(child)
        self._previous = None

    def evaluate(self, signals):
        current = self.child.evaluate(signals)
        result = -math.inf if self._previous is None else self._previous
        self._previous = current
        return result

    def reset(self) -> None:
        self._previous = None
        super().reset()


class _Rise(_Unary):
    def __init__(self, child) -> None:
        super().__init__(child)
        self._previous = None

    def evaluate(self, signals):
        current = self.child.evaluate(signals)
        result = (
            current
            if self._previous is None
            else min(current, -self._previous)
        )
        self._previous = current
        return result

    def reset(self) -> None:
        self._previous = None
        super().reset()


class _Fall(_Unary):
    def __init__(self, child) -> None:
        super().__init__(child)
        self._previous = None

    def evaluate(self, signals):
        current = self.child.evaluate(signals)
        result = (
            -current
            if self._previous is None
            else min(-current, self._previous)
        )
        self._previous = current
        return result

    def reset(self) -> None:
        self._previous = None
        super().reset()


class _Since(_Binary):
    """Bounded or unbounded quantitative past-time since."""

    def __init__(self, left, right, upper=None, lower=0) -> None:
        super().__init__(left, right)
        self._lower = lower
        self._upper = upper
        self._value = -math.inf
        maxlen = upper + 1 if upper is not None else None
        self._left_buffer = deque(maxlen=maxlen)
        self._right_buffer = deque(maxlen=maxlen)

    def evaluate(self, signals):
        left = self.left.evaluate(signals)
        right = self.right.evaluate(signals)

        if self._upper is None:
            self._value = max(right, min(left, self._value))
            return self._value

        self._left_buffer.append(left)
        self._right_buffer.append(right)
        if len(self._left_buffer) <= self._lower:
            return -math.inf

        left_values = list(self._left_buffer)
        right_values = list(self._right_buffer)
        suffix_min = [math.inf] * (len(left_values) + 1)
        for index in range(len(left_values) - 1, -1, -1):
            suffix_min[index] = min(
                left_values[index], suffix_min[index + 1]
            )

        last_eligible = len(left_values) - 1 - self._lower
        return max(
            min(right_values[index], suffix_min[index + 1])
            for index in range(last_eligible + 1)
        )

    def reset(self) -> None:
        self._value = -math.inf
        self._left_buffer.clear()
        self._right_buffer.clear()
        super().reset()


class NativeBackend:
    """TemporalBackend evaluating the fragment directly over the AST."""

    def __init__(self, root: ast.FormulaNode, dt: float) -> None:
        counter: List[int] = [0]
        self._root = _build(root, dt, counter)

    def update(
        self,
        timestep: int,
        signals: Mapping[str, float],
    ) -> float:
        del timestep  # windows advance one sample per call
        return float(self._root.evaluate(signals))

    def reset(self) -> None:
        self._root.reset()


def _build(node: ast.FormulaNode, dt: float, counter: List[int]):
    if isinstance(node, ast.Comparison):
        name = f"mu_{counter[0]}"
        counter[0] += 1
        return _Atom(name)

    if isinstance(node, ast.Not):
        return _Not(_build(node.child, dt, counter))

    if isinstance(node, (ast.And, ast.Or, ast.Implies, ast.Iff, ast.Xor)):
        left = _build(node.left, dt, counter)
        right = _build(node.right, dt, counter)
        node_type = {
            ast.And: _And,
            ast.Or: _Or,
            ast.Implies: _Implies,
            ast.Iff: _Iff,
            ast.Xor: _Xor,
        }[type(node)]
        return node_type(left, right)

    if isinstance(node, (ast.Always, ast.Eventually)):
        child = _build(node.child, dt, counter)
        if node.bounds is None:
            return _RunningWindow(
                child,
                is_min=isinstance(node, ast.Always),
            )

        lower, upper = convert_interval(node.bounds, dt)
        return _Window(
            child,
            lower=lower,
            upper=upper,
            is_min=isinstance(node, ast.Always),
        )

    if isinstance(node, ast.Since):
        left = _build(node.left, dt, counter)
        right = _build(node.right, dt, counter)
        if node.bounds is None:
            return _Since(left, right)
        lower, upper = convert_interval(node.bounds, dt)
        return _Since(left, right, lower=lower, upper=upper)

    if isinstance(node, (ast.Previous, ast.Rise, ast.Fall)):
        child = _build(node.child, dt, counter)
        node_type = {
            ast.Previous: _Previous,
            ast.Rise: _Rise,
            ast.Fall: _Fall,
        }[type(node)]
        return node_type(child)

    raise ValueError(
        f"AST node {type(node).__name__} is outside the native fragment"
    )
