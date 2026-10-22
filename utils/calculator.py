# used by: utils\schroedinger.py
"""Safe calculator for user-entered formulas such as ``-5*rect(x/2) + x^2/10``.

The text is parsed into a Python AST and checked against a whitelist (numbers,
variables, constants, ``+ - * / % ^ **`` and whitelisted function calls), so
arbitrary code is rejected. Evaluation is vectorised with numpy.

    f = compile_expression("exp(-x^2) + delta(x - 1)", variables=("x",))
    y = f(np.linspace(-5, 5, 1001), delta_width=0.01)
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

_FloatArray = NDArray[np.float64]

CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

FUNCTIONS: dict[str, Callable[..., _FloatArray]] = {
    "exp": np.exp,
    "log": np.log,
    "ln": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "sign": np.sign,
    "rect": lambda u: (np.abs(u) <= 0.5).astype(float),
    "step": lambda u: (np.asarray(u, dtype=float) >= 0).astype(float),
}
FUNCTIONS["heaviside"] = FUNCTIONS["step"]

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)

# delta needs a width (usually the grid spacing), so it is supplied per call.
_CALL_ONLY_FUNCTIONS = ("delta",)


def triangular_delta(u: _FloatArray, width: float) -> _FloatArray:
    """Unit-area triangular pulse of half-width ``width`` approximating delta(u)."""
    return np.maximum(0.0, 1.0 - np.abs(np.asarray(u, dtype=float)) / width) / width


class Expression:
    """A validated, compiled formula. Call it with one array per variable."""

    def __init__(
        self,
        text: str,
        code,
        variables: tuple[str, ...],
        functions: dict[str, Callable[..., _FloatArray]],
        constants: dict[str, float],
    ) -> None:
        self.text = text
        self._code = code
        self.variables = variables
        self._functions = functions
        self._constants = constants

    def __call__(
        self, *args: _FloatArray, delta_width: float = 1e-2, check_finite: bool = True
    ) -> _FloatArray:
        if len(args) != len(self.variables):
            raise TypeError(f"expected {len(self.variables)} argument(s)")
        values = [np.asarray(a, dtype=float) for a in args]
        ns = {
            **self._constants,
            **self._functions,
            "delta": lambda u: triangular_delta(u, delta_width),
            **dict(zip(self.variables, values)),
        }
        with np.errstate(all="ignore"):
            result = eval(self._code, {"__builtins__": {}}, ns)  # noqa: S307
        shape = np.broadcast(*values).shape
        result = np.broadcast_to(np.asarray(result, dtype=float), shape).copy()
        if check_finite and not np.all(np.isfinite(result)):
            raise ValueError("Ausdruck ist an manchen Stellen nicht endlich")
        return result


def compile_expression(
    text: str,
    variables: Sequence[str] = ("x",),
    functions: dict[str, Callable[..., _FloatArray]] | None = None,
    constants: dict[str, float] | None = None,
) -> Expression:
    """Validate ``text`` and compile it; raises ``ValueError`` on bad input.

    ``functions`` and ``constants`` extend the defaults (``FUNCTIONS``,
    ``CONSTANTS``, plus ``delta``). ``^`` is accepted as power operator.
    """
    variables = tuple(variables)
    funcs = {**FUNCTIONS, **(functions or {})}
    consts = {**CONSTANTS, **(constants or {})}
    callable_names = set(funcs) | set(_CALL_ONLY_FUNCTIONS)
    known_names = set(consts) | set(variables) | callable_names

    source = text.strip().replace("^", "**")
    if not source:
        raise ValueError("Leerer Ausdruck")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Syntaxfehler: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Nicht erlaubt: {type(node).__name__}")
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise ValueError("Nur Zahlen als Konstanten erlaubt")
        if isinstance(node, ast.Name) and node.id not in known_names:
            raise ValueError(f"Unbekannter Name: {node.id}")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.keywords
            or node.func.id not in callable_names
        ):
            raise ValueError("Ungültiger Funktionsaufruf")

    return Expression(
        text, compile(tree, "<expression>", "eval"), variables, funcs, consts
    )
