"""
specola.spec — specification language layer.

parser.parse(text)      syntax only: formula text → immutable AST
semantic.compile_formula(...)  AST + context → executable SpecTemplate
"""

from __future__ import annotations

from . import ast  # noqa: F401
from .parser import parse  # noqa: F401
from .semantic import compile_formula  # noqa: F401
