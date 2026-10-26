"""
monitor_types.py — shared binding types (owner decision 6).

A runtime binding is an immutable tuple of entity IDs aligned with the
immutable parameter-slot ordering of a compiled specification;
``binding[i]`` binds ``formula.params[i]``. Binding interpretation
always occurs in the context of that compiled specification, so the
conversion helpers take the parameter list.

Kept in a leaf module so the bindings package and monitor.py can share
these without an import cycle.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple


ParamKey = Tuple[Any, int]          # (entity_class, per-class index)
Binding = Tuple[int, ...]           # slot-aligned entity IDs


def make_binding(
    mapping: Dict[ParamKey, int],
    params: Sequence[Any],
) -> Binding:
    """Order a {param_key: entity_id} assignment by the slot ordering."""
    return tuple(mapping[p.param_key] for p in params)


def binding_to_dict(
    binding: Binding,
    params: Sequence[Any],
) -> Dict[ParamKey, int]:
    return {p.param_key: eid for p, eid in zip(params, binding)}
