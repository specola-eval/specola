"""
stl_parser.py — compatibility entry point for the specification layer
=====================================================================

``parse_stl`` is the stable public API: formula text in, executable
``SpecTemplate`` out. Internally the work is split per the target
architecture:

    specola.spec.parser      syntax only → immutable AST
    specola.spec.semantic    slots, validation, dt conversion,
                                 predicate compilation
    specola.backends.rtamt_text   RTAMT formula-text fragments

The syntax parser takes nothing but the formula text; every
context-dependent concern (dt, constants, class vocabulary, component
catalog, backend syntax) lives in semantic compilation. Parsing a spec
file once and compiling it per episode against the connected adapter is
therefore possible; this wrapper simply does both steps at once for
callers that want the legacy single-call behavior.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

from .components import ComponentCatalog, ComponentResolver
from .entities import SpecConfigError
from .monitor import SpecTemplate, ParamKey
from .spec import ast as _ast
from .spec import parser as _spec_parser
from .spec import semantic as _semantic


PredicateFn = Callable[
    [Dict[ParamKey, int], ComponentResolver],
    float,
]


def parse_stl(
    formula_str: str,
    dt: float = 0.05,
    constants: Optional[Mapping[str, float]] = None,
    symmetric: bool = False,
    *,
    catalog: Optional[ComponentCatalog] = None,
    valid_classes: Optional[Iterable[str]] = None,
    backend: str = "auto",
) -> SpecTemplate:
    """
    Parse and compile a causal past-time STL spec template.

    Parameters
    ----------
    formula_str:
        Spec template source text, in the surface STL syntax.

    dt:
        Sampling period used for bounds suffixed with ``s``.

    constants:
        Optional named finite scalar constants used in arithmetic
        expressions.

    symmetric:
        Set True only when permuting equal-class wildcard slots leaves the
        complete formula invariant. Symmetry is not inferred automatically.

    catalog:
        Episode-level component catalog produced after adapter discovery.
        When provided, compilation performs exact class, component, and
        pinned entity validation.

    valid_classes:
        Optional exact class vocabulary when a full catalog is unavailable.
        This supports class resolution but cannot validate components.

    backend:
        Temporal backend selection mode: ``"auto"`` (native for the causal
        language), ``"native"``, or ``"rtamt"`` (fail if unavailable).
        Resolved here, at compilation.
    """
    node, compile_kwargs = _prepare(
        formula_str,
        dt,
        constants,
        symmetric,
        catalog,
        valid_classes,
        backend,
        None,
    )

    # Trivial formulas (no selectors, no "**") stay on the original
    # single-formula path byte-for-byte. A selector formula is desugared;
    # parse_stl accepts it only when it expands to exactly one monitor.
    if not _semantic._needs_desugaring(node):
        return _semantic.compile_formula(node, **compile_kwargs)

    formulas = _semantic.compile_formulas(node, **compile_kwargs)
    if len(formulas) != 1:
        raise SpecConfigError(
            f"this spec expands to {len(formulas)} monitors; use "
            f"compile_specs"
        )
    return formulas[0]


def compile_specs(
    formula_str: str,
    dt: float = 0.05,
    constants: Optional[Mapping[str, float]] = None,
    symmetric: bool = False,
    *,
    catalog: Optional[ComponentCatalog] = None,
    valid_classes: Optional[Iterable[str]] = None,
    backend: str = "auto",
    capabilities=None,
) -> List[SpecTemplate]:
    """Parse and compile a spec into one or more ``SpecTemplate``.

    This is the selector-aware entry point: a formula that fans out (a bare
    selector enumerated per key, ``**`` over discovered entities, etc.)
    returns the whole list, each formula's ``name`` carrying its expansion
    suffix. A trivial formula returns a one-element list, identical to
    ``parse_stl``'s result. Selectors require a ``catalog``.

    Accepts exactly the same keyword arguments as ``parse_stl``, plus
    ``capabilities``: the adapter's declared ``AdapterCapabilities``.
    Passing it aligns compilation with ``specola run``'s episode-start
    authority resolution (section 5.6.1) — a CLOSED class vocabulary
    comes from the declaration rather than the reset snapshot, and a
    class with a CLOSED declared component schema may compile against
    that schema even when it has no live instance at reset.
    """
    node, compile_kwargs = _prepare(
        formula_str,
        dt,
        constants,
        symmetric,
        catalog,
        valid_classes,
        backend,
        capabilities,
    )
    return _semantic.compile_formulas(node, **compile_kwargs)


def _prepare(
    formula_str: str,
    dt: float,
    constants: Optional[Mapping[str, float]],
    symmetric: bool,
    catalog: Optional[ComponentCatalog],
    valid_classes: Optional[Iterable[str]],
    backend: str,
    capabilities=None,
) -> Tuple[_ast.FormulaNode, Dict[str, object]]:
    """Shared validation/parsing for ``parse_stl`` and ``compile_specs``."""
    if not isinstance(formula_str, str) or not formula_str.strip():
        raise ValueError("formula_str must be a non-empty string")

    if isinstance(dt, bool) or not isinstance(dt, Real):
        raise TypeError("dt must be a real scalar")

    numeric_dt = float(dt)
    if not math.isfinite(numeric_dt) or numeric_dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt!r}")

    if not isinstance(symmetric, bool):
        raise TypeError("symmetric must be bool")

    if catalog is not None and not isinstance(catalog, ComponentCatalog):
        raise TypeError("catalog must be ComponentCatalog or None")

    validated_constants = _semantic.validate_constants(constants or {})
    normalized_classes = _normalize_classes(valid_classes)

    if catalog is not None:
        catalog_classes = catalog.classes

        if (
            normalized_classes is not None
            and not catalog_classes.issubset(normalized_classes)
        ):
            raise ValueError(
                "catalog contains classes outside valid_classes: "
                f"{sorted(catalog_classes - normalized_classes)}"
            )

        # The episode catalog is authoritative for resolving exact class
        # names and legacy uppercase label suffixes.
        class_vocabulary: Optional[FrozenSet[str]] = catalog_classes
    else:
        class_vocabulary = normalized_classes

    # Declared capabilities outrank the reset snapshot, mirroring
    # MonitorEngine.start_episode: a CLOSED class vocabulary comes from
    # the declaration, so classes with empty reset populations (dynamic
    # contact pairs) resolve during check exactly as during run.
    if capabilities is not None:
        from .capabilities import VocabularyMode as _VM

        if capabilities.class_mode == _VM.CLOSED:
            class_vocabulary = frozenset(capabilities.supported_classes)

    node = _spec_parser.parse(formula_str)

    compile_kwargs: Dict[str, object] = {
        "formula_name": formula_str.strip(),
        "dt": numeric_dt,
        "constants": validated_constants,
        "symmetric": symmetric,
        "catalog": catalog,
        "class_vocabulary": class_vocabulary,
        "backend": backend,
        "capabilities": capabilities,
    }
    return node, compile_kwargs


def describe_formula(formula: SpecTemplate) -> str:
    """Return a human-readable description of a parsed formula."""
    predicates = formula.predicate_fns or {}

    lines = [
        f"Formula: {formula.name}",
        f"RTAMT:   {formula.rtamt_spec or '<legacy-single-mu>'}",
        (
            f"History: {formula.window_steps} steps "
            f"({formula.window_seconds:.6g}s)"
        ),
        f"Predicates: {len(predicates)}",
        f"Symmetric bindings: {formula.symmetric}",
        f"Parameters ({len(formula.params)}):",
    ]

    for parameter in formula.params:
        mode = (
            "wildcard"
            if parameter.is_wildcard
            else f"pinned(id={parameter.pinned_entity_id})"
        )
        lines.append(
            f"  x=({parameter.entity_class},{parameter.param_index})  "
            f"{mode}  req={parameter.required_components}"
        )

    return "\n".join(lines)


def _normalize_classes(
    classes: Optional[Iterable[str]],
) -> Optional[FrozenSet[str]]:
    if classes is None:
        return None

    normalized = frozenset(classes)

    for class_name in normalized:
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(
                "valid_classes must contain non-empty strings"
            )

    return normalized
