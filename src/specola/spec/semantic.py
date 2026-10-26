"""
semantic.py — semantic compilation of parsed STL formulas.

Turns a syntax AST plus context (dt, constants, class vocabulary,
component catalog) into an executable ``SpecTemplate``:

- resolves entity-reference heads against the class vocabulary;
- merges repeated references into parameter slots, ordered by first
  source occurrence; the order is immutable once compiled;
- enforces same-class injectivity for pinned slots;
- validates constants and rejects collisions with generated signals;
- converts authored temporal bounds to steps using dt;
- compiles atomic predicates to fast numeric functions;
- assembles the RTAMT formula text through the backend fragment module.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import sympy as sp

from ..backends import rtamt_text
from ..backends.native import NativeBackend
from ..backends.rtamt import RtamtBackend
from ..backends.selection import choose_backend
from ..components import ComponentCatalog, ComponentResolver
from ..entities import (
    SpecConfigError,
    UnboundStaticSlotError,
    UnknownClassError,
)
from . import ast, render
from .grammar import PARAM_RE, find_matching_parenthesis
from .intervals import convert_interval
from .selectors import parse_selector, resolve_selector

# Imported late to avoid a cycle: monitor.py does not import this module.
from ..monitor import ParameterSpec, SpecTemplate, ParamKey, TemporalOp


PredicateFn = Callable[[Dict[ParamKey, int], ComponentResolver], float]

_SYMPY_LOCALS = {
    "min": sp.Min,
    "max": sp.Max,
    "Min": sp.Min,
    "Max": sp.Max,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "atan2": sp.atan2,
    "exp": sp.exp,
    "log": sp.log,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "pi": sp.pi,
    "E": sp.E,
}

RESERVED_EXPRESSION_NAMES = frozenset(_SYMPY_LOCALS)


# ---------------------------------------------------------------------------
# Reference resolution (head → class + label)
# ---------------------------------------------------------------------------

def _resolve_exact_class(
    class_name: str,
    class_vocabulary: Optional[FrozenSet[str]],
    strict: bool = True,
) -> str:
    if (
        class_vocabulary is None
        or class_name in class_vocabulary
        or not strict
    ):
        return class_name

    raise UnknownClassError(
        f"Unknown entity class {class_name!r}. Available classes: "
        f"{sorted(class_vocabulary)}"
    )


def resolve_reference_head(
    reference: ast.EntityReference,
    class_vocabulary: Optional[FrozenSet[str]],
    strict: bool = True,
) -> Tuple[str, str]:
    """Resolve an authored head into (entity_class, label).

    Exact class matches always take precedence, so a class literally named
    ``robot_A`` is never reinterpreted as class ``robot`` with label ``A``.
    """
    if reference.explicit_label is not None:
        entity_class = _resolve_exact_class(
            reference.head, class_vocabulary, strict
        )
        return entity_class, reference.explicit_label

    head = reference.head

    if class_vocabulary is not None and head in class_vocabulary:
        return head, ""

    base, separator, suffix = head.rpartition("_")
    looks_like_legacy_label = (
        bool(separator) and len(suffix) == 1 and suffix.isupper()
    )

    if class_vocabulary is not None:
        if looks_like_legacy_label and base in class_vocabulary:
            return base, suffix

        if not strict and not looks_like_legacy_label:
            # Open class vocabulary: defer unknown references — the
            # head is its own class, eligible once such entities appear.
            return head, ""

        raise UnknownClassError(
            f"Cannot resolve parameter prefix {head!r}. Available "
            f"classes: {sorted(class_vocabulary)}. Use class{{A}}_*[...] "
            f"for an unambiguous labelled parameter."
        )

    if looks_like_legacy_label:
        raise SpecConfigError(
            f"Parameter prefix {head!r} is ambiguous without a discovered "
            f"class vocabulary. It may represent class {head!r}, or class "
            f"{base!r} with label {suffix!r}. Pass catalog=..., pass "
            f"valid_classes=..., or use the unambiguous syntax "
            f"{base}{{{suffix}}}_*[...]."
        )

    return head, ""


# ---------------------------------------------------------------------------
# Selector-syntax guards (stage 3, task 2)
#
# The syntax parser now accepts "**" bindings and non-trivial component
# selectors (spec/selectors.py), but resolving either against a discovered
# catalog is Task 3's job. Anything that reaches this OLD semantic-compile
# path must refuse them loudly rather than silently mis-treating "**" as a
# wildcard/pinned binding, or a selector's raw text as a literal component
# name.
# ---------------------------------------------------------------------------

_DOUBLE_STAR_MESSAGE = (
    "** requires selector compilation; use compile_specs"
)


def _reject_double_star_binding(binding: str) -> None:
    if binding == "**":
        raise SpecConfigError(_DOUBLE_STAR_MESSAGE)


def _reject_unresolved_reference(reference: ast.EntityReference) -> None:
    """Guard applied to every reference entering the old compile path."""
    _reject_double_star_binding(reference.binding)

    if not reference.selector.is_trivial:
        raise SpecConfigError(
            f"Component selector {reference.selector.raw!r} requires "
            "selector compilation; use compile_specs"
        )


# ---------------------------------------------------------------------------
# Parameter-slot table
# ---------------------------------------------------------------------------

@dataclass
class _SlotTable:
    """Parameter slots keyed by (class, label, binding), in source order."""

    class_vocabulary: Optional[FrozenSet[str]]
    strict_classes: bool = True

    key_map: Dict[Tuple[str, str, str], ParamKey] = field(
        default_factory=dict
    )
    info: Dict[ParamKey, Tuple[str, str, str]] = field(default_factory=dict)
    components: Dict[ParamKey, Set[str]] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    labelled_bindings: Dict[Tuple[str, str], str] = field(
        default_factory=dict
    )
    occurrence_order: List[ParamKey] = field(default_factory=list)

    def register(self, reference: ast.EntityReference) -> Tuple[ParamKey, str]:
        """Register one reference; return (param_key, entity_class)."""
        _reject_unresolved_reference(reference)

        entity_class, label = resolve_reference_head(
            reference, self.class_vocabulary, self.strict_classes
        )

        if label:
            label_key = (entity_class, label)
            previous = self.labelled_bindings.get(label_key)
            if previous is not None and previous != reference.binding:
                raise ValueError(
                    f"Parameter label {entity_class!r}{{{label}}} is "
                    f"reused with bindings {previous!r} and "
                    f"{reference.binding!r}."
                )
            self.labelled_bindings[label_key] = reference.binding

        lookup_key = (entity_class, label, reference.binding)

        if lookup_key not in self.key_map:
            parameter_index = self.counters.get(entity_class, 0) + 1
            self.counters[entity_class] = parameter_index
            param_key = (entity_class, parameter_index)
            self.key_map[lookup_key] = param_key
            self.info[param_key] = lookup_key
            self.components[param_key] = set()
            self.occurrence_order.append(param_key)

        param_key = self.key_map[lookup_key]
        self.components[param_key].add(reference.component)
        return param_key, entity_class

    def signal_name(self, reference: ast.EntityReference) -> str:
        _reject_unresolved_reference(reference)

        entity_class, label = resolve_reference_head(
            reference, self.class_vocabulary, self.strict_classes
        )
        param_key = self.key_map[
            (entity_class, label, reference.binding)
        ]
        return (
            f"{param_key[0]}_{param_key[1]}_{reference.component}"
        )

    def symbol_map(self) -> Dict[str, Tuple[ParamKey, str]]:
        mapping: Dict[str, Tuple[ParamKey, str]] = {}
        for param_key, components in self.components.items():
            entity_class, parameter_index = param_key
            for component in components:
                mapping[
                    f"{entity_class}_{parameter_index}_{component}"
                ] = (param_key, component)
        return mapping


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def validate_constants(
    constants: Mapping[str, float],
) -> Dict[str, float]:
    from numbers import Real
    from .grammar import CONSTANT_NAME_RE

    if not isinstance(constants, Mapping):
        raise TypeError("constants must be a mapping")

    validated: Dict[str, float] = {}

    for name, value in constants.items():
        if not isinstance(name, str) or not CONSTANT_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid constant name {name!r}")

        if name in RESERVED_EXPRESSION_NAMES:
            raise ValueError(f"Constant name {name!r} is reserved")

        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Constant {name!r} must be a real scalar")

        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Constant {name!r} must be finite")

        validated[name] = numeric_value

    return validated


# ---------------------------------------------------------------------------
# Predicate compilation
# ---------------------------------------------------------------------------

def _compile_predicate(
    expression: str,
    symbol_map: Mapping[str, Tuple[ParamKey, str]],
    constants: Mapping[str, float],
) -> PredicateFn:
    substitutions = {
        sp.Symbol(name): value for name, value in constants.items()
    }

    # Constant names must always parse as plain symbols, even when they
    # collide with sympy's default namespace (e.g. 'limit', 'beta').
    sympify_locals = dict(_SYMPY_LOCALS)
    for name in constants:
        sympify_locals[name] = sp.Symbol(name)

    try:
        symbolic_expression = sp.sympify(
            expression, locals=sympify_locals
        ).subs(substitutions)
    except Exception as exc:
        raise ValueError(
            f"Could not parse numeric expression {expression!r}: {exc}"
        ) from exc

    free_symbols = sorted(
        symbolic_expression.free_symbols,
        key=lambda symbol: symbol.name,
    )

    unknown_symbols = sorted(
        symbol.name
        for symbol in free_symbols
        if symbol.name not in symbol_map
    )
    if unknown_symbols:
        raise SpecConfigError(
            f"Unknown symbol(s) {unknown_symbols}. Use parameter syntax "
            f"such as object_*[mass], or provide them through constants."
        )

    if not free_symbols:
        try:
            constant_value = float(symbolic_expression)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Expression {expression!r} does not evaluate to a scalar"
            ) from exc

        if not math.isfinite(constant_value):
            raise ValueError(
                f"Expression {expression!r} evaluates to non-finite "
                f"value {constant_value!r}"
            )

        return lambda binding_dict, resolver: constant_value

    try:
        fast_function = sp.lambdify(
            free_symbols, symbolic_expression, modules="numpy"
        )
    except Exception as exc:
        raise ValueError(
            f"Could not compile predicate expression "
            f"{expression!r}: {exc}"
        ) from exc

    symbol_names = [symbol.name for symbol in free_symbols]
    bound_symbols = [
        (name,) + symbol_map[name] for name in symbol_names
    ]

    def predicate_fn(
        binding_dict: Dict[ParamKey, int],
        resolver: ComponentResolver,
    ) -> float:
        if not isinstance(resolver, ComponentResolver):
            raise TypeError(
                "Compiled STL predicates require a ComponentResolver. "
                "monitor.py must create one resolver per observation and "
                "pass it to every active MonitorInstance."
            )

        arguments: List[float] = []

        for _name, param_key, component in bound_symbols:
            if param_key not in binding_dict:
                raise SpecConfigError(
                    f"Binding is missing parameter {param_key!r}"
                )

            entity_id = binding_dict[param_key]
            entity = resolver.entity(entity_id)
            expected_class = param_key[0]

            if entity.entity_class != expected_class:
                raise SpecConfigError(
                    f"Parameter {param_key!r} requires class "
                    f"{expected_class!r}, but entity {entity_id} has "
                    f"class {entity.entity_class!r}."
                )

            # Binding enumeration must already have checked supports_all().
            arguments.append(resolver.value(entity_id, component))

        try:
            result = np.asarray(fast_function(*arguments))
        except SpecConfigError:
            raise
        except Exception as exc:
            raise SpecConfigError(
                f"Numeric evaluation failed for predicate "
                f"{expression!r}: {exc}"
            ) from exc

        if result.ndim != 0:
            raise SpecConfigError(
                f"Predicate {expression!r} returned non-scalar shape "
                f"{result.shape}"
            )

        value = float(result)
        if not np.isfinite(value):
            raise SpecConfigError(
                f"Predicate {expression!r} evaluated to non-finite "
                f"value {value!r}"
            )

        return value

    return predicate_fn


# ---------------------------------------------------------------------------
# Compile walk
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _WalkResult:
    text: str
    max_age: int
    is_atomic: bool = False
    top_temporal: Optional[str] = None
    simple_window: Optional[str] = None


class _Compiler:
    def __init__(
        self,
        dt: float,
        constants: Mapping[str, float],
        class_vocabulary: Optional[FrozenSet[str]],
    ) -> None:
        self.dt = dt
        self.constants = constants
        self.slots = _SlotTable(class_vocabulary=class_vocabulary)
        self.predicate_fns: Dict[str, PredicateFn] = {}
        self._next_mu = 0

    def walk(self, node: ast.FormulaNode) -> _WalkResult:
        if isinstance(node, ast.Comparison):
            return self._compile_comparison(node)

        if isinstance(node, ast.Not):
            child = self.walk(node.child)
            return _WalkResult(
                text=rtamt_text.negation(child.text),
                max_age=child.max_age,
            )

        if isinstance(node, (ast.And, ast.Or, ast.Implies, ast.Iff, ast.Xor)):
            left = self.walk(node.left)
            right = self.walk(node.right)
            combine = {
                ast.And: rtamt_text.conjunction,
                ast.Or: rtamt_text.disjunction,
                ast.Implies: rtamt_text.implication,
                ast.Iff: rtamt_text.equivalence,
                ast.Xor: rtamt_text.exclusive_or,
            }[type(node)]
            return _WalkResult(
                text=combine(left.text, right.text),
                max_age=max(left.max_age, right.max_age),
            )

        if isinstance(node, (ast.Always, ast.Eventually)):
            child = self.walk(node.child)

            if node.bounds is None:
                lower = upper = 0
                max_age = child.max_age
                if isinstance(node, ast.Always):
                    kind = "always"
                    text = rtamt_text.historically_unbounded(child.text)
                else:
                    kind = "eventually"
                    text = rtamt_text.once_unbounded(child.text)
            else:
                lower, upper = convert_interval(node.bounds, self.dt)
                interval_text = rtamt_text.interval(lower, upper)
                max_age = upper + child.max_age
                if isinstance(node, ast.Always):
                    kind = "always"
                    text = rtamt_text.historically(
                        interval_text, child.text
                    )
                else:
                    kind = "eventually"
                    text = rtamt_text.once(interval_text, child.text)

            return _WalkResult(
                text=text,
                max_age=max_age,
                top_temporal=kind,
                # A trailing window directly over a comparison, and only
                # that shape, has a warm-up value equal to the running
                # min (always) or max (eventually) of the child's
                # robustness over the samples seen so far. That
                # monotonicity is what lets a verdict settle before the
                # window fills; see MonitorInstance.verdict_final.
                #
                # A nonzero lower bound breaks it: the window is
                # [t-upper, t-lower], so a sample has to age *into* the
                # window before it counts, and a value seen now may not
                # be in the window when it fills. A non-atomic child
                # breaks it too, because an inner window's own warm-up
                # value is provisional.
                simple_window=(
                    kind
                    if (
                        node.bounds is not None
                        and child.is_atomic
                        and lower == 0
                    )
                    else None
                ),
            )

        if isinstance(node, ast.Since):
            left = self.walk(node.left)
            right = self.walk(node.right)
            if node.bounds is None:
                text = rtamt_text.since_unbounded(left.text, right.text)
                max_age = max(left.max_age, right.max_age)
            else:
                lower, upper = convert_interval(node.bounds, self.dt)
                interval_text = rtamt_text.interval(lower, upper)
                text = rtamt_text.since(
                    left.text, interval_text, right.text
                )
                max_age = upper + max(left.max_age, right.max_age)
            return _WalkResult(
                text=text,
                max_age=max_age,
                top_temporal="since",
            )

        if isinstance(node, (ast.Previous, ast.Rise, ast.Fall)):
            child = self.walk(node.child)
            emit = {
                ast.Previous: rtamt_text.previous,
                ast.Rise: rtamt_text.rise,
                ast.Fall: rtamt_text.fall,
            }[type(node)]
            return _WalkResult(
                text=emit(child.text),
                # All three operators have a total origin value when their
                # child is immediately defined. A child with a nonzero
                # warm-up horizon needs one additional sample: at the
                # child's fill boundary these operators still consume its
                # provisional value from the preceding sample.
                max_age=(
                    child.max_age + 1
                    if child.max_age > 0
                    else 0
                ),
                top_temporal={
                    ast.Previous: "previous",
                    ast.Rise: "rise",
                    ast.Fall: "fall",
                }[type(node)],
            )

        raise TypeError(f"Unknown AST node {type(node).__name__}")

    def _compile_comparison(self, node: ast.Comparison) -> _WalkResult:
        # Slots are registered in authored source order (left operand
        # first), which fixes the canonical, immutable slot ordering.
        for reference in node.references:
            self.slots.register(reference)

        if node.operator in {">=", ">"}:
            robustness = f"({node.left}) - ({node.right})"
        elif node.operator in {"<=", "<"}:
            robustness = f"({node.right}) - ({node.left})"
        else:  # '=='; '!=' is rejected by the syntax parser
            robustness = f"-abs(({node.left}) - ({node.right}))"

        rewritten = (
            PARAM_RE.sub(
                lambda match: self.slots.signal_name(
                    ast.EntityReference(
                        head=match.group("head"),
                        explicit_label=match.group("explicit_label"),
                        binding=match.group("binding"),
                        component=match.group("component"),
                        selector=parse_selector(
                            match.group("component")
                        ),
                    )
                ),
                robustness,
            )
            .replace("^", "**")
        )

        predicate_name = f"mu_{self._next_mu}"
        self._next_mu += 1

        self.predicate_fns[predicate_name] = _compile_predicate(
            rewritten,
            self.slots.symbol_map(),
            self.constants,
        )

        return _WalkResult(
            text=predicate_name,
            max_age=0,
            is_atomic=True,
        )


# ---------------------------------------------------------------------------
# Catalog validation
# ---------------------------------------------------------------------------

def _unbound_static_slot(
    entity_class: str,
    required_components: Sequence[str],
    catalog: ComponentCatalog,
) -> UnboundStaticSlotError:
    """The refusal a wildcard slot on a static class earns."""
    return UnboundStaticSlotError(
        f"No discovered entity of static class "
        f"{entity_class!r} exposes all required components "
        f"{sorted(required_components)}, and the class is "
        f"not runtime-populated, so none can appear later. "
        f"Components exposed by at least one entity of that "
        f"class: "
        f"{sorted(catalog.components_for_class(entity_class))}"
    )


def _validate_slot_against_catalog(
    entity_class: str,
    binding: str,
    required_components: Sequence[str],
    catalog: Optional[ComponentCatalog],
    entity_ids_closed: bool = True,
    components_closed: bool = True,
    declared_components: Optional[FrozenSet[str]] = None,
    static_class: bool = False,
) -> None:
    """Validate that a slot is realizable in the discovered model.

    Wildcards require at least one entity that jointly exposes all
    required components; the class-level union is not sufficient.
    A class whose component schema is declared CLOSED in the adapter's
    capabilities (``declared_components``) may have zero live instances
    at reset — dynamic populations such as contact pairs — in which case
    the declared schema stands in for a live instance: covered
    references compile and wait, references outside the schema are
    rejected immediately.

    Deferral needs a population that can still change. A wildcard slot on
    a static class (``static_class``: the adapter's class vocabulary is
    CLOSED and the class is not runtime-populated) has its whole
    population at reset, so no candidate can appear later and the slot is
    rejected.
    """
    _reject_double_star_binding(binding)

    if catalog is None:
        return

    if binding == "*":
        compatible = [
            entity_id
            for entity_id in catalog.entity_ids_for_class(entity_class)
            if catalog.entity_info(entity_id).supports_all(
                required_components
            )
        ]
        if compatible:
            return

        if declared_components is not None:
            outside_schema = sorted(
                component
                for component in required_components
                if component not in declared_components
            )
            if not outside_schema:
                if static_class:
                    raise _unbound_static_slot(
                        entity_class, required_components, catalog
                    )
                # Schema-covered dynamic class: compile and wait for
                # the population to appear.
                return
            raise SpecConfigError(
                f"Components {outside_schema} are outside the declared "
                f"component schema of class {entity_class!r}. Declared "
                f"components: {sorted(declared_components)}."
            )

        if not components_closed:
            if static_class:
                raise _unbound_static_slot(
                    entity_class, required_components, catalog
                )
            # Open component vocabulary: defer — a compatible entity
            # may appear later in the episode.
            return

        raise SpecConfigError(
            f"No discovered entity of class {entity_class!r} exposes all "
            f"required components {sorted(required_components)}. "
            f"Components exposed by at least one entity of that class: "
            f"{sorted(catalog.components_for_class(entity_class))}"
        )

    entity_id = int(binding)

    if not catalog.has_entity(entity_id):
        if not entity_ids_closed:
            return  # open entity catalog: compile and wait (§5.6.1)
        raise SpecConfigError(
            f"Pinned entity ID {entity_id} is not present in the "
            f"discovered component catalog."
        )

    entity_info = catalog.entity_info(entity_id)

    if entity_info.entity_class != entity_class:
        raise SpecConfigError(
            f"Pinned parameter requires class {entity_class!r}, but "
            f"entity {entity_id} has class "
            f"{entity_info.entity_class!r}."
        )

    missing = sorted(
        component
        for component in required_components
        if not entity_info.supports(component)
    )
    if missing:
        raise SpecConfigError(
            f"Pinned entity {entity_id} ({entity_class}) does not expose "
            f"required components {missing}. Available components: "
            f"{sorted(entity_info.components)}"
        )


# ---------------------------------------------------------------------------
# Public compilation entry point
# ---------------------------------------------------------------------------

def compile_formula(
    node: ast.FormulaNode,
    *,
    formula_name: str,
    dt: float,
    constants: Mapping[str, float],
    symmetric: bool = False,
    catalog: Optional[ComponentCatalog] = None,
    class_vocabulary: Optional[FrozenSet[str]] = None,
    backend: str = "auto",
    capabilities=None,
    runtime_tag_filters: Optional[
        Mapping[Tuple[str, str], Mapping[str, Tuple[str, ...]]]
    ] = None,
) -> SpecTemplate:
    """Compile a parsed AST into an executable SpecTemplate.

    ``runtime_tag_filters`` maps ``(entity_class, label)`` slots to their
    retained per-component ``@`` labels (from ``_expand_formula``'s
    runtime-filter routing); matching parameter slots carry them as
    ``ParameterSpec.component_tag_filters``.
    """
    # Backend availability and fragment support are resolved here, at
    # compilation — never during a monitor update.
    chosen_backend = choose_backend(backend, node)

    # §5.6.1 authority matrix: with no declared capabilities the catalog
    # is treated as authoritative (closed); declared OPEN modes defer.
    from ..capabilities import VocabularyMode as _VM
    strict_classes = (
        capabilities is None
        or capabilities.class_mode == _VM.CLOSED
    )
    entity_ids_closed = (
        capabilities is None
        or capabilities.entity_id_mode == _VM.CLOSED
    )

    def components_closed_for(cls):
        if capabilities is None:
            return True
        return (
            capabilities.component_mode_by_class.get(cls, _VM.OPEN)
            == _VM.CLOSED
        )

    compiler = _Compiler(
        dt=dt,
        constants=constants,
        class_vocabulary=class_vocabulary,
    )
    compiler.slots.strict_classes = strict_classes

    root = compiler.walk(node)

    if not compiler.predicate_fns:
        raise ValueError("Formula contains no atomic predicates")

    # A constant named like a generated parameter signal would silently
    # shadow that signal in every predicate compiled before the parameter
    # reference was first seen. Reject the collision outright.
    collisions = sorted(
        set(compiler.slots.symbol_map()) & set(constants)
    )
    if collisions:
        raise ValueError(
            f"Constant name(s) {collisions} collide with generated "
            f"parameter signal names. Rename the constant(s)."
        )

    # Same-class injectivity for pinned slots: two distinct slots of one
    # class pinned to the same entity can never be satisfied.
    pinned_by_class: Dict[Tuple[str, str], Tuple[str, int]] = {}
    for param_key in compiler.slots.occurrence_order:
        entity_class, _label, binding = compiler.slots.info[param_key]
        _reject_double_star_binding(binding)
        if binding == "*":
            continue
        duplicate_key = (entity_class, binding)
        if duplicate_key in pinned_by_class:
            raise SpecConfigError(
                f"Two distinct parameters of class {entity_class!r} are "
                f"both pinned to entity {binding}. Same-class bindings "
                f"are injective; pinned slots must reference distinct "
                f"entities."
            )
        pinned_by_class[duplicate_key] = param_key

    parameters: List[ParameterSpec] = []

    for param_key in compiler.slots.occurrence_order:
        entity_class, _label, binding = compiler.slots.info[param_key]
        required_components = sorted(compiler.slots.components[param_key])

        declared_components: Optional[FrozenSet[str]] = None
        if (
            capabilities is not None
            and capabilities.component_mode_by_class.get(entity_class)
            == _VM.CLOSED
        ):
            declared = capabilities.components_by_class.get(entity_class)
            if declared is not None:
                declared_components = frozenset(declared)

        static_class = _is_static_class(entity_class, capabilities)

        _validate_slot_against_catalog(
            entity_class=entity_class,
            binding=binding,
            required_components=required_components,
            catalog=catalog,
            entity_ids_closed=entity_ids_closed,
            components_closed=components_closed_for(entity_class),
            declared_components=declared_components,
            static_class=static_class,
        )

        slot_filters = (runtime_tag_filters or {}).get(
            (entity_class, _label), {}
        )

        parameters.append(
            ParameterSpec(
                param_key=param_key,
                is_wildcard=(binding == "*"),
                required_components=required_components,
                pinned_entity_id=(
                    None if binding == "*" else int(binding)
                ),
                component_tag_filters=dict(slot_filters),
            )
        )

    # RESET_AND_REWARM lifecycle: nothing is fed to a backend during
    # suspension, so every formula can retain suspended instances for
    # the default retention horizon (window_steps) and rewarm on
    # resumption. No neutral values exist anymore.
    neutral_values = None
    grace_override: Optional[int] = None

    if chosen_backend == "native":
        def make_backend(node=node, dt=dt):
            return NativeBackend(node, dt)
    else:
        rtamt_text_spec = root.text
        variable_names = sorted(compiler.predicate_fns)

        def make_backend(
            text=rtamt_text_spec, names=variable_names
        ):
            return RtamtBackend(text, names)

    return SpecTemplate(
        params=parameters,
        predicate_fn=compiler.predicate_fns[
            sorted(compiler.predicate_fns)[0]
        ],
        temporal_op=TemporalOp.NONE,
        window_steps=max(1, root.max_age + 1),
        dt=dt,
        name=formula_name,
        symmetric=symmetric,
        predicate_fns=compiler.predicate_fns,
        rtamt_spec=root.text,
        neutral_values=neutral_values,
        grace_steps_override=grace_override,
        outer_op=root.top_temporal,
        settles_early=root.simple_window,
        backend=chosen_backend,
        make_backend=make_backend,
    )


# ---------------------------------------------------------------------------
# Selector desugaring (stage 3, task 3)
#
# A formula carrying non-trivial selectors or "**" bindings is rewritten,
# BEFORE compile_formula ever sees it, into one or more formulas whose
# comparison text contains only concrete, trivial references. The rewrite is
# purely textual: each authored reference token is replaced by a pinned
# reference, an aggregator call, or a summed expression, so both backends and
# SymPy receive exactly the longhand a user could have typed by hand.
# ---------------------------------------------------------------------------

_KEYWORD_RE = re.compile(r"^\s*(any|all)\s*:\s*", re.IGNORECASE)

# ``Min``/``Max`` are accepted here because ``_SYMPY_LOCALS`` maps them to the
# same operators as ``min``/``max``; recognizing only the lowercase spellings
# would let ``Min(robot_*[...])`` slip through as a bare reference and desugar
# with different semantics than ``min(...)``.
_AGGREGATOR_CALL_RE = re.compile(r"\b(min|max|Min|Max|sum)\s*\(")


def _reference_from_match(match: "re.Match") -> ast.EntityReference:
    return ast.EntityReference(
        head=match.group("head"),
        explicit_label=match.group("explicit_label"),
        binding=match.group("binding"),
        component=match.group("component"),
        selector=parse_selector(match.group("component")),
    )


def _reference_token(reference: ast.EntityReference) -> str:
    """Reconstruct the exact authored source token for a reference."""
    label = (
        "{%s}" % reference.explicit_label
        if reference.explicit_label is not None
        else ""
    )
    return (
        f"{reference.head}{label}_{reference.binding}"
        f"[{reference.component}]"
    )


def _concrete_token(
    entity_class: str,
    entity_id: int,
    key: str,
    label: Optional[str] = None,
) -> str:
    """A pinned, trivial reference the syntax parser re-reads verbatim.

    When the authored reference carried an explicit ``{A}`` label, the pinned
    token keeps it (``robot{A}_1[key]``): distinct labels of one class stay
    distinct slots, so same-class injectivity in ``compile_formula`` still
    guards a self-comparison even if two labelled slots ever collapsed onto
    one entity.
    """
    head = entity_class if label is None else f"{entity_class}{{{label}}}"
    return f"{head}_{entity_id}[{key}]"


def _star_token(
    entity_class: str,
    key: str,
    label: str = "",
) -> str:
    """An unpinned trivial reference for a declared-schema wildcard slot.

    The slot stays ``*``: for a class with a declared CLOSED schema the
    member population is a runtime fact, so which entity satisfies the
    slot is per-observation binding eligibility, never a compile-time
    choice. The keys, by contrast, are schema facts and expand here.

    ``label`` is the RESOLVED slot label from ``resolve_reference_head``
    — never the raw ``explicit_label``, which is None for the legacy
    ``class_A_*`` spelling. Rebuilding with the resolved label in
    canonical brace form keeps legacy-labelled slots distinct; dropping
    it silently collapsed two labelled slots into one (found by the
    external review).
    """
    head = entity_class if not label else f"{entity_class}{{{label}}}"
    return f"{head}_*[{key}]"


def _references_in(text: str) -> Tuple[ast.EntityReference, ...]:
    return tuple(
        _reference_from_match(match)
        for match in PARAM_RE.finditer(text)
    )


def _is_non_trivial(reference: ast.EntityReference) -> bool:
    return (
        reference.binding == "**"
        or not reference.selector.is_trivial
    )


def _needs_desugaring(node: ast.FormulaNode) -> bool:
    """Whether any comparison in the tree carries a selector or ``**``."""
    if isinstance(node, ast.Comparison):
        return any(
            _is_non_trivial(reference)
            for reference in node.references
        )
    if isinstance(
        node,
        (
            ast.Not,
            ast.Always,
            ast.Eventually,
            ast.Previous,
            ast.Rise,
            ast.Fall,
        ),
    ):
        return _needs_desugaring(node.child)
    if isinstance(
        node,
        (ast.And, ast.Or, ast.Implies, ast.Iff, ast.Xor, ast.Since),
    ):
        return (
            _needs_desugaring(node.left)
            or _needs_desugaring(node.right)
        )
    return False


def _mirror_operator(operator: str) -> str:
    return {
        "<=": ">=",
        "<": ">",
        ">=": "<=",
        ">": "<",
        "==": "==",
        "!=": "!=",
    }[operator]


def _join_aggregate(func: str, pieces: Sequence[str]) -> str:
    if func == "sum":
        return "(" + " + ".join(pieces) + ")"
    return f"{func}(" + ", ".join(pieces) + ")"


def _split_top_level_commas(text: str) -> List[Tuple[int, int]]:
    """Spans of comma-separated arguments at paren/bracket depth zero.

    ``(`` and ``[`` open a region a comma cannot split; a selector's own
    ``{list}`` is always inside its ``[...]`` so its commas are never at
    top level here.
    """
    spans: List[Tuple[int, int]] = []
    depth = 0
    start = 0
    for index, character in enumerate(text):
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif character == "," and depth == 0:
            spans.append((start, index))
            start = index + 1
    spans.append((start, len(text)))
    return spans


def _find_aggregator(
    text: str,
    start: int,
    end: int,
) -> Optional[Tuple[int, str, int, int]]:
    """Return the INNERMOST ``min``/``max``/``sum`` call containing a span.

    Result is ``(call_start, func_name, open_paren_index, close_paren_index)``
    or None when the span sits under no aggregator. Binding to the innermost
    enclosing call is what keeps ``max(0, min(sel))`` from replicating the
    whole ``max`` argument list (the ``max(0, min(k1), 0, min(k2))`` bug):
    only ``min`` — the innermost — is expanded, leaving ``max(0, ...)`` intact.
    """
    candidates: List[Tuple[int, str, int, int]] = []
    for match in _AGGREGATOR_CALL_RE.finditer(text):
        open_index = match.end() - 1
        close_index = find_matching_parenthesis(text, open_index)
        if open_index < start and end <= close_index:
            candidates.append(
                (match.start(), match.group(1), open_index, close_index)
            )
    if not candidates:
        return None
    # Largest open-paren index == the closest enclosing call == innermost.
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[0]


def _reject_outside_keyword(side_text: str) -> None:
    """Reject a quantifier keyword written as a comparison-operand prefix.

    The canonical spelling puts ``any:``/``all:`` INSIDE the selector
    brackets (``robot_*[any: touch_*]``). A leading ``any:``/``all:`` on an
    operand (``any: robot_*[touch_*] >= c``) is not accepted; the author is
    pointed at the in-bracket form.
    """
    match = _KEYWORD_RE.match(side_text)
    if match is None:
        return

    keyword = match.group(1).lower()
    remainder = side_text[match.end():]

    references = _references_in(remainder)
    if references:
        reference = references[0]
        label = (
            "{%s}" % reference.explicit_label
            if reference.explicit_label is not None
            else ""
        )
        inside = (
            f"{reference.head}{label}_{reference.binding}"
            f"[{keyword}: {reference.component}]"
        )
        hint = f" Write it inside the brackets, e.g. {inside!r}."
    else:
        hint = ""

    raise SpecConfigError(
        f"Quantifier keyword {keyword + ':'!r} must be written inside the "
        f"selector brackets, not as a comparison-operand prefix." + hint
    )


_SlotKey = Tuple[str, str]                       # (entity_class, label)
_Assignment = Dict[_SlotKey, int]                # slot -> chosen entity id


def _try_resolve(
    selector,
    entity_id: int,
    catalog: ComponentCatalog,
) -> Tuple[str, ...]:
    """Resolve a selector against one entity, treating empty as ``()``.

    An entity that exposes none of the selector's keys is not an error at
    the per-entity eligibility stage (rule 3 / heterogeneous classes): it is
    simply skipped. A genuinely empty class-wide match is raised elsewhere.
    """
    try:
        return resolve_selector(
            selector,
            catalog.components_for_entity(entity_id),
            catalog.component_groups_for_entity(entity_id),
            catalog.component_tags_for_entity(entity_id),
        )
    except SpecConfigError:
        return ()


def _eligible_entities_for_slot(
    entity_class: str,
    label: str,
    slot_references: List[ast.EntityReference],
    catalog: ComponentCatalog,
    static_class: bool = False,
) -> List[int]:
    """Entities of a class on which EVERY reference of the slot resolves.

    Per-slot eligibility (Important 4): an entity whose resolution is empty
    for any reference of the slot is skipped, so a heterogeneous class (one
    member exposing none of the keys) still monitors its eligible members.
    Only when NO entity of the class matches is a ``SpecConfigError`` raised,
    naming the real keys (and groups, when a group filter is in play).

    On a static class (``static_class``) that refusal is the same fact a
    trivial exact key earns, so it is raised as ``UnboundStaticSlotError``:
    the population is fixed at reset, so a selector matching nothing now
    matches nothing later, whatever its spelling. The diagnostic text is
    unchanged.
    """
    entity_ids = sorted(catalog.entity_ids_for_class(entity_class))
    eligible = [
        entity_id
        for entity_id in entity_ids
        if all(
            _try_resolve(reference.selector, entity_id, catalog)
            for reference in slot_references
        )
    ]
    if eligible:
        return eligible

    error_type = UnboundStaticSlotError if static_class else SpecConfigError

    # No entity matched: surface an informative, key-naming error. A single
    # entity reproduces ``resolve_selector``'s native (group-aware) message.
    if len(entity_ids) == 1:
        entity_id = entity_ids[0]
        for reference in slot_references:
            try:
                resolve_selector(
                    reference.selector,
                    catalog.components_for_entity(entity_id),
                    catalog.component_groups_for_entity(entity_id),
                    catalog.component_tags_for_entity(entity_id),
                )
            except SpecConfigError as exc:
                raise error_type(str(exc)) from exc

    per_entity = {
        entity_id: sorted(catalog.components_for_entity(entity_id))
        for entity_id in entity_ids
    }
    available_groups = sorted(
        {
            group
            for entity_id in entity_ids
            for group in catalog.component_groups_for_entity(
                entity_id
            ).values()
        }
    )
    available_tags = sorted(
        {
            tag
            for entity_id in entity_ids
            for key_tags in catalog.component_tags_for_entity(
                entity_id
            ).values()
            for tag in key_tags
        }
    )
    raws = [reference.selector.raw for reference in slot_references]
    message = (
        f"No entity of class {entity_class!r} matches selector(s) {raws}. "
        f"Components per entity: {per_entity}."
    )
    if available_groups:
        message += f" Available groups: {available_groups}."
    if available_tags:
        message += f" Available tags: {available_tags}."
    raise error_type(message)


def _entity_assignments(
    fanning_keys: List[_SlotKey],
    slot_candidates: List[List[int]],
    pinned_digits: Mapping[str, Set[int]],
) -> List[_Assignment]:
    """Every injective assignment of entities to fanning wildcard slots.

    Enumeration is at FORMULA scope: an authored slot maps to one entity per
    output formula, and same-class slots (including any pinned-digit slot of
    that class) receive distinct entities — so labelled two-slot fan-out
    never yields a self-comparison, and no single authored slot is split
    across two entities by injectivity pressure. With no fanning slot the
    result is a single empty assignment.
    """
    assignments: List[_Assignment] = []
    for combination in itertools.product(*slot_candidates):
        used: Dict[str, Set[int]] = {}
        valid = True
        for (entity_class, _label), entity_id in zip(
            fanning_keys, combination
        ):
            seen = used.setdefault(entity_class, set())
            if (
                entity_id in seen
                or entity_id in pinned_digits.get(entity_class, set())
            ):
                valid = False
                break
            seen.add(entity_id)
        if valid:
            assignments.append(dict(zip(fanning_keys, combination)))
    return assignments


def _runtime_filtered_slots(
    star_slots: Mapping[_SlotKey, List[ast.EntityReference]],
    capabilities,
) -> Dict[_SlotKey, Dict[str, Tuple[str, ...]]]:
    """Slots whose ``@`` filters are retained for runtime eligibility.

    A wildcard slot routes to runtime filtering — instead of compile-time
    desugaring — when its class declares a CLOSED component schema in the
    adapter capabilities and any of its references carries an ``@``
    filter. Such classes are dynamic populations (contact pairs): their
    members are born mid-episode, so expanding the filter against the
    reset catalog could never reach the entities the spec is about, and
    would make the same spec behave differently depending on what
    happened to exist at reset.

    Routing is validated against the DECLARED vocabularies, mirroring how
    component references on these classes already validate against the
    declared schema: every reference of a routed slot must be an
    exact-key selector with a single ``@`` label and no excludes, the key
    must be inside the declared component schema, and the label inside
    the declared ``component_tags_by_class`` vocabulary. Anything else
    fails compilation loudly — there is no live entity whose real keys
    could catch the mistake later.

    Returns ``{slot: {component_key: (labels...)}}``; empty when nothing
    routes.
    """
    if capabilities is None:
        return {}
    from ..capabilities import VocabularyMode as _VM

    filters: Dict[_SlotKey, Dict[str, Tuple[str, ...]]] = {}
    for slot, references in star_slots.items():
        entity_class, _label = slot
        if (
            capabilities.component_mode_by_class.get(entity_class)
            != _VM.CLOSED
        ):
            continue
        declared = capabilities.components_by_class.get(entity_class)
        if declared is None:
            continue
        if not any(
            reference.selector.include.group is not None
            for reference in references
        ):
            continue

        declared_tags = frozenset(
            getattr(capabilities, "component_tags_by_class", {}).get(
                entity_class, frozenset()
            )
        )
        # ``@`` labels naming declared metadata GROUPS are not runtime
        # tags: leave the slot to declared expansion (runtime classes)
        # or ordinary catalog desugaring (static classes), where the
        # group resolves statically. Without this escape, declaring a
        # schema made an adapter's own published group vocabulary
        # unusable (found by the external review).
        include_labels = {
            reference.selector.include.group
            for reference in references
            if reference.selector.include.group is not None
        }
        declared_groups = set(
            _declared_group_map(entity_class, capabilities).values()
        )
        if (
            include_labels
            and include_labels.isdisjoint(declared_tags)
            and include_labels <= declared_groups
        ):
            continue
        unknown = include_labels - declared_tags - declared_groups
        if unknown and (declared_groups or declared_tags):
            raise SpecConfigError(
                f"Filter label(s) {sorted(unknown)} on class "
                f"{entity_class!r} match neither the declared metadata "
                f"groups {sorted(declared_groups)} nor the declared "
                f"tags {sorted(declared_tags)}."
            )
        slot_filters: Dict[str, Set[str]] = {}
        for reference in references:
            if not _is_non_trivial(reference):
                continue
            token = _reference_token(reference)
            selector = reference.selector
            include = selector.include
            label = include.group
            unsupported = None
            if reference.binding == "**":
                unsupported = "a '**' binding"
            elif selector.quantifier is not None:
                unsupported = "a quantifier keyword"
            elif selector.exclude:
                unsupported = "an exclude term"
            elif label is None:
                unsupported = "a non-trivial selector without a filter"
            elif "*" in include.atom or include.atom.startswith("{"):
                unsupported = "a glob or list atom"
            if unsupported is not None or label is None:
                raise SpecConfigError(
                    f"Selector {token!r}: class {entity_class!r} has a "
                    f"declared component schema and a runtime-filtered "
                    f"population, which supports only exact-key "
                    f"selectors with a single '@' label; found "
                    f"{unsupported}."
                )
            if include.atom not in declared:
                raise SpecConfigError(
                    f"Component {include.atom!r} in {token!r} is outside "
                    f"the declared component schema of class "
                    f"{entity_class!r}. Declared components: "
                    f"{sorted(declared)}."
                )
            if label not in declared_tags:
                raise SpecConfigError(
                    f"Tag filter {label!r} in {token!r} is "
                    f"outside the declared tag vocabulary of class "
                    f"{entity_class!r}. Declared tags: "
                    f"{sorted(declared_tags)}."
                )
            slot_filters.setdefault(include.atom, set()).add(label)
        filters[slot] = {
            key: tuple(sorted(labels))
            for key, labels in slot_filters.items()
        }
    return filters


def _is_declared_runtime_class(entity_class: str, capabilities) -> bool:
    """True for a declared-CLOSED-schema class with a RUNTIME population.

    The two capability dimensions are independent: a CLOSED component
    vocabulary says the keys are schema facts, while membership in
    ``runtime_populated_classes`` says the entity population may gain
    members after reset. Only the combination routes selectors to
    declared (unpinned) expansion — a static declared class (CARLA's
    regions and traffic lights) keeps ordinary catalog desugaring,
    including ``**`` and aggregator support over its reset population.
    """
    if capabilities is None:
        return False
    from ..capabilities import VocabularyMode as _VM

    return (
        entity_class in getattr(
            capabilities, "runtime_populated_classes", frozenset()
        )
        and capabilities.component_mode_by_class.get(entity_class)
        == _VM.CLOSED
        and bool(capabilities.components_by_class.get(entity_class))
    )


def _is_static_class(entity_class: str, capabilities) -> bool:
    """True when the class's whole population is fixed at reset.

    The class vocabulary is CLOSED and the class is not runtime-populated,
    so no entity of it can appear after reset and a slot with no candidate
    now has none later. Undeclared capabilities are not a static claim.
    """
    if capabilities is None:
        return False
    from ..capabilities import VocabularyMode as _VM

    return (
        capabilities.class_mode == _VM.CLOSED
        and entity_class not in (capabilities.runtime_populated_classes or ())
    )


def _declared_group_map(entity_class: str, capabilities) -> Dict[str, str]:
    """The class's declared key->group mapping, empty when undeclared."""
    if capabilities is None:
        return {}
    return dict(
        getattr(capabilities, "component_groups_by_class", {}).get(
            entity_class, {}
        )
    )


def _declared_expansion_slots(
    star_slots: Mapping[_SlotKey, List[ast.EntityReference]],
    capabilities,
    runtime_slots: FrozenSet[_SlotKey],
) -> Dict[_SlotKey, Tuple[Tuple[str, ...], Dict[str, str]]]:
    """Wildcard slots whose selectors resolve against the DECLARED schema.

    A slot routes here when its class is declared-CLOSED **and**
    runtime-populated, it was not claimed by the runtime-tag route
    above, and at least one reference is a non-trivial selector. Key
    expansion then happens once, class-level, against the declaration
    (components and, when declared, the key->group map, so published
    metadata groups stay usable in ``@`` filters), and the rewritten
    references stay unpinned: the population of such a class is a
    runtime fact — members may be born mid-episode (CARLA actors,
    contact pairs) — so desugaring per entity against the reset catalog
    would tie the spec to whatever happened to exist at reset and never
    reach later arrivals. Static declared classes never route here and
    keep ordinary catalog desugaring.

    Returns ``{slot: (declared_keys, declared_key_to_group)}``; empty
    when nothing routes.
    """
    if capabilities is None:
        return {}

    declared_slots: Dict[
        _SlotKey, Tuple[Tuple[str, ...], Dict[str, str]]
    ] = {}
    for slot, references in star_slots.items():
        if slot in runtime_slots:
            continue
        entity_class, _label = slot
        if not _is_declared_runtime_class(entity_class, capabilities):
            continue
        if not any(_is_non_trivial(reference) for reference in references):
            continue
        declared = capabilities.components_by_class.get(entity_class)
        groups = _declared_group_map(entity_class, capabilities)
        group_labels = set(groups.values())
        # ``@`` labels here must name declared GROUPS — a declared tag
        # would have routed the slot to the runtime-tag path above, and
        # anything else has no static meaning to resolve against.
        for reference in references:
            selector = reference.selector
            terms = [selector.include] + list(selector.exclude)
            for term in terms:
                if term.group is not None and term.group not in group_labels:
                    raise SpecConfigError(
                        f"Filter {term.group!r} in "
                        f"{_reference_token(reference)!r}: class "
                        f"{entity_class!r} has a declared component "
                        f"schema and a runtime population, so '@' "
                        f"labels must name a declared metadata group. "
                        f"Declared groups: {sorted(group_labels)}."
                    )
        declared_slots[slot] = (tuple(sorted(declared)), groups)
    return declared_slots


def _resolve_ref_entities(
    reference: ast.EntityReference,
    entity_class: str,
    label: str,
    assignment: _Assignment,
    slot_multiple: Mapping[_SlotKey, bool],
    catalog: ComponentCatalog,
) -> Tuple[List[Tuple[int, Tuple[str, ...]]], bool]:
    """Targeted ``(entity_id, matched_keys)`` pairs for one reference.

    ``*`` reads its entity from the formula-scope ``assignment``; a digit
    binding targets exactly that entity; ``**`` collapses every eligible
    entity of the class. The boolean records whether the enclosing ``*`` slot
    fanned over more than one entity (drives the ``[@id]`` name suffix).
    """
    if reference.binding == "**":
        resolved: List[Tuple[int, Tuple[str, ...]]] = []
        for entity_id in sorted(catalog.entity_ids_for_class(entity_class)):
            keys = _try_resolve(reference.selector, entity_id, catalog)
            if keys:
                resolved.append((entity_id, keys))
        if not resolved:
            raise SpecConfigError(
                f"Binding '**' in {_reference_token(reference)!r} matched no "
                f"component on any discovered entity of class "
                f"{entity_class!r}."
            )
        return resolved, False

    if reference.binding == "*":
        entity_id = assignment[(entity_class, label)]
        keys = resolve_selector(
            reference.selector,
            catalog.components_for_entity(entity_id),
            catalog.component_groups_for_entity(entity_id),
            catalog.component_tags_for_entity(entity_id),
        )
        return [(entity_id, keys)], slot_multiple.get(
            (entity_class, label), False
        )

    entity_id = int(reference.binding)
    if not catalog.has_entity(entity_id):
        raise SpecConfigError(
            f"Pinned entity ID {entity_id} is not present in the "
            f"discovered component catalog."
        )
    keys = resolve_selector(
        reference.selector,
        catalog.components_for_entity(entity_id),
        catalog.component_groups_for_entity(entity_id),
        catalog.component_tags_for_entity(entity_id),
    )
    return [(entity_id, keys)], False


# One textual rewrite of one region, tagged with its comparison + side so a
# formula-scope plan can carry rewrites across every comparison at once.
_Replacement = Tuple[int, str, int, int, str]    # (comp, side, start, end, text)
_Choice = Tuple[List[_Replacement], str]         # (replacements, name_suffix)


def _expand_keyword_occurrence(
    comp_index: int,
    side: str,
    text: str,
    reference: ast.EntityReference,
    operator: str,
    entity_class: str,
    label: str,
    assignment: _Assignment,
    slot_multiple: Mapping[_SlotKey, bool],
    catalog: ComponentCatalog,
) -> _Replacement:
    """Rewrite a quantified selector (the whole side) into min/max(...).

    Returns one replacement plus its ``[@id]`` suffix folded in as the last
    tuple element via ``_Replacement``'s text — the caller reads the suffix
    from :func:`_expand_side`'s accumulation, so here we only build the call.
    """
    keyword = reference.selector.quantifier.lower()
    effective = operator if side == "left" else _mirror_operator(operator)
    if effective in ("==", "!="):
        raise SpecConfigError(
            f"Quantifier keyword {keyword!r} has no direction for "
            f"operator {operator!r}; use min(...)/max(...) explicitly."
        )
    if keyword == "all":
        func = "max" if effective in ("<=", "<") else "min"
    else:
        func = "min" if effective in ("<=", "<") else "max"

    per_entity, multiple = _resolve_ref_entities(
        reference, entity_class, label, assignment, slot_multiple, catalog
    )
    pieces = [
        _concrete_token(entity_class, entity_id, key, reference.explicit_label)
        for entity_id, keys in per_entity
        for key in keys
    ]
    joined = _join_aggregate(func, pieces)

    suffix = ""
    if reference.binding == "*" and multiple:
        suffix = f" [@{per_entity[0][0]}]"
    return (comp_index, side, 0, len(text), joined), suffix


def _expand_aggregator_call(
    comp_index: int,
    side: str,
    text: str,
    aggregator: Tuple[int, str, int, int],
    assignment: _Assignment,
    slot_multiple: Mapping[_SlotKey, bool],
    catalog: ComponentCatalog,
    class_vocabulary: Optional[FrozenSet[str]],
) -> Tuple[_Replacement, str]:
    """Replicate every selector-bearing argument of ONE aggregator call.

    Only the comma-delimited arguments that hold a selector are expanded, each
    into its per-key pieces; sibling arguments are copied verbatim. This keeps
    ``max(0, min(sel))`` and ``min(5, sel)`` correct (siblings untouched) and
    lets one call carry several distinct selector arguments
    (``max(sel_a, sel_b)``). An argument with two DISTINCT selector tokens is
    the genuinely undecidable case and is rejected.
    """
    call_start, func, open_index, close_index = aggregator
    content = text[open_index + 1:close_index]

    new_args: List[str] = []
    suffix = ""
    for arg_start, arg_end in _split_top_level_commas(content):
        argument = content[arg_start:arg_end]
        non_trivial = [
            reference
            for reference in _references_in(argument)
            if _is_non_trivial(reference)
        ]
        if not non_trivial:
            new_args.append(argument.strip())
            continue

        distinct_tokens = {
            _reference_token(reference) for reference in non_trivial
        }
        if len(distinct_tokens) > 1:
            raise SpecConfigError(
                f"Aggregator argument {argument.strip()!r} contains more than "
                f"one distinct selector reference; pairing is undecidable. "
                f"Write one aggregator per selector."
            )

        reference = non_trivial[0]
        token = _reference_token(reference)
        entity_class, label = resolve_reference_head(
            reference, class_vocabulary, strict=True
        )
        per_entity, multiple = _resolve_ref_entities(
            reference, entity_class, label, assignment, slot_multiple, catalog
        )
        if reference.binding == "**":
            entity_ids = sorted(
                {entity_id for entity_id, _keys in per_entity}
            )
            if len(entity_ids) >= 2:
                current = aggregator
                while True:
                    enclosing = _find_aggregator(
                        text,
                        start=current[0],
                        end=current[3] + 1,
                    )
                    if enclosing is None:
                        break
                    if enclosing[1].lower() != current[1].lower():
                        inner = current[1].lower()
                        outer = enclosing[1].lower()
                        raise SpecConfigError(
                            f"Binding '**' expanded to {len(entity_ids)} "
                            f"{entity_class} entities {entity_ids} inside "
                            f"{inner} under {outer}. This expression would "
                            f"flatten per-entity groups into one reduction "
                            f"and change its meaning. The grouped scene-level "
                            f"form is not expressible in this release. Write "
                            f"one explicit formula per entity."
                        )
                    current = enclosing
        for entity_id, keys in per_entity:
            for key in keys:
                concrete = _concrete_token(
                    entity_class, entity_id, key, reference.explicit_label
                )
                new_args.append(argument.replace(token, concrete).strip())
            if reference.binding == "*" and multiple:
                suffix += f" [@{entity_id}]"

    joined = _join_aggregate(func, new_args)
    return (comp_index, side, call_start, close_index + 1, joined), suffix


def _expand_side(
    comp_index: int,
    side: str,
    text: str,
    operator: str,
    assignment: _Assignment,
    slot_multiple: Mapping[_SlotKey, bool],
    catalog: ComponentCatalog,
    class_vocabulary: Optional[FrozenSet[str]],
    runtime_slots: FrozenSet[_SlotKey] = frozenset(),
    declared_expansions: Mapping[
        _SlotKey, Tuple[Tuple[str, ...], Mapping[str, str]]
    ] = {},
) -> Tuple[List[_Replacement], str, List[List[_Choice]]]:
    """Plan one comparison side under a fixed entity assignment.

    ``text`` is already pre-pinned (trivial references of a fanning slot are
    concrete). Returns the deterministic base replacements (keyword + each
    aggregator call), their combined suffix, and one fan-out DIMENSION per
    distinct bare selector token — bare references being the only construct
    that multiplies formulas once the entity is fixed. Repeated identical
    bare tokens share ONE dimension, so all occurrences rewrite to the same
    key (no cartesian blow-up into mixed keys).

    References of a ``runtime_slots`` slot do not expand at all: their
    validated ``@`` filter moves to binding eligibility, so the token is
    rewritten to its trivial exact-key form in place (no suffix, no
    fan-out dimension).

    References of a ``declared_expansions`` slot resolve their keys
    against the class's declared schema and rewrite to UNPINNED trivial
    references: which entity satisfies the slot stays a per-observation
    eligibility question, so members born mid-episode bind the moment
    they appear.
    """
    base: List[_Replacement] = []
    base_suffix = ""
    bare_groups: Dict[
        str, Tuple[ast.EntityReference, List[Tuple[int, int]]]
    ] = {}
    declared_bare: Dict[
        str,
        Tuple[ast.EntityReference, Tuple[str, ...], List[Tuple[int, int]]],
    ] = {}
    handled_calls: Set[int] = set()

    for match in PARAM_RE.finditer(text):
        reference = _reference_from_match(match)
        if not _is_non_trivial(reference):
            continue
        start, end = match.start(), match.end()
        entity_class, label = resolve_reference_head(
            reference, class_vocabulary, strict=True
        )

        declared_entry = declared_expansions.get((entity_class, label))
        if declared_entry is not None:
            declared_keys, declared_groups = declared_entry
            if (
                reference.binding == "**"
                or _find_aggregator(text, start, end) is not None
            ):
                raise SpecConfigError(
                    f"Selector {_reference_token(reference)!r}: class "
                    f"{entity_class!r} has a declared component schema "
                    f"and a runtime population, so aggregation across "
                    f"entities ('**', or a min/max/sum call) is not "
                    f"known at compilation. Use an any:/all: keyword "
                    f"over one bound entity instead."
                )
            keys = resolve_selector(
                reference.selector,
                frozenset(declared_keys),
                declared_groups,
                {},
            )
            if reference.selector.quantifier is not None:
                keyword = reference.selector.quantifier.lower()
                effective = (
                    operator if side == "left" else _mirror_operator(operator)
                )
                if effective in ("==", "!="):
                    raise SpecConfigError(
                        f"Quantifier keyword {keyword!r} has no direction "
                        f"for operator {operator!r}; use min(...)/max(...) "
                        f"explicitly."
                    )
                if keyword == "all":
                    func = "max" if effective in ("<=", "<") else "min"
                else:
                    func = "min" if effective in ("<=", "<") else "max"
                pieces = [
                    _star_token(entity_class, key, label) for key in keys
                ]
                base.append(
                    (comp_index, side, 0, len(text),
                     _join_aggregate(func, pieces))
                )
                continue
            token = _reference_token(reference)
            declared_bare.setdefault(token, (reference, keys, []))[2].append(
                (start, end)
            )
            continue

        if (entity_class, label) in runtime_slots:
            if _find_aggregator(text, start, end) is not None:
                raise SpecConfigError(
                    f"Selector {_reference_token(reference)!r} carries a "
                    f"runtime tag filter and cannot sit inside a "
                    f"min/max/sum call: the eligible population is not "
                    f"known at compilation. Compare the component "
                    f"directly."
                )
            explicit = (
                "{%s}" % reference.explicit_label
                if reference.explicit_label is not None
                else ""
            )
            trivial = (
                f"{reference.head}{explicit}_{reference.binding}"
                f"[{reference.selector.include.atom}]"
            )
            base.append((comp_index, side, start, end, trivial))
            continue

        if reference.selector.quantifier is not None:
            replacement, suffix = _expand_keyword_occurrence(
                comp_index, side, text, reference, operator,
                entity_class, label, assignment, slot_multiple, catalog,
            )
            base.append(replacement)
            base_suffix += suffix
            continue

        aggregator = _find_aggregator(text, start, end)
        if aggregator is not None:
            if aggregator[3] in handled_calls:
                continue  # this call already expanded (all its args at once)
            handled_calls.add(aggregator[3])
            replacement, suffix = _expand_aggregator_call(
                comp_index, side, text, aggregator, assignment,
                slot_multiple, catalog, class_vocabulary,
            )
            base.append(replacement)
            base_suffix += suffix
            continue

        if reference.binding == "**":
            raise SpecConfigError(
                f"Binding '**' in {_reference_token(reference)!r} is only "
                f"valid inside an aggregator (min/max/sum) or under an "
                f"any:/all: keyword; a bare '**' enumeration means exactly "
                f"what '*' already does."
            )

        token = _reference_token(reference)
        bare_groups.setdefault(token, (reference, []))[1].append((start, end))

    dimensions: List[List[_Choice]] = []
    for reference, keys, spans in declared_bare.values():
        entity_class, slot_label = resolve_reference_head(
            reference, class_vocabulary, strict=True
        )
        choices: List[_Choice] = []
        for key in keys:
            unpinned = _star_token(entity_class, key, slot_label)
            replacements = [
                (comp_index, side, start, end, unpinned)
                for start, end in spans
            ]
            choices.append((replacements, f" [{key}]"))
        dimensions.append(choices)

    for reference, spans in bare_groups.values():
        entity_class, label = resolve_reference_head(
            reference, class_vocabulary, strict=True
        )
        per_entity, _multiple = _resolve_ref_entities(
            reference, entity_class, label, assignment, slot_multiple, catalog
        )
        entity_id, keys = per_entity[0]
        choices = []
        for key in keys:
            concrete = _concrete_token(
                entity_class, entity_id, key, reference.explicit_label
            )
            replacements = [
                (comp_index, side, start, end, concrete)
                for start, end in spans
            ]
            choices.append((replacements, f" [{key}@{entity_id}]"))
        dimensions.append(choices)

    return base, base_suffix, dimensions


def _apply_replacements(
    text: str,
    replacements: List[Tuple[int, int, str]],
) -> str:
    ordered = sorted(replacements, key=lambda item: item[0])
    previous_end = -1
    for start, end, _new_text in ordered:
        if start < previous_end:
            raise SpecConfigError(
                f"Overlapping selector rewrites in {text!r}; a selector "
                f"reference must not sit inside another selector's aggregator "
                f"argument. Split the comparison into separate clauses."
            )
        previous_end = end

    for start, end, new_text in reversed(ordered):
        text = text[:start] + new_text + text[end:]
    return text


def _collect_comparisons(node: ast.FormulaNode) -> List[ast.Comparison]:
    """Every ``Comparison`` leaf, left-before-right, depth-first."""
    out: List[ast.Comparison] = []

    def visit(current: ast.FormulaNode) -> None:
        if isinstance(current, ast.Comparison):
            out.append(current)
        elif isinstance(
            current,
            (
                ast.Not,
                ast.Always,
                ast.Eventually,
                ast.Previous,
                ast.Rise,
                ast.Fall,
            ),
        ):
            visit(current.child)
        elif isinstance(
            current,
            (ast.And, ast.Or, ast.Implies, ast.Iff, ast.Xor),
        ):
            visit(current.left)
            visit(current.right)
        elif isinstance(current, ast.Since):
            visit(current.left)
            visit(current.right)
        else:
            raise TypeError(
                f"Unknown AST node {type(current).__name__}"
            )

    visit(node)
    return out


def _rebuild_tree(
    node: ast.FormulaNode,
    rewritten: "Iterator[ast.Comparison]",
) -> ast.FormulaNode:
    """Rebuild the tree, drawing rewritten comparisons in collection order."""
    if isinstance(node, ast.Comparison):
        return next(rewritten)
    if isinstance(node, ast.Not):
        return ast.Not(child=_rebuild_tree(node.child, rewritten))
    if isinstance(node, (ast.Always, ast.Eventually)):
        return type(node)(
            bounds=node.bounds,
            child=_rebuild_tree(node.child, rewritten),
        )
    if isinstance(node, (ast.Previous, ast.Rise, ast.Fall)):
        return type(node)(child=_rebuild_tree(node.child, rewritten))
    if isinstance(
        node,
        (ast.And, ast.Or, ast.Implies, ast.Iff, ast.Xor),
    ):
        left = _rebuild_tree(node.left, rewritten)
        right = _rebuild_tree(node.right, rewritten)
        return type(node)(left=left, right=right)
    if isinstance(node, ast.Since):
        left = _rebuild_tree(node.left, rewritten)
        right = _rebuild_tree(node.right, rewritten)
        return ast.Since(left=left, bounds=node.bounds, right=right)
    raise TypeError(f"Unknown AST node {type(node).__name__}")


def _prepin(
    text: str,
    assignment: _Assignment,
    fanning_slots: Set[_SlotKey],
    class_vocabulary: Optional[FrozenSet[str]],
) -> str:
    """Pin every TRIVIAL reference of a fanning slot to the assigned entity.

    Non-trivial references stay verbatim (they are expanded later against the
    now-known entity), and a ``*`` reference whose slot does not fan is left
    wildcard — the trivial-regression path is untouched. Pinning trivial
    siblings here (rather than as separate span rewrites) is what lets a
    trivial reference sit inside an aggregator argument without tripping the
    overlap guard.
    """
    replacements: List[Tuple[int, int, str]] = []
    for match in PARAM_RE.finditer(text):
        reference = _reference_from_match(match)
        if reference.binding != "*" or not reference.selector.is_trivial:
            continue
        entity_class, label = resolve_reference_head(
            reference, class_vocabulary, strict=True
        )
        if (entity_class, label) not in fanning_slots:
            continue
        entity_id = assignment[(entity_class, label)]
        token = _concrete_token(
            entity_class, entity_id, reference.component,
            reference.explicit_label,
        )
        replacements.append((match.start(), match.end(), token))
    return _apply_replacements(text, replacements)


def _expand_formula(
    node: ast.FormulaNode,
    catalog: ComponentCatalog,
    class_vocabulary: Optional[FrozenSet[str]],
    capabilities=None,
) -> Tuple[
    List[Tuple[ast.FormulaNode, str]],
    Dict[_SlotKey, Dict[str, Tuple[str, ...]]],
]:
    """Desugar one authored formula into concrete formulas (formula scope).

    Entity fan-out is enumerated ONCE for the whole formula: an authored
    wildcard slot picks one entity per output formula, injectively across
    same-class slots, and that choice governs every reference to the slot in
    every comparison. Only then are keys expanded per reference. This is what
    keeps a slot reused across two comparisons on ONE entity, prevents
    labelled same-class self-comparisons, and honours per-slot eligibility.
    """
    comparisons = _collect_comparisons(node)

    # Whole-formula validation (once).
    star_slots: Dict[_SlotKey, List[ast.EntityReference]] = {}
    pinned_digits: Dict[str, Set[int]] = {}
    for comparison in comparisons:
        for side_text in (comparison.left, comparison.right):
            _reject_outside_keyword(side_text)
            for reference in _references_in(side_text):
                if (
                    reference.selector.quantifier is not None
                    and side_text.strip() != _reference_token(reference)
                ):
                    raise SpecConfigError(
                        "a quantified selector must be the entire comparison "
                        "side; wrap arithmetic around min(...)/max(...) "
                        "instead"
                    )
                entity_class, label = resolve_reference_head(
                    reference, class_vocabulary, strict=True
                )
                if reference.binding == "*":
                    star_slots.setdefault(
                        (entity_class, label), []
                    ).append(reference)
                elif reference.binding == "**":
                    # '**' never enters slot routing, so guard it here:
                    # on a declared-CLOSED-schema class it would expand
                    # against whatever population existed at reset —
                    # silent pinning, the bug class the declared route
                    # exists to eliminate.
                    if _is_declared_runtime_class(entity_class, capabilities):
                        raise SpecConfigError(
                            f"Binding '**' in "
                            f"{_reference_token(reference)!r}: class "
                            f"{entity_class!r} has a declared component "
                            f"schema and a runtime population, so "
                            f"aggregation across entities is not known "
                            f"at compilation. Use an any:/all: keyword "
                            f"over one bound entity instead."
                        )
                else:
                    pinned_digits.setdefault(entity_class, set()).add(
                        int(reference.binding)
                    )

    # Slots on declared-schema dynamic classes keep their validated
    # ``@`` filters for runtime eligibility and never fan here.
    runtime_filters = _runtime_filtered_slots(star_slots, capabilities)
    runtime_slots = frozenset(runtime_filters)

    # Unfiltered non-trivial selectors on declared-schema classes expand
    # class-level against the declaration, unpinned, and never fan
    # either — their population is a runtime fact.
    declared_expansions = _declared_expansion_slots(
        star_slots, capabilities, runtime_slots
    )

    # A slot fans iff any of its references is a non-trivial selector.
    fanning_keys = [
        slot
        for slot, references in star_slots.items()
        if slot not in runtime_slots
        and slot not in declared_expansions
        and any(_is_non_trivial(reference) for reference in references)
    ]
    slot_candidates = [
        _eligible_entities_for_slot(
            entity_class, label, star_slots[(entity_class, label)], catalog,
            _is_static_class(entity_class, capabilities),
        )
        for (entity_class, label) in fanning_keys
    ]
    slot_multiple = {
        slot: len(candidates) > 1
        for slot, candidates in zip(fanning_keys, slot_candidates)
    }
    fanning_set = set(fanning_keys)

    assignments = _entity_assignments(
        fanning_keys, slot_candidates, pinned_digits
    )
    if fanning_keys and not assignments:
        # Never vacuous, never silent: injectivity across same-class slots
        # left no valid entity assignment, so the spec would monitor nothing.
        details: List[str] = []
        for entity_class in sorted(
            {entity_class for entity_class, _label in fanning_keys}
        ):
            slot_count = sum(
                1 for cls, _label in fanning_keys if cls == entity_class
            ) + len(pinned_digits.get(entity_class, set()))
            eligible = {
                entity_id
                for slot, candidates in zip(fanning_keys, slot_candidates)
                if slot[0] == entity_class
                for entity_id in candidates
            } | pinned_digits.get(entity_class, set())
            plural_e = "entity" if len(eligible) == 1 else "entities"
            plural_s = "slot" if slot_count == 1 else "slots"
            details.append(
                f"class {entity_class!r} has {len(eligible)} eligible "
                f"{plural_e} for {slot_count} injective {plural_s}"
            )
        raise SpecConfigError(
            f"expands to zero monitors: {'; '.join(details)}"
        )

    results: List[Tuple[ast.FormulaNode, str]] = []
    for assignment in assignments:
        pinned_text: Dict[Tuple[int, str], str] = {}
        base: List[_Replacement] = []
        base_suffix = ""
        dimensions: List[List[_Choice]] = []

        for comp_index, comparison in enumerate(comparisons):
            for side in ("left", "right"):
                original = (
                    comparison.left if side == "left" else comparison.right
                )
                text = _prepin(
                    original, assignment, fanning_set, class_vocabulary
                )
                pinned_text[(comp_index, side)] = text
                side_base, side_suffix, side_dims = _expand_side(
                    comp_index, side, text, comparison.operator,
                    assignment, slot_multiple, catalog, class_vocabulary,
                    runtime_slots, declared_expansions,
                )
                base.extend(side_base)
                base_suffix += side_suffix
                dimensions.extend(side_dims)

        for combination in itertools.product(*dimensions):
            replacements = list(base)
            suffix = base_suffix
            for choice_replacements, choice_suffix in combination:
                replacements.extend(choice_replacements)
                suffix += choice_suffix

            rewritten_comparisons: List[ast.Comparison] = []
            for comp_index, comparison in enumerate(comparisons):
                left_text = _apply_replacements(
                    pinned_text[(comp_index, "left")],
                    [
                        (start, end, text)
                        for ci, side, start, end, text in replacements
                        if ci == comp_index and side == "left"
                    ],
                )
                right_text = _apply_replacements(
                    pinned_text[(comp_index, "right")],
                    [
                        (start, end, text)
                        for ci, side, start, end, text in replacements
                        if ci == comp_index and side == "right"
                    ],
                )
                rewritten_comparisons.append(
                    ast.Comparison(
                        left=left_text,
                        operator=comparison.operator,
                        right=right_text,
                        references=(
                            _references_in(left_text)
                            + _references_in(right_text)
                        ),
                    )
                )

            tree = _rebuild_tree(node, iter(rewritten_comparisons))
            results.append((tree, suffix))

    return results, runtime_filters


def compile_formulas(
    node: ast.FormulaNode,
    *,
    formula_name: str,
    dt: float,
    constants: Mapping[str, float],
    symmetric: bool = False,
    catalog: Optional[ComponentCatalog] = None,
    class_vocabulary: Optional[FrozenSet[str]] = None,
    backend: str = "auto",
    capabilities=None,
) -> List[SpecTemplate]:
    """Compile one authored formula into one or more concrete formulas.

    Formulas with no selectors compile straight through
    ``compile_formula`` (a one-element list). A formula carrying any
    non-trivial selector or ``**`` binding is desugared against the
    discovered ``catalog`` — which is therefore required — into a list of
    longhand formulas, each with a name suffix recording its expansion.
    """
    if not _needs_desugaring(node):
        return [
            compile_formula(
                node,
                formula_name=formula_name,
                dt=dt,
                constants=constants,
                symmetric=symmetric,
                catalog=catalog,
                class_vocabulary=class_vocabulary,
                backend=backend,
                capabilities=capabilities,
            )
        ]

    if catalog is None:
        raise SpecConfigError(
            "Formula contains component selectors or a '**' binding, which "
            "require a discovered component catalog. Compile per episode "
            "with catalog=... (this is why selector specs are compiled per "
            "sim, not at parse time)."
        )

    expansions, runtime_filters = _expand_formula(
        node, catalog, class_vocabulary, capabilities
    )

    compiled: List[SpecTemplate] = []
    for rewritten, suffix in expansions:
        formula = compile_formula(
            rewritten,
            formula_name=formula_name + suffix,
            dt=dt,
            constants=constants,
            symmetric=symmetric,
            catalog=catalog,
            class_vocabulary=class_vocabulary,
            backend=backend,
            capabilities=capabilities,
            runtime_tag_filters=runtime_filters,
        )
        formula.desugared_text = render.formula_text(rewritten)
        compiled.append(formula)
    return compiled
