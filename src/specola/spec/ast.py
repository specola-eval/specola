"""
ast.py — immutable syntax tree for past-time STL spec templates.

Nodes carry authored information only:

- temporal bounds keep their authored value and unit (``Bound``); the
  conversion to steps needs the adapter's dt and happens at semantic
  compilation;
- entity references keep their authored head (``obj``, ``object_A``),
  optional explicit label, binding token (``*`` or a digit string), and
  component name; resolution against a class vocabulary is semantic;
- comparison operands remain expression text; arithmetic is compiled by
  the semantic layer.

Nothing in this module knows about simulators, catalogs, dt, or RTAMT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class Bound:
    """One authored temporal bound, e.g. ``5`` (steps) or ``2.5s``."""

    value: float
    in_seconds: bool


@dataclass(frozen=True)
class Interval:
    lower: Bound
    upper: Bound


@dataclass(frozen=True)
class SelectorTerm:
    """One atom of a component selector, e.g. ``qacc_* @ some group``.

    atom:   exact name, glob (contains ``*``), or list-literal text
            (``{a, b}``), exactly as authored
    names:  expanded ``{list}`` members; single-element tuple holding
            ``atom`` itself for an exact name or a glob
    is_glob: whether ``atom`` contains a ``*`` wildcard
    group:  ``@ group`` filter text, stripped; None when absent
    """

    atom: str
    names: Tuple[str, ...]
    is_glob: bool
    group: Optional[str]


@dataclass(frozen=True)
class ComponentSelector:
    """Parsed contents of a component bracket, e.g. ``[qacc_* \\ {qacc_a}]``.

    Pure syntax: resolution against a discovered component catalog is a
    Task-3 concern. ``parse_selector`` (``spec/selectors.py``) is the only
    producer of this type.

    quantifier: ``"any"`` | ``"all"`` | None
    include:    first (and, for a trivial selector, only) term
    exclude:    ``"\\"``-separated remainder, each subtracted from include
    raw:        the complete, unparsed selector text
    """

    quantifier: Optional[str]
    include: SelectorTerm
    exclude: Tuple[SelectorTerm, ...]
    raw: str

    @property
    def is_trivial(self) -> bool:
        """Whether this selector is exactly an old-style exact name.

        True only for a bare name with no quantifier, glob, list, group,
        or exclude clause — i.e. the pre-selector reference forms that the
        old semantic-compile path already knows how to handle.
        """
        return (
            self.quantifier is None
            and not self.exclude
            and not self.include.is_glob
            and self.include.group is None
            and len(self.include.names) == 1
            and self.include.names[0] == self.include.atom
        )


@dataclass(frozen=True)
class EntityReference:
    """One authored parameter reference such as ``object{A}_*[hinge]``.

    head:            authored class-or-legacy-label prefix (``object``,
                     ``object_A``)
    explicit_label:  label from the unambiguous ``class{A}`` form, or None
    binding:         ``"*"`` for wildcards, ``"**"`` for a fan-out group
                     binding, or a digit string for pinned IDs
    component:       raw text of the requested component bracket, exactly
                     as authored (e.g. ``hinge_angle`` or a selector such
                     as ``qacc_* \\ {qacc_rootx}``)
    selector:        ``component`` parsed into a ``ComponentSelector``
    """

    head: str
    explicit_label: Optional[str]
    binding: str
    component: str
    selector: ComponentSelector


@dataclass(frozen=True)
class Comparison:
    """Atomic predicate ``left <op> right`` over expression text."""

    left: str
    operator: str
    right: str
    references: Tuple[EntityReference, ...]


@dataclass(frozen=True)
class Not:
    child: "FormulaNode"


@dataclass(frozen=True)
class And:
    left: "FormulaNode"
    right: "FormulaNode"


@dataclass(frozen=True)
class Or:
    left: "FormulaNode"
    right: "FormulaNode"


@dataclass(frozen=True)
class Implies:
    left: "FormulaNode"
    right: "FormulaNode"


@dataclass(frozen=True)
class Iff:
    left: "FormulaNode"
    right: "FormulaNode"


@dataclass(frozen=True)
class Xor:
    left: "FormulaNode"
    right: "FormulaNode"


@dataclass(frozen=True)
class Always:
    """Past historically, unbounded when ``bounds`` is None."""

    bounds: Optional[Interval]
    child: "FormulaNode"


@dataclass(frozen=True)
class Eventually:
    """Past once, unbounded when ``bounds`` is None."""

    bounds: Optional[Interval]
    child: "FormulaNode"


@dataclass(frozen=True)
class Since:
    left: "FormulaNode"
    bounds: Optional[Interval]
    right: "FormulaNode"


@dataclass(frozen=True)
class Previous:
    """Strong previous: false at the start of a contiguous segment."""

    child: "FormulaNode"


@dataclass(frozen=True)
class Rise:
    """Robust false-to-true edge over a Boolean-valued predicate."""

    child: "FormulaNode"


@dataclass(frozen=True)
class Fall:
    """Robust true-to-false edge over a Boolean-valued predicate."""

    child: "FormulaNode"


FormulaNode = Union[
    Comparison,
    Not,
    And,
    Or,
    Implies,
    Iff,
    Xor,
    Always,
    Eventually,
    Since,
    Previous,
    Rise,
    Fall,
]
