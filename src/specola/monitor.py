"""
monitor.py — spec templates and the per-binding monitors built from them
=============================================================
Snapshot-driven monitoring of STL spec templates over dynamically
evolving multi-entity scenes.  At each timestep the system maintains one
MonitorInstance per valid binding θ ∈ Θ_φ(t) and evaluates ρ(φ, θ, t) online.

  IncrementalBindingManager
      Binding-set maintenance keyed on per-slot candidate sets
      (bindings.eligibility). The fast path skips enumeration while the
      candidate map is unchanged; any eligibility change triggers a
      recomputation from the complete-enumeration reference
      implementation (bindings.enumerate), so the active binding set
      equals complete enumeration by construction.

Formal correspondence (definitions document):
  ParameterSpec             → one slot x = (λ, i) ∈ X_φ  (Defs 6, 9)
  SpecTemplate              → (φ, X_φ) with predicate fn and temporal structure (Def 10)
  MonitorInstance           → M(φ, θ): evaluates ρ(φ, θ, t) online (Def 11)
  IncrementalBindingManager → top-level: receives ω(t), returns {θ: ρ}

Note on parametric trace slicing: the parameter-binding semantics of
Definition 1.9 are adapted from Chen & Rosu (TACAS 2009).  Their event-driven
algorithms (with partial-binding extension via ⊥/⊑/⊔) do not transfer
directly to this snapshot-driven setting, where every observation reveals
all currently-perceived entities at once and partial bindings collapse to
triviality.  The two managers above are independent designs for snapshot-
driven binding maintenance, not implementations of Chen & Rosu's algorithms.

Temporal backend
----------------
  MonitorInstance uses the compile-time-selected native or RTAMT temporal
  backend. The predicate functions produced by stl_parser.py yield scalar
  µ[θ,t] values; the backend evaluates the causal past-time STL formula over
  those streams.

  Formula mapping (past-time equivalents, no pastify() needed):
    historically[0,b](mu) → historically[0, b-1](mu)
    once[0,b](mu)         → once[0, b-1](mu)
    bare    →  mu                           (instant predicate)

  The off-by-one (b-1) aligns RTAMT's [t-(b-1), t] window (b values) with
  the previous deque(maxlen=b) behaviour.

  RTAMT availability is resolved at semantic compilation only when explicitly
  requested. Auto mode selects the native backend for the complete causal
  fragment. The deque sliding-window fallback exists only for legacy
  hand-constructed single-window formulas (rtamt_spec is None).

Monitor lifecycle (RESET_AND_REWARM — see MonitorInstance.tick_suspended)
--------------------------------------------------------------------------
  ACTIVE    : θ ∈ Θ_φ(t) — entity perceived; µ fed to RTAMT; ρ returned.
  SUSPENDED : θ ∉ Θ_φ(t) — entity absent.  The backend is NOT called during
              suspension: no fabricated sample is fed to it, so no artificial
              violation or satisfaction is introduced.  Returns None during
              this period.
  PRUNED    : grace period (= window_steps) elapsed — instance removed.

Why window_steps as grace period?
  G[0,b] robustness = min over the last b samples.  After b absent steps,
  all pre-suspension samples would have scrolled out of the window even
  under continuous aging, so a fresh instance on re-detection loses nothing
  a still-active one would have retained.

On resumption within the grace period, the backend is reset and the
resuming observation becomes sample zero of a new contiguous segment
(pre-gap history is discarded, not aged); see MonitorInstance.evaluate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple, Any
)

import numpy as np

from .entities import SpecConfigError
from .components import ComponentResolver

# ─────────────────────────────────────────────────────────────────────────────
# RTAMT BACKEND DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# RTAMT probing and specification construction live in backends.rtamt;
# these names remain importable here for compatibility (tests patch
# monitor._rtamt_available to simulate a missing backend).
from .backends.rtamt import (  # noqa: E402
    make_legacy_spec as _make_legacy_spec,
    rtamt_available as _rtamt_available,
)


# ─────────────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ─────────────────────────────────────────────────────────────────────────────

from .monitor_types import (  # noqa: F401  (re-exported public API)
    Binding,
    ParamKey,
    binding_to_dict,
    make_binding,
)
from .bindings import enumerate_complete, slot_candidates
from .bindings.incremental import IncrementalBindingSet


def binding_label(b: Binding, params: List["ParameterSpec"]) -> str:
    if not params:
        return "∅"
    d = binding_to_dict(b, params)
    parts = []
    for p in params:
        eid = d.get(p.param_key, "?")
        cls = p.param_key[0]
        cls_text = str(getattr(cls, "value", cls))
        parts.append(f"{cls_text[0]}:{p.param_key[1]}→e{eid}")
    return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER SPECIFICATION  (Definition 6 — one slot x = (λ, i) ∈ X_φ)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParameterSpec:
    """
    One parameter slot x = (λ, i) in X_φ.

    param_key            : (EntityClass, i)
    is_wildcard          : True  → bound at runtime to every entity of class λ
                           False → pinned to a specific entity at spec time
    required_components  : req(φ, x) — state components needed from this param
    pinned_entity_id     : only meaningful when is_wildcard=False
    component_tag_filters: retained ``@`` labels per required component,
                           for declared-schema dynamic classes whose tag
                           filters cannot desugar at reset. An entity is
                           eligible only when, for every listed
                           component, its observation-time tags contain
                           every listed label. Empty for slots resolved
                           at compile time.
    """
    param_key:           ParamKey
    is_wildcard:         bool
    required_components: List[str]
    pinned_entity_id:    Optional[int] = None
    component_tag_filters: Mapping[str, Tuple[str, ...]] = field(
        default_factory=dict
    )

    @property
    def entity_class(self) -> Any:
        return self.param_key[0]

    @property
    def param_index(self) -> int:
        return self.param_key[1]


# ─────────────────────────────────────────────────────────────────────────────
# STL TEMPORAL OPERATORS  (Definition 9)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalOp(str, Enum):
    """Temporal operator for a spec template."""
    ALWAYS     = "always"       # G[0,b]φ — past-time historically (sliding min)
    EVENTUALLY = "eventually"   # F[0,b]φ — past-time once         (sliding max)
    NONE       = "none"         # φ       — no temporal wrapping


# ─────────────────────────────────────────────────────────────────────────────
# SPEC TEMPLATE  (Definitions 9–10)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpecTemplate:
    """
    A spec template Φ(φ) = (φ, X_φ).

    params        : ordered list of ParameterSpec objects (= X_φ)
    predicate_fn  : predicate_fn(binding_dict, entity_states) → float
    temporal_op   : TemporalOp.ALWAYS / EVENTUALLY / NONE
    window_steps  : b in G[0,b] or F[0,b] (timestep units)
                    Also used as the suspension grace period.
    dt            : timestep in seconds
    name          : human-readable label
    """
    params:       List[ParameterSpec]
    predicate_fn: Optional[Callable[[Dict[ParamKey, int], Dict[int, Any]], float]] = None
    temporal_op:  str   = TemporalOp.ALWAYS
    window_steps: int   = 100
    dt:           float = 0.05
    name:         str   = "φ"

    # Nested-RTAMT mode. When rtamt_spec is not None, MonitorInstance declares
    # all variables in predicate_fns and gives rtamt_spec directly to RTAMT.
    # This lets RTAMT handle logical/temporal nesting while the parser handles
    # parameter extraction and SymPy compilation of atomic predicates.
    predicate_fns: Optional[Dict[str, Callable[[Dict[ParamKey, int], Dict[int, Any]], float]]] = None
    rtamt_spec:    Optional[str] = None
    neutral_values: Optional[Dict[str, float]] = None
    grace_steps_override: Optional[int] = None
    # If True, the predicate is invariant under permutation of equal-class
    # wildcards; binding enumeration emits canonical (sorted) assignments
    # via combinations instead of permutations, halving the binding count
    # for k_λ=2 and giving k_λ! reduction in general.
    symmetric:    bool  = False
    # Top-level temporal operator name, or None, captured by the parser.
    # The legacy always/eventually values are used by display/reporting
    # layers to distinguish a negative-but-progressing once spec from a
    # violated historically spec; other causal operators are retained for
    # inspection without special display treatment.
    outer_op:     Optional[str] = None
    # Temporal backend plan, resolved at semantic compilation:
    # backend is "native" or "rtamt"; make_backend constructs one fresh
    # TemporalBackend per MonitorInstance. None on legacy hand-built
    # formulas, which keep the historical evaluation paths.
    backend:      Optional[str] = None
    make_backend: Optional[Callable[[], Any]] = None
    # Desugared formula text (selectors and "**" expanded, entity fan-out
    # pinned) for the concrete formula this rewrite compiled to. Set by
    # compile_formulas for every fanned-out expansion; None on the trivial
    # single-formula path, where no rewriting happened.
    desugared_text: Optional[str] = None
    # "always" / "eventually" when this formula is a zero-lower-bound
    # trailing window directly over a comparison, the one shape whose
    # warm-up robustness is a monotone bound on the settled answer, and so
    # the one shape whose verdict may become final before the window
    # fills. None for every other shape. Set by semantic compilation; see
    # MonitorInstance.verdict_final.
    #
    # Appended rather than grouped with outer_op, which it belongs with
    # logically: SpecTemplate is exported in specola.__all__, so inserting
    # a field ahead of backend/make_backend/desugared_text would silently
    # re-map any existing positional construction.
    settles_early: Optional[str] = None

    def __post_init__(self):
        self._params_by_class: Dict[Any, List[ParameterSpec]] = {}
        for p in self.params:
            self._params_by_class.setdefault(p.entity_class, []).append(p)

    @property
    def params_by_class(self) -> Dict[Any, List[ParameterSpec]]:
        return self._params_by_class

    @property
    def window_seconds(self) -> float:
        return self.window_steps * self.dt

    @property
    def grace_steps(self) -> int:
        """Grace period before pruning a suspended instance."""
        if self.grace_steps_override is not None:
            return self.grace_steps_override
        return self.window_steps


# ─────────────────────────────────────────────────────────────────────────────
# MONITOR INSTANCE  (Definition 11)
# ─────────────────────────────────────────────────────────────────────────────

class MonitorInstance:
    """
    M(φ, θ) — one monitor instance for one complete binding θ.

    Owns one RTAMT StlDiscreteTimeSpecification (or a deque fallback) and
    evaluates ρ(φ, θ, t) online at each timestep by:
      1. Computing the scalar predicate value µ[θ,t] via formula.predicate_fn
      2. Feeding µ to RTAMT as the signal 'mu' at the current step index
      3. Reading back the robustness ρ from RTAMT's online update()

    During suspension (entity temporarily absent from E(t)), tick_suspended()
    does NOT call the backend: no fabricated sample is fed to it, so absence
    is read as neither satisfaction nor violation. On resumption within the
    grace period, evaluate() resets the backend and treats the resuming
    observation as sample zero of a new contiguous segment; pre-gap history
    is discarded rather than aged (RESET_AND_REWARM).

    Lifecycle:
      ACTIVE    : entity present in E(t), steps_suspended == 0
      SUSPENDED : entity absent from E(t), 0 < steps_suspended < window_steps
      PRUNED    : steps_suspended >= window_steps — removed by manager
    """

    def __init__(self, formula: "SpecTemplate", binding: Binding,
                 created_at_step: int):
        self.formula         = formula
        self.binding         = binding
        self.binding_dict    = binding_to_dict(binding, formula.params)
        self.created_at_step = created_at_step

        # ── Temporal backend, chosen at semantic compilation ─────────────────
        if formula.make_backend is not None:
            self._backend = formula.make_backend()
            self._rtamt = None
            self._use_rtamt = False
        else:
            # Legacy hand-constructed formulas: RTAMT wrapper when
            # available, deque sliding window otherwise.
            self._backend = None
            self._rtamt = _make_legacy_spec(formula)
            self._use_rtamt = self._rtamt is not None

        # Fallback: deque sliding-window (used when RTAMT unavailable)
        self._buf: deque     = deque(maxlen=formula.window_steps)

        # Local step counter for RTAMT (counts both active + suspended steps)
        self._step: int      = 0

        self.suspended:         bool           = False
        self.steps_suspended:   int            = 0
        self.last_rho:          Optional[float] = None
        self.n_updates:         int            = 0

        # Terminal error state (design §5.9/§6): an unrecoverable backend
        # error marks the instance ERROR; it is never updated again and
        # last_rho is only ever stale diagnostic data from then on.
        self.errored:           bool           = False
        self.error_message:     Optional[str]  = None

        # Contiguous samples fed since the current segment started
        # (creation or last post-suspension rewarm). Drives verdict
        # finality: a window of W samples is final at exactly W.
        self.segment_samples:   int            = 0

        # Set when the verdict settled before the window filled. Cleared
        # with the segment, since a rewarmed instance starts from a new
        # contiguous prefix and has decided nothing yet.
        self.settled_early:     bool           = False

    # ── Online robustness update ──────────────────────────────────────────────

    def update(self, resolver: "ComponentResolver") -> Optional[float]:
        """
        Compute ρ(φ, θ, t) for the current observation.

        Parameters
        ----------
        resolver:
            The observation-scoped ComponentResolver created once per
            MonitorObservation by the manager and shared across every
            active instance, so repeated component reads are cached.

        Called by manager only when binding is in current_bindings (entity
        is perceived).  Resets suspension counter on call.
        """
        if self.errored:
            raise RuntimeError(
                f"Monitor instance is in ERROR state: {self.error_message}"
            )

        if self.suspended:
            # RESET_AND_REWARM: the backend never consumed fabricated
            # samples during the gap; discard pre-gap history and treat
            # this observation as sample zero of a new contiguous
            # segment. Previous robustness stays available only as
            # explicitly stale diagnostic data (last_rho).
            self._reset_backend()
            self.segment_samples = 0
            self.settled_early = False

        self.suspended       = False
        self.steps_suspended = 0

        # No exception is ever converted into an apparently valid
        # robustness value: evaluation and backend errors propagate.
        if self._backend is not None:
            signals = {
                name: fn(self.binding_dict, resolver)
                for name, fn in self.formula.predicate_fns.items()
            }
            rho = float(self._backend.update(self._step, signals))
            self._step += 1
        elif self.formula.rtamt_spec is not None:
            values = [
                (name, fn(self.binding_dict, resolver))
                for name, fn in sorted(self.formula.predicate_fns.items())
            ]
            rho = self._eval_rtamt_values(values)
        else:
            mu = self.formula.predicate_fn(self.binding_dict, resolver)
            rho = self._eval_temporal(mu)

        self.last_rho  = rho
        self.n_updates += 1
        self.segment_samples += 1

        if not self.settled_early and self._settles_now(rho):
            self.settled_early = True

        return rho

    def _settles_now(self, rho: float) -> bool:
        """
        Whether ``rho`` settles this instance's verdict before its window
        has filled.

        Only a zero-lower-bound trailing window directly over a
        comparison qualifies (``formula.settles_early``), because only
        there is the warm-up robustness the running min or max of the
        child over the samples so far.

        Given that, one direction of each operator is monotone and one is
        not. A running minimum only falls, so a negative ``G`` stays
        negative until the window fills and FAIL is settled; a
        non-negative one says nothing, since a later sample can still
        violate. A running maximum only rises, so a non-negative ``F``
        settles PASS, and a negative one says nothing.

        The unsettled direction deliberately returns False rather than
        guessing: reporting a verdict that a later sample can overturn
        would be worse than reporting it late.
        """
        kind = self.formula.settles_early

        if kind is None:
            return False

        if kind == TemporalOp.ALWAYS:
            return rho < 0

        if kind == TemporalOp.EVENTUALLY:
            return rho >= 0

        return False

    def _reset_backend(self) -> None:
        if self._backend is not None:
            self._backend.reset()
        elif self._use_rtamt:
            self._rtamt = _make_legacy_spec(self.formula)
            self._step = 0
        else:
            self._buf.clear()

    def _eval_rtamt_values(self, values: List[Tuple[str, float]]) -> float:
        """Feed one or more atomic predicate signals to RTAMT and return ρ."""
        if not self._use_rtamt:
            raise RuntimeError("Nested STL requires RTAMT; fallback is intentionally unsupported.")
        rho = self._rtamt.update(self._step, values)
        self._step += 1
        return float(rho)

    def _eval_temporal(self, mu: float) -> float:
        """
        Feed µ to the temporal backend and return ρ.
        Uses RTAMT when available, deque otherwise.
        """
        if self._use_rtamt:
            rho = self._rtamt.update(self._step, [('mu', mu)])
            self._step += 1
            return float(rho)

        # ── Deque fallback ────────────────────────────────────────────────────
        op = self.formula.temporal_op
        if op == TemporalOp.NONE:
            return mu
        self._buf.append(mu)
        if op == TemporalOp.ALWAYS:
            return min(self._buf)
        if op == TemporalOp.EVENTUALLY:
            return max(self._buf)
        return mu

    def tick_suspended(self) -> None:
        """
        Advance the suspension counter by one step.

        The temporal backend is never called during suspension: no
        fabricated samples, no artificial window aging. Logical lifecycle
        time advances through steps_suspended only; on resumption the
        backend is reset and rewarmed from the new contiguous segment.
        """
        self.suspended       = True
        self.steps_suspended += 1

    def mark_error(self, message: str) -> None:
        """Enter the terminal ERROR state (unrecoverable backend error)."""
        self.errored = True
        self.error_message = message

    @property
    def verdict_final(self) -> bool:
        """Whether the current temporal-window verdict is no longer
        provisional.

        This is not lifetime finality: a settled verdict may change later as
        the window slides. Two ways make the current verdict ready. The
        window has filled -- counted per
        instance, in contiguous samples since creation or the last
        post-suspension rewarm, never the global episode step. Or the
        verdict settled early, which reporting a settled answer the
        moment it settles is the point of monitoring online rather than
        after the fact (see ``_settles_now`` for which shapes qualify and
        why).
        """
        return (
            self.settled_early
            or self.segment_samples >= self.formula.window_steps
        )

    @property
    def grace_expired(self) -> bool:
        """True when suspension has outlasted the formula's window."""
        return self.steps_suspended >= self.formula.grace_steps

    @property
    def grace_remaining(self) -> int:
        return max(0, self.formula.grace_steps - self.steps_suspended)

    # ── Display ───────────────────────────────────────────────────────────────

    def label(self) -> str:
        return binding_label(self.binding, self.formula.params)

    def short_label(self) -> str:
        d = self.binding_dict
        return "↔".join(str(d[p.param_key]) for p in self.formula.params)

    @property
    def age_steps(self) -> int:
        return self.n_updates

    @property
    def is_safe(self) -> Optional[bool]:
        if self.last_rho is None:
            return None
        return self.last_rho >= 0.0

    def robustness_str(self) -> str:
        if self.suspended:
            remaining = self.grace_remaining
            return f"  ? [{remaining}]"     # ? [steps until pruned]
        if self.last_rho is None:
            return "  ?"
        return f"{self.last_rho:+.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# BINDING ENUMERATION  (Definitions 7, 10)
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_bindings(
    formula: SpecTemplate,
    obs: Any,
    require_confirmed: bool = True,
    resolver: Optional["ComponentResolver"] = None,
) -> Set[Binding]:
    """
    Compute Θ_φ(t) — all valid complete bindings at this timestep.

    Eligibility is evaluated per parameter slot (bindings.eligibility):
    an entity must expose the components required by the slot it fills,
    never the union across its class. Same-class assignments are
    injective. A formula with zero entity parameters yields exactly one
    empty binding.

    Parameters
    ----------
    resolver:
        Optional observation-scoped ComponentResolver. When omitted, one
        is constructed for eligibility checks; callers already holding a
        resolver for this observation should pass it to share the cache.
    """
    if resolver is None:
        resolver = ComponentResolver(obs.perceived_entities)

    candidates = slot_candidates(
        formula,
        obs.perceived_entities,
        require_confirmed,
        resolver,
        component_tags=getattr(obs, "component_tags", None),
    )

    return enumerate_complete(formula, candidates)


# ─────────────────────────────────────────────────────────────────────────────
# INCREMENTAL BINDING MANAGER  (inverse-indexed binding maintenance)
# ─────────────────────────────────────────────────────────────────────────────

class IncrementalBindingManager:
    """
    Binding-set maintenance keyed on per-slot candidate sets.

    Per-step algorithm
    ------------------
      1. Compute per-slot candidate sets (bindings.eligibility). Any
         eligibility change — entity arrival/departure, track-state
         change, or per-entity component availability change — shows up
         as a candidate-map change.
      2. Fast path: unchanged candidate map ⇒ unchanged active binding
         set. Otherwise recompute the active set with the complete-
         enumeration reference implementation (bindings.enumerate),
         creating instances for bindings not seen before; existing
         instances are retained (reactivation) or left to suspend.
      3. Lifecycle ticks: tick_suspended on instances not active; prune
         those whose grace expired.
      4. Update active instances and collect ρ.

    Correctness
    -----------
      The active binding set equals complete enumeration of the current
      observation *by construction* (it is recomputed from the reference
      implementation whenever eligibility changes). The optimized
      incremental maintenance — departures via an inverse index and
      arrivals via inclusion-exclusion — returns with the parity-proof
      stage of the migration; ``n_recomputes`` makes the recompute rate
      observable in the meantime.

    Complexity
    ----------
      No eligibility churn:  O(|params| · |entities|) for the candidate
                             map plus O(|active|) monitor updates.
      With churn:            plus one complete enumeration.
    """

    def __init__(
        self,
        formula:           SpecTemplate,
        require_confirmed: bool = True,
    ):
        self.formula           = formula
        self.require_confirmed = require_confirmed

        # Compiled formulas carry a backend factory resolved at semantic
        # compilation. This guard covers only legacy hand-constructed
        # formulas that request nested-RTAMT evaluation without a factory.
        if (
            formula.make_backend is None
            and formula.rtamt_spec is not None
            and not _rtamt_available()
        ):
            raise RuntimeError(
                f"Formula {formula.name!r} requires the RTAMT temporal "
                f"backend, but RTAMT is not installed or its C++ backend "
                f"is not functional. Install it with: "
                f"pip install 'specola[rtamt]'"
            )

        self._instances:            Dict[Binding, MonitorInstance] = {}
        self._active_bindings:      Set[Binding]                   = set()
        self._prev_candidates:      Optional[
            Dict[ParamKey, FrozenSet[int]]
        ] = None
        self.binding_set = IncrementalBindingSet(formula)
        self._timestep:             int                            = 0

        # Parameter slots whose candidate set was non-empty at least
        # once this episode; never-matched slots are surfaced as
        # possible spec typos at end of episode.
        self.ever_matched: Set[ParamKey] = set()

        self.n_created:  int = 0
        self.n_pruned:   int = 0
        self.n_total:    int = 0
        self.n_recomputes: int = 0

    # ── Main algorithm step ───────────────────────────────────────────────────

    def update(self, obs: Any) -> Dict[Binding, Optional[float]]:
        """Process one scene observation ω(t). See class docstring."""

        t = self._timestep

        # One observation-scoped resolver, shared by eligibility checks
        # and every active instance's predicate evaluation this step.
        resolver = ComponentResolver(obs.perceived_entities)

        # Step 1: per-slot candidate sets. Any eligibility change —
        # arrival, departure, track-state change, or a component
        # appearing/disappearing on an entity — shows up here.
        candidates = slot_candidates(
            self.formula,
            obs.perceived_entities,
            self.require_confirmed,
            resolver,
            component_tags=getattr(obs, "component_tags", None),
        )

        for key, ids in candidates.items():
            if ids:
                self.ever_matched.add(key)

        # Step 2: bring the active binding set up to date whenever any
        # slot's candidate set changed (fast path: unchanged map ⇒
        # unchanged binding set). Maintenance is delta-based
        # (bindings.incremental — proof sketch in its module docstring);
        # its invariant is equality with complete enumeration, validated
        # by the property-based parity suite. Counters on binding_set
        # (n_full, n_incremental) make the behavior observable.
        if candidates != self._prev_candidates:
            self.n_recomputes += 1
            target = self.binding_set.update(candidates)
            for b in target:
                if b not in self._instances:
                    self._create_instance(b, t)
            self._active_bindings = target

        # Step 5: lifecycle ticks (suspend / prune)
        to_prune: List[Binding] = []
        for binding, inst in self._instances.items():
            if binding not in self._active_bindings:
                inst.tick_suspended()
                if inst.grace_expired:
                    to_prune.append(binding)

        for b in to_prune:
            self._remove_instance(b)
            self.n_pruned += 1

        # Step 6: compute ρ for active instances
        results: Dict[Binding, Optional[float]] = {}
        for binding, inst in self._instances.items():
            if binding in self._active_bindings and not inst.errored:
                try:
                    rho = inst.update(resolver)
                except SpecConfigError:
                    raise  # configuration errors fail loudly (§6)
                except Exception as exc:
                    # Unrecoverable backend/evaluation error: terminal
                    # ERROR state; never an apparently-valid robustness.
                    inst.mark_error(f"{type(exc).__name__}: {exc}")
                    rho = None
                results[binding] = rho
            else:
                results[binding] = None

        self._prev_candidates = candidates
        self._timestep       += 1
        self.n_total          = len(self._instances)
        return results

    # ── Instance helpers ──────────────────────────────────────────────────────

    def _create_instance(self, binding: Binding, created_at_step: int) -> None:
        if binding in self._instances:
            return
        self._instances[binding] = MonitorInstance(
            self.formula, binding, created_at_step=created_at_step
        )
        self.n_created += 1

    def _remove_instance(self, binding: Binding) -> None:
        del self._instances[binding]
        self._active_bindings.discard(binding)

    # ── Accessors (mirror legacy manager interface) ───────────────────────────

    @property
    def active_instances(self) -> List[MonitorInstance]:
        return list(self._instances.values())

    @property
    def n_instances(self) -> int:
        return len(self._instances)

    @property
    def n_active(self) -> int:
        return sum(1 for i in self._instances.values() if not i.suspended)

    @property
    def n_suspended(self) -> int:
        return sum(1 for i in self._instances.values() if i.suspended)

    @property
    def current_bindings(self) -> Set[Binding]:
        return set(self._active_bindings)

    def get_instance(self, binding: Binding) -> Optional[MonitorInstance]:
        return self._instances.get(binding)

    def summary(self) -> str:
        return (
            f"Formula: {self.formula.name}  |  "
            f"Instances: {self.n_instances} "
            f"({self.n_active} active, {self.n_suspended} suspended "
            f"[grace={self.formula.grace_steps} steps])  |  "
            f"Created: {self.n_created}  Pruned: {self.n_pruned}  "
            f"[incremental]"
        )
