"""
engine.py — parse-once / compile-per-episode monitoring engine (§5.10).

    engine = MonitorEngine.load("specs.yaml")
    reset = reset_adapter(adapter)
    engine.start_episode(
        dt=adapter.dt,
        capabilities=reset.capabilities,
        catalog=reset.discovered_catalog,
    )
    result = engine.update(reset.initial_observation)
    while running:
        result = engine.update(adapter.get_observation())

Specs are parsed to syntax ASTs exactly once, at load; semantic
compilation happens at every episode start against the connected
adapter's dt, capabilities, and discovered catalog — so catalog and
class validation run in the real pipeline, and second-based bounds
adapt to each adapter's sampling period.
"""

from __future__ import annotations

import hashlib
import json
import time as _time
from collections.abc import Mapping
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import yaml

from .capabilities import AdapterCapabilities, AdapterResetResult, VocabularyMode
from .components import ComponentCatalog
from .entities import UnknownClassError
from .monitor import IncrementalBindingManager
from .results import BindingEvaluation, EngineResult, EvaluationStatus
from .spec import parser as _spec_parser
from .spec import semantic as _semantic


class _LoadedSpec:
    def __init__(self, name, node, formula_text, symmetric, constants, backend):
        self.name = name
        self.node = node
        self.formula_text = formula_text
        self.symmetric = symmetric
        self.constants = constants
        self.backend = backend


def _spec_provenance(
    spec_file: str, specs: List[_LoadedSpec]
) -> Dict[str, str]:
    """Identity of the spec set, for recording alongside results.

    The digest covers the parsed content — name, formula text, constants,
    backend and symmetry per spec — rather than the file's bytes, so it
    is unaffected by comments and whitespace and cannot disagree with
    what was actually monitored. Recompute it by loading any candidate
    spec file and comparing.

    Constants are recorded separately because robustness alone does not
    identify the threshold that produced it: for a predicate in a
    monotone position the margin is meaningful only against the bound it
    was measured from.

    Values are recorded as authored, never coerced. Coercing here would
    both break this function's own contract — constant values are not
    validated until ``start_episode``, so ``load()`` must not reject
    them — and record something other than what was written: ``float()``
    accepts ``True`` as ``1.0``, which would bake a fabricated threshold
    into the digest for a spec that ``validate_constants`` will later
    refuse to run at all.
    """
    canonical = [
        {
            "name": s.name,
            "formula": s.formula_text,
            "constants": dict(sorted(s.constants.items())),
            "backend": s.backend,
            "symmetric": bool(s.symmetric),
        }
        for s in sorted(specs, key=lambda s: s.name)
    ]
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), default=str,
    )
    constants = {s.name: dict(s.constants) for s in specs if s.constants}
    return {
        "spec_source": str(spec_file),
        "spec_digest": hashlib.sha256(payload.encode()).hexdigest(),
        "spec_constants": json.dumps(constants, sort_keys=True, default=str),
    }


class MonitorEngine:
    """Multi-spec monitoring engine over one adapter episode at a time."""

    def __init__(
        self,
        specs: List[_LoadedSpec],
        provenance: Optional[Mapping] = None,
    ) -> None:
        self._specs = specs
        self._provenance: Dict[str, str] = dict(provenance or {})
        self._managers: Optional[
            Dict[str, IncrementalBindingManager]
        ] = None
        self._prev_bindings: Dict[str, set] = {}
        self._monitor_counts: Dict[str, int] = {}
        self._skipped: Dict[str, str] = {}
        self._spec_of_manager: Dict[str, str] = {}

    def _fork_unstarted(self) -> "MonitorEngine":
        """Return an unstarted engine over the same parsed specifications.

        Loaded specification records and their syntax trees are read-only
        during semantic compilation. Sharing them preserves the parse-once
        contract while every returned engine receives fresh episode state.
        """
        return MonitorEngine(list(self._specs), provenance=self._provenance)

    @classmethod
    def iter_spec_entries(
        cls, spec_file: str
    ) -> List[Tuple[Optional[str], Optional[_LoadedSpec], Optional[str]]]:
        """Parse a specs YAML into one (name, entry, error) triple per entry.

        Unlike ``load()``'s all-or-nothing contract — one bad entry raises
        before any entry is usable — this walks every entry and reports
        each one's own outcome independently: ``entry`` is a parsed
        ``_LoadedSpec`` and ``error`` is ``None``, or vice versa, never
        both. That is exactly what a tool whose job is reporting every
        entry's status (``specola check``) needs, without disturbing
        ``load()``'s stricter contract for ``start_episode()`` callers —
        ``load()`` is defined in terms of this method below and keeps its
        original behavior verbatim.

        ``name`` is the entry's declared ``name`` field, present even on
        failure (missing-formula, duplicate, parse error) so a caller can
        label the row; entries with no usable name at all (the field is
        missing or empty, or the entry isn't a mapping at all — e.g. a
        bare string or number under ``specs:``) get a positional
        placeholder (``<entry N>``, 1-indexed) instead, so ``name`` is
        never ``None``.
        """
        with open(spec_file) as handle:
            data = yaml.safe_load(handle)

        entries = (data or {}).get("specs", [])
        if not entries:
            raise ValueError(f"No specs found in {spec_file}")

        results: List[
            Tuple[Optional[str], Optional[_LoadedSpec], Optional[str]]
        ] = []
        seen = set()

        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                results.append((
                    f"<entry {index + 1}>",
                    None,
                    "Spec entry must be a mapping",
                ))
                continue
            name = entry.get("name")
            formula = entry.get("formula")
            if not name:
                results.append((
                    f"<entry {index + 1}>",
                    None,
                    "Spec entry missing 'name' field",
                ))
                continue
            if not formula:
                results.append(
                    (name, None, f"Spec '{name}' missing 'formula' field")
                )
                continue
            if name in seen:
                results.append(
                    (name, None, f"Duplicate spec name '{name}'")
                )
                continue
            seen.add(name)
            try:
                node = _spec_parser.parse(formula)
            except ValueError as exc:
                results.append(
                    (name, None, f"Spec '{name}': parse error — {exc}")
                )
                continue
            results.append((
                name,
                _LoadedSpec(
                    name=name,
                    node=node,
                    formula_text=formula.strip(),
                    symmetric=bool(entry.get("symmetric", False)),
                    constants=dict(entry.get("constants", {}) or {}),
                    backend=str(entry.get("backend", "auto")),
                ),
                None,
            ))

        return results

    @classmethod
    def load(cls, spec_file: str) -> "MonitorEngine":
        """Parse a specs YAML into syntax ASTs (no compilation yet)."""
        loaded: List[_LoadedSpec] = []
        errors: List[str] = []

        for _name, entry, error in cls.iter_spec_entries(spec_file):
            if error is not None:
                errors.append(error)
            else:
                loaded.append(entry)

        if errors:
            raise ValueError(
                f"Errors loading {spec_file}:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

        return cls(loaded, provenance=_spec_provenance(spec_file, loaded))

    @property
    def spec_provenance(self) -> Dict[str, str]:
        """Which spec set produced these results.

        Empty for an engine constructed directly rather than through
        ``load()``, since there is no spec file to name — including one
        built by ``__new__`` without ``__init__``, which tests do to
        drive a hand-assembled manager. Missing provenance reads as "no
        spec file behind these results", never as an AttributeError
        raised while writing them.
        """
        return dict(getattr(self, "_provenance", {}))

    @property
    def spec_names(self) -> List[str]:
        return [s.name for s in self._specs]

    def specification_metadata(self) -> Tuple[Dict[str, Any], ...]:
        """Authored specification records in file order for presentation."""
        return tuple(
            {
                "name": spec.name,
                "formula": spec.formula_text,
                "backend": spec.backend,
                "constants": dict(spec.constants),
            }
            for spec in self._specs
        )

    @property
    def managers(self) -> Dict[str, IncrementalBindingManager]:
        if self._managers is None:
            raise RuntimeError(
                "No episode started. Call start_episode() first."
            )
        return self._managers

    def start_episode(
        self,
        *,
        dt: float,
        capabilities: Optional[AdapterCapabilities] = None,
        catalog: Optional[ComponentCatalog] = None,
        require_confirmed: bool = True,
        skip_unsatisfiable: bool = False,
    ) -> None:
        """Semantically compile every spec against this episode's context.

        By default an unsatisfiable spec (one referencing an entity class
        outside this episode's class vocabulary) fails the whole episode
        start, per the fail-fast/fail-loud error philosophy — a spec that
        cannot be evaluated should never be silently dropped. Passing
        ``skip_unsatisfiable=True`` narrows that to a single, explicit
        exception: such specs are recorded via ``skipped_specs()`` and
        excluded from this episode's managers instead of raising. Every
        other spec-config error still raises regardless of this flag.
        """
        if (
            capabilities is not None
            and capabilities.class_mode == VocabularyMode.CLOSED
        ):
            class_vocabulary = frozenset(capabilities.supported_classes)
        elif catalog is not None:
            class_vocabulary = catalog.classes
        else:
            class_vocabulary = None

        # A failed compilation must not leave the previous episode's
        # managers silently serving updates, nor its accessors serving
        # the previous episode's data — every one of these is documented
        # as empty when no episode has started.
        self._managers = None
        self._skipped = {}
        self._spec_of_manager = {}
        self._monitor_counts = {}
        # Not observable today — every read of _prev_bindings is behind
        # the `managers` property, which raises while _managers is None —
        # but leaving one attribute out of this block is what left
        # _monitor_counts stale for two commits. Every attribute carrying
        # per-episode state is reset here, wherever that attribute is
        # otherwise assigned.
        self._prev_bindings = {}
        # update() derives per-call binding_recomputes from these cumulative
        # manager counters. An episode reset creates new managers whose
        # counters restart at zero, so their comparison baseline must restart
        # too or the first update under-reports recomputation work.
        self._prev_incremental = {}

        managers: Dict[str, IncrementalBindingManager] = {}
        monitor_counts: Dict[str, int] = {}
        skipped: Dict[str, str] = {}
        spec_of_manager: Dict[str, str] = {}
        for spec in self._specs:
            try:
                compiled_list = _semantic.compile_formulas(
                    spec.node,
                    formula_name=spec.formula_text,
                    dt=float(dt),
                    constants=_semantic.validate_constants(spec.constants),
                    symmetric=spec.symmetric,
                    catalog=catalog,
                    class_vocabulary=class_vocabulary,
                    backend=spec.backend,
                    capabilities=capabilities,
                )
            except UnknownClassError as exc:
                if skip_unsatisfiable:
                    skipped[spec.name] = str(exc)
                    continue
                raise
            monitor_counts[spec.name] = len(compiled_list)

            # A selector or "**" binding fans one authored spec into many
            # concrete formulas (semantic.compile_formulas); a trivial
            # formula fans into exactly one, whose name equals the
            # authored formula text verbatim — so `suffix` is empty and
            # `manager_key` reproduces today's plain `spec.name` key. No
            # cap is applied here: N expansions get N managers.
            for compiled in compiled_list:
                suffix = compiled.name[len(spec.formula_text):]
                manager_key = f"{spec.name}{suffix}"
                if manager_key in managers:
                    raise ValueError(
                        f"Spec {spec.name!r} fan-out produced a duplicate "
                        f"monitor name {manager_key!r}"
                    )
                managers[manager_key] = IncrementalBindingManager(
                    compiled, require_confirmed=require_confirmed
                )
                spec_of_manager[manager_key] = spec.name

        self._managers = managers
        self._monitor_counts = monitor_counts
        self._skipped = skipped
        self._spec_of_manager = spec_of_manager
        self._prev_bindings = {key: set() for key in managers}
        self._prev_incremental = {key: 0 for key in managers}

    def extraction_scope(self) -> Dict[str, FrozenSet[str]]:
        """Property keys this episode's monitors need, per entity class.

        The union over every compiled parameter slot of its class and its
        required components — which is what an adapter needs to know to
        stop extracting anything else (``SimulatorAdapter``'s
        ``set_extraction_scope``).

        Keyed by class rather than by entity on purpose. A slot's
        candidate set is recomputed from the unbound entities of its
        class at every observation, so an entity that is not currently
        bound still has to expose the components the slot requires or it
        could never become bound. Narrowing per entity would freeze the
        binding set at whatever it was when the scope was computed.

        Requires a started episode: parameter slots exist only after
        semantic compilation.
        """
        scope: Dict[str, Set[str]] = {}
        for manager in self.managers.values():
            for parameter in manager.formula.params:
                scope.setdefault(
                    parameter.entity_class, set()
                ).update(parameter.required_components)
        return {
            entity_class: frozenset(components)
            for entity_class, components in scope.items()
        }

    def extraction_entity_scope(
        self,
    ) -> Dict[str, FrozenSet[Tuple[Optional[int], Tuple[Tuple[str, Tuple[str, ...]], ...]]]]:
        """Which entities of each class a compiled slot can bind (issue #136).

        One selector per distinct parameter slot: its pinned entity ID
        (``None`` for a wildcard) and its retained ``@`` tag filters as
        sorted ``(component, labels)`` pairs. An entity is in scope when a
        selector of its class admits it: the pin is absent or names it,
        and its tags on every filtered component include every label,
        which is the test ``passes_tag_filters`` applies when binding. The
        other eligibility conditions (confirmed track, required components)
        are left out, so the scope can only be wider than the bindable set.

        Requires a started episode, like ``extraction_scope``.
        """
        scope: Dict[str, Set[Tuple[Optional[int], Tuple[Tuple[str, Tuple[str, ...]], ...]]]] = {}
        for manager in self.managers.values():
            for parameter in manager.formula.params:
                filters = tuple(sorted(
                    (component, tuple(labels))
                    for component, labels in parameter.component_tag_filters.items()
                ))
                scope.setdefault(parameter.entity_class, set()).add(
                    (parameter.pinned_entity_id, filters)
                )
        return {
            entity_class: frozenset(selectors)
            for entity_class, selectors in scope.items()
        }

    def monitor_counts(self) -> Dict[str, int]:
        """Authored spec name -> number of monitors compiled this episode.

        A spec with no selector fan-out always maps to 1. There is no
        fan-out cap: a spec that resolves to N monitors gets N — this
        accessor exists purely so callers (e.g. run_monitor) can surface
        that count, not to police it. Empty before the first
        ``start_episode()`` call.
        """
        return dict(self._monitor_counts)

    def skipped_specs(self) -> Dict[str, str]:
        """Authored spec name -> reason, for specs skipped as not
        applicable this episode.

        Populated only when ``start_episode(skip_unsatisfiable=True)`` hit
        an ``UnknownClassError``. Empty otherwise, including before the
        first ``start_episode()`` call.
        """
        return dict(self._skipped)

    def manager_specs(self) -> Dict[str, str]:
        """Manager key -> authored spec name.

        Identity (each manager key equals its spec's name) unless the
        spec fanned out over a selector or a "**" binding, in which case
        several manager keys map back to the same authored spec name.
        Empty before the first ``start_episode()`` call.
        """
        return dict(self._spec_of_manager)

    def unmatched_slots(self) -> Dict[str, list]:
        """Parameter slots that never had a single candidate entity.

        A slot with zero candidates across the whole episode usually
        means a typo'd class or component (or data that never contained
        it). Keyed by monitor name (the manager key — equal to the spec
        name unless the spec fanned out); monitors with no such slots
        are omitted.
        """
        report: Dict[str, list] = {}
        for name, manager in self.managers.items():
            missing = [
                parameter
                for parameter in manager.formula.params
                if parameter.param_key not in manager.ever_matched
            ]
            if missing:
                report[name] = missing
        return report

    def update(self, observation: Any) -> EngineResult:
        """Run all monitors for one observation; return structured results."""
        started = _time.perf_counter()

        evaluations: List[BindingEvaluation] = []
        created = []
        resumed = []
        pruned = 0
        recomputes = 0
        if not hasattr(self, "_prev_incremental"):
            self._prev_incremental = {}

        for name, manager in self.managers.items():
            pruned_before = manager.n_pruned
            known_before = set(manager._instances)
            active_before = self._prev_bindings.get(name, set())

            results = manager.update(observation)

            for binding, rho in results.items():
                instance = manager.get_instance(binding)
                if instance is None:
                    continue
                if instance.errored:
                    status = EvaluationStatus.ERROR
                    rho_out = None
                    final = False
                elif instance.suspended:
                    status = EvaluationStatus.SUSPENDED
                    rho_out = None
                    final = False
                else:
                    final = instance.verdict_final
                    status = (
                        EvaluationStatus.ACTIVE
                        if final
                        else EvaluationStatus.WARMING_UP
                    )
                    rho_out = rho
                component_values = tuple(
                    (
                        int(entity_id),
                        key,
                        float(
                            observation.perceived_entities[
                                entity_id
                            ].properties[key]
                        ),
                    )
                    for entity_id, parameter in zip(
                        binding, manager.formula.params
                    )
                    for key in parameter.required_components
                    if (
                        entity_id in observation.perceived_entities
                        and key in observation.perceived_entities[
                            entity_id
                        ].properties
                    )
                )
                evaluations.append(
                    BindingEvaluation(
                        spec_name=name,
                        binding=binding,
                        robustness=rho_out,
                        status=status,
                        verdict_final=final,
                        timestep=int(observation.timestep),
                        time=float(observation.time),
                        segment_samples=int(instance.segment_samples),
                        window_steps=int(instance.formula.window_steps),
                        error_message=instance.error_message,
                        component_values=component_values,
                    )
                )

                if binding not in known_before:
                    created.append((name, binding))
                elif (
                    binding in manager.current_bindings
                    and binding not in active_before
                ):
                    resumed.append((name, binding))

            pruned += manager.n_pruned - pruned_before
            total = (
                manager.binding_set.n_incremental
                + manager.binding_set.n_full
            )
            recomputes += total - self._prev_incremental.get(name, 0)
            self._prev_incremental[name] = total
            self._prev_bindings[name] = set(manager.current_bindings)

        return EngineResult(
            timestep=int(observation.timestep),
            time=float(observation.time),
            evaluations=tuple(evaluations),
            created=tuple(created),
            resumed=tuple(resumed),
            pruned_count=pruned,
            binding_recomputes=recomputes,
            update_seconds=_time.perf_counter() - started,
        )
