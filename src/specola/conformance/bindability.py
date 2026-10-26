"""
bindability.py: static-slot bindability probe (CORE.STATIC_SLOT_BINDABILITY).

An author who names a declared key on its declared class must get a
binding. The X15 never-bound gripper slot
(``docs/reviews/2026-09-02-x15-never-bound-slot-postmortem.md``) is the
failure this obligation certifies against: the key existed, the class
existed, compilation accepted the specification, and no monitor instance
was ever created.

The probe drives the same two legs a monitored episode drives. The
compile leg compiles a one-slot formula against the reset catalog and the
adapter's declared capabilities, exactly as ``specola check`` does. The
runtime leg starts a real session over a fresh adapter and steps it, so a
key that survives compilation but disappears under the episode's
extraction scope is reported rather than silently monitored as nothing.

The denominator is one case per (static class, component group): every
group of every class the adapter enumerates and does not declare
runtime-populated. One case per group rather than one per key keeps the
probe proportional to the surface an author selects from, since a group
is the unit a selector filter names.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

from ..adapter import SimulatorAdapter
from ..capabilities import VocabularyMode, reset_adapter
from ..components import ComponentCatalog
from ..entities import SpecConfigError
from ..registry import resolve
from ..session import Monitor
from ..stl_parser import compile_specs
from .core_runner import ConformanceValidationError

NO_LIVE_PRODUCER = "no live producer in this environment"


@dataclass(frozen=True)
class BindabilityCase:
    """One (static class, group) probe case and the key that represents it."""

    entity_class: str
    group: str
    component: str
    entity_id: int


@dataclass(frozen=True)
class BindabilityResult:
    """The outcome of one case, with the leg that decided it."""

    case: BindabilityCase
    status: str  # "pass" | "fail" | "unexecuted"
    reason: str


@dataclass(frozen=True)
class BindabilityArtifact:
    """The probe's complete record for one adapter.

    ``outcome`` is one of:

    ``pass``
        every case bound at least once during the runtime leg.
    ``fail``
        at least one case did not compile, had no eligible entity, or
        never produced an evaluation.
    ``not_applicable``
        the adapter's class vocabulary is OPEN, so section 5.6.1
        compile-and-wait applies and no class is static; or no static
        class exposes any component, leaving nothing to bind.
    ``setup_unavailable``
        the adapter factory or the reset raised, or the runtime leg never
        started, so at least one case stayed undecided; ``detail`` keeps
        the exception text. A factory that still builds a different scene
        after ``seed`` is applied lands here whenever the second scene
        drops a probed key: its two legs then describe two scenes, and a
        verdict over them would be about the factory, not about
        bindability. Scene equivalence is assumed, not measured: two
        scenes that differ only in entity identity while exposing the
        same class and key vocabulary are indistinguishable to this
        probe and come back as an ordinary verdict.

    ``seed`` is the value both legs seeded the global random number
    generators with before their factory call.
    """

    adapter_type: str
    class_mode: str
    static_classes: Tuple[str, ...]
    runtime_populated_classes: Tuple[str, ...]
    cases: Tuple[BindabilityCase, ...]
    results: Tuple[BindabilityResult, ...]
    outcome: str
    detail: str = ""
    seed: int = 0

    def to_json_dict(self) -> Dict[str, object]:
        """A plain JSON-serializable view of the record."""
        return {
            "adapter_type": self.adapter_type,
            "class_mode": self.class_mode,
            "static_classes": list(self.static_classes),
            "runtime_populated_classes": list(self.runtime_populated_classes),
            "cases": [_case_json(case) for case in self.cases],
            "results": [
                {
                    "case": _case_json(item.case),
                    "status": item.status,
                    "reason": item.reason,
                }
                for item in self.results
            ],
            "outcome": self.outcome,
            "detail": self.detail,
            "seed": self.seed,
        }


def _case_json(case: BindabilityCase) -> Dict[str, object]:
    return {
        "entity_class": case.entity_class,
        "group": case.group,
        "component": case.component,
        "entity_id": case.entity_id,
    }


def probe_static_slot_bindability(
    adapter_factory: Callable[[], SimulatorAdapter],
    *,
    steps: int = 1,
    seed: int = 0,
) -> BindabilityArtifact:
    """Probe whether every static class slot can bind on this adapter.

    ``adapter_factory`` is called twice: once for the compile leg's reset
    and discovery, once for the runtime leg's session, because a session
    installs an extraction scope the compile leg must not see. Both
    adapters are closed before the artifact returns, including on the
    paths where a leg raises, so a probe over several registry entries
    does not accumulate simulator processes.

    ``seed`` is applied to the global ``random`` and ``numpy.random``
    generators immediately before each factory call, so a factory that
    samples its scene from either global generator draws the same
    sequence in both legs when the legs perform the same number of
    resets. ``benchmarks/x15/x15_discover.py`` seeds by the same rule.

    Global seeding is not a determinism guarantee. A factory that samples
    from a generator of its own is unaffected: robosuite builds one per
    environment with ``np.random.default_rng(seed)`` and RoboCasa's
    factory leaves ``seed`` at ``None``
    (``robosuite/environments/base.py`` lines 144-145), so its two legs
    still describe two kitchens and the probe reports
    ``setup_unavailable`` rather than a verdict. Scene equivalence is
    assumed where the legs agree, never measured: two scenes exposing the
    same class and key vocabulary are indistinguishable here even when
    their entities differ.
    """
    adapter = None
    try:
        _seed(seed)
        adapter = adapter_factory()
        reset = reset_adapter(adapter)
    except Exception as exc:  # the adapter, not the probe, is the subject
        _close(adapter)
        return BindabilityArtifact(
            _adapter_type(adapter_factory), "unknown", (), (), (), (),
            "setup_unavailable", "{}: {}".format(type(exc).__name__, exc),
            seed,
        )

    try:
        capabilities = reset.capabilities
        catalog = reset.discovered_catalog
        dynamic = tuple(sorted(capabilities.runtime_populated_classes or ()))
        adapter_type = _adapter_type(type(adapter))

        if capabilities.class_mode is not VocabularyMode.CLOSED:
            return BindabilityArtifact(
                adapter_type, capabilities.class_mode.value, (), dynamic,
                (), (), "not_applicable",
                "open class vocabulary: section 5.6.1 compile-and-wait applies",
                seed,
            )

        static_classes = tuple(sorted(
            entity_class for entity_class in catalog.classes
            if entity_class not in set(dynamic)
        ))
        cases = _denominator(catalog, static_classes)
        if not cases:
            return BindabilityArtifact(
                adapter_type, capabilities.class_mode.value, static_classes,
                dynamic, (), (), "not_applicable",
                "no static class exposes a component",
                seed,
            )

        compiled: Dict[int, BindabilityCase] = {}
        results: Dict[int, BindabilityResult] = {}
        detail = ""
        for index, case in enumerate(cases):
            status, reason = _compile_case(
                case, catalog, capabilities, adapter.dt
            )
            if status is None:
                compiled[index] = case
            else:
                if status == "unexecuted":
                    detail = detail or reason
                results[index] = BindabilityResult(case, status, reason)

        if compiled:
            bound, error = _run_bindings(
                adapter_factory, compiled, steps, seed
            )
            for index, case in compiled.items():
                if error is not None:
                    # No case was decided: an undecided case is never a
                    # never-bindable slot, so it stays unexecuted and the
                    # compile leg's own failures keep their verdict.
                    detail = detail or error
                    results[index] = BindabilityResult(
                        case, "unexecuted",
                        "runtime leg unavailable: " + error,
                    )
                elif _spec_name(index) in bound:
                    results[index] = BindabilityResult(
                        case, "pass", "{} bound on entity {}".format(
                            _slot_text(case), case.entity_id,
                        ),
                    )
                else:
                    results[index] = BindabilityResult(
                        case, "fail",
                        "{} compiled but no monitor evaluated it in {} "
                        "step(s); entity {} exposes the key at reset".format(
                            _slot_text(case), steps, case.entity_id,
                        ),
                    )

        ordered = tuple(results[index] for index in range(len(cases)))
        statuses = {item.status for item in ordered}
        if "fail" in statuses:
            outcome = "fail"
        elif "unexecuted" in statuses:
            outcome = "setup_unavailable"
        else:
            outcome = "pass"
        return BindabilityArtifact(
            adapter_type, capabilities.class_mode.value, static_classes,
            dynamic, cases, ordered, outcome, detail, seed,
        )
    finally:
        _close(adapter)


def _seed(seed: int) -> None:
    """Put the global generators a factory may sample from in one state."""
    random.seed(seed)
    np.random.seed(seed)


def _close(adapter: Optional[SimulatorAdapter]) -> None:
    """Release an adapter the probe built, if it holds a resource."""
    close = getattr(adapter, "close", None)
    if callable(close):
        close()


def _adapter_type(value: object) -> str:
    return "{}.{}".format(
        getattr(value, "__module__", "?"),
        getattr(value, "__qualname__", type(value).__name__),
    )


def _slot_text(case: BindabilityCase) -> str:
    return "class {} key {} (group {!r})".format(
        case.entity_class, case.component, case.group,
    )


def _spec_name(index: int) -> str:
    return "bind_{}".format(index)


def _formula_text(case: BindabilityCase) -> str:
    return "abs({}_*[{}]) >= 0".format(case.entity_class, case.component)


def _denominator(
    catalog: ComponentCatalog,
    static_classes: Tuple[str, ...],
) -> Tuple[BindabilityCase, ...]:
    """The lexicographically first (entity, key) pair of every group."""
    cases: List[BindabilityCase] = []
    for entity_class in static_classes:
        first: Dict[str, Tuple[int, str]] = {}
        for entity_id in sorted(catalog.entity_ids_for_class(entity_class)):
            groups = catalog.component_groups_for_entity(entity_id)
            for key in sorted(catalog.entity_info(entity_id).components):
                group = groups.get(key) or ""
                candidate = (entity_id, key)
                if group not in first or candidate < first[group]:
                    first[group] = candidate
        for group in sorted(first):
            entity_id, key = first[group]
            cases.append(
                BindabilityCase(entity_class, group, key, entity_id)
            )
    return tuple(cases)


def _compile_case(
    case: BindabilityCase,
    catalog: ComponentCatalog,
    capabilities,
    dt: float,
) -> Tuple[Optional[str], str]:
    """Compile one slot the way ``specola check`` does; report the refusal.

    Returns the case's status and its reason. ``None`` is the status of a
    slot that compiles with an eligible entity. A ``SpecConfigError`` is
    the refusal an author would see, so it decides the case ``fail``;
    any other exception is the compile leg breaking rather than the slot
    being unbindable, and leaves the case ``unexecuted`` the way an
    unavailable runtime leg does.
    """
    formula_text = _formula_text(case)
    try:
        compiled = compile_specs(
            formula_text, dt=dt, capabilities=capabilities, catalog=catalog,
        )
    except SpecConfigError as exc:
        return "fail", "{} did not compile: {}".format(_slot_text(case), exc)
    except Exception as exc:
        return "unexecuted", "compile leg unavailable: {}: {}".format(
            type(exc).__name__, exc,
        )

    # Same eligibility question as check_specs._eligible_entities, for a
    # wildcard slot with no tag filter: does a discovered entity of the
    # class expose every component the slot requires?
    for formula in compiled:
        for param in formula.params:
            eligible = [
                entity_id
                for entity_id in catalog.entity_ids_for_class(
                    param.entity_class
                )
                if catalog.entity_info(entity_id).supports_all(
                    param.required_components
                )
            ]
            if not eligible:
                return "fail", "{} compiled with no eligible entity".format(
                    _slot_text(case),
                )
    return None, ""


def _run_bindings(
    adapter_factory: Callable[[], SimulatorAdapter],
    compiled: Dict[int, BindabilityCase],
    steps: int,
    seed: int,
) -> Tuple[frozenset, Optional[str]]:
    """Monitor every compiling case on a fresh adapter; report what bound."""
    names = {index: _spec_name(index) for index in compiled}
    document = {
        "specs": [
            {"name": names[index], "formula": _formula_text(case)}
            for index, case in compiled.items()
        ],
    }
    bound: set = set()
    adapter = None
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bindability.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            _seed(seed)
            adapter = adapter_factory()
            session = Monitor(specs=str(path), results_dir=None).watch(adapter)
            for _ in range(steps):
                result = session.step()
                bound.update(
                    name for name in names.values()
                    if result.rho(name) is not None
                )
            session.end()
    except Exception as exc:
        return frozenset(bound), "{}: {}".format(type(exc).__name__, exc)
    finally:
        _close(adapter)
    return frozenset(bound), None


def probe_for_registry_name(
    name: str,
    scene: Optional[str] = None,
    *,
    steps: int = 1,
    seed: int = 0,
) -> BindabilityArtifact:
    """Probe the adapter one registry name builds for ``scene``.

    Row drivers name their own subject: ``scene`` is the argument their
    factory receives, so the probe measures the configuration the row
    certifies. A registry lookup or a construction that raises comes back
    as ``setup_unavailable`` naming the registry entry, not the closure
    that called it.
    """

    def factory() -> SimulatorAdapter:
        return resolve(name).factory(scene, None, False)

    artifact = probe_static_slot_bindability(factory, steps=steps, seed=seed)
    if artifact.outcome == "setup_unavailable":
        return replace(artifact, adapter_type="registry:" + name)
    return artifact


def bindability_evidence(artifact: BindabilityArtifact) -> Dict[str, object]:
    """The raw-evidence record for one probe: its facts and their digest.

    The digest is taken over the same canonical facts the ledger
    references, so a row's obligation provenance and this record name one
    artifact.
    """
    facts = artifact.to_json_dict()
    canonical = json.dumps(
        facts, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return {
        "artifact": facts,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


_OUTCOMES = ("pass", "fail", "not_applicable", "setup_unavailable")
_STATUSES = ("pass", "fail", "unexecuted")


def validate_bindability_artifact(artifact: BindabilityArtifact) -> None:
    """Refuse a probe record the probe itself cannot emit.

    The exact frozen type is not internal consistency: a record can be
    built by hand with a ``pass`` outcome over no case, or over a case
    whose own result failed. A ledger that reads only ``outcome`` would
    certify it. Every consumer of the outcome validates first, so the
    outcome is never trusted beyond the results it summarizes.

    The outcome rules are the probe's own precedence: a proven refusal
    outranks an undecided case, so an artifact with one ``fail`` and one
    ``unexecuted`` result is ``fail``. ``setup_unavailable`` is therefore
    the outcome of a record whose only undecided-or-better results are
    undecided.

    Raises
    ------
    ConformanceValidationError
        If the record is not the exact artifact type, names an unknown
        outcome or status, does not carry one result per case in case
        order, or carries an outcome its results do not support.
    """
    if type(artifact) is not BindabilityArtifact:
        raise ConformanceValidationError(
            "bindability evidence needs the exact BindabilityArtifact type"
        )
    if artifact.outcome not in _OUTCOMES:
        raise ConformanceValidationError(
            "unknown bindability outcome: {!r}".format(artifact.outcome)
        )
    statuses = [result.status for result in artifact.results]
    unknown = [status for status in statuses if status not in _STATUSES]
    if unknown:
        raise ConformanceValidationError(
            "unknown bindability result status: {!r}".format(unknown[0])
        )
    if len(artifact.results) != len(artifact.cases):
        raise ConformanceValidationError(
            "bindability record has {} results for {} cases".format(
                len(artifact.results), len(artifact.cases),
            )
        )
    if any(
        result.case != case
        for result, case in zip(artifact.results, artifact.cases)
    ):
        raise ConformanceValidationError(
            "bindability results are not the cases in case order"
        )

    passed = bool(artifact.cases) and set(statuses) == {"pass"}
    failed = "fail" in statuses
    if (artifact.outcome == "pass") != passed:
        raise ConformanceValidationError(
            "bindability outcome 'pass' needs a case and an all-pass result set"
        )
    if (artifact.outcome == "fail") != failed:
        raise ConformanceValidationError(
            "bindability outcome 'fail' needs a failed result"
        )
    if artifact.outcome == "not_applicable" and artifact.cases:
        raise ConformanceValidationError(
            "bindability outcome 'not_applicable' needs an empty denominator"
        )
    if artifact.outcome == "setup_unavailable":
        if not artifact.detail:
            raise ConformanceValidationError(
                "bindability outcome 'setup_unavailable' needs its detail"
            )
        if any(status != "unexecuted" for status in statuses):
            raise ConformanceValidationError(
                "bindability outcome 'setup_unavailable' needs every case "
                "undecided"
            )
