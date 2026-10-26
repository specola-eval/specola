"""session.py — researcher-facing facade over MonitorEngine.

    monitor = Monitor("specs.yaml")
    session = monitor.watch(env, meta={"train_step": 30000, "task": t})
    for ...:
        env.step(action)                # the researcher's loop
        result = session.step()
        if result.violated("safety"): break
    session.end(success=env.check_success())

The facade owns no policy, env stepping, or threading; it reads
observations through the adapter and feeds MonitorEngine. See
docs/superpowers/specs/2026-07-16-researcher-eval-integration-design.md.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .adapter import SimulatorAdapter
from .capabilities import reset_adapter
from .engine import MonitorEngine
from .monitor import binding_to_dict
from .registry import wrap_env
from .results import EngineResult
from .query_config import QueryConfiguration
from .results_io import (
    RESERVED_META_KEYS,
    episode_filename,
    save_rows,
)

_LOG = logging.getLogger(__name__)


def _binding_label(binding, params, entity_labels: dict) -> str:
    """Format a binding using short entity names.

    For objects with a category label stored as 'body_name  (category)',
    shows just the category (e.g. 'bread', 'can', 'jar').
    For all other entities (robot, gripper, region) uses the full label.
    Falls back to e{id} if no label exists.

    Example: 'bread  ·  target_cabinet_main_group'
    """
    d = binding_to_dict(binding, params)
    parts = []
    for p in params:
        eid   = d.get(p.param_key, "?")
        label = entity_labels.get(eid, f"e{eid}")
        # "obj_main  (bread)" → "bread"
        if "  (" in label and label.endswith(")"):
            label = label.split("  (", 1)[1][:-1]
        parts.append(label)
    return "  ·  ".join(parts)


def _binding_key(binding) -> str:
    """A stable, injective identity for a binding within its spec.

    The raw binding is a tuple of entity IDs aligned with the formula's
    parameter slots; two entities of the same category can collapse to the
    same human-readable ``_binding_label`` (e.g. mujoco's
    ``"body  (category)"`` labels both strip to the category), but they
    never share an entity-ID tuple. Keying the last-final verdict on this
    identity — rather than on the display label — keeps a failing binding
    from being absorbed by a co-labeled passing one.

    A parameter-free formula yields the single empty binding and an empty
    key, which is correct: there is only one binding to key.
    """
    return "_".join(str(entity_id) for entity_id in binding)


class _LiveEntityLabels(dict):
    """Entity labels that follow the adapter's population.

    ``adapter.entity_labels`` is a point-in-time snapshot, and dynamic
    populations (CARLA actors, contact pairs) gain members after the
    run starts — a static copy captured at reset displays every later
    arrival as ``e<id>``. A lookup miss re-reads the adapter, so an
    entity's real label appears the step the entity does. Adapters
    whose labels are a plain static dict just refresh to the same
    contents.
    """

    def __init__(self, adapter) -> None:
        super().__init__()
        self._adapter = adapter
        self._refresh()

    def _refresh(self) -> None:
        fresh = getattr(self._adapter, "entity_labels", None) or {}
        if fresh:
            self.update(fresh)

    def get(self, key, default=None):
        if key not in self:
            self._refresh()
        return super().get(key, default)

    def __missing__(self, key):
        self._refresh()
        if key in self:
            return dict.__getitem__(self, key)
        raise KeyError(key)


def _validate_meta(meta: Optional[dict]) -> Dict[str, Any]:
    """Validate a caller-supplied meta dict: flat, str keys, scalar values.

    Keys colliding with ``results_io.RESERVED_META_KEYS`` are rejected here,
    at watch() time, rather than surfacing later as a confusing overwrite.
    That reserved set covers the row schema's own columns
    (``CORE_COLUMNS``), every column ``report.py`` derives at aggregation
    time (``DERIVED_REPORT_COLUMNS``) — a meta key named e.g. ``"verdict"``
    would otherwise pass validation but corrupt the report — and the
    spec-set identity attached to every results file
    (``PROVENANCE_KEYS``), which must describe what actually produced the
    rows rather than whatever a caller passed.
    """
    if meta is None:
        return {}
    if not isinstance(meta, dict):
        raise ValueError(f"meta must be a dict, got {type(meta).__name__}")
    validated: Dict[str, Any] = {}
    for key, value in meta.items():
        if not isinstance(key, str):
            raise ValueError(
                f"meta keys must be str, got {type(key).__name__} ({key!r})"
            )
        if key in RESERVED_META_KEYS:
            raise ValueError(
                f"meta key {key!r} collides with a reserved results column "
                f"({sorted(RESERVED_META_KEYS)})"
            )
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                f"meta value for key {key!r} must be str/int/float/bool, "
                f"got {type(value).__name__}"
            )
        validated[key] = value
    return validated


@dataclass(frozen=True)
class Violation:
    spec: str      # authored spec name
    monitor: str   # manager key (== spec unless fan-out)
    binding: str   # human-readable label
    rho: float
    step: int
    time: float


@dataclass(frozen=True)
class StepResult:
    raw: EngineResult
    violations: Tuple[Violation, ...]
    _rho_by_spec: Dict[str, float]          # min robustness per authored spec

    def violated(self, spec_name: Optional[str] = None) -> bool:
        if spec_name is None:
            return bool(self.violations)
        return any(v.spec == spec_name for v in self.violations)

    def rho(self, spec_name: str) -> Optional[float]:
        return self._rho_by_spec.get(spec_name)


@dataclass(frozen=True)
class SpecOutcome:
    verdict: str          # "pass" | "fail" | "undetermined" | "not_applicable"
    worst_rho: Optional[float]
    first_violation_step: Optional[int]


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    status: str
    success: Optional[bool]
    steps: int
    meta: Dict[str, Any]
    specs: Dict[str, SpecOutcome]
    path: Optional[Path]


class Monitor:
    """Parse specs once; hand out one Session per episode."""

    def __init__(
        self,
        specs: str,
        results_dir: Optional[str] = "results",
        run_id: Optional[str] = None,
    ) -> None:
        # Fail fast: a spec YAML typo dies here, before any rollout starts.
        self._engine = MonitorEngine.load(specs)
        self._results_dir = results_dir
        # A bare second-resolution timestamp collides when two Monitors are
        # constructed in the same second (parallel sweeps, or a per-task
        # Monitor built in a loop). Salt the default run_id with random hex
        # so distinct Monitors never share one.
        self._run_id = run_id or "{ts}_{salt}".format(
            ts=datetime.now().strftime("%Y%m%d_%H%M%S"),
            salt=secrets.token_hex(2),
        )
        self._episode_index = 0
        self._session: Optional["Session"] = None

    def watch(
        self,
        env_or_adapter: Any,
        meta: Optional[dict] = None,
        dt: Optional[float] = None,
        _phase_hook: Optional[Callable[[str], None]] = None,
        _reset_hook: Optional[Callable[[Any, Any], None]] = None,
        queries: Optional[QueryConfiguration] = None,
        **adapter_kwargs: Any,
    ) -> "Session":
        """Reset the adapter and compile specs to start a new episode.

        ``dt`` is a researcher override for the spec-evaluation timestep
        (e.g. to match a recorded demo's control rate rather than the live
        adapter's). Defaults to the adapter's own ``dt`` when omitted.

        ``queries`` installs the immutable native query subject before reset.
        Omission retains a configuration already installed on the adapter.

        ``_reset_hook`` is private and stays private: it hands the CLI
        the exact reset the episode compiles against, which is the only
        moment a durable description of *this* scene can be taken. It is
        not a place to influence the episode — the reset has already
        happened when it runs, and raising from it is the caller's own
        problem to contain.
        """
        from .conformance.verification import current_verification
        from .conformance.provider import ExtractionGuardError

        verification = current_verification()
        if verification is not None and not isinstance(env_or_adapter, SimulatorAdapter):
            raise ExtractionGuardError("verification_subject_mismatch", ("candidate monitoring requires the declared adapter instance",))
        if isinstance(env_or_adapter, SimulatorAdapter):
            if adapter_kwargs:
                raise TypeError(
                    "watch() was given an already-constructed SimulatorAdapter; "
                    "adapter_kwargs are only accepted when passing a raw env "
                    f"for auto-detection (got {sorted(adapter_kwargs)})"
                )
            adapter = env_or_adapter
        else:
            adapter = wrap_env(env_or_adapter, **adapter_kwargs)

        validated_meta = _validate_meta(meta)
        if verification is not None:
            current_verification(adapter)
            if self._results_dir is not None or _reset_hook is not None:
                raise ExtractionGuardError("verification_only", ("candidate monitoring requires results_dir=None and no publication hook",))
            subject = verification.subject
            if ((queries is not None and queries != subject.queries)
                    or (queries is None and adapter.query_configuration != subject.queries)):
                raise ExtractionGuardError("verification_subject_mismatch", ("monitor query configuration differs from candidate subject",))
            if (self._session is not None and not self._session.ended
                    and self._session._verification_scope is not verification):
                raise ExtractionGuardError("verification_scope_changed", ("candidate monitoring cannot truncate a session from another scope",))
            validated_meta["verification_only"] = True
        if queries is not None:
            adapter.set_query_configuration(queries)

        # A break-heavy eval loop must not be able to wedge the monitor:
        # an unended previous session is truncated, not silently dropped.
        if self._session is not None and not self._session.ended:
            self._session.end(status="truncated")

        if _phase_hook is not None:
            _phase_hook("adapter reset and scene discovery")
        # Reset discovers the episode's full component catalog, which is
        # what start_episode() compiles against — so the previous
        # episode's scope has to come off first, or the second episode of
        # a session would compile against a catalog its own scope had
        # already narrowed.
        adapter.set_extraction_scope(None)
        # Catalog-only reset (issue #136, maintainer decision 2026-10-07):
        # no compiled slot exists yet, so an entity scope admitting nothing
        # keeps the MuJoCo family from reading its geom pairs here; the
        # catalog still lists every pair (reset_catalog) and the first step
        # reads the scoped ones. Until 2026-10-13 a reset hook forced the
        # full reset so the dashboard's discovery document could carry
        # every pair; on RoboCasa that read 365,423 entities and cost about
        # 120 s of startup (maintainer decision 2026-10-13: the document
        # describes the catalog-only reset, pairs unread).
        adapter.set_extraction_entity_scope({})
        try:
            reset = reset_adapter(adapter)
        finally:
            # The empty scope serves this reset only: a watch that fails
            # from here on leaves the adapter unscoped, as before.
            adapter.set_extraction_entity_scope(None)
        observed = adapter.last_observation
        if ((verification is not None and (observed is None or observed.extraction is None))
                or (observed is not None and observed.extraction is not None
                    and observed.extraction.verification_only != (verification is not None))):
            raise ExtractionGuardError("verification_mode_mismatch", ("reset does not match the monitor execution mode",))
        if _reset_hook is not None:
            # Here, and nowhere else. This is the reset the episode
            # compiles against; a reset the caller performed earlier can
            # describe a different scene entirely — RoboCasa randomizes
            # its kitchen per reset, so a snapshot taken there would name
            # objects this episode never spawns. It also runs *before*
            # the extraction scope is installed below, so it sees every
            # entity the catalog-only reset perceived rather than the
            # narrowed set the compiled specs need.
            _reset_hook(adapter, reset)
        if _phase_hook is not None:
            _phase_hook("specification compilation")
        self._engine.start_episode(
            dt=dt if dt is not None else adapter.dt,
            capabilities=reset.capabilities,
            catalog=reset.discovered_catalog,
            skip_unsatisfiable=True,
        )
        # Monitoring is the one mode that knows what it needs: from here
        # the adapter extracts the compiled specs' components and
        # nothing else. Every other entrypoint (discover, check,
        # write-specs, explore) leaves the scope unset and still sees the
        # full surface.
        adapter.set_extraction_scope(self._engine.extraction_scope())
        adapter.set_extraction_entity_scope(self._engine.extraction_entity_scope())

        episode_index = self._episode_index
        self._episode_index += 1
        # Salt the episode_id too, so even callers that pass an explicit
        # (possibly same-second) run_id — specola run, or two parallel
        # invocations sharing a timestamped run_id — never produce two
        # episodes with the same id that report.episode_table would merge.
        episode_id = f"{self._run_id}_ep{episode_index}_{secrets.token_hex(2)}"

        path: Optional[Path] = None
        if self._results_dir is not None:
            path = (
                Path(self._results_dir)
                / self._run_id
                / episode_filename(episode_index)
            )

        entity_labels = _LiveEntityLabels(adapter)

        session = Session(
            self._engine,
            adapter,
            meta=validated_meta,
            episode_id=episode_id,
            path=path,
            entity_labels=entity_labels,
            phase_hook=_phase_hook,
        )
        self._session = session
        return session


class Session:
    """One monitored episode. Created by Monitor.watch(), never directly."""

    def __init__(
        self,
        engine: MonitorEngine,
        adapter: SimulatorAdapter,
        *,
        meta: Dict[str, Any],
        episode_id: str,
        path: Optional[Path],
        entity_labels: Optional[dict] = None,
        phase_hook: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.engine = engine        # power-user escape hatch
        self.adapter = adapter
        from .conformance.verification import current_verification, require_ordinary_execution
        self._verification_scope = current_verification(adapter)
        if self._verification_scope is None:
            require_ordinary_execution(meta)
        if self._verification_scope is not None and (path is not None or meta.get("verification_only") is not True):
            from .conformance.provider import ExtractionGuardError
            raise ExtractionGuardError("verification_only", ("candidate session must retain its mode and remain in memory",))
        from .conformance.provider import adapter_extraction_status
        self._extraction_status = adapter_extraction_status(adapter)
        self._meta = meta
        self._episode_id = episode_id
        self._path = path
        # ``is not None`` rather than truthiness: a live label view may
        # be empty at reset and fill as the population arrives.
        self._entity_labels = (
            entity_labels if entity_labels is not None else {}
        )
        # Snapshot this episode's compile-time results; the shared engine
        # gets recompiled by the next watch() call, so these must not be
        # re-read from self.engine after that point.
        self._manager_specs = engine.manager_specs()
        self._skipped = engine.skipped_specs()
        self._rows: List[dict] = []
        self._next_step = 0
        self._ended = False
        self._record: Optional[EpisodeRecord] = None
        self._phase_hook = phase_hook

    @property
    def ended(self) -> bool:
        return self._ended

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._ended:
            if exc_type is not None:
                self.end(status="truncated")
            else:
                self.end()
        return False  # never suppress the caller's exception

    def step(self) -> StepResult:
        if self._ended:
            raise RuntimeError(
                "Session already ended; call Monitor.watch() to start a "
                "new episode before stepping again."
            )

        from .conformance.verification import current_verification
        if current_verification(self.adapter) is not self._verification_scope:
            from .conformance.provider import ExtractionGuardError
            raise ExtractionGuardError("verification_scope_changed", ("session cannot leave its original execution scope",))
        obs = self.adapter.get_observation()
        raw = self.engine.update(obs)

        step_index = self._next_step
        self._next_step += 1

        violations: List[Violation] = []
        rho_by_spec: Dict[str, float] = {}

        for ev in raw.evaluations:
            manager_key = ev.spec_name
            authored = self._manager_specs.get(manager_key, manager_key)
            manager = self.engine.managers[manager_key]
            params = manager.formula.params
            label = _binding_label(ev.binding, params, self._entity_labels)
            rho_value = ev.robustness

            self._rows.append({
                "step":          step_index,
                "time":          round(float(obs.time), 6),
                "spec":          manager_key,
                "authored_spec": authored,
                "binding":       label,
                "binding_key":   _binding_key(ev.binding),
                "rho":           float(rho_value) if rho_value is not None else float("nan"),
                "suspended":     rho_value is None,
                "verdict_final": ev.verdict_final,
                "window_steps":  manager.formula.window_steps,
                "not_applicable": False,
            })

            if rho_value is not None:
                current = rho_by_spec.get(authored)
                rho_by_spec[authored] = (
                    rho_value if current is None else min(current, rho_value)
                )

                if ev.verdict_final and rho_value < 0:
                    violations.append(Violation(
                        spec=authored,
                        monitor=manager_key,
                        binding=label,
                        rho=float(rho_value),
                        step=step_index,
                        time=float(obs.time),
                    ))

        return StepResult(
            raw=raw, violations=tuple(violations), _rho_by_spec=rho_by_spec,
        )

    def end(
        self, success: Optional[bool] = None, status: str = "completed",
    ) -> EpisodeRecord:
        from .conformance.verification import current_verification, require_ordinary_execution
        execution = current_verification(self.adapter)
        if self._verification_scope is None:
            require_ordinary_execution(self._meta)
        elif execution is not None and execution is not self._verification_scope:
            from .conformance.provider import ExtractionGuardError
            raise ExtractionGuardError("verification_scope_changed", ("session cannot finalize in another candidate scope",))
        if self._ended:
            assert self._record is not None
            return self._record
        self._ended = True

        if self._phase_hook is not None:
            self._phase_hook("result finalization")
        specs = self._compute_outcomes()
        # A slot that never had a candidate entity usually means a typo'd
        # class or component; the episode record alone reports it as
        # undetermined, which reads like a monitor that ran. An episode
        # with no steps has nothing to say about matching, so it is
        # silent.
        unmatched = (
            self.engine.unmatched_slots() if self._next_step else {}
        )
        for name, params in unmatched.items():
            keys = [parameter.param_key for parameter in params]
            # A logger, not warnings: the default warning filter shows one
            # warning per call site, which would report the first episode
            # of a campaign loop and silence every later one.
            _LOG.warning(
                "spec %s: parameter slot(s) %s matched no entity at any "
                "step", name, keys,
            )
        record = EpisodeRecord(
            episode_id=self._episode_id,
            status=status,
            success=success,
            steps=self._next_step,
            meta=dict(self._meta),
            specs=specs,
            path=self._path,
        )
        # Store before writing: a write failure must not destroy the
        # in-memory record the researcher already has a reference to.
        self._record = record

        if self._path is not None:
            if self._phase_hook is not None:
                self._phase_hook("result write")
            full_rows = self._build_full_rows(status=status, success=success)
            schema_meta: Dict[str, Any] = {
                "episode_id": self._episode_id,
                "status": status,
                "success": success,
                "steps": self._next_step,
            }
            # Which spec set produced these rows. Caller meta cannot
            # shadow these keys (results_io.PROVENANCE_KEYS is reserved),
            # so the update below cannot rewrite provenance.
            schema_meta.update(self.engine.spec_provenance)
            if self._extraction_status is not None:
                schema_meta["extraction_status"] = self._extraction_status
            schema_meta.update(self._meta)

            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                written = save_rows(
                    full_rows, self._path, schema_meta, quiet=True,
                )
            except Exception:
                # The record survives the failure (stored above), but it
                # must not survive holding a path to a file that was
                # never created: a caller reading record.path would open
                # nothing, or a stale file left by an earlier run.
                self._record = replace(self._record, path=None)
                raise

            # save_rows returns None when it wrote nothing (an empty row
            # set on the CSV fallback), which is exactly what the record
            # should carry: path names the results file this episode
            # wrote, or it is None. Trusting the returned path rather
            # than probing the filesystem matters — an earlier episode's
            # file could otherwise occupy that name and be claimed here.
            self._record = replace(
                self._record,
                path=written,
            )

        return self._record

    def _build_full_rows(
        self, *, status: str, success: Optional[bool],
    ) -> List[dict]:
        rows = list(self._rows)
        # A spec that compiled but never bound any entity produces no
        # evaluation rows at all, which made it vanish from the report
        # entirely while the CLI and the Session record both said
        # undetermined (X1 consistency suite). One marker row keeps it
        # visible: non-final, no robustness, not not_applicable, so the
        # report derives undetermined rather than nothing.
        seen = {row["authored_spec"] for row in rows}
        # engine.spec_names is set at load and never mutated by
        # start_episode, so reading it here does not race the recompile
        # that watch() performs between sessions.
        for name in self.engine.spec_names:
            if name in seen or name in self._skipped:
                continue
            rows.append({
                # Last real step index, so episode_table's
                # steps = max(step) + 1 stays honest; -1 only when the
                # episode had no steps at all.
                "step":          max(self._next_step - 1, -1),
                "time":          float("nan"),
                "spec":          name,
                "authored_spec": name,
                "binding":       "",
                "binding_key":   "",
                "rho":           float("nan"),
                "suspended":     False,
                "verdict_final": False,
                "window_steps":  0,
                "not_applicable": False,
            })
        for name in self._skipped:
            rows.append({
                "step":          -1,
                "time":          float("nan"),
                "spec":          name,
                "authored_spec": name,
                "binding":       "",
                "binding_key":   "",
                "rho":           float("nan"),
                "suspended":     False,
                "verdict_final": False,
                "window_steps":  0,
                "not_applicable": True,
            })

        stamped = []
        for row in rows:
            new_row = dict(row)
            new_row["episode_id"] = self._episode_id
            new_row["status"] = status
            new_row["success"] = success
            new_row.update(self._meta)
            stamped.append(new_row)
        return stamped

    def _compute_outcomes(self) -> Dict[str, SpecOutcome]:
        # Per (authored spec, monitor, binding_key): the last row whose
        # verdict was final. Per-episode verdict is decided from that set,
        # not from every row seen — a binding that recovered from an early
        # dip still counts as "pass" if its last final rho is >= 0. The key
        # is binding_key (the entity-ID identity), NOT the display binding
        # label: two same-category entities can share a label, and keying on
        # the label would let a passing binding absorb a co-labeled failing
        # one (report.episode_table keys on binding_key for the same reason).
        last_final: Dict[str, Dict[Tuple[str, str], dict]] = {}
        # first_violation_step scans every final row (not just the last
        # per binding), since a diagnosis needs to know a violation ever
        # happened even if that binding later recovered.
        first_violation: Dict[str, int] = {}

        for row in self._rows:
            if not row["verdict_final"]:
                continue
            authored = row["authored_spec"]
            key = (row["spec"], row["binding_key"])
            bucket = last_final.setdefault(authored, {})
            existing = bucket.get(key)
            if existing is None or row["step"] > existing["step"]:
                bucket[key] = row
            if row["rho"] < 0:
                if authored not in first_violation or row["step"] < first_violation[authored]:
                    first_violation[authored] = row["step"]

        outcomes: Dict[str, SpecOutcome] = {}
        for name in self.engine.spec_names:
            if name in self._skipped:
                outcomes[name] = SpecOutcome(
                    verdict="not_applicable",
                    worst_rho=None,
                    first_violation_step=None,
                )
                continue

            bucket = last_final.get(name, {})
            rhos = [r["rho"] for r in bucket.values()]
            if any(r < 0 for r in rhos):
                verdict = "fail"
            elif rhos:
                verdict = "pass"
            else:
                verdict = "undetermined"

            outcomes[name] = SpecOutcome(
                verdict=verdict,
                worst_rho=min(rhos) if rhos else None,
                first_violation_step=first_violation.get(name),
            )
        return outcomes
