"""Native geometry query components pending adapter-row activation.

These readers preserve native results while the full effect audit remains
pending. They do not grant extraction completeness or replace the engine
inventory guard. Plugin and shared-thread-pool isolation is unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack
import hashlib
from importlib import metadata
import math
from pathlib import Path
import sys
from typing import Iterator, Optional, Tuple

import numpy as np

from ..native_values import NativeValue
from ..query_config import QueryConfiguration
from ..conformance.core_runner import EvidenceIdentity
from ..conformance.query_bindings import (
    ArgumentSchema, CallableParameterSurface, ParameterRole, QueryBindingError,
    QueryBindingRecord, QueryParameter, bind_query_configuration,
    validate_query_arguments,
)
from .mujoco_native import MujocoNativeGuard
from .mujoco_scratch import MujocoScratchSession


# Source-reviewed implementation custody, not an engine member allowlist.
# Other producer builds remain unresolved until their effect contracts are
# reconciled. The producer input record retains the source references.
#
# One row per producer version: the installed introspection files and
# runtime library, and the two engine sources the contract was read from at
# that release's source commit. A row is added only after mj_geomDistance
# and the mj_copyData path at that commit are reconciled against the
# reviewed 3.8.1 sources. The reconciliations are recorded in
# docs/superpowers/specs/2026-10-02-mujoco-query-custody-3.10.0.md and
# docs/superpowers/specs/2026-10-02-mujoco-query-custody-3.3.md.
#
# "clamps" says whether the build bounds a beyond-cutoff return at distmax.
# 3.3.0 does not on the native CCD path: it returns a sentinel, which the
# binding declares as native_above_cutoff for non-clamping builds only.
#
# 3.3.1 is admitted with two recorded limits (maintainer decision
# 2026-10-02, option B of the 3.3 custody note). Its convex solver keeps
# scratch in five thread-local arrays that the scratch copy does not
# isolate; that is accepted on measured evidence, not proven. And its
# mj_geomDistance path calls an unset allocator when ccd_iterations exceeds
# the solver's static capacity, so "ccd_iteration_limit" refuses such a
# model at bind and at every read.
_REVIEWED = {
    "3.8.1": {
        "source_commit": "688209574d35555fb897be5795f1ce4ecca9f29a",
        "functions": "9f4f952b79f1c643d6579430eaa57cdaf054559ad0f6b5b0745178101b4ba452",
        "structs": "e5c01aea7e41cd6b36b8d3330eba7b1c702156a71b531471391b6e48e4394b5c",
        "library": "8969010a8325d32b9b154159862ffe3b112277cc993983a0168f7dbe2efe0592",
        "support": "b8c9571ea7d6a220e52795494e5af705b180081af35314666983de6e1d2adf8b",
        "copy": "c214eaae5f42e2245641f5f8bb7c4e1fc9ec2420e48a4aac8c821474cd35c294",
        "clamps": True,
    },
    "3.10.0": {
        "source_commit": "28009f9105cd92784b7b0b30c0605a5e29107a77",
        "functions": "141e0e6653b2a5b6ed0d3392e575bd0f97c8b52fae26bb023dcad42591222935",
        "structs": "71e3db74abac2cc2fb87ac59937700bcbbe2f008a5e166faf63e6f204044ebeb",
        "library": "872b2954e9760b1df122e3c807a907ff806e1980099b00e15789ae67e08b75d5",
        "support": "a14f4e15241ae6a84350fdc12ca9ac09facc6fb0b9e1936c3e1336501554bea7",
        "copy": "701de2290a35e538bddb6e032cfa3715b673397b654e6ec713be304ddaaefcf0",
        "clamps": True,
    },
    "3.3.0": {
        "source_commit": "8de9b4e4e373caf01035c6c13efd363347d89032",
        "functions": "e14d57716421e404526e82c3f313902e172e69fbb80e2ceda61d9f3b456dfe88",
        "structs": "82f2f2b92d43b786bcc9e285c281b127cdfa391ad6d5f3e614dfa5d88e1eb1a2",
        "library": "f37b05c836fd6e35ec799f144af65ff0b98eda29c3bbd049981ce8dbc29dd136",
        "support": "0a85e97a33bcb2e1f28cbb6bdd1f9430eb72f0a3ce34194931df63e9a65d44d0",
        "copy": "894d7a551db4cf9bf0cf3473aa0b3cc594f287ab7b002727972d1111852ac862",
        "clamps": False,
    },
    "3.3.1": {
        "source_commit": "a209134315dda126a714529bfa80f7478902644d",
        "functions": "f4041c253930d9bdf57875bdf29ee35cce8255a37422d45c59a906930b517aa7",
        "structs": "a78a5ea7b9c20d133cb9002430576f28016af506e93d1aa7909105cd1d6ff98b",
        "library": "e5ad1f81436875347cb3bd9d824f5dafe434c220e8a5ebcf3c6208baeac06059",
        "support": "5fabdb33646c691636808a7e3445483acc50543b9807b10749dd34d098dd0f76",
        "copy": "f8eabe5f92b43d145e010f8656475c67c6cee1ca69d314cf8547084e33ba348f",
        "clamps": False,
        # mjMAX_EPA_ITERATIONS in engine_collision_gjk.h at the source commit.
        "ccd_iteration_limit": 170,
    },
}
_BINDING = "mujoco.mj_geomDistance"
# The scalar keys geom_distance_capped and geom_distance_clipped are
# declared projections of the return quantity, not quantities of their own.
_QUANTITIES = (_BINDING + ".return", _BINDING + ".fromto")


def reviewed_custody(version: str) -> dict:
    """Return the custody row for a producer version, or refuse it by name."""
    try:
        return _REVIEWED[version]
    except KeyError:
        raise QueryBindingError(
            "unresolved MuJoCo query effect contract for installed version "
            + repr(version) + "; reviewed versions: " + ", ".join(sorted(_REVIEWED))
        ) from None


def check_ccd_iterations(version: str, iterations: int) -> None:
    """Refuse a solver iteration count the build cannot run safely.

    On a build with a ``ccd_iteration_limit``, a count above the limit
    makes the native distance path call an allocator it never set.
    """
    limit = reviewed_custody(version).get("ccd_iteration_limit")
    if limit is not None and iterations > limit:
        raise QueryBindingError(
            "MuJoCo " + version + " cannot run mj_geomDistance with ccd_iterations="
            + str(iterations) + ": above " + str(limit)
            + " its convex solver calls an unset allocator; lower opt.ccd_iterations"
        )


def validate_mujoco_arguments(binding, arguments):
    """Reject arguments the declared native scalar representation cannot hold.

    JSON Schema permits an integral floating-point value as an integer. The
    producer's integer binding requires an integer object. Floating-point
    arguments retain the producer's finite rounding, including signed zero.
    """
    schema = binding.arguments.document()
    for name, value in arguments.items():
        kind = schema["properties"][name].get("type")
        if kind == "integer":
            if type(value) is not int or not np.iinfo(np.int32).min <= value <= np.iinfo(np.int32).max:
                raise QueryBindingError(binding.binding_id + ": native integer query argument is not representable")
        elif kind == "number":
            try:
                converted = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise QueryBindingError(binding.binding_id + ": native floating-point query argument is not representable") from error
            if not math.isfinite(converted):
                raise QueryBindingError(binding.binding_id + ": native floating-point query argument must be finite")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_library_path(distribution) -> Path:
    """Resolve the runtime library using producer-owned package inventory."""
    # Runtime patterns from MuJoCo python/setup.py at the reviewed 3.8.1 source commit; the 3.10.0 wheel matches the same pattern.
    if sys.platform == "win32":
        pattern = "mujoco.dll"
    elif sys.platform == "darwin":
        pattern = "libmujoco.*.dylib"
    else:
        pattern = "libmujoco.so.*"
    if distribution.files is None:
        raise QueryBindingError("MuJoCo distribution has no file inventory")
    candidates = [Path(distribution.locate_file(item)).resolve() for item in distribution.files
                  if item.parent.as_posix() == "mujoco" and item.match(pattern)]
    if len(candidates) != 1 or not candidates[0].is_file():
        raise QueryBindingError("MuJoCo runtime library inventory is absent or ambiguous")
    return candidates[0]


def _binding_record(evidence: Tuple[EvidenceIdentity, ...]):
    schema = ArgumentSchema.from_document({
        "type": "object", "additionalProperties": False,
        "properties": {"distmax": {"type": "number"}},
        "required": ["distmax"],
    })
    # ``population: all`` is population-only (issue #136): it keeps the whole
    # pair universe under a specification scope and never reaches the call.
    population_schema = ArgumentSchema.from_document({
        "type": "object", "additionalProperties": False,
        "properties": {"distmax": {"type": "number", "exclusiveMinimum": 0},
                       "population": {"const": "all"}},
        "required": ["distmax"],
    })
    # Parameter roles are native semantic mappings. Their population and order
    # are checked against the independent producer declaration by the binder.
    roles = (
        QueryParameter("m", ParameterRole.CONTEXT, "const mjModel*"),
        QueryParameter("d", ParameterRole.CONTEXT, "mjData*"),
        QueryParameter("geom1", ParameterRole.OWNER, "int", owner_kind="geom"),
        QueryParameter("geom2", ParameterRole.OWNER, "int", owner_kind="geom"),
        QueryParameter("distmax", ParameterRole.ARGUMENT, "mjtNum"),
        QueryParameter("fromto", ParameterRole.OUTPUT_BUFFER, "nullable mjtNum array"),
    )
    return QueryBindingRecord(
        _BINDING, "mujoco.c.functions.mj_geomDistance", roles, schema, _QUANTITIES,
        "native_geom_name_or_id", "isolated_mj_geomDistance",
        "caller supplies current kinematics at the observation instant",
        "distance helper and copy path reviewed; collider bodies unreviewed beyond executed sphere and box cases",
        "retain scalar return and fromto; add min(return,distmax) and return>=distmax; "
        "a pair whose bounding-sphere lower bound reaches distmax is not executed and reports "
        "capped=distmax, clipped=1 without a native return",
        "clipped is the return>=distmax projection; it does not prove native clamping or witness absence; "
        "inferred_above_cutoff rests on |xpos_i-xpos_j|-rbound_i-rbound_j>=distmax, not on a native call",
        evidence, completion_dependencies=("mujoco.c.functions.mj_copyData",),
        population_resolver_id="collision_filtered_native_geom_pairs_bound_pruned",
        population_arguments=population_schema,
    )


@dataclass(frozen=True)
class GeomDistanceResult:
    """Native output plus declared capped projections, not exact geometry.

    A pruned pair (``executed`` false) has no native return or witness. Its
    bounding-sphere lower bound reached the cutoff, so the capped projection
    is the cutoff and the clipped indicator is one, the same values the native
    clamp reports on the reviewed 3.8.1 and 3.10.0 builds for a pair beyond the cutoff.
    """

    geom_ids: Tuple[int, int]
    body_ids: Tuple[int, int]
    distmax: float
    native_return: Optional[float]
    fromto: Optional[NativeValue]
    options: NativeValue
    collider_available: bool
    instance_token: str = ""
    binding_id: str = ""
    warnings: Tuple[bytes, ...] = ()
    completion_warnings: Tuple[bytes, ...] = ()
    executed: bool = True
    lower_bound: Optional[float] = None

    def _native(self) -> Optional[float]:
        """The executed finite native return, or None for any other case."""
        returned, witness = self.native_return, self.fromto
        if not self.executed or returned is None or witness is None:
            return None
        if not (math.isfinite(returned) and np.all(np.isfinite(witness.restore()))):
            return None
        return returned

    @property
    def finite(self) -> bool:
        return self._native() is not None

    @property
    def capped(self) -> Optional[float]:
        if not self.executed:
            return self.distmax
        returned = self._native()
        return None if returned is None else min(returned, self.distmax)

    @property
    def clipped(self) -> Optional[float]:
        if not self.executed:
            return 1.0
        returned = self._native()
        return None if returned is None else float(returned >= self.distmax)

    @property
    def outcome(self) -> str:
        if not self.executed:
            return "inferred_above_cutoff"
        returned = self._native()
        if returned is None:
            return "unresolved_nonfinite_native_result"
        if not self.collider_available:
            return "unsupported_collider_native_cutoff"
        if returned > self.distmax:
            return "native_above_cutoff"
        return "native_cutoff" if returned >= self.distmax else "native_below_cutoff"


class IsolatedGeomDistanceReader:
    """Resolve a native subject and execute queries on separate data buffers.

    This implementation has source custody for the builds recorded in _REVIEWED.
    Source custody is not all-collider numerical evidence or row readiness.
    Caller code must enforce the producer inventory and complete row gate.
    Collider-specific effects beyond the recorded fixtures remain unverified.
    """

    def __init__(self, model, data, configuration: QueryConfiguration, *,
                 _guard=None, _session=None, _defer_configuration=False):
        self._held = ExitStack()
        self._closed = False
        self._guard = None
        self._session = None
        try:
            self._initialize(model, data, configuration, _guard, _session, _defer_configuration)
        except BaseException:
            self.close()
            raise

    def _initialize(self, model, data, configuration, guard, session, defer_configuration):
        import mujoco
        from mujoco.introspect import functions, structs

        distribution = metadata.distribution("mujoco")
        if mujoco.__version__ != distribution.version:
            raise QueryBindingError("MuJoCo module and distribution version differ")
        reviewed = reviewed_custody(distribution.version)
        source = "mujoco@" + reviewed["source_commit"]
        self.clamps_beyond_cutoff = reviewed["clamps"]
        self._version = distribution.version
        if type(model) is not mujoco.MjModel or type(data) is not mujoco.MjData:
            raise QueryBindingError("query reader needs native MuJoCo model and data")
        library_path = _native_library_path(distribution)
        if (_digest(Path(functions.__file__)) != reviewed["functions"]
                or _digest(Path(structs.__file__)) != reviewed["structs"]
                or _digest(library_path) != reviewed["library"]):
            raise QueryBindingError("MuJoCo query implementation custody differs")
        self._mujoco = mujoco
        self._model = model
        self._geom_types = np.array(model.geom_type, copy=True)
        self._body_ids = np.array(model.geom_bodyid, copy=True)
        self._dimension = int(mujoco.mjtGeom.mjNGEOMTYPES)
        # The pair universe is a function of these model arrays only; the
        # per-read prune reads current geom positions from the scratch copy.
        self._population_arrays = {
            name: np.array(getattr(model, name), copy=True)
            for name in ("geom_contype", "geom_conaffinity", "geom_rbound", "body_weldid", "body_parentid")
        }
        self._unbounded = np.isin(self._geom_types, [int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD)])
        declaration = functions.FUNCTIONS["mj_geomDistance"]
        self._buffer_shape = next(parameter.type.extents for parameter in declaration.parameters if parameter.name == "fromto")
        self._option_fields = tuple(item.name for item in structs.STRUCTS["mjOption"].fields)
        evidence = (
            EvidenceIdentity("mujoco/introspect/functions.py", reviewed["functions"]),
            EvidenceIdentity(source + "/src/engine/engine_support.c", reviewed["support"]),
            EvidenceIdentity(source + "/src/engine/engine_io.c", reviewed["copy"]),
        )
        self.binding = _binding_record(evidence)
        self.declaration = CallableParameterSurface(self.binding.callable_id, tuple(item.name for item in declaration.parameters), (evidence[0],))
        if not defer_configuration:
            validate_query_arguments(configuration, (self.binding,), (self.declaration,), _QUANTITIES,
                                     validate_native_arguments=validate_mujoco_arguments)
        self._guard = guard if guard is not None else self._held.enter_context(MujocoNativeGuard(mujoco))
        self._session = session if session is not None else self._held.enter_context(MujocoScratchSession(model, data, self._guard))
        self._table = tuple(witness.value or None for witness in self._guard.check().table)
        if len(self._table) != self._dimension * self._dimension:
            raise QueryBindingError("native geometry enumeration and collision declaration disagree")
        self._pair_first, self._pair_second = self._pair_universe()
        self._validate_effect_domain(data)
        self.configured = None
        self.population_all = False
        if not defer_configuration:
            with self._session.observe(data):
                configured = bind_query_configuration(configuration, (self.binding,), (self.declaration,),
                    _QUANTITIES, self._resolve_operand, validate_native_arguments=validate_mujoco_arguments)
            self._install_configuration(configuration, configured)
        self._query = self._guard.function("mj_geomDistance")

    def _install_configuration(self, configuration, configured):
        if any(item.binding_id != self.binding.binding_id for item in configured):
            raise QueryBindingError("geometry reader received another binding's configured query")
        self.configuration = configuration
        self.configured = tuple(configured)
        self._cutoffs = tuple(float(item.arguments()["distmax"]) for item in configured)
        self.population_all = any(item.instance_token is None and item.arguments().get("population") == "all"
                                  for item in configured)

    def _validate_effect_domain(self, data):
        self._session._validate_subject(data)
        self._validate_population_identity()

    def _validate_population_identity(self):
        """Reject a model whose pair universe or bounds may have changed."""
        check_ccd_iterations(self._version, int(self._model.opt.ccd_iterations))
        if (not np.array_equal(self._model.geom_type, self._geom_types)
                or not np.array_equal(self._model.geom_bodyid, self._body_ids)
                or any(not np.array_equal(getattr(self._model, name), retained)
                       for name, retained in self._population_arrays.items())):
            raise QueryBindingError("native geometry ownership or type changed; reset query identity")
        if np.any(self._geom_types < 0) or np.any(self._geom_types >= self._dimension):
            raise QueryBindingError("native geometry type is outside the declared collision table")

    def _resolve_operand(self, kind, operand):
        if kind != "geom":
            raise QueryBindingError("mj_geomDistance requires native geom operands")
        if "name" in operand:
            # Both standalone and combined binders enter one enclosing scope.
            index = int(self._guard.function("mj_name2id")(
                self._model, self._mujoco.mjtObj.mjOBJ_GEOM, operand["name"]))
        else:
            index = operand["id"]
        if type(index) is not int or not 0 <= index < len(self._geom_types):
            raise QueryBindingError("native geom operand is absent or out of bounds")
        return (index,)

    def collider_available(self, first: int, second: int) -> bool:
        types = sorted((int(self._geom_types[first]), int(self._geom_types[second])))
        return self._table[types[0] * self._dimension + types[1]] is not None

    def _pair_universe(self):
        """Unordered pairs the engine's default pair filter would collide.

        The body filter follows ``filterBodyPair`` in
        ``src/engine/engine_collision_driver.c`` at the reviewed source
        commit: two bodies are discarded when they share a weld group, or when
        one weld group is the weld group of the other's parent and neither is
        the world. Geom bitmask compatibility is the narrow-phase geom filter.
        Sleep-state filtering is not modelled; the reviewed rows do not
        enable sleeping. The canonical (min, max) order is the requested
        native operand order; it is one ordered operand per pair, not a
        symmetry claim.
        """
        arrays = self._population_arrays
        contype, conaffinity = arrays["geom_contype"], arrays["geom_conaffinity"]
        body = self._body_ids
        weld_of_body, parent_of_body = arrays["body_weldid"], arrays["body_parentid"]
        weld = weld_of_body[body]
        parent_weld = weld_of_body[parent_of_body[weld]]
        compatible = ((contype[:, None] & conaffinity[None, :]) != 0) | ((contype[None, :] & conaffinity[:, None]) != 0)
        distinct = (body[:, None] != body[None, :]) & (weld[:, None] != weld[None, :])
        nonworld = (weld[:, None] != 0) & (weld[None, :] != 0)
        related = nonworld & ((weld[:, None] == parent_weld[None, :]) | (weld[None, :] == parent_weld[:, None]))
        types = self._geom_types
        low, high = np.minimum(types[:, None], types[None, :]), np.maximum(types[:, None], types[None, :])
        table = np.array([item is not None for item in self._table]).reshape(self._dimension, self._dimension)
        supported = table[low, high]
        universe = np.triu(compatible & distinct & ~related & supported, 1)
        first, second = np.nonzero(universe)
        return first.astype(int), second.astype(int)

    def population_pairs(self) -> Iterator[Tuple[int, int]]:
        """Yield the canonical pair universe; pruning happens per read."""
        for first, second in zip(self._pair_first.tolist(), self._pair_second.tolist()):
            yield first, second

    def _lower_bounds(self, data, first, second):
        """Bounding-sphere lower bound on the distance of each given pair."""
        xpos = np.asarray(data.geom_xpos)
        rbound = self._population_arrays["geom_rbound"]
        return np.linalg.norm(xpos[first] - xpos[second], axis=1) - rbound[first] - rbound[second]

    def read(self, data) -> Tuple[GeomDistanceResult, ...]:
        """Copy the current state once, then execute the configured subject."""
        if self._closed:
            raise QueryBindingError("native geometry query reader is closed")
        # The scratch snapshot is validated once in _read_snapshot; the
        # session's observe already authenticates the live subject.
        with self._session.observe(data, completion=("mj_kinematics",)) as snapshot:
            return self._read_snapshot(snapshot)

    def _read_snapshot(self, snapshot, pairs=None):
        """Capture owned results within the caller's shared scratch interval.

        ``pairs`` restricts the population read to the given ``(first,
        second)`` geom ID arrays, a subset of the universe the caller chose
        (the adapter's specification scope, issue #136); ``None`` reads the
        whole universe. Explicit instances are read either way.
        """
        data = snapshot.data
        if self._closed or self.configured is None:
            raise QueryBindingError("native geometry query reader is closed or unconfigured")
        if data is not self._session._scratch or "mj_kinematics" not in snapshot.completed:
            raise QueryBindingError("geometry query requires this session's completed scratch")
        if type(data) is not self._mujoco.MjData:
            raise QueryBindingError("query reader needs native MuJoCo data")
        # The session validated the live subject when it opened this
        # snapshot; only the population-defining model arrays are rechecked.
        self._validate_population_identity()
        if self._query is not self._guard.function("mj_geomDistance"):
            raise QueryBindingError("native geometry query reader identity changed")
        options = NativeValue.capture({name: getattr(self._model.opt, name) for name in self._option_fields})
        buffer = np.empty(self._buffer_shape, dtype=data.qpos.dtype)
        result = []
        completion_warnings = tuple(warning.message for warning in snapshot.completion_warnings)
        bounds = None
        first_ids, second_ids = (self._pair_first, self._pair_second) if pairs is None else pairs
        for configured, cutoff in zip(self.configured, self._cutoffs):
            if configured.instance_token is None:
                if bounds is None:
                    bounds = self._lower_bounds(data, first_ids, second_ids)
                # Plane and heightfield pairs have no finite bounding sphere.
                pruned = (bounds >= cutoff) & ~self._unbounded[first_ids] & ~self._unbounded[second_ids]
                rows = zip(first_ids.tolist(), second_ids.tolist(), pruned.tolist(),
                           [float(bound) for bound in bounds.tolist()])
            else:
                first, second = (owner.native_index[0] for owner in configured.operands)
                rows = ((first, second, False, None),)
            for first, second, skip, bound in rows:
                body_ids = (int(self._body_ids[first]), int(self._body_ids[second]))
                if skip:
                    result.append(GeomDistanceResult(
                        (first, second), body_ids, cutoff, None, None, options, True,
                        configured.instance_token or "", configured.binding_id, (), completion_warnings,
                        executed=False, lower_bound=bound,
                    ))
                    continue
                warning_start = len(snapshot.warnings)
                returned = float(self._query(self._model, snapshot.data, first, second, cutoff, buffer))
                result.append(GeomDistanceResult(
                    (first, second), body_ids,
                    cutoff, returned, NativeValue.capture(buffer), options,
                    self.collider_available(first, second), configured.instance_token or "", configured.binding_id,
                    tuple(warning.message for warning in snapshot.warnings[warning_start:]), completion_warnings,
                    lower_bound=bound,
                ))
        return tuple(result)

    def close(self):
        """Retire query scratch before releasing native image descriptors."""
        if not self._closed:
            try:
                self._held.close()
            finally:
                self._closed = True
