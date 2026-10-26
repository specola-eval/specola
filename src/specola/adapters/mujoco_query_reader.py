"""Join configured MuJoCo queries around one owned observation interval.

The source-backed family readers supply bindings to the existing shared
binder. This integration is a bounded query reader, not a producer inventory
or a provider readiness decision.
"""

from contextlib import ExitStack
from dataclasses import dataclass, replace
from graphlib import TopologicalSorter
from importlib import import_module

from .mujoco_native import MujocoNativeGuard
from .mujoco_queries import IsolatedGeomDistanceReader, validate_mujoco_arguments
from .mujoco_scratch import MujocoScratchSession
from .mujoco_spatial_queries import MujocoSpatialQueryReader
from ..conformance.query_bindings import (
    QueryBindingError, bind_query_configuration, validate_query_arguments,
    validate_query_binding,
)


@dataclass(frozen=True)
class MujocoQueryResults:
    """Owned outputs from geometry and spatial queries on one copied state."""

    geometry: tuple
    spatial: tuple
    configuration_sha256: str
    completed: tuple


class MujocoQueryReader:
    """Resolve one complete configuration and share detached native data."""

    def __init__(self, model, data, configuration, declarations, *,
                 _guard=None, _session=None):
        self._held = ExitStack()
        self._closed = False
        try:
            mujoco = import_module("mujoco")
            # Geometry declarations load child modules onto the producer
            # package. Finish those imports before enrolling its namespace.
            import_module("mujoco.introspect.functions")
            import_module("mujoco.introspect.structs")
            self._guard = _guard if _guard is not None else self._held.enter_context(MujocoNativeGuard(mujoco))
            self._session = _session if _session is not None else self._held.enter_context(MujocoScratchSession(model, data, self._guard))
            shared = dict(_guard=self._guard, _session=self._session, _defer_configuration=True)
            self.geometry = IsolatedGeomDistanceReader(model, data, configuration, **shared)
            self._held.callback(self.geometry.close)
            self.spatial = MujocoSpatialQueryReader(model, data, configuration, declarations, **shared)
            self._held.callback(self.spatial.close)
            if "mj_geomDistance" not in declarations:
                raise QueryBindingError("native geometry query has no independent source declaration")
            declared = declarations["mj_geomDistance"]
            self.geometry.binding = replace(self.geometry.binding, callable_id=declared.callable_id)
            validate_query_binding(self.geometry.binding, declared, self.geometry.binding.quantity_ids)
            self.geometry.declaration = declared
            self.bindings = (self.geometry.binding,) + self.spatial.bindings
            function_names = ("mj_geomDistance",) + tuple(self.spatial._functions)
            self.declarations = tuple(declarations[name] for name in function_names)
            quantities = tuple(dict.fromkeys(q for binding in self.bindings for q in binding.quantity_ids))
            validate_query_arguments(configuration, self.bindings, self.declarations, quantities,
                                     validate_native_arguments=validate_mujoco_arguments)
            with self._session.observe(data):
                configured = bind_query_configuration(configuration, self.bindings, self.declarations,
                    quantities, self._resolve_owner, validate_native_arguments=validate_mujoco_arguments)
            self.configuration = configuration
            self.configured = configured
            self.geometry._install_configuration(configuration, tuple(
                item for item in configured if item.binding_id == self.geometry.binding.binding_id))
            self.spatial._install_configuration(configuration, tuple(
                item for item in configured if item.binding_id in self.spatial._plans))
            stages = TopologicalSorter()
            for sequence in (("mj_kinematics",), self.spatial._completion):
                for index, stage in enumerate(sequence):
                    stages.add(stage, *sequence[:index])
            self.completion = tuple(stages.static_order())
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def _resolve_owner(self, kind, operand):
        if kind == "geom":
            return self.geometry._resolve_operand(kind, operand)
        return self.spatial._resolve_owner(kind, operand)

    def read(self, data):
        """Read the configured populations with one copy and one completion."""
        if self._closed:
            raise QueryBindingError("native combined query reader is closed")
        self.geometry._validate_effect_domain(data)
        self.spatial._validate_current_subject()
        with self._session.observe(data, completion=self.completion) as snapshot:
            return self._read_snapshot(snapshot)

    def _read_snapshot(self, snapshot):
        geometry = self.geometry._read_snapshot(snapshot)
        spatial = self.spatial._read_snapshot(snapshot)
        return MujocoQueryResults(geometry, spatial, self.configuration.sha256, snapshot.completed)

    def close(self):
        """Retire readers, scratch and native custody in dependency order."""
        if not self._closed:
            try:
                self._held.close()
            finally:
                self._closed = True
