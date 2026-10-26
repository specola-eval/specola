"""Frozen, Isaac-local monitor extraction plans.

This module intentionally models only the current scalar legacy surface.  It
does not infer an asset's semantics from ``.data``: documented
``InteractiveScene`` family mappings determine whether an entry is an
articulation, rigid object, deformable compatibility entry, camera/sensor, or
an unsupported scene kind.  Deformables retain their existing root-position
compatibility behavior but are explicitly marked incomplete; nodal state and
topology belong to later work.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional, Tuple

import numpy as np

from ..adapter import MonitorObservation, PropertyMeta
from ..entities import EntityState, TrackState


_STATIC_SCENE_KEYS = frozenset(("terrain", "ground", "sky_light", "light", "dome_light"))
_PHYSICAL_FAMILIES = ("articulations", "rigid_objects", "deformable_objects")
_NON_PHYSICAL_FAMILIES = ("sensors", "extras")
_UNSUPPORTED_FAMILIES = ("rigid_object_collections", "surface_grippers")
_AXES = ("0", "1", "2")
_WRENCH_AXES = ("0", "1", "2", "3", "4", "5")


@dataclass(frozen=True)
class SemanticField:
    """One compiled monitor scalar with declared source semantics."""

    key: str
    source: str
    owner: str
    unit: str
    frame: Optional[str]
    group: str
    note: str
    slot: int
    component: int
    provenance: str = "legacy_unvalidated"
    exactness: str = "legacy_unvalidated"
    applicability: str = "applicable_legacy_projection"


@dataclass(frozen=True)
class VectorSource:
    """Declared semantics for a monitor entity position or velocity vector."""

    source: str
    owner: str
    unit: str
    frame: str
    provenance: str
    exactness: str
    applicability: str = "applicable_legacy_projection"


@dataclass(frozen=True)
class EntityExtractionPlan:
    """Immutable static identity, names, metadata, and source layout."""

    entity_id: int
    key: str
    entity_class: str
    family: str
    profile: str
    asset: Any
    position: VectorSource
    velocity: Optional[VectorSource]
    fields: Tuple[SemanticField, ...]
    unresolved_semantic_candidates: Tuple[SemanticField, ...]
    metadata: Tuple[Tuple[str, PropertyMeta], ...]


@dataclass(frozen=True)
class ExtractionPlan:
    """The closed scalar extraction layout for one adapter lifetime."""

    entities: Tuple[EntityExtractionPlan, ...]
    excluded_scene_keys: Tuple[Tuple[str, str], ...]
    unresolved_scene_keys: Tuple[Tuple[str, str], ...]
    sensors: Tuple[str, ...]

    @property
    def entity_id_of(self) -> Mapping[str, int]:
        return MappingProxyType({entity.key: entity.entity_id for entity in self.entities})


class IsaacLabExtractionStrategy:
    """Narrow advanced seam; batch and incremental strategies remain future work."""

    def set_extraction_scope(
        self, scope: Optional[Mapping[str, FrozenSet[str]]]
    ) -> None:
        """Install the monitor property scope for native extraction.

        The adapter owns the public scope contract. This small strategy seam
        lets a scalar extractor prune semantic fields before it accesses or
        converts their native arrays; strategies that do not implement the
        optimization retain the adapter base class's fail-safe projection.
        """
        del scope

    def set_plan_extraction_scope(
        self,
        plan: ExtractionPlan,
        scope: Optional[Mapping[str, FrozenSet[str]]],
    ) -> None:
        """Install one plan's scope, with legacy strategy compatibility."""
        del plan
        self.set_extraction_scope(scope)

    def compile_plan(
        self,
        scene: Any,
        robot_class: str,
        object_class: str,
    ) -> ExtractionPlan:
        raise NotImplementedError

    def extract(
        self,
        plan: ExtractionPlan,
        env_index: int,
        timestep: int,
        time_seconds: float,
    ) -> MonitorObservation:
        raise NotImplementedError

    def metadata(
        self, plan: ExtractionPlan, env_index: int, entity_id: int
    ) -> Dict[str, PropertyMeta]:
        raise NotImplementedError


class ScalarPlanExtractor(IsaacLabExtractionStrategy):
    """Scalar legacy extractor backed by a precompiled immutable plan."""

    def __init__(
        self,
        to_numpy: Callable[[Any], Optional[np.ndarray]],
        token: Callable[[str], str],
    ) -> None:
        self._to_numpy = to_numpy
        self._token = token
        self._scope_by_plan: Dict[
            int,
            Tuple[ExtractionPlan, Optional[Dict[str, FrozenSet[str]]]],
        ] = {}

    def set_plan_extraction_scope(
        self,
        plan: ExtractionPlan,
        scope: Optional[Mapping[str, FrozenSet[str]]],
    ) -> None:
        normalized = (
            None
            if scope is None
            else {
                str(entity_class): frozenset(keys)
                for entity_class, keys in scope.items()
            }
        )
        # Keep the plan strongly referenced beside its identity. This makes
        # identity reuse impossible while the strategy lives and lets one
        # injected strategy safely serve independent adapter instances.
        self._scope_by_plan[id(plan)] = (plan, normalized)

    def _scope_for_plan(
        self, plan: ExtractionPlan
    ) -> Optional[Dict[str, FrozenSet[str]]]:
        record = self._scope_by_plan.get(id(plan))
        if record is None:
            return None
        bound_plan, scope = record
        if bound_plan is not plan:
            raise RuntimeError("Isaac extraction-plan identity collision")
        return scope

    @staticmethod
    def _mapping(scene: Any, name: str) -> Mapping[str, Any]:
        value = getattr(scene, name, None)
        if value is None:
            return MappingProxyType({})
        try:
            return MappingProxyType(dict(value))
        except (TypeError, ValueError) as error:
            raise ValueError("Isaac Lab scene family {0!r} is not a mapping: {1}".format(name, error))

    @staticmethod
    def _names(asset: Any, name: str) -> Tuple[str, ...]:
        return tuple(getattr(asset, name, ()) or ())

    def _array_available(self, asset: Any, source: str, minimum_ndim: int) -> bool:
        value = self._to_numpy(getattr(asset.data, source, None))
        return value is not None and value.ndim >= minimum_ndim

    def _array_width(self, asset: Any, source: str, axis: int, minimum_ndim: int) -> int:
        # Plan compilation needs only static shape. Reading the shape from
        # the native object avoids an unnecessary device-to-host transfer
        # before a monitor scope can prune semantic fields. A converter
        # fallback keeps lightweight stand-ins and unusual wrappers working.
        native = getattr(asset.data, source, None)
        shape = self._native_shape(native)
        if shape is None:
            value = self._to_numpy(native)
            shape = None if value is None else value.shape
        if shape is None or len(shape) < minimum_ndim:
            return 0
        return int(shape[axis])

    def _fields(self, asset: Any, family: str) -> Tuple[Tuple[SemanticField, ...], Tuple[Tuple[str, PropertyMeta], ...]]:
        if family == "deformable_objects":
            # Explicitly incomplete legacy profile: no nodal/topology claims.
            return (), ()
        fields = []
        joint_names = self._names(asset, "joint_names")
        body_names = self._names(asset, "body_names")
        joint_pos = self._array_width(asset, "joint_pos", -1, 2)
        joint_vel = self._array_width(asset, "joint_vel", -1, 2)
        body_pos = self._array_width(asset, "body_pos_w", 1, 3)
        body_components = self._array_width(asset, "body_pos_w", -1, 3)
        for slot, name in enumerate(joint_names):
            token = self._token(name)
            if slot < joint_pos:
                fields.append(SemanticField("joint_pos_{0}".format(token), "joint_pos", "joint:{0}".format(name), "rad or m", None, "joint position", "rad or m", slot, 0))
            if slot < joint_vel:
                fields.append(SemanticField("joint_vel_{0}".format(token), "joint_vel", "joint:{0}".format(name), "rad/s or m/s", None, "joint velocity", "rad/s or m/s", slot, 0))
        for slot, name in enumerate(body_names):
            token = self._token(name)
            if slot < body_pos and body_components >= len(_AXES):
                for component, label in enumerate(_AXES):
                    fields.append(SemanticField("body_pos_w_{0}_{1}".format(token, label), "body_pos_w", "body:{0}".format(name), "m", "world", "body position", "m", slot, component))
        if self._wrench_declared(asset):
            for slot, name in enumerate(body_names):
                token = self._token(name)
                for component, label in enumerate(_WRENCH_AXES):
                    group, note = ("link forces", "N") if component < 3 else ("link torques", "N·m")
                    fields.append(SemanticField(
                        "link_wrench_{0}_{1}".format(token, label),
                        "get_link_incoming_joint_force",
                        "incoming_joint/body:{0}".format(name), note,
                        "body1 at body1 origin", group, note, slot, component,
                        "IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/joint_wrench/kernels.py:24-38; "
                        "reset freshness verified: benchmarks/adapter_fidelity/evidence_pins/probes/"
                        "2026-08-20-wrench-reset-freshness-probe.log",
                        "exact", "applicable_when_view_getter_present",
                    ))
        frozen_fields = tuple(fields)
        metadata = []
        for field in frozen_fields:
            metadata.append((field.key, PropertyMeta(group=field.group, note=field.note)))
        return frozen_fields, tuple(metadata)

    def _field_is_requested(
        self,
        scope: Optional[Mapping[str, FrozenSet[str]]],
        entity: EntityExtractionPlan,
        field: SemanticField,
    ) -> bool:
        if scope is None:
            return True
        return field.key in scope.get(
            entity.entity_class, frozenset()
        )

    @staticmethod
    def _wrench_declared(asset: Any) -> bool:
        """Structural check only: the view publishes the getter.

        Plan compilation must not invoke the getter — a scoped-out wrench
        surface costs nothing, and reset-time reads stay out of discovery.
        """
        view = getattr(asset, "root_view", None)
        return view is not None and callable(
            getattr(view, "get_link_incoming_joint_force", None)
        )

    def _link_wrench(self, asset: Any) -> Optional[np.ndarray]:
        """Read the engine's raw incoming joint wrench for every link.

        Returns a (num_envs, num_bodies, 6) host array, or ``None`` when the
        view does not publish the getter or the read fails. Reset-lifecycle
        freshness was the reason this surface was previously withheld
        (issue #105); the 2026-08-20 probe refuted the staleness hypothesis
        on the pinned certification configuration — see the provenance
        recorded on the emitted fields.
        """
        view = getattr(asset, "root_view", None)
        if view is None:
            return None
        getter = getattr(view, "get_link_incoming_joint_force", None)
        if not callable(getter):
            return None
        try:
            wrench = self._to_numpy(getter())
        except Exception:  # pragma: no cover - backend/version dependent
            return None
        if wrench is None or wrench.ndim < 3:
            return None
        return np.asarray(wrench, dtype=float)

    def compile_plan(self, scene: Any, robot_class: str, object_class: str) -> ExtractionPlan:
        families = {}
        for name in _PHYSICAL_FAMILIES + _NON_PHYSICAL_FAMILIES + _UNSUPPORTED_FAMILIES:
            families[name] = self._mapping(scene, name)
        membership: Dict[str, str] = {}
        for family, assets in families.items():
            for key in assets:
                if key in membership:
                    raise ValueError("Isaac Lab scene key {0!r} appears in both {1!r} and {2!r}".format(key, membership[key], family))
                membership[key] = family
        scene_keys = tuple(scene.keys())
        scene_key_set = set(scene_keys)
        for key, family in membership.items():
            if key not in scene_key_set:
                raise ValueError("Isaac Lab {0} entry {1!r} is absent from scene.keys()".format(family, key))
        entities = []
        excluded = []
        unresolved = []
        for key in scene_keys:
            family = membership.get(key)
            # A documented physical family wins over a coincidental static
            # spelling: an articulation named ``ground`` is still an asset.
            if family is None and key in _STATIC_SCENE_KEYS:
                excluded.append((str(key), "static_scene_entry"))
                continue
            if family in _NON_PHYSICAL_FAMILIES:
                if family == "sensors":
                    unresolved.append((str(key), "unresolved_sensor_kind_not_materialized"))
                else:
                    excluded.append((str(key), "extra_scene_entry"))
                continue
            if family in _UNSUPPORTED_FAMILIES:
                raise ValueError("Isaac Lab {0} entry {1!r} is unsupported for monitor extraction".format(family, key))
            if family not in _PHYSICAL_FAMILIES:
                raise ValueError("Isaac Lab scene key {0!r} is unclassified; declare an InteractiveScene family mapping".format(key))
            asset = families[family][key]
            if asset is None:
                raise ValueError("Isaac Lab {0} entry {1!r} is missing its declared asset".format(family, key))
            fields, metadata = self._fields(asset, family)
            entities.append(EntityExtractionPlan(
                entity_id=len(entities), key=str(key),
                entity_class=robot_class if key == "robot" else object_class,
                family=family,
                profile="deformable_legacy_incomplete" if family == "deformable_objects" else "scalar_legacy_projection_incomplete",
                asset=asset,
                position=VectorSource(
                    source="root_pos_w", owner=("deformable_nodal_mean_root" if family == "deformable_objects" else "asset_root"), unit="m", frame="world",
                    provenance=(
                        "installed_deformable_root_pos_w_mean_nodal_positions"
                        if family == "deformable_objects" else "legacy_root_pos_w"
                    ),
                    exactness="derived" if family == "deformable_objects" else "legacy_unvalidated",
                ),
                velocity=(
                    None
                    if family == "deformable_objects"
                    else VectorSource(
                        source="root_lin_vel_w", owner="asset_root", unit="m/s", frame="world",
                        provenance="legacy_root_lin_vel_w", exactness="legacy_unvalidated",
                    )
                ),
                fields=fields,
                # The raw-wrench surface graduated from unresolved to
                # included (reset-freshness probe, 2026-08-20); no
                # semantic candidates remain undecided.
                unresolved_semantic_candidates=(),
                metadata=metadata,
            ))
        if not entities:
            raise ValueError("Isaac Lab scene exposes no articulated assets to monitor; scene keys were {0}".format(sorted(scene.keys())))
        sensor_names = []
        for key in families["sensors"]:
            sensor_names.append(str(key))
        return ExtractionPlan(tuple(entities), tuple(excluded), tuple(unresolved), tuple(sorted(sensor_names)))

    @staticmethod
    def _native_shape(value: Any) -> Optional[Tuple[int, ...]]:
        # Isaac's ProxyArray logical Warp shape omits vector-dtype component
        # axes (for example ``(N,)`` vec3f), while its zero-copy torch view is
        # the actual indexable scalar layout ``(N, 3)``. Prefer that view.
        tensor = getattr(value, "torch", None)
        shape = getattr(tensor, "shape", None)
        if shape is None:
            shape = getattr(value, "shape", None)
        if shape is None:
            return None
        try:
            return tuple(int(width) for width in shape)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _native_storage(value: Any) -> Any:
        """Return the indexable device value without copying it to the host."""
        tensor = getattr(value, "torch", None)
        return tensor if tensor is not None and not callable(tensor) else value

    def _convert_selection(
        self, data: Any, source: str, index: Any
    ) -> Optional[np.ndarray]:
        """Index the native value first, then cross the device/host boundary."""
        native = getattr(data, source, None)
        if native is None:
            return None
        storage = self._native_storage(native)
        try:
            selected = storage[index]
        except (IndexError, TypeError) as error:
            raise ValueError(
                "Isaac source {0!r} does not support the required indexed read"
                .format(source)
            ) from error
        converted = self._to_numpy(selected)
        return None if converted is None else np.asarray(converted)

    def _field_available(
        self, data: Any, field: SemanticField, env_index: int,
        wrench: Optional[np.ndarray] = None,
    ) -> bool:
        if field.source == "get_link_incoming_joint_force":
            return (
                wrench is not None
                and env_index < wrench.shape[0]
                and field.slot < wrench.shape[1]
                and field.component < wrench.shape[2]
            )
        shape = self._native_shape(getattr(data, field.source, None))
        if shape is None:
            return False
        if field.source in ("joint_pos", "joint_vel"):
            return (
                len(shape) >= 2
                and env_index < shape[0]
                and field.slot < shape[-1]
            )
        return (
            len(shape) >= 3
            and env_index < shape[0]
            and field.slot < shape[1]
            and field.component < shape[2]
        )

    def _active_fields(
        self,
        scope: Optional[Mapping[str, FrozenSet[str]]],
        entity: EntityExtractionPlan,
        env_index: int,
    ) -> Tuple[SemanticField, ...]:
        data = entity.asset.data
        active = []
        for field in entity.fields:
            if not self._field_is_requested(scope, entity, field):
                continue
            if self._field_available(data, field, env_index):
                active.append(field)
        return tuple(active)

    def _vector(
        self, data: Any, source: str, env_index: int
    ) -> Optional[np.ndarray]:
        shape = self._native_shape(getattr(data, source, None))
        if shape is None or len(shape) < 2 or env_index >= shape[0]:
            return None
        value = self._convert_selection(
            data, source, (env_index, slice(0, 3))
        )
        return None if value is None else np.asarray(value, dtype=float)

    def _property_values(
        self,
        data: Any,
        active_fields: Tuple[SemanticField, ...],
        env_index: int,
        wrench: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        by_source: Dict[str, list] = {}
        for field in active_fields:
            by_source.setdefault(field.source, []).append(field)

        values_by_key: Dict[str, float] = {}
        for source, fields in by_source.items():
            if source == "get_link_incoming_joint_force":
                if wrench is None:
                    continue
                for field in fields:
                    values_by_key[field.key] = float(
                        wrench[env_index, field.slot, field.component]
                    )
                continue
            if source in ("joint_pos", "joint_vel"):
                index = (env_index, [field.slot for field in fields])
            else:
                index = (
                    env_index,
                    [field.slot for field in fields],
                    [field.component for field in fields],
                )
            selected = self._convert_selection(data, source, index)
            if selected is None:
                continue
            flat = np.asarray(selected).reshape(-1)
            if len(flat) != len(fields):
                raise ValueError(
                    "Isaac source {0!r} returned {1} scoped values for {2} fields"
                    .format(source, len(flat), len(fields))
                )
            for field, value in zip(fields, flat):
                values_by_key[field.key] = float(value)
        return {
            field.key: values_by_key[field.key]
            for field in active_fields
            if field.key in values_by_key
        }

    def metadata(
        self, plan: ExtractionPlan, env_index: int, entity_id: int
    ) -> Dict[str, PropertyMeta]:
        del env_index
        for entity in plan.entities:
            if entity.entity_id == entity_id:
                return dict(entity.metadata)
        return {}

    def extract(self, plan: ExtractionPlan, env_index: int, timestep: int, time_seconds: float) -> MonitorObservation:
        scope = self._scope_for_plan(plan)
        entities: Dict[int, EntityState] = {}
        for entity in plan.entities:
            data = entity.asset.data
            # One view-getter call per entity per step, and only when the
            # installed scope actually requests a wrench field; every
            # wrench field of this entity indexes the same host array.
            wrench = (
                self._link_wrench(entity.asset)
                if any(field.source == "get_link_incoming_joint_force"
                       and self._field_is_requested(scope, entity, field)
                       for field in entity.fields)
                else None
            )
            active_fields = tuple(
                field
                for field in entity.fields
                if self._field_is_requested(scope, entity, field)
                and self._field_available(data, field, env_index,
                                          wrench=wrench)
            )
            position = self._vector(data, entity.position.source, env_index)
            velocity = (
                None
                if entity.velocity is None
                else self._vector(data, entity.velocity.source, env_index)
            )
            properties = self._property_values(data, active_fields, env_index,
                                               wrench=wrench)
            entities[entity.entity_id] = EntityState(entity_id=entity.entity_id, entity_class=entity.entity_class, position=position, velocity=velocity, track_age=timestep, track_state=TrackState.CONFIRMED, properties=properties)
        return MonitorObservation(timestep=timestep, time=time_seconds, perceived_entities=entities)
