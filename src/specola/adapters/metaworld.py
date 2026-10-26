"""Meta-World v3 adapter with source-declared task entities.

Meta-World tasks run a Sawyer arm through Gymnasium-style MuJoCo
environments.  Shared MuJoCo extraction supplies robot, site, geometry,
sensor, actuator, and engine state.  This module supplies only Meta-World's
task semantics:

* one point target from ``_get_pos_goal()``;
* the exact ordered object vectors from ``_get_pos_objects()``;
* object identity and source provenance from the packaged, statically
  extracted source map; and
* physical object orientations where ``_get_quat_objects()`` has a named,
  reconciled source.

The adapter encodes no success criterion.  Meta-World criteria differ by task
and are often not box containment or even a distance to the target point.
Thresholds belong in specifications, with provenance from
``benchmarks/metaworld_success_criteria.json``.  A raw MuJoCo site named
``goal`` remains a first-class ``site`` entity when present, but it is never
promoted to a target or region.

The flattened Meta-World vectors remain authoritative live state.  The source
map supplies names and provenance, and every object vector is checked against
an independently evaluated, whitelisted source accessor on every observation.
There is no candidate-body scan, nearest assignment, or ``obj*`` fallback.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..adapter import PropertyMeta
from ..entities import EntityState, TrackState
from ..registry import register as _register
from .metaworld_declarations import (
    MetaworldDeclarationError,
    assert_exact_agreement,
    evaluate_accessor,
    exact_vector,
    normalize_orientation,
    task_declaration,
    task_name_for_environment,
    verify_environment_class,
    verify_installed_version,
)
from .mujoco import (
    BodyRule,
    DeclaredEntitySpec,
    MujocoAdapter,
    MujocoBodyOwner,
)


DEFAULT_TASK = "pick-place-v3"
_POSITION_COMPONENTS = ("0", "1", "2")
_QUATERNION_COMPONENTS = ("w", "x", "y", "z")
_GOAL_DISAGREEMENT_TOLERANCE_M = 1e-6
_GRIPPER_JOINT_NAMES = ("l_close", "r_close")
_MULTI_OBJECT_MAIN_GEOMS = {
    "hammer-v3": (
        "HammerHandle",
        "metaworld/envs/sawyer_hammer_v3.py:88-89",
    ),
    "stick-pull-v3": (
        "objGeom",
        "metaworld/sawyer_xyz_env.py:442-443 (inherited)",
    ),
    "stick-push-v3": (
        "objGeom",
        "metaworld/sawyer_xyz_env.py:442-443 (inherited)",
    ),
}


class MetaworldAdapter(MujocoAdapter):
    """Adapter for the exact Meta-World 3.1.1 v3 task declarations."""

    def __init__(
        self,
        env: Any,
        *,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        environment = getattr(env, "unwrapped", env)
        verify_installed_version()
        resolved_task = (
            task_name
            if task_name is not None
            else task_name_for_environment(environment)
        )
        declaration = task_declaration(resolved_task)
        verify_environment_class(environment, declaration)
        self._metaworld_env = environment
        self._task_declaration = declaration
        self.entity_roles: Dict[int, str] = {}
        self.entity_role_basis: Dict[int, str] = {}
        self.entity_role_provenance: Dict[int, str] = {}
        self._contact_role_by_body: Dict[int, str] = {}
        super().__init__(environment, **kwargs)

    @property
    def task_name(self) -> str:
        """Exact Meta-World registry key represented by this adapter."""
        return self._task_declaration.task

    def discovery_warnings(
        self,
        entities: Mapping[int, EntityState],
    ) -> Sequence[str]:
        """Flag a misleading raw ``goal`` site beside the declared target.

        The relationship is declared here, where Meta-World task semantics
        are known.  The generic discovery presenter merely renders adapter
        diagnostics and does not infer authority from entity names.
        """
        raw_site_id = self.name_to_id.get("site:goal")
        target_id = self.name_to_id.get("declared:target")
        if raw_site_id is None or target_id is None:
            return ()

        raw_site = entities.get(raw_site_id)
        target = entities.get(target_id)
        if (
            raw_site is None
            or target is None
            or raw_site.position is None
            or target.position is None
        ):
            return ()

        separation = float(
            np.linalg.norm(
                np.asarray(raw_site.position, dtype=float)
                - np.asarray(target.position, dtype=float)
            )
        )
        if separation <= _GOAL_DISAGREEMENT_TOLERANCE_M:
            return ()

        return (
            "Raw site 'site:goal' differs from declared target 'target' by "
            "{:.6g} m (tolerance {:.0e} m). The declared target from "
            "_get_pos_goal() is authoritative for task-target semantics; "
            "the raw site remains available as simulator state.".format(
                separation,
                _GOAL_DISAGREEMENT_TOLERANCE_M,
            ),
)

    def _default_body_rules(self) -> Sequence[BodyRule]:
        """Classify only the Sawyer subtree; task objects are declarations."""
        return (
            BodyRule(
                match=lambda name: name == "base",
                cls="robot",
                priority=0,
                joint_scope="subtree",
            ),
        )

    def _default_gripper_sites(self) -> Sequence[str]:
        return ("endEffector",)

    def _gripper_joint_positions(self) -> np.ndarray:
        """Read Meta-World's two physical finger-slide coordinates."""
        # The two qpos addresses depend only on the compiled model, but
        # resolving them walked every joint's name each step (issue
        # #92). Keyed on model identity so a task change recomputes.
        if getattr(self, "_gripper_addr_model", None) is not self._model:
            self._gripper_addr_model = self._model
            available = set(self._names("joint"))
            if any(
                name not in available for name in _GRIPPER_JOINT_NAMES
            ):
                self._gripper_qpos_addrs: Tuple[int, ...] = ()
            else:
                self._gripper_qpos_addrs = tuple(
                    int(
                        self._model.jnt_qposadr[
                            self._name2id("joint", name)
                        ]
                    )
                    for name in _GRIPPER_JOINT_NAMES
                )
        if not self._gripper_qpos_addrs:
            return np.empty(0, dtype=float)
        values = np.asarray(
            [
                self._data.qpos[address]
                for address in self._gripper_qpos_addrs
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            return np.empty(0, dtype=float)
        return values

    def _gripper_properties(
        self,
        entity_id: int,
        site_name: str,
        site_id: int,
    ) -> Dict[str, float]:
        """Expose the finger-slide state on the semantic gripper entity."""
        properties = super()._gripper_properties(
            entity_id,
            site_name,
            site_id,
        )
        values = self._gripper_joint_positions()
        if values.size == 0:
            return properties
        for index, value in enumerate(values):
            properties["gripper_qpos_{}".format(index)] = float(value)
        signed_mean = float(np.mean(values))
        properties["gripper_qpos_mean"] = signed_mean
        properties["gripper_qpos_abs_mean"] = float(
            np.mean(np.abs(values))
        )
        properties["grip"] = signed_mean
        return properties

    def _gripper_property_meta(self) -> Dict[str, PropertyMeta]:
        """Describe the Meta-World finger-slide properties."""
        metadata = super()._gripper_property_meta()
        values = self._gripper_joint_positions()
        for index in range(values.size):
            metadata["gripper_qpos_{}".format(index)] = PropertyMeta(
                group="gripper",
                note="m",
            )
        if values.size:
            metadata["gripper_qpos_mean"] = PropertyMeta(
                group="gripper",
                note="m",
            )
            metadata["gripper_qpos_abs_mean"] = PropertyMeta(
                group="gripper",
                note="m",
            )
            metadata["grip"] = PropertyMeta(group="gripper")
        return metadata

    # Sawyer robot bodies shared by the Meta-World v3 scenes, measured
    # from the compiled models (tests assert this set covers every
    # registered v3 task). Exact names rather than substrings: task
    # object names ("obj", "basket_goal", ...) vary per task and a
    # substring rule could silently absorb one. "mocap" is the control
    # target and carries no geoms, so it can never appear in a contact.
    _SAWYER_ROBOT_BODIES = frozenset(
        {
            "base",
            "controller_box",
            "pedestal_feet",
            "torso",
            "pedestal",
            "right_arm_base_link",
            "right_l0",
            "right_l1",
            "right_l1_2",
            "right_l2",
            "right_l2_2",
            "right_l3",
            "right_l4",
            "right_l4_2",
            "right_l5",
            "right_l6",
            "right_torso_itb",
            "right_arm_itb",
            "right_hand_camera",
            "right_wrist",
            "right_hand",
            "head",
            "screen",
            "head_camera",
            "hand",
            "rightclaw",
            "rightpad",
            "leftclaw",
            "leftpad",
        }
    )

    def _default_robot_bodies(self) -> Sequence[Any]:
        return (self._SAWYER_ROBOT_BODIES.__contains__,)

    def _declare_task_entities(self) -> Sequence[DeclaredEntitySpec]:
        declaration = self._task_declaration
        entities = [
            DeclaredEntitySpec(
                declaration_key="target",
                name="target",
                entity_class="target",
                source_kind="task_accessor",
                source_name="_get_pos_goal",
                source_provenance=declaration.target.provenance,
                tags=("metaworld:declared", "target"),
            )
        ]
        for item in declaration.objects:
            role = "target" if item.index == 0 else "distractor"
            entities.append(
                DeclaredEntitySpec(
                    declaration_key="object:{}".format(item.index),
                    name=item.declared_name,
                    entity_class="object",
                    source_kind=item.kind,
                    source_name=item.declared_name,
                    source_provenance=item.provenance,
                    tags=(
                        "metaworld:declared",
                        "source_kind:{}".format(item.kind),
                        "role:" + role,
                    ),
                )
            )
        return tuple(entities)

    @staticmethod
    def _object_role(index: int) -> str:
        return "target" if index == 0 else "distractor"

    def _verify_primary_object(self) -> str:
        """Cross-check object 0 against Meta-World's main-object geom."""
        objects = self._task_declaration.objects
        if len(objects) == 1:
            return "sole declared object"
        declaration = _MULTI_OBJECT_MAIN_GEOMS.get(
            self._task_declaration.task
        )
        if declaration is None:
            raise MetaworldDeclarationError(
                "{} has multiple objects but no reviewed main-object "
                "source declaration"
                .format(self._task_declaration.task)
            )
        main_geom_name, main_source = declaration
        main_geom_id = self._name2id("geom", main_geom_name)
        main_body_id = int(self._model.geom_bodyid[main_geom_id])
        source_body_id = self._source_body_id(objects[0])
        if source_body_id is None:
            raise MetaworldDeclarationError(
                "{} primary object has no exact source body".format(
                    self._task_declaration.task
                )
            )
        current = main_body_id
        parent_ids = np.asarray(self._model.body_parentid, dtype=int)
        while current != source_body_id:
            parent = int(parent_ids[current])
            if parent == current:
                raise MetaworldDeclarationError(
                    "{} main-object geom body {} is outside object 0 "
                    "source-body subtree {}".format(
                        self._task_declaration.task,
                        main_body_id,
                        source_body_id,
                    )
                )
            current = parent
        return (
            "_get_id_main_object() source {} selects geom {!r}, which "
            "belongs to object 0 "
            "source-body subtree"
        ).format(main_source, main_geom_name)

    def _spatial_source(self, item: Any) -> Optional[Tuple[str, str]]:
        if len(item.sources) != 1:
            return None
        sources = [
            (source.kind, source.name)
            for source in item.sources
            if source.kind in {
                "body_com",
                "body_frame",
                "geom_frame",
                "site",
                "site_helper",
            }
        ]
        if len(sources) != 1:
            return None
        return sources[0]

    def _source_body_id(self, item: Any) -> Optional[int]:
        source = self._spatial_source(item)
        if source is None:
            return None
        kind, name = source
        if kind in {"body_com", "body_frame"}:
            return self._name2id("body", name)
        if kind == "geom_frame":
            return int(
                self._model.geom_bodyid[self._name2id("geom", name)]
            )
        return int(
            self._model.site_bodyid[self._name2id("site", name)]
        )

    def _source_linear_velocity(self, item: Any) -> Optional[np.ndarray]:
        source = self._spatial_source(item)
        if source is None:
            return None
        kind, name = source
        import mujoco

        if kind == "body_com":
            object_type = mujoco.mjtObj.mjOBJ_BODY
            object_id = self._name2id("body", name)
        elif kind == "body_frame":
            object_type = mujoco.mjtObj.mjOBJ_XBODY
            object_id = self._name2id("body", name)
        elif kind == "geom_frame":
            object_type = mujoco.mjtObj.mjOBJ_GEOM
            object_id = self._name2id("geom", name)
        else:
            object_type = mujoco.mjtObj.mjOBJ_SITE
            object_id = self._name2id("site", name)
        velocity = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self._model,
            self._data,
            object_type,
            object_id,
            velocity,
            0,
        )
        return velocity[3:].copy()

    def _declared_object_body_roots(self) -> Dict[int, int]:
        """Map each exact object source-body root to its semantic entity."""
        registrations = {
            record.spec.declaration_key: record.entity_id
            for record in self._declared_entities
        }
        result: Dict[int, int] = {}
        for item in self._task_declaration.objects:
            body_id = self._source_body_id(item)
            if body_id is None:
                continue
            entity_id = registrations["object:{}".format(item.index)]
            previous = result.get(body_id)
            if previous is not None and previous != entity_id:
                raise MetaworldDeclarationError(
                    "Meta-World source body {} maps to multiple declared "
                    "objects".format(body_id)
                )
            result[body_id] = entity_id
        return result

    def _additional_body_owners(self) -> Sequence[MujocoBodyOwner]:
        """Route each declared object's physical subtree to that object."""
        roots = self._declared_object_body_roots()
        parent_ids = np.asarray(self._model.body_parentid, dtype=int)
        bodies_by_root: Dict[int, list[int]] = {
            root_id: [] for root_id in roots
        }
        for body_id in range(len(parent_ids)):
            current = body_id
            while True:
                if current in roots:
                    bodies_by_root[current].append(body_id)
                    break
                parent = int(parent_ids[current])
                if parent == current:
                    break
                current = parent
        return tuple(
            MujocoBodyOwner(
                entity_id=roots[root_id],
                body_ids=tuple(body_ids),
                owner_name="declared object source body {}".format(root_id),
            )
            for root_id, body_ids in sorted(bodies_by_root.items())
        )

    def _build_labels(self) -> None:
        super()._build_labels()
        primary_provenance = self._verify_primary_object()
        records = {
            record.spec.declaration_key: record.entity_id
            for record in self._declared_entities
        }
        self.entity_roles = {}
        self.entity_role_basis = {}
        self.entity_role_provenance = {}
        main_roles: Dict[int, str] = {}
        for item in self._task_declaration.objects:
            entity_id = records["object:{}".format(item.index)]
            role = self._object_role(item.index)
            self.entity_roles[entity_id] = role
            self.entity_role_basis[entity_id] = "structural_derivation"
            if item.index == 0:
                role_provenance = (
                    "{}; {}; Specola primary target"
                ).format(item.provenance, primary_provenance)
            else:
                role_provenance = (
                    "{}; Specola non-primary ordinary-object complement; "
                    "declared task participant, not task-irrelevant"
                ).format(item.provenance)
            self.entity_role_provenance[entity_id] = role_provenance
            body_id = self._source_body_id(item)
            if body_id is not None:
                previous = main_roles.get(body_id)
                if previous is not None and previous != role:
                    raise MetaworldDeclarationError(
                        "Meta-World source body {} has conflicting roles "
                        "{!r} and {!r}".format(body_id, previous, role)
                    )
                main_roles[body_id] = role

        self._contact_role_by_body = {}
        parent_ids = np.asarray(self._model.body_parentid, dtype=int)
        for body_id in range(len(parent_ids)):
            current = body_id
            while True:
                role = main_roles.get(current)
                if role is not None:
                    self._contact_role_by_body[body_id] = role
                    break
                parent = int(parent_ids[current])
                if parent == current:
                    break
                current = parent

    def _contact_tag_vocabulary(self) -> FrozenSet[str]:
        return frozenset(
            {"role:target", "role:distractor", "role:robot"}
        )

    def _contact_pair_tags(
        self,
        body_pair: Tuple[int, int],
    ) -> Tuple[str, ...]:
        tags = set()
        for body_id in body_pair:
            if body_id in self._robot_body_ids:
                tags.add("role:robot")
            role = self._contact_role_by_body.get(body_id)
            if role is not None:
                tags.add("role:" + role)
        return tuple(sorted(tags))

    def _declared_entity_states(
        self,
        registrations: Mapping[str, int],
    ) -> Mapping[str, EntityState]:
        declaration = self._task_declaration
        environment = self._metaworld_env

        target = exact_vector(
            environment._get_pos_goal(),
            size=3,
            context="{} _get_pos_goal()".format(declaration.task),
        )
        flattened_positions = exact_vector(
            environment._get_pos_objects(),
            size=3 * len(declaration.objects),
            context="{} _get_pos_objects()".format(declaration.task),
        )
        flattened_orientations = exact_vector(
            environment._get_quat_objects(),
            size=4 * len(declaration.orientations),
            context="{} _get_quat_objects()".format(declaration.task),
        )

        states: Dict[str, EntityState] = {
            "target": EntityState(
                entity_id=registrations["target"],
                entity_class="target",
                position=target.copy(),
                velocity=None,
                track_age=self._timestep,
                track_state=TrackState.CONFIRMED,
                properties={
                    "target_pos_{}".format(component): float(target[index])
                    for index, component in enumerate(_POSITION_COMPONENTS)
                },
            )
        }

        for item, orientation in zip(
            declaration.objects,
            declaration.orientations,
        ):
            declaration_key = "object:{}".format(item.index)
            start = 3 * item.index
            position = flattened_positions[start : start + 3].copy()
            independent_position = exact_vector(
                evaluate_accessor(item.accessor, environment),
                size=3,
                context="{} object {} accessor {}".format(
                    declaration.task,
                    item.index,
                    item.accessor,
                ),
            )
            assert_exact_agreement(
                position,
                independent_position,
                context="{} object {} ({})".format(
                    declaration.task,
                    item.index,
                    item.declared_name,
                ),
            )

            orientation_start = 4 * orientation.index
            authored_orientation = flattened_orientations[
                orientation_start : orientation_start + 4
            ].copy()
            independent_orientation = exact_vector(
                evaluate_accessor(orientation.accessor, environment),
                size=4,
                context="{} orientation {} accessor {}".format(
                    declaration.task,
                    orientation.index,
                    orientation.accessor,
                ),
            )
            assert_exact_agreement(
                authored_orientation,
                independent_orientation,
                context="{} orientation {} ({})".format(
                    declaration.task,
                    orientation.index,
                    item.declared_name,
                ),
            )
            normalized = normalize_orientation(
                authored_orientation,
                orientation.representation,
            )

            properties = {
                "object_pos_{}".format(component): float(position[index])
                for index, component in enumerate(_POSITION_COMPONENTS)
            }
            linear_velocity = self._source_linear_velocity(item)
            if linear_velocity is not None:
                properties.update(
                    {
                        "object_linvel_{}".format(component): float(
                            linear_velocity[index]
                        )
                        for index, component in enumerate(
                            _POSITION_COMPONENTS
                        )
                    }
                )
            if normalized is not None:
                properties.update(
                    {
                        "object_quat_{}".format(component): float(
                            normalized[index]
                        )
                        for index, component in enumerate(
                            _QUATERNION_COMPONENTS
                        )
                    }
                )

            states[declaration_key] = EntityState(
                entity_id=registrations[declaration_key],
                entity_class="object",
                position=position,
                velocity=linear_velocity,
                track_age=self._timestep,
                track_state=TrackState.CONFIRMED,
                properties=properties,
            )

        return states

    def _declared_entity_property_metadata(
        self,
        declaration_key: str,
    ) -> Mapping[str, PropertyMeta]:
        if declaration_key == "target":
            return {
                "target_pos_{}".format(component): PropertyMeta(
                    group="declared target position",
                    note="m",
                    tags=("metaworld:declared", "target"),
                )
                for component in _POSITION_COMPONENTS
            }

        prefix = "object:"
        if not declaration_key.startswith(prefix):
            raise MetaworldDeclarationError(
                "Unknown Meta-World declaration key {!r}".format(
                    declaration_key
                )
            )
        try:
            index = int(declaration_key[len(prefix) :])
            item = self._task_declaration.objects[index]
            orientation = self._task_declaration.orientations[index]
        except (ValueError, IndexError) as exc:
            raise MetaworldDeclarationError(
                "Unknown Meta-World declaration key {!r}".format(
                    declaration_key
                )
            ) from exc

        role = self._object_role(index)
        tags = (
            "metaworld:declared",
            "position_source_kind:{}".format(item.kind),
            "role:" + role,
        )
        metadata: Dict[str, PropertyMeta] = {
            "object_pos_{}".format(component): PropertyMeta(
                group="declared object position",
                note="m",
                tags=tags,
            )
            for component in _POSITION_COMPONENTS
        }
        if orientation.has_semantic_orientation:
            orientation_tags = (
                "metaworld:declared",
                "role:" + role,
            ) + tuple(
                "orientation_source_kind:{}".format(source.kind)
                for source in orientation.sources
            )
            metadata.update(
                {
                    "object_quat_{}".format(component): PropertyMeta(
                        group="declared object orientation",
                        note="quat {}".format(component),
                        tags=orientation_tags,
                    )
                    for component in _QUATERNION_COMPONENTS
                }
            )
        if self._spatial_source(item) is not None:
            metadata.update(
                {
                    "object_linvel_{}".format(component): PropertyMeta(
                        group="declared object linear velocity",
                        note="m/s",
                        tags=tags,
                    )
                    for component in _POSITION_COMPONENTS
                }
            )
        return metadata

    def property_metadata(
        self,
        entity_id: int,
    ) -> Dict[str, PropertyMeta]:
        """Apply the declared object role to every routed source field."""
        metadata = super().property_metadata(entity_id)
        role = getattr(self, "entity_roles", {}).get(entity_id)
        if role is None:
            return metadata
        role_tag = "role:" + role
        return {
            key: meta._replace(
                tags=(
                    meta.tags
                    if role_tag in meta.tags
                    else tuple(meta.tags) + (role_tag,)
                )
            )
            for key, meta in metadata.items()
        }


def _make_metaworld(
    task: Optional[str],
    robot: Optional[str],
    target_only: bool,
    seed: Optional[int] = None,
) -> MetaworldAdapter:
    """Construct one reset Meta-World MT1 environment and its adapter.

    ``seed`` reaches ``metaworld.MT1``, which draws the task variants
    (initial object and goal positions; the goal site lands in the compiled
    model) from it. ``None`` draws another variant per construction.
    """
    del robot, target_only

    try:
        import metaworld
    except ImportError as exc:
        raise ImportError(
            "Meta-World is required for the 'metaworld' adapter."
        ) from exc

    resolved_task = task or DEFAULT_TASK
    benchmark = metaworld.MT1(resolved_task, seed=seed)
    env = benchmark.train_classes[resolved_task]()
    env.set_task(benchmark.train_tasks[0])
    env.reset()
    return MetaworldAdapter(env, task_name=resolved_task)


def _handles(obj: Any) -> bool:
    """Cheap module-path sniff that never imports Meta-World."""
    return (type(obj).__module__ or "").startswith("metaworld")


def _wrap(env: Any, **kwargs: Any) -> MetaworldAdapter:
    """Wrap an existing exact Meta-World v3 environment."""
    return MetaworldAdapter(env, **kwargs)


_register(
    name="metaworld",
    factory=_make_metaworld,
    description="Meta-World Sawyer benchmark (MT1 tasks, MuJoCo)",
    install_hint="pip install metaworld",
    handles=_handles,
    wrap=_wrap,
    requires=("metaworld", "mujoco"),
)


# Bounded migration provider (maintainer decision 2026-09-07): the wrapper
# row inherits the engine provider and adds its own source custody.
from .mujoco_provider import install_bounded_provider as _install_bounded_provider  # noqa: E402

_install_bounded_provider(MetaworldAdapter)
