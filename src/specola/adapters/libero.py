"""
adapters/libero.py — LIBERO SimulatorAdapter
============================================

MujocoAdapter subclass for the LIBERO benchmark (Liu et al., NeurIPS 2023).
LIBERO wraps robosuite 1.4 with a Franka Panda arm and language-conditioned
manipulation tasks across four suites (Spatial, Object, Goal, Long/100).

This module defines only behavior that depends on LIBERO:

- declaration-backed task-object and fixture classification;
- declaration-backed semantic-region registration;
- robosuite 1.4 gripper-site conventions;
- gripper finger-joint properties;
- LIBERO environment construction and registry integration.

Generic MuJoCo body, site, joint, pose, velocity, and scalar property
extraction remains in adapters/mujoco.py.

The wrapper-versus-inner-environment trap
-----------------------------------------
LIBERO's outer ``ControlEnv``/``OffScreenRenderEnv`` does not expose the
declaration dictionaries. The task environment stored at ``env.env`` does: it owns
``parsed_problem``, ``objects_dict``, ``object_states_dict``,
``object_sites_dict``, ``fixtures_dict``, and ``obj_of_interest``. Names and
roles come from those declarations. Every instantiated declaration is
reconciled to an exact compiled body or site, and the compiled model and
``MjData`` remain authoritative for physical values. Missing or ambiguous
reconciliation raises; there is no compiled-name fallback.

Usage
-----
    from specola.adapters.libero import LIBEROAdapter, make_libero_env

    env, task = make_libero_env("libero_object", task_id=0)
    adapter   = LIBEROAdapter(env, task=task)

    obs    = adapter.reset()
    runner = MonitorRunner("libero_specs.yaml",
                           dt=adapter.dt,
                           spatial_dims=adapter.spatial_dims)
    for _ in range(horizon):
        env.step(action)
        results = runner.update(adapter.get_observation())
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..adapter import PropertyMeta
from ..entities import EntityState
from .mujoco import (
    BodyRule,
    DeclaredEntitySpec,
    MujocoAdapter,
    MujocoJointOwner,
    RegionSpec,
    _CURATED_FIELD_OWNER_ENTITY_CLASSES,
    _ORPHAN_ACTUATOR_ENTITY_CLASS,
    _ORPHAN_SENSOR_ENTITY_CLASS,
    _property_token,
)


_OBJECT_RELATION_CLASS = "object_relation"
_OBJECT_REGION_RELATION_CLASS = "object_region_relation"


@dataclass(frozen=True)
class _LIBERODeclaredObject:
    """Criterion-independent identity needed by generic relation state."""

    name: str
    body_id: int
    contact_geom_ids: Tuple[int, ...]


@dataclass(frozen=True)
class _LIBEROObjectRelation:
    source: _LIBERODeclaredObject
    destination: _LIBERODeclaredObject

    @property
    def declaration_key(self) -> str:
        return "object_relation:{}__to__{}".format(
            self.source.name, self.destination.name
        )


@dataclass(frozen=True)
class _LIBEROObjectRegionRelation:
    source: _LIBERODeclaredObject
    region_name: str
    site_id: int
    half_extents: Tuple[float, float, float]
    parent_name: Optional[str]
    parent_contact_geom_ids: Tuple[int, ...]

    @property
    def declaration_key(self) -> str:
        return "object_region_relation:{}__to__{}".format(
            self.source.name, self.region_name
        )


# Candidate gripper sites in preference order. robosuite 1.4 (LIBERO's
# pinned version) uses gripper0_grip_site for the Panda.
_DEFAULT_GRIPPER_SITES: tuple = (
    "gripper0_grip_site",
    "gripper0_right_grip_site",
    "robot0_right_hand",
)

# Version-matched LIBERO arena source maps each BDDL workspace declaration
# to this exact compiled body name. Table and kitchen arenas both compile
# their workspace body as ``table``; the remaining arenas preserve their
# specific body names.
_ARENA_ROOT_BODY_BY_TYPE = {
    "table": "table",
    "kitchen": "table",
    "floor": "floor",
    "coffee_table": "coffee_table",
    "living_room": "living_room_table",
    "study": "study_table",
}


@dataclass(frozen=True)
class _LIBEROArenaFixtureDisposition:
    """A parsed workspace fixture owned by LIBERO's arena, not fixtures_dict."""

    name: str
    category: str
    root_body: str
    disposition: str
    reason: str


@dataclass(frozen=True)
class _LIBERODeclarationSnapshot:
    """One reset's reconciled task declarations.

    ``parsed_problem["regions"]`` can contain scene templates for fixtures
    that were not instantiated in the task. Only ``object_sites_dict`` is
    the instantiated region population. ``uninstantiated_region_names``
    records the template-only remainder without manufacturing runtime
    entities for it.
    """

    parsed_object_names: Tuple[str, ...]
    parsed_fixture_names: Tuple[str, ...]
    objects: Tuple[Tuple[str, str], ...]
    object_roles: Tuple[Tuple[str, str, str, str], ...]
    fixtures: Tuple[Tuple[str, str], ...]
    arena_fixtures: Tuple[_LIBEROArenaFixtureDisposition, ...]
    regions: Tuple[RegionSpec, ...]
    uninstantiated_region_names: Tuple[str, ...]
    object_relations: Tuple[_LIBEROObjectRelation, ...]
    object_region_relations: Tuple[_LIBEROObjectRegionRelation, ...]

    @property
    def object_roots(self) -> FrozenSet[str]:
        return frozenset(root for _name, root in self.objects)

    @property
    def fixture_roots(self) -> FrozenSet[str]:
        return frozenset(
            [root for _name, root in self.fixtures]
            + [record.root_body for record in self.arena_fixtures]
        )

    @property
    def declared_name_by_root(self) -> Dict[str, str]:
        return {
            root: name
            for name, root in (
                self.objects
                + self.fixtures
                + tuple(
                    (record.name, record.root_body)
                    for record in self.arena_fixtures
                )
            )
        }

    @property
    def fixture_name_by_root(self) -> Dict[str, str]:
        return {
            root: name
            for name, root in (
                self.fixtures
                + tuple(
                    (record.name, record.root_body)
                    for record in self.arena_fixtures
                )
            )
        }

    @property
    def role_by_object_name(self) -> Dict[str, Tuple[str, str, str]]:
        return {
            name: (role, basis, provenance)
            for name, role, basis, provenance in self.object_roles
        }


def _libero_object_roles(
    parsed_object_names: Sequence[str],
    declared_interest_names: Sequence[str],
    obj_of_interest: Any,
    goal_state: Any,
) -> Tuple[Tuple[str, str, str, str], ...]:
    """Partition only parsed ordinary objects by exact BDDL interests.

    Fixtures and regions may also occur in ``obj_of_interest``. They are
    validated as declared symbols, but they are not silently promoted into
    this ordinary-object role universe.
    """
    if not isinstance(obj_of_interest, (list, tuple)):
        raise RuntimeError(
            "LIBERO obj_of_interest must be a list or tuple"
        )
    if not isinstance(goal_state, (list, tuple)):
        raise RuntimeError(
            'LIBERO parsed_problem["goal_state"] must be a list or tuple'
        )
    interests = tuple(obj_of_interest)
    if (
        any(not isinstance(name, str) or not name for name in interests)
        or len(set(interests)) != len(interests)
    ):
        raise RuntimeError(
            "LIBERO obj_of_interest must contain unique non-empty names"
        )

    object_names = frozenset(parsed_object_names)
    interests_set = set(interests)
    unknown_interests = sorted(
        interests_set - set(declared_interest_names)
    )
    if unknown_interests:
        raise RuntimeError(
            "LIBERO obj_of_interest contains undeclared symbols: {}".format(
                ", ".join(repr(name) for name in unknown_interests)
            )
        )
    for index, state in enumerate(goal_state):
        if (
            not isinstance(state, (list, tuple))
            or len(state) < 2
            or not isinstance(state[0], str)
            or any(
                not isinstance(operand, str) or not operand
                for operand in state[1:]
            )
        ):
            raise RuntimeError(
                "LIBERO goal_state[{}] is not a predicate with named "
                "operands".format(index)
            )
        for operand in state[1:]:
            if operand in object_names and operand not in interests_set:
                raise RuntimeError(
                    "LIBERO goal operand {!r} is absent from "
                    "obj_of_interest".format(operand)
                )

    records = []
    for name in parsed_object_names:
        if name in interests_set:
            records.append(
                (
                    name,
                    "target",
                    "structural_derivation",
                    "obj_of_interest member; goal operands cross-checked",
                )
            )
        else:
            records.append(
                (
                    name,
                    "distractor",
                    "structural_derivation",
                    'parsed_problem["objects"] minus obj_of_interest',
                )
            )
    return tuple(records)


class LIBEROAdapter(MujocoAdapter):
    """
    MuJoCo adapter specialized for the LIBERO benchmark.

    Parameters
    ----------
    env:
        Constructed LIBERO ``OffScreenRenderEnv`` (or any LIBERO env
        exposing ``env.sim`` and robosuite ``env.robots``).

    task:
        Optional LIBERO task object. Provides ``task.language`` for the
        task-description line at load time. When omitted, the adapter
        falls back to ``env.language_instruction``.

    target_only:
        Retained for registry compatibility. It does not select discovery:
        every instantiated declaration-backed region is registered.

    All other parameters are forwarded to MujocoAdapter.
    """

    def __init__(
        self,
        env: Any,
        task: Any = None,
        body_rules: Optional[Sequence[BodyRule]] = None,
        regions: Optional[Sequence[RegionSpec]] = None,
        gripper_site: Optional[str] = None,
        perception_gate: Optional[Callable[[EntityState, Any], bool]] = None,
        target_only: bool = False,
    ) -> None:
        self.task = task
        self.target_only = target_only

        # Built lazily after MujocoAdapter resolves the live model. The
        # snapshot is invalidated before every adapter reset.
        self._declaration_snapshot_value: Optional[
            _LIBERODeclarationSnapshot
        ] = None
        self.entity_roles: Dict[int, str] = {}
        self.entity_role_basis: Dict[int, str] = {}
        self.entity_role_provenance: Dict[int, str] = {}
        self.entity_declared_names: Dict[int, str] = {}
        self._contact_role_by_body: Dict[int, str] = {}

        super().__init__(
            env,
            body_rules=body_rules,
            regions=regions,
            gripper_site=gripper_site,
            perception_gate=perception_gate,
        )

    # ------------------------------------------------------------------
    # Adapter capability metadata
    # ------------------------------------------------------------------

    @property
    def supported_classes(self) -> FrozenSet[str]:
        return frozenset(
            {
                "robot",
                "object",
                "fixture",
                "region",
                _OBJECT_RELATION_CLASS,
                _OBJECT_REGION_RELATION_CLASS,
                self.gripper_class,
                # Synthetic entity carrying sensors with no resolvable
                # body/site owner (see adapters/mujoco.py). Any reset
                # that produces an orphan sensor must not be rejected by
                # a closed vocabulary that omits this class.
                _ORPHAN_SENSOR_ENTITY_CLASS,
                # Synthetic entity carrying actuators with no resolvable
                # body owner (see adapters/mujoco.py). Any reset that
                # produces an orphan actuator must not be rejected by a
                # closed vocabulary that omits this class.
                _ORPHAN_ACTUATOR_ENTITY_CLASS,
            }
            | _CURATED_FIELD_OWNER_ENTITY_CLASSES
        )

    # ------------------------------------------------------------------
    # Declaration/model reconciliation
    # ------------------------------------------------------------------

    _INNER_DECLARATION_ATTRIBUTES = (
        "parsed_problem",
        "objects_dict",
        "object_states_dict",
        "object_sites_dict",
        "fixtures_dict",
        "obj_of_interest",
        "workspace_name",
    )

    def _libero_inner_env(self) -> Any:
        """Return LIBERO's declaration-owning environment.

        Walk wrapper ``env`` links until the declaration owner is found.
        Failing loudly prevents a wrapper change from silently re-enabling
        compiled-name inference.
        """
        candidate = self.env
        seen = set()
        last_missing = list(self._INNER_DECLARATION_ATTRIBUTES)
        while candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            last_missing = [
                name
                for name in self._INNER_DECLARATION_ATTRIBUTES
                if not hasattr(candidate, name)
            ]
            if not last_missing:
                return candidate
            next_candidate = getattr(candidate, "env", None)
            if next_candidate is candidate:
                break
            candidate = next_candidate

        raise RuntimeError(
            "LIBERO declaration-owning environment is missing "
            f"attributes {last_missing}; inspected env wrapper chain"
        )

    @staticmethod
    def _require_mapping(value: Any, source: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise RuntimeError(
                f"LIBERO {source} must be a mapping, got "
                f"{type(value).__name__}"
            )
        return value

    @classmethod
    def _parsed_named_population(
        cls,
        value: Any,
        source: str,
    ) -> Tuple[Tuple[str, str], ...]:
        """Flatten a parsed ``category -> [name]`` declaration mapping."""
        mapping = cls._require_mapping(value, source)
        entries = []
        for category, names in mapping.items():
            if not isinstance(category, str) or not category:
                raise RuntimeError(
                    f"LIBERO {source} contains invalid category "
                    f"{category!r}"
                )
            if not isinstance(names, (list, tuple)):
                raise RuntimeError(
                    f"LIBERO {source}[{category!r}] must be a list or "
                    f"tuple, got {type(names).__name__}"
                )
            for name in names:
                if not isinstance(name, str) or not name:
                    raise RuntimeError(
                        f"LIBERO {source}[{category!r}] contains invalid "
                        f"name {name!r}"
                    )
                entries.append((name, category))

        duplicate_names = sorted(
            name
            for name in {name for name, _category in entries}
            if sum(
                candidate == name
                for candidate, _category in entries
            ) != 1
        )
        if duplicate_names:
            raise RuntimeError(
                f"LIBERO {source} declares names more than once: "
                f"{duplicate_names}"
            )
        return tuple(sorted(entries))

    @staticmethod
    def _declared_root(
        declaration_name: Any,
        declaration: Any,
        source: str,
    ) -> Tuple[str, str]:
        if not isinstance(declaration_name, str) or not declaration_name:
            raise RuntimeError(
                f"LIBERO {source} contains a non-string declaration name "
                f"{declaration_name!r}"
            )
        root_body = getattr(declaration, "root_body", None)
        if not isinstance(root_body, str) or not root_body:
            raise RuntimeError(
                f"LIBERO {source} declaration {declaration_name!r} "
                "has no non-empty root_body"
            )
        return declaration_name, root_body

    def _declared_contact_geom_ids(
        self,
        declaration_name: str,
        declaration: Any,
        source: str,
    ) -> Tuple[int, ...]:
        """Resolve the exact geom population used by robosuite contact.

        ``Environment.check_contact`` iterates every native contact row and
        compares geom names against each model declaration's
        ``contact_geoms``.  The adapter freezes those same declaration-backed
        geom IDs at reset; it does not call the benchmark predicate.
        """
        geom_names = getattr(declaration, "contact_geoms", None)
        if not isinstance(geom_names, (list, tuple)):
            raise RuntimeError(
                "LIBERO {} declaration {!r} has no list/tuple "
                "contact_geoms".format(source, declaration_name)
            )
        available = tuple(self._names("geom"))
        resolved = []
        for geom_name in geom_names:
            if not isinstance(geom_name, str) or not geom_name:
                raise RuntimeError(
                    "LIBERO {} declaration {!r} contains invalid contact "
                    "geom {!r}".format(source, declaration_name, geom_name)
                )
            count = available.count(geom_name)
            if count != 1:
                raise RuntimeError(
                    "LIBERO {} declaration {!r} contact geom {!r} "
                    "resolved {} times; expected exactly one".format(
                        source, declaration_name, geom_name, count
                    )
                )
            resolved.append(self._name2id("geom", geom_name))
        if len(set(resolved)) != len(resolved):
            raise RuntimeError(
                "LIBERO {} declaration {!r} repeats contact geoms".format(
                    source, declaration_name
                )
            )
        return tuple(resolved)

    def _declaration_snapshot(self) -> _LIBERODeclarationSnapshot:
        """Reconcile one reset's instantiated declarations to the model."""
        cached = self._declaration_snapshot_value
        if cached is not None:
            return cached

        inner = self._libero_inner_env()
        objects_dict = self._require_mapping(
            inner.objects_dict,
            "objects_dict",
        )
        fixtures_dict = self._require_mapping(
            inner.fixtures_dict,
            "fixtures_dict",
        )
        object_sites_dict = self._require_mapping(
            inner.object_sites_dict,
            "object_sites_dict",
        )
        parsed_problem = self._require_mapping(
            inner.parsed_problem,
            "parsed_problem",
        )
        parsed_object_entries = self._parsed_named_population(
            parsed_problem.get("objects"),
            'parsed_problem["objects"]',
        )
        parsed_fixture_entries = self._parsed_named_population(
            parsed_problem.get("fixtures"),
            'parsed_problem["fixtures"]',
        )
        parsed_regions = self._require_mapping(
            parsed_problem.get("regions"),
            'parsed_problem["regions"]',
        )
        parsed_object_names = tuple(
            name for name, _category in parsed_object_entries
        )
        parsed_fixture_names = tuple(
            name for name, _category in parsed_fixture_entries
        )

        objects = tuple(
            sorted(
                self._declared_root(name, declaration, "objects_dict")
                for name, declaration in objects_dict.items()
            )
        )
        fixtures = tuple(
            sorted(
                self._declared_root(name, declaration, "fixtures_dict")
                for name, declaration in fixtures_dict.items()
            )
        )
        runtime_object_names = tuple(name for name, _root in objects)
        runtime_fixture_names = tuple(name for name, _root in fixtures)
        if set(runtime_object_names) != set(parsed_object_names):
            raise RuntimeError(
                "LIBERO parsed/runtime object declaration mismatch: "
                f"missing_from_runtime={sorted(set(parsed_object_names) - set(runtime_object_names))}, "
                f"unexpected_runtime={sorted(set(runtime_object_names) - set(parsed_object_names))}"
            )

        workspace_name = getattr(inner, "workspace_name", None)
        if not isinstance(workspace_name, str) or not workspace_name:
            raise RuntimeError(
                "LIBERO workspace_name must be a non-empty string"
            )
        parsed_fixture_categories = dict(parsed_fixture_entries)
        arena_fixture_names = (
            set(parsed_fixture_names) - set(runtime_fixture_names)
        )
        unexpected_runtime_fixtures = (
            set(runtime_fixture_names) - set(parsed_fixture_names)
        )
        if (
            arena_fixture_names != {workspace_name}
            or unexpected_runtime_fixtures
        ):
            raise RuntimeError(
                "LIBERO parsed/runtime fixture declaration mismatch: "
                f"workspace_name={workspace_name!r}, "
                f"parsed_without_runtime_object={sorted(arena_fixture_names)}, "
                f"unexpected_runtime={sorted(unexpected_runtime_fixtures)}"
            )
        arena_type = getattr(inner, "_arena_type", None)
        arena_root_body = _ARENA_ROOT_BODY_BY_TYPE.get(arena_type)
        if arena_root_body is None:
            raise RuntimeError(
                "LIBERO arena type {!r} has no verified compiled workspace "
                "body mapping".format(arena_type)
            )
        model_body_names = tuple(self._names("body"))
        if model_body_names.count(arena_root_body) != 1:
            raise RuntimeError(
                "LIBERO workspace {!r} arena type {!r} maps to compiled "
                "body {!r}, which resolved {} times; expected exactly one"
                .format(
                    workspace_name,
                    arena_type,
                    arena_root_body,
                    model_body_names.count(arena_root_body),
                )
            )
        arena_fixtures = (
            _LIBEROArenaFixtureDisposition(
                name=workspace_name,
                category=parsed_fixture_categories[workspace_name],
                root_body=arena_root_body,
                disposition="arena_owned_compiled_body",
                reason=(
                    "LIBERO workspace fixture is instantiated by the arena "
                    "rather than fixtures_dict; its exact compiled root body "
                    "comes from the version-matched arena implementation"
                ),
            ),
        )

        declared_pairs = (
            objects
            + fixtures
            + ((workspace_name, arena_root_body),)
        )
        declared_names = [name for name, _root in declared_pairs]
        declared_roots = [root for _name, root in declared_pairs]
        duplicate_names = sorted(
            name
            for name in set(declared_names)
            if declared_names.count(name) != 1
        )
        duplicate_roots = sorted(
            root
            for root in set(declared_roots)
            if declared_roots.count(root) != 1
        )
        if duplicate_names or duplicate_roots:
            raise RuntimeError(
                "LIBERO declarations must map one semantic entity to one "
                f"root body; duplicate names={duplicate_names}, "
                f"duplicate roots={duplicate_roots}"
            )

        for declaration_name, root_body in declared_pairs:
            count = model_body_names.count(root_body)
            if count != 1:
                raise RuntimeError(
                    f"LIBERO declaration {declaration_name!r} root body "
                    f"{root_body!r} resolved {count} times in the model; "
                    "expected exactly one"
                )

        declared_objects = tuple(
            _LIBERODeclaredObject(
                name=name,
                body_id=self._name2id("body", root_body),
                contact_geom_ids=self._declared_contact_geom_ids(
                    name, objects_dict[name], "objects_dict"
                ),
            )
            for name, root_body in objects
        )
        object_relations = tuple(
            _LIBEROObjectRelation(
                source=source,
                destination=destination,
            )
            for source in declared_objects
            for destination in declared_objects
            if source.name != destination.name
        )

        model_site_names = tuple(self._names("site"))
        regions = []
        region_relation_sources = []
        for site_name, declaration in sorted(object_sites_dict.items()):
            if not isinstance(site_name, str) or not site_name:
                raise RuntimeError(
                    "LIBERO object_sites_dict contains a non-string "
                    f"declaration name {site_name!r}"
                )
            if site_name not in parsed_regions:
                raise RuntimeError(
                    f"LIBERO instantiated region {site_name!r} is absent "
                    'from parsed_problem["regions"]'
                )
            count = model_site_names.count(site_name)
            if count != 1:
                raise RuntimeError(
                    f"LIBERO instantiated region {site_name!r} resolved "
                    f"{count} times in the model; expected exactly one"
                )

            site_type = getattr(declaration, "site_type", None)
            if site_type != "box":
                raise RuntimeError(
                    f"LIBERO instantiated region {site_name!r} declares "
                    f"site_type={site_type!r}; only box regions have "
                    "RegionSpec semantics"
                )

            declared_size = np.asarray(
                getattr(declaration, "size", None),
                dtype=float,
            ).reshape(-1)
            if (
                declared_size.shape != (3,)
                or not np.all(np.isfinite(declared_size))
                or np.any(declared_size < 0.0)
            ):
                raise RuntimeError(
                    f"LIBERO instantiated region {site_name!r} has invalid "
                    f"declared box half-extents {declared_size.tolist()}"
                )

            site_id = self._name2id("site", site_name)
            model_site_type = self._geom_type_name(site_id, site=True)
            if model_site_type != "box":
                raise RuntimeError(
                    f"LIBERO model site {site_name!r} has compiled "
                    f"type={model_site_type!r}; expected 'box' to match "
                    "the declaration"
                )
            model_size = np.asarray(
                self._model.site_size[site_id],
                dtype=float,
            ).reshape(-1)
            if (
                model_size.shape != (3,)
                or not np.all(np.isfinite(model_size))
                or np.any(model_size < 0.0)
            ):
                raise RuntimeError(
                    f"LIBERO model site {site_name!r} has invalid box "
                    f"half-extents {model_size.tolist()}"
                )
            if not np.allclose(
                declared_size,
                model_size,
                rtol=1e-9,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"LIBERO region {site_name!r} declaration/model size "
                    f"mismatch: {declared_size.tolist()} != "
                    f"{model_size.tolist()}"
                )

            regions.append(
                RegionSpec(
                    name=site_name,
                    source="site",
                    source_name=site_name,
                    half_extents=tuple(float(value) for value in model_size),
                    cls="region",
                )
            )
            parent_name = getattr(declaration, "parent_name", None)
            if parent_name is not None and (
                not isinstance(parent_name, str) or not parent_name
            ):
                raise RuntimeError(
                    f"LIBERO instantiated region {site_name!r} has invalid "
                    f"parent_name={parent_name!r}"
                )
            if parent_name is None:
                parent_contact_geom_ids = ()
            elif parent_name in objects_dict:
                parent_contact_geom_ids = self._declared_contact_geom_ids(
                    parent_name,
                    objects_dict[parent_name],
                    "objects_dict",
                )
            elif parent_name in fixtures_dict:
                parent_contact_geom_ids = self._declared_contact_geom_ids(
                    parent_name,
                    fixtures_dict[parent_name],
                    "fixtures_dict",
                )
            else:
                raise RuntimeError(
                    f"LIBERO instantiated region {site_name!r} names "
                    f"undeclared parent {parent_name!r}"
                )
            region_relation_sources.append(
                (
                    site_name,
                    site_id,
                    tuple(float(value) for value in model_size),
                    parent_name,
                    parent_contact_geom_ids,
                )
            )

        uninstantiated = tuple(
            sorted(set(parsed_regions) - set(object_sites_dict))
        )
        object_roles = _libero_object_roles(
            parsed_object_names,
            tuple(
                sorted(
                    set(parsed_object_names)
                    | set(parsed_fixture_names)
                    | set(parsed_regions)
                )
            ),
            inner.obj_of_interest,
            parsed_problem.get("goal_state"),
        )
        object_region_relations = tuple(
            _LIBEROObjectRegionRelation(
                source=source,
                region_name=region_name,
                site_id=site_id,
                half_extents=half_extents,
                parent_name=parent_name,
                parent_contact_geom_ids=parent_contact_geom_ids,
            )
            for source in declared_objects
            for (
                region_name,
                site_id,
                half_extents,
                parent_name,
                parent_contact_geom_ids,
            ) in region_relation_sources
        )
        snapshot = _LIBERODeclarationSnapshot(
            parsed_object_names=parsed_object_names,
            parsed_fixture_names=parsed_fixture_names,
            objects=objects,
            object_roles=object_roles,
            fixtures=fixtures,
            arena_fixtures=arena_fixtures,
            regions=tuple(regions),
            uninstantiated_region_names=uninstantiated,
            object_relations=object_relations,
            object_region_relations=object_region_relations,
        )
        self._declaration_snapshot_value = snapshot
        return snapshot

    def _invalidate_declaration_snapshot(self) -> None:
        self._declaration_snapshot_value = None

    @property
    def uninstantiated_declared_regions(self) -> Tuple[str, ...]:
        """Parsed region templates that are not instantiated this reset."""
        return self._declaration_snapshot().uninstantiated_region_names

    def _declared_fixture_name(self, body_name: str) -> Optional[str]:
        """Fixture whose declared root body is ``body_name``."""
        return self._declaration_snapshot().fixture_name_by_root.get(
            body_name
        )

    # ------------------------------------------------------------------
    # MujocoAdapter hooks
    # ------------------------------------------------------------------

    def _default_body_rules(self) -> Sequence[BodyRule]:
        snapshot = self._declaration_snapshot()
        fixture_roots = snapshot.fixture_roots
        object_roots = snapshot.object_roots
        return (
            # The complete articulated Panda is represented by its base
            # body; subtree exposes arm joint state as properties.
            BodyRule(
                match=lambda name: name == "robot0_base",
                cls="robot",
                priority=0,
                joint_scope="subtree",
            ),
            # Fixture roles and their exact compiled roots come from the
            # declarations. Subtree scope exposes articulated child joints.
            BodyRule(
                match=lambda name: name in fixture_roots,
                cls="fixture",
                priority=5,
                joint_scope="subtree",
            ),
            # Movable task-object roles and exact roots are declaration
            # backed. Their free joint belongs to the root body.
            BodyRule(
                match=lambda name: name in object_roots,
                cls="object",
                priority=10,
                joint_scope="body",
            ),
        )

    @staticmethod
    def _relation_tags(source: str, destination: str) -> Tuple[str, ...]:
        return (
            "source_declared_name:" + source,
            "destination_declared_name:" + destination,
            "declared_pair:{}__to__{}".format(source, destination),
        )

    def _relation_contact_provenance_tags(
        self,
        source_geom_ids: Sequence[int],
        destination_geom_ids: Sequence[int],
        *,
        destination_role: str = "destination",
    ) -> Tuple[str, ...]:
        """Expose the declaration-to-geom reduction in discovery metadata."""
        def geom_name(geom_id: int) -> str:
            name = self._id2name("geom", geom_id)
            if name is None:
                raise RuntimeError(
                    "LIBERO declared contact geom id {} has no compiled "
                    "name".format(geom_id)
                )
            return name

        return tuple(
            "source_contact_geom:" + geom_name(geom_id)
            for geom_id in source_geom_ids
        ) + tuple(
            "{}_contact_geom:{}".format(
                destination_role,
                geom_name(geom_id),
            )
            for geom_id in destination_geom_ids
        )

    def _declare_task_entities(self) -> Sequence[DeclaredEntitySpec]:
        """Declare every criterion-independent ordered relation at reset."""
        snapshot = self._declaration_snapshot()
        object_specs = tuple(
            DeclaredEntitySpec(
                declaration_key=relation.declaration_key,
                name="{} -> {}".format(
                    relation.source.name, relation.destination.name
                ),
                entity_class=_OBJECT_RELATION_CLASS,
                source_kind="libero_declaration_relation",
                source_name="{}__to__{}".format(
                    relation.source.name, relation.destination.name
                ),
                source_provenance=(
                    "LIBERO objects_dict declarations and installed MuJoCo "
                    "model; static goal-independent ordered pair"
                ),
                tags=self._relation_tags(
                    relation.source.name, relation.destination.name
                ),
            )
            for relation in snapshot.object_relations
        )
        region_specs = tuple(
            DeclaredEntitySpec(
                declaration_key=relation.declaration_key,
                name="{} -> {}".format(
                    relation.source.name, relation.region_name
                ),
                entity_class=_OBJECT_REGION_RELATION_CLASS,
                source_kind="libero_declaration_relation",
                source_name="{}__to__{}".format(
                    relation.source.name, relation.region_name
                ),
                source_provenance=(
                    "LIBERO objects_dict/object_sites_dict declarations "
                    "and installed MuJoCo model; static goal-independent "
                    "ordered pair"
                ),
                tags=self._relation_tags(
                    relation.source.name, relation.region_name
                ),
            )
            for relation in snapshot.object_region_relations
        )
        return object_specs + region_specs

    def _collect_native_contact_pairs(self) -> FrozenSet[FrozenSet[int]]:
        """Return all native contact geom pairs, matching robosuite 1.4."""
        pairs = set()
        for index in range(int(self._data.ncon)):
            contact = self._data.contact[index]
            geoms = getattr(contact, "geom", None)
            if geoms is None:
                geoms = (contact.geom1, contact.geom2)
            pairs.add(frozenset((int(geoms[0]), int(geoms[1]))))
        return frozenset(pairs)

    def _active_contact_geom_pairs(self) -> FrozenSet[FrozenSet[int]]:
        """Contact pairs from the buffer as the caller presented it.

        LIBERO's success check runs on the contact buffer exactly as
        ``env.step`` left it. The adapter's observation pass runs a
        kinematic ``forward()`` for position self-consistency, and that
        forward re-runs collision detection from post-integration
        positions, flipping marginal resting contacts in both directions
        (probe, 2026-08-17: over 8,730 instrumented step reads the
        declared geom sets matched ``check_contact`` on every read of the
        as-presented buffer, while the post-forward re-detection dropped
        7 real contacts — among them success frames — and invented 3).
        Contact state therefore reads the snapshot taken at observation
        entry, before the forward; positions keep the recorded
        self-consistency decision. On the replay path the driver runs
        ``forward()`` after ``set_state`` before observing, so the
        presented buffer is the current frame there as well.

        Documented asymmetry: the generic contact entities (forces,
        pair population) still read the post-forward buffer, because
        force magnitudes require the constraint solve the forward
        performs. On a marginal resting frame the declared relation can
        therefore report contact while the generic pair entity is
        absent; the declared read is the benchmark-faithful one.
        """
        pairs = getattr(self, "_presented_contact_pairs", None)
        if pairs is None:
            return self._collect_native_contact_pairs()
        return pairs

    def _get_observation_impl(self):
        self._presented_contact_pairs = self._collect_native_contact_pairs()
        try:
            return super()._get_observation_impl()
        finally:
            self._presented_contact_pairs = None

    @staticmethod
    def _declared_geoms_touch(
        first: Sequence[int],
        second: Sequence[int],
        active_pairs: FrozenSet[FrozenSet[int]],
    ) -> float:
        return float(
            any(
                frozenset((first_geom, second_geom)) in active_pairs
                for first_geom in first
                for second_geom in second
            )
        )

    def _declared_entity_states(
        self,
        registrations: Mapping[str, int],
    ) -> Mapping[str, EntityState]:
        """Materialize generic raw relation state without criterion calls."""
        snapshot = self._declaration_snapshot()
        active_pairs = self._active_contact_geom_pairs()
        states: Dict[str, EntityState] = {}

        for relation in snapshot.object_relations:
            source_position = np.asarray(
                self._data.body_xpos[relation.source.body_id], dtype=float
            )
            destination_position = np.asarray(
                self._data.body_xpos[relation.destination.body_id],
                dtype=float,
            )
            delta = source_position - destination_position
            state = self._empty_owner_state(
                registrations[relation.declaration_key],
                _OBJECT_RELATION_CLASS,
            )
            state.properties.update(
                {
                    "horizontal_center_distance": float(
                        np.linalg.norm(delta[:2])
                    ),
                    "source_minus_destination_z": float(delta[2]),
                    "declared_geom_contact": self._declared_geoms_touch(
                        relation.source.contact_geom_ids,
                        relation.destination.contact_geom_ids,
                        active_pairs,
                    ),
                }
            )
            states[relation.declaration_key] = state

        for relation in snapshot.object_region_relations:
            source_position = np.asarray(
                self._data.body_xpos[relation.source.body_id], dtype=float
            )
            region_position = np.asarray(
                self._data.site_xpos[relation.site_id], dtype=float
            )
            installed_matrix = np.asarray(
                self._data.site_xmat[relation.site_id], dtype=float
            ).reshape(3, 3)
            half_extents = np.asarray(relation.half_extents, dtype=float)
            world_delta = source_position - region_position
            installed_delta = installed_matrix @ world_delta
            projected_half_extents = np.abs(
                installed_matrix @ half_extents
            )
            parent_present = relation.parent_name is not None
            state = self._empty_owner_state(
                registrations[relation.declaration_key],
                _OBJECT_REGION_RELATION_CLASS,
            )
            state.properties.update(
                {
                    "world_delta_x": float(world_delta[0]),
                    "world_delta_y": float(world_delta[1]),
                    "world_delta_z": float(world_delta[2]),
                    "installed_matrix_delta_x": float(installed_delta[0]),
                    "installed_matrix_delta_y": float(installed_delta[1]),
                    "installed_matrix_delta_z": float(installed_delta[2]),
                    "region_half_extent_x": float(half_extents[0]),
                    "region_half_extent_y": float(half_extents[1]),
                    "region_half_extent_z": float(half_extents[2]),
                    "installed_projected_half_extent_x": float(
                        projected_half_extents[0]
                    ),
                    "installed_projected_half_extent_y": float(
                        projected_half_extents[1]
                    ),
                    "installed_projected_half_extent_z": float(
                        projected_half_extents[2]
                    ),
                    "parent_present": float(parent_present),
                    "source_parent_declared_geom_contact": (
                        self._declared_geoms_touch(
                            relation.source.contact_geom_ids,
                            relation.parent_contact_geom_ids,
                            active_pairs,
                        )
                        if parent_present
                        else 0.0
                    ),
                }
            )
            states[relation.declaration_key] = state
        return states

    def _declared_entity_property_metadata(
        self,
        declaration_key: str,
    ) -> Mapping[str, PropertyMeta]:
        snapshot = self._declaration_snapshot()
        object_by_key = {
            relation.declaration_key: relation
            for relation in snapshot.object_relations
        }
        region_by_key = {
            relation.declaration_key: relation
            for relation in snapshot.object_region_relations
        }
        if declaration_key in object_by_key:
            relation = object_by_key[declaration_key]
            tags = self._relation_tags(
                relation.source.name, relation.destination.name
            ) + self._relation_contact_provenance_tags(
                relation.source.contact_geom_ids,
                relation.destination.contact_geom_ids,
            )
            return {
                "horizontal_center_distance": PropertyMeta(
                    group="object relation horizontal distance",
                    note="m",
                    tags=tags,
                ),
                "source_minus_destination_z": PropertyMeta(
                    group="object relation vertical displacement",
                    note="m",
                    tags=tags,
                ),
                "declared_geom_contact": PropertyMeta(
                    group="declared geometry contact",
                    note="1",
                    tags=tags,
                ),
            }
        if declaration_key in region_by_key:
            relation = region_by_key[declaration_key]
            tags = self._relation_tags(
                relation.source.name, relation.region_name
            ) + (
                "parent_declared_name:" + (
                    relation.parent_name
                    if relation.parent_name is not None
                    else "none"
                ),
            ) + self._relation_contact_provenance_tags(
                relation.source.contact_geom_ids,
                relation.parent_contact_geom_ids,
                destination_role="parent",
            )
            metadata: Dict[str, PropertyMeta] = {}
            for prefix, group, frame in (
                (
                    "world_delta",
                    "object-region world displacement",
                    "world",
                ),
                (
                    "installed_matrix_delta",
                    "object-region installed-matrix displacement",
                    "installed LIBERO site_xmat convention",
                ),
                (
                    "region_half_extent",
                    "declared region half extent",
                    "declared region coordinates",
                ),
                (
                    "installed_projected_half_extent",
                    "installed projected region half extent",
                    "world under installed LIBERO site_xmat convention",
                ),
            ):
                for component in "xyz":
                    metadata[f"{prefix}_{component}"] = PropertyMeta(
                        group=group,
                        note="m",
                        tags=tags,
                        unit="m",
                        component=component,
                        frame=frame,
                    )
            metadata["parent_present"] = PropertyMeta(
                group="declared region parent presence",
                note="1",
                tags=tags,
            )
            metadata[
                "source_parent_declared_geom_contact"
            ] = PropertyMeta(
                group="declared geometry contact",
                note="1",
                tags=tags,
            )
            return metadata
        raise KeyError(
            "unknown LIBERO declared relation {!r}".format(declaration_key)
        )

    def _declared_joint_spec(
        self,
        entity_name: str,
    ) -> Tuple[Optional[str], Tuple[str, ...]]:
        """Pure declaration identity for one fixture or articulated region."""
        inner = self._libero_inner_env()

        if entity_name in inner.fixtures_dict:
            fixture_name = entity_name
            requested_names = tuple(
                getattr(inner.fixtures_dict[fixture_name], "joints", ()) or ()
            )
        elif entity_name in inner.object_sites_dict:
            site = inner.object_sites_dict[entity_name]
            fixture_name = getattr(site, "parent_name", None)
            requested_names = tuple(getattr(site, "joints", ()) or ())
            if fixture_name is None:
                if requested_names:
                    raise RuntimeError(
                        f"LIBERO region {entity_name!r} declares joints "
                        f"{requested_names} without a parent fixture"
                    )
                return None, ()
            if not isinstance(fixture_name, str):
                if not requested_names:
                    return None, ()
                raise RuntimeError(
                    f"LIBERO region {entity_name!r} names unknown parent "
                    f"fixture {fixture_name!r}"
                )
            if fixture_name not in inner.fixtures_dict:
                if not requested_names:
                    return None, ()
                raise RuntimeError(
                    f"LIBERO region {entity_name!r} names unknown parent "
                    f"fixture {fixture_name!r}"
                )
        else:
            return None, ()

        fixture = inner.fixtures_dict[fixture_name]
        fixture_names = tuple(getattr(fixture, "joints", ()) or ())
        invalid_names = [
            name
            for name in fixture_names + requested_names
            if not isinstance(name, str) or not name
        ]
        if invalid_names:
            raise RuntimeError(
                f"LIBERO entity {entity_name!r} contains invalid joint "
                f"names {invalid_names}"
            )
        duplicate_names = sorted(
            name
            for name in set(fixture_names)
            if fixture_names.count(name) != 1
        )
        if duplicate_names:
            raise RuntimeError(
                f"LIBERO fixture {fixture_name!r} declares duplicate "
                f"joints {duplicate_names}"
            )
        missing = [
            name
            for name in requested_names
            if name not in fixture_names
        ]
        if missing:
            raise RuntimeError(
                f"LIBERO entity {entity_name!r} names joints {missing} "
                f"outside fixture {fixture_name!r}"
            )
        return fixture_name, requested_names

    def _declared_joint_names(self, entity_name: str) -> Tuple[str, ...]:
        """Declared joint identity without reading live state."""
        _fixture_name, names = self._declared_joint_spec(entity_name)
        return names

    def _declared_joint_values(
        self,
        entity_name: str,
    ) -> Tuple[Tuple[str, np.ndarray], ...]:
        """Declared joint names and raw qpos values for reconciliation only."""
        inner = self._libero_inner_env()
        fixture_name, requested_names = self._declared_joint_spec(entity_name)
        if fixture_name is None or not requested_names:
            return ()

        fixture = inner.fixtures_dict[fixture_name]
        fixture_names = tuple(getattr(fixture, "joints", ()) or ())
        state = inner.object_states_dict.get(fixture_name)
        getter = getattr(state, "get_joint_state", None)
        if not callable(getter):
            raise RuntimeError(
                f"LIBERO fixture {fixture_name!r} has no raw "
                "get_joint_state() source"
            )

        values = tuple(getter())
        if len(values) != len(fixture_names):
            raise RuntimeError(
                f"LIBERO fixture {fixture_name!r} declares "
                f"{len(fixture_names)} joints but get_joint_state() "
                f"returned {len(values)} values"
            )
        by_name = dict(zip(fixture_names, values))

        return tuple(
            (
                name,
                np.asarray(by_name[name], dtype=float).reshape(-1),
            )
            for name in requested_names
        )

    def _declared_joint_ids(self, entity_name: str) -> Tuple[int, ...]:
        return tuple(
            self._name2id("joint", name)
            for name in self._declared_joint_names(entity_name)
        )

    def _additional_joint_owners(
        self,
    ) -> Sequence[MujocoJointOwner]:
        """Map articulated semantic regions to exact declared model joints."""
        inner = self._libero_inner_env()
        owners = []
        for record in self._regions:
            fixture_name, joint_names = self._declared_joint_spec(
                record.spec.name
            )
            if fixture_name is None or not joint_names:
                continue

            fixture = inner.fixtures_dict[fixture_name]
            root_body = getattr(fixture, "root_body", None)
            if not isinstance(root_body, str) or not root_body:
                raise RuntimeError(
                    f"LIBERO region {record.spec.name!r} has no declared "
                    "fixture root body for articulation ownership"
                )
            owner_body_id = self._name2id("body", root_body)
            joint_ids = tuple(
                self._name2id("joint", name)
                for name in joint_names
            )
            subtree_ids = set(self._subtree_body_ids(owner_body_id))
            misattached = [
                name
                for name, joint_id in zip(joint_names, joint_ids)
                if int(self._model.jnt_bodyid[joint_id]) not in subtree_ids
            ]
            if misattached:
                raise RuntimeError(
                    f"LIBERO region {record.spec.name!r} declares joints "
                    f"{misattached} outside fixture {fixture_name!r} subtree"
                )
            owners.append(
                MujocoJointOwner(
                    entity_id=record.entity_id,
                    owner_body_id=owner_body_id,
                    joint_ids=joint_ids,
                    owner_name=record.spec.name,
                )
            )
        return tuple(owners)

    def _reconcile_declared_qpos(
        self,
        properties: Dict[str, float],
        entity_name: str,
        entity_class: Optional[str] = None,
    ) -> None:
        """Validate LIBERO's raw joint state against model-valued qpos.

        Declarations establish which joints belong to the entity, while
        ``MjData`` remains authoritative for physical values. The raw
        ``ObjectState`` getter is an independent reconciliation source,
        never a replacement for extracted model state.

        ``entity_class`` lets the check see the active extraction scope.
        A key the scope excluded was never extracted, so there is no
        model value to reconcile it against and nothing to assert — but
        a key that is *in* scope and still missing is the materialization
        failure this check exists to catch, scope or no scope. Omitting
        the argument keeps the strict reading, so a caller that forgets
        it fails loudly rather than silently skipping.
        """
        scope = getattr(self, "_extraction_scope", None)
        in_scope = (
            None
            if scope is None or entity_class is None
            else scope.get(entity_class, frozenset())
        )
        for joint_name, values in self._declared_joint_values(entity_name):
            keys = self._signal_keys(
                "qpos",
                _property_token(joint_name),
                int(values.size),
            )
            for key, value in zip(keys, values):
                if key not in properties:
                    if in_scope is not None and key not in in_scope:
                        continue
                    raise RuntimeError(
                        f"LIBERO declared joint {joint_name!r} did not "
                        f"materialize expected property {key!r}"
                    )
                model_value = float(properties[key])
                declared_value = float(value)
                if not np.isclose(
                    model_value,
                    declared_value,
                    rtol=1e-9,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        f"LIBERO raw state for declared joint "
                        f"{joint_name!r} disagrees with MjData at "
                        f"{key!r}: {declared_value!r} != "
                        f"{model_value!r}"
                    )

    def _materialize_curated_fields(
        self,
        perceived: Dict[int, EntityState],
    ) -> None:
        """Reconcile fixture declarations after shared eager extraction.

        The shared MuJoCo materializer owns qpos observation and metadata
        sinks. LIBERO adds only an independent declaration-state assertion,
        after those globally available values have been emitted.
        """
        super()._materialize_curated_fields(perceived)
        fixture_by_root = (
            self._declaration_snapshot().fixture_name_by_root
        )
        for body in self._bodies:
            fixture_name = fixture_by_root.get(body.body_name)
            if fixture_name is None:
                continue
            state = perceived.get(body.entity_id)
            if state is None:
                # The public perception gate intentionally removed this body.
                # Shared materialization already records the omitted owner.
                continue
            self._reconcile_declared_qpos(
                state.properties,
                fixture_name,
                body.entity_class,
            )
        for record in self._regions:
            if not self._declared_joint_names(record.spec.name):
                continue
            state = perceived.get(record.entity_id)
            if state is None:
                continue
            self._reconcile_declared_qpos(
                state.properties,
                record.spec.name,
                record.spec.cls,
            )

    def _default_gripper_sites(self) -> Sequence[str]:
        return _DEFAULT_GRIPPER_SITES

    def _default_robot_bodies(self) -> Sequence[str]:
        """Robosuite prefixes every robot-owned body name.

        Measured on LIBERO compiled models: arm links carry
        ``robot0_``, fingers and end effector ``gripper0_``, and the
        mount stack ``mount0_``. Contacts touching any of these classify
        as ``robot_contact``.
        """
        return ("robot0_", "gripper0_", "mount0_")

    def _object_category(self, body_name: str) -> Optional[str]:
        """Human-readable category derived from the declared entity name."""
        declared_name = (
            self._declaration_snapshot()
            .declared_name_by_root
            .get(body_name)
        )
        if declared_name is None:
            return None
        parts = declared_name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            declared_name = parts[0]
        return declared_name.replace("_", " ")

    def _build_labels(self) -> None:
        """Use declaration names and exact BDDL-derived object roles."""
        super()._build_labels()
        snapshot = self._declaration_snapshot()
        declared_name_by_root = snapshot.declared_name_by_root
        object_roles = snapshot.role_by_object_name
        main_roles: Dict[int, str] = {}
        self.entity_roles = {}
        self.entity_role_basis = {}
        self.entity_role_provenance = {}
        self.entity_declared_names = {}
        for body in self._bodies:
            declared_name = declared_name_by_root.get(body.body_name)
            if declared_name is None:
                continue
            category = self._object_category(body.body_name)
            self.entity_labels[body.entity_id] = (
                f"{declared_name}  ({category})"
                if category
                else declared_name
            )
            self.entity_declared_names[body.entity_id] = declared_name
            role_record = object_roles.get(declared_name)
            if role_record is not None:
                role, basis, provenance = role_record
                self.entity_roles[body.entity_id] = role
                self.entity_role_basis[body.entity_id] = basis
                self.entity_role_provenance[body.entity_id] = provenance
                main_roles[body.body_id] = role

        declared_region_names = {spec.name for spec in snapshot.regions}
        for region in getattr(self, "_regions", ()):
            if region.spec.name in declared_region_names:
                self.entity_declared_names[
                    region.entity_id
                ] = region.spec.name

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

    def _entity_extra_tags(self, entity_id: int) -> Tuple[str, ...]:
        tags = []
        role = self.entity_roles.get(entity_id)
        if role is not None:
            tags.append("role:" + role)
        declared_name = self.entity_declared_names.get(entity_id)
        if declared_name is not None:
            tags.append("declared_name:" + declared_name)
        return tuple(tags)

    def property_metadata(
        self,
        entity_id: int,
    ) -> Dict[str, PropertyMeta]:
        metadata = super().property_metadata(entity_id)
        extra_tags = self._entity_extra_tags(entity_id)
        if not extra_tags:
            return metadata
        return {
            key: meta._replace(
                tags=tuple(meta.tags) + tuple(
                    tag for tag in extra_tags if tag not in meta.tags
                )
            )
            for key, meta in metadata.items()
        }

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

    def _discover_extra_regions(self) -> Sequence[RegionSpec]:
        """Return every instantiated declaration-backed box region.

        Region identity and box semantics come from ``object_sites_dict``.
        Position, orientation, and half-extents come from the reconciled
        compiled site. Generic raw-site extraction uses a separate
        namespaced identity, so the same source site may also remain visible
        as class ``site`` without colliding with this semantic entity.
        """
        return self._declaration_snapshot().regions

    def _gripper_properties(
        self,
        entity_id: int,
        site_name: str,
        site_id: int,
    ) -> Dict[str, float]:
        """
        Panda finger-joint state, using the same property names as
        RoboCasaAdapter so gripper specs transfer between backends:

            gripper_qpos_<i>, gripper_qpos_mean,
            gripper_qpos_abs_mean, grip
        """
        properties = super()._gripper_properties(
            entity_id,
            site_name,
            site_id,
        )

        robots = getattr(self.env, "robots", None)
        if not robots:
            return properties

        gripper = getattr(robots[0], "gripper", None)
        if gripper is None:
            return properties

        values = self._gripper_joint_positions(gripper)

        if values.size == 0 or not np.all(np.isfinite(values)):
            return properties

        for index, value in enumerate(values):
            properties[f"gripper_qpos_{index}"] = float(value)

        signed_mean = float(np.mean(values))
        properties["gripper_qpos_mean"] = signed_mean
        properties["gripper_qpos_abs_mean"] = float(np.mean(np.abs(values)))
        properties["grip"] = signed_mean

        return properties

    def _gripper_joint_positions(self, gripper: Any) -> np.ndarray:
        """Read Panda finger coordinates from runtime state or joint rows."""
        qpos = getattr(gripper, "qpos", None)
        if qpos is not None:
            try:
                values = np.asarray(qpos, dtype=float).ravel()
            except Exception:
                values = np.empty(0, dtype=float)
            if values.size:
                return values

        names = tuple(getattr(gripper, "joints", ()) or ())
        available = set(self._names("joint"))
        if not names or any(name not in available for name in names):
            return np.empty(0, dtype=float)
        return np.asarray(
            [
                self._data.qpos[
                    int(
                        self._model.jnt_qposadr[
                            self._name2id("joint", name)
                        ]
                    )
                ]
                for name in names
            ],
            dtype=float,
        )

    def _gripper_property_meta(self) -> Dict[str, PropertyMeta]:
        """
        Display metadata for the Panda finger-joint keys added in
        ``_gripper_properties``, keeping the metadata/property parity
        invariant intact.
        """
        meta = super()._gripper_property_meta()

        robots = getattr(self.env, "robots", None)
        if not robots:
            return meta

        gripper = getattr(robots[0], "gripper", None)
        if gripper is None:
            return meta

        values = self._gripper_joint_positions(gripper)

        if values.size == 0 or not np.all(np.isfinite(values)):
            return meta

        for index in range(values.size):
            meta[f"gripper_qpos_{index}"] = PropertyMeta(
                group="gripper", note="m",
            )

        meta["gripper_qpos_mean"] = PropertyMeta(group="gripper", note="m")
        meta["gripper_qpos_abs_mean"] = PropertyMeta(group="gripper", note="m")
        meta["grip"] = PropertyMeta(group="gripper")

        return meta

    # ------------------------------------------------------------------
    # Reset hook: rebuild declarations for the new scene
    # ------------------------------------------------------------------

    def _reset_impl(self):
        # Task layouts and declaration owners may change between resets.
        # No contact snapshot here: reset rebuilds the simulator handles,
        # so the reset observation fresh-reads the new buffer instead.
        self._invalidate_declaration_snapshot()
        self._presented_contact_pairs = None
        return super()._reset_impl()

    # ------------------------------------------------------------------
    # Task metadata
    # ------------------------------------------------------------------

    @property
    def language_instruction(self) -> Optional[str]:
        """Natural-language task description for display and logging."""
        if self.task is not None:
            language = getattr(self.task, "language", None)
            if language:
                return str(language)
        language = getattr(self.env, "language_instruction", None)
        return str(language) if language else None

    def get_ep_meta(self) -> Dict[str, str]:
        """RoboCasa-compatible episode metadata for load_sim()."""
        return {"lang": self.language_instruction or ""}


# ----------------------------------------------------------------------
# Environment construction
# ----------------------------------------------------------------------

def make_libero_env(
    suite: str = "libero_object",
    task_id: int = 0,
    camera_heights: int = 128,
    camera_widths: int = 128,
    seed: int = 0,
    offscreen_renderer: bool = False,
):
    """
    Build a LIBERO environment for the given suite and task.

    The default ``ControlEnv`` disables camera observations and offscreen
    rendering. Set ``offscreen_renderer=True`` only when a caller needs the
    EGL-backed ``OffScreenRenderEnv``.

    Returns ``(env, task)`` ready to pass to LIBEROAdapter.
    """
    try:
        from libero.libero import benchmark, get_libero_path
    except Exception as exc:
        raise ImportError(
            f"LIBERO import failed: {type(exc).__name__}: {exc}\n"
            f"  (If LIBERO is installed, this is likely a missing "
            f"dependency or a MuJoCo/EGL issue in the active env.)"
        ) from exc

    try:
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs.env_wrapper import ControlEnv
    except ImportError as exc:
        raise ImportError(
            f"LIBERO installed but sub-module missing ({exc})"
        ) from exc

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite not in benchmark_dict:
        raise ValueError(
            f"Unknown LIBERO suite {suite!r}. "
            f"Available: {sorted(benchmark_dict)}"
        )

    task_suite = benchmark_dict[suite]()
    task = task_suite.get_task(task_id)

    # LIBERO's configured data root is authoritative. Deriving this path
    # from the import location can silently pair code from one clone with
    # task declarations from another.
    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    if not os.path.isfile(bddl_file):
        raise FileNotFoundError(
            f"LIBERO task declaration does not exist: {bddl_file}"
        )

    environment_class = (
        OffScreenRenderEnv if offscreen_renderer else ControlEnv
    )
    env = environment_class(
        bddl_file_name=bddl_file,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        has_renderer=False,
        has_offscreen_renderer=offscreen_renderer,
        use_camera_obs=False,
        ignore_done=True,
    )
    env.seed(seed)
    env.reset()
    return env, task


# ----------------------------------------------------------------------
# Registry integration
# ----------------------------------------------------------------------

def _make_libero(
    task: Optional[str],
    robot: Optional[str],
    target_only: bool,
    capture_frames: bool = True,
) -> LIBEROAdapter:
    """
    Registry factory. ``task`` is interpreted as ``suite:task_id``,
    for example ``libero_object:0``. ``robot`` is ignored: LIBERO tasks
    fix the robot model. ``capture_frames=False`` builds the env without
    an offscreen renderer, for a run that captures no camera frame.
    """
    suite = "libero_object"
    task_id = 0
    if task:
        parts = task.split(":", 1)
        suite = parts[0]
        if len(parts) == 2 and parts[1].isdigit():
            task_id = int(parts[1])

    if not capture_frames:
        env, libero_task = make_libero_env(suite=suite, task_id=task_id)
        return LIBEROAdapter(env, task=libero_task, target_only=target_only)

    # Offscreen renderer preferred: robosuite-family envs upload
    # reset-time texture data through the live render context only, so
    # capture through a fresh renderer samples stale textures (see
    # adding-an-adapter, "Camera capture obligations"). The context must
    # therefore exist from construction whenever a frame may be captured.
    # Monitoring must not require GL, so failure degrades to the
    # rendererless env.
    try:
        env, libero_task = make_libero_env(
            suite=suite, task_id=task_id, offscreen_renderer=True
        )
    except Exception as exc:
        import sys

        # The env handle is internal to make_libero_env, so a failure
        # mid-construction can leak one GL context (bounded; process
        # continues rendererless). A non-GL failure (bad suite/task)
        # will fail the retry too — this line names the real cause.
        print(
            "  Web warning: offscreen-renderer env construction failed "
            "({}: {}); retrying without it.".format(
                type(exc).__name__, exc
            ),
            file=sys.stderr,
        )
        env, libero_task = make_libero_env(suite=suite, task_id=task_id)
    return LIBEROAdapter(env, task=libero_task, target_only=target_only)


def _handles(obj) -> bool:
    """Cheap module-path sniff — never imports LIBERO itself."""
    return (type(obj).__module__ or "").startswith("libero.")


def _wrap(env, **kwargs) -> LIBEROAdapter:
    """Wrap an existing LIBERO env.

    Adapter kwargs pass through, e.g. ``task=`` (the LIBERO task object,
    for ``language_instruction``) and ``target_only=``.
    """
    return LIBEROAdapter(env, **kwargs)


from ..registry import register as _register

_register(
    name="libero",
    factory=_make_libero,
    description="LIBERO benchmark (Liu et al., NeurIPS 2023)",
    install_hint=(
        "conda env with python=3.8.13 and robosuite==1.4.1; "
        "pip install -e <LIBERO clone>"
    ),
    handles=_handles,
    wrap=_wrap,
    requires=("libero",),
)


# Bounded migration provider (maintainer decision 2026-09-07): the wrapper
# row inherits the engine provider and adds its own source custody.
from .mujoco_provider import install_bounded_provider as _install_bounded_provider  # noqa: E402

_install_bounded_provider(LIBEROAdapter)
