"""
adapters/mujoco.py — Generic MuJoCo SimulatorAdapter

MuJoCo-specific, task-independent extraction of monitor entities.
Simulator-specific naming, task regions, and extra physical signals are
provided through subclass hooks.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from ..adapter import MonitorObservation, PropertyMeta, SimulatorAdapter
from ..capabilities import AdapterCapabilities, VocabularyMode
from ..entities import EntityState, TrackState
from .mujoco_geom_pairs import GEOM_PAIR_ENTITY_CLASS
from .mujoco_contract import (
    FieldSurfaceReport,
    INCLUDED_FIELDS,
    classify_public_field_surface,
    public_noncallable_field_names,
)

# Pause after viewer.close() before returning control: exiting the
# process while the viewer thread is still shutting down dumps core on
# MuJoCo 3.10 (Wayland); two seconds outlasts the race.
_VIEWER_TEARDOWN_SETTLE = 2.0


BodyMatcher = Union[str, Callable[[str], bool]]
PerceptionGate = Callable[[EntityState, Any], bool]

# Dynamic contact-pair entities. MjData contact rows have unstable
# indices (one body pair can occupy dozens of distinct row indices over
# an episode), so rows are never exposed. The adapter materializes one
# dynamic entity per unordered body pair, keyed on
# model.geom_bodyid[contact.geom[i]], and aggregates the pair's retained
# contact points via mj_contactForce. A pair whose participants include
# a policy-declared robot body classifies as ``robot_contact``; every
# other pair classifies as ``contact``. Both classes are emitted
# whenever a pair exists; under a monitor session's extraction scope
# their properties are projected onto the keys the compiled specs
# require, like every other class (see ``set_extraction_scope``).
CONTACT_ENTITY_CLASS = "contact"
ROBOT_CONTACT_ENTITY_CLASS = "robot_contact"

# Shared empty result for a class no compiled spec references.
_NO_KEYS: FrozenSet[str] = frozenset()
# The same for entity selectors; one object, so an unreferenced class keeps
# one cached geom-pair scope (``_geom_pairs_in_scope``).
_NO_SELECTORS: FrozenSet[tuple] = frozenset()

# mj_contactForce returns a 6-vector in the contact frame whose rows are
# contact.frame; component 0 is the normal force ("normal is in [0-2],
# points from geom[0] to geom[1]", mjdata.h). The scalar aggregations
# below deliberately expose several source-grounded readings because
# external references (ISO/TS 15066 Table A.2, SafeVLA-Bench's per-step
# point maxima) do not agree on one:
#
# - contact_point_force_max: max over the pair's retained points of the
#   Euclidean norm of the translational force (components 0-2). This is
#   the per-point reading SafeVLA-Bench's formulas take maxima over.
# - contact_normal_force_max / contact_normal_force_sum: max and sum of
#   component 0 (the compressive normal force) over the pair's points.
# - contact_resultant_force: norm of the world-frame vector sum of the
#   pair's translational point forces, oriented as the force applied by
#   the pair's lower body ID to its higher body ID. (A same-body pair is
#   unreachable: MuJoCo's collision filter never emits contacts between
#   geoms of one body.)
# - contact_point_count: number of retained points for the pair.
# - contact_min_dist: minimum contact.dist over the pair's points
#   (negative is penetration). Emitted only while the pair has retained
#   points; the force keys and the count persist at exactly zero once a
#   pair has been seen, because a separated pair's contact force is
#   measured state, not fabrication.
CONTACT_PROPERTY_KEYS = frozenset(
    {
        "contact_point_force_max",
        "contact_normal_force_max",
        "contact_normal_force_sum",
        "contact_resultant_force",
        "contact_point_count",
        "contact_min_dist",
    }
)

_CONTACT_FORCE_KEYS = (
    "contact_point_force_max",
    "contact_normal_force_max",
    "contact_normal_force_sum",
    "contact_resultant_force",
)


def _contact_pair_key(model: Any, contact: Any) -> Tuple[int, int]:
    """Return the unordered body pair owning one MjData contact row.

    ``contact.geom`` is authoritative (``geom1``/``geom2`` are
    deprecated aliases). A flex participant carries geom index -1;
    indexing ``geom_bodyid`` with it would silently return the last
    geom's body, so it fails loudly until a flex-contact representation
    is designed. None of the supported simulators compile flex objects.
    """
    geoms = getattr(contact, "geom", None)
    if geoms is None:
        geoms = (contact.geom1, contact.geom2)

    first_geom = int(geoms[0])
    second_geom = int(geoms[1])
    if first_geom < 0 or second_geom < 0:
        raise RuntimeError(
            "MuJoCo flex contact encountered (geom index -1). Dynamic "
            "contact-pair entities represent geom-geom contacts only; "
            "a flex-contact representation has not been designed."
        )

    bodies = (
        int(model.geom_bodyid[first_geom]),
        int(model.geom_bodyid[second_geom]),
    )
    return (min(bodies), max(bodies))


@dataclass
class _ContactPairRecord:
    """One dynamic contact pair seen this episode."""

    entity_id: int
    body_pair: Tuple[int, int]
    entity_class: str
    label: str
    tags: Tuple[str, ...] = ()

_VALID_JOINT_SCOPES = frozenset(
    {
        "none",
        "body",
        "subtree",
    }
)

_VALID_REGION_SOURCES = frozenset(
    {
        "int_sites",
        "site",
        "geom",
        "fixed",
    }
)

_MJ_JNT_FREE = 0
_MJ_JNT_BALL = 1
_MJ_JNT_SLIDE = 2
_MJ_JNT_HINGE = 3

# ``mjData.xaxis`` has one meaningful axis only for MuJoCo's scalar joint
# kinds.  Keep the complete engine joint-kind partition beside the reader so
# every MuJoCo-family wrapper inherits the same applicability decision.
_XAXIS_JOINT_TYPE_APPLICABILITY = {
    _MJ_JNT_FREE: False,
    _MJ_JNT_BALL: False,
    _MJ_JNT_SLIDE: True,
    _MJ_JNT_HINGE: True,
}

# mjtBias values. MuJoCo has no actuator "kind" field: a motor is bias
# NONE, and position/velocity servos are AFFINE distinguished by which
# feedback term negates the gain (see _ctrl_unit_note).
_MJ_BIAS_NONE = 0
_MJ_BIAS_AFFINE = 1

# The bias pattern alone is not enough to identify a kind, because two
# other shorthands wear a servo's or a motor's bias while commanding
# something else entirely. Both are separable, but only on the other two
# type fields (measured against every MuJoCo actuator shorthand,
# 2026-07-27):
#
#   intvelocity  dyntype INTEGRATOR, bias AFFINE with the *position*
#                term negated -- indistinguishable from a position servo
#                by bias alone, yet its ctrl is a velocity.
#   damper       gaintype AFFINE, bias NONE -- indistinguishable from a
#                motor by bias alone, yet its ctrl is a dimensionless
#                multiplier on the authored damping coefficient.
#
# The full dynamic, gain and bias encoding is therefore required.
_MJ_DYN_NONE = 0
_MJ_DYN_INTEGRATOR = 1
_MJ_DYN_FILTER = 2
_MJ_DYN_FILTEREXACT = 3
_MJ_GAIN_FIXED = 0
_MJ_GAIN_AFFINE = 1

_QUAT_COMPONENTS = ("w", "x", "y", "z")

# mjtGeom value -> name, for the 3D scene export (MuJoCo 3.3.1,
# https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtgeom).
# Only _EXPORTED_GEOM_TYPES have a renderable primitive on the dashboard
# side; the rest are counted as skipped so the panel can say what it is
# not showing rather than silently dropping it.
_GEOM_TYPE_NAMES = {
    0: "plane",
    1: "hfield",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
    8: "sdf",
}
_EXPORTED_GEOM_TYPES = frozenset(
    ("plane", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh")
)

# MuJoCo's compiled-in default geom_rgba. A geom left at exactly this
# value while carrying a material takes the material's colour instead,
# which is what the native viewer draws.
_GEOM_RGBA_DEFAULT = (0.5, 0.5, 0.5, 1.0)

# Appearance arrays zeroed on a copy of the model before ``state_signature``
# digests it: colours, materials and textures, which no observation reads.
# RoboCasa draws its counter-top colours from the process-global NumPy RNG
# (robocasa/models/fixtures/counter.py:369 and :474, robocasa 1.0.1), so two
# processes with one ``--seed`` build kitchens that differ only in
# ``geom_rgba`` (measured 2026-10-14). Everything an observation can reach
# (sizes, masses, friction, contype and conaffinity, groups, names,
# options, timestep, meshes) stays in the digest. Arrays a MuJoCo build
# lacks are skipped.
_SIGNATURE_APPEARANCE_ARRAYS = (
    "geom_rgba", "geom_matid", "site_rgba", "tendon_rgba", "skin_rgba",
    "light_ambient", "light_diffuse", "light_specular",
    "mat_rgba", "mat_texid", "mat_texuniform", "mat_texrepeat", "mat_emission",
    "mat_specular", "mat_shininess", "mat_reflectance",
    "tex_type", "tex_height", "tex_width", "tex_adr", "tex_data",
)

# Body-pose field -> (MjData attribute aliases in preference order, row
# width). robosuite/mujoco-py wrappers alias the modern ``mujoco`` names
# and disagree about which one they keep, so the preference order is
# per field and matches what the adapter has always read. Widths are the
# documented mjData row shapes (nbody x 3 / 4 / 9):
# https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata
_BODY_POSE_SOURCES = {
    "xpos": (("body_xpos", "xpos"), 3),
    "xquat": (("xquat", "body_xquat"), 4),
    "xmat": (("xmat",), 9),
}

_SIGNAL_ADDRESSES = {
    "qpos": ("jnt_qposadr", "nq"),
    "qvel": ("jnt_dofadr", "nv"),
    "qacc": ("jnt_dofadr", "nv"),
    "qacc_smooth": ("jnt_dofadr", "nv"),
    "qfrc_actuator": ("jnt_dofadr", "nv"),
    "qfrc_constraint": ("jnt_dofadr", "nv"),
    "qfrc_bias": ("jnt_dofadr", "nv"),
    "qfrc_applied": ("jnt_dofadr", "nv"),
    "qfrc_smooth": ("jnt_dofadr", "nv"),
    # Passive half of the joint-force family. qfrc_passive is the total;
    # the other four are its components. All are computed by mj_step, so
    # a zero reading means the model configures none of that effect (no
    # fluid density, no gravity compensation), not that it was skipped.
    "qfrc_passive": ("jnt_dofadr", "nv"),
    "qfrc_spring": ("jnt_dofadr", "nv"),
    "qfrc_damper": ("jnt_dofadr", "nv"),
    "qfrc_fluid": ("jnt_dofadr", "nv"),
    "qfrc_gravcomp": ("jnt_dofadr", "nv"),
}

# qfrc_inverse is deliberately absent: it is only written by mj_inverse(),
# which mj_step never calls, so exposing it would produce a signal that
# reads zero forever under forward simulation. See issue #76.

# MuJoCo's per-body 6-vectors, all in the same com-based frame and all
# ordered [rotational (0, 1, 2), translational (3, 4, 5)]. Each entry is
# (field, rotational group, rotational unit, linear group, linear unit);
# the groups split by unit so that no group a spec can select mixes two
# physical quantities.
#
# cvel is filled by mj_step. cfrc_ext, cfrc_int and cacc are not -- they
# need the explicit mj_rnePostConstraint call made in
# _populate_constraint_dependent_fields.
_BODY_SPATIAL_FIELDS: Tuple[Tuple[str, str, str, str, str], ...] = (
    (
        "cvel",
        "body angular velocity", "rad/s",
        "body com-frame linear velocity", "m/s",
    ),
    (
        "cfrc_ext",
        "body external torque", "N·m",
        "body external force", "N",
    ),
    (
        "cfrc_int",
        "body internal torque", "N·m",
        "body internal force", "N",
    ),
    (
        "cacc",
        "body angular acceleration", "rad/s²",
        "body com-frame linear acceleration", "m/s²",
    ),
)

_SENSOR_UNIT_NOTES = {
    "touch": "N",
    "force": "N",
    "torque": "N·m",
    "gyro": "rad/s",
    "accelerometer": "m/s²",
    "velocimeter": "m/s",
}

_ORPHAN_SENSOR_ENTITY_CLASS = "sensors"

_ORPHAN_ACTUATOR_ENTITY_CLASS = "actuators"

# First-class entity classes introduced by the shared curated field
# materializers. Closed-vocabulary MuJoCo subclasses must include this exact
# set; keeping it beside the materializer mechanism prevents a subclass from
# accidentally rejecting globally available owners.
_CURATED_FIELD_OWNER_ENTITY_CLASSES = frozenset(
    {
        "simulation",
        "body",
        "site",
        "geom",
        "camera",
        "light",
        "equality",
        "flex",
        "tendon",
        "mocap",
        # Dynamic contact-pair sinks of the included raw `contact`
        # field. Global availability: a closed adapter vocabulary that
        # omitted them would reject the entities the shared reader
        # emits.
        CONTACT_ENTITY_CLASS,
        ROBOT_CONTACT_ENTITY_CLASS,
        # Geometry distance pairs of the bound query universe, present
        # from reset under the bounded provider.
        GEOM_PAIR_ENTITY_CLASS,
    }
)

# Metadata group per actuator key, keyed by MjData field then unit note.
# Every group holds one unit, so `@ <group>` aggregates something
# comparable -- the rule every other family here follows, and which the
# single "actuators" group these replace could not: it held a control
# range beside N·m, m and m/s at once.
#
# The `actuator ...` provenance does not collide with the per-DOF
# `actuator forces`/`actuator torques` groups, which are the generalized
# force contributed at each DOF rather than one actuator's own output.
#
# `None` is the unit for a transmission whose units cannot be read off a
# scalar joint (free/ball joints, and non-joint transmissions); those
# fall back to a group that makes no unit claim.
_ACTUATOR_GROUPS: Dict[str, Dict[Optional[str], str]] = {
    "actuator_force": {
        "N": "actuator output force",
        "N·m": "actuator output torque",
        None: "actuator output",
    },
    "actuator_length": {
        "m": "actuator displacement",
        "rad": "actuator angle",
        None: "actuator length",
    },
    "actuator_velocity": {
        "m/s": "actuator linear velocity",
        "rad/s": "actuator angular velocity",
        None: "actuator velocity",
    },
}

# ctrl's *note* is the actuator's control range rather than a unit,
# because the range is the authoring answer to "what threshold do I write
# here". Its *group* is still unit-homogeneous, keyed by the units the
# command is expressed in.
#
# Those two being different is the whole point. An earlier version argued
# that because ctrl's note is a label rather than a unit, ctrl could take
# one group whatever the range — reasoning by analogy with the quaternion
# labels on "body orientation". The analogy fails: quaternion components
# are dimensionless, so aggregating them is at worst meaningless, whereas
# a torque command in N·m and a position command in m aggregated under one
# group is the unit-mixing hazard that splitting the old "actuators" group
# was meant to remove (issue #84, the same defect class as #70).
#
# What a command is expressed in follows the actuator's *kind*, not its
# transmitted joint alone: a motor on a hinge commands N·m, a position
# servo on the same hinge commands rad. See `_ctrl_unit_note`.
_CTRL_GROUPS: Dict[str, str] = {
    "N": "actuator force command",
    "N·m": "actuator torque command",
    "m": "actuator position command",
    "rad": "actuator angle command",
    "m/s": "actuator linear velocity command",
    "rad/s": "actuator angular velocity command",
    "1": "actuator damping-scale command",
}


@dataclass(frozen=True)
class BodyRule:
    """
    Classify matching MuJoCo bodies.

    Parameters
    ----------
    match:
        A body-name substring or a callable ``body_name -> bool``.

    cls:
        String entity class assigned to matching bodies.

    priority:
        Registration priority. Lower values receive lower entity IDs.
        Equal priorities preserve MuJoCo model order.

    joint_scope:
        Controls which joints are exposed through the entity's dynamic
        properties:

        ``"none"``
            Do not expose joint state.

        ``"body"``
            Expose joints attached directly to the matched body.

        ``"subtree"``
            Expose joints attached to the matched body and all descendants.
            This is appropriate when a complete articulated robot is
            represented by its base body.
    """

    match: BodyMatcher
    cls: str
    priority: int = 100
    joint_scope: str = "body"

    def __post_init__(self) -> None:
        if not isinstance(self.match, str) and not callable(self.match):
            raise TypeError(
                "BodyRule.match must be a string or callable"
            )

        if isinstance(self.match, str) and not self.match:
            raise ValueError(
                "BodyRule.match cannot be empty"
            )

        if not isinstance(self.cls, str) or not self.cls.strip():
            raise ValueError(
                "BodyRule.cls must be a non-empty string"
            )

        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, Integral)
        ):
            raise TypeError(
                "BodyRule.priority must be an integer"
            )

        if self.joint_scope not in _VALID_JOINT_SCOPES:
            raise ValueError(
                f"joint_scope must be one of "
                f"{sorted(_VALID_JOINT_SCOPES)}, "
                f"got {self.joint_scope!r}"
            )

    def matches(self, body_name: str) -> bool:
        if callable(self.match):
            return bool(self.match(body_name))

        return self.match in body_name


@dataclass(frozen=True)
class RegionSpec:
    """
    Define one world-axis-aligned monitor region.

    Supported sources
    -----------------
    ``"int_sites"``:
        Uses four MuJoCo sites:

        - ``<source_name>_int_p0``
        - ``<source_name>_int_px``
        - ``<source_name>_int_py``
        - ``<source_name>_int_pz``

        These sites define a parallelepiped. The adapter computes its
        world-axis-aligned bounding box.

    ``"site"``:
        Uses the named site as the region center. ``half_extents`` are
        interpreted in the site's local coordinate frame.

    ``"geom"``:
        Uses a named box-like geom. ``half_extents`` may override
        ``model.geom_size``.

    ``"fixed"``:
        Uses an explicit world-frame ``center`` and ``half_extents``.

    The generated entity stores ``hw``, ``hh``, and ``hd`` in its dynamic
    properties dictionary. Additional scalar properties may be supplied
    through ``properties``.
    """

    name: str
    source: str
    source_name: Optional[str] = None
    half_extents: Optional[Tuple[float, float, float]] = None
    center: Optional[Tuple[float, float, float]] = None
    cls: str = "region"
    properties: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "RegionSpec.name must be a non-empty string"
            )

        if not isinstance(self.cls, str) or not self.cls.strip():
            raise ValueError(
                "RegionSpec.cls must be a non-empty string"
            )

        if self.source not in _VALID_REGION_SOURCES:
            raise ValueError(
                f"RegionSpec.source must be one of "
                f"{sorted(_VALID_REGION_SOURCES)}, "
                f"got {self.source!r}"
            )

        if self.source == "fixed":
            _vector3(
                self.center,
                f"RegionSpec {self.name!r} center",
            )
            _nonnegative_vector3(
                self.half_extents,
                f"RegionSpec {self.name!r} half_extents",
            )
        else:
            if not self.source_name:
                raise ValueError(
                    f"RegionSpec {self.name!r} source "
                    f"{self.source!r} requires source_name"
                )

            if self.source == "site":
                _nonnegative_vector3(
                    self.half_extents,
                    f"RegionSpec {self.name!r} half_extents",
                )

        if self.half_extents is not None:
            _nonnegative_vector3(
                self.half_extents,
                f"RegionSpec {self.name!r} half_extents",
            )

        if not isinstance(self.properties, Mapping):
            raise TypeError(
                "RegionSpec.properties must be a mapping"
            )


@dataclass(frozen=True)
class DeclaredEntitySpec:
    """Reset-time identity and provenance for a task-declared entity.

    Live state is deliberately absent.  Task 7's generic declaration hook
    materialises these identities after reset; this value only gives the
    simulator-semantic policy a typed declaration slot.
    """

    declaration_key: str
    name: str
    entity_class: str
    source_kind: str
    source_name: str
    source_provenance: str
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "declaration_key",
            "name",
            "entity_class",
            "source_kind",
            "source_name",
            "source_provenance",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "DeclaredEntitySpec.{} must be a non-empty string".format(
                        field_name
                    )
                )
        canonical_tags = tuple(self.tags)
        if (
            any(
                not isinstance(tag, str) or not tag.strip()
                for tag in canonical_tags
            )
            or len(set(canonical_tags)) != len(canonical_tags)
        ):
            raise ValueError(
                "DeclaredEntitySpec.tags must be unique non-empty strings"
            )
        object.__setattr__(self, "tags", canonical_tags)


@dataclass(frozen=True)
class MujocoJointOwner:
    """One additional semantic owner for named MuJoCo joints.

    Body rules already establish the ordinary body-to-joint ownership map.
    Simulator declarations may additionally make the same physical joint
    author-visible on another registered entity, such as an articulated
    region.  This record carries only identity and ownership policy; the
    shared materializers remain the sole readers of ``MjData``.
    """

    entity_id: int
    owner_body_id: int
    joint_ids: Tuple[int, ...]
    owner_name: str

    def __post_init__(self) -> None:
        for field_name in ("entity_id", "owner_body_id"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(
                    "MujocoJointOwner.{} must be an integer".format(
                        field_name
                    )
                )
            if value < 0:
                raise ValueError(
                    "MujocoJointOwner.{} must be non-negative".format(
                        field_name
                    )
                )

        joint_ids = tuple(self.joint_ids)
        if not joint_ids:
            raise ValueError(
                "MujocoJointOwner.joint_ids must not be empty"
            )
        if any(
            isinstance(joint_id, bool)
            or not isinstance(joint_id, Integral)
            or joint_id < 0
            for joint_id in joint_ids
        ):
            raise TypeError(
                "MujocoJointOwner.joint_ids must contain non-negative "
                "integers"
            )
        if len(set(joint_ids)) != len(joint_ids):
            raise ValueError(
                "MujocoJointOwner.joint_ids must be unique"
            )
        if not isinstance(self.owner_name, str) or not self.owner_name.strip():
            raise ValueError(
                "MujocoJointOwner.owner_name must be a non-empty string"
            )
        object.__setattr__(self, "joint_ids", joint_ids)


@dataclass(frozen=True)
class MujocoBodyOwner:
    """One semantic entity owning exact compiled-model body rows."""

    entity_id: int
    body_ids: Tuple[int, ...]
    owner_name: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.entity_id, bool)
            or not isinstance(self.entity_id, Integral)
            or self.entity_id < 0
        ):
            raise TypeError(
                "MujocoBodyOwner.entity_id must be a nonnegative integer"
            )
        body_ids = tuple(self.body_ids)
        if (
            not body_ids
            or any(
                isinstance(body_id, bool)
                or not isinstance(body_id, Integral)
                or body_id < 0
                for body_id in body_ids
            )
        ):
            raise TypeError(
                "MujocoBodyOwner.body_ids must contain nonnegative "
                "integers"
            )
        if len(set(body_ids)) != len(body_ids):
            raise ValueError("MujocoBodyOwner.body_ids must be unique")
        if not isinstance(self.owner_name, str) or not self.owner_name.strip():
            raise ValueError(
                "MujocoBodyOwner.owner_name must be a non-empty string"
            )
        object.__setattr__(self, "body_ids", body_ids)


@dataclass(frozen=True)
class MujocoSemanticPolicy:
    """Simulator semantics, excluding field availability and materialisation.

    ``robot_bodies`` names the bodies whose contacts classify as
    ``robot_contact`` rather than ``contact`` — a semantic ownership
    mapping, not an availability switch: both contact classes exist on
    every MuJoCo adapter, and an empty robot set only means every pair
    lands in the generic class. Matchers follow ``BodyRule.match``
    semantics (substring or callable over the body name).
    """

    body_rules: Tuple[BodyRule, ...] = ()
    regions: Tuple[RegionSpec, ...] = ()
    declared_entities: Tuple[DeclaredEntitySpec, ...] = ()
    robot_bodies: Tuple[BodyMatcher, ...] = ()

    def __post_init__(self) -> None:
        body_rules = tuple(self.body_rules)
        regions = tuple(self.regions)
        declared_entities = tuple(self.declared_entities)
        robot_bodies = tuple(self.robot_bodies)
        object.__setattr__(self, "body_rules", body_rules)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "declared_entities", declared_entities)
        object.__setattr__(self, "robot_bodies", robot_bodies)

        for matcher in robot_bodies:
            if not isinstance(matcher, str) and not callable(matcher):
                raise TypeError(
                    "MujocoSemanticPolicy.robot_bodies must contain "
                    "strings or callables"
                )
            if isinstance(matcher, str) and not matcher:
                raise ValueError(
                    "MujocoSemanticPolicy.robot_bodies cannot contain "
                    "empty strings"
                )

        if not all(isinstance(rule, BodyRule) for rule in body_rules):
            raise TypeError(
                "MujocoSemanticPolicy.body_rules must contain BodyRule objects"
            )
        if not all(isinstance(spec, RegionSpec) for spec in regions):
            raise TypeError(
                "MujocoSemanticPolicy.regions must contain RegionSpec objects"
            )
        if not all(
            isinstance(spec, DeclaredEntitySpec)
            for spec in declared_entities
        ):
            raise TypeError(
                "MujocoSemanticPolicy.declared_entities must contain "
                "DeclaredEntitySpec objects"
            )

        string_matches = [
            rule.match for rule in body_rules if isinstance(rule.match, str)
        ]
        duplicate_matches = _duplicates(string_matches)
        if duplicate_matches:
            raise ValueError(
                "duplicate body-rule match: {}".format(
                    ", ".join(repr(value) for value in duplicate_matches)
                )
            )

        duplicate_regions = _duplicates(spec.name for spec in regions)
        if duplicate_regions:
            raise ValueError(
                "duplicate region name: {}".format(
                    ", ".join(repr(value) for value in duplicate_regions)
                )
            )

        duplicate_keys = _duplicates(
            spec.declaration_key for spec in declared_entities
        )
        if duplicate_keys:
            raise ValueError(
                "duplicate declaration key: {}".format(
                    ", ".join(repr(value) for value in duplicate_keys)
                )
            )
        duplicate_names = _duplicates(spec.name for spec in declared_entities)
        if duplicate_names:
            raise ValueError(
                "duplicate declared entity name: {}".format(
                    ", ".join(repr(value) for value in duplicate_names)
                )
            )


def _duplicates(values: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


@dataclass(frozen=True)
class _BodyRecord:
    """Internal registration record for one monitored MuJoCo body."""

    entity_id: int
    body_name: str
    body_id: int
    entity_class: str
    joint_ids: Tuple[int, ...]
    pose_body_ids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class _SiteRecord:
    """Internal registration record for one monitored MuJoCo site."""

    entity_id: int
    site_name: str
    site_id: int
    entity_class: str


@dataclass(frozen=True)
class _SensorRecord:
    """Internal registration record for one monitored MuJoCo sensor."""

    sensor_id: int
    name: str
    name_token: str
    adr: int
    dim: int
    type_name: str
    unit_note: Optional[str]
    owner_entity_id: Optional[int]


@dataclass(frozen=True)
class _ActuatorRecord:
    """Internal registration record for one monitored MuJoCo actuator."""

    actuator_id: int
    name: str
    name_token: str
    ctrllimited: bool
    ctrlrange: Tuple[float, float]
    force_unit_note: Optional[str]
    owner_entity_id: Optional[int]
    length_unit_note: Optional[str] = None
    velocity_unit_note: Optional[str] = None
    ctrl_unit_note: Optional[str] = None


@dataclass(frozen=True)
class MujocoFieldMaterialization:
    """Runtime disposition of one globally included ``MjData`` field.

    ``binding_present`` describes the live Python binding, while
    ``scene_applicable`` describes whether the reset model contains at least
    one resolvable owner.  ``emitted_keys`` names the exact observation sinks
    written for the current snapshot; entity properties use
    ``entity[<id>].properties.<key>`` and observation-level values use
    ``observation.<key>``.
    """

    field_name: str
    binding_present: bool
    scene_applicable: bool
    emitted_keys: Tuple[str, ...]
    reason: Optional[str] = None
    owner_omissions: Tuple[Tuple[str, str], ...] = ()
    # Why this field reads its values one at a time instead of gathering
    # them (issue #96). Kept apart from ``reason``, which the field
    # contract renders as the field's capability: refusing to vectorize
    # says nothing about what the field can observe.
    plan_refusal: Optional[str] = None


@dataclass(frozen=True)
class _NamedOwnerRecord:
    """One first-class owner for a curated raw field family."""

    entity_id: int
    source_name: str
    source_id: int
    entity_class: str


@dataclass(frozen=True)
class _MocapOwnerRecord:
    """A named body mapped to one ``MjData.mocap_*`` row."""

    entity_id: int
    body_name: str
    body_id: int
    mocap_id: int


@dataclass(frozen=True)
class _DeclaredEntityRecord:
    """Reset registration for one task-declared entity."""

    entity_id: int
    spec: DeclaredEntitySpec


@dataclass(frozen=True)
class _FieldGatherPlan:
    """Where one field's steady-state values live in a flat MjData array.

    Built on a field's first full materialization pass and valid for the
    rest of the episode: ``flat_indices[i]`` is the offset, into
    ``getattr(data, attr).reshape(-1)``, of the value written to
    ``targets[i]``. ``position_rows`` carries the ``EntityState.position``
    side effect that the scalar path performs alongside the property
    writes: one ``(entity id, row)`` pair per owner whose position the
    field sets, always a row of ``attr`` itself (``xpos``, ``geom_xpos``,
    ``site_xpos``, ``cam_xpos``, ``mocap_pos``). The plan is cleared on
    reset, and installed only after its gathered values are verified
    against the ones the scalar pass just produced.
    """

    attr: str
    flat_indices: np.ndarray
    targets: List[Tuple[int, str]]
    position_rows: List[Tuple[int, int]]


def _plan_row_width(
    values: Any,
) -> Optional[int]:
    """Row stride for flat-indexing one MjData array, or ``None``.

    A gather plan reads its values out of
    ``np.asarray(array, dtype=float).reshape(-1)``, whose C order puts
    ``values[row][component]`` of a two-dimensional array at
    ``row * width + component`` and ``values[i]`` of a one-dimensional
    array at ``i``, i.e. at stride 1. Any other rank has no such
    arithmetic; returning ``None`` leaves the emission unplanned, which
    costs the field its gather and nothing else.

    The rank is read off the live array rather than a documented width
    because the plan re-reads that same array every step: a binding whose
    layout differs from the documentation must not be indexed as though
    it did not.
    """
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) not in (1, 2):
        return None
    return 1 if len(shape) == 1 else int(shape[1])


def _raw_struct(
    obj: Any,
    wrapper_attr: str,
) -> Any:
    """
    Unwrap a wrapped ``MjModel``/``MjData`` to the raw struct the C API needs.

    ``mujoco``'s module-level functions reject wrapper objects, and
    robosuite (so RoboCasa and LIBERO) hands the adapter a
    ``robosuite.utils.binding_utils.MjModel`` whose raw struct is at
    ``_model`` / ``_data``. Every other bundled backend passes raw
    structs to ``_resolve_model_data`` already: Meta-World and MA-MuJoCo
    natively, dm_control because ``adapters/dmcontrol._PhysicsProxy``
    resolves ``physics.model.ptr`` / ``physics.data.ptr`` before the
    adapter sees them. Those fall through unchanged.

    Checked by attribute rather than by isinstance so that specola never
    imports a simulator package to talk to MuJoCo.
    """
    return getattr(
        obj,
        wrapper_attr,
        obj,
    )


# What mj_fwdVelocity writes besides the velocity kinematics, on MuJoCo 3.3.0
# to 3.10.0: mj_passive's forces, mj_referenceConstraint's efc_vel and
# efc_aref, and the bias force of mj_rne and mj_tendonBias.
# MujocoAdapter._refresh_live_kinematics puts them back.
_FWD_VELOCITY_STEP_OUTPUTS = (
    "qfrc_spring", "qfrc_damper", "qfrc_gravcomp", "qfrc_fluid", "qfrc_passive",
    "qfrc_bias", "efc_vel", "efc_aref",
)


@dataclass(frozen=True)
class _RegionRecord:
    """Internal registration record for one region."""

    entity_id: int
    spec: RegionSpec


class MujocoAdapter(SimulatorAdapter):
    """
    Generic adapter for MuJoCo or RoboSuite-style environments.

    Parameters
    ----------
    env:
        Environment exposing either:

        - ``env.sim.model`` and ``env.sim.data``; or
        - ``env.model`` and ``env.data``.

    body_rules:
        Ordered body classification rules. The first matching rule wins.

    regions:
        Explicit region definitions. When omitted, the subclass hook
        ``_discover_extra_regions()`` is used.

    gripper_site:
        Explicit MuJoCo site to expose as a separate entity. When omitted,
        names returned by ``_default_gripper_sites()`` are searched.

    gripper_class:
        Entity class assigned to the gripper site.

    perception_gate:
        Optional function ``(entity_state, env) -> bool``. Returning False
        removes the body or gripper from the current observation.

    Field availability is global to the MuJoCo mechanism. Every curated
    joint family present on the live binding is eager; callers and simulator
    policy cannot select a subset.
    """

    def __init__(
        self,
        env: Any,
        body_rules: Optional[Sequence[BodyRule]] = None,
        regions: Optional[Sequence[RegionSpec]] = None,
        gripper_site: Optional[str] = None,
        gripper_class: str = "gripper",
        perception_gate: Optional[PerceptionGate] = None,
        robot_bodies: Optional[Sequence[BodyMatcher]] = None,
        **legacy_options: Any,
    ) -> None:
        self._reject_legacy_options(legacy_options)
        super().__init__()

        self.env = env
        self.gripper_site = gripper_site
        self.gripper_class = gripper_class
        self.perception_gate = perception_gate
        self._explicit_robot_bodies = (
            None if robot_bodies is None else tuple(robot_bodies)
        )

        if (
            not isinstance(gripper_class, str)
            or not gripper_class.strip()
        ):
            raise ValueError(
                "gripper_class must be a non-empty string"
            )

        if (
            perception_gate is not None
            and not callable(perception_gate)
        ):
            raise TypeError(
                "perception_gate must be callable or None"
            )

        explicit_body_rules = (
            None if body_rules is None else tuple(body_rules)
        )
        explicit_regions = (
            None if regions is None else tuple(regions)
        )
        if (
            explicit_body_rules is not None
            and not all(
                isinstance(rule, BodyRule)
                for rule in explicit_body_rules
            )
        ):
            raise TypeError(
                "body_rules must contain BodyRule objects"
            )

        if (
            explicit_regions is not None
            and not all(
                isinstance(spec, RegionSpec)
                for spec in explicit_regions
            )
        ):
            raise TypeError(
                "regions must contain RegionSpec objects"
            )
        self._explicit_body_rules = explicit_body_rules
        self._explicit_regions = explicit_regions
        self.body_rules: List[BodyRule] = []
        self.regions: List[RegionSpec] = []
        self._semantic_policy_value = MujocoSemanticPolicy()

        self._model: Any = None
        self._data: Any = None
        self._joint_signals: Tuple[str, ...] = ()
        self._field_surface_report = FieldSurfaceReport((), (), (), ())
        self._source_token_cache: Dict[str, Tuple[str, ...]] = {}
        self._joint_token_cache: Optional[Tuple[str, ...]] = None
        # Reset-invariant lookups memoized per episode (issue #92): name
        # resolution and row-ownership depend only on the compiled model
        # and the reset-time entity registration, but were recomputed
        # per property per owner per step.
        self._id2name_cache: Dict[Tuple[str, int], Optional[str]] = {}
        # Entities whose property/metadata parity already passed this
        # episode: the property-key schema is fixed once an episode
        # begins, so re-validating every step is pure hot-path cost
        # (measured 2026-07-31: ~0.6s/step on RoboCasa's ~2700
        # entities, plus the property_metadata recomputation it drags
        # in). Cleared on reset; a mid-episode entity is validated on
        # first appearance.
        self._parity_validated: set = set()
        # Reset-invariant hot-path memos (issue #92/#96 class), all
        # cleared on reset: geom/site type enum names are model-static;
        # a property's tag-merged metadata is fixed per (entity, key)
        # once an episode begins — memoizing it also restores object
        # identity across steps, which fast-paths the metadata
        # collision guard in _emit_property.
        self._geom_type_cache: Dict[Tuple[bool, int], str] = {}
        self._merged_meta_cache: Dict[Tuple[int, str], PropertyMeta] = {}
        self._identity_tags_cache: Dict[tuple, Tuple[str, ...]] = {}
        # Per-field steady state (issue #96): a field enters
        # _steady_fields after a full bookkeeping pass; its later runs
        # set _emit_value_only so _emit_property skips metadata,
        # provenance, and report work for known sinks. See
        # _materialize_curated_fields.
        self._materialization_steady = False
        self._steady_fields: set = set()
        self._emit_value_only = False
        # Sink tables for the scalar fields that cannot be gathered
        # (issue #96): the (entity_id, key) pairs one full pass emitted,
        # in emission order, so a value-only rerun reads its keys instead
        # of formatting them and skips PropertyMeta entirely. Recorded
        # only by a full pass that omitted no owner, which is what makes
        # every pair a guaranteed _merged_meta_cache hit later — see
        # _emit_property's metadata guard.
        self._value_only_sinks: Dict[str, List[Tuple[int, str]]] = {}
        # Per-field gather plans (issue #96): a planned field skips its
        # materializer entirely in steady state and runs one fancy-indexed
        # NumPy gather instead. _plan_capture is the recording sideband,
        # non-None only while a candidate field runs its full pass; the
        # per-emission source tuples it collects become the plan.
        self._gather_plan: Dict[str, _FieldGatherPlan] = {}
        self._plan_capture: Optional[
            List[Tuple[int, str, Tuple[str, int]]]
        ] = None
        self._plan_position_rows: List[Tuple[int, int]] = []
        # Spec-scoped extraction (issue #96, Task 5). _scope_dropped
        # counts the keys the active scope removed from the field
        # currently running its full pass; _scope_skipped_fields names
        # the fields whose every key it removed, which are then never
        # entered again this episode — excluding fields that materialize
        # their own entity population, whose owners are not all known
        # after one pass.
        self._scope_dropped: int = 0
        self._scope_skipped_fields: set = set()
        self._body_owner_pairs_cache: Optional[
            List[Tuple[int, int]]
        ] = None
        self._joint_owner_triples_cache: Optional[
            List[Tuple[int, int, int]]
        ] = None

        self._refresh_backend(
            allow_model_change=True
        )
        self._rebuild_reset_contract()

        self.name_to_id: Dict[str, int] = {}
        self.entity_labels: Dict[int, str] = {}

        self._bodies: List[_BodyRecord] = []
        self._gripper: Optional[_SiteRecord] = None
        self._regions: List[_RegionRecord] = []
        self._sensors: List[_SensorRecord] = []
        self._orphan_sensor_entity_id: Optional[int] = None
        self._actuators: List[_ActuatorRecord] = []
        self._orphan_actuator_entity_id: Optional[int] = None
        self._simulation_entity_id: Optional[int] = None
        self._raw_bodies: List[_NamedOwnerRecord] = []
        self._raw_sites: List[_NamedOwnerRecord] = []
        self._raw_geoms: List[_NamedOwnerRecord] = []
        self._raw_cameras: List[_NamedOwnerRecord] = []
        self._raw_lights: List[_NamedOwnerRecord] = []
        self._equalities: List[_NamedOwnerRecord] = []
        self._flexes: List[_NamedOwnerRecord] = []
        self._tendons: List[_NamedOwnerRecord] = []
        self._mocap_owners: List[_MocapOwnerRecord] = []
        self._declared_entities: List[_DeclaredEntityRecord] = []
        self._materialized_property_meta: Dict[
            int, Dict[str, PropertyMeta]
        ] = {}
        self._prospective_metadata_keys: Optional[List[Tuple[int, str]]] = None
        self._field_materialization: Dict[
            str, MujocoFieldMaterialization
        ] = {}
        self._current_owner_omissions: List[Tuple[str, str]] = []
        self._previous_positions: Dict[
            int,
            np.ndarray,
        ] = {}

        self._contact_pairs: Dict[
            Tuple[int, int], _ContactPairRecord
        ] = {}
        self._next_runtime_entity_id: int = 0
        self._robot_body_ids: FrozenSet[int] = frozenset()

        self._timestep = 0

        self._native_resources = ExitStack()
        self._native_subject_model = None
        self._native_projection = None
        # Geometry distance subject: bound with the native projection, read
        # on the same scratch snapshot, declared by the bounded provider.
        self._geometry_reader = None
        self._geom_pair_records = None
        # (registry, selectors, scoped records, reader pair arrays): the
        # specification-scoped subset, reused while both inputs are unchanged.
        self._geom_pair_scope = None
        self._configuration_identity = None
        self._geometry_samples: Tuple[Any, ...] = ()
        self._native_snapshot = None
        self._native_stored_data = None
        self._native_stored_wrapper = None
        self._native_closed = False

    def reset(self) -> MonitorObservation:
        """Begin an adapter episode inside the native observation effect scope.

        The guard the reset binds is fully checked at its construction, then
        once when the reset's reads begin and once after the reset observation
        is recorded; callers in between reuse that receipt (maintainer
        decision 2026-10-12, issue #136). A failed end check withdraws the
        observation, so the episode requires another reset. Retiring the
        previous subject keeps the checks on its own guard.
        """
        with self._native_observation_scope(), ExitStack() as boundaries:
            self._reset_boundaries = boundaries
            try:
                return super().reset()
            finally:
                self._reset_boundaries = None

    def get_observation(self) -> MonitorObservation:
        """Preserve native effects through guards, projection and validation.

        The live kinematics are refreshed first (``_refresh_live_kinematics``),
        outside the observation window, because the refresh writes to the
        live data. The bound native guard is then fully checked once before
        this observation and once after it is recorded (issue #136), as at
        reset.
        """
        with self._native_observation_scope():
            # A subclass may skip this initializer and bind no native subject.
            projection = getattr(self, "_native_projection", None)
            if projection is None:
                return super().get_observation()
            self._refresh_live_kinematics(projection)
            with projection._guard.observation():
                return super().get_observation()

    def _refresh_live_kinematics(self, projection) -> None:
        """Recompute the live kinematics at the live positions and velocities.

        ``mj_step`` computes poses, velocities derived from them, the mass
        matrix and the transmission from the positions it then integrates,
        so after a step they describe the instant before ``qpos``
        (robosuite, hence LIBERO and RoboCasa, and gymnasium's MA-MuJoCo end
        their step on ``mj_step``). This runs the position and velocity
        stages of the forward pass without collision detection or the
        constraint solver (maintainer decision 2026-10-14, option (c2) of
        ``docs/superpowers/plans/2026-10-13-state-replay.md``, kinematics
        only): ``mj_kinematics``, ``mj_comPos``, ``mj_camlight``, ``mj_flex``,
        ``mj_tendon``, the build's mass matrix stages, ``mj_transmission``
        and ``mj_fwdVelocity``. ``mj_fwdVelocity`` is the only call that
        computes ``actuator_velocity`` and ``ten_velocity``; the passive
        forces, constraint references and bias force it also writes are put
        back, so contacts, forces, ``qacc`` and sensor data stay what the
        step produced. The integration state is not written, so the
        trajectory is unchanged. On an environment whose step ends with a
        forward pass (Meta-World, dm_control) the refresh recomputes the
        same values.
        """
        import mujoco

        model, data = self._resolve_model_data(self.env)
        model = _raw_struct(model, "_model")
        if model is not self._native_subject_model:
            return  # the observation refuses the changed model
        data = _raw_struct(data, "_data")
        for stage in (mujoco.mj_kinematics, mujoco.mj_comPos, mujoco.mj_camlight,
                      mujoco.mj_flex, mujoco.mj_tendon):
            stage(model, data)
        projection.prepare_mass(data)
        mujoco.mj_transmission(model, data)
        kept = [np.copy(getattr(data, name)) for name in _FWD_VELOCITY_STEP_OUTPUTS]
        mujoco.mj_fwdVelocity(model, data)
        for name, value in zip(_FWD_VELOCITY_STEP_OUTPUTS, kept):
            getattr(data, name)[...] = value

    @contextmanager
    def _native_observation_scope(self):
        """Keep the last accepted observation through a rejected native read.

        A read rejected before the base records an observation (metadata
        parity, validation) has already been rolled back by the adapter's
        own bookkeeping, so the previously accepted observation stays and
        the episode continues (the accepted write-free and scope contract).
        A native exit guard can reject after the base has recorded an
        observation; by then the timestep and dynamic registries have
        advanced, so the recorded observation is withdrawn (lifecycle
        contract: no last observation) and the episode requires a reset.
        """
        previous = self._last_observation
        try:
            yield
        except BaseException:
            if self._last_observation is not previous:
                # Recorded, then rejected: bookkeeping has advanced.
                self._last_observation = None
                self._episode_started = False
            raise

    def _bind_extraction_subject(self, *, reset: bool) -> None:
        from importlib import import_module
        from .mujoco_native import MujocoNativeGuard
        from .mujoco_projection import MujocoProjection
        from .mujoco_scratch import MujocoScratchSession

        self._refresh_backend(allow_model_change=reset)
        model = _raw_struct(self._model, "_model")
        data = _raw_struct(self._data, "_data")
        if reset or self._native_projection is None:
            from .mujoco_geom_pairs import bind_geometry_reader, configuration_identity, geom_pair_records

            self._native_resources.close()
            self._native_projection = None
            self._native_subject_model = None
            self._geometry_reader = None
            self._geom_pair_records = None
            self._configuration_identity = None
            held = ExitStack()
            try:
                mujoco = import_module("mujoco")
                # The geometry reader loads these declaration modules onto
                # the producer package; finish that before the guard enrolls
                # the package namespace.
                import_module("mujoco.introspect.functions")
                import_module("mujoco.introspect.structs")
                guard = held.enter_context(MujocoNativeGuard(mujoco))
                boundaries = getattr(self, "_reset_boundaries", None)
                if reset and boundaries is not None:
                    # Closed by reset() after the reset observation is recorded.
                    boundaries.enter_context(guard.observation())
                session = held.enter_context(MujocoScratchSession(model, data, guard))
                projection = MujocoProjection(model, mujoco, session, guard)
                held.callback(projection.close)
                reader = bind_geometry_reader(model, data, self.query_configuration,
                                              guard=guard, session=session)
                held.callback(reader.close)
                records = geom_pair_records(reader, model)
                identity = configuration_identity(mujoco, model, self.query_configuration)
            except BaseException:
                held.close()
                raise
            self._native_resources = held
            self._native_subject_model = model
            self._native_projection = projection
            self._geometry_reader = reader
            self._geom_pair_records = records
            self._configuration_identity = identity
        elif model is not self._native_subject_model:
            raise RuntimeError("Native MuJoCo extraction model changed; reset required")

    @contextmanager
    def _projected_observation_data(self):
        from .mujoco_projection import MujocoProjectionData

        if self._native_projection is None:
            self._bind_extraction_subject(reset=False)
        projection = self._native_projection
        if projection is None:
            raise RuntimeError("Native MuJoCo projection did not bind a subject")
        wrapped = self._data
        source = _raw_struct(wrapped, "_data")
        try:
            with projection.observe(source) as snapshot:
                self._native_snapshot = snapshot
                self._native_stored_data = source
                self._native_stored_wrapper = wrapped
                self._data = MujocoProjectionData(snapshot)
                yield snapshot
        finally:
            self._data = wrapped
            self._native_snapshot = None
            self._native_stored_data = None
            self._native_stored_wrapper = None

    @contextmanager
    def _completion_stage_data(self, contract):
        """Read a completion-derived field from the completed scratch stage.

        Fields whose contract names a completion call (``energy``,
        ``subtree_*``, ``cacc``, ``cfrc_*``) are the adapter's own
        completions (issue #76): their scalars come from the detached
        scratch copy after the declared stages, not from the stored live
        stage, which the caller may never have completed. Stored fields
        keep reading the live stage.
        """
        from .mujoco_projection import MujocoProjectionData

        if contract.completion_call_id is None or self._native_snapshot is None:
            yield
            return
        stored = self._data
        self._data = MujocoProjectionData(self._native_snapshot)
        try:
            yield
        finally:
            self._data = stored

    @contextmanager
    def _stored_observation_data(self):
        """Read native field declarations and values from captured source data."""
        projected = self._data
        if self._native_stored_wrapper is None:
            raise RuntimeError("Stored native extraction requires an active observation")
        try:
            self._data = self._native_stored_wrapper
            yield self._native_stored_data
        finally:
            self._data = projected

    def close_native_extraction(self) -> None:
        """Retire observation resources without closing the caller's simulator."""
        if self._native_closed:
            return
        try:
            self._native_resources.close()
        finally:
            # ExitStack attempts each callback and the retired subject must
            # never be reused.
            self._native_closed = True
            self._native_projection = None
            self._native_subject_model = None

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def spatial_dims(self) -> int:
        return 3

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Declared authority for episode-start compilation (§5.6.1).

        The class vocabulary mirrors ``supported_classes`` (closed when
        a subclass declares one, open otherwise), extended with the two
        dynamic contact classes, whose component schemas are declared
        CLOSED so that ``specola check`` and ``specola run`` resolve a
        contact specification identically even when the reset snapshot
        contains no contact — dynamic populations are runtime facts,
        not schema facts. Entity IDs stay open because contact pairs
        allocate IDs as they first appear.
        """
        supported = self.supported_classes
        if supported is not None:
            class_mode = VocabularyMode.CLOSED
            declared_classes = frozenset(supported) | {
                CONTACT_ENTITY_CLASS,
                ROBOT_CONTACT_ENTITY_CLASS,
            }
        else:
            class_mode = VocabularyMode.OPEN
            declared_classes = frozenset()

        from .mujoco_geom_pairs import GEOM_PAIR_ENTITY_CLASS, GEOM_PAIR_PROPERTY_KEYS

        if supported is not None:
            declared_classes = declared_classes | {GEOM_PAIR_ENTITY_CLASS}
        contact_components = frozenset(CONTACT_PROPERTY_KEYS)
        contact_tags = frozenset(self._contact_tag_vocabulary())
        geom_pair_tags = frozenset(
            tag for record in (self._geom_pair_records or {}).values() for tag in record.tags
        )
        component_tags_by_class = (
            {
                CONTACT_ENTITY_CLASS: contact_tags,
                ROBOT_CONTACT_ENTITY_CLASS: contact_tags,
            }
            if contact_tags
            else {}
        )
        if geom_pair_tags:
            component_tags_by_class[GEOM_PAIR_ENTITY_CLASS] = geom_pair_tags
        return AdapterCapabilities(
            spatial_dims=int(self.spatial_dims),
            class_mode=class_mode,
            supported_classes=declared_classes,
            components_by_class={
                CONTACT_ENTITY_CLASS: contact_components,
                ROBOT_CONTACT_ENTITY_CLASS: contact_components,
                GEOM_PAIR_ENTITY_CLASS: frozenset(GEOM_PAIR_PROPERTY_KEYS),
            },
            component_mode_by_class={
                CONTACT_ENTITY_CLASS: VocabularyMode.CLOSED,
                ROBOT_CONTACT_ENTITY_CLASS: VocabularyMode.CLOSED,
                GEOM_PAIR_ENTITY_CLASS: VocabularyMode.CLOSED,
            },
            component_tags_by_class=component_tags_by_class,
            runtime_populated_classes=frozenset(
                {CONTACT_ENTITY_CLASS, ROBOT_CONTACT_ENTITY_CLASS}
            ),
            entity_id_mode=VocabularyMode.OPEN,
        )

    def _contact_tag_vocabulary(self) -> FrozenSet[str]:
        """Possible ``_contact_pair_tags`` values (subclass hook).

        Declared in capabilities so an ``@``-filtered contact selector
        can be validated at compilation while the pair population is
        still empty. Must be a superset of every tag the pair hook can
        emit for the current simulator policy; the base adapter emits
        none.
        """
        return frozenset()

    @property
    def joint_signals(self) -> Tuple[str, ...]:
        """Live curated joint fields, fixed by global binding capability."""
        return self._joint_signals

    @property
    def semantic_policy(self) -> MujocoSemanticPolicy:
        """Current reset-derived simulator semantics."""
        return self._semantic_policy_value

    @property
    def field_surface_report(self) -> FieldSurfaceReport:
        """Name-only dispositions for the current binding object.

        ``scope_active`` says whether the per-field dispositions below
        describe the full extraction surface or the projection a monitor
        session installed: under an active scope a field can report no
        emitted keys simply because no compiled spec asked for any.
        """
        if self._extraction_scope is None:
            return self._field_surface_report
        return replace(self._field_surface_report, scope_active=True)

    @property
    def field_materialization(
        self,
    ) -> Mapping[str, MujocoFieldMaterialization]:
        """Per-field runtime materialisation for the latest observation."""
        return dict(self._field_materialization)

    def set_extraction_scope(
        self,
        scope: Optional[Mapping[str, FrozenSet[str]]],
    ) -> None:
        """Install a new extraction scope and rebuild against it.

        Every steady-state shortcut this adapter builds — gather plans,
        value-only sink tables, the per-field steady set — was derived
        from a pass under the *previous* scope, and a gather plan in
        particular writes its targets without consulting the scope at
        all. Changing the scope therefore has to discard them, so that
        the next observation runs one full pass and rebuilds each of
        them over the keys the new scope keeps.

        This matters on the ordinary path, not just an exotic one: a
        session compiles its specs against the catalog that ``reset()``
        discovers, so the scope always arrives after a full unscoped
        pass has already planned every field.

        The entity scope (``set_extraction_entity_scope``) changes the
        ``geom_pair`` population the same way, so it also rebuilds.
        """
        super().set_extraction_scope(scope)
        self._invalidate_materialization()

    def set_extraction_entity_scope(
        self,
        entities: Optional[Mapping[str, FrozenSet[tuple]]],
    ) -> None:
        """Restrict the ``geom_pair`` population to pairs a compiled slot can bind."""
        super().set_extraction_entity_scope(entities)
        self._invalidate_materialization()

    def _invalidate_materialization(self) -> None:
        """Drop every per-episode extraction shortcut and its bookkeeping.

        Called on reset, and whenever the extraction scope changes. The
        model-derived name and identity caches are deliberately not
        included: they survive a scope change and are cleared separately
        on reset, where the model itself may have changed.
        """
        self._parity_validated.clear()
        self._merged_meta_cache.clear()
        self._materialization_steady = False
        self._steady_fields.clear()
        self._emit_value_only = False
        self._value_only_sinks.clear()
        self._gather_plan.clear()
        self._plan_capture = None
        self._plan_position_rows.clear()
        self._scope_dropped = 0
        self._scope_skipped_fields.clear()

    # ------------------------------------------------------------------
    # SimulatorAdapter template methods
    # ------------------------------------------------------------------

    def _reset_impl(self) -> MonitorObservation:
        self._refresh_backend(allow_model_change=True)
        model = _raw_struct(self._model, "_model")
        if self._native_projection is None or model is not self._native_subject_model:
            self._bind_extraction_subject(reset=True)
        with self._projected_observation_data():
            return self._reset_snapshot_impl()

    def _reset_snapshot_impl(self) -> MonitorObservation:
        """
        Re-enumerate entities after the caller resets the environment.

        This method resets only the adapter. The caller remains responsible
        for resetting the simulator environment itself.
        """
        # Cleared before the backend refresh: the refresh and the reset
        # contract resolve names against the (possibly new) model, and a
        # stale entry here would feed them the previous model's names.
        self._id2name_cache.clear()
        self._geom_type_cache.clear()
        self._identity_tags_cache.clear()
        self._invalidate_materialization()
        self._body_owner_pairs_cache = None
        self._joint_owner_triples_cache = None
        self._rebuild_reset_contract()

        self.name_to_id.clear()
        self.entity_labels.clear()
        self._bodies.clear()
        self._gripper = None
        self._regions.clear()
        self._sensors.clear()
        self._orphan_sensor_entity_id = None
        self._actuators.clear()
        self._orphan_actuator_entity_id = None
        self._simulation_entity_id = None
        self._raw_bodies.clear()
        self._raw_sites.clear()
        self._raw_geoms.clear()
        self._raw_cameras.clear()
        self._raw_lights.clear()
        self._equalities.clear()
        self._flexes.clear()
        self._tendons.clear()
        self._mocap_owners.clear()
        self._declared_entities.clear()
        self._materialized_property_meta.clear()
        self._field_materialization.clear()
        self._current_owner_omissions.clear()
        self._previous_positions.clear()
        self._contact_pairs.clear()
        self._source_token_cache.clear()
        self._joint_token_cache = None
        self._timestep = 0

        next_id = 0
        classified = []

        for order, body_name in enumerate(
            self._names("body")
        ):
            matched = self._classify(body_name)

            if matched is None:
                continue

            (
                entity_class,
                priority,
                joint_scope,
            ) = matched

            body_id = self._name2id(
                "body",
                body_name,
            )

            classified.append(
                (
                    priority,
                    order,
                    body_name,
                    body_id,
                    entity_class,
                    joint_scope,
                )
            )

        classified.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        for (
            _priority,
            _order,
            body_name,
            body_id,
            entity_class,
            joint_scope,
        ) in classified:
            self._reserve_name(
                body_name,
                next_id,
            )

            self._bodies.append(
                _BodyRecord(
                    entity_id=next_id,
                    body_name=body_name,
                    body_id=body_id,
                    entity_class=entity_class,
                    joint_ids=self._joint_ids(
                        body_id,
                        joint_scope,
                    ),
                    pose_body_ids=self._resolve_pose_body_ids(
                        body_id,
                        joint_scope,
                    ),
                )
            )

            next_id += 1

        site_name = (
            self.gripper_site
            or self._auto_gripper_site()
        )

        if site_name is not None:
            site_id = self._name2id(
                "site",
                site_name,
            )

            self._reserve_name(
                site_name,
                next_id,
            )

            self._gripper = _SiteRecord(
                entity_id=next_id,
                site_name=site_name,
                site_id=site_id,
                entity_class=self.gripper_class,
            )

            next_id += 1

        region_specs = list(self.regions)

        if not all(
            isinstance(spec, RegionSpec)
            for spec in region_specs
        ):
            raise TypeError(
                "_discover_extra_regions() must return "
                "RegionSpec objects"
            )

        for spec in region_specs:
            self._reserve_name(
                spec.name,
                next_id,
            )

            self._regions.append(
                _RegionRecord(
                    entity_id=next_id,
                    spec=spec,
                )
            )

            next_id += 1

        next_id = self._register_sensors(next_id)
        next_id = self._register_actuators(next_id)
        next_id = self._register_declared_entities(next_id)
        next_id = self._register_curated_field_owners(next_id)

        # Contact pairs are allocated at runtime as pairs first touch;
        # they continue the reset allocation so no reset-time ID can be
        # recycled for a pair (and vice versa) within the episode.
        self._next_runtime_entity_id = next_id
        self._robot_body_ids = self._resolve_robot_body_ids()

        self._build_labels()
        self._register_geom_pair_entities()

        return self._build_observation()

    def _get_observation_impl(
        self,
    ) -> MonitorObservation:
        accepted_pairs = frozenset(self._contact_pairs)
        next_entity_id = self._next_runtime_entity_id
        contact_status = self._field_materialization.get("contact")
        accepted_metadata = self._materialized_property_meta
        accepted_materialization = self._field_materialization
        accepted_steady = self._materialization_steady
        accepted_scope_dropped = self._scope_dropped
        self._prospective_metadata_keys = []
        # Pose and velocity projections are completed on detached data. Stored
        # native contacts, qacc and constraint inputs retain their source stage.
        try:
            self._refresh_backend(allow_model_change=False)
            with self._projected_observation_data():
                return self._build_observation()
        except Exception:
            self._rollback_rejected_contact_state(
                accepted_pairs,
                next_entity_id,
                contact_status,
                accepted_metadata,
                accepted_materialization,
                accepted_steady,
                accepted_scope_dropped,
            )
            raise
        finally:
            self._prospective_metadata_keys = None

    def _rollback_rejected_contact_state(
        self,
        accepted_pairs: FrozenSet[Tuple[int, int]],
        next_entity_id: int,
        contact_status: Optional[MujocoFieldMaterialization],
        accepted_metadata: Dict[int, Dict[str, PropertyMeta]],
        accepted_materialization: Dict[str, MujocoFieldMaterialization],
        accepted_steady: bool,
        accepted_scope_dropped: int,
    ) -> None:
        """Remove dynamic contact state introduced by a rejected build."""
        new_pairs = set(self._contact_pairs) - accepted_pairs
        for pair in new_pairs:
            record = self._contact_pairs.pop(pair)
            self.name_to_id.pop(record.label, None)
            self.entity_labels.pop(record.entity_id, None)
            self._entity_class_by_id.pop(record.entity_id, None)
            self._parity_validated.discard(record.entity_id)
        additions = tuple(self._prospective_metadata_keys or ())
        if self._materialized_property_meta is accepted_metadata:
            for entity_id, key in additions:
                metadata = self._materialized_property_meta.get(entity_id)
                if metadata is not None:
                    metadata.pop(key, None)
                    if not metadata:
                        self._materialized_property_meta.pop(entity_id)
        else:
            self._materialized_property_meta = accepted_metadata
        for key in additions:
            self._merged_meta_cache.pop(key, None)
        self._next_runtime_entity_id = next_entity_id
        if self._field_materialization is not accepted_materialization:
            self._field_materialization = accepted_materialization
        elif contact_status is None:
            accepted_materialization.pop("contact", None)
        else:
            accepted_materialization["contact"] = contact_status
        if not accepted_steady:
            self._invalidate_materialization()
            self._materialized_property_meta = accepted_metadata
            self._field_materialization = accepted_materialization
        self._scope_dropped = accepted_scope_dropped

    def _kinematic_forward(self) -> None:
        """Require the detached completion already performed for this read."""
        if self._native_snapshot is None:
            raise RuntimeError("MuJoCo kinematic projection requires an active detached observation")

    def _semantic_policy(self) -> MujocoSemanticPolicy:
        """Build simulator semantics from the current reset model.

        Availability and materialisation do not appear in this hook.  A
        subclass may classify bodies, declare semantic regions, or supply
        task identities; the shared field registry remains unchanged.
        """
        return MujocoSemanticPolicy(
            body_rules=(
                tuple(self._default_body_rules())
                if self._explicit_body_rules is None
                else self._explicit_body_rules
            ),
            regions=(
                tuple(self._discover_extra_regions())
                if self._explicit_regions is None
                else self._explicit_regions
            ),
            declared_entities=tuple(self._declare_task_entities()),
            robot_bodies=(
                tuple(self._default_robot_bodies())
                if self._explicit_robot_bodies is None
                else self._explicit_robot_bodies
            ),
        )

    def _default_body_rules(
        self,
    ) -> Sequence[BodyRule]:
        """
        Return simulator-specific body classification rules.

        The generic MuJoCo adapter does not register any bodies unless rules
        are supplied by the caller or a subclass.
        """
        return ()

    def _default_gripper_sites(
        self,
    ) -> Sequence[str]:
        """Return candidate gripper site names in preference order."""
        return ()

    def _default_robot_bodies(
        self,
    ) -> Sequence[BodyMatcher]:
        """Return matchers naming robot bodies for contact ownership.

        With no matchers, every contact pair classifies as the generic
        ``contact`` class; the ``robot_contact`` class still exists and
        is still declared in capabilities, it is simply unpopulated.
        """
        return ()

    def _resolve_robot_body_ids(self) -> FrozenSet[int]:
        """Resolve the policy's robot-body matchers against the model."""
        matchers = self.semantic_policy.robot_bodies
        if not matchers:
            return frozenset()

        # _names() drops unnamed bodies, so its position is not a body
        # ID; resolve each matched name through the model's name table.
        robot_ids = set()
        for body_name in self._names("body"):
            for matcher in matchers:
                if callable(matcher):
                    matched = bool(matcher(body_name))
                else:
                    matched = matcher in body_name
                if matched:
                    robot_ids.add(self._name2id("body", body_name))
                    break
        return frozenset(robot_ids)

    def _discover_extra_regions(
        self,
    ) -> Sequence[RegionSpec]:
        """
        Return task-specific regions when explicit regions were not supplied.

        Subclasses may use ``source="fixed"`` for regions discovered through
        task metadata rather than named MuJoCo geometry.
        """
        return ()

    def _declare_task_entities(
        self,
    ) -> Sequence[DeclaredEntitySpec]:
        """Return reset-time task identities; Task 7 materialises them."""
        return ()

    def _additional_joint_owners(
        self,
    ) -> Sequence[MujocoJointOwner]:
        """Return declaration-backed aliases for model-joint ownership.

        The hook is simulator-semantic policy only: it maps registered
        entities to exact model IDs and cannot select fields or read values.
        Every globally included joint field is still read, emitted, and
        metadata-tracked by the shared MuJoCo materializers.
        """
        return ()

    def _additional_body_owners(
        self,
    ) -> Sequence[MujocoBodyOwner]:
        """Return declaration-backed owners for compiled body rows."""
        return ()

    def _declared_entity_states(
        self,
        registrations: Mapping[str, int],
    ) -> Mapping[str, EntityState]:
        """Return one live state for every registered declaration key.

        Subclasses with declarations override this hook.  Exact key, entity
        id, and entity-class reconciliation is enforced every observation.
        """
        return {}

    def _declared_entity_property_metadata(
        self,
        declaration_key: str,
    ) -> Mapping[str, PropertyMeta]:
        """Return metadata for one task-declared entity's live properties."""
        return {}

    def _object_category(
        self,
        body_name: str,
    ) -> Optional[str]:
        """
        Return an optional human-readable body category.

        Despite the historical name, this hook may return a label for any
        monitored body.
        """
        return None

    def _body_properties(
        self,
        entity_id: int,
        body_name: str,
        body_id: int,
        joint_ids: Sequence[int],
    ) -> Dict[str, float]:
        """
        Return scalar properties for one monitored body.

        Subclasses may override this method to add simulator-specific
        properties. Implementations should normally call ``super()`` first.
        """
        del entity_id, body_name, body_id, joint_ids
        return {}

    def _gripper_properties(
        self,
        entity_id: int,
        site_name: str,
        site_id: int,
    ) -> Dict[str, float]:
        """
        Return scalar properties for the gripper-site entity.

        RoboSuite-derived adapters may expose finger joint positions,
        end-effector force, end-effector torque, or other physical signals.
        Sensors defined on the gripper's exact site are merged in here
        too, matching the gripper-site ownership rule applied in
        ``_sensor_owner_entity_id``.
        """
        del entity_id, site_name, site_id
        return {}

    def _gripper_property_meta(
        self,
    ) -> Dict[str, PropertyMeta]:
        """
        Display metadata for the gripper-site entity's base properties.

        Covers the ``site_xpos_*`` keys this base class extracts in
        ``_gripper_properties``, plus metadata for any sensor defined on
        the gripper's own site. A subclass that adds extra gripper
        properties (finger joints, end-effector force/torque, ...) must
        override this to add matching entries, or ``property_metadata``
        stops covering every key the entity actually exposes.
        """
        if self._gripper is None:
            return {}

        token = self._source_property_token(
            "site", self._gripper.site_id
        )

        meta: Dict[str, PropertyMeta] = {
            f"site_xpos_{token}_{axis}": PropertyMeta(
                group="site position",
                note="m",
            )
            for axis in range(3)
        }

        meta.update(
            self._sensor_property_meta(
                self._owned_sensor_records(
                    self._gripper.entity_id
                )
            )
        )

        return meta

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _register_curated_field_owners(
        self,
        next_id: int,
    ) -> int:
        """Register every positive owner family required by the registry."""
        self._reserve_name("simulation:global", next_id)
        self._simulation_entity_id = next_id
        next_id += 1

        claimed_body_ids = {
            body_id
            for body in self._bodies
            for body_id in body.pose_body_ids
        }
        for owner in self._validated_additional_body_owners():
            for body_id in owner.body_ids:
                if body_id in claimed_body_ids:
                    raise ValueError(
                        "Body row {} has more than one semantic owner"
                        .format(body_id)
                    )
                claimed_body_ids.add(body_id)
        body_count = int(getattr(self._model, "nbody", 0) or 0)
        for body_id in range(body_count):
            if body_id in claimed_body_ids:
                continue
            body_name = self._indexed_source_name("body", body_id)
            self._reserve_name("body:{}".format(body_name), next_id)
            self._raw_bodies.append(
                _NamedOwnerRecord(
                    entity_id=next_id,
                    source_name=body_name,
                    source_id=body_id,
                    entity_class="body",
                )
            )
            next_id += 1

        gripper_site_id = (
            None if self._gripper is None else self._gripper.site_id
        )
        for site_id in range(int(getattr(self._model, "nsite", 0) or 0)):
            if site_id == gripper_site_id:
                continue
            site_name = self._indexed_source_name("site", site_id)
            self._reserve_name("site:{}".format(site_name), next_id)
            self._raw_sites.append(
                _NamedOwnerRecord(
                    entity_id=next_id,
                    source_name=site_name,
                    source_id=site_id,
                    entity_class="site",
                )
            )
            next_id += 1

        for geom_id in range(int(getattr(self._model, "ngeom", 0) or 0)):
            geom_name = self._indexed_source_name("geom", geom_id)
            self._reserve_name("geom:{}".format(geom_name), next_id)
            self._raw_geoms.append(
                _NamedOwnerRecord(
                    entity_id=next_id,
                    source_name=geom_name,
                    source_id=geom_id,
                    entity_class="geom",
                )
            )
            next_id += 1

        for camera_id in range(int(getattr(self._model, "ncam", 0) or 0)):
            camera_name = self._indexed_source_name("camera", camera_id)
            self._reserve_name("camera:{}".format(camera_name), next_id)
            self._raw_cameras.append(
                _NamedOwnerRecord(
                    entity_id=next_id,
                    source_name=camera_name,
                    source_id=camera_id,
                    entity_class="camera",
                )
            )
            next_id += 1

        for light_id in range(
            int(getattr(self._model, "nlight", 0) or 0)
        ):
            light_name = self._indexed_source_name("light", light_id)
            self._reserve_name("light:{}".format(light_name), next_id)
            self._raw_lights.append(
                _NamedOwnerRecord(
                    entity_id=next_id,
                    source_name=light_name,
                    source_id=light_id,
                    entity_class="light",
                )
            )
            next_id += 1

        for kind, entity_class, destination in (
            ("equality", "equality", self._equalities),
            ("flex", "flex", self._flexes),
            ("tendon", "tendon", self._tendons),
        ):
            count_name = {
                "equality": "neq",
                "flex": "nflex",
                "tendon": "ntendon",
            }[kind]
            for source_id in range(
                int(getattr(self._model, count_name, 0) or 0)
            ):
                source_name = self._indexed_source_name(kind, source_id)
                self._reserve_name(
                    "{}:{}".format(kind, source_name), next_id
                )
                destination.append(
                    _NamedOwnerRecord(
                        entity_id=next_id,
                        source_name=source_name,
                        source_id=source_id,
                        entity_class=entity_class,
                    )
                )
                next_id += 1

        mocap_ids = np.asarray(
            getattr(self._model, "body_mocapid", ()),
            dtype=int,
        ).reshape(-1)
        for body_id, mocap_id in enumerate(mocap_ids):
            if mocap_id < 0:
                continue
            body_name = self._id2name("body", body_id)
            if body_name is None:
                body_name = self._indexed_source_name("body", body_id)
            self._reserve_name(
                "mocap:{}".format(body_name), next_id
            )
            self._mocap_owners.append(
                _MocapOwnerRecord(
                    entity_id=next_id,
                    body_name=body_name,
                    body_id=body_id,
                    mocap_id=int(mocap_id),
                )
            )
            next_id += 1

        return next_id

    def _indexed_source_name(self, kind: str, object_id: int) -> str:
        """Return model name or a reset-local, index-backed identity."""
        model_name = self._id2name(kind, object_id)
        if model_name is not None:
            return model_name
        count_name = {
            "body": "nbody",
            "joint": "njnt",
            "geom": "ngeom",
            "site": "nsite",
            "sensor": "nsensor",
            "actuator": "nu",
            "camera": "ncam",
            "light": "nlight",
            "equality": "neq",
            "flex": "nflex",
            "tendon": "ntendon",
        }[kind]
        existing = {
            name
            for index in range(
                int(getattr(self._model, count_name, 0) or 0)
            )
            for name in (self._id2name(kind, index),)
            if name is not None
        }
        candidate = "specola_unnamed_{}_index_{}".format(kind, object_id)
        while candidate in existing:
            candidate += "_reserved"
        return candidate

    def _source_property_tokens(self, kind: str) -> Tuple[str, ...]:
        """Collision-free parser tokens for one compiled object kind."""
        cached = self._source_token_cache.get(kind)
        if cached is not None:
            return cached
        count_name = {
            "body": "nbody",
            "joint": "njnt",
            "geom": "ngeom",
            "site": "nsite",
            "sensor": "nsensor",
            "actuator": "nu",
            "camera": "ncam",
            "light": "nlight",
            "equality": "neq",
            "flex": "nflex",
            "tendon": "ntendon",
        }[kind]
        count = int(getattr(self._model, count_name, 0) or 0)
        bases = [
            _property_token(self._indexed_source_name(kind, object_id))
            for object_id in range(count)
        ]
        base_counts = {
            base: bases.count(base) for base in set(bases)
        }
        tokens = [
            (
                base
                if base_counts[base] == 1
                else "{}_model_index_{}".format(base, object_id)
            )
            for object_id, base in enumerate(bases)
        ]
        collisions = {
            token for token in tokens if tokens.count(token) > 1
        }
        if collisions:
            tokens = [
                (
                    "{}_source_index_{}".format(token, object_id)
                    if token in collisions
                    else token
                )
                for object_id, token in enumerate(tokens)
            ]
        if len(set(tokens)) != len(tokens):
            raise RuntimeError(
                "{} source-token allocation is not injective".format(kind)
            )
        result = tuple(tokens)
        self._source_token_cache[kind] = result
        return result

    def _source_property_token(self, kind: str, object_id: int) -> str:
        if not hasattr(self, "_model"):
            if (
                kind == "site"
                and getattr(self, "_gripper", None) is not None
                and self._gripper.site_id == object_id
            ):
                return _property_token(self._gripper.site_name)
            return "{}_index_{}".format(kind, object_id)
        return self._source_property_tokens(kind)[object_id]

    def _joint_property_tokens(self) -> Tuple[str, ...]:
        """Joint tokens whose scalar/multi-coordinate keys cannot collide."""
        if self._joint_token_cache is not None:
            return self._joint_token_cache
        bases = self._source_property_tokens("joint")
        qpos_used = set()
        dof_used = set()
        tokens = []
        for joint_id, base in enumerate(bases):
            qpos_start, qpos_stop = self._joint_slice(
                joint_id, "jnt_qposadr", "nq"
            )
            dof_start, dof_stop = self._joint_slice(
                joint_id, "jnt_dofadr", "nv"
            )
            candidate = base
            suffix = 0
            while True:
                qpos_keys = set(
                    self._signal_keys(
                        "qpos", candidate, qpos_stop - qpos_start
                    )
                )
                dof_keys = set(
                    self._signal_keys(
                        "qvel", candidate, dof_stop - dof_start
                    )
                )
                if not (qpos_keys & qpos_used or dof_keys & dof_used):
                    break
                suffix += 1
                candidate = "{}_model_index_{}_{}".format(
                    base, joint_id, suffix
                )
            tokens.append(candidate)
            qpos_used.update(qpos_keys)
            dof_used.update(dof_keys)
        result = tuple(tokens)
        self._joint_token_cache = result
        return result

    def _joint_property_token(self, joint_id: int) -> str:
        return self._joint_property_tokens()[joint_id]

    def _source_identity_tags(
        self,
        kind: str,
        object_id: int,
    ) -> Tuple[str, ...]:
        """Metadata proving whether identity came from a name or model ID."""
        cache_key = (kind, object_id)
        cached = self._identity_tags_cache.get(cache_key)
        if cached is not None:
            return cached
        tags = (
            "source_kind:{}".format(kind),
            "model_index:{}".format(object_id),
            "identity:{}".format(
                "model_name"
                if self._id2name(kind, object_id) is not None
                else "compiled_model_index"
            ),
        )
        self._identity_tags_cache[cache_key] = tags
        return tags

    def _qualified_source_identity_tags(
        self,
        role: str,
        kind: str,
        object_id: int,
    ) -> Tuple[str, ...]:
        """Namespaced source identity for properties with multiple owners."""
        cache_key = (role, kind, object_id)
        cached = self._identity_tags_cache.get(cache_key)
        if cached is not None:
            return cached
        tags = (
            "{}_source_kind:{}".format(role, kind),
            "{}_model_index:{}".format(role, object_id),
            "{}_identity:{}".format(
                role,
                (
                    "model_name"
                    if self._id2name(kind, object_id) is not None
                    else "compiled_model_index"
                ),
            ),
        )
        self._identity_tags_cache[cache_key] = tags
        return tags

    def _sensor_target_tags(
        self,
        sensor_id: int,
    ) -> Tuple[str, ...]:
        """Retain a sensor's exact compiled object and reference relation."""
        import mujoco

        tags = []
        for role, type_field, id_field in (
            ("object", "sensor_objtype", "sensor_objid"),
            ("reference", "sensor_reftype", "sensor_refid"),
        ):
            object_type = int(getattr(self._model, type_field)[sensor_id])
            object_id = int(getattr(self._model, id_field)[sensor_id])
            type_name = mujoco.mjtObj(object_type).name.replace(
                "mjOBJ_", "", 1
            ).lower()
            tags.extend(
                (
                    "sensor_{}_type:{}".format(role, type_name),
                    "sensor_{}_index:{}".format(role, object_id),
                )
            )
        return tuple(tags)

    def _actuator_transmission_tags(
        self,
        actuator_id: int,
    ) -> Tuple[str, ...]:
        """Retain an actuator's exact compiled transmission relation."""
        import mujoco

        transmission_type = int(
            self._model.actuator_trntype[actuator_id]
        )
        type_name = mujoco.mjtTrn(transmission_type).name.replace(
            "mjTRN_", "", 1
        ).lower()
        object_ids = np.asarray(
            self._model.actuator_trnid[actuator_id], dtype=int
        ).reshape(-1)
        return (
            ("actuator_transmission_type:{}".format(type_name),)
            + tuple(
                "actuator_transmission_object{}_index:{}".format(
                    index, object_id
                )
                for index, object_id in enumerate(object_ids)
            )
        )

    def _register_declared_entities(
        self,
        next_id: int,
    ) -> int:
        """Reserve stable identities for all reset-derived declarations."""
        for spec in self.semantic_policy.declared_entities:
            self._reserve_name(
                "declared:{}".format(spec.declaration_key), next_id
            )
            self._declared_entities.append(
                _DeclaredEntityRecord(entity_id=next_id, spec=spec)
            )
            next_id += 1
        return next_id

    def _populate_constraint_dependent_fields(self) -> None:
        """Require completed scratch force, velocity and energy projections.

        Stored generalized acceleration and contact/constraint inputs remain
        unchanged. Completion does not solve a new native physical stage.
        """
        if self._native_snapshot is None:
            raise RuntimeError("MuJoCo scalar projection requires an active detached observation")

    def _build_observation(
        self,
    ) -> MonitorObservation:
        self._populate_constraint_dependent_fields()

        perceived: Dict[int, EntityState] = {}

        for body in self._bodies:
            position = np.asarray(
                self._body_positions()[body.body_id],
                dtype=float,
            ).copy()

            velocity = self._body_velocity(
                body,
                position,
            )

            properties = self._body_properties(
                body.entity_id,
                body.body_name,
                body.body_id,
                body.joint_ids,
            )

            entity = EntityState(
                entity_id=body.entity_id,
                entity_class=body.entity_class,
                position=position,
                velocity=velocity,
                track_age=self._timestep,
                track_state=TrackState.CONFIRMED,
                properties=dict(properties),
            )

            if self._is_perceived(entity):
                perceived[body.entity_id] = entity

        if self._gripper is not None:
            gripper = self._gripper

            position = np.asarray(
                self._data.site_xpos[gripper.site_id],
                dtype=float,
            ).copy()

            velocity = self._site_velocity(
                gripper,
                position,
            )

            properties = self._gripper_properties(
                gripper.entity_id,
                gripper.site_name,
                gripper.site_id,
            )

            entity = EntityState(
                entity_id=gripper.entity_id,
                entity_class=gripper.entity_class,
                position=position,
                velocity=velocity,
                track_age=self._timestep,
                track_state=TrackState.CONFIRMED,
                properties=dict(properties),
            )

            if self._is_perceived(entity):
                perceived[gripper.entity_id] = entity

        for record in self._regions:
            perceived[record.entity_id] = (
                self._build_region_state(record)
            )

        if self._orphan_sensor_entity_id is not None:
            orphan_entity = EntityState(
                entity_id=self._orphan_sensor_entity_id,
                entity_class=_ORPHAN_SENSOR_ENTITY_CLASS,
                position=None,
                velocity=None,
                track_age=self._timestep,
                track_state=TrackState.CONFIRMED,
                properties={},
            )

            if self._is_perceived(orphan_entity):
                perceived[
                    self._orphan_sensor_entity_id
                ] = orphan_entity

        if self._orphan_actuator_entity_id is not None:
            orphan_actuator_entity = EntityState(
                entity_id=self._orphan_actuator_entity_id,
                entity_class=_ORPHAN_ACTUATOR_ENTITY_CLASS,
                position=None,
                velocity=None,
                track_age=self._timestep,
                track_state=TrackState.CONFIRMED,
                properties={},
            )

            if self._is_perceived(orphan_actuator_entity):
                perceived[
                    self._orphan_actuator_entity_id
                ] = orphan_actuator_entity

        if self._simulation_entity_id is None:
            raise RuntimeError(
                "MuJoCo field owners were not registered; call reset()"
            )
        perceived[self._simulation_entity_id] = self._empty_owner_state(
            self._simulation_entity_id, "simulation"
        )

        for record in self._raw_bodies:
            perceived[record.entity_id] = EntityState(
                entity_id=record.entity_id,
                entity_class=record.entity_class,
                position=np.asarray(
                    self._body_positions()[record.source_id],
                    dtype=float,
                ).copy(),
                velocity=None,
                track_age=self._timestep,
                track_state=TrackState.CONFIRMED,
                properties={},
            )

        for record in self._raw_sites:
            position = np.asarray(
                self._data.site_xpos[record.source_id], dtype=float
            ).copy()
            properties: Dict[str, float] = {}
            token = self._source_property_token(
                "site", record.source_id
            )
            size = np.asarray(
                self._model.site_size[record.source_id], dtype=float
            ).reshape(-1)
            for index, value in enumerate(size):
                properties[
                    "site_size_{}_{}".format(token, index)
                ] = float(value)
            perceived[record.entity_id] = EntityState(
                entity_id=record.entity_id,
                entity_class=record.entity_class,
                position=position,
                velocity=None,
                track_age=self._timestep,
                track_state=TrackState.CONFIRMED,
                properties=properties,
            )

        for record in self._raw_geoms:
            properties = {}
            token = self._source_property_token(
                "geom", record.source_id
            )
            size = np.asarray(
                self._model.geom_size[record.source_id], dtype=float
            ).reshape(-1)
            for index, value in enumerate(size):
                properties[
                    "geom_size_{}_{}".format(token, index)
                ] = float(value)
            perceived[record.entity_id] = EntityState(
                entity_id=record.entity_id,
                entity_class=record.entity_class,
                position=np.asarray(
                    self._data.geom_xpos[record.source_id], dtype=float
                ).copy(),
                velocity=None,
                track_age=self._timestep,
                track_state=TrackState.CONFIRMED,
                properties=properties,
            )

        for record in self._raw_cameras:
            perceived[record.entity_id] = EntityState(
                entity_id=record.entity_id,
                entity_class=record.entity_class,
                position=np.asarray(
                    self._data.cam_xpos[record.source_id], dtype=float
                ).copy(),
                velocity=None,
                track_age=self._timestep,
                track_state=TrackState.CONFIRMED,
                properties={},
            )

        for record in self._raw_lights:
            perceived[record.entity_id] = self._empty_owner_state(
                record.entity_id, record.entity_class
            )

        for records in (self._equalities, self._flexes, self._tendons):
            for record in records:
                perceived[record.entity_id] = self._empty_owner_state(
                    record.entity_id, record.entity_class
                )

        for record in self._mocap_owners:
            perceived[record.entity_id] = EntityState(
                entity_id=record.entity_id,
                entity_class="mocap",
                position=np.asarray(
                    self._data.mocap_pos[record.mocap_id], dtype=float
                ).copy(),
                velocity=None,
                track_age=self._timestep,
                track_state=TrackState.CONFIRMED,
                properties={},
            )

        perceived.update(self._materialize_declared_entity_states())
        if self._extraction_scope is not None:
            self._restrict_to_scope(perceived)
        projected_positions = {identity: state.position for identity, state in perceived.items()
                               if state.position is not None}
        with self._stored_observation_data():
            self._materialize_curated_fields(perceived)
        # Raw pose properties retain the native stored stage. Their legacy
        # materializers also assign position; keep the explicitly completed
        # position projection separate from those raw scalar components.
        for identity, position in projected_positions.items():
            if identity in perceived:
                perceived[identity].position = position
        # After the curated pass: it resets the per-observation
        # materialized-metadata store, which would orphan these keys'
        # metadata if they were emitted earlier.
        self._emit_flex_rest_lengths(perceived)
        self._validate_property_metadata_parity(perceived)
        # After the legacy parity check: the entity scope decides which
        # pairs are read (issue #136); each read pair carries both typed
        # keys whatever the key scope says, and the post-read join validates
        # them against their declared projections instead.
        self._materialize_geom_pairs(perceived)

        component_tags: Dict[int, Dict[str, Tuple[str, ...]]] = {}
        pair_records = list(self._contact_pairs.values())
        if self._geom_pair_records is not None:
            pair_records.extend(self._geom_pair_records.values())
        for record in pair_records:
            if record.tags and record.entity_id in perceived:
                component_tags[record.entity_id] = {
                    key: record.tags
                    for key in perceived[
                        record.entity_id
                    ].properties
                }

        observation = MonitorObservation(
            timestep=self._timestep,
            time=float(self._data.time),
            perceived_entities=perceived,
            component_tags=component_tags,
            extraction=self._completed_extraction(),
        )

        self._timestep += 1

        return observation

    def _completed_extraction(self):
        """Join native completed arrays to the provider's prepared subject.

        The shared post-read relation rejects missing quantities from any
        other reader. Capturing this subset does not relax the required
        provider guard or establish producer completeness.
        """
        from dataclasses import asdict, replace
        from ..conformance.extraction import quantity_metadata
        from ..conformance.verification import current_verification
        from ..extraction_observation import ExtractionObservation

        prepared = self._prepared_extraction
        if prepared is None:
            return None
        if prepared.configuration_identity is None:
            raise RuntimeError("completed extraction requires the prepared model configuration identity")
        projection, snapshot = self._native_projection, self._native_snapshot
        if projection is None or snapshot is None:
            raise RuntimeError("completed extraction requires an active native projection")
        declared = {quantity.quantity_id: quantity for quantity in prepared.manifest.quantities}
        for quantity in projection._completed_fields.quantities:
            if declared.get(quantity.quantity_id) != quantity:
                raise RuntimeError("prepared completed-field semantics differ: " + quantity.quantity_id)
        captured = projection.capture(snapshot, asdict(prepared.configuration_identity))
        population = prepared.population
        return ExtractionObservation(
            self._timestep, float(self._data.time), population.episode_epoch,
            population.configuration_epoch, population.producer_surface_sha256,
            population.manifest_sha256, population.queries,
            tuple(quantity_metadata(quantity) for quantity in prepared.manifest.quantities),
            tuple(replace(sample, binding_id=self._completed_binding_ids[sample.quantity_id])
                  for sample in captured.samples) + tuple(self._geometry_samples),
            verification_only=current_verification(self) is not None,
        )

    def _extraction_scope_during_read(self, prepared) -> bool:
        # Legacy scalars keep the issue #101 scope behavior under the bounded
        # provider's rule; geometry projections bypass the scope below.
        return bool(getattr(prepared.population, "legacy_scalar_rule", ""))

    def _register_geom_pair_entities(self) -> None:
        """Reserve names, labels and tags for the bound pair universe."""
        from dataclasses import replace as _replace
        from .mujoco_geom_pairs import GEOM_PAIR_ENTITY_CLASS

        records = self._geom_pair_records
        if records is None:
            return
        registered = {}
        for pair, record in records.items():
            names = tuple(self._indexed_source_name("geom", geom_id) for geom_id in pair)
            label = "{}:{}|{}".format(GEOM_PAIR_ENTITY_CLASS, names[0], names[1])
            bodies = tuple(self._indexed_source_name("body", body_id) for body_id in record.body_ids)
            robot = tuple(body_id in self._robot_body_ids for body_id in record.body_ids)
            tags = ("body:" + bodies[0], "body:" + bodies[1])
            if all(robot):
                tags += ("robot", "self_collision")
            elif any(robot):
                tags += ("robot",)
            else:
                tags += ("environment",)
            self._reserve_name(label, record.entity_id)
            self.entity_labels[record.entity_id] = label
            registered[pair] = _replace(record, label=label, tags=tags)
        records.clear()
        records.update(registered)

    def _geom_pairs_in_scope(self):
        """The pair records this read declares and materializes (issue #136).

        Maintainer decision 2026-10-07: under an entity scope, only the pairs
        a compiled specification slot can bind are read, the way the
        RoboDojo adapter computes only the pair distances its query
        configuration names. A query population with ``population: all``
        keeps the whole universe under a scope.

        Reset honours an entity scope that admits no pair: a monitor
        session resets with one before it has compiled anything, reads no
        pair, and compiles against a catalog that still lists the universe
        (``unread_entity_metadata``; maintainer decision 2026-10-07). Any
        other scope reads the whole universe at reset, because the provider
        declares the reset population before the pair tags a selector
        filters on are registered.

        Returns the records and, for a proper subset, the ``(first, second)``
        geom ID arrays the reader takes (``None`` for the whole universe).
        The subset object is reused while the registry and the selectors are
        the same objects, so the provider builds, and validates, one
        population per scope; the cache holds both, so an identity match
        cannot be a new object at a recycled address.
        """
        records = self._geom_pair_records
        entity_scope = self._extraction_entity_scope
        reader = self._geometry_reader
        if records is None or entity_scope is None or reader is None or reader.population_all:
            return records, None
        selectors = entity_scope.get(GEOM_PAIR_ENTITY_CLASS, _NO_SELECTORS)
        if selectors and not self._episode_started:
            return records, None
        cached = getattr(self, "_geom_pair_scope", None)
        if cached is not None and cached[0] is records and cached[1] is selectors:
            return cached[2], cached[3]
        # Selector semantics are MonitorEngine.extraction_entity_scope's. A
        # pair's two keys carry the same tags, the pair's, so a tag filter on
        # either component is a test of the pair's tags.
        wildcard = [filters for pinned, filters in selectors if pinned is None]
        pinned: Dict[int, list] = {}
        for entity_id, filters in selectors:
            if entity_id is not None:
                pinned.setdefault(entity_id, []).append(filters)

        def admitted(record) -> bool:
            return any(all(label in record.tags for _component, labels in filters for label in labels)
                       for filters in wildcard + pinned.get(record.entity_id, []))

        scoped = {pair: record for pair, record in records.items() if admitted(record)}
        if len(scoped) == len(records):
            scoped, pairs = records, None
        else:
            pairs = (np.fromiter((pair[0] for pair in scoped), dtype=int, count=len(scoped)),
                     np.fromiter((pair[1] for pair in scoped), dtype=int, count=len(scoped)))
        self._geom_pair_scope = (records, selectors, scoped, pairs)
        return scoped, pairs

    def unread_entity_metadata(self) -> Dict[int, Tuple[str, Dict[str, PropertyMeta]]]:
        """The universe pairs the last observation left unread (issue #136).

        Each comes with its class and the metadata an unscoped read would
        register for it (``_materialize_geom_pairs`` through
        ``_emit_property``), so the reset catalog
        (``capabilities.reset_catalog``) lists the whole universe after a
        reset that read no pair.
        """
        from .mujoco_geom_pairs import pair_property_meta

        records, observation = self._geom_pair_records, self._last_observation
        if records is None or observation is None:
            return {}
        read = observation.perceived_entities
        unread = {}
        for record in records.values():
            if record.entity_id in read:
                continue
            meta = pair_property_meta(record)
            extra = self._entity_extra_tags(record.entity_id)
            if extra:
                meta = {key: item._replace(tags=tuple(item.tags) + tuple(tag for tag in extra if tag not in item.tags))
                        for key, item in meta.items()}
            unread[record.entity_id] = (GEOM_PAIR_ENTITY_CLASS, meta)
        return unread

    def _materialize_geom_pairs(self, perceived: Dict[int, EntityState]) -> None:
        """Read the scoped pair population on the active snapshot."""
        from .mujoco_geom_pairs import GEOM_PAIR_ENTITY_CLASS, geometry_samples, pair_property_meta, scalar_values

        self._geometry_samples = ()
        reader = self._geometry_reader
        records, pairs = self._geom_pairs_in_scope()
        if reader is None or records is None:
            return
        snapshot = self._native_snapshot
        if snapshot is None:
            raise RuntimeError("geometry distance extraction requires an active detached observation")
        samples = []
        # Typed projections of every read pair are emitted whatever the key
        # scope says; the post-join projection removes scoped-out keys.
        scope, self._extraction_scope = self._extraction_scope, None
        try:
            for result in reader._read_snapshot(snapshot, pairs):
                record = records.get(result.geom_ids)
                if record is None:
                    raise RuntimeError("geometry reader returned a pair outside the scoped population")
                values = scalar_values(result)
                if values is None:
                    raise RuntimeError("non-finite native geometry distance for pair {}".format(result.geom_ids))
                perceived[record.entity_id] = self._empty_owner_state(record.entity_id, GEOM_PAIR_ENTITY_CLASS)
                meta = pair_property_meta(record)
                for key, value in values.items():
                    self._emit_property(perceived, record.entity_id, key, value, meta[key])
                samples.extend(geometry_samples(reader.binding, result, record))
        finally:
            self._extraction_scope = scope
        self._parity_validated.update(record.entity_id for record in records.values())
        self._geometry_samples = tuple(samples)

    def _validate_prepared_extraction(self, prepared) -> None:
        """Check completed-array owners and layouts before scalar or raw reads."""
        from dataclasses import asdict
        from .mujoco_completed_population import completed_population

        projection = self._native_projection
        if projection is None or prepared.configuration_identity is None:
            raise RuntimeError("completed population requires a bound prepared native subject")
        quantities = projection._completed_fields.quantities
        declared = {quantity.quantity_id: quantity for quantity in prepared.manifest.quantities}
        identities = {quantity.quantity_id for quantity in quantities}
        if any(declared.get(quantity.quantity_id) != quantity for quantity in quantities):
            raise RuntimeError("prepared completed-field semantics differ")
        instances = [row for row in prepared.population.instances if row.quantity_id in identities]
        if len(instances) != len(identities) or {row.quantity_id for row in instances} != identities:
            raise RuntimeError("prepared completed-field instance population differs")
        bindings = {row.quantity_id: row.binding_id for row in instances}
        resolved = completed_population(
            self._native_subject_model, _raw_struct(self._data, "_data"), projection._guard,
            quantities, asdict(prepared.configuration_identity), bindings,
        )
        actual = {row.address: row for row in instances}
        if actual != {row.address: row for row in resolved.instances}:
            raise RuntimeError("prepared completed-field native layout differs")
        domains = {row.quantity_id: row for row in prepared.population.domains if row.quantity_id in identities}
        if domains != {row.quantity_id: row for row in resolved.domains}:
            raise RuntimeError("prepared completed-field owner domain differs")
        query_domains = {(row.binding_id, row.quantity_id): row for row in prepared.population.query_domains
                         if row.quantity_id in identities}
        if query_domains != {(row.binding_id, row.quantity_id): row for row in resolved.query_domains}:
            raise RuntimeError("prepared completed-field query domain differs")
        # Binding identity is configuration, not a measured native address.
        # The capture reader still supplies its own field and physical owners.
        self._completed_binding_ids = bindings

    def _observed_class(
        self,
        perceived: Mapping[int, EntityState],
        entity_id: int,
    ) -> Optional[str]:
        """One entity's class, as the extraction scope resolves it.

        The observation is authoritative. The episode's identity history
        covers the one case it cannot: an owner a perception gate hid
        from this step, whose sinks a value-only rerun still walks. An
        entity that appears in neither has no class yet, and the caller
        falls back to unscoped behaviour for it.
        """
        state = perceived.get(entity_id)
        if state is not None:
            return state.entity_class
        return self._entity_class_by_id.get(entity_id)

    def _restrict_to_scope(
        self,
        perceived: Dict[int, EntityState],
    ) -> None:
        """Project the properties built outside the curated-field path.

        Regions, raw geom and site sizes, task declarations and the
        ``_body_properties``/``_gripper_properties`` subclass hooks write
        straight into an entity's dict rather than through
        ``_emit_property``, so the scope has to reach them here. Running
        before ``_materialize_curated_fields`` keeps one invariant true
        for every source: an entity's keys are exactly its class's scope
        intersected with what the scene offers, which is what
        ``property_metadata`` parity checks against.
        """
        scope = self._extraction_scope
        assert scope is not None
        for state in perceived.values():
            properties = state.properties
            if not properties:
                continue
            allowed = scope.get(state.entity_class, _NO_KEYS)
            if not allowed:
                properties.clear()
            elif not properties.keys() <= allowed:
                for key in [
                    key for key in properties if key not in allowed
                ]:
                    del properties[key]

    def _empty_owner_state(
        self,
        entity_id: int,
        entity_class: str,
    ) -> EntityState:
        return EntityState(
            entity_id=entity_id,
            entity_class=entity_class,
            position=None,
            velocity=None,
            track_age=self._timestep,
            track_state=TrackState.CONFIRMED,
            properties={},
        )

    def _materialize_declared_entity_states(
        self,
    ) -> Dict[int, EntityState]:
        registrations = {
            record.spec.declaration_key: record.entity_id
            for record in self._declared_entities
        }
        states = dict(self._declared_entity_states(registrations))
        expected = set(registrations)
        actual = set(states)
        if actual != expected:
            raise ValueError(
                "Declared entity state reconciliation mismatch: "
                "missing={} unexpected={}".format(
                    sorted(expected - actual), sorted(actual - expected)
                )
            )

        result: Dict[int, EntityState] = {}
        record_by_key = {
            record.spec.declaration_key: record
            for record in self._declared_entities
        }
        for declaration_key, state in states.items():
            if not isinstance(state, EntityState):
                raise TypeError(
                    "_declared_entity_states() value for {!r} must be "
                    "EntityState".format(declaration_key)
                )
            record = record_by_key[declaration_key]
            if state.entity_id != record.entity_id:
                raise ValueError(
                    "Declared entity {!r} returned id {}, expected {}".format(
                        declaration_key, state.entity_id, record.entity_id
                    )
                )
            if state.entity_class != record.spec.entity_class:
                raise ValueError(
                    "Declared entity {!r} returned class {!r}, expected "
                    "{!r}".format(
                        declaration_key,
                        state.entity_class,
                        record.spec.entity_class,
                    )
                )
            result[state.entity_id] = state
        return result

    _MATERIALIZER_METHODS = {
        "observation_time": "_materialize_observation_time",
        "simulation_energy": "_materialize_simulation_energy",
        "joint_signal": "_materialize_joint_signal",
        "actuator_activation": "_materialize_actuator_activation",
        "actuator_state": "_materialize_actuator_state",
        "actuator_moment": "_materialize_actuator_moment",
        "equality_constraint_state": "_materialize_equality_constraint_state",
        "body_cartesian_state": "_materialize_body_cartesian_state",
        "body_pose": "_materialize_body_pose",
        "body_inertial_pose": "_materialize_body_inertial_pose",
        "joint_geometry": "_materialize_joint_geometry",
        "subtree_state": "_materialize_subtree_state",
        "joint_motion_axis": "_materialize_joint_motion_axis",
        "body_inertia": "_materialize_body_inertia",
        "body_spatial_state": "_materialize_body_spatial_state",
        "body_post_constraint_state":
            "_materialize_body_post_constraint_state",
        "inertia_matrix": "_materialize_inertia_matrix",
        "geom_pose": "_materialize_geom_pose",
        "camera_pose": "_materialize_camera_pose",
        "light_pose": "_materialize_light_pose",
        "site_pose": "_materialize_site_pose",
        "mocap_state": "_materialize_mocap_state",
        "flex_state": "_materialize_flex_state",
        "flex_edge_jacobian": "_materialize_flex_edge_jacobian",
        "tendon_state": "_materialize_tendon_state",
        "tendon_jacobian": "_materialize_tendon_jacobian",
        "sensor_state": "_materialize_sensor_state",
        "contact_state": "_materialize_contact_state",
    }

    # Fields whose materializer passes ``source=`` at every emission and
    # whose owner population is fixed for the episode, so their values
    # can be gathered straight out of one MjData array in steady state.
    # Converting a field means passing ``source=`` and listing it here.
    _VECTOR_CANDIDATES: FrozenSet[str] = frozenset(
        {
            # Row gather: flat index is row * width + component.
            "geom_xpos", "geom_xmat",
            "site_xpos", "site_xmat",
            "cinert", "crb",
            "cacc", "cfrc_ext", "cfrc_int",
            "xpos", "xquat", "xmat",
            "xipos", "ximat",
            "cvel",
            "subtree_com", "subtree_linvel", "subtree_angmom",
            "xfrc_applied",
            "xanchor", "xaxis",
            "cam_xpos", "cam_xmat",
            "mocap_pos", "mocap_quat",
            "light_xpos", "light_xdir",
            # Address-slice gather: flat index is start + offset, where
            # start comes from a model address array.
            "qpos", "qvel", "qacc", "qacc_smooth",
            "qfrc_actuator", "qfrc_bias", "qfrc_constraint",
            "qfrc_damper", "qfrc_fluid", "qfrc_gravcomp",
            "qfrc_passive", "qfrc_smooth", "qfrc_spring",
            "qfrc_applied",
            "cdof", "cdof_dot",
            "sensordata",
            "act", "act_dot",
            "flexvert_xpos", "flexedge_length", "flexedge_velocity",
            # Scalar per owner: the flat index is the owner id.
            "ctrl", "actuator_force", "actuator_length",
            "actuator_velocity",
            "ten_length", "ten_velocity",
            "energy",
        }
    )

    def _materialize_curated_fields(
        self,
        perceived: Dict[int, EntityState],
    ) -> None:
        """Dispatch every included field through its declared materializer.

        Per-field steady state (issue #96): the first full pass over a
        field builds its metadata, provenance strings, and report entry,
        then the field enters ``_steady_fields`` and later steps run its
        materializer in value-only mode. A steady run that registers a
        NEW sink (a contact pair materializing mid-episode) drops the
        field back to the full path, so its report entry catches up on
        the next observation — the report is at most one step stale for
        dynamic-population fields, never wrong about applicability.
        ``_materialized_property_meta`` accumulates for the episode: a
        property's metadata means "emitted at least once this episode",
        which is the schema contract discovery and spec validation rely
        on (a separated contact pair keeps its keys).
        """
        expected_ids = {
            record.eager_materializer_id
            for record in INCLUDED_FIELDS.values()
        }
        dispatch_ids = set(self._MATERIALIZER_METHODS)
        if dispatch_ids != expected_ids:
            raise RuntimeError(
                "MuJoCo materializer registry mismatch: missing={} "
                "unexpected={}".format(
                    sorted(expected_ids - dispatch_ids),
                    sorted(dispatch_ids - expected_ids),
                )
            )

        first_pass = not self._materialization_steady
        if first_pass:
            self._materialized_property_meta = {}
            self._field_materialization = {}
        statuses = self._field_materialization
        for field_name, contract in INCLUDED_FIELDS.items():
            if field_name in self._scope_skipped_fields:
                continue
            binding_present = hasattr(self._data, field_name)
            if not binding_present:
                if first_pass:
                    statuses[field_name] = MujocoFieldMaterialization(
                        field_name=field_name,
                        binding_present=False,
                        scene_applicable=False,
                        emitted_keys=(),
                        reason=(
                            "live binding does not expose "
                            "MjData.{}".format(field_name)
                        ),
                    )
                continue
            plan = self._gather_plan.get(field_name)
            if plan is not None:
                with self._completion_stage_data(contract):
                    self._run_gather(plan, perceived)
                continue
            method = getattr(
                self, self._MATERIALIZER_METHODS[
                    contract.eager_materializer_id
                ]
            )
            if field_name in self._steady_fields:
                known_sinks = len(self._merged_meta_cache)
                self._emit_value_only = True
                try:
                    self._current_owner_omissions = []
                    with self._completion_stage_data(contract):
                        method(field_name, perceived)
                finally:
                    self._emit_value_only = False
                if len(self._merged_meta_cache) != known_sinks:
                    # A new sink registered mid-episode: rerun the full
                    # bookkeeping path next step so the report entry
                    # reflects the grown population.
                    self._steady_fields.discard(field_name)
                continue
            self._current_owner_omissions = []
            self._scope_dropped = 0
            candidate = field_name in self._VECTOR_CANDIDATES
            # Reset on every full pass, not just candidate ones: an
            # aborted observation must not leave a half-built capture
            # for the next pass to append to.
            self._plan_capture = [] if candidate else None
            if candidate:
                self._plan_position_rows = []
            with self._completion_stage_data(contract):
                applicable, emitted, reason = method(field_name, perceived)
            emitted = [key for key in emitted if key]
            if (
                applicable
                and not emitted
                and not self._current_owner_omissions
                and self._scope_dropped
            ):
                # The scope removed every sink this field wrote on this
                # pass, and it wrote no positions either, so there is
                # nothing left for it to do right now.
                #
                # The capture has to be dropped by hand here, because
                # this is the one full pass that does not end in
                # _install_gather_plan — and every emission after it
                # relies on _plan_capture being None outside a
                # candidate's own pass.
                self._plan_capture = None
                self._plan_position_rows = []
                statuses[field_name] = MujocoFieldMaterialization(
                    field_name=field_name,
                    binding_present=True,
                    scene_applicable=True,
                    emitted_keys=(),
                    reason=(
                        "every key of this field falls outside the "
                        "active extraction scope"
                    ),
                )
                if "dynamic_entity_population" not in contract.output_sink:
                    # Retiring the field for the episode is only sound
                    # when this pass saw every owner it will ever have.
                    # A field that materializes its own entities at
                    # runtime has not: MjData.contact allocates one
                    # entity per body pair as pairs first touch, and
                    # decides each pair's class then, so a scope naming
                    # only `robot_contact` sees nothing to emit while
                    # the scene holds plain contacts and would lose the
                    # robot pair that arrives later — silently, since a
                    # spec bound to it simply never evaluates. Such a
                    # field re-runs every step; the scope still projects
                    # its keys, it just never stops looking for owners.
                    self._scope_skipped_fields.add(field_name)
                continue
            if (
                applicable
                and not emitted
                and not self._current_owner_omissions
            ):
                raise RuntimeError(
                    "Applicable MuJoCo field {!r} emitted no observation "
                    "sink".format(field_name)
                )
            omissions = tuple(self._current_owner_omissions)
            if omissions and reason is None:
                reason = "some owners were not materialized: {}".format(
                    "; ".join(
                        "{} ({})".format(owner, detail)
                        for owner, detail in omissions
                    )
                )
            refusal = None
            if candidate:
                # The plan trial re-reads the field's source array; a
                # completion-derived field must be trialled on the same
                # completed stage its materializer read.
                with self._completion_stage_data(contract):
                    refusal = self._install_gather_plan(
                        field_name,
                        applicable,
                        omissions,
                        len(emitted),
                        perceived,
                    )
            statuses[field_name] = MujocoFieldMaterialization(
                field_name=field_name,
                binding_present=True,
                scene_applicable=applicable,
                emitted_keys=tuple(emitted),
                reason=reason,
                owner_omissions=omissions,
                plan_refusal=refusal,
            )
            if (
                applicable
                and "dynamic_entity_population" not in contract.output_sink
            ):
                self._steady_fields.add(field_name)
        self._materialization_steady = True

    def _install_gather_plan(
        self,
        field_name: str,
        applicable: bool,
        omissions: Tuple[Tuple[str, str], ...],
        emitted_count: int,
        perceived: Dict[int, EntityState],
    ) -> Optional[str]:
        """Turn one field's capture into a gather plan, if it is sound.

        Refuses to plan unless the capture accounts for every sink the
        pass emitted, every value came from a single MjData array, and a
        trial gather reproduces the values and positions the scalar pass
        just wrote.
        The pass must also have omitted no owner, so the sink set cannot
        grow later: a perception gate that hides an owner mid-episode
        only ever removes writes, which ``_run_gather`` already handles,
        but one that hides an owner during this pass would freeze it out
        of the plan for the episode.

        A refusal costs the field its speedup, never its correctness, so
        it is reported rather than raised: the returned string becomes
        the field's ``plan_refusal``, which is the only place an odd
        binding layout becomes visible.

        Under an active extraction scope the capture still holds every
        key the pass visited, including the ones the scope dropped —
        that is what keeps the single-source-array check meaningful when
        the scope removed all of them. The plan itself is then built over
        the surviving keys alone, so a scoped-out key costs nothing per
        step.
        """
        capture, self._plan_capture = self._plan_capture, None
        position_rows = self._plan_position_rows
        self._plan_position_rows = []
        if not applicable:
            return None
        if omissions:
            return (
                "no gather plan: an owner was omitted from the first "
                "full pass"
            )
        if capture is None:
            return "no gather plan: a value was emitted with no source"
        visited = emitted_count + self._scope_dropped
        if len(capture) + len(position_rows) != visited:
            return (
                "no gather plan: the capture covers {} of {} emitted "
                "sinks".format(
                    len(capture) + len(position_rows), visited
                )
            )
        attrs = {source[0] for _entity_id, _key, source in capture}
        if len(attrs) != 1:
            return (
                "no gather plan: values came from {} source arrays "
                "({})".format(len(attrs), ", ".join(sorted(attrs)))
            )
        scope = self._extraction_scope
        if scope is not None:
            # An owner with no resolvable class cannot reach here: the
            # scope gate leaves it to the registering path, which records
            # an omission, and omissions have already refused above.
            capture = [
                item
                for item in capture
                if item[1]
                in scope.get(
                    self._observed_class(perceived, item[0]) or "",
                    _NO_KEYS,
                )
            ]
        targets = [(entity_id, key) for entity_id, key, _ in capture]
        # Uniqueness here replaces the per-step duplicate-sink guard the
        # scalar path runs in _emit_property: the property dicts are
        # built fresh every observation, so a plan whose targets are
        # distinct can never collide with itself.
        if len(set(targets)) != len(targets):
            return "no gather plan: the same sink was emitted twice"
        plan = _FieldGatherPlan(
            attr=attrs.pop(),
            flat_indices=np.fromiter(
                (source[1] for _entity_id, _key, source in capture),
                dtype=np.intp,
                count=len(capture),
            ),
            targets=targets,
            position_rows=list(position_rows),
        )
        arr = np.asarray(
            getattr(self._data, plan.attr), dtype=float
        ).reshape(-1)
        if plan.flat_indices.size:
            if plan.flat_indices.max() >= arr.size:
                return (
                    "no gather plan: a flat index falls outside "
                    "MjData.{}".format(plan.attr)
                )
        elif not plan.position_rows:
            # Unreachable without a scope: an empty capture yields no
            # source array and the check above has already refused.
            return "no gather plan: nothing left to gather"
        written = np.fromiter(
            (
                perceived[entity_id].properties[key]
                for entity_id, key in targets
            ),
            dtype=float,
            count=len(targets),
        )
        if not np.array_equal(arr[plan.flat_indices], written):
            return (
                "no gather plan: gathering MjData.{} did not reproduce "
                "the emitted values".format(plan.attr)
            )
        if plan.position_rows:
            # The position sinks need their own trial: they are gathered
            # by row rather than by flat index, so the check above says
            # nothing about them -- and a field whose every property key
            # is scoped out has nothing else left to validate.
            rows = getattr(self._data, plan.attr)
            if any(
                not 0 <= row_id < len(rows)
                for _entity_id, row_id in plan.position_rows
            ):
                # Refused by range rather than by the comparison below.
                # A row past the end raises out of the gather, and a
                # negative one is worse than that: Python indexes it from
                # the end of the array, so it quietly reads some other
                # owner's position and is caught only when the two differ.
                return (
                    "no gather plan: a position row falls outside "
                    "MjData.{}".format(plan.attr)
                )
            gathered = np.asarray(
                [rows[row_id] for _entity_id, row_id in plan.position_rows],
                dtype=float,
            )
            written_positions = np.asarray(
                [
                    perceived[entity_id].position
                    for entity_id, _row_id in plan.position_rows
                ],
                dtype=float,
            )
            if not np.array_equal(gathered, written_positions):
                return (
                    "no gather plan: gathering MjData.{} did not reproduce "
                    "the emitted positions".format(plan.attr)
                )
            # Two owners at the same point make the comparison above
            # blind: it reproduces the emitted positions whichever of the
            # two rows each owner was recorded against, and the wrong
            # pairing only shows once the scene separates them, with the
            # plan already installed. Coincident owners are ordinary --
            # a collision geom and its visual twin, sites sharing an
            # anchor, unused mocap bodies parked together -- so refusing
            # on that alone would cost real scenes their largest gathers.
            #
            # The owner's own property indices settle it wherever it has
            # any: those are gathered from row * width + component and
            # were value-checked above, so the row they imply is the row
            # this field read that owner from, whatever the values look
            # like.
            width = _plan_row_width(rows)
            anchors: Dict[int, Set[int]] = {}
            if width:
                for (entity_id, _key), flat in zip(
                    plan.targets, plan.flat_indices.tolist()
                ):
                    anchors.setdefault(entity_id, set()).add(flat // width)
            anchorless = []
            for index, (entity_id, row_id) in enumerate(plan.position_rows):
                implied = anchors.get(entity_id)
                if not implied:
                    anchorless.append(index)
                elif row_id not in implied:
                    return (
                        "no gather plan: a position row of MjData.{} is "
                        "not a row the same owner's properties gather "
                        "from".format(plan.attr)
                    )
            if anchorless:
                # No property anchor for these: a scope dropped every key
                # the owner had, leaving the position sink alone -- the
                # shape a scoped monitor run puts geom_xpos and site_xpos
                # in, where the coincident owners above are ordinary and
                # refusing costs the scene its two largest gathers.
                #
                # The reset's own owner records answer it without reading
                # the capture that is under validation: they are what
                # each position write is driven from, so the row a record
                # holds for an entity is the row that write read.
                recorded = self._position_row_records(field_name)
                unanchored = []
                for index in anchorless:
                    entity_id, row_id = plan.position_rows[index]
                    owned_row = recorded.get(entity_id)
                    if owned_row is None:
                        unanchored.append(index)
                    elif owned_row != row_id:
                        return (
                            "no gather plan: a position row of MjData.{} "
                            "is not the row this field's owner record "
                            "holds for that entity".format(plan.attr)
                        )
                if unanchored:
                    # Nothing left that can tell one owner's row from
                    # another's, so equal vectors have to refuse -- the
                    # conservative answer, and only for the rows that
                    # have no better one.
                    vectors = [tuple(row) for row in gathered.tolist()]
                    counts = Counter(vectors)
                    if any(counts[vectors[i]] > 1 for i in unanchored):
                        return (
                            "no gather plan: two position rows of "
                            "MjData.{} hold the same vector".format(
                                plan.attr
                            )
                        )
        self._gather_plan[field_name] = plan
        return None

    def _position_row_records(
        self,
        field_name: str,
    ) -> Dict[int, int]:
        """Entity id to model row, from the reset's registration records.

        The structural anchor for a position row whose owner has no
        surviving property key to place it. These are the records each
        position-writing materializer iterates -- the row here is the row
        it read the owner's position out of -- and they are built once at
        reset, so consulting them says nothing about the capture whose
        soundness is the question.

        A field with no record table here anchors nothing and its rows
        fall through to the equal-vector refusal, which is the answer for
        a row nothing at all can place.
        """
        if field_name == "xpos":
            return {
                record.entity_id: record.body_id
                for record in self._bodies
            }
        if field_name == "geom_xpos":
            return {
                record.entity_id: record.source_id
                for record in self._raw_geoms
            }
        if field_name == "site_xpos":
            return {
                record.entity_id: record.source_id
                for record in self._site_owner_records()
            }
        if field_name == "cam_xpos":
            return {
                record.entity_id: record.source_id
                for record in self._raw_cameras
            }
        if field_name == "mocap_pos":
            return {
                record.entity_id: record.mocap_id
                for record in self._mocap_owners
            }
        return {}

    def _run_gather(
        self,
        plan: _FieldGatherPlan,
        perceived: Dict[int, EntityState],
    ) -> None:
        """Steady-state replacement for one field's materializer pass."""
        arr = np.asarray(
            getattr(self._data, plan.attr), dtype=float
        ).reshape(-1)
        values = arr[plan.flat_indices]
        if not np.isfinite(values).all():
            # The scalar path drops a hidden owner before it ever reads
            # the value, so only a non-finite value that would actually
            # be written is an error — and the first such one, in
            # emission order, is the one it would have reported.
            for bad in np.flatnonzero(~np.isfinite(values)):
                entity_id, key = plan.targets[int(bad)]
                if entity_id in perceived:
                    raise RuntimeError(
                        "MuJoCo property {!r} for entity {} is "
                        "non-finite".format(key, entity_id)
                    )
        # tolist() converts in one C loop and yields plain floats, which
        # is the contract every consumer of properties relies on.
        for (entity_id, key), value in zip(plan.targets, values.tolist()):
            state = perceived.get(entity_id)
            if state is not None:
                state.properties[key] = value
        if plan.position_rows:
            # Every position sink copies a row of the field's own source
            # array (xpos, geom_xpos, site_xpos, cam_xpos, mocap_pos), so
            # the plan's attribute is the right one to re-read -- and it
            # is the wrapper alias (body_xpos) where the pass resolved one.
            rows = getattr(self._data, plan.attr)
            for entity_id, row_id in plan.position_rows:
                state = perceived.get(entity_id)
                if state is not None:
                    state.position = np.array(
                        rows[row_id], dtype=float
                    ).reshape(-1)

    def _omit_materialization_owner(
        self,
        owner: str,
        reason: str,
    ) -> None:
        omission = (owner, reason)
        if omission not in self._current_owner_omissions:
            self._current_owner_omissions.append(omission)

    def _entity_extra_tags(
        self,
        entity_id: int,
    ) -> Tuple[str, ...]:
        """Structural tags appended to every property key of one entity.

        Subclass hook for per-entity classifications that apply to the
        whole key set (RoboCasa attaches namespaced task-role tags,
        e.g. ``role:distractor``). The base adapter has none.
        """
        return ()

    def _contact_pair_tags(
        self,
        body_pair: Tuple[int, int],
    ) -> Tuple[str, ...]:
        """Structural tags for one dynamic contact pair (subclass hook).

        Consulted exactly once, when the unordered body pair is first
        materialized as an entity; the result is frozen on the pair
        record and rides every contact property key of that entity for
        the rest of the episode. Simulator policy decides the content
        (RoboCasa maps participant body IDs through its declaration-
        backed body-ownership table to ``role:*`` tags). The base
        adapter has no such policy and returns no tags.
        """
        return ()

    def _replay_value_only_sinks(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
        values: List[float],
    ) -> bool:
        """Emit ``values`` against ``field_name``'s recorded sink table.

        Returns ``False`` when no table exists yet, which leaves the
        caller on its ordinary path. A table is only recorded by a full
        pass that omitted no owner, so every pair it holds is already in
        ``_merged_meta_cache`` and needs no metadata argument. ``values``
        must be in the same order the recording pass emitted.

        The length check has no reachable failure: the table is cleared
        on reset and the model cannot change without one. It is there
        because the alternative failure mode is silent — ``zip`` would
        truncate to the shorter side and pair values with keys that do
        not belong to them.
        """
        if not self._emit_value_only:
            return False
        sinks = self._value_only_sinks.get(field_name)
        if sinks is None:
            return False
        if len(sinks) != len(values):
            raise RuntimeError(
                "MuJoCo field {!r} produced {} values for a sink table "
                "of {}".format(field_name, len(values), len(sinks))
            )
        emit = self._emit_property
        for (entity_id, key), value in zip(sinks, values):
            emit(perceived, entity_id, key, value, None)
        return True

    def _emit_property(
        self,
        perceived: Dict[int, EntityState],
        entity_id: int,
        key: str,
        value: Any,
        meta: Optional[PropertyMeta],
        source: Optional[Tuple[str, int]] = None,
    ) -> str:
        """Write one property value, registering its metadata if new.

        ``source`` names where the value came from as
        ``(MjData attribute, flat index)``. It is recorded only while a
        gather plan is being captured; a field that emits even one value
        without it cannot be planned and stays on this path forever.

        ``meta`` may be ``None`` only from a value-only rerun driven by a
        ``_value_only_sinks`` table, where every sink is already
        registered and the metadata argument would be built and thrown
        away. Reaching the registering path without it is a bug, and
        raises rather than registering a property with no schema.
        """
        if self._extraction_scope is not None:
            entity_class = self._observed_class(perceived, entity_id)
            if (
                entity_class is not None
                and key
                not in self._extraction_scope.get(entity_class, _NO_KEYS)
            ):
                # Out of scope: no value, no metadata, no report entry,
                # and — deliberately — no owner omission. A scoped-out
                # key is not a failure to materialize something, so it
                # must not block this field from being planned. The
                # capture still records where the value would have come
                # from, which is what lets a field resolve its single
                # source array even when the scope removed every key.
                #
                # The counter is written on every path through here,
                # including a steady value-only rerun, but it is only
                # ever read inside the full pass that zeroed it — the
                # field loop resets it immediately before calling a
                # materializer and _install_gather_plan reads it
                # immediately after. A count made anywhere else is
                # therefore dead rather than wrong, and is left in
                # rather than branched around so the gate stays one
                # test on the hot path.
                self._scope_dropped += 1
                if self._plan_capture is not None:
                    if source is None:
                        self._plan_capture = None
                    else:
                        self._plan_capture.append((entity_id, key, source))
                return ""

        meta_key = (entity_id, key)
        if self._emit_value_only:
            # Value-only fast path for sinks a full pass already
            # registered: metadata and provenance were settled then;
            # per-step work is the fresh value, its finiteness, and the
            # within-step duplicate-sink guard. A NEW (entity, key) — a
            # contact pair materializing mid-episode — misses the cache
            # and takes the full registering path below.
            if meta_key in self._merged_meta_cache:
                if entity_id not in perceived:
                    return ""
                scalar = float(value)
                if not np.isfinite(scalar):
                    raise RuntimeError(
                        "MuJoCo property {!r} for entity {} is "
                        "non-finite".format(key, entity_id)
                    )
                properties = perceived[entity_id].properties
                existing = properties.get(key)
                if existing is not None and existing != scalar:
                    raise ValueError(
                        "Materialized property collision for entity {} "
                        "key {!r}: {} != {}".format(
                            entity_id, key, existing, scalar
                        )
                    )
                properties[key] = scalar
                return ""

        if meta is None:
            raise RuntimeError(
                "MuJoCo property {!r} for entity {} reached the "
                "registering path without metadata: a value-only rerun "
                "emitted a sink its full pass never registered".format(
                    key, entity_id
                )
            )
        merged = self._merged_meta_cache.get(meta_key)
        if merged is None:
            extra_tags = self._entity_extra_tags(entity_id)
            if extra_tags:
                meta = meta._replace(
                    tags=tuple(meta.tags) + tuple(
                        tag for tag in extra_tags if tag not in meta.tags
                    )
                )
        else:
            meta = merged
        if entity_id not in perceived:
            self._omit_materialization_owner(
                "entity[{}]".format(entity_id),
                "registered scene owner was filtered from this observation",
            )
            return ""
        scalar = float(value)
        if not np.isfinite(scalar):
            raise RuntimeError(
                "MuJoCo property {!r} for entity {} is non-finite".format(
                    key, entity_id
                )
            )
        properties = perceived[entity_id].properties
        if key in properties and properties[key] != scalar:
            raise ValueError(
                "Materialized property collision for entity {} key {!r}: "
                "{} != {}".format(
                    entity_id, key, properties[key], scalar
                )
            )
        properties[key] = scalar
        entity_meta = self._materialized_property_meta.setdefault(
            entity_id, {}
        )
        existing = entity_meta.get(key)
        if existing is None and self._prospective_metadata_keys is not None:
            self._prospective_metadata_keys.append(meta_key)
        # Identity fast path: the memoized meta is the same object every
        # step, so the field-by-field namedtuple walk only runs on a
        # genuine collision candidate.
        if (existing is not None and existing is not meta
                and existing != meta):
            raise ValueError(
                "Materialized metadata collision for entity {} key "
                "{!r}".format(entity_id, key)
            )
        entity_meta[key] = meta
        # Cached only after full registration: an entity filtered out of
        # this observation (perception gate) must not leave a cache
        # entry, or its later appearance would take the value-only path
        # with no metadata registered (review catch on 4789291's
        # successor).
        self._merged_meta_cache[meta_key] = meta
        if self._plan_capture is not None:
            if source is None:
                self._plan_capture = None
            else:
                self._plan_capture.append((entity_id, key, source))
        return "entity[{}].properties.{}".format(entity_id, key)

    @staticmethod
    def _not_applicable(owner: str) -> Tuple[bool, List[str], str]:
        return (
            False,
            [],
            "current reset model has no resolvable {}".format(owner),
        )

    def _body_owner_pairs(
        self,
    ) -> List[Tuple[int, int]]:
        # Ownership is fixed after reset-time registration, but was
        # rebuilt and partition-checked (an O(nbody log nbody) sort)
        # ~21 times per step across the materializers (issue #92).
        # Callers iterate the result; nothing mutates it.
        if self._body_owner_pairs_cache is not None:
            return self._body_owner_pairs_cache
        owners = [
            (body.entity_id, body_id)
            for body in self._bodies
            for body_id in body.pose_body_ids
        ]
        owners.extend(
            (owner.entity_id, body_id)
            for owner in self._validated_additional_body_owners()
            for body_id in owner.body_ids
        )
        owners.extend(
            (record.entity_id, record.source_id)
            for record in self._raw_bodies
        )
        body_ids = [body_id for _entity_id, body_id in owners]
        expected = list(
            range(int(getattr(self._model, "nbody", 0) or 0))
        )
        if sorted(body_ids) != expected:
            raise ValueError(
                "MuJoCo body-row ownership is not an exact partition: "
                "owned={} expected={}".format(sorted(body_ids), expected)
            )
        self._body_owner_pairs_cache = owners
        return owners

    def _validated_additional_body_owners(
        self,
    ) -> Tuple[MujocoBodyOwner, ...]:
        owners = tuple(self._additional_body_owners())
        if not all(isinstance(owner, MujocoBodyOwner) for owner in owners):
            raise TypeError(
                "_additional_body_owners() must return MujocoBodyOwner "
                "objects"
            )
        registered_entity_ids = set(self.name_to_id.values())
        body_count = int(getattr(self._model, "nbody", 0) or 0)
        for owner in owners:
            if owner.entity_id not in registered_entity_ids:
                raise ValueError(
                    "Additional body owner {!r} uses unregistered entity "
                    "id {}".format(owner.owner_name, owner.entity_id)
                )
            for body_id in owner.body_ids:
                if body_id >= body_count:
                    raise ValueError(
                        "Additional body owner {!r} body id {} is outside "
                        "model.nbody={}".format(
                            owner.owner_name, body_id, body_count
                        )
                    )
        return owners

    def _cframe_tags(
        self,
        body_id: int,
    ) -> Tuple[str, ...]:
        root_ids = getattr(self._model, "body_rootid", None)
        if root_ids is None:
            return ()
        root_id = int(root_ids[body_id])
        return (
            "cframe_origin:subtree_com_{}".format(
                self._body_token(root_id)
            ),
        )

    def _joint_owner_triples(
        self,
    ) -> List[Tuple[int, int, int]]:
        # Same reset-invariance as _body_owner_pairs: this ran once per
        # joint-signal field (14 fields) per step (issue #92).
        if self._joint_owner_triples_cache is not None:
            return self._joint_owner_triples_cache
        owners = [
            (body.body_id, body.entity_id, joint_id)
            for body in self._bodies
            for joint_id in body.joint_ids
        ]
        additional = tuple(self._additional_joint_owners())
        if not all(
            isinstance(owner, MujocoJointOwner)
            for owner in additional
        ):
            raise TypeError(
                "_additional_joint_owners() must return "
                "MujocoJointOwner objects"
            )

        registered_entity_ids = set(self.name_to_id.values())
        claimed = {
            (entity_id, joint_id)
            for _body_id, entity_id, joint_id in owners
        }
        body_count = int(getattr(self._model, "nbody", 0) or 0)
        joint_count = int(getattr(self._model, "njnt", 0) or 0)
        for owner in additional:
            if owner.entity_id not in registered_entity_ids:
                raise ValueError(
                    "Additional joint owner {!r} uses unregistered entity "
                    "id {}".format(owner.owner_name, owner.entity_id)
                )
            if owner.owner_body_id >= body_count:
                raise ValueError(
                    "Additional joint owner {!r} body id {} is outside "
                    "model.nbody={}".format(
                        owner.owner_name,
                        owner.owner_body_id,
                        body_count,
                    )
                )
            for joint_id in owner.joint_ids:
                if joint_id >= joint_count:
                    raise ValueError(
                        "Additional joint owner {!r} joint id {} is outside "
                        "model.njnt={}".format(
                            owner.owner_name,
                            joint_id,
                            joint_count,
                        )
                    )
                claim = (owner.entity_id, joint_id)
                if claim in claimed:
                    raise ValueError(
                        "Additional joint owner {!r} duplicates entity/joint "
                        "claim {}".format(owner.owner_name, claim)
                    )
                claimed.add(claim)
                owners.append(
                    (owner.owner_body_id, owner.entity_id, joint_id)
                )
        claimed_joint_ids = {
            joint_id for _body_id, _entity_id, joint_id in owners
        }
        body_owner_by_id = {
            body_id: entity_id
            for entity_id, body_id in self._body_owner_pairs()
        }
        for joint_id in range(joint_count):
            if joint_id in claimed_joint_ids:
                continue
            body_id = int(self._model.jnt_bodyid[joint_id])
            entity_id = body_owner_by_id.get(body_id)
            if entity_id is None:
                raise ValueError(
                    "MuJoCo joint row {} on body {} has no owner".format(
                        joint_id, body_id
                    )
                )
            owners.append((body_id, entity_id, joint_id))
            claimed_joint_ids.add(joint_id)
        if claimed_joint_ids != set(range(joint_count)):
            raise ValueError(
                "MuJoCo joint-row ownership is incomplete"
            )
        self._joint_owner_triples_cache = owners
        return owners

    def _xaxis_owner_triples(
        self,
    ) -> List[Tuple[int, int, int]]:
        """Return the hinge/slide subset of the complete joint owners.

        MuJoCo supports exactly free, ball, slide, and hinge joint kinds.
        Ball and free joints do not expose one axis, while an unexpected
        binding value is a compatibility failure rather than a silently
        omitted authoring surface.
        """
        owners = []
        for owner in self._joint_owner_triples():
            joint_id = owner[2]
            joint_type = int(self._model.jnt_type[joint_id])
            try:
                applies = _XAXIS_JOINT_TYPE_APPLICABILITY[joint_type]
            except KeyError as exc:
                raise RuntimeError(
                    "Unknown MuJoCo joint type {} while classifying "
                    "xaxis for joint {!r}; expected free (0), ball (1), "
                    "slide (2), or hinge (3)".format(
                        joint_type,
                        self._indexed_source_name("joint", joint_id),
                    )
                ) from exc
            if applies:
                owners.append(owner)
        return owners

    def _materialize_observation_time(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        del field_name, perceived
        return True, ["observation.time"], None

    def _materialize_simulation_energy(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        del field_name
        if self._simulation_entity_id is None:
            return self._not_applicable("simulation-energy owner")
        values = np.asarray(self._data.energy, dtype=float).reshape(-1)
        if values.size != 2:
            raise RuntimeError(
                "MjData.energy must have two components, found {}".format(
                    values.size
                )
            )
        emitted = [
            self._emit_property(
                perceived,
                self._simulation_entity_id,
                "energy_{}".format(name),
                values[index],
                PropertyMeta(group="simulation energy", note="J"),
                # values is already the flattened array, and _run_gather
                # flattens the same attribute the same way, so the
                # emitted position is the flat index.
                source=("energy", index),
            )
            for index, name in enumerate(("potential", "kinetic"))
        ]
        return True, emitted, None

    def _materialize_joint_signal(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        owners = self._joint_owner_triples()
        if not owners:
            return self._not_applicable("joint owner")
        emitted: List[str] = []
        address_name, total_name = _SIGNAL_ADDRESSES[field_name]
        values = getattr(self._data, field_name)
        # values[start:stop] is a view into one qpos/qvel-family array,
        # so the emitted element's flat index is the joint's address plus
        # its offset -- which holds exactly when the array's row stride is
        # one element.
        flat_source = _plan_row_width(values) == 1
        for owner_body_id, entity_id, joint_id in owners:
            token = self._joint_property_token(joint_id)
            start, stop = self._joint_slice(
                joint_id, address_name, total_name
            )
            keys = self._signal_keys(field_name, token, stop - start)
            joint_type = int(self._model.jnt_type[joint_id])
            own_body = (
                int(self._model.jnt_bodyid[joint_id]) == owner_body_id
            )
            tags = (
                tuple(self._joint_subsystem_tags(joint_id))
                + self._source_identity_tags("joint", joint_id)
            )
            for index, (key, value) in enumerate(
                zip(keys, values[start:stop])
            ):
                meta = self._joint_key_meta(
                    field_name, joint_type, index, own_body
                )
                if tags:
                    meta = meta._replace(tags=tags)
                emitted.append(
                    self._emit_property(
                        perceived,
                        entity_id,
                        key,
                        value,
                        meta,
                        source=(
                            (field_name, start + index)
                            if flat_source
                            else None
                        ),
                    )
                )
        return True, emitted, None

    def _actuator_entity_id(
        self,
        record: _ActuatorRecord,
    ) -> int:
        entity_id = record.owner_entity_id
        if entity_id is None:
            entity_id = self._orphan_actuator_entity_id
        if entity_id is None:
            raise RuntimeError(
                "Actuator {!r} has no registered monitor owner".format(
                    record.name
                )
            )
        return entity_id

    def _activation_meta(
        self,
        record: _ActuatorRecord,
        field_name: str,
    ) -> PropertyMeta:
        unit = record.ctrl_unit_note
        if unit is None:
            return PropertyMeta(
                group="actuator {} ({}; opaque)".format(
                    "activation" if field_name == "act" else "activation rate",
                    record.name_token,
                ),
                note=None,
                tags=(
                    self._source_identity_tags(
                        "actuator", record.actuator_id
                    )
                    + self._actuator_transmission_tags(record.actuator_id)
                    + ("semantics:opaque", "unit:unresolved")
                ),
            )
        dyn_type = int(
            self._model.actuator_dyntype[record.actuator_id]
        )
        if dyn_type == _MJ_DYN_INTEGRATOR:
            integrated = {
                "m/s": "m",
                "rad/s": "rad",
            }.get(unit)
            if integrated is None:
                raise RuntimeError(
                    "Integrator actuator {!r} control unit {!r} cannot be "
                    "resolved to an activation unit".format(
                        record.name, unit
                    )
                )
            note = unit if field_name == "act_dot" else integrated
        else:
            note = unit if field_name == "act" else "{}/s".format(unit)
        return PropertyMeta(
            group="actuator activation ({})".format(note),
            note=note,
            tags=self._source_identity_tags(
                "actuator", record.actuator_id
            ) + self._actuator_transmission_tags(record.actuator_id),
        )

    def _materialize_actuator_activation(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        actadr = getattr(self._model, "actuator_actadr", None)
        actnum = getattr(self._model, "actuator_actnum", None)
        if actadr is None or actnum is None:
            return (
                False,
                [],
                "binding exposes MjData.{} but model has no "
                "actuator_actadr/actuator_actnum capability".format(
                    field_name
                ),
            )
        owners = [
            record
            for record in self._actuators
            if int(actnum[record.actuator_id]) > 0
        ]
        if not owners:
            return self._not_applicable("stateful actuator")
        emitted: List[str] = []
        capable_owner = False
        data_values = getattr(self._data, field_name)
        flat_source = _plan_row_width(data_values) == 1
        for record in owners:
            start = int(actadr[record.actuator_id])
            count = int(actnum[record.actuator_id])
            if start < 0:
                self._omit_materialization_owner(
                    record.name,
                    "invalid activation address {}".format(start),
                )
                continue
            keys = (
                ["{}_{}".format(field_name, record.name_token)]
                if count == 1
                else [
                    "{}_{}_{}".format(
                        field_name, record.name_token, index
                    )
                    for index in range(count)
                ]
            )
            meta = self._activation_meta(record, field_name)
            capable_owner = True
            for offset, (key, value) in enumerate(
                zip(keys, data_values[start:start + count])
            ):
                emitted.append(
                    self._emit_property(
                        perceived,
                        self._actuator_entity_id(record),
                        key,
                        value,
                        meta,
                        source=(
                            (field_name, start + offset)
                            if flat_source
                            else None
                        ),
                    )
                )
        return True, emitted, None

    def _materialize_actuator_state(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        if not self._actuators:
            return self._not_applicable("actuator")
        emitted: List[str] = []
        values = getattr(self._data, field_name)
        flat_source = _plan_row_width(values) == 1
        for record in self._actuators:
            matches = [
                item
                for item in self._actuator_key_fields(record)
                if item[1] == field_name
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "No unique actuator materializer for {} on {!r}".format(
                        field_name, record.name
                    )
                )
            key, _, group, note = matches[0]
            unit = self._actuator_field_unit(record, field_name)
            emitted.append(
                self._emit_property(
                    perceived,
                    self._actuator_entity_id(record),
                    key,
                    values[record.actuator_id],
                    PropertyMeta(
                        group=group,
                        note=note,
                        tags=(
                            self._source_identity_tags(
                                "actuator", record.actuator_id
                            )
                            + self._actuator_transmission_tags(
                                record.actuator_id
                            )
                            + (
                                ()
                                if unit is not None
                                else (
                                    "semantics:opaque",
                                    "unit:unresolved",
                                )
                            )
                        ),
                    ),
                    source=(
                        (field_name, record.actuator_id)
                        if flat_source
                        else None
                    ),
                )
            )
        return True, emitted, None

    def _dof_descriptors(
        self,
    ) -> List[Tuple[str, str, int, int]]:
        """Return ``(token, motion kind, joint id, local index)`` per DOF."""
        descriptors: List[Tuple[str, str, int, int]] = []
        joint_count = int(getattr(self._model, "njnt", 0) or 0)
        for joint_id in range(joint_count):
            start, stop = self._joint_slice(
                joint_id, "jnt_dofadr", "nv"
            )
            count = stop - start
            joint_type = int(self._model.jnt_type[joint_id])
            joint_token = self._joint_property_token(joint_id)
            keys = self._signal_keys("qvel", joint_token, count)
            for local_index, key in enumerate(keys):
                if joint_type == _MJ_JNT_SLIDE:
                    motion_kind = "linear"
                elif joint_type in (_MJ_JNT_HINGE, _MJ_JNT_BALL):
                    motion_kind = "rotational"
                elif joint_type == _MJ_JNT_FREE:
                    motion_kind = (
                        "linear" if local_index < 3 else "rotational"
                    )
                else:
                    raise RuntimeError(
                        "Unknown MuJoCo joint type {}".format(joint_type)
                    )
                descriptors.append(
                    (
                        key[len("qvel_") :],
                        motion_kind,
                        joint_id,
                        local_index,
                    )
                )
        expected = int(getattr(self._model, "nv", 0) or 0)
        if len(descriptors) != expected:
            raise RuntimeError(
                "Generalized-DOF descriptor count {} != model.nv {}"
                .format(len(descriptors), expected)
            )
        return descriptors

    def _materialize_actuator_moment(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        if not self._actuators:
            return self._not_applicable("actuator")
        dofs = self._dof_descriptors()
        actuator_count = len(self._actuators)
        dof_count = len(dofs)
        raw_values = np.asarray(
            self._data.actuator_moment, dtype=float
        )
        if raw_values.shape == (actuator_count, dof_count):
            dense = raw_values
        else:
            required = (
                "moment_rownnz",
                "moment_rowadr",
                "moment_colind",
            )
            if not all(hasattr(self._data, name) for name in required):
                raise RuntimeError(
                    "MjData.actuator_moment shape {} is neither dense "
                    "(nu,nv)=({}, {}) nor accompanied by sparse row "
                    "metadata".format(
                        raw_values.shape, actuator_count, dof_count
                    )
                )
            rownnz = np.asarray(self._data.moment_rownnz, dtype=int)
            rowadr = np.asarray(self._data.moment_rowadr, dtype=int)
            colind = np.asarray(self._data.moment_colind, dtype=int)
            sparse_values = raw_values.reshape(-1)
            dense = np.zeros(
                (actuator_count, dof_count), dtype=float
            )
            for actuator_id in range(actuator_count):
                start = int(rowadr[actuator_id])
                count = int(rownnz[actuator_id])
                for sparse_index in range(start, start + count):
                    dof_id = int(colind[sparse_index])
                    if dof_id < 0 or dof_id >= dof_count:
                        raise RuntimeError(
                            "Actuator moment column {} is outside nv={}"
                            .format(dof_id, dof_count)
                        )
                    dense[actuator_id, dof_id] = sparse_values[
                        sparse_index
                    ]
        # Row-major, which is the actuator-then-DOF emission order below:
        # _actuators holds one record per actuator id, appended in id
        # order, so a record's position in the list is its dense row.
        # The axis order counts here — dense is (nu, nv), not square.
        if self._replay_value_only_sinks(
            field_name, perceived, dense.reshape(-1).tolist()
        ):
            return True, [], None
        emitted: List[str] = []
        sinks: List[Tuple[int, str]] = []
        capable_owner = False
        for record in self._actuators:
            length_unit = record.length_unit_note
            capable_owner = True
            for dof_id in range(dof_count):
                dof_token, motion_kind, joint_id, local_index = dofs[dof_id]
                coordinate_unit = (
                    "m" if motion_kind == "linear" else "rad"
                )
                note = (
                    None
                    if length_unit is None
                    else (
                        "1"
                        if length_unit == coordinate_unit
                        else "{}/{}".format(
                            length_unit, coordinate_unit
                        )
                    )
                )
                key = "actuator_moment_actuator_{}_dof_{}".format(
                    record.actuator_id, dof_id
                )
                sinks.append((self._actuator_entity_id(record), key))
                emitted.append(
                    self._emit_property(
                        perceived,
                        self._actuator_entity_id(record),
                        key,
                        dense[record.actuator_id, dof_id],
                        PropertyMeta(
                            group=(
                                "actuator transmission moment arm ({})"
                                .format(
                                    note
                                    if note is not None
                                    else "{}; opaque".format(
                                        record.name_token
                                    )
                                )
                            ),
                            note=note,
                            tags=(
                                self._qualified_source_identity_tags(
                                    "actuator",
                                    "actuator",
                                    record.actuator_id,
                                )
                                + self._actuator_transmission_tags(
                                    record.actuator_id
                                )
                                + self._qualified_source_identity_tags(
                                    "joint", "joint", joint_id
                                )
                                + (
                                    "dof_index:{}".format(dof_id),
                                    "joint_local_dof:{}".format(local_index),
                                )
                                + (
                                    ()
                                    if note is not None
                                    else (
                                        "semantics:opaque",
                                        "unit:unresolved",
                                    )
                                )
                            ),
                        ),
                    )
                )
        if not capable_owner or not emitted:
            return (
                False,
                [],
                "current reset model has no actuator transmission moment "
                "with a resolvable unit",
            )
        if not self._current_owner_omissions:
            self._value_only_sinks[field_name] = sinks
        return True, emitted, None

    def _materialize_inertia_matrix(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        if self._simulation_entity_id is None:
            return self._not_applicable("simulation-global owner")
        dofs = self._dof_descriptors()
        if not dofs:
            return self._not_applicable("generalized DOF")
        import mujoco

        dense = np.zeros((len(dofs), len(dofs)), dtype=float)
        version_parts = tuple(
            int(part)
            for part in re.findall(r"\d+", mujoco.__version__)[:2]
        )
        raw_model = _raw_struct(self._model, "_model")
        if version_parts >= (3, 10):
            # MuJoCo 3.10 changed the Python binding from
            # mj_fullM(model, destination, qM) to
            # mj_fullM(model, data, destination).
            mujoco.mj_fullM(
                raw_model,
                _raw_struct(self._data, "_data"),
                dense,
            )
        else:
            mujoco.mj_fullM(
                raw_model,
                dense,
                np.asarray(self._data.qM, dtype=float),
            )
        # Row-major, matching the row-then-column emission order below.
        # qM is symmetric, so a transposed flatten would emit the same
        # values here; the table is what pairs them with keys either way.
        if self._replay_value_only_sinks(
            field_name, perceived, dense.reshape(-1).tolist()
        ):
            return True, [], None
        emitted = []
        sinks: List[Tuple[int, str]] = []
        for row, (
            row_token,
            row_kind,
            row_joint_id,
            row_local_index,
        ) in enumerate(dofs):
            for column, (
                column_token,
                column_kind,
                column_joint_id,
                column_local_index,
            ) in enumerate(dofs):
                if row_kind == column_kind == "linear":
                    note = "kg"
                elif row_kind == column_kind == "rotational":
                    note = "kg·m²"
                else:
                    note = "kg·m"
                key = "inertia_matrix_dof_{}_dof_{}".format(
                    row, column
                )
                sinks.append((self._simulation_entity_id, key))
                emitted.append(
                    self._emit_property(
                        perceived,
                        self._simulation_entity_id,
                        key,
                        dense[row, column],
                        PropertyMeta(
                            group="joint-space inertia",
                            note=note,
                            tags=(
                                "row_dof_index:{}".format(row),
                                "column_dof_index:{}".format(column),
                                "row_joint_index:{}".format(row_joint_id),
                                "column_joint_index:{}".format(
                                    column_joint_id
                                ),
                                "row_joint_local_dof:{}".format(
                                    row_local_index
                                ),
                                "column_joint_local_dof:{}".format(
                                    column_local_index
                                ),
                                "row_dof_name:{}".format(row_token),
                                "column_dof_name:{}".format(column_token),
                            ),
                        ),
                    )
                )
        if not self._current_owner_omissions:
            self._value_only_sinks[field_name] = sinks
        return True, emitted, None

    @staticmethod
    def _actuator_field_unit(
        record: _ActuatorRecord,
        field_name: str,
    ) -> Optional[str]:
        return {
            "ctrl": record.ctrl_unit_note,
            "actuator_force": record.force_unit_note,
            "actuator_length": record.length_unit_note,
            "actuator_velocity": record.velocity_unit_note,
        }[field_name]

    def _materialize_equality_constraint_state(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        del field_name
        if not self._equalities:
            return self._not_applicable("equality constraint")
        import mujoco

        emitted = []
        for record in self._equalities:
            equality_type = int(
                self._model.eq_type[record.source_id]
            )
            equality_type_name = (
                mujoco.mjtEq(equality_type)
                .name.replace("mjEQ_", "", 1)
                .lower()
            )
            object_kind = {
                "connect": "body",
                "weld": "body",
                "joint": "joint",
                "tendon": "tendon",
                "flex": "flex",
                "distance": "geom",
            }.get(equality_type_name, "type_dependent")
            object1 = int(self._model.eq_obj1id[record.source_id])
            object2 = int(self._model.eq_obj2id[record.source_id])
            key = "eq_active_{}_active".format(
                self._source_property_token(
                    "equality", record.source_id
                )
            )
            emitted.append(
                self._emit_property(
                    perceived,
                    record.entity_id,
                    key,
                    float(bool(self._data.eq_active[record.source_id])),
                    PropertyMeta(
                        group="equality constraint state",
                        note="bool",
                        tags=(
                            self._source_identity_tags(
                                "equality", record.source_id
                            )
                            + (
                                "equality_type:{}".format(
                                    equality_type_name
                                ),
                                "object_kind:{}".format(object_kind),
                                "object1_index:{}".format(object1),
                                "object2_index:{}".format(object2),
                            )
                        ),
                    ),
                )
            )
        return True, emitted, None

    def _materialize_body_cartesian_state(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        owners = self._body_owner_pairs()
        if not owners:
            return self._not_applicable("body owner")
        emitted = []
        values = getattr(self._data, field_name)
        width = _plan_row_width(values)
        for entity_id, body_id in owners:
            token = self._body_token(body_id)
            for index in range(6):
                emitted.append(
                    self._emit_property(
                        perceived,
                        entity_id,
                        "{}_{}_{}".format(field_name, token, index),
                        values[body_id][index],
                        PropertyMeta(
                            group=(
                                "body applied force"
                                if index < 3
                                else "body applied torque"
                            ),
                            note="N" if index < 3 else "N·m",
                            tags=self._source_identity_tags(
                                "body", body_id
                            ),
                        ),
                        source=(
                            None
                            if width is None
                            else (field_name, body_id * width + index)
                        ),
                    )
                )
        return True, emitted, None

    def _materialize_body_pose(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        owners = self._body_owner_pairs()
        if not owners:
            return self._not_applicable("body owner")
        attr = self._body_pose_attr(field_name)
        width = _BODY_POSE_SOURCES[field_name][1]
        values = getattr(self._data, attr)
        emitted = []
        for entity_id, body_id in owners:
            token = self._body_token(body_id)
            flat = np.asarray(values[body_id], dtype=float).reshape(-1)
            # A row of the documented width is what makes the flat index
            # below (row-major, body_id * width + offset) correct; any
            # other layout emits as before but cannot be planned.
            row_planned = flat.size == width
            components = (
                list(_QUAT_COMPONENTS)
                if field_name == "xquat"
                else list(range(flat.size))
            )
            for offset, (component, value) in enumerate(
                zip(components, flat)
            ):
                key = "{}_{}_{}".format(field_name, token, component)
                if field_name == "xpos":
                    axis = int(component)
                    meta = PropertyMeta(
                        group="body position",
                        note="m",
                        family="xpos",
                        component=("x", "y", "z")[axis]
                        + " (axis {})".format(axis),
                        frame="world",
                    )
                elif field_name == "xquat":
                    meta = PropertyMeta(
                        group="body orientation",
                        note="quat {}".format(component),
                        unit="1",
                        family="xquat",
                        component=str(component),
                        frame="world",
                    )
                else:
                    meta = PropertyMeta(
                        group="body orientation matrix",
                        note="1",
                        tags=(
                            "matrix_component:{}".format(component),
                        ),
                    )
                meta = meta._replace(
                    tags=tuple(meta.tags)
                    + self._source_identity_tags("body", body_id)
                )
                emitted.append(
                    self._emit_property(
                        perceived,
                        entity_id,
                        key,
                        value,
                        meta,
                        source=(
                            (attr, body_id * width + offset)
                            if row_planned
                            else None
                        ),
                    )
                )
            body_record = next(
                (
                    record
                    for record in self._bodies
                    if record.entity_id == entity_id
                ),
                None,
            )
            if (
                field_name == "xpos"
                and body_record is not None
                and body_record.body_id == body_id
                and entity_id in perceived
            ):
                perceived[entity_id].position = flat.copy()
                emitted.append("entity[{}].position".format(entity_id))
                if self._plan_capture is not None:
                    self._plan_position_rows.append((entity_id, body_id))
        return True, emitted, None

    def _materialize_body_inertial_pose(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        owners = self._body_owner_pairs()
        if not owners:
            return self._not_applicable("body owner")
        emitted = []
        values = getattr(self._data, field_name)
        width = _plan_row_width(values)
        for entity_id, body_id in owners:
            token = self._body_token(body_id)
            flat = np.asarray(values[body_id], dtype=float).reshape(-1)
            for index, value in enumerate(flat):
                emitted.append(
                    self._emit_property(
                        perceived,
                        entity_id,
                        "{}_{}_{}".format(field_name, token, index),
                        value,
                        PropertyMeta(
                            group=(
                                "body inertial position"
                                if field_name == "xipos"
                                else "body inertial orientation matrix"
                            ),
                            note=(
                                "m"
                                if field_name == "xipos"
                                else "1"
                            ),
                            tags=(
                                ()
                                if field_name == "xipos"
                                else (
                                    "matrix_component:{}".format(index),
                                )
                            )
                            + self._source_identity_tags("body", body_id),
                        ),
                        source=(
                            None
                            if width is None
                            else (field_name, body_id * width + index)
                        ),
                    )
                )
        return True, emitted, None

    def _materialize_joint_geometry(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        owners = (
            self._xaxis_owner_triples()
            if field_name == "xaxis"
            else self._joint_owner_triples()
        )
        if not owners:
            return self._not_applicable(
                "hinge/slide joint owner"
                if field_name == "xaxis"
                else "joint owner"
            )
        values = getattr(self._data, field_name)
        width = _plan_row_width(values)
        emitted = []
        for _, entity_id, joint_id in owners:
            token = self._joint_property_token(joint_id)
            for axis in range(3):
                emitted.append(
                    self._emit_property(
                        perceived,
                        entity_id,
                        "{}_{}_{}".format(field_name, token, axis),
                        values[joint_id][axis],
                        PropertyMeta(
                            group=(
                                "joint anchor"
                                if field_name == "xanchor"
                                else "joint axis"
                            ),
                            note="m" if field_name == "xanchor" else "1",
                            tags=(
                                tuple(
                                    self._joint_subsystem_tags(joint_id)
                                )
                                + self._source_identity_tags(
                                    "joint", joint_id
                                )
                            ),
                        ),
                        source=(
                            None
                            if width is None
                            else (field_name, joint_id * width + axis)
                        ),
                    )
                )
        if not emitted:
            return (
                False,
                [],
                "current reset model has no hinge/slide joint owner"
                if field_name == "xaxis"
                else "current reset model has no resolvable joint anchor",
            )
        return True, emitted, None

    def _materialize_subtree_state(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        owners = self._body_owner_pairs()
        if not owners:
            return self._not_applicable("body owner")
        notes = {
            "subtree_com": ("subtree com position", "m"),
            "subtree_linvel": ("subtree linear velocity", "m/s"),
            "subtree_angmom": (
                "subtree angular momentum", "kg·m²/s"
            ),
        }
        group, note = notes[field_name]
        values = getattr(self._data, field_name)
        width = _plan_row_width(values)
        emitted = []
        for entity_id, body_id in owners:
            token = self._body_token(body_id)
            for axis in range(3):
                emitted.append(
                    self._emit_property(
                        perceived,
                        entity_id,
                        "{}_{}_{}".format(field_name, token, axis),
                        values[body_id][axis],
                        PropertyMeta(
                            group=group,
                            note=note,
                            tags=self._source_identity_tags(
                                "body", body_id
                            ),
                        ),
                        source=(
                            None
                            if width is None
                            else (field_name, body_id * width + axis)
                        ),
                    )
                )
        return True, emitted, None

    def _motion_axis_meta(
        self,
        field_name: str,
        joint_type: int,
        dof_index: int,
        component: int,
    ) -> PropertyMeta:
        translational_dof = (
            joint_type == _MJ_JNT_SLIDE
            or (joint_type == _MJ_JNT_FREE and dof_index < 3)
        )
        angular_component = component < 3
        if translational_dof:
            note = "rad/m" if angular_component else "1"
        else:
            note = "1" if angular_component else "m/rad"
        if field_name == "cdof_dot":
            note = {
                "rad/m": "rad/(m·s)",
                "1": "1/s",
                "m/rad": "m/(rad·s)",
            }[note]
        return PropertyMeta(
            group="joint {} motion axis ({})".format(
                "angular" if angular_component else "linear",
                note,
            )
            if field_name == "cdof"
            else "joint {} motion-axis derivative ({})".format(
                "angular" if angular_component else "linear",
                note,
            ),
            note=note,
        )

    def _materialize_joint_motion_axis(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        owners = self._joint_owner_triples()
        if not owners:
            return self._not_applicable("joint owner")
        values = getattr(self._data, field_name)
        width = _plan_row_width(values)
        emitted = []
        for _, entity_id, joint_id in owners:
            name = self._indexed_source_name("joint", joint_id)
            token = self._joint_property_token(joint_id)
            start, stop = self._joint_slice(
                joint_id, "jnt_dofadr", "nv"
            )
            joint_type = int(self._model.jnt_type[joint_id])
            for local_dof, global_dof in enumerate(range(start, stop)):
                for component in range(6):
                    emitted.append(
                        self._emit_property(
                            perceived,
                            entity_id,
                            "{}_{}_{}_{}".format(
                                field_name, token, local_dof, component
                            ),
                            values[global_dof][component],
                            self._motion_axis_meta(
                                field_name,
                                joint_type,
                                local_dof,
                                component,
                            )._replace(
                                tags=(
                                    tuple(
                                        self._joint_subsystem_tags(
                                            joint_id
                                        )
                                    )
                                    + self._source_identity_tags(
                                        "joint", joint_id
                                    )
                                    + self._cframe_tags(
                                        int(
                                            self._model.jnt_bodyid[
                                                joint_id
                                            ]
                                        )
                                    )
                                )
                            ),
                            source=(
                                None
                                if width is None
                                else (
                                    field_name,
                                    global_dof * width + component,
                                )
                            ),
                        )
                    )
        if not emitted:
            return self._not_applicable("joint-DOF owner")
        return True, emitted, None

    def _materialize_body_inertia(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        owners = [
            (entity_id, body_id)
            for entity_id, body_id in self._body_owner_pairs()
            if body_id != 0
        ]
        if not owners:
            return self._not_applicable("non-world body inertia owner")
        values = getattr(self._data, field_name)
        width = _plan_row_width(values)
        emitted = []
        for entity_id, body_id in owners:
            token = self._body_token(body_id)
            flat = np.asarray(values[body_id], dtype=float).reshape(-1)
            if flat.size != 10:
                raise RuntimeError(
                    "MjData.{} body row must have 10 components".format(
                        field_name
                    )
                )
            for index, value in enumerate(flat):
                if index < 6:
                    note = "kg·m²"
                elif index < 9:
                    note = "kg·m"
                else:
                    note = "kg"
                emitted.append(
                    self._emit_property(
                        perceived,
                        entity_id,
                        "{}_{}_{}".format(field_name, token, index),
                        value,
                        PropertyMeta(
                            group="{} spatial inertia ({})".format(
                                "body"
                                if field_name == "cinert"
                                else "composite body",
                                note,
                            ),
                            note=note,
                            tags=(
                                self._cframe_tags(body_id)
                                + self._source_identity_tags(
                                    "body", body_id
                                )
                            ),
                        ),
                        source=(
                            None
                            if width is None
                            else (field_name, body_id * width + index)
                        ),
                    )
                )
        return True, emitted, None

    def _materialize_body_spatial_state(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        return self._materialize_body_six_vector(
            field_name, perceived
        )

    def _materialize_body_post_constraint_state(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        return self._materialize_body_six_vector(
            field_name, perceived
        )

    def _materialize_body_six_vector(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        owners = self._body_owner_pairs()
        if not owners:
            return self._not_applicable("body owner")
        definition = next(
            item for item in _BODY_SPATIAL_FIELDS
            if item[0] == field_name
        )
        _, rot_group, rot_note, lin_group, lin_note = definition
        values = getattr(self._data, field_name)
        width = _plan_row_width(values)
        emitted = []
        for entity_id, body_id in owners:
            token = self._body_token(body_id)
            for index in range(6):
                emitted.append(
                    self._emit_property(
                        perceived,
                        entity_id,
                        "{}_{}_{}".format(field_name, token, index),
                        values[body_id][index],
                        PropertyMeta(
                            group=rot_group if index < 3 else lin_group,
                            note=rot_note if index < 3 else lin_note,
                            tags=(
                                self._cframe_tags(body_id)
                                + self._source_identity_tags(
                                    "body", body_id
                                )
                            ),
                        ),
                        source=(
                            None
                            if width is None
                            else (field_name, body_id * width + index)
                        ),
                    )
                )
        return True, emitted, None

    def _materialize_geom_pose(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        if not self._raw_geoms:
            return self._not_applicable("geom")
        values = getattr(self._data, field_name)
        width = _plan_row_width(values)
        emitted = []
        for record in self._raw_geoms:
            token = self._source_property_token(
                "geom", record.source_id
            )
            flat = np.asarray(
                values[record.source_id], dtype=float
            ).reshape(-1)
            for index, value in enumerate(flat):
                emitted.append(
                    self._emit_property(
                        perceived,
                        record.entity_id,
                        "{}_{}_{}".format(field_name, token, index),
                        value,
                        PropertyMeta(
                            group=(
                                "geom position"
                                if field_name == "geom_xpos"
                                else "geom orientation matrix"
                            ),
                            note=(
                                "m"
                                if field_name == "geom_xpos"
                                else "1"
                            ),
                            tags=(
                                ("geom_type:{}".format(
                                    self._geom_type_name(
                                        record.source_id
                                    )
                                ),)
                                + self._source_identity_tags(
                                    "geom", record.source_id
                                )
                                + (
                                    ()
                                    if field_name == "geom_xpos"
                                    else (
                                        "matrix_component:{}".format(
                                            index
                                        ),
                                    )
                                )
                            ),
                        ),
                        source=(
                            None
                            if width is None
                            else (
                                field_name,
                                record.source_id * width + index,
                            )
                        ),
                    )
                )
            if field_name == "geom_xpos":
                if record.entity_id in perceived:
                    perceived[record.entity_id].position = flat.copy()
                    emitted.append(
                        "entity[{}].position".format(record.entity_id)
                    )
                    if self._plan_capture is not None:
                        self._plan_position_rows.append(
                            (record.entity_id, record.source_id)
                        )
        return True, emitted, None

    def _materialize_camera_pose(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        """Expose each compiled camera's current world-frame pose."""
        if not self._raw_cameras:
            return self._not_applicable("camera")
        values = getattr(self._data, field_name)
        width = _plan_row_width(values)
        emitted = []
        for record in self._raw_cameras:
            token = self._source_property_token(
                "camera", record.source_id
            )
            flat = np.asarray(
                values[record.source_id], dtype=float
            ).reshape(-1)
            identity_tags = self._source_identity_tags(
                "camera", record.source_id
            )
            for index, value in enumerate(flat):
                tags = identity_tags
                if field_name == "cam_xmat":
                    tags += ("matrix_component:{}".format(index),)
                emitted.append(
                    self._emit_property(
                        perceived,
                        record.entity_id,
                        "{}_{}_{}".format(field_name, token, index),
                        value,
                        PropertyMeta(
                            group=(
                                "camera position"
                                if field_name == "cam_xpos"
                                else "camera orientation matrix"
                            ),
                            note="m" if field_name == "cam_xpos" else "1",
                            tags=tags,
                        ),
                        source=(
                            None
                            if width is None
                            else (
                                field_name,
                                record.source_id * width + index,
                            )
                        ),
                    )
                )
            if field_name == "cam_xpos" and record.entity_id in perceived:
                perceived[record.entity_id].position = flat.copy()
                emitted.append(
                    "entity[{}].position".format(record.entity_id)
                )
                if self._plan_capture is not None:
                    self._plan_position_rows.append(
                        (record.entity_id, record.source_id)
                    )
        return True, emitted, None

    def _materialize_light_pose(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        """Expose rendering-sensor light position or world direction."""
        if not self._raw_lights:
            return self._not_applicable("light")
        values = getattr(self._data, field_name)
        width = _plan_row_width(values)
        emitted = []
        for record in self._raw_lights:
            token = self._source_property_token(
                "light", record.source_id
            )
            tags = self._source_identity_tags(
                "light", record.source_id
            ) + ("sensor_configuration:rendering",)
            flat = np.asarray(
                values[record.source_id], dtype=float
            ).reshape(-1)
            for axis, value in enumerate(flat):
                emitted.append(
                    self._emit_property(
                        perceived,
                        record.entity_id,
                        "{}_{}_{}".format(field_name, token, axis),
                        value,
                        PropertyMeta(
                            group=(
                                "light position"
                                if field_name == "light_xpos"
                                else "light direction"
                            ),
                            note="m" if field_name == "light_xpos" else "1",
                            tags=tags,
                        ),
                        source=(
                            None
                            if width is None
                            else (
                                field_name,
                                record.source_id * width + axis,
                            )
                        ),
                    )
                )
        return True, emitted, None

    def _site_owner_records(
        self,
    ) -> List[_NamedOwnerRecord]:
        records = list(self._raw_sites)
        if self._gripper is not None:
            records.append(
                _NamedOwnerRecord(
                    entity_id=self._gripper.entity_id,
                    source_name=self._gripper.site_name,
                    source_id=self._gripper.site_id,
                    entity_class=self._gripper.entity_class,
                )
            )
        return records

    def _site_type_name(
        self,
        site_id: int,
    ) -> str:
        size = np.asarray(self._model.site_size[site_id], dtype=float)
        if not np.any(size):
            return "marker"
        return self._geom_type_name(site_id, site=True)

    def _geom_type_name(
        self,
        object_id: int,
        site: bool = False,
    ) -> str:
        cache_key = (site, int(object_id))
        cached = self._geom_type_cache.get(cache_key)
        if cached is not None:
            return cached
        import mujoco

        values = self._model.site_type if site else self._model.geom_type
        enum_name = mujoco.mjtGeom(int(values[object_id])).name
        name = enum_name.replace("mjGEOM_", "", 1).lower()
        self._geom_type_cache[cache_key] = name
        return name

    def _materialize_site_pose(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        owners = self._site_owner_records()
        if not owners:
            return self._not_applicable("site")
        values = getattr(self._data, field_name)
        width = _plan_row_width(values)
        emitted = []
        for record in owners:
            token = self._source_property_token(
                "site", record.source_id
            )
            flat = np.asarray(
                values[record.source_id], dtype=float
            ).reshape(-1)
            for index, value in enumerate(flat):
                emitted.append(
                    self._emit_property(
                        perceived,
                        record.entity_id,
                        "{}_{}_{}".format(field_name, token, index),
                        value,
                        PropertyMeta(
                            group=(
                                "site position"
                                if field_name == "site_xpos"
                                else "site orientation matrix"
                            ),
                            note=(
                                "m"
                                if field_name == "site_xpos"
                                else "1"
                            ),
                            tags=(
                                ("site_type:{}".format(
                                    self._site_type_name(
                                        record.source_id
                                    )
                                ),)
                                + self._source_identity_tags(
                                    "site", record.source_id
                                )
                                + (
                                    ()
                                    if field_name == "site_xpos"
                                    else (
                                        "matrix_component:{}".format(
                                            index
                                        ),
                                    )
                                )
                            ),
                        ),
                        source=(
                            None
                            if width is None
                            else (
                                field_name,
                                record.source_id * width + index,
                            )
                        ),
                    )
                )
            if field_name == "site_xpos":
                if record.entity_id in perceived:
                    perceived[record.entity_id].position = flat.copy()
                    emitted.append(
                        "entity[{}].position".format(record.entity_id)
                    )
                    if self._plan_capture is not None:
                        self._plan_position_rows.append(
                            (record.entity_id, record.source_id)
                        )
        return True, emitted, None

    def _materialize_mocap_state(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        if not self._mocap_owners:
            return self._not_applicable("mocap body")
        values = getattr(self._data, field_name)
        width = _plan_row_width(values)
        emitted = []
        for record in self._mocap_owners:
            token = self._source_property_token(
                "body", record.body_id
            )
            flat = np.asarray(
                values[record.mocap_id], dtype=float
            ).reshape(-1)
            components = (
                list(_QUAT_COMPONENTS)
                if field_name == "mocap_quat"
                else list(range(flat.size))
            )
            for offset, (component, value) in enumerate(
                zip(components, flat)
            ):
                emitted.append(
                    self._emit_property(
                        perceived,
                        record.entity_id,
                        "{}_{}_{}".format(
                            field_name, token, component
                        ),
                        value,
                        PropertyMeta(
                            group=(
                                "mocap position"
                                if field_name == "mocap_pos"
                                else "mocap orientation"
                            ),
                            note=(
                                "m"
                                if field_name == "mocap_pos"
                                else "quat {}".format(component)
                            ),
                        ),
                        source=(
                            None
                            if width is None
                            else (
                                field_name,
                                record.mocap_id * width + offset,
                            )
                        ),
                    )
                )
            if field_name == "mocap_pos":
                if record.entity_id in perceived:
                    perceived[record.entity_id].position = flat.copy()
                    emitted.append(
                        "entity[{}].position".format(record.entity_id)
                    )
                    if self._plan_capture is not None:
                        self._plan_position_rows.append(
                            (record.entity_id, record.mocap_id)
                        )
        return True, emitted, None

    def _emit_flex_rest_lengths(
        self,
        perceived: Dict[int, EntityState],
    ) -> None:
        """Per-edge rest lengths as entity parameters.

        MjModel constants, not MjData state — the X14 decision
        (2026-08-08) exposes them beside the live edge lengths so strain
        is expressible in a formula. Emitted outside the planned-field
        machinery (they have no MjData source) and documented as
        MjModel-provenance, outside the E8 field-coverage population.
        """
        if not self._flexes:
            return
        rest = getattr(self._model, "flexedge_length0", None)
        addresses = getattr(self._model, "flex_edgeadr", None)
        counts = getattr(self._model, "flex_edgenum", None)
        if rest is None or addresses is None or counts is None:
            return
        for record in self._flexes:
            if record.entity_id not in perceived:
                continue
            start = int(addresses[record.source_id])
            count = int(counts[record.source_id])
            token = self._source_property_token("flex", record.source_id)
            for local_index in range(count):
                self._emit_property(
                    perceived,
                    record.entity_id,
                    "flexedge_length0_{}_{}".format(token, local_index),
                    float(rest[start + local_index]),
                    PropertyMeta(
                        group="flex edge rest length (MjModel constant)",
                        note="m",
                    ),
                    source=None,
                )

    def _materialize_flex_state(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        if not self._flexes:
            return self._not_applicable("flex")
        emitted = []
        values = getattr(self._data, field_name)
        width = _plan_row_width(values)
        vertex_field = field_name == "flexvert_xpos"
        adr_name = "flex_vertadr" if vertex_field else "flex_edgeadr"
        num_name = "flex_vertnum" if vertex_field else "flex_edgenum"
        addresses = getattr(self._model, adr_name, None)
        counts = getattr(self._model, num_name, None)
        if addresses is None or counts is None:
            raise RuntimeError(
                "Binding exposes MjData.{} but model lacks {}/{}".format(
                    field_name, adr_name, num_name
                )
            )
        applicable_records = []
        import mujoco

        version_parts = tuple(
            int(part)
            for part in re.findall(r"\d+", mujoco.__version__)[:2]
        )
        equality_requires_mode_one = version_parts >= (3, 10)
        interpolation_skip_exists = version_parts >= (3, 3)
        flex_damping_requests_jacobian = version_parts >= (3, 3)
        for record in self._flexes:
            count = int(counts[record.source_id])
            if count == 0:
                self._omit_materialization_owner(
                    record.source_name,
                    "the flex declares no {}"
                    .format("vertices" if vertex_field else "edges"),
                )
                continue
            if vertex_field:
                applicable_records.append(record)
                continue
            rigid = bool(self._model.flex_rigid[record.source_id])
            flex_interp = getattr(self._model, "flex_interp", None)
            interpolated = (
                bool(flex_interp[record.source_id])
                if (
                    interpolation_skip_exists
                    and flex_interp is not None
                )
                else False
            )
            if rigid or interpolated:
                self._omit_materialization_owner(
                    record.source_name,
                    "MuJoCo does not populate edge state for {} flexes"
                    .format("rigid" if rigid else "interpolated"),
                )
                continue
            if field_name == "flexedge_velocity":
                equality_mode = int(
                    self._model.flex_edgeequality[record.source_id]
                )
                equality_requests_jacobian = (
                    equality_mode == 1
                    if equality_requires_mode_one
                    else bool(equality_mode)
                )
                edge_jacobian_needed = any(
                    (
                        equality_requests_jacobian,
                        bool(
                            self._model.flex_edgedamping[
                                record.source_id
                            ]
                        ),
                        bool(
                            self._model.flex_edgestiffness[
                                record.source_id
                            ]
                        ),
                        bool(
                            self._model.flex_damping[
                                record.source_id
                            ]
                        )
                        if flex_damping_requests_jacobian
                        else False,
                    )
                )
                if not edge_jacobian_needed:
                    self._omit_materialization_owner(
                        record.source_name,
                        "MuJoCo did not request this flex's edge Jacobian",
                    )
                    continue
            applicable_records.append(record)

        if not applicable_records:
            return (
                False,
                [],
                "current reset model has no flex owner whose {} is "
                "populated".format(field_name),
            )

        for record in applicable_records:
            start = int(addresses[record.source_id])
            count = int(counts[record.source_id])
            token = self._source_property_token(
                "flex", record.source_id
            )
            for local_index in range(count):
                value = values[start + local_index]
                flat = np.asarray(value, dtype=float).reshape(-1)
                for component, scalar in enumerate(flat):
                    key = "{}_{}_{}".format(
                        field_name, token, local_index
                    )
                    if flat.size > 1:
                        key += "_{}".format(component)
                    emitted.append(
                        self._emit_property(
                            perceived,
                            record.entity_id,
                            key,
                            scalar,
                            PropertyMeta(
                                group=(
                                    "flex vertex position"
                                    if vertex_field
                                    else "flex edge length"
                                    if field_name == "flexedge_length"
                                    else "flex edge velocity"
                                ),
                                note=(
                                    "m"
                                    if field_name != "flexedge_velocity"
                                    else "m/s"
                                ),
                            ),
                            source=(
                                None
                                if width is None
                                else (
                                    field_name,
                                    (start + local_index) * width
                                    + component,
                                )
                            ),
                        )
                    )
        return True, emitted, None

    def _tendon_unit(
        self,
        record: _NamedOwnerRecord,
        field_name: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        unit, failure = self._tendon_coordinate_unit(record.source_id)
        if unit is None:
            return None, failure
        return (
            unit if field_name == "ten_length" else "{}/s".format(unit),
            None,
        )

    def _tendon_coordinate_unit(
        self,
        tendon_id: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo is required to resolve tendon units"
            ) from exc
        start = int(self._model.tendon_adr[tendon_id])
        count = int(self._model.tendon_num[tendon_id])
        wrap_types = np.asarray(
            self._model.wrap_type[start:start + count], dtype=int
        )
        joint_wrap = int(mujoco.mjtWrap.mjWRAP_JOINT)
        if not np.any(wrap_types == joint_wrap):
            unit = "m"
        elif np.all(wrap_types == joint_wrap):
            joint_ids = np.asarray(
                self._model.wrap_objid[start:start + count], dtype=int
            )
            coefficients = np.asarray(
                self._model.wrap_prm[start:start + count], dtype=float
            )
            units = set()
            for joint_id, coefficient in zip(joint_ids, coefficients):
                if coefficient == 0.0:
                    continue
                joint_type = int(self._model.jnt_type[joint_id])
                if joint_type == _MJ_JNT_SLIDE:
                    units.add("m")
                elif joint_type == _MJ_JNT_HINGE:
                    units.add("rad")
                else:
                    return (
                        None,
                        "nonzero term references non-scalar joint {}"
                        .format(joint_id),
                    )
            if not units:
                return (
                    None,
                    "no nonzero fixed-tendon term establishes a unit",
                )
            if len(units) != 1:
                return (
                    None,
                    "nonzero terms mix joint coordinate units {}".format(
                        sorted(units)
                    ),
                )
            unit = next(iter(units))
        else:
            return (
                None,
                "path mixes spatial and fixed-joint wraps",
            )
        return unit, None

    def _expand_row_jacobian(
        self,
        *,
        values_name: str,
        row_count: int,
        column_count: int,
        sparse_prefix: str,
    ) -> np.ndarray:
        """Return a dense row-by-DOF Jacobian across supported MuJoCo layouts."""
        raw_values = np.asarray(
            getattr(self._data, values_name), dtype=float
        )
        if raw_values.shape == (row_count, column_count):
            return raw_values

        metadata = tuple(
            "{}_{}".format(sparse_prefix, suffix)
            for suffix in ("rownnz", "rowadr", "colind")
        )
        metadata_owner = next(
            (
                owner
                for owner in (self._data, self._model)
                if all(hasattr(owner, name) for name in metadata)
            ),
            None,
        )
        if metadata_owner is None:
            raise RuntimeError(
                "MjData.{} shape {} is neither dense ({}, {}) nor "
                "accompanied by sparse row metadata".format(
                    values_name,
                    raw_values.shape,
                    row_count,
                    column_count,
                )
            )
        rownnz = np.asarray(
            getattr(metadata_owner, metadata[0]), dtype=int
        ).reshape(-1)
        rowadr = np.asarray(
            getattr(metadata_owner, metadata[1]), dtype=int
        ).reshape(-1)
        colind = np.asarray(
            getattr(metadata_owner, metadata[2]), dtype=int
        ).reshape(-1)
        sparse_values = raw_values.reshape(-1)
        dense = np.zeros((row_count, column_count), dtype=float)
        for row in range(row_count):
            start = int(rowadr[row])
            count = int(rownnz[row])
            for sparse_index in range(start, start + count):
                column = int(colind[sparse_index])
                if column < 0 or column >= column_count:
                    raise RuntimeError(
                        "{} column {} is outside nv={}".format(
                            values_name, column, column_count
                        )
                    )
                dense[row, column] = sparse_values[sparse_index]
        return dense

    @staticmethod
    def _jacobian_unit(
        numerator_unit: str,
        motion_kind: str,
    ) -> str:
        coordinate_unit = "m" if motion_kind == "linear" else "rad"
        return (
            "1"
            if numerator_unit == coordinate_unit
            else "{}/{}".format(numerator_unit, coordinate_unit)
        )

    def _materialize_flex_edge_jacobian(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        del field_name
        if not self._flexes:
            return self._not_applicable("flex")
        edge_count = int(getattr(self._model, "nflexedge", 0) or 0)
        dofs = self._dof_descriptors()
        if edge_count == 0 or not dofs:
            return self._not_applicable("flex edge and generalized DOF")
        dense = self._expand_row_jacobian(
            values_name="flexedge_J",
            row_count=edge_count,
            column_count=len(dofs),
            sparse_prefix="flexedge_J",
        )
        emitted: List[str] = []
        capable_owner = False
        for record in self._flexes:
            omission = self._flex_edge_jacobian_omission(record)
            if omission is not None:
                self._omit_materialization_owner(
                    record.source_name, omission
                )
                continue
            capable_owner = True
            start = int(self._model.flex_edgeadr[record.source_id])
            count = int(self._model.flex_edgenum[record.source_id])
            token = self._source_property_token(
                "flex", record.source_id
            )
            for local_edge in range(count):
                global_edge = start + local_edge
                for dof_id, (
                    _dof_token,
                    motion_kind,
                    joint_id,
                    local_dof,
                ) in enumerate(dofs):
                    note = self._jacobian_unit("m", motion_kind)
                    emitted.append(
                        self._emit_property(
                            perceived,
                            record.entity_id,
                            (
                                "flexedge_jacobian_{}_edge_{}_dof_{}"
                                .format(token, local_edge, dof_id)
                            ),
                            dense[global_edge, dof_id],
                            PropertyMeta(
                                group=(
                                    "flex-edge length Jacobian ({})"
                                    .format(note)
                                ),
                                note=note,
                                tags=(
                                    self._qualified_source_identity_tags(
                                        "flex",
                                        "flex",
                                        record.source_id,
                                    )
                                    + self._qualified_source_identity_tags(
                                        "joint", "joint", joint_id
                                    )
                                    + (
                                        "flex_global_edge_index:{}".format(
                                            global_edge
                                        ),
                                        "dof_index:{}".format(dof_id),
                                        "joint_local_dof:{}".format(
                                            local_dof
                                        ),
                                    )
                                ),
                            ),
                        )
                    )
        if not capable_owner or not emitted:
            return (
                False,
                [],
                "current reset model has no flex owner whose edge "
                "Jacobian is populated",
            )
        return True, emitted, None

    def _flex_edge_jacobian_omission(
        self,
        record: _NamedOwnerRecord,
    ) -> Optional[str]:
        count = int(self._model.flex_edgenum[record.source_id])
        if count == 0:
            return "the flex declares no edges"
        if bool(self._model.flex_rigid[record.source_id]):
            return "MuJoCo does not populate edge state for rigid flexes"

        import mujoco

        version_parts = tuple(
            int(part)
            for part in re.findall(r"\d+", mujoco.__version__)[:2]
        )
        flex_interp = getattr(self._model, "flex_interp", None)
        if (
            version_parts >= (3, 3)
            and flex_interp is not None
            and bool(flex_interp[record.source_id])
        ):
            return (
                "MuJoCo does not populate edge state for interpolated "
                "flexes"
            )
        equality_mode = int(
            self._model.flex_edgeequality[record.source_id]
        )
        equality_requests_jacobian = (
            equality_mode == 1
            if version_parts >= (3, 10)
            else bool(equality_mode)
        )
        edge_jacobian_needed = any(
            (
                equality_requests_jacobian,
                bool(self._model.flex_edgedamping[record.source_id]),
                bool(self._model.flex_edgestiffness[record.source_id]),
                (
                    bool(self._model.flex_damping[record.source_id])
                    if version_parts >= (3, 3)
                    else False
                ),
            )
        )
        if not edge_jacobian_needed:
            return "MuJoCo did not request this flex's edge Jacobian"
        return None

    def _materialize_tendon_jacobian(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        del field_name
        if not self._tendons:
            return self._not_applicable("tendon")
        dofs = self._dof_descriptors()
        if not dofs:
            return self._not_applicable("generalized DOF")
        dense = self._expand_row_jacobian(
            values_name="ten_J",
            row_count=len(self._tendons),
            column_count=len(dofs),
            sparse_prefix="ten_J",
        )
        emitted: List[str] = []
        for record in self._tendons:
            numerator_unit, _failure = self._tendon_unit(
                record, "ten_length"
            )
            for dof_id, (
                _dof_token,
                motion_kind,
                joint_id,
                local_dof,
            ) in enumerate(dofs):
                note = (
                    None
                    if numerator_unit is None
                    else self._jacobian_unit(
                        numerator_unit, motion_kind
                    )
                )
                emitted.append(
                    self._emit_property(
                        perceived,
                        record.entity_id,
                        "tendon_jacobian_tendon_{}_dof_{}".format(
                            record.source_id, dof_id
                        ),
                        dense[record.source_id, dof_id],
                        PropertyMeta(
                            group=(
                                "tendon-coordinate Jacobian ({})"
                                .format(
                                    note
                                    if note is not None
                                    else "{}; opaque".format(
                                        record.source_name
                                    )
                                )
                            ),
                            note=note,
                            tags=(
                                self._qualified_source_identity_tags(
                                    "tendon",
                                    "tendon",
                                    record.source_id,
                                )
                                + self._qualified_source_identity_tags(
                                    "joint", "joint", joint_id
                                )
                                + (
                                    "dof_index:{}".format(dof_id),
                                    "joint_local_dof:{}".format(local_dof),
                                )
                                + (
                                    ()
                                    if note is not None
                                    else (
                                        "semantics:opaque",
                                        "unit:unresolved",
                                    )
                                )
                            ),
                        ),
                    )
                )
        if not emitted:
            return (
                False,
                [],
                "current reset model has no tendon Jacobian entries",
            )
        return True, emitted, None

    def _materialize_tendon_state(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        if not self._tendons:
            return self._not_applicable("tendon")
        values = getattr(self._data, field_name)
        flat_source = _plan_row_width(values) == 1
        emitted = []
        for record in self._tendons:
            note, _failure = self._tendon_unit(record, field_name)
            emitted.append(
                self._emit_property(
                    perceived,
                    record.entity_id,
                    "{}_{}".format(
                        field_name,
                        self._source_property_token(
                            "tendon", record.source_id
                        ),
                    ),
                    values[record.source_id],
                    PropertyMeta(
                        group=(
                            "tendon {}".format(
                                "length"
                                if field_name == "ten_length"
                                else "velocity"
                            )
                            if note is not None
                            else "tendon {} ({}; opaque)".format(
                                (
                                    "length"
                                    if field_name == "ten_length"
                                    else "velocity"
                                ),
                                record.source_name,
                            )
                        ),
                        note=note,
                        tags=(
                            ()
                            if note is not None
                            else (
                                "semantics:opaque",
                                "unit:unresolved",
                            )
                        ),
                    ),
                    source=(
                        (field_name, record.source_id)
                        if flat_source
                        else None
                    ),
                )
            )
        if not emitted:
            return (
                False,
                [],
                "current reset model has no tendon state entries",
            )
        return True, emitted, None

    def _materialize_sensor_state(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        del field_name
        if not self._sensors:
            return self._not_applicable("sensor")
        emitted = []
        sensordata = self._data.sensordata
        flat_source = _plan_row_width(sensordata) == 1
        for record in self._sensors:
            entity_id = record.owner_entity_id
            if entity_id is None:
                entity_id = self._orphan_sensor_entity_id
            if entity_id is None:
                raise RuntimeError(
                    "Sensor {!r} has no registered owner".format(record.name)
                )
            values = sensordata[
                record.adr:record.adr + record.dim
            ]
            meta = PropertyMeta(
                group=(
                    "sensors: {}".format(record.type_name)
                    if record.unit_note is not None
                    else "sensor: {} (opaque)".format(record.name_token)
                ),
                note=record.unit_note,
                tags=(
                    self._source_identity_tags(
                        "sensor", record.sensor_id
                    )
                    + ("sensor_type:{}".format(record.type_name),)
                    + self._sensor_target_tags(record.sensor_id)
                    + (
                        ()
                        if record.unit_note is not None
                        else ("semantics:opaque", "unit:unresolved")
                    )
                ),
            )
            for offset, (key, value) in enumerate(
                zip(self._sensor_keys(record), values)
            ):
                emitted.append(
                    self._emit_property(
                        perceived,
                        entity_id,
                        key,
                        value,
                        meta,
                        source=(
                            ("sensordata", record.adr + offset)
                            if flat_source
                            else None
                        ),
                    )
                )
        return True, emitted, None

    _CONTACT_PROPERTY_META: Mapping[str, PropertyMeta] = {
        "contact_point_force_max": PropertyMeta(
            group="contact force", note="N"
        ),
        "contact_normal_force_max": PropertyMeta(
            group="contact force", note="N"
        ),
        "contact_normal_force_sum": PropertyMeta(
            group="contact force", note="N"
        ),
        "contact_resultant_force": PropertyMeta(
            group="contact force", note="N"
        ),
        "contact_point_count": PropertyMeta(
            group="contact points", note="count"
        ),
        "contact_min_dist": PropertyMeta(
            group="contact separation", note="m"
        ),
    }

    def _contact_body_token(self, body_id: int) -> str:
        name = self._id2name("body", body_id)
        if name:
            return name
        return "body#{}".format(body_id)

    def _materialize_contact_state(
        self,
        field_name: str,
        perceived: Dict[int, EntityState],
    ) -> Tuple[bool, List[str], Optional[str]]:
        """Materialize retained contact rows as dynamic pair entities.

        Retained rows (``exclude == 0`` and ``efc_address >= 0``) are
        grouped by unordered body pair; each pair gets one entity whose
        integer ID is allocated on first appearance, never reassigned to
        another pair within the episode, and reused when the same pair
        reappears. Once a pair has been seen it stays in the population
        for the rest of the episode: a separated pair's contact force is
        exactly zero, which is measured state — emitting it avoids the
        monitor suspension/rewarm cycle that would otherwise discard an
        intermittent contact's temporal history. ``contact_min_dist`` is
        emitted only while the pair has retained points, because no
        contact row measures a separated pair's clearance.
        """
        import mujoco

        raw_model = _raw_struct(self._model, "_model")
        raw_data = _raw_struct(self._data, "_data")

        if not hasattr(raw_data, "contact"):
            return (
                False,
                [],
                "live binding does not expose MjData.contact",
            )

        groups: Dict[Tuple[int, int], List[int]] = {}
        for index in range(int(raw_data.ncon)):
            contact = raw_data.contact[index]
            if (
                int(contact.exclude) != 0
                or int(contact.efc_address) < 0
            ):
                continue
            pair = _contact_pair_key(raw_model, contact)
            groups.setdefault(pair, []).append(index)

        for pair in groups:
            if pair in self._contact_pairs:
                continue
            is_robot = (
                pair[0] in self._robot_body_ids
                or pair[1] in self._robot_body_ids
            )
            entity_class = (
                ROBOT_CONTACT_ENTITY_CLASS
                if is_robot
                else CONTACT_ENTITY_CLASS
            )
            tokens = sorted(
                self._contact_body_token(body_id) for body_id in pair
            )
            label = "{}:{}|{}".format(
                entity_class, tokens[0], tokens[1]
            )
            entity_id = self._next_runtime_entity_id
            self._next_runtime_entity_id += 1
            self._reserve_name(label, entity_id)
            self.entity_labels[entity_id] = label
            self._contact_pairs[pair] = _ContactPairRecord(
                entity_id=entity_id,
                body_pair=pair,
                entity_class=entity_class,
                label=label,
                tags=tuple(self._contact_pair_tags(pair)),
            )

        if not self._contact_pairs:
            return (
                False,
                [],
                "no retained contact rows and no previously seen "
                "pairs this episode",
            )

        emitted: List[str] = []
        result = np.zeros(6, dtype=np.float64)
        for pair, record in self._contact_pairs.items():
            rows = groups.get(pair, ())
            if rows:
                normals: List[float] = []
                magnitudes: List[float] = []
                resultant = np.zeros(3, dtype=np.float64)
                min_dist: Optional[float] = None
                for index in rows:
                    contact = raw_data.contact[index]
                    mujoco.mj_contactForce(
                        raw_model, raw_data, index, result
                    )
                    rotation = np.asarray(
                        contact.frame, dtype=np.float64
                    ).reshape(3, 3)
                    world = rotation.T @ result[0:3]
                    # Orient every point force as the force applied by
                    # the pair's lower body ID to its higher body ID.
                    # mj_contactForce's vector acts on geom[1] with the
                    # normal pointing geom[0] -> geom[1], and the geom
                    # order is collision-table order, not ID order.
                    geoms = getattr(contact, "geom", None)
                    if geoms is None:
                        geoms = (contact.geom1, contact.geom2)
                    receiving_body = int(
                        raw_model.geom_bodyid[int(geoms[1])]
                    )
                    if receiving_body != pair[1]:
                        world = -world
                    resultant += world
                    normals.append(float(result[0]))
                    magnitudes.append(
                        float(np.linalg.norm(result[0:3]))
                    )
                    distance = float(contact.dist)
                    if min_dist is None or distance < min_dist:
                        min_dist = distance
                values: Dict[str, float] = {
                    "contact_point_force_max": max(magnitudes),
                    "contact_normal_force_max": max(normals),
                    "contact_normal_force_sum": sum(normals),
                    "contact_resultant_force": float(
                        np.linalg.norm(resultant)
                    ),
                    "contact_point_count": float(len(rows)),
                }
                if min_dist is not None:
                    values["contact_min_dist"] = min_dist
            else:
                values = {
                    "contact_point_force_max": 0.0,
                    "contact_normal_force_max": 0.0,
                    "contact_normal_force_sum": 0.0,
                    "contact_resultant_force": 0.0,
                    "contact_point_count": 0.0,
                }

            perceived[record.entity_id] = EntityState(
                entity_id=record.entity_id,
                entity_class=record.entity_class,
                position=None,
                velocity=None,
                track_age=self._timestep,
                track_state=TrackState.CONFIRMED,
                properties={},
            )
            for key, value in values.items():
                meta = self._CONTACT_PROPERTY_META[key]
                if record.tags:
                    meta = meta._replace(
                        tags=tuple(meta.tags) + tuple(
                            tag for tag in record.tags
                            if tag not in meta.tags
                        )
                    )
                emitted.append(
                    self._emit_property(
                        perceived,
                        record.entity_id,
                        key,
                        value,
                        meta,
                    )
                )

        return (True, sorted(set(filter(None, emitted))), None)

    def _validate_property_metadata_parity(
        self,
        perceived: Mapping[int, EntityState],
    ) -> None:
        previous_scope = self._metadata_scope_override
        self._metadata_scope_override = self._extraction_scope
        parity_passed = []
        temporary_classes = []
        try:
            for entity_id, entity in perceived.items():
                if entity_id in self._parity_validated:
                    continue
                if (
                    self._extraction_scope is not None
                    and entity_id not in self._entity_class_by_id
                ):
                    # The base commits identity only after this in-flight
                    # parity check; seed its class without committing scope.
                    self._entity_class_by_id[entity_id] = entity.entity_class
                    temporary_classes.append(entity_id)
                parity_passed.append(entity_id)
                property_keys = set(entity.properties)
                metadata_keys = set(self.property_metadata(entity_id))
                if property_keys != metadata_keys:
                    raise RuntimeError(
                        "Property metadata mismatch for entity {}: missing={} "
                        "unexpected={}".format(
                            entity_id,
                            sorted(property_keys - metadata_keys),
                            sorted(metadata_keys - property_keys),
                        )
                    )
        finally:
            self._metadata_scope_override = previous_scope
            for entity_id in temporary_classes:
                self._entity_class_by_id.pop(entity_id, None)
        # Committed only when every new entity passed — a rejected
        # observation must not mark its earlier entities validated.
        self._parity_validated.update(parity_passed)

    def _is_perceived(
        self,
        entity: EntityState,
    ) -> bool:
        if self.perception_gate is None:
            return True

        return bool(
            self.perception_gate(
                entity,
                self.env,
            )
        )

    # ------------------------------------------------------------------
    # Joint-property extraction
    # ------------------------------------------------------------------

    def _body_token(
        self,
        body_id: int,
    ) -> str:
        return self._source_property_token("body", body_id)

    def _own_pose_properties(
        self,
        body_id: int,
    ) -> Dict[str, float]:
        """World pose of the entity's own body, from forward kinematics."""
        token = self._body_token(body_id)

        properties: Dict[str, float] = {}

        positions = self._body_positions()
        orientations = self._body_orientations()

        for axis in range(3):
            properties[f"xpos_{token}_{axis}"] = float(
                positions[body_id][axis]
            )

        for index, component in enumerate(_QUAT_COMPONENTS):
            properties[f"xquat_{token}_{component}"] = float(
                orientations[body_id][index]
            )

        return properties

    def _own_pose_meta(
        self,
        body_id: int,
    ) -> Dict[str, PropertyMeta]:
        token = self._body_token(body_id)

        meta: Dict[str, PropertyMeta] = {}

        for axis in range(3):
            meta[f"xpos_{token}_{axis}"] = PropertyMeta(
                group="body position",
                note="m",
                family="xpos",
                component=("x", "y", "z")[axis] + " (axis {})".format(axis),
                frame="world",
            )

        for component in _QUAT_COMPONENTS:
            meta[f"xquat_{token}_{component}"] = PropertyMeta(
                group="body orientation",
                note=f"quat {component}",
                unit="1",
                family="xquat",
                component=component,
                frame="world",
            )

        return meta

    def _body_spatial_properties(
        self,
        body_id: int,
    ) -> Dict[str, float]:
        """
        One body's com-based 6-vectors: ``cvel``, ``cfrc_ext``,
        ``cfrc_int`` and ``cacc``, as ``<field>_<body>_0..5``.

        Every field is skipped independently when the live ``MjData``
        does not carry it, so an older or wrapped binding that exposes
        only some of them still produces a consistent key set --
        ``_body_spatial_meta`` applies the same guard, keeping
        ``property_metadata`` parity intact.
        """
        token = self._body_token(body_id)

        properties: Dict[str, float] = {}

        for field, _, _, _, _ in _BODY_SPATIAL_FIELDS:
            values = getattr(
                self._data,
                field,
                None,
            )

            if values is None:
                continue

            for index in range(6):
                properties[f"{field}_{token}_{index}"] = float(
                    values[body_id][index]
                )

        return properties

    def _body_spatial_meta(
        self,
        body_id: int,
    ) -> Dict[str, PropertyMeta]:
        token = self._body_token(body_id)

        meta: Dict[str, PropertyMeta] = {}

        for (
            field,
            rotational_group,
            rotational_note,
            linear_group,
            linear_note,
        ) in _BODY_SPATIAL_FIELDS:
            if getattr(self._data, field, None) is None:
                continue

            for index in range(6):
                meta[f"{field}_{token}_{index}"] = PropertyMeta(
                    group=(
                        rotational_group
                        if index < 3
                        else linear_group
                    ),
                    note=(
                        rotational_note
                        if index < 3
                        else linear_note
                    ),
                )

        return meta

    def _extract_dynamic_properties(
        self,
        joint_ids: Sequence[int],
    ) -> Dict[str, float]:
        """
        Extract configured MuJoCo joint signals as scalar properties.

        Scalar joints produce names such as:

            qpos_robot0_joint1
            qvel_robot0_joint1
            qfrc_applied_robot0_joint1

        Multi-coordinate joints add stable zero-based suffixes:

            qpos_object_free_joint_0
            qpos_object_free_joint_1
            ...
        """
        properties: Dict[str, float] = {}

        for joint_id in joint_ids:
            joint_name = self._indexed_source_name("joint", joint_id)

            token = self._joint_property_token(joint_id)

            for signal in self.joint_signals:
                if not hasattr(
                    self._data,
                    signal,
                ):
                    continue

                addr_attr, size_attr = _SIGNAL_ADDRESSES[signal]

                start, stop = self._joint_slice(
                    joint_id,
                    addr_attr,
                    size_attr,
                )

                self._add_values(
                    properties,
                    signal,
                    token,
                    getattr(
                        self._data,
                        signal,
                    )[start:stop],
                )

        return properties

    def property_metadata(
        self,
        entity_id: int,
    ) -> Dict[str, PropertyMeta]:
        """
        Display metadata for one entity's extracted properties.

        Joint-derived groups and units come from ``model.jnt_type`` —
        never from joint names. Per-body metadata — pose
        (``xpos_*``/``xquat_*``) and the com-based 6-vectors
        (``cvel_*``/``cfrc_ext_*``/``cfrc_int_*``/``cacc_*``) — for the
        entity's matched body, and for every body in its kinematic
        subtree when the matching rule has ``joint_scope="subtree"``, is
        merged in, along with metadata for
        any sensors and joint-transmission actuators attached to a body
        in that set; a gripper entity instead returns its own gripper
        metadata (including any sensor defined on its own site), a
        region entity returns metadata for its ``hw``/``hh``/``hd`` and
        ``x0``/``x1``/``y0``/``y1``/``z0``/``z1`` keys, and the
        synthetic orphan-sensor / orphan-actuator entities return
        metadata for sensors / actuators with no owner.
        """
        base: Dict[str, PropertyMeta] = {}

        for body in getattr(self, "_bodies", ()):
            if body.entity_id == entity_id:
                base = self._joint_property_meta(
                    body.body_id,
                    body.joint_ids,
                )
                for pose_body_id in body.pose_body_ids:
                    base.update(
                        self._own_pose_meta(
                            pose_body_id,
                        )
                    )
                    base.update(
                        self._body_spatial_meta(
                            pose_body_id,
                        )
                    )
                base.update(
                    self._sensor_property_meta(
                        self._owned_sensor_records(
                            entity_id,
                        )
                    )
                )
                base.update(
                    self._actuator_property_meta(
                        self._owned_actuator_records(
                            entity_id,
                        )
                    )
                )
                break

        if (
            getattr(self, "_gripper", None) is not None
            and self._gripper.entity_id == entity_id
        ):
            base = self._gripper_property_meta()

        for region in getattr(self, "_regions", ()):
            if region.entity_id == entity_id:
                base = self._region_property_meta()
                break

        if (
            getattr(self, "_orphan_sensor_entity_id", None) is not None
            and self._orphan_sensor_entity_id == entity_id
        ):
            base = self._sensor_property_meta(
                self._orphan_sensor_records()
            )

        if (
            getattr(self, "_orphan_actuator_entity_id", None) is not None
            and self._orphan_actuator_entity_id == entity_id
        ):
            base = self._actuator_property_meta(
                self._orphan_actuator_records()
            )

        for record in getattr(self, "_raw_sites", ()):
            if record.entity_id == entity_id:
                token = self._source_property_token(
                    "site", record.source_id
                )
                tags = (
                    "site_type:{}".format(
                        self._site_type_name(record.source_id)
                    ),
                ) + self._source_identity_tags(
                    "site", record.source_id
                )
                for index in range(
                    np.asarray(
                        self._model.site_size[record.source_id]
                    ).size
                ):
                    base[
                        "site_size_{}_{}".format(token, index)
                    ] = PropertyMeta(
                        group="site size", note="m", tags=tags
                    )
                break

        for record in getattr(self, "_raw_geoms", ()):
            if record.entity_id == entity_id:
                token = self._source_property_token(
                    "geom", record.source_id
                )
                tags = (
                    "geom_type:{}".format(
                        self._geom_type_name(record.source_id)
                    ),
                ) + self._source_identity_tags(
                    "geom", record.source_id
                )
                for index in range(
                    np.asarray(
                        self._model.geom_size[record.source_id]
                    ).size
                ):
                    base[
                        "geom_size_{}_{}".format(token, index)
                    ] = PropertyMeta(
                        group="geom size", note="m", tags=tags
                    )
                break

        for record in getattr(self, "_declared_entities", ()):
            if record.entity_id == entity_id:
                declared_meta = dict(
                    self._declared_entity_property_metadata(
                        record.spec.declaration_key
                    )
                )
                if not all(
                    isinstance(meta, PropertyMeta)
                    for meta in declared_meta.values()
                ):
                    raise TypeError(
                        "_declared_entity_property_metadata() must return "
                        "PropertyMeta values"
                    )
                base.update(declared_meta)
                break

        base.update(
            getattr(self, "_materialized_property_meta", {}).get(
                entity_id, {}
            )
        )
        scope = self._metadata_extraction_scope()
        if scope is not None:
            # Metadata describes what was extracted, so under a scope it
            # describes the projection. An entity this adapter has never
            # observed has no class to project through and keeps the
            # full model-derived schema.
            entity_class = self._entity_class_by_id.get(entity_id)
            if entity_class is not None:
                allowed = scope.get(entity_class, _NO_KEYS)
                base = {
                    key: meta
                    for key, meta in base.items()
                    if key in allowed
                }
        return base

    def _joint_subsystem_tags(
        self,
        joint_id: int,
    ) -> Tuple[str, ...]:
        """Structural tags for one joint's keys (subclass hook).

        The base adapter has no notion of robot subsystems, so joint
        keys carry no tags. Robosuite-derived adapters override this to
        classify joints from the robot's composition structure and
        return tags such as ``("arm", "right")`` or ``("base",)``.
        """
        return ()

    def _joint_property_meta(
        self,
        body_id: int,
        joint_ids: Sequence[int],
    ) -> Dict[str, PropertyMeta]:
        jnt_types = getattr(
            self._model,
            "jnt_type",
            None,
        )

        jnt_bodies = getattr(
            self._model,
            "jnt_bodyid",
            None,
        )

        if jnt_types is None or jnt_bodies is None:
            return {}

        metadata: Dict[str, PropertyMeta] = {}

        for joint_id in joint_ids:
            joint_name = self._indexed_source_name("joint", joint_id)

            token = self._joint_property_token(joint_id)

            joint_type = int(
                jnt_types[joint_id]
            )

            own_body = (
                int(jnt_bodies[joint_id])
                == int(body_id)
            )

            subsystem_tags = (
                tuple(self._joint_subsystem_tags(joint_id))
                + self._source_identity_tags("joint", joint_id)
            )

            for signal in self.joint_signals:
                address_name, total_name = (
                    _SIGNAL_ADDRESSES[signal]
                )

                start, stop = self._joint_slice(
                    joint_id,
                    address_name,
                    total_name,
                )

                keys = self._signal_keys(
                    signal,
                    token,
                    stop - start,
                )

                for index, key in enumerate(keys):
                    meta = self._joint_key_meta(
                        signal,
                        joint_type,
                        index,
                        own_body,
                    )

                    if subsystem_tags:
                        meta = meta._replace(
                            tags=subsystem_tags,
                        )

                    metadata[key] = meta

        return metadata

    _FORCE_GROUP_PREFIX = {
        "qfrc_actuator": "actuator",
        "qfrc_constraint": "constraint",
        "qfrc_bias": "bias",
        "qfrc_applied": "applied",
        "qfrc_smooth": "smooth",
        "qfrc_passive": "passive",
        "qfrc_spring": "spring",
        "qfrc_damper": "damper",
        "qfrc_fluid": "fluid",
        "qfrc_gravcomp": "gravcomp",
    }

    @staticmethod
    def _joint_key_meta(
        signal: str,
        joint_type: int,
        index: int,
        own_body: bool,
    ) -> PropertyMeta:
        free = joint_type == _MJ_JNT_FREE
        ball = joint_type == _MJ_JNT_BALL
        slide = joint_type == _MJ_JNT_SLIDE

        if signal in MujocoAdapter._FORCE_GROUP_PREFIX:
            prefix = MujocoAdapter._FORCE_GROUP_PREFIX[signal]

            # Rotational DOF take torques: a hinge or ball entirely, a
            # free joint in its last three components.
            rotational = (
                ball
                or (free and index >= 3)
                or not (free or slide)
            )

            if rotational:
                return PropertyMeta(
                    group=f"{prefix} torques",
                    note="N·m",
                )

            return PropertyMeta(
                group=f"{prefix} forces",
                note="N",
            )

        if signal == "qpos":
            if free and index < 3:
                return PropertyMeta(
                    group="joint position",
                    note="m",
                )

            if free:
                return PropertyMeta(
                    group="joint orientation",
                    note=(
                        "quat "
                        + _QUAT_COMPONENTS[index - 3]
                    ),
                )

            if ball:
                return PropertyMeta(
                    group="joint orientation",
                    note=(
                        "quat "
                        + _QUAT_COMPONENTS[index]
                    ),
                )

            # A slide/hinge on the matched body itself is a pose
            # coordinate of that body relative to its spawn frame
            # (cheetah root joints, robosuite virtual mobile-base
            # joints) — not articulation.
            if own_body:
                if slide:
                    return PropertyMeta(
                        group="base displacement (rel. spawn)",
                        note="m",
                    )

                return PropertyMeta(
                    group="base angle (rel. spawn)",
                    note="rad",
                )

            if slide:
                return PropertyMeta(
                    group="joint displacement",
                    note="m",
                )

            return PropertyMeta(
                group="joint angle",
                note="rad",
            )

        if signal in ("qacc", "qacc_smooth"):
            if free and index < 3:
                return PropertyMeta(
                    group="joint linear acceleration",
                    note="m/s²",
                )

            if free or ball:
                return PropertyMeta(
                    group="joint angular acceleration",
                    note="rad/s²",
                )

            if own_body:
                if slide:
                    return PropertyMeta(
                        group="base linear acceleration",
                        note="m/s²",
                    )

                return PropertyMeta(
                    group="base angular acceleration",
                    note="rad/s²",
                )

            if slide:
                return PropertyMeta(
                    group="joint linear acceleration",
                    note="m/s²",
                )

            return PropertyMeta(
                group="joint angular acceleration",
                note="rad/s²",
            )

        if free and index < 3:
            return PropertyMeta(
                group="joint linear velocity",
                note="m/s",
            )

        if free or ball:
            return PropertyMeta(
                group="joint angular velocity",
                note="rad/s",
            )

        if own_body:
            if slide:
                return PropertyMeta(
                    group="base linear velocity",
                    note="m/s",
                )

            return PropertyMeta(
                group="base angular velocity",
                note="rad/s",
            )

        if slide:
            return PropertyMeta(
                group="joint linear velocity",
                note="m/s",
            )

        return PropertyMeta(
            group="joint angular velocity",
            note="rad/s",
        )

    def _subtree_body_ids(
        self,
        body_id: int,
    ) -> Tuple[int, ...]:
        """Bodies in body_id's kinematic subtree, own body first."""
        body_ids = [body_id]

        parent_ids = np.asarray(
            self._model.body_parentid,
            dtype=int,
        )

        for candidate in range(
            int(self._model.nbody)
        ):
            current = candidate

            while (
                current > 0
                and current != body_id
            ):
                current = int(
                    parent_ids[current]
                )

            if (
                current == body_id
                and candidate != body_id
            ):
                body_ids.append(candidate)

        return tuple(body_ids)

    def _resolve_pose_body_ids(
        self,
        body_id: int,
        joint_scope: str,
    ) -> Tuple[int, ...]:
        """
        Compute one ``_BodyRecord``'s collision-free ``pose_body_ids``.

        Model names that normalize to the same parser token receive
        compiled-index suffixes from ``_source_property_tokens``.
        """
        pose_body_ids = (
            self._subtree_body_ids(body_id)
            if joint_scope == "subtree"
            else (body_id,)
        )

        claimed: Dict[str, str] = {}

        for pose_body_id in pose_body_ids:
            name = self._indexed_source_name("body", pose_body_id)
            token = self._body_token(pose_body_id)

            if token in claimed:
                raise ValueError(
                    f"Pose-body property name collision: bodies "
                    f"{claimed[token]!r} and {name!r} both tokenize to "
                    f"{token!r} (e.g. xpos_{token}_0, cvel_{token}_0)"
                )

            claimed[token] = name

        return pose_body_ids

    def _joint_ids(
        self,
        body_id: int,
        scope: str,
    ) -> Tuple[int, ...]:
        """Resolve the joints included by a BodyRule joint scope."""
        if scope == "none":
            return ()

        if scope == "subtree":
            body_ids = self._subtree_body_ids(body_id)
        else:
            body_ids = (body_id,)

        joint_ids: List[int] = []

        for current_body in body_ids:
            joint_count = int(
                self._model.body_jntnum[
                    current_body
                ]
            )

            first_joint = int(
                self._model.body_jntadr[
                    current_body
                ]
            )

            if (
                joint_count > 0
                and first_joint >= 0
            ):
                joint_ids.extend(
                    range(
                        first_joint,
                        first_joint + joint_count,
                    )
                )

        return tuple(joint_ids)

    def _joint_slice(
        self,
        joint_id: int,
        address_name: str,
        total_name: str,
    ) -> Tuple[int, int]:
        addresses = getattr(
            self._model,
            address_name,
        )

        start = int(
            addresses[joint_id]
        )

        if (
            joint_id + 1
            < int(self._model.njnt)
        ):
            stop = int(
                addresses[joint_id + 1]
            )
        else:
            stop = int(
                getattr(
                    self._model,
                    total_name,
                )
            )

        if stop < start:
            raise RuntimeError(
                f"Invalid {address_name} range for "
                f"joint {joint_id}: [{start}, {stop})"
            )

        return start, stop

    @staticmethod
    def _add_values(
        destination: Dict[str, float],
        signal: str,
        joint_token: str,
        values: Any,
    ) -> None:
        flat = np.asarray(
            values,
            dtype=float,
        ).reshape(-1)

        keys = MujocoAdapter._signal_keys(
            signal,
            joint_token,
            int(flat.size),
        )

        for key, value in zip(keys, flat):
            if key in destination:
                raise ValueError(
                    f"Dynamic property name collision: "
                    f"{key!r}"
                )

            destination[key] = float(value)

    @staticmethod
    def _signal_keys(
        signal: str,
        joint_token: str,
        size: int,
    ) -> List[str]:
        """Property key names for one signal of one joint, in index order."""
        if size == 1:
            return [f"{signal}_{joint_token}"]

        return [
            f"{signal}_{joint_token}_{index}"
            for index in range(size)
        ]

    # ------------------------------------------------------------------
    # Sensor extraction
    # ------------------------------------------------------------------

    def _register_sensors(
        self,
        next_id: int,
    ) -> int:
        """
        Enumerate ``model`` sensors once and resolve each one's owning
        entity.

        A sensor's object is resolved to a body id, and the owner is the
        registered ``_BodyRecord`` whose ``pose_body_ids`` contains that
        body id (first match in registration order) -- except a sensor
        defined directly on the gripper's own site, which belongs to the
        gripper entity regardless of which body's subtree that site's
        body falls under (see ``_sensor_owner_entity_id``). Sensors that
        resolve to no owner are collected under one synthetic "sensors"
        entity, whose id is reserved here -- after body/gripper/region
        ids -- so that ids stay stable across resets when no orphans
        exist.
        """
        sensor_count = int(
            getattr(
                self._model,
                "nsensor",
                0,
            )
            or 0
        )

        if sensor_count == 0:
            return next_id

        try:
            import mujoco
        except ImportError:
            return next_id

        has_orphan = False

        for sensor_id in range(sensor_count):
            owner_entity_id = self._sensor_owner_entity_id(
                sensor_id,
                mujoco,
            )

            adr = int(
                self._model.sensor_adr[sensor_id]
            )

            dim = int(
                self._model.sensor_dim[sensor_id]
            )

            type_name = (
                mujoco.mjtSensor(
                    int(
                        self._model.sensor_type[
                            sensor_id
                        ]
                    )
                )
                .name.replace(
                    "mjSENS_",
                    "",
                    1,
                )
                .lower()
            )

            name = self._id2name("sensor", sensor_id)
            if name is None:
                name = self._indexed_source_name("sensor", sensor_id)

            self._sensors.append(
                _SensorRecord(
                    sensor_id=sensor_id,
                    name=name,
                    name_token=self._source_property_token(
                        "sensor", sensor_id
                    ),
                    adr=adr,
                    dim=dim,
                    type_name=type_name,
                    unit_note=self._sensor_unit_note(
                        sensor_id, type_name, mujoco
                    ),
                    owner_entity_id=owner_entity_id,
                )
            )

            if owner_entity_id is None:
                has_orphan = True

        if has_orphan:
            self._orphan_sensor_entity_id = next_id
            next_id += 1

        return next_id

    def _sensor_unit_note(
        self,
        sensor_id: int,
        type_name: str,
        mujoco_module: Any,
    ) -> Optional[str]:
        """Resolve one sensor's physical unit without using its name."""
        direct = {
            **_SENSOR_UNIT_NOTES,
            "magnetometer": "T",
            "rangefinder": "m",
            "ballquat": "1",
            "ballangvel": "rad/s",
            "framepos": "m",
            "framequat": "1",
            "framexaxis": "1",
            "frameyaxis": "1",
            "framezaxis": "1",
            "framelinvel": "m/s",
            "frameangvel": "rad/s",
            "framelinacc": "m/s²",
            "frameangacc": "rad/s²",
            "subtreecom": "m",
            "subtreelinvel": "m/s",
            "subtreeangmom": "kg·m²/s",
            "clock": "s",
            "e_potential": "J",
            "e_kinetic": "J",
            "geomdist": "m",
            "geomnormal": "1",
            "geomfromto": "m",
            "camprojection": "px",
        }
        if type_name in direct:
            return direct[type_name]

        object_id = int(self._model.sensor_objid[sensor_id])
        joint_position_types = {
            "jointpos", "jointlimitpos",
        }
        joint_velocity_types = {
            "jointvel", "jointlimitvel",
        }
        joint_force_types = {
            "jointactfrc", "jointlimitfrc",
        }
        if type_name in (
            joint_position_types
            | joint_velocity_types
            | joint_force_types
        ):
            joint_type = int(self._model.jnt_type[object_id])
            if joint_type == _MJ_JNT_SLIDE:
                base = "m"
                force = "N"
            elif joint_type == _MJ_JNT_HINGE:
                base = "rad"
                force = "N·m"
            else:
                return None
            if type_name in joint_position_types:
                return base
            if type_name in joint_velocity_types:
                return "{}/s".format(base)
            return force

        if type_name in {
            "tendonpos",
            "tendonvel",
            "tendonlimitpos",
            "tendonlimitvel",
            "tendonlimitfrc",
        }:
            name = (
                self._id2name("tendon", object_id)
                or "tendon[{}]".format(object_id)
            )
            record = _NamedOwnerRecord(
                entity_id=-1,
                source_name=name,
                source_id=object_id,
                entity_class="tendon",
            )
            coordinate, _ = self._tendon_unit(
                record,
                "ten_velocity"
                if type_name in {"tendonvel", "tendonlimitvel"}
                else "ten_length",
            )
            if coordinate is None:
                return None
            if type_name == "tendonlimitfrc":
                return "N" if coordinate == "m" else "N·m"
            return coordinate

        if type_name in {
            "actuatorpos", "actuatorvel", "actuatorfrc",
        }:
            return self._actuator_transmission_unit(
                object_id, type_name
            )

        # User and plugin sensors deliberately carry no engine-defined unit.
        del mujoco_module
        return None

    def _actuator_transmission_unit(
        self,
        actuator_id: int,
        quantity: str,
    ) -> Optional[str]:
        import mujoco

        if int(self._model.actuator_trntype[actuator_id]) != int(
            mujoco.mjtTrn.mjTRN_JOINT
        ):
            return None
        joint_id = int(self._model.actuator_trnid[actuator_id][0])
        joint_type = int(self._model.jnt_type[joint_id])
        if joint_type == _MJ_JNT_SLIDE:
            return {
                "actuatorpos": "m",
                "actuatorvel": "m/s",
                "actuatorfrc": "N",
            }[quantity]
        if joint_type == _MJ_JNT_HINGE:
            return {
                "actuatorpos": "rad",
                "actuatorvel": "rad/s",
                "actuatorfrc": "N·m",
            }[quantity]
        return None

    def _sensor_owner_entity_id(
        self,
        sensor_id: int,
        mujoco_module: Any,
    ) -> Optional[int]:
        """
        Resolve a sensor's object to a body id, then to an owning entity.

        A sensor defined directly on the gripper's own site belongs to
        the gripper entity. This exact-site match is checked before the
        general body walk below, because the gripper's site may sit on a
        body that also falls inside a subtree-scoped robot body's
        ``pose_body_ids`` -- without this precedence, such a sensor would
        attach to the robot (or, if the gripper site's body matches no
        registered body, be treated as an orphan) instead of the gripper
        that actually owns it.
        """
        object_type = int(
            self._model.sensor_objtype[sensor_id]
        )

        object_id = int(
            self._model.sensor_objid[sensor_id]
        )

        if (
            self._gripper is not None
            and object_type == int(
                mujoco_module.mjtObj.mjOBJ_SITE
            )
            and object_id == self._gripper.site_id
        ):
            return self._gripper.entity_id

        if object_type == int(
            mujoco_module.mjtObj.mjOBJ_SITE
        ):
            body_id = int(
                self._model.site_bodyid[object_id]
            )
        elif object_type in (
            int(mujoco_module.mjtObj.mjOBJ_BODY),
            int(mujoco_module.mjtObj.mjOBJ_XBODY),
        ):
            body_id = object_id
        elif object_type == int(
            mujoco_module.mjtObj.mjOBJ_JOINT
        ):
            body_id = int(
                self._model.jnt_bodyid[object_id]
            )
        else:
            return None

        for body in self._bodies:
            if body_id in body.pose_body_ids:
                return body.entity_id

        return None

    def _owned_sensor_records(
        self,
        entity_id: int,
    ) -> List[_SensorRecord]:
        # getattr guards callers that reach this hook on a bare adapter
        # built with object.__new__ (see tests/test_gripper_property_meta.py),
        # where __init__ -- and so self._sensors -- never ran.
        return [
            record
            for record in getattr(self, "_sensors", None) or []
            if record.owner_entity_id == entity_id
        ]

    def _orphan_sensor_records(self) -> List[_SensorRecord]:
        return [
            record
            for record in getattr(self, "_sensors", None) or []
            if record.owner_entity_id is None
        ]

    def _sensor_properties(
        self,
        records: Sequence[_SensorRecord],
    ) -> Dict[str, float]:
        properties: Dict[str, float] = {}
        claimed: Dict[str, str] = {}

        for record in records:
            values = self._data.sensordata[
                record.adr : record.adr + record.dim
            ]

            for key, value in zip(
                self._sensor_keys(record),
                values,
            ):
                self._claim_sensor_key(key, record, claimed)
                properties[key] = float(value)

        return properties

    def _sensor_property_meta(
        self,
        records: Sequence[_SensorRecord],
    ) -> Dict[str, PropertyMeta]:
        meta: Dict[str, PropertyMeta] = {}
        claimed: Dict[str, str] = {}

        for record in records:
            for key in self._sensor_keys(record):
                self._claim_sensor_key(key, record, claimed)
                meta[key] = PropertyMeta(
                    group=(
                        f"sensors: {record.type_name}"
                        if record.unit_note is not None
                        else f"sensor: {record.name_token} (opaque)"
                    ),
                    note=record.unit_note,
                    tags=(
                        self._source_identity_tags(
                            "sensor", record.sensor_id
                        )
                        + (f"sensor_type:{record.type_name}",)
                        + self._sensor_target_tags(record.sensor_id)
                        + (
                            ()
                            if record.unit_note is not None
                            else ("semantics:opaque", "unit:unresolved")
                        )
                    ),
                )

        return meta

    @staticmethod
    def _claim_sensor_key(
        key: str,
        record: "_SensorRecord",
        claimed: Dict[str, str],
    ) -> None:
        """
        Raise if two sensors would collide on the same property key.

        ``_property_token`` sanitizes punctuation to underscores, so
        distinctly named sensors (for example ``a.b`` and ``a_b``) can
        tokenize to the same key. Silently overwriting one sensor's
        value with another's would drop data with no signal that it
        happened; this mirrors the collision guard ``_add_values``
        applies to joint-signal keys.
        """
        if key in claimed:
            raise ValueError(
                f"Sensor property name collision: sensors "
                f"{claimed[key]!r} and {record.name!r} both produce "
                f"property key {key!r}"
            )

        claimed[key] = record.name

    @staticmethod
    def _sensor_keys(
        record: _SensorRecord,
    ) -> List[str]:
        """Property key names for one sensor, in index order."""
        if record.dim == 1:
            return [f"sensordata_{record.name_token}"]

        return [
            f"sensordata_{record.name_token}_{index}"
            for index in range(record.dim)
        ]

    # ------------------------------------------------------------------
    # Actuator extraction
    # ------------------------------------------------------------------

    def _register_actuators(
        self,
        next_id: int,
    ) -> int:
        """
        Enumerate ``model`` actuators once and resolve each one's owning
        entity.

        A joint-transmission actuator's joint is resolved to a body id,
        and the owner is the registered ``_BodyRecord`` whose
        ``pose_body_ids`` contains that body id (first match in
        registration order) -- the same body walk sensors use
        (``_sensor_owner_entity_id``), except actuators have no
        gripper-site precedence: an actuator's transmission is a joint,
        never a site, so it can never resolve to the gripper's site-based
        entity. Non-joint-transmission actuators (tendon, site,
        slider-crank, ...), and joint-transmission actuators whose
        joint's body matches no registered body, are collected under one
        synthetic "actuators" entity, whose id is reserved here -- after
        body/gripper/region/sensor ids -- so that ids stay stable across
        resets when no orphans exist.
        """
        actuator_count = int(
            getattr(
                self._model,
                "nu",
                0,
            )
            or 0
        )

        if actuator_count == 0:
            return next_id

        try:
            import mujoco
        except ImportError:
            return next_id

        has_orphan = False

        for actuator_id in range(actuator_count):
            transmission_type = int(
                self._model.actuator_trntype[
                    actuator_id
                ]
            )
            is_joint_transmission = transmission_type == int(
                mujoco.mjtTrn.mjTRN_JOINT
            )
            is_tendon_transmission = transmission_type == int(
                mujoco.mjtTrn.mjTRN_TENDON
            )

            joint_id = (
                int(
                    self._model.actuator_trnid[
                        actuator_id
                    ][0]
                )
                if is_joint_transmission
                else None
            )

            owner_entity_id = self._actuator_owner_entity_id(
                joint_id,
            )

            force_unit_note = None
            length_unit_note = None
            velocity_unit_note = None

            if joint_id is not None:
                joint_type = int(
                    self._model.jnt_type[joint_id]
                )

                slide = joint_type == _MJ_JNT_SLIDE

                # Units come from the transmitted joint only when that
                # joint is *scalar* (slide or hinge), where the actuator
                # coordinate, rate and output really are the joint's own.
                #
                # Free and ball joints are neither. MuJoCo projects the
                # multi-DOF joint through the actuator's six-component
                # gear vector, so a translation gear on a free joint
                # produces a force in N and a rate in m/s where the joint
                # type alone would claim N·m and rad/s, a mixed gear has
                # no single unit at all, and actuator_length stays 0
                # whatever the joint's position (measured: a freejoint
                # motor with gear="1 0 0 0 0 0" reads length 0.0 at qpos
                # 0.37 and velocity 0.25). All three are left unlabelled
                # rather than mislabelled.
                if joint_type in (_MJ_JNT_SLIDE, _MJ_JNT_HINGE):
                    force_unit_note = "N" if slide else "N·m"
                    length_unit_note = "m" if slide else "rad"
                    velocity_unit_note = "m/s" if slide else "rad/s"
            elif is_tendon_transmission:
                tendon_id = int(
                    self._model.actuator_trnid[actuator_id][0]
                )
                tendon_unit, _failure = self._tendon_coordinate_unit(
                    tendon_id
                )
                if tendon_unit is not None:
                    length_unit_note = tendon_unit
                    velocity_unit_note = "{}/s".format(tendon_unit)
                    force_unit_note = (
                        "N" if tendon_unit == "m" else "N·m"
                    )

            ctrllimited = bool(
                self._model.actuator_ctrllimited[
                    actuator_id
                ]
            )

            ctrlrange = (
                float(
                    self._model.actuator_ctrlrange[
                        actuator_id
                    ][0]
                ),
                float(
                    self._model.actuator_ctrlrange[
                        actuator_id
                    ][1]
                ),
            )

            name = self._id2name("actuator", actuator_id)
            if name is None:
                name = self._indexed_source_name(
                    "actuator", actuator_id
                )

            self._actuators.append(
                _ActuatorRecord(
                    actuator_id=actuator_id,
                    name=name,
                    name_token=self._source_property_token(
                        "actuator", actuator_id
                    ),
                    ctrllimited=ctrllimited,
                    ctrlrange=ctrlrange,
                    force_unit_note=force_unit_note,
                    owner_entity_id=owner_entity_id,
                    length_unit_note=length_unit_note,
                    velocity_unit_note=velocity_unit_note,
                    ctrl_unit_note=self._ctrl_unit_note(
                        actuator_id,
                        force_unit_note,
                        length_unit_note,
                        velocity_unit_note,
                    ),
                )
            )

            if owner_entity_id is None:
                has_orphan = True

        if has_orphan:
            self._orphan_actuator_entity_id = next_id
            next_id += 1

        return next_id

    def _actuator_owner_entity_id(
        self,
        joint_id: Optional[int],
    ) -> Optional[int]:
        """
        Resolve a joint-transmission actuator's joint to a body id, then
        to an owning entity.

        ``joint_id`` is None for non-joint transmissions, which carry no
        body to walk and are always orphans.
        """
        if joint_id is None:
            return None

        body_id = int(
            self._model.jnt_bodyid[joint_id]
        )

        for body in self._bodies:
            if body_id in body.pose_body_ids:
                return body.entity_id

        return None

    def _owned_actuator_records(
        self,
        entity_id: int,
    ) -> List[_ActuatorRecord]:
        # getattr guards callers that reach this hook on a bare adapter
        # built with object.__new__ (see tests/test_gripper_property_meta.py),
        # where __init__ -- and so self._actuators -- never ran.
        return [
            record
            for record in getattr(self, "_actuators", None) or []
            if record.owner_entity_id == entity_id
        ]

    def _orphan_actuator_records(self) -> List[_ActuatorRecord]:
        return [
            record
            for record in getattr(self, "_actuators", None) or []
            if record.owner_entity_id is None
        ]

    def _actuator_properties(
        self,
        records: Sequence[_ActuatorRecord],
    ) -> Dict[str, float]:
        properties: Dict[str, float] = {}
        claimed: Dict[str, str] = {}

        for record in records:
            for key, field, _, _ in self._actuator_key_fields(record):
                values = getattr(
                    self._data,
                    field,
                    None,
                )

                if values is None:
                    continue

                self._claim_actuator_key(key, record, claimed)
                properties[key] = float(
                    values[record.actuator_id]
                )

        return properties

    def _actuator_property_meta(
        self,
        records: Sequence[_ActuatorRecord],
    ) -> Dict[str, PropertyMeta]:
        meta: Dict[str, PropertyMeta] = {}
        claimed: Dict[str, str] = {}

        for record in records:
            for key, field, group, note in self._actuator_key_fields(record):
                if getattr(self._data, field, None) is None:
                    continue

                self._claim_actuator_key(key, record, claimed)
                unit = self._actuator_field_unit(record, field)
                meta[key] = PropertyMeta(
                    group=group,
                    note=note,
                    tags=(
                        self._source_identity_tags(
                            "actuator", record.actuator_id
                        )
                        + self._actuator_transmission_tags(
                            record.actuator_id
                        )
                        + (
                            ()
                            if unit is not None
                            else ("semantics:opaque", "unit:unresolved")
                        )
                    ),
                )

        return meta

    def _actuator_key_fields(
        self,
        record: _ActuatorRecord,
    ) -> Tuple[Tuple[str, str, str, Optional[str]], ...]:
        """
        ``(property key, MjData field, group, unit note)`` per actuator.

        The group follows from the unit, so a family spanning two units
        on one model (a slide actuator beside a hinge one) lands in two
        unit-homogeneous groups rather than one mixed group. ``ctrl``
        follows the same rule via ``_CTRL_GROUPS``, keyed by the units
        the command is expressed in rather than by its note, which is the
        control range.

        ``act``/``act_dot`` are absent *here* because they hold
        activation state only stateful actuators (muscles, filtered
        actuators) have; on models with ``model.na > 0`` they are
        materialized separately by ``_materialize_actuator_activation``
        with metadata from ``_activation_meta``. See issue #73.
        """
        ctrl_key, force_key = self._actuator_keys(record)

        fields = (
            (force_key, "actuator_force", record.force_unit_note),
            (
                f"actuator_length_{record.name_token}",
                "actuator_length",
                record.length_unit_note,
            ),
            (
                f"actuator_velocity_{record.name_token}",
                "actuator_velocity",
                record.velocity_unit_note,
            ),
        )

        return (
            (
                ctrl_key,
                "ctrl",
                self._ctrl_group(record),
                self._ctrl_note(record),
            ),
        ) + tuple(
            (
                key,
                field,
                (
                    _ACTUATOR_GROUPS[field][note]
                    if note is not None
                    else "{} ({}; opaque)".format(
                        _ACTUATOR_GROUPS[field][None],
                        record.name_token,
                    )
                ),
                note,
            )
            for key, field, note in fields
        )

    def _ctrl_unit_note(
        self,
        actuator_id: int,
        force_unit_note: Optional[str],
        length_unit_note: Optional[str],
        velocity_unit_note: Optional[str],
    ) -> Optional[str]:
        """
        Units the actuator's ``ctrl`` command is expressed in.

        MuJoCo encodes actuator kind in the gain/bias parameters rather
        than in a type field, and the kind decides what ``ctrl`` means:

        - ``mjBIAS_NONE`` — a motor. ``ctrl`` is generalized force, so it
          carries the transmitted joint's force units.
        - ``mjBIAS_AFFINE`` with ``biasprm[1] == -gainprm[0]`` — a
          position servo. ``ctrl`` is a target position, so it carries
          the joint's length units.
        - ``mjBIAS_AFFINE`` with ``biasprm[2] == -gainprm[0]`` and no
          position term — a velocity servo, carrying rate units.
        - ``mjDYN_INTEGRATOR`` with a position-servo bias — an integrated
          velocity actuator. ``ctrl`` is the position setpoint's rate, so it
          also carries velocity units.
        - ``mjGAIN_AFFINE`` with velocity gain and no bias — a damper.
          ``ctrl`` is a dimensionless multiplier on the authored damping
          coefficient.

        Verified against LIBERO 2026-07-27: `robot0_torq_j*` report
        gaintype 0 / biastype 0 with gainprm[0] = 1 (motors, ctrl in
        N·m), while `gripper0_gripper_finger_joint*` report biastype 1
        with gainprm[0] = 1e3 and biasprm[1] = -1e3 (position servos,
        ctrl in m over range [0, 0.04]).

        Anything else — muscle, user, or a general law matching neither
        pattern — returns ``None`` rather than a guess. Its command receives
        an actuator-specific group so unknown meanings are never aggregated
        together. The same applies when the transmitted joint is free or
        ball, where the caller has already left every unit note ``None``.
        """
        try:
            dyntype = int(self._model.actuator_dyntype[actuator_id])
            gaintype = int(self._model.actuator_gaintype[actuator_id])
            biastype = int(self._model.actuator_biastype[actuator_id])
            gainprm = self._model.actuator_gainprm[actuator_id]
            biasprm = self._model.actuator_biasprm[actuator_id]
            kp = float(gainprm[0])
            gain_pos = float(gainprm[1])
            gain_vel = float(gainprm[2])
            bias_pos = float(biasprm[1])
            bias_vel = float(biasprm[2])
        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
        ):
            return None

        gain_tol = max(
            abs(kp),
            abs(gain_pos),
            abs(gain_vel),
        ) * 1e-6

        # A damper is affine in actuator velocity, multiplied by a
        # dimensionless nonnegative ctrl. Its NONE bias otherwise looks like
        # a motor, so gaintype and the affine coefficients are decisive.
        if (
            dyntype == _MJ_DYN_NONE
            and gaintype == _MJ_GAIN_AFFINE
            and biastype == _MJ_BIAS_NONE
            and abs(kp) <= gain_tol
            and abs(gain_pos) <= gain_tol
            and gain_vel < -gain_tol
        ):
            return "1"

        if gaintype != _MJ_GAIN_FIXED:
            return None

        # Zero fixed gain erases the distinction between position, velocity,
        # and a general affine law: MuJoCo compiles them to identical zero
        # gain/bias arrays. Do not infer an authored kind that the live model
        # no longer represents.
        if kp == 0.0:
            return None

        if biastype == _MJ_BIAS_NONE and dyntype == _MJ_DYN_NONE:
            return force_unit_note

        if biastype != _MJ_BIAS_AFFINE:
            return None

        # A servo's feedback term negates the gain. Compare against the
        # gain rather than against zero so a stiff and a soft servo are
        # recognised alike.
        position_tol = max(
            abs(bias_pos),
            abs(kp),
        ) * 1e-6
        velocity_tol = max(
            abs(bias_vel),
            abs(kp),
        ) * 1e-6
        if abs(bias_pos + kp) <= position_tol:
            if dyntype == _MJ_DYN_INTEGRATOR:
                return velocity_unit_note
            if dyntype in (
                _MJ_DYN_NONE,
                _MJ_DYN_FILTER,
                _MJ_DYN_FILTEREXACT,
            ):
                return length_unit_note
            return None
        if (
            dyntype == _MJ_DYN_NONE
            and abs(bias_vel + kp) <= velocity_tol
            and abs(bias_pos) <= velocity_tol
        ):
            return velocity_unit_note
        return None

    @staticmethod
    def _ctrl_group(record: "_ActuatorRecord") -> str:
        """Unit-homogeneous selector group for one actuator command."""
        unit = record.ctrl_unit_note
        if unit is not None:
            return _CTRL_GROUPS[unit]

        # Unknown controls cannot safely share a selector group: their units
        # and meanings may differ. One group per actuator keeps the key
        # discoverable without inviting cross-actuator aggregation.
        return f"actuator command ({record.name_token})"

    @staticmethod
    def _ctrl_note(
        record: "_ActuatorRecord",
    ) -> Optional[str]:
        if not record.ctrllimited:
            return None

        lo, hi = record.ctrlrange

        return f"ctrl ∈ [{lo:g}, {hi:g}]"

    @staticmethod
    def _claim_actuator_key(
        key: str,
        record: "_ActuatorRecord",
        claimed: Dict[str, str],
    ) -> None:
        """
        Raise if two actuators would collide on the same property key.

        ``_property_token`` sanitizes punctuation to underscores, so
        distinctly named actuators (for example ``a.b`` and ``a_b``) can
        tokenize to the same key. Silently overwriting one actuator's
        value with another's would drop data with no signal that it
        happened; this mirrors the collision guard
        ``_claim_sensor_key`` applies to sensor-data keys.
        """
        if key in claimed:
            raise ValueError(
                f"Actuator property name collision: actuators "
                f"{claimed[key]!r} and {record.name!r} both produce "
                f"property key {key!r}"
            )

        claimed[key] = record.name

    @staticmethod
    def _actuator_keys(
        record: "_ActuatorRecord",
    ) -> Tuple[str, str]:
        """``(ctrl_key, actuator_force_key)`` for one actuator."""
        return (
            f"ctrl_{record.name_token}",
            f"actuator_force_{record.name_token}",
        )

    # ------------------------------------------------------------------
    # Velocity extraction
    # ------------------------------------------------------------------

    def _body_velocity(
        self,
        body: _BodyRecord,
        position: np.ndarray,
    ) -> np.ndarray:
        if self._native_snapshot is not None:
            native_type = self._native_projection._mujoco.mjtObj.mjOBJ_XBODY
            return self._native_projection.object_linear_velocity(
                self._native_snapshot, native_type, body.body_id)
        getter = getattr(
            self._data,
            "get_body_xvelp",
            None,
        )

        if callable(getter):
            try:
                velocity = np.asarray(
                    getter(body.body_name),
                    dtype=float,
                ).copy()

                self._previous_positions[
                    body.entity_id
                ] = position.copy()

                return velocity
            except Exception:
                pass

        return self._finite_difference_velocity(
            body.entity_id,
            position,
        )

    def _site_velocity(
        self,
        site: _SiteRecord,
        position: np.ndarray,
    ) -> np.ndarray:
        if self._native_snapshot is not None:
            native_type = self._native_projection._mujoco.mjtObj.mjOBJ_SITE
            return self._native_projection.object_linear_velocity(
                self._native_snapshot, native_type, site.site_id)
        getter = getattr(
            self._data,
            "get_site_xvelp",
            None,
        )

        if callable(getter):
            try:
                velocity = np.asarray(
                    getter(site.site_name),
                    dtype=float,
                ).copy()

                self._previous_positions[
                    site.entity_id
                ] = position.copy()

                return velocity
            except Exception:
                pass

        return self._finite_difference_velocity(
            site.entity_id,
            position,
        )

    def _finite_difference_velocity(
        self,
        entity_id: int,
        position: np.ndarray,
    ) -> np.ndarray:
        previous = self._previous_positions.get(
            entity_id
        )

        self._previous_positions[
            entity_id
        ] = position.copy()

        if previous is None:
            return np.zeros(
                self.spatial_dims,
                dtype=float,
            )

        return (
            position - previous
        ) / self._dt

    # ------------------------------------------------------------------
    # Region extraction
    # ------------------------------------------------------------------

    # (low_name, high_name, axis) for each world-frame bound pair a
    # region entity may expose. Shared by ``_build_region_state`` (which
    # skips axes beyond ``self.spatial_dims``) and
    # ``_region_property_meta``, so the metadata keys and the actual
    # property keys can never drift apart.
    _REGION_BOUND_AXES: Tuple[
        Tuple[str, str, int], ...
    ] = (
        ("x0", "x1", 0),
        ("y0", "y1", 1),
        ("z0", "z1", 2),
    )

    def _build_region_state(
        self,
        record: _RegionRecord,
    ) -> EntityState:
        (
            position,
            half_extents,
            dynamic,
        ) = self._region_geometry(
            record.spec
        )

        if dynamic:
            velocity = (
                self._finite_difference_velocity(
                    record.entity_id,
                    position,
                )
            )
        else:
            self._previous_positions[
                record.entity_id
            ] = position.copy()

            velocity = np.zeros(
                self.spatial_dims,
                dtype=float,
            )

        properties = dict(
            record.spec.properties
        )

        reserved = (
            {"hw", "hh", "hd"}
            & set(properties)
        )

        if reserved:
            raise ValueError(
                f"RegionSpec {record.spec.name!r} "
                f"cannot redefine reserved properties "
                f"{sorted(reserved)}"
            )

        properties.update(
            hw=float(half_extents[0]),
            hh=float(half_extents[1]),
            hd=float(half_extents[2]),
        )

        # Materialize world-frame box bounds directly, following the
        # component-model contract that derived spatial values are adapter
        # properties rather than parser-side special cases. Axes beyond
        # the adapter's spatial dimensionality are omitted. The axis
        # tuple is shared with ``_region_property_meta`` so the metadata
        # keys and the actual property keys can never drift apart.
        for low_name, high_name, axis in self._REGION_BOUND_AXES:
            if axis >= self.spatial_dims:
                break
            half = float(half_extents[axis])
            center = float(position[axis])
            properties[low_name]  = center - half
            properties[high_name] = center + half

        return EntityState(
            entity_id=record.entity_id,
            entity_class=record.spec.cls,
            position=position,
            velocity=velocity,
            track_age=self._timestep,
            track_state=TrackState.CONFIRMED,
            properties=properties,
        )

    def _region_property_meta(
        self,
    ) -> Dict[str, PropertyMeta]:
        """
        Display metadata for a region entity's ``hw``/``hh``/``hd`` and
        world-frame bound (``x0``/``x1``/...) keys.

        Every region entity gets exactly these keys, regardless of its
        ``RegionSpec.source`` -- ``_build_region_state`` always sets
        ``hw``/``hh``/``hd``, and the bound pairs it materializes from
        ``_REGION_BOUND_AXES`` (the same class attribute this method
        walks) so the two can never drift apart.
        """
        meta: Dict[str, PropertyMeta] = {
            key: PropertyMeta(
                group="region half-extents",
                note="m",
            )
            for key in ("hw", "hh", "hd")
        }

        for low_name, high_name, axis in self._REGION_BOUND_AXES:
            if axis >= self.spatial_dims:
                break

            meta[low_name] = PropertyMeta(
                group="region bounds",
                note="m",
            )
            meta[high_name] = PropertyMeta(
                group="region bounds",
                note="m",
            )

        return meta

    def _region_geometry(
        self,
        spec: RegionSpec,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        bool,
    ]:
        if spec.source == "fixed":
            return (
                _vector3(
                    spec.center,
                    f"RegionSpec "
                    f"{spec.name!r} center",
                ),
                _nonnegative_vector3(
                    spec.half_extents,
                    f"RegionSpec "
                    f"{spec.name!r} half_extents",
                ),
                False,
            )

        if spec.source == "int_sites":
            base = str(
                spec.source_name
            )

            p0 = self._site_position(
                f"{base}_int_p0"
            )

            px = self._site_position(
                f"{base}_int_px"
            )

            py = self._site_position(
                f"{base}_int_py"
            )

            pz = self._site_position(
                f"{base}_int_pz"
            )

            edge_x = px - p0
            edge_y = py - p0
            edge_z = pz - p0

            center = (
                p0
                + 0.5
                * (
                    edge_x
                    + edge_y
                    + edge_z
                )
            )

            # Exact world-axis-aligned half-extents for a parallelepiped
            # generated by these edge vectors.
            half_extents = (
                0.5
                * (
                    np.abs(edge_x)
                    + np.abs(edge_y)
                    + np.abs(edge_z)
                )
            )

            return (
                center,
                half_extents,
                True,
            )

        if spec.source == "site":
            site_id = self._name2id(
                "site",
                str(spec.source_name),
            )

            center = np.asarray(
                self._data.site_xpos[site_id],
                dtype=float,
            ).copy()

            local_half_extents = (
                _nonnegative_vector3(
                    spec.half_extents,
                    f"RegionSpec "
                    f"{spec.name!r} half_extents",
                )
            )

            rotation = self._rotation(
                "site",
                site_id,
            )

            world_half_extents = (
                np.abs(rotation)
                @ local_half_extents
            )

            return (
                center,
                world_half_extents,
                True,
            )

        if spec.source == "geom":
            geom_id = self._name2id(
                "geom",
                str(spec.source_name),
            )

            center = np.asarray(
                self._data.geom_xpos[geom_id],
                dtype=float,
            ).copy()

            if spec.half_extents is not None:
                local_half_extents = (
                    _nonnegative_vector3(
                        spec.half_extents,
                        f"RegionSpec "
                        f"{spec.name!r} half_extents",
                    )
                )
            else:
                local_half_extents = (
                    _nonnegative_vector3(
                        np.asarray(
                            self._model.geom_size[
                                geom_id
                            ],
                            dtype=float,
                        )[:3],
                        f"geom "
                        f"{spec.source_name!r} size",
                    )
                )

            rotation = self._rotation(
                "geom",
                geom_id,
            )

            world_half_extents = (
                np.abs(rotation)
                @ local_half_extents
            )

            return (
                center,
                world_half_extents,
                True,
            )

        raise AssertionError(
            f"Unhandled region source "
            f"{spec.source!r}"
        )

    def _site_position(
        self,
        site_name: str,
    ) -> np.ndarray:
        site_id = self._name2id(
            "site",
            site_name,
        )

        return np.asarray(
            self._data.site_xpos[site_id],
            dtype=float,
        ).copy()

    def _rotation(
        self,
        kind: str,
        object_id: int,
    ) -> np.ndarray:
        values = getattr(
            self._data,
            f"{kind}_xmat",
            None,
        )

        if values is None:
            return np.eye(
                3,
                dtype=float,
            )

        return np.asarray(
            values[object_id],
            dtype=float,
        ).reshape(3, 3)

    def _find_corner_site_fixtures(
        self,
        suffix: str = "_int_p0",
    ) -> List[str]:
        return sorted(
            name[: -len(suffix)]
            for name in self._names("site")
            if name.endswith(suffix)
        )

    # ------------------------------------------------------------------
    # Classification and labels
    # ------------------------------------------------------------------

    def _classify(
        self,
        body_name: str,
    ) -> Optional[
        Tuple[str, int, str]
    ]:
        for rule in self.body_rules:
            if rule.matches(body_name):
                return (
                    rule.cls,
                    int(rule.priority),
                    rule.joint_scope,
                )

        return None

    def _auto_gripper_site(
        self,
    ) -> Optional[str]:
        available = set(
            self._names("site")
        )

        return next(
            (
                name
                for name
                in self._default_gripper_sites()
                if name in available
            ),
            None,
        )

    def _reserve_name(
        self,
        name: str,
        entity_id: int,
    ) -> None:
        if name in self.name_to_id:
            raise ValueError(
                f"Duplicate monitor entity name "
                f"{name!r}"
            )

        self.name_to_id[name] = entity_id

    def _build_labels(self) -> None:
        self.entity_labels.clear()

        for body in self._bodies:
            category = self._object_category(
                body.body_name
            )

            if category:
                label = (
                    f"{body.body_name}  "
                    f"({category})"
                )
            else:
                label = body.body_name

            self.entity_labels[
                body.entity_id
            ] = label

        if self._gripper is not None:
            self.entity_labels[
                self._gripper.entity_id
            ] = self._gripper.site_name

        for region in self._regions:
            self.entity_labels[
                region.entity_id
            ] = region.spec.name

        if self._orphan_sensor_entity_id is not None:
            self.entity_labels[
                self._orphan_sensor_entity_id
            ] = _ORPHAN_SENSOR_ENTITY_CLASS

        if self._orphan_actuator_entity_id is not None:
            self.entity_labels[
                self._orphan_actuator_entity_id
            ] = _ORPHAN_ACTUATOR_ENTITY_CLASS

        if self._simulation_entity_id is not None:
            self.entity_labels[
                self._simulation_entity_id
            ] = "simulation (global state)"

        for records, provenance in (
            (self._raw_bodies, "MuJoCo body"),
            (self._raw_sites, "MuJoCo site"),
            (self._raw_geoms, "MuJoCo geom"),
            (self._raw_cameras, "MuJoCo camera"),
            (self._raw_lights, "MuJoCo light"),
            (self._equalities, "MuJoCo equality"),
            (self._flexes, "MuJoCo flex"),
            (self._tendons, "MuJoCo tendon"),
        ):
            for record in records:
                self.entity_labels[record.entity_id] = "{} ({})".format(
                    record.source_name, provenance
                )

        for record in self._mocap_owners:
            self.entity_labels[
                record.entity_id
            ] = "{} (MuJoCo mocap body)".format(record.body_name)

        for record in self._declared_entities:
            self.entity_labels[record.entity_id] = "{} ({})".format(
                record.spec.name, record.spec.source_provenance
            )

    # ------------------------------------------------------------------
    # MuJoCo backend compatibility
    # ------------------------------------------------------------------

    @staticmethod
    def _reject_legacy_options(
        options: Mapping[str, Any],
    ) -> None:
        if "joint_signals" in options:
            raise TypeError(
                "joint_signals= was removed: MuJoCo field availability is "
                "global and every live curated family is eagerly materialised"
            )
        if options:
            raise TypeError(
                "Unexpected MujocoAdapter option(s): {}".format(
                    ", ".join(sorted(options))
                )
            )

    def _rebuild_reset_contract(self) -> None:
        """Recompute all model-derived capability and semantic declarations."""
        self._dt = self._infer_dt(self.env, self._model)
        source_data = getattr(self, "_native_stored_wrapper", None)
        if source_data is None:
            source_data = self._data
        self._joint_signals = tuple(
            signal
            for signal in _SIGNAL_ADDRESSES
            if hasattr(source_data, signal)
        )
        self._field_surface_report = classify_public_field_surface(
            public_noncallable_field_names(source_data)
        )

        default_policy = self._semantic_policy()
        if not isinstance(default_policy, MujocoSemanticPolicy):
            raise TypeError(
                "_semantic_policy() must return MujocoSemanticPolicy"
            )
        policy = MujocoSemanticPolicy(
            body_rules=(
                default_policy.body_rules
                if self._explicit_body_rules is None
                else self._explicit_body_rules
            ),
            regions=(
                default_policy.regions
                if self._explicit_regions is None
                else self._explicit_regions
            ),
            declared_entities=default_policy.declared_entities,
            robot_bodies=(
                default_policy.robot_bodies
                if self._explicit_robot_bodies is None
                else self._explicit_robot_bodies
            ),
        )
        self._semantic_policy_value = policy
        self.body_rules = list(policy.body_rules)
        self.regions = list(policy.regions)

    def _refresh_backend(
        self,
        allow_model_change: bool,
    ) -> None:
        model, data = self._resolve_model_data(
            self.env
        )

        if (
            not allow_model_change
            and self._model is not None
            and model is not self._model
        ):
            raise RuntimeError(
                "The MuJoCo model changed during an active "
                "adapter episode. Call adapter.reset() to "
                "re-enumerate entities."
            )

        self._model = model
        self._data = data
        self._forward_handle = self._resolve_forward_handle(self.env)

    @staticmethod
    def _resolve_forward_handle(env: Any):
        """A sim object exposing ``forward()``, or None.

        robosuite-style ``env.sim`` wraps model/data behind bindings that
        reject raw ``mujoco.mj_forward``; its own ``forward()`` is the
        supported path.
        """
        for candidate in (env, getattr(env, "unwrapped", None)):
            if candidate is None:
                continue
            sim = getattr(candidate, "sim", None)
            if sim is not None and callable(getattr(sim, "forward", None)):
                return sim
        return None

    @staticmethod
    def _resolve_model_data(
        env: Any,
    ) -> Tuple[Any, Any]:
        candidates = (
            env,
            getattr(
                env,
                "unwrapped",
                None,
            ),
        )

        for candidate in candidates:
            if candidate is None:
                continue

            sim = getattr(
                candidate,
                "sim",
                None,
            )

            if (
                sim is not None
                and hasattr(sim, "model")
                and hasattr(sim, "data")
            ):
                return (
                    sim.model,
                    sim.data,
                )

            if (
                hasattr(candidate, "model")
                and hasattr(candidate, "data")
            ):
                return (
                    candidate.model,
                    candidate.data,
                )

        raise TypeError(
            "Expected env.sim.model/env.sim.data "
            "or env.model/env.data"
        )

    def render(self) -> None:
        """
        Open (lazily) and refresh a passive MuJoCo viewer window.

        The viewer attaches to the raw ``mujoco.MjModel``/``MjData`` the
        adapter already monitors, so it works for every MuJoCo-derived
        adapter without simulator-side renderer configuration (robosuite's
        ``has_renderer`` machinery is not involved). Wrapper objects such
        as robosuite's binding classes are unwrapped via their ``_model``/
        ``_data`` attributes. Once the user closes the window, further
        calls are no-ops; requires a display (fails with MuJoCo's own
        error otherwise).
        """
        import mujoco.viewer

        viewer = getattr(self, "_passive_viewer", None)

        if viewer is not None:
            if viewer.is_running():
                viewer.sync()
            return

        self._passive_viewer = mujoco.viewer.launch_passive(
            _raw_struct(self._model, "_model"),
            _raw_struct(self._data, "_data"),
        )

        self._passive_viewer.sync()

    def close_render(self) -> None:
        """Close the passive viewer if one is open; safe to call twice.

        After ``close()`` the call pauses ``_VIEWER_TEARDOWN_SETTLE``
        seconds: process exit racing the viewer thread's shutdown dumps
        core on MuJoCo 3.10 even after a clean close (verified by
        minimal repro), and the pause deterministically outlasts the
        race.
        """
        self._close_offscreen_renderer()

        viewer = getattr(self, "_passive_viewer", None)

        if viewer is None:
            return

        self._passive_viewer = None

        try:
            viewer.close()
        except Exception:
            # Teardown must never mask the run's real outcome.
            pass

        time.sleep(_VIEWER_TEARDOWN_SETTLE)

    def camera_names(self) -> Tuple[str, ...]:
        """
        Genuine renderer camera names only: monitoring identities for
        unnamed cameras (specola_unnamed_camera_index_N) are not names
        ``mujoco.Renderer.update_scene`` accepts. First name is the
        ``capture_frame`` default.
        """
        import mujoco

        raw_model = _raw_struct(self._model, "_model")
        return tuple(
            name
            for name in (
                mujoco.mj_id2name(
                    raw_model, mujoco.mjtObj.mjOBJ_CAMERA, index
                )
                for index in range(int(raw_model.ncam))
            )
            if name
        )

    def capture_frame(
        self,
        camera: Optional[str] = None,
        *,
        width: int = 320,
        height: int = 240,
    ) -> Optional[np.ndarray]:
        """
        Render one offscreen HxWx3 uint8 RGB frame from monitored state.

        Uses ``mujoco.Renderer`` against the same (unwrapped) model/data
        the monitors read, so it works for every MuJoCo-derived adapter
        without enabling simulator-side camera observations. The
        renderer is created lazily, cached, and rebuilt when the
        requested size changes. Default camera: the model's first named
        camera, else MuJoCo's free camera. Requires an offscreen GL
        backend (EGL/osmesa work on a worker thread; measured 2026-07-30).
        """
        import mujoco

        raw_model = _raw_struct(self._model, "_model")
        names = list(self.camera_names())
        # -1 is MuJoCo's free camera for models without named cameras.
        selected: Union[str, int]
        if camera is None:
            selected = names[0] if names else -1
        else:
            if camera not in names:
                raise ValueError(
                    "unknown camera {!r}; model cameras: {}".format(
                        camera, ", ".join(names) or "(none)"
                    )
                )
            selected = camera

        # robosuite-family envs (RoboCasa, LIBERO) randomize textures at
        # reset by uploading through their LIVE render context only —
        # model.tex_data goes stale, so any renderer context created
        # afterwards samples wrong textures (verified 2026-07-31:
        # post-reset contexts render a green-spotted robot). When the
        # env carries a live offscreen context, render through it.
        sim = getattr(self.env, "sim", None)
        if (isinstance(selected, str)
                and not getattr(self, "_sim_render_broken", False)
                and getattr(sim, "_render_context_offscreen", None)
                is not None):
            try:
                frame = sim.render(
                    width=width, height=height, camera_name=selected)
            except Exception:
                # Remember the miss: retrying every frame would double
                # render cost for the whole run.
                self._sim_render_broken = True
            else:
                # robosuite renders bottom-up; contract is top-down RGB.
                return np.ascontiguousarray(frame[::-1])

        renderer = getattr(self, "_offscreen_renderer", None)
        if renderer is not None and (
            self._offscreen_size != (width, height)
            # mujoco.Renderer is model-bound; reset() may replace the
            # backend model (DMControl/MA-MuJoCo), stale imagery follows.
            or self._offscreen_model is not raw_model
        ):
            self._close_offscreen_renderer()
            renderer = None
        if renderer is None:
            # The model's offscreen framebuffer cap is visual-only;
            # raising it does not touch physics and lets captures
            # exceed authored offwidth/offheight defaults.
            raw_model.vis.global_.offwidth = max(
                int(raw_model.vis.global_.offwidth), width
            )
            raw_model.vis.global_.offheight = max(
                int(raw_model.vis.global_.offheight), height
            )
            renderer = mujoco.Renderer(
                raw_model,
                height=height,
                width=width,
            )
            self._offscreen_renderer = renderer
            self._offscreen_size = (width, height)
            self._offscreen_model = raw_model

        raw_data = _raw_struct(self._data, "_data")
        renderer.update_scene(raw_data, camera=selected)
        # update_scene rebuilds the scene, so re-enable per call.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        frame = renderer.render()
        # Cameras authored for mujoco-py's bottom-up buffer (Meta-World's
        # corner cams) have their image-up axis pointing at world-down;
        # mujoco.Renderer renders them upside down. Flip those, judged
        # from the AUTHORED reference orientation (cam_mat0) and only
        # for world-mounted cameras — a body-mounted camera (wrist cams
        # like robot0_eye_in_hand) rotates at runtime and must never
        # toggle mid-episode. Top-down cameras (up_z near 0) have no
        # world-up to violate and are never flipped.
        if isinstance(selected, str):
            camera_id = mujoco.mj_name2id(
                raw_model, mujoco.mjtObj.mjOBJ_CAMERA, selected
            )
            if (camera_id >= 0
                    and int(raw_model.cam_bodyid[camera_id]) == 0):
                up_z = raw_model.cam_mat0[camera_id].reshape(3, 3)[2, 1]
                if up_z < -0.1:
                    frame = frame[::-1]
        return frame

    def _close_offscreen_renderer(self) -> None:
        renderer = getattr(self, "_offscreen_renderer", None)
        if renderer is None:
            return
        self._offscreen_renderer = None
        try:
            renderer.close()
        except Exception:
            # Teardown must never mask the run's real outcome.
            pass

    # -- state tier (replay) ------------------------------------------------

    def state_signature(self) -> Optional[dict]:
        """MuJoCo's integration state, keyed to this engine build and model.

        ``model_sha256`` digests the bytes ``mj_saveModel`` writes for a copy
        of the compiled model whose appearance arrays
        (``_SIGNATURE_APPEARANCE_ARRAYS``) are zeroed. It covers names,
        topology, meshes, sizes, masses, friction, collision filters,
        options and timestep, so a recording made on a model that differs
        anywhere an observation can see, another RoboCasa kitchen included,
        is refused before a state is restored. Colours, materials and
        textures may differ: the dashboard's scene export and camera frames
        of a replay show the replaying process's appearance.
        """
        import copy
        import hashlib

        import mujoco

        model = _raw_struct(self._model, "_model")
        digested = copy.copy(model)   # mj_copyModel: the live model is untouched
        for name in _SIGNATURE_APPEARANCE_ARRAYS:
            array = getattr(digested, name, None)
            if array is not None:
                array[...] = 0
        buffer = np.zeros(mujoco.mj_sizeModel(digested), dtype=np.uint8)
        mujoco.mj_saveModel(digested, None, buffer)
        return {
            "engine": "mujoco",
            "version": mujoco.__version__,
            "spec": "mjSTATE_INTEGRATION",
            "length": int(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)),
            "model_sha256": hashlib.sha256(buffer.tobytes()).hexdigest(),
        }

    def capture_state(self) -> Optional[np.ndarray]:
        """``mj_getState`` for ``mjSTATE_INTEGRATION``, into a new float64 array.

        The integration state is time, positions, velocities, actuator
        activations, the warm start, controls, applied forces, mocap poses,
        user data and plugin state. Contacts, constraint forces and
        accelerations are step outputs and are not part of it.
        """
        import mujoco

        model = _raw_struct(self._model, "_model")
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
        mujoco.mj_getState(model, _raw_struct(self._data, "_data"), state, spec)
        return state

    def restore_state(self, state: np.ndarray) -> None:
        """``mj_setState`` for ``mjSTATE_INTEGRATION``, then one forward pass.

        The forward pass goes through the environment's own ``sim.forward()``
        where it has one (robosuite), else ``mujoco.mj_forward``. No physics
        step: positions, velocities and time are the recorded ones, and
        kinematics, contacts and constraint forces are recomputed for them.
        On MuJoCo 3.3.0 and 3.3.1 the forward pass also writes its
        constraint solution into the warm start, so a state captured right
        after a restore differs from the restored one in that slice; 3.8.1
        and 3.10.0 leave it unchanged (measured 2026-10-14).
        """
        import mujoco

        model = _raw_struct(self._model, "_model")
        data = _raw_struct(self._data, "_data")
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.ascontiguousarray(state, dtype=np.float64)
        expected = mujoco.mj_stateSize(model, spec)
        if state.shape != (expected,):
            raise ValueError(
                f"state of shape {state.shape}; this model's integration "
                f"state has length {expected}"
            )
        mujoco.mj_setState(model, data, state, spec)
        if self._forward_handle is not None:
            self._forward_handle.forward()
        else:
            mujoco.mj_forward(model, data)

    def scene_geometry(self) -> Optional[dict]:
        """
        Export the compiled model's rigid geometry for the 3D panel.

        Bodies and geoms come straight off ``MjModel``: ``geom_pos`` /
        ``geom_quat`` are constant offsets in the parent body frame, so
        pairing them with the per-step body poses of ``scene_poses`` is
        exact for rigid geoms. Mesh assets are packed into one buffer,
        all vertices (little-endian float32 triples) followed at
        ``faces_start`` by all faces (little-endian uint32 index
        triples); each mesh's face indices stay local to its own vertex
        block. Mesh scale, position and rotation are already baked into
        ``mesh_vert`` and the geom offsets by the compiler, so vertices
        are exported untransformed.

        Geom types without a primitive on the dashboard side (height
        fields, SDFs) are counted in ``skipped`` and left out.
        """
        import mujoco

        if self._model is None:
            return None
        raw_model = _raw_struct(self._model, "_model")

        bodies = [
            {
                "name": mujoco.mj_id2name(
                    raw_model, mujoco.mjtObj.mjOBJ_BODY, index
                ) or "",
                "parent": int(raw_model.body_parentid[index]),
            }
            for index in range(int(raw_model.nbody))
        ]

        # Resolve material colours in one pass rather than per geom.
        rgba = np.array(raw_model.geom_rgba, dtype=float)
        matid = np.asarray(raw_model.geom_matid)
        from_material = (matid >= 0) & np.all(
            rgba == _GEOM_RGBA_DEFAULT, axis=1
        )
        if from_material.any():
            rgba[from_material] = raw_model.mat_rgba[matid[from_material]]

        geoms = []
        skipped: Counter[str] = Counter()
        for index in range(int(raw_model.ngeom)):
            type_value = int(raw_model.geom_type[index])
            type_name = _GEOM_TYPE_NAMES.get(
                type_value, "type_{}".format(type_value)
            )
            if type_name not in _EXPORTED_GEOM_TYPES:
                skipped[type_name] += 1
                continue
            dataid = int(raw_model.geom_dataid[index])
            geoms.append(
                {
                    "body": int(raw_model.geom_bodyid[index]),
                    "type": type_name,
                    "size": raw_model.geom_size[index].tolist(),
                    "pos": raw_model.geom_pos[index].tolist(),
                    "quat": raw_model.geom_quat[index].tolist(),
                    "rgba": rgba[index].tolist(),
                    "group": int(raw_model.geom_group[index]),
                    "mesh": dataid if type_name == "mesh" else None,
                }
            )

        # Flexes and skins are deformable surfaces, not geoms, so the loop
        # above never sees them; count them so the panel's omissions are
        # visible in the export metadata.
        for attribute, label in (("nflex", "flex"), ("nskin", "skin")):
            count = int(getattr(raw_model, attribute, 0) or 0)
            if count:
                skipped[label] += count

        meshes = []
        vertices = []
        faces = []
        vert_offset = 0
        face_offset = 0
        for index in range(int(raw_model.nmesh)):
            vert_start = int(raw_model.mesh_vertadr[index])
            vert_count = int(raw_model.mesh_vertnum[index])
            face_start = int(raw_model.mesh_faceadr[index])
            face_count = int(raw_model.mesh_facenum[index])
            vertices.append(
                raw_model.mesh_vert[vert_start:vert_start + vert_count]
            )
            faces.append(
                raw_model.mesh_face[face_start:face_start + face_count]
            )
            meshes.append(
                {
                    "vert_offset": vert_offset,
                    "vert_count": vert_count,
                    "face_offset": face_offset,
                    "face_count": face_count,
                }
            )
            vert_offset += vert_count
            face_offset += face_count

        vertex_bytes = (
            np.concatenate(vertices).astype("<f4").tobytes()
            if vertices
            else b""
        )
        face_bytes = (
            np.concatenate(faces).astype("<u4").tobytes() if faces else b""
        )

        return {
            "json": {
                "format_version": 1,
                "nbody": int(raw_model.nbody),
                "ngeom": int(raw_model.ngeom),
                "bodies": bodies,
                "geoms": geoms,
                "meshes": meshes,
                "faces_start": len(vertex_bytes),
                "skipped": dict(skipped),
                "visible_groups": self._visible_geom_groups(),
            },
            "meshes": vertex_bytes + face_bytes,
        }

    def _visible_geom_groups(self) -> List[int]:
        """Geom groups the simulator's own cameras draw by default.

        The plain MuJoCo viewer shows groups 0-2, but robosuite renders
        with group 0 (collision meshes) hidden and group 1 (visual
        meshes) shown -- robosuite/environments/base.py drives
        ``vopt.geomgroup`` from ``render_collision_mesh`` /
        ``render_visual_mesh``. A panel that ignores that convention
        wraps every robosuite robot in its opaque green collision hulls.
        """
        # Wrappers (LIBERO's ControlEnv and friends) take these flags as
        # constructor kwargs without storing or forwarding them, so walk
        # the .env chain to the first object that exposes either — the
        # same reason _find_in_env_chain exists in run_monitor.
        env = getattr(self, "env", None)
        collision = visual = None
        seen = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            collision = getattr(env, "render_collision_mesh", None)
            visual = getattr(env, "render_visual_mesh", None)
            if collision is not None or visual is not None:
                break
            env = getattr(env, "env", None)
        if collision is None and visual is None:
            return [0, 1, 2]
        groups = []
        # An absent flag means the viewer default, which is visible.
        if collision or collision is None:
            groups.append(0)
        if visual or visual is None:
            groups.append(1)
        groups.append(2)
        return groups

    def scene_poses(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Copy this step's body poses out of ``MjData``.

        ``xpos``/``xquat`` are float64 views the simulator overwrites in
        place, so the ``astype`` narrowing to float32 doubles as the
        fresh owning copy the caller needs. Row 0 is the world body;
        mocap bodies update like any other row.
        """
        if self._data is None:
            return None
        raw_data = _raw_struct(self._data, "_data")
        return (
            raw_data.xpos.astype(np.float32),
            raw_data.xquat.astype(np.float32),
        )

    @staticmethod
    def _infer_dt(
        env: Any,
        model: Any,
    ) -> float:
        control_timestep = getattr(
            env,
            "control_timestep",
            None,
        )

        if (
            control_timestep is not None
            and float(control_timestep) > 0
        ):
            return float(
                control_timestep
            )

        control_frequency = getattr(
            env,
            "control_freq",
            None,
        )

        if (
            control_frequency is not None
            and float(control_frequency) > 0
        ):
            return (
                1.0
                / float(control_frequency)
            )

        environment_dt = getattr(
            env,
            "dt",
            None,
        )

        if (
            isinstance(environment_dt, Real)
            and float(environment_dt) > 0
        ):
            return float(
                environment_dt
            )

        frame_skip = getattr(
            env,
            "frame_skip",
            getattr(
                env,
                "_frame_skip",
                1,
            ),
        )

        return (
            float(model.opt.timestep)
            * int(frame_skip)
        )

    def _names(
        self,
        kind: str,
    ) -> Tuple[str, ...]:
        names_attribute = {
            "body": "body_names",
            "site": "site_names",
            "geom": "geom_names",
            "joint": "joint_names",
            "equality": "equality_names",
            "flex": "flex_names",
            "tendon": "tendon_names",
        }[kind]

        raw_names = getattr(
            self._model,
            names_attribute,
            None,
        )

        if raw_names is not None:
            return tuple(
                _decode_name(name)
                for name in raw_names
                if name
            )

        count_attribute = {
            "body": "nbody",
            "site": "nsite",
            "geom": "ngeom",
            "joint": "njnt",
            "equality": "neq",
            "flex": "nflex",
            "tendon": "ntendon",
        }[kind]

        count = int(
            getattr(
                self._model,
                count_attribute,
            )
        )

        names = []

        for object_id in range(count):
            name = self._id2name(
                kind,
                object_id,
            )

            if name is not None:
                names.append(name)

        return tuple(names)

    def _body_pose_attr(
        self,
        field_name: str,
    ) -> str:
        """Name of the MjData attribute holding one body-pose field.

        Resolving to a name rather than to the array is what lets a
        gather plan record where its values came from: robosuite and
        mujoco-py wrappers alias these arrays (``body_xpos`` for
        ``xpos``, ``body_xquat`` for ``xquat``), so the attribute the
        plan re-reads every step has to be the one this call picked.
        """
        aliases, _width = _BODY_POSE_SOURCES[field_name]
        for name in aliases[:-1]:
            if getattr(self._data, name, None) is not None:
                return name
        return aliases[-1]

    def _body_positions(self):
        """World body positions for both MuJoCo data APIs."""
        return getattr(self._data, self._body_pose_attr("xpos"))

    def _body_orientations(self):
        """World body orientations (quaternions) for both MuJoCo APIs."""
        return getattr(self._data, self._body_pose_attr("xquat"))

    def _name2id(
        self,
        kind: str,
        name: str,
    ) -> int:
        method = getattr(
            self._model,
            f"{kind}_name2id",
            None,
        )

        if callable(method):
            try:
                return int(
                    method(name)
                )
            except Exception as exc:
                raise ValueError(
                    f"Unknown MuJoCo {kind} name "
                    f"{name!r}"
                ) from exc

        try:
            import mujoco

            object_type = {
                "body": mujoco.mjtObj.mjOBJ_BODY,
                "site": mujoco.mjtObj.mjOBJ_SITE,
                "geom": mujoco.mjtObj.mjOBJ_GEOM,
                "camera": mujoco.mjtObj.mjOBJ_CAMERA,
                "light": mujoco.mjtObj.mjOBJ_LIGHT,
                "joint": mujoco.mjtObj.mjOBJ_JOINT,
                "equality": mujoco.mjtObj.mjOBJ_EQUALITY,
                "flex": mujoco.mjtObj.mjOBJ_FLEX,
                "tendon": mujoco.mjtObj.mjOBJ_TENDON,
            }[kind]

            object_id = int(
                mujoco.mj_name2id(
                    self._model,
                    object_type,
                    name,
                )
            )

            if object_id < 0:
                raise ValueError

            return object_id

        except (
            ImportError,
            AttributeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Cannot resolve MuJoCo "
                f"{kind} name {name!r}"
            ) from exc

    def _id2name(
        self,
        kind: str,
        object_id: int,
    ) -> Optional[str]:
        # Names are fixed for the life of the compiled model; the
        # uncached resolution below re-does attribute lookup (and, on
        # native bindings, an import plus an enum-map rebuild) per call,
        # and the identity-tag helpers invoke it per property per owner
        # per step. Cache cleared on reset (issue #92).
        key = (kind, int(object_id))
        if key in self._id2name_cache:
            return self._id2name_cache[key]
        name = self._id2name_uncached(kind, object_id)
        self._id2name_cache[key] = name
        return name

    def _id2name_uncached(
        self,
        kind: str,
        object_id: int,
    ) -> Optional[str]:
        method = getattr(
            self._model,
            f"{kind}_id2name",
            None,
        )

        if callable(method):
            try:
                name = method(object_id)

                if name is None:
                    return None

                return _decode_name(name)

            except Exception:
                pass

        try:
            import mujoco

            object_type = {
                "body": mujoco.mjtObj.mjOBJ_BODY,
                "site": mujoco.mjtObj.mjOBJ_SITE,
                "geom": mujoco.mjtObj.mjOBJ_GEOM,
                "camera": mujoco.mjtObj.mjOBJ_CAMERA,
                "light": mujoco.mjtObj.mjOBJ_LIGHT,
                "joint": mujoco.mjtObj.mjOBJ_JOINT,
                "sensor": mujoco.mjtObj.mjOBJ_SENSOR,
                "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
                "equality": mujoco.mjtObj.mjOBJ_EQUALITY,
                "flex": mujoco.mjtObj.mjOBJ_FLEX,
                "tendon": mujoco.mjtObj.mjOBJ_TENDON,
            }[kind]

            name = mujoco.mj_id2name(
                self._model,
                object_type,
                int(object_id),
            )

            if name is None:
                return None

            return _decode_name(name)

        except (
            ImportError,
            AttributeError,
        ):
            return None


def _decode_name(
    name: Any,
) -> str:
    if isinstance(name, bytes):
        return name.decode("utf-8")

    return str(name)


def _property_token(
    name: str,
) -> str:
    """
    Convert a MuJoCo joint name into a parser-compatible property token.
    """
    token = re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        name,
    )

    token = token.strip(
        "_"
    ).lower()

    token = re.sub(
        r"_+",
        "_",
        token,
    )

    if not token:
        token = "unnamed"

    if token[0].isdigit():
        token = f"joint_{token}"

    return token


def _vector3(
    value: Optional[Sequence[float]],
    label: str,
) -> np.ndarray:
    if value is None:
        raise ValueError(
            f"{label} is required"
        )

    array = np.asarray(
        value,
        dtype=float,
    )

    if (
        array.shape != (3,)
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            f"{label} must contain exactly "
            f"three finite values"
        )

    return array.copy()


def _nonnegative_vector3(
    value: Optional[Sequence[float]],
    label: str,
) -> np.ndarray:
    array = _vector3(
        value,
        label,
    )

    if np.any(array < 0):
        raise ValueError(
            f"{label} must be nonnegative"
        )

    return array


# Bounded migration provider (maintainer decision 2026-09-07). The wrapper
# adapters register the same provider for their own classes.
from .mujoco_provider import install_bounded_provider as _install_bounded_provider  # noqa: E402

_install_bounded_provider(MujocoAdapter)
