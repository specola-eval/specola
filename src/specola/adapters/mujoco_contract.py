"""Declared MuJoCo field dispositions.

This module is the field-name classification layer for the shared MuJoCo
adapter mechanism.  It records what must be read and what is deliberately
excluded; it does not infer semantics from runtime values and it does not let
simulator policy or compiled specifications select fields.

The measured denominator is the union of public, non-callable ``MjData``
names in the installed MuJoCo 3.2.3, 3.3.0, 3.3.1, 3.8.1, and 3.10.0
bindings.
Robosuite contributes three duplicate aliases, recorded separately.  A
``reader_status`` of ``implemented`` means the shared adapter has a concrete
capability-safe materializer for the record. It is not a claim that every
scene contains an applicable owner; live applicability and omissions are
reported separately by the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


ALLOWED_EXCLUSION_REASONS: FrozenSet[str] = frozenset(
    {
        "sparse_matrix_or_factor",
        "constraint_or_island_representation",
        "broadphase_cache",
        "memory_or_arena_diagnostic",
        "operational_counter_or_diagnostic",
        "opaque_user_or_plugin_semantics",
        "newer_version_physical_surface_unvalidated",
        "lazy_cache_or_sleep_scheduler",
        "not_populated_by_forward",
        "duplicate_binding_alias",
        "no_stable_owner_or_stateable_meaning",
    }
)

_OUTPUT_SINKS = frozenset(
    {
        "observation_time",
        "entity_position",
        "entity_orientation",
        "entity_property",
        "entity_metadata",
        # The raw field is materialized as a dynamic entity population
        # (one entity per unordered contact body pair) rather than as
        # keys on a pre-registered owner; the raw row array itself is
        # never exposed.
        "dynamic_entity_population",
    }
)
_READER_STATUSES = frozenset({"implemented"})

_MJDATA_SOURCE = (
    "MuJoCo mjData API and installed mujoco/mjdata.h declaration for "
    "mjData.{field}"
)
_CFRAME_SOURCE = (
    "MuJoCo API types, 'c-frame variables'; engine_core_smooth.c "
    "mj_comPos/mj_crb; and installed mujoco/mjdata.h declaration for "
    "mjData.{field}"
)
_ROBOSUITE_ALIAS_SOURCE = (
    "robosuite.utils.binding_utils.MjData wrapper properties"
)


class FieldContractError(ValueError):
    """Raised when the declared MuJoCo field contract is internally invalid."""


class UnclassifiedMujocoFieldError(FieldContractError):
    """Raised when a binding adds a public field with no disposition."""


@dataclass(frozen=True)
class IncludedFieldRecord:
    """Complete semantic declaration for one globally included field."""

    field_name: str
    family_id: str
    physical_meaning: str
    owner_resolver_id: str
    address_resolver_id: str
    population_prerequisite: str
    completion_call_id: Optional[str]
    unit_resolver_id: str
    frame: str
    group_resolver_id: str
    property_pattern: str
    output_sink: Tuple[str, ...]
    eager_materializer_id: str
    primary_source: str
    reader_status: str


@dataclass(frozen=True)
class ExcludedFieldRecord:
    """Concrete reason why one public field is not monitor-facing state."""

    field_name: str
    reason_code: str
    concrete_detail: str
    primary_source: str


@dataclass(frozen=True)
class FieldSurfaceReport:
    """Name-only classification result for one live binding surface.

    ``scope_active`` is the one runtime fact here: it is True while a
    monitor session has restricted extraction to the property keys its
    compiled specs require, and it tells a reader of the field
    dispositions that an empty ``emitted_keys`` may mean "no spec asked"
    rather than "the scene has no owner". Every author-facing
    entrypoint leaves extraction unscoped, so it is False there.
    """

    included: Tuple[str, ...]
    excluded: Tuple[str, ...]
    unclassified: Tuple[str, ...]
    overlap: Tuple[str, ...]
    scope_active: bool = False


def _included_record(
    field_name: str,
    *,
    physical_meaning: str,
    owner_resolver_id: str,
    address_resolver_id: str,
    population_prerequisite: str,
    completion_call_id: Optional[str],
    unit_resolver_id: str,
    frame: str,
    group_resolver_id: str,
    property_pattern: str,
    output_sink: Union[str, Sequence[str]],
    eager_materializer_id: str,
    primary_source: Optional[str],
) -> IncludedFieldRecord:
    return IncludedFieldRecord(
        field_name=field_name,
        family_id="mjdata.%s" % field_name,
        physical_meaning=physical_meaning,
        owner_resolver_id=owner_resolver_id,
        address_resolver_id=address_resolver_id,
        population_prerequisite=population_prerequisite,
        completion_call_id=completion_call_id,
        unit_resolver_id=unit_resolver_id,
        frame=frame,
        group_resolver_id=group_resolver_id,
        property_pattern="%s_%s" % (field_name, property_pattern),
        output_sink=(
            (output_sink,)
            if isinstance(output_sink, str)
            else tuple(output_sink)
        ),
        eager_materializer_id=eager_materializer_id,
        primary_source=(
            _MJDATA_SOURCE.format(field=field_name)
            if primary_source is None
            else primary_source.format(field=field_name)
        ),
        reader_status="implemented",
    )


def _add_included(
    target: Dict[str, IncludedFieldRecord],
    names: Sequence[str],
    *,
    meanings: Mapping[str, str],
    owner_resolver_id: str,
    address_resolver_id: str,
    population_prerequisite: str,
    completion_call_id: Optional[str],
    unit_resolver_id: str,
    frame: str,
    group_resolver_id: str,
    property_pattern: str,
    output_sink: Union[str, Sequence[str]],
    eager_materializer_id: str,
    primary_source: Optional[str] = None,
) -> None:
    for name in names:
        if name in target:
            raise FieldContractError("duplicate included field: %s" % name)
        target[name] = _included_record(
            name,
            physical_meaning=meanings[name],
            owner_resolver_id=owner_resolver_id,
            address_resolver_id=address_resolver_id,
            population_prerequisite=population_prerequisite,
            completion_call_id=completion_call_id,
            unit_resolver_id=unit_resolver_id,
            frame=frame,
            group_resolver_id=group_resolver_id,
            property_pattern=property_pattern,
            output_sink=output_sink,
            eager_materializer_id=eager_materializer_id,
            primary_source=primary_source,
        )


_included_fields: Dict[str, IncludedFieldRecord] = {}

_add_included(
    _included_fields,
    ("time",),
    meanings={"time": "Simulation time for the current state."},
    owner_resolver_id="simulation",
    address_resolver_id="scalar",
    population_prerequisite="forward_or_step",
    completion_call_id=None,
    unit_resolver_id="seconds",
    frame="simulation_clock",
    group_resolver_id="simulation_time",
    property_pattern="scalar",
    output_sink="observation_time",
    eager_materializer_id="observation_time",
)
_add_included(
    _included_fields,
    ("energy",),
    meanings={
        "energy": "Potential and kinetic energy of the simulated system."
    },
    owner_resolver_id="simulation_energy",
    address_resolver_id="energy_component",
    population_prerequisite="energy_position_and_velocity_completed",
    completion_call_id="mj_energyPos+mj_energyVel",
    unit_resolver_id="joule",
    frame="system",
    group_resolver_id="simulation_energy",
    property_pattern="{potential|kinetic}",
    output_sink="entity_property",
    eager_materializer_id="simulation_energy",
)
_add_included(
    _included_fields,
    ("qpos",),
    meanings={"qpos": "Generalized joint position coordinates."},
    owner_resolver_id="semantic_policy_joint_scope_or_declared_alias",
    address_resolver_id="jnt_qposadr+nq",
    population_prerequisite="forward_position",
    completion_call_id=None,
    unit_resolver_id="joint_position_component",
    frame="joint_coordinate",
    group_resolver_id="joint_position_component",
    property_pattern="{joint}[_{component}]",
    output_sink="entity_property",
    eager_materializer_id="joint_signal",
)
_add_included(
    _included_fields,
    ("qvel",),
    meanings={"qvel": "Generalized joint velocity coordinates."},
    owner_resolver_id="semantic_policy_joint_scope_or_declared_alias",
    address_resolver_id="jnt_dofadr+nv",
    population_prerequisite="forward_velocity",
    completion_call_id=None,
    unit_resolver_id="joint_velocity_component",
    frame="joint_coordinate",
    group_resolver_id="joint_velocity_component",
    property_pattern="{joint}[_{component}]",
    output_sink="entity_property",
    eager_materializer_id="joint_signal",
)
_add_included(
    _included_fields,
    ("qacc", "qacc_smooth"),
    meanings={
        "qacc": "Generalized joint acceleration after constraint solving.",
        "qacc_smooth": "Smooth unconstrained generalized acceleration.",
    },
    owner_resolver_id="semantic_policy_joint_scope_or_declared_alias",
    address_resolver_id="jnt_dofadr+nv",
    population_prerequisite="forward_acceleration",
    completion_call_id=None,
    unit_resolver_id="joint_acceleration_component",
    frame="joint_coordinate",
    group_resolver_id="joint_acceleration_component",
    property_pattern="{joint}[_{component}]",
    output_sink="entity_property",
    eager_materializer_id="joint_signal",
)
_add_included(
    _included_fields,
    (
        "qfrc_actuator",
        "qfrc_bias",
        "qfrc_constraint",
        "qfrc_damper",
        "qfrc_fluid",
        "qfrc_gravcomp",
        "qfrc_passive",
        "qfrc_smooth",
        "qfrc_spring",
    ),
    meanings={
        "qfrc_actuator": "Generalized force contributed by actuators.",
        "qfrc_bias": "Coriolis, centrifugal, and gravitational generalized force.",
        "qfrc_constraint": "Generalized force contributed by constraints.",
        "qfrc_damper": "Joint damping component of passive generalized force.",
        "qfrc_fluid": "Fluid-effect component of passive generalized force.",
        "qfrc_gravcomp": "Gravity-compensation component of passive generalized force.",
        "qfrc_passive": "Total passive generalized force.",
        "qfrc_smooth": "Smooth generalized force before constraint contribution.",
        "qfrc_spring": "Joint spring component of passive generalized force.",
    },
    owner_resolver_id="semantic_policy_joint_scope_or_declared_alias",
    address_resolver_id="jnt_dofadr+nv",
    population_prerequisite="forward_force_stage",
    completion_call_id=None,
    unit_resolver_id="joint_generalized_force_component",
    frame="joint_coordinate",
    group_resolver_id="joint_force_or_torque_component",
    property_pattern="{joint}[_{component}]",
    output_sink="entity_property",
    eager_materializer_id="joint_signal",
)
_add_included(
    _included_fields,
    ("qfrc_applied",),
    meanings={"qfrc_applied": "User-written generalized force input."},
    owner_resolver_id="semantic_policy_joint_scope_or_declared_alias",
    address_resolver_id="jnt_dofadr+nv",
    population_prerequisite="user_written_input_buffer",
    completion_call_id=None,
    unit_resolver_id="joint_generalized_force_component",
    frame="joint_coordinate",
    group_resolver_id="joint_force_or_torque_component",
    property_pattern="{joint}[_{component}]",
    output_sink="entity_property",
    eager_materializer_id="joint_signal",
)
_add_included(
    _included_fields,
    ("act", "act_dot"),
    meanings={
        "act": "Stateful actuator activation.",
        "act_dot": "Time derivative of stateful actuator activation.",
    },
    owner_resolver_id="stateful_actuator",
    address_resolver_id="actuator_actadr+actnum",
    population_prerequisite="stateful_actuator_forward",
    completion_call_id=None,
    unit_resolver_id="actuator_activation_or_capability_failure",
    frame="actuator_state",
    group_resolver_id="actuator_activation_unit",
    property_pattern="{actuator}[_{component}]",
    output_sink="entity_property",
    eager_materializer_id="actuator_activation",
)
_add_included(
    _included_fields,
    ("ctrl",),
    meanings={"ctrl": "Actuator control input."},
    owner_resolver_id="compiled_actuator_index",
    address_resolver_id="actuator_id",
    population_prerequisite="control_input",
    completion_call_id=None,
    unit_resolver_id="actuator_control_kind",
    frame="actuator_input",
    group_resolver_id="actuator_control_unit",
    property_pattern="{actuator}",
    output_sink="entity_property",
    eager_materializer_id="actuator_state",
)
_add_included(
    _included_fields,
    ("eq_active",),
    meanings={
        "eq_active": "Runtime enabled state of a compiled equality constraint."
    },
    owner_resolver_id="compiled_equality_index",
    address_resolver_id="equality_id",
    population_prerequisite="initialized_from_model_eq_active0",
    completion_call_id=None,
    unit_resolver_id="boolean_dimensionless",
    frame="constraint_configuration",
    group_resolver_id="equality_constraint_state",
    property_pattern="{constraint}_active",
    output_sink="entity_property",
    eager_materializer_id="equality_constraint_state",
)
_add_included(
    _included_fields,
    ("actuator_force", "actuator_length", "actuator_velocity"),
    meanings={
        "actuator_force": "Scalar force or torque produced by an actuator.",
        "actuator_length": "Actuator transmission length or angle.",
        "actuator_velocity": "Actuator transmission velocity.",
    },
    owner_resolver_id="compiled_actuator_index",
    address_resolver_id="actuator_id",
    population_prerequisite="forward_actuation",
    completion_call_id=None,
    unit_resolver_id="actuator_transmission",
    frame="actuator_transmission",
    group_resolver_id="actuator_output_unit",
    property_pattern="{actuator}",
    output_sink="entity_property",
    eager_materializer_id="actuator_state",
)
_add_included(
    _included_fields,
    ("xfrc_applied",),
    meanings={
        "xfrc_applied": "User-applied Cartesian force and torque on a body."
    },
    owner_resolver_id="semantic_policy_pose_body",
    address_resolver_id="body_id+cartesian_6",
    population_prerequisite="applied_force_input",
    completion_call_id=None,
    unit_resolver_id="cartesian_force_torque_component",
    frame="MuJoCo Cartesian wrench convention",
    group_resolver_id="body_applied_force_or_torque",
    property_pattern="{body}_{component}",
    output_sink="entity_property",
    eager_materializer_id="body_cartesian_state",
)
_add_included(
    _included_fields,
    ("xpos",),
    meanings={"xpos": "World position of a body frame origin."},
    owner_resolver_id="semantic_policy_pose_body",
    address_resolver_id="body_id+xyz",
    population_prerequisite="forward_position",
    completion_call_id=None,
    unit_resolver_id="metre",
    frame="world_body_frame_origin",
    group_resolver_id="body_position",
    property_pattern="{body}_{axis}",
    output_sink=("entity_position", "entity_property"),
    eager_materializer_id="body_pose",
)
_add_included(
    _included_fields,
    ("xquat",),
    meanings={"xquat": "World orientation quaternion of a body frame."},
    owner_resolver_id="semantic_policy_pose_body",
    address_resolver_id="body_id+orientation_component",
    population_prerequisite="forward_position",
    completion_call_id=None,
    unit_resolver_id="dimensionless_orientation",
    frame="world_body_frame",
    group_resolver_id="body_orientation",
    property_pattern="{body}_{component}",
    output_sink="entity_property",
    eager_materializer_id="body_pose",
)
_add_included(
    _included_fields,
    ("xmat",),
    meanings={"xmat": "World orientation matrix of a body frame."},
    owner_resolver_id="semantic_policy_pose_body",
    address_resolver_id="body_id+orientation_component",
    population_prerequisite="forward_position",
    completion_call_id=None,
    unit_resolver_id="dimensionless_orientation",
    frame="world_body_frame",
    group_resolver_id="body_orientation",
    property_pattern="{body}_{component}",
    output_sink="entity_property",
    eager_materializer_id="body_pose",
)
_add_included(
    _included_fields,
    ("xipos", "ximat"),
    meanings={
        "xipos": "World position of a body's centre of mass.",
        "ximat": "World orientation matrix of a body's inertial frame.",
    },
    owner_resolver_id="semantic_policy_pose_body",
    address_resolver_id="body_id+inertial_component",
    population_prerequisite="forward_position",
    completion_call_id=None,
    unit_resolver_id="body_inertial_pose_component",
    frame="world_inertial_frame",
    group_resolver_id="body_inertial_pose",
    property_pattern="{body}_{component}",
    output_sink="entity_property",
    eager_materializer_id="body_inertial_pose",
)
_add_included(
    _included_fields,
    ("xanchor",),
    meanings={"xanchor": "World position of a joint anchor."},
    owner_resolver_id="semantic_policy_joint_scope_or_declared_alias",
    address_resolver_id="joint_id+xyz",
    population_prerequisite="forward_position",
    completion_call_id=None,
    unit_resolver_id="joint_anchor_or_axis",
    frame="world",
    group_resolver_id="joint_geometry",
    property_pattern="{joint}_{axis}",
    output_sink="entity_property",
    eager_materializer_id="joint_geometry",
)
_add_included(
    _included_fields,
    ("xaxis",),
    meanings={"xaxis": "World direction of a hinge or slide joint axis."},
    owner_resolver_id=(
        "semantic_policy_hinge_slide_joint_scope_or_declared_alias"
    ),
    address_resolver_id="joint_id+xyz",
    population_prerequisite="forward_position",
    completion_call_id=None,
    unit_resolver_id="joint_anchor_or_axis",
    frame="world",
    group_resolver_id="joint_geometry",
    property_pattern="{joint}_{axis}",
    output_sink="entity_property",
    eager_materializer_id="joint_geometry",
)
_add_included(
    _included_fields,
    ("subtree_com",),
    meanings={"subtree_com": "World centre of mass of a body subtree."},
    owner_resolver_id="semantic_policy_pose_body",
    address_resolver_id="body_id+xyz",
    population_prerequisite="forward_com_position",
    completion_call_id=None,
    unit_resolver_id="metre",
    frame="world_subtree_com",
    group_resolver_id="subtree_com_position",
    property_pattern="{body}_{axis}",
    output_sink="entity_property",
    eager_materializer_id="subtree_state",
)
_add_included(
    _included_fields,
    ("subtree_linvel", "subtree_angmom"),
    meanings={
        "subtree_linvel": "World linear velocity of a body subtree centre of mass.",
        "subtree_angmom": "Angular momentum of a body subtree about its centre of mass.",
    },
    owner_resolver_id="semantic_policy_pose_body",
    address_resolver_id="body_id+xyz",
    population_prerequisite="subtree_velocity_completed",
    completion_call_id="mj_subtreeVel",
    unit_resolver_id="subtree_velocity_or_angular_momentum",
    frame="world_subtree_com",
    group_resolver_id="subtree_motion",
    property_pattern="{body}_{axis}",
    output_sink="entity_property",
    eager_materializer_id="subtree_state",
)
_add_included(
    _included_fields,
    ("cdof",),
    meanings={
        "cdof": (
            "Six-component instantaneous motion axis of a generalized "
            "degree of freedom, ordered rotation then translation."
        ),
    },
    owner_resolver_id="semantic_policy_joint_scope_or_declared_alias",
    address_resolver_id="jnt_dofadr+nv+cartesian_6",
    population_prerequisite="forward_position_mj_comPos",
    completion_call_id=None,
    unit_resolver_id="dof_kind_angular_or_linear_ratio_component",
    frame=(
        "c_frame_at_root_kinematic_tree_com_world_aligned_"
        "rotational_then_translational"
    ),
    group_resolver_id="joint_motion_axis_homogeneous_component",
    property_pattern="{joint}_{dof}_{component}",
    output_sink="entity_property",
    eager_materializer_id="joint_motion_axis",
    primary_source=_CFRAME_SOURCE,
)
_add_included(
    _included_fields,
    ("cdof_dot",),
    meanings={
        "cdof_dot": (
            "Time derivative of a generalized degree of freedom's "
            "six-component c-frame motion axis, ordered rotation then "
            "translation."
        ),
    },
    owner_resolver_id="semantic_policy_joint_scope_or_declared_alias",
    address_resolver_id="jnt_dofadr+nv+cartesian_6",
    population_prerequisite="forward_velocity_mj_comVel",
    completion_call_id=None,
    unit_resolver_id="dof_kind_motion_axis_ratio_per_second_component",
    frame=(
        "c_frame_at_root_kinematic_tree_com_world_aligned_"
        "rotational_then_translational"
    ),
    group_resolver_id="joint_motion_axis_derivative_homogeneous_component",
    property_pattern="{joint}_{dof}_{component}",
    output_sink="entity_property",
    eager_materializer_id="joint_motion_axis",
    primary_source=_CFRAME_SOURCE,
)
_add_included(
    _included_fields,
    ("cinert",),
    meanings={
        "cinert": (
            "Body spatial inertia in the c-frame: [Ixx,Iyy,Izz,Ixy,Ixz,Iyz] "
            "in [0:6], [m*dx,m*dy,m*dz] from the c-frame origin in [6:9], "
            "and body mass m in [9]."
        ),
    },
    owner_resolver_id="semantic_policy_pose_body",
    address_resolver_id="body_id+inertia_10",
    population_prerequisite="forward_position_mj_comPos",
    completion_call_id=None,
    unit_resolver_id="spatial_inertia_kg_m2_kg_m_kg_component",
    frame="c_frame_at_root_kinematic_tree_com_world_aligned",
    group_resolver_id="body_spatial_inertia_component",
    property_pattern="{body}_{component}",
    output_sink="entity_property",
    eager_materializer_id="body_inertia",
    primary_source=_CFRAME_SOURCE,
)
_add_included(
    _included_fields,
    ("crb",),
    meanings={
        "crb": (
            "Composite rigid-body spatial inertia accumulated over the "
            "body subtree, using MuJoCo's [Ixx,Iyy,Izz,Ixy,Ixz,Iyz], "
            "[m*dx,m*dy,m*dz], m packing."
        ),
    },
    owner_resolver_id="semantic_policy_pose_body",
    address_resolver_id="body_id+inertia_10",
    population_prerequisite="forward_position_mj_crb",
    completion_call_id=None,
    unit_resolver_id="spatial_inertia_kg_m2_kg_m_kg_component",
    frame="c_frame_at_root_kinematic_tree_com_world_aligned",
    group_resolver_id="composite_spatial_inertia_component",
    property_pattern="{body}_{component}",
    output_sink="entity_property",
    eager_materializer_id="body_inertia",
    primary_source=_CFRAME_SOURCE,
)
_add_included(
    _included_fields,
    ("cvel",),
    meanings={"cvel": "Body spatial velocity in MuJoCo's centre-of-mass frame."},
    owner_resolver_id="semantic_policy_pose_body",
    address_resolver_id="body_id+cartesian_6",
    population_prerequisite="forward_velocity",
    completion_call_id=None,
    unit_resolver_id="spatial_velocity_component",
    frame="com_based_rotational_then_translational",
    group_resolver_id="body_angular_or_linear_velocity",
    property_pattern="{body}_{component}",
    output_sink="entity_property",
    eager_materializer_id="body_spatial_state",
)
_add_included(
    _included_fields,
    ("cacc", "cfrc_ext", "cfrc_int"),
    meanings={
        "cacc": "Body spatial acceleration in MuJoCo's centre-of-mass frame.",
        "cfrc_ext": "External Cartesian wrench on a body.",
        "cfrc_int": "Internal Cartesian interaction wrench on a body.",
    },
    owner_resolver_id="semantic_policy_pose_body",
    address_resolver_id="body_id+cartesian_6",
    population_prerequisite="rne_post_constraint_completed",
    completion_call_id="mj_rnePostConstraint",
    unit_resolver_id="spatial_acceleration_or_wrench_component",
    frame="com_based_rotational_then_translational",
    group_resolver_id="body_acceleration_force_or_torque",
    property_pattern="{body}_{component}",
    output_sink="entity_property",
    eager_materializer_id="body_post_constraint_state",
)
_add_included(
    _included_fields,
    ("geom_xpos",),
    meanings={"geom_xpos": "World position of a compiled geometry frame."},
    owner_resolver_id="model_name_or_compiled_geom_index",
    address_resolver_id="geom_id+pose_component",
    population_prerequisite="forward_position",
    completion_call_id=None,
    unit_resolver_id="geom_pose_component",
    frame="world_geom_frame",
    group_resolver_id="geom_pose",
    property_pattern="{geom}_{component}",
    output_sink=("entity_position", "entity_property"),
    eager_materializer_id="geom_pose",
)
_add_included(
    _included_fields,
    ("geom_xmat",),
    meanings={"geom_xmat": "World orientation matrix of a compiled geometry frame."},
    owner_resolver_id="model_name_or_compiled_geom_index",
    address_resolver_id="geom_id+pose_component",
    population_prerequisite="forward_position",
    completion_call_id=None,
    unit_resolver_id="geom_pose_component",
    frame="world_geom_frame",
    group_resolver_id="geom_pose",
    property_pattern="{geom}_{component}",
    output_sink="entity_property",
    eager_materializer_id="geom_pose",
)
_add_included(
    _included_fields,
    ("site_xpos",),
    meanings={"site_xpos": "World position of a compiled site frame."},
    owner_resolver_id="model_name_or_compiled_site_index",
    address_resolver_id="site_id+pose_component",
    population_prerequisite="forward_position",
    completion_call_id=None,
    unit_resolver_id="site_pose_component",
    frame="world_site_frame",
    group_resolver_id="site_pose",
    property_pattern="{site}_{component}",
    output_sink=("entity_position", "entity_property"),
    eager_materializer_id="site_pose",
)
_add_included(
    _included_fields,
    ("site_xmat",),
    meanings={"site_xmat": "World orientation matrix of a compiled site frame."},
    owner_resolver_id="model_name_or_compiled_site_index",
    address_resolver_id="site_id+pose_component",
    population_prerequisite="forward_position",
    completion_call_id=None,
    unit_resolver_id="site_pose_component",
    frame="world_site_frame",
    group_resolver_id="site_pose",
    property_pattern="{site}_{component}",
    output_sink="entity_property",
    eager_materializer_id="site_pose",
)
_add_included(
    _included_fields,
    ("cam_xpos",),
    meanings={"cam_xpos": "Current world position of a compiled camera frame."},
    owner_resolver_id="model_name_or_compiled_camera_index",
    address_resolver_id="camera_id+pose_component",
    population_prerequisite="forward_position_mj_camlight",
    completion_call_id=None,
    unit_resolver_id="camera_pose_component",
    frame="world_camera_frame",
    group_resolver_id="camera_pose",
    property_pattern="{camera}_{component}",
    output_sink=("entity_position", "entity_property"),
    eager_materializer_id="camera_pose",
)
_add_included(
    _included_fields,
    ("cam_xmat",),
    meanings={
        "cam_xmat": (
            "Current world orientation matrix of a compiled camera frame."
        )
    },
    owner_resolver_id="model_name_or_compiled_camera_index",
    address_resolver_id="camera_id+pose_component",
    population_prerequisite="forward_position_mj_camlight",
    completion_call_id=None,
    unit_resolver_id="camera_pose_component",
    frame="world_camera_frame",
    group_resolver_id="camera_pose",
    property_pattern="{camera}_{component}",
    output_sink="entity_property",
    eager_materializer_id="camera_pose",
)
_add_included(
    _included_fields,
    ("light_xpos",),
    meanings={
        "light_xpos": (
            "Current world position of a compiled light affecting "
            "rendered sensor observations."
        )
    },
    owner_resolver_id="model_name_or_compiled_light_index",
    address_resolver_id="light_id+xyz",
    population_prerequisite="forward_position_mj_camlight",
    completion_call_id=None,
    unit_resolver_id="metre",
    frame="world_light_frame",
    group_resolver_id="light_position",
    property_pattern="{light}_{axis}",
    output_sink="entity_property",
    eager_materializer_id="light_pose",
)
_add_included(
    _included_fields,
    ("light_xdir",),
    meanings={
        "light_xdir": (
            "Current world direction of a compiled directional light "
            "affecting rendered sensor observations."
        )
    },
    owner_resolver_id="model_name_or_compiled_light_index",
    address_resolver_id="light_id+xyz",
    population_prerequisite="forward_position_mj_camlight",
    completion_call_id=None,
    unit_resolver_id="dimensionless_direction",
    frame="world_light_direction",
    group_resolver_id="light_direction",
    property_pattern="{light}_{axis}",
    output_sink="entity_property",
    eager_materializer_id="light_pose",
)
_add_included(
    _included_fields,
    ("mocap_pos",),
    meanings={"mocap_pos": "World position input for a mocap body."},
    owner_resolver_id="mocap_body_via_model_mapping",
    address_resolver_id="body_mocapid+pose_component",
    population_prerequisite="mocap_input",
    completion_call_id=None,
    unit_resolver_id="mocap_pose_component",
    frame="world",
    group_resolver_id="mocap_pose",
    property_pattern="{body}_{component}",
    output_sink=("entity_position", "entity_property"),
    eager_materializer_id="mocap_state",
)
_add_included(
    _included_fields,
    ("mocap_quat",),
    meanings={"mocap_quat": "World orientation input for a mocap body."},
    owner_resolver_id="mocap_body_via_model_mapping",
    address_resolver_id="body_mocapid+pose_component",
    population_prerequisite="mocap_input",
    completion_call_id=None,
    unit_resolver_id="mocap_pose_component",
    frame="world",
    group_resolver_id="mocap_pose",
    property_pattern="{body}_{component}",
    output_sink="entity_property",
    eager_materializer_id="mocap_state",
)
_add_included(
    _included_fields,
    ("flexvert_xpos",),
    meanings={"flexvert_xpos": "World position of a flex vertex."},
    owner_resolver_id="compiled_flex_index_and_local_vertex",
    address_resolver_id=(
        "model.flex_vertadr[flex_id]+local_vertex within "
        "model.flex_vertnum[flex_id]"
    ),
    population_prerequisite="forward_position_mj_flex",
    completion_call_id=None,
    unit_resolver_id="metre",
    frame="world",
    group_resolver_id="flex_vertex_position",
    property_pattern="{flex}_{vertex}_{axis}",
    output_sink="entity_property",
    eager_materializer_id="flex_state",
)
_add_included(
    _included_fields,
    ("flexedge_length",),
    meanings={"flexedge_length": "Length of a stable flex edge."},
    owner_resolver_id="compiled_flex_index_and_local_edge",
    address_resolver_id=(
        "model.flex_edgeadr[flex_id]+local_edge within "
        "model.flex_edgenum[flex_id]"
    ),
    population_prerequisite=(
        "forward_position_mj_flex_when_flex_not_rigid_or_interpolated"
    ),
    completion_call_id=None,
    unit_resolver_id="metre",
    frame="scalar_frame_invariant_length",
    group_resolver_id="flex_edge_length",
    property_pattern="{flex}_{edge}",
    output_sink="entity_property",
    eager_materializer_id="flex_state",
)
_add_included(
    _included_fields,
    ("flexedge_velocity",),
    meanings={
        "flexedge_velocity": "Rate of change of a stable flex-edge length."
    },
    owner_resolver_id="compiled_flex_index_and_local_edge",
    address_resolver_id=(
        "model.flex_edgeadr[flex_id]+local_edge within "
        "model.flex_edgenum[flex_id]"
    ),
    population_prerequisite=(
        "forward_velocity_mj_fwdVelocity_when_flex_edge_jacobian_populated"
    ),
    completion_call_id=None,
    unit_resolver_id="metre_per_second",
    frame="scalar_frame_invariant_length_rate",
    group_resolver_id="flex_edge_velocity",
    property_pattern="{flex}_{edge}",
    output_sink="entity_property",
    eager_materializer_id="flex_state",
)
_add_included(
    _included_fields,
    ("flexedge_J",),
    meanings={
        "flexedge_J": (
            "Jacobian of each compiled flex's stable edge length with respect "
            "to generalized coordinates."
        )
    },
    owner_resolver_id="compiled_flex_index_and_local_edge",
    address_resolver_id=(
        "model.flex_edgeadr[flex_id]+local_edge by generalized DOF"
    ),
    population_prerequisite="forward_position_mj_flex",
    completion_call_id=None,
    unit_resolver_id="metre_per_dof_coordinate_unit",
    frame="generalized_coordinate_jacobian",
    group_resolver_id="flex_edge_length_jacobian",
    property_pattern="{flex}_{edge}_{dof}",
    output_sink="entity_property",
    eager_materializer_id="flex_edge_jacobian",
)
_add_included(
    _included_fields,
    ("ten_length",),
    meanings={
        "ten_length": (
            "Compiled spatial tendon's path length or compiled fixed tendon's "
            "homogeneous scalar joint-coordinate combination."
        ),
    },
    owner_resolver_id="compiled_tendon_index",
    address_resolver_id="tendon_id",
    population_prerequisite="forward_position_mj_tendon",
    completion_call_id=None,
    unit_resolver_id="tendon_kind_or_capability_failure",
    frame="scalar_coordinate_no_spatial_frame",
    group_resolver_id="tendon_length",
    property_pattern="{tendon}",
    output_sink="entity_property",
    eager_materializer_id="tendon_state",
)
_add_included(
    _included_fields,
    ("ten_velocity",),
    meanings={
        "ten_velocity": (
            "Time derivative of a compiled tendon's scalar coordinate."
        ),
    },
    owner_resolver_id="compiled_tendon_index",
    address_resolver_id="tendon_id",
    population_prerequisite=(
        "forward_velocity_mj_fwdVelocity_from_mj_tendon_jacobian"
    ),
    completion_call_id=None,
    unit_resolver_id="tendon_velocity_kind_or_capability_failure",
    frame="scalar_coordinate_rate_no_spatial_frame",
    group_resolver_id="tendon_velocity",
    property_pattern="{tendon}",
    output_sink="entity_property",
    eager_materializer_id="tendon_state",
)
_add_included(
    _included_fields,
    ("ten_J",),
    meanings={
        "ten_J": (
            "Jacobian of a compiled tendon's scalar coordinate with respect "
            "to generalized coordinates."
        )
    },
    owner_resolver_id="compiled_tendon_index",
    address_resolver_id="tendon_id by generalized DOF",
    population_prerequisite="forward_position_mj_tendon",
    completion_call_id=None,
    unit_resolver_id="tendon_coordinate_unit_per_dof_coordinate_unit",
    frame="generalized_coordinate_jacobian",
    group_resolver_id="tendon_coordinate_jacobian",
    property_pattern="{tendon}_{dof}",
    output_sink="entity_property",
    eager_materializer_id="tendon_jacobian",
)
_add_included(
    _included_fields,
    ("sensordata",),
    meanings={
        "sensordata": (
            "Scalar components emitted by compiled model sensors; names are "
            "used when present and compiled indices otherwise."
        )
    },
    owner_resolver_id="compiled_sensor_index_and_target",
    address_resolver_id="sensor_adr+sensor_dim",
    population_prerequisite="sensor_type_forward_stage",
    completion_call_id=None,
    unit_resolver_id="sensor_type_or_capability_failure",
    frame="sensor_type_specific",
    group_resolver_id="sensor_type",
    property_pattern="{sensor}_{component}",
    output_sink="entity_property",
    eager_materializer_id="sensor_state",
)
_add_included(
    _included_fields,
    ("actuator_moment",),
    meanings={
        "actuator_moment": (
            "Jacobian of each actuator transmission length with respect to "
            "generalized coordinates (MuJoCo moment arms): dense (nu, nv) "
            "in the supported 3.2/3.3 layouts and sparse in later layouts."
        ),
    },
    owner_resolver_id="compiled_actuator_index",
    address_resolver_id=(
        "dense actuator-by-DOF row, or sparse "
        "moment_rowadr/rownnz/colind by actuator and generalized DOF"
    ),
    population_prerequisite="forward_position_mj_transmission",
    completion_call_id=None,
    unit_resolver_id="actuator_length_unit_per_dof_coordinate_unit",
    frame="generalized_coordinate_jacobian",
    group_resolver_id="actuator_transmission_moment_arm",
    property_pattern="{actuator}_{dof}",
    output_sink="entity_property",
    eager_materializer_id="actuator_moment",
)
_add_included(
    _included_fields,
    ("qM",),
    meanings={
        "qM": (
            "Joint-space inertia matrix M(q), expanded from MuJoCo's "
            "sparse qM representation into exact generalized-DOF pairs."
        ),
    },
    owner_resolver_id="simulation_global_with_dof_pair",
    address_resolver_id="mj_fullM_dense_row_and_column_dof",
    population_prerequisite="forward_position_mj_makeM",
    completion_call_id=None,
    unit_resolver_id="row_and_column_dof_motion_kinds",
    frame="generalized_coordinate_matrix",
    group_resolver_id="joint_space_inertia",
    property_pattern="{row_dof}_{column_dof}",
    output_sink="entity_property",
    eager_materializer_id="inertia_matrix",
)

_add_included(
    _included_fields,
    ("contact",),
    meanings={
        "contact": (
            "Active collision records. Row identity and ordering are "
            "unstable across steps (one body pair can occupy dozens of "
            "distinct row indices over an episode), so rows are never "
            "exposed; the reader materializes one dynamic entity per "
            "unordered body pair of the retained rows (exclude == 0 "
            "and efc_address >= 0), carrying scalar per-pair force "
            "aggregations computed via mj_contactForce, the retained "
            "point count, and the minimum contact distance."
        ),
    },
    owner_resolver_id="contact_geom_bodyid_pair",
    address_resolver_id="retained_rows_grouped_by_unordered_body_pair",
    population_prerequisite="forward_or_step",
    completion_call_id=None,
    unit_resolver_id="contact_pair_property_units",
    frame="contact_frame_scalars_and_world_resultant",
    group_resolver_id="contact_pair_groups",
    property_pattern="{aggregation}",
    output_sink="dynamic_entity_population",
    eager_materializer_id="contact_state",
)

INCLUDED_FIELDS: Mapping[str, IncludedFieldRecord] = MappingProxyType(
    dict(_included_fields)
)


_EXCLUSION_GROUPS: Mapping[str, Tuple[str, Tuple[str, ...]]] = MappingProxyType(
    {
        "sparse_matrix_or_factor": (
            "Computational matrix, factor, or multi-owner Jacobian whose "
            "entries lack one stable entity owner and/or homogeneous unit.",
            (
                "B_colind",
                "B_rowadr",
                "B_rownnz",
                "C_colind",
                "C_rowadr",
                "C_rownnz",
                "D_colind",
                "D_diag",
                "D_rowadr",
                "D_rownnz",
                "M",
                "M_colind",
                "M_rowadr",
                "M_rownnz",
                "flexedge_J_colind",
                "flexedge_J_rowadr",
                "flexedge_J_rownnz",
                "iLD",
                "iLDiagInv",
                "iM",
                "iM_colind",
                "iM_rowadr",
                "iM_rownnz",
                "mapD2M",
                "mapM2C",
                "mapM2D",
                "mapM2M",
                "moment_colind",
                "moment_rowadr",
                "moment_rownnz",
                "qDeriv",
                "qH",
                "qHDiagInv",
                "qLD",
                "qLDiagInv",
                "qLDiagSqrtInv",
                "qLU",
                "ten_J_colind",
                "ten_J_rowadr",
                "ten_J_rownnz",
            ),
        ),
        "constraint_or_island_representation": (
            "Solver constraint row, row ordering, island map, or related "
            "representation rather than state with a stable named owner.",
            (
                "dof_island",
                "dof_islandind",
                "efc_AR",
                "efc_AR_colind",
                "efc_AR_rowadr",
                "efc_AR_rownnz",
                "efc_D",
                "efc_J",
                "efc_JT",
                "efc_JT_colind",
                "efc_JT_rowadr",
                "efc_JT_rownnz",
                "efc_JT_rowsuper",
                "efc_J_colind",
                "efc_J_rowadr",
                "efc_J_rownnz",
                "efc_J_rowsuper",
                "efc_KBIP",
                "efc_R",
                "efc_Y",
                "efc_Y_colind",
                "efc_Y_rowadr",
                "efc_Y_rownnz",
                "efc_aref",
                "efc_b",
                "efc_diagA",
                "efc_diagApprox",
                "efc_force",
                "efc_frictionloss",
                "efc_id",
                "efc_island",
                "efc_margin",
                "efc_pos",
                "efc_state",
                "efc_type",
                "efc_vel",
                "flexvert_length",
                "iacc",
                "iacc_smooth",
                "iefc_D",
                "iefc_R",
                "iefc_aref",
                "iefc_force",
                "iefc_frictionloss",
                "iefc_id",
                "iefc_state",
                "iefc_type",
                "ifrc_constraint",
                "ifrc_smooth",
                "iefc_J",
                "iefc_J_colind",
                "iefc_J_rowadr",
                "iefc_J_rownnz",
                "iefc_J_rowsuper",
                "island_dofadr",
                "island_dofind",
                "island_dofnum",
                "island_efcadr",
                "island_efcind",
                "island_efcnum",
                "island_idofadr",
                "island_iefcadr",
                "island_itreeadr",
                "island_ne",
                "island_nefc",
                "island_nf",
                "island_ntree",
                "island_nv",
                "map_dof2idof",
                "map_efc2iefc",
                "map_idof2dof",
                "map_iefc2efc",
                "map_itree2tree",
                "tendon_efcadr",
                "tree_island",
            ),
        ),
        "broadphase_cache": (
            "Collision broadphase acceleration state, not monitor-facing "
            "physical state.",
            ("bvh_aabb_dyn", "bvh_active", "flexelem_aabb"),
        ),
        "memory_or_arena_diagnostic": (
            "Allocation size, high-water mark, thread handle, or raw arena "
            "pointer rather than physical simulator state.",
            (
                "maxuse_arena",
                "maxuse_con",
                "maxuse_efc",
                "maxuse_stack",
                "maxuse_threadstack",
                "narena",
                "nbuffer",
                "parena",
                "pbase",
                "pstack",
                "threadpool",
            ),
        ),
        "operational_counter_or_diagnostic": (
            "Operational count, compilation identity, diagnostic structure, "
            "or model reference rather than named physical state.",
            (
                "model",
                "nA",
                "nJ",
                "nY",
                "ncon",
                "ne",
                "nefc",
                "nf",
                "nidof",
                "nisland",
                "nl",
                "nnzJ",
                "nplugin",
                "signature",
                "solver",
                "solver_fwdinv",
                "solver_nisland",
                "solver_niter",
                "solver_nnz",
                "timer",
                "warning",
            ),
        ),
        "opaque_user_or_plugin_semantics": (
            "User- or plugin-defined storage whose physical meaning, owner, "
            "and unit cannot be stated by the shared engine adapter.",
            ("history", "plugin", "plugin_data", "plugin_state", "userdata"),
        ),
        "newer_version_physical_surface_unvalidated": (
            "Documented physical field added after every installed "
            "simulator's pinned MuJoCo version. Stable ownership appears "
            "available, but exact population, address, component, and unit "
            "behavior has not been runtime-validated in a supported "
            "environment.",
            ("flexvert_J",),
        ),
        "lazy_cache_or_sleep_scheduler": (
            "Evaluation-cache flag, warm-start value, or sleeping-body "
            "scheduler state rather than the physical quantity it controls.",
            (
                "body_awake",
                "body_awake_ind",
                "dof_awake_ind",
                "flg_energypos",
                "flg_energyvel",
                "flg_rnepost",
                "flg_subtreevel",
                "nbody_awake",
                "nparent_awake",
                "ntree_awake",
                "nv_awake",
                "parent_awake_ind",
                "qacc_warmstart",
                "tree_asleep",
                "tree_awake",
            ),
        ),
        "not_populated_by_forward": (
            "Written by inverse dynamics rather than ordinary forward "
            "simulation; reading it during normal stepping would expose a "
            "misleading constant.",
            ("qfrc_inverse",),
        ),
        "duplicate_binding_alias": (
            "Robosuite wrapper alias of the canonical included MuJoCo body "
            "pose field, not an additional physical quantity.",
            ("body_xmat", "body_xpos", "body_xquat"),
        ),
        "no_stable_owner_or_stateable_meaning": (
            "The raw row/path lacks a durable named owner, or the installed "
            "primary source does not state enough component semantics for an "
            "author-facing quantity.",
            (
                "ten_wrapadr",
                "ten_wrapnum",
                "wrap_obj",
                "wrap_xpos",
            ),
        ),
    }
)

_EXCLUSION_DETAIL_OVERRIDES: Mapping[str, str] = MappingProxyType(
    {
        "flexvert_length": (
            "MuJoCo 3.10 writes these two rows only for vertex-based flex "
            "equality mode 2: mass-scaled Cauchy-strain residuals "
            "trace(C)-2 and det(C)-1 with matching constraint Jacobian rows. "
            "They are solver residuals, not generally populated vertex "
            "lengths or homogeneous monitor-facing state."
        ),
        "ten_wrapadr": (
            "Per-named-tendon start offset into the flattened current-step "
            "path workspace; it shifts when preceding tendons' slot counts "
            "change and is buffer-layout metadata, not physical state."
        ),
        "ten_wrapnum": (
            "Per-named-tendon current path-slot count; slots mix sites, pulley "
            "sentinels, and geom tangency points, so it is dynamic workspace "
            "cardinality rather than a uniform physical quantity."
        ),
        "wrap_obj": (
            "Flattened dynamic tendon path slots encode geom ids, a site "
            "sentinel without its site id, and pulley sentinels. Python packs "
            "two independent slots per row; slot identity, role, and tendon "
            "association change with ten_wrapadr and ten_wrapnum."
        ),
        "wrap_xpos": (
            "Physical world-coordinate tendon path points are observable, "
            "but ten_wrapadr/ten_wrapnum and wrapping topology can change "
            "their slot identity during an episode. Exposing a slot as one "
            "stable monitor property would splice temporal history across "
            "different physical path roles; a topology-aware identity and "
            "history-reset contract is required first."
        ),
    }
)

_excluded_fields: Dict[str, ExcludedFieldRecord] = {}
for _reason_code, (_detail, _names) in _EXCLUSION_GROUPS.items():
    for _name in _names:
        if _name in _excluded_fields:
            raise FieldContractError("duplicate excluded field: %s" % _name)
        _excluded_fields[_name] = ExcludedFieldRecord(
            field_name=_name,
            reason_code=_reason_code,
            concrete_detail=_EXCLUSION_DETAIL_OVERRIDES.get(_name, _detail),
            primary_source=(
                _ROBOSUITE_ALIAS_SOURCE
                if _reason_code == "duplicate_binding_alias"
                else _MJDATA_SOURCE.format(field=_name)
            ),
        )

EXCLUDED_FIELDS: Mapping[str, ExcludedFieldRecord] = MappingProxyType(
    dict(_excluded_fields)
)
WRAPPER_ALIAS_FIELDS: FrozenSet[str] = frozenset(
    {
        name
        for name, record in EXCLUDED_FIELDS.items()
        if record.reason_code == "duplicate_binding_alias"
    }
)
RAW_PUBLIC_FIELD_UNION: FrozenSet[str] = frozenset(
    set(INCLUDED_FIELDS) | (set(EXCLUDED_FIELDS) - set(WRAPPER_ALIAS_FIELDS))
)


_ABSENT_BY_VERSION: Mapping[str, FrozenSet[str]] = MappingProxyType(
    {
        "3.2.3": frozenset(
            {
                "D_diag",
                "M",
                "M_colind",
                "M_rowadr",
                "M_rownnz",
                "body_awake",
                "body_awake_ind",
                "dof_awake_ind",
                "efc_Y",
                "efc_Y_colind",
                "efc_Y_rowadr",
                "efc_Y_rownnz",
                "efc_diagA",
                "flexvert_J",
                "flexvert_length",
                "flg_energypos",
                "flg_energyvel",
                "flg_rnepost",
                "flg_subtreevel",
                "history",
                "iacc",
                "iacc_smooth",
                "iLD",
                "iLDiagInv",
                "iM",
                "iM_colind",
                "iM_rowadr",
                "iM_rownnz",
                "iefc_D",
                "iefc_J",
                "iefc_J_colind",
                "iefc_J_rowadr",
                "iefc_J_rownnz",
                "iefc_J_rowsuper",
                "iefc_R",
                "iefc_aref",
                "iefc_force",
                "iefc_frictionloss",
                "iefc_id",
                "iefc_state",
                "iefc_type",
                "ifrc_constraint",
                "ifrc_smooth",
                "island_idofadr",
                "island_iefcadr",
                "island_itreeadr",
                "island_ne",
                "island_nefc",
                "island_nf",
                "island_ntree",
                "island_nv",
                "mapM2M",
                "map_dof2idof",
                "map_efc2iefc",
                "map_idof2dof",
                "map_iefc2efc",
                "map_itree2tree",
                "moment_colind",
                "moment_rowadr",
                "moment_rownnz",
                "nA",
                "nJ",
                "nY",
                "nbody_awake",
                "nidof",
                "nparent_awake",
                "ntree_awake",
                "nv_awake",
                "parent_awake_ind",
                "signature",
                "tree_asleep",
                "tree_awake",
                "tree_island",
            }
        ),
        "3.3.0": frozenset(
            {
                "M",
                "M_colind",
                "M_rowadr",
                "M_rownnz",
                "body_awake",
                "body_awake_ind",
                "dof_awake_ind",
                "efc_Y",
                "efc_Y_colind",
                "efc_Y_rowadr",
                "efc_Y_rownnz",
                "efc_diagA",
                "flexvert_J",
                "flexvert_length",
                "flg_energypos",
                "flg_energyvel",
                "flg_rnepost",
                "flg_subtreevel",
                "history",
                "iacc",
                "iacc_smooth",
                "iLD",
                "iLDiagInv",
                "iM",
                "iM_colind",
                "iM_rowadr",
                "iM_rownnz",
                "iefc_D",
                "iefc_J",
                "iefc_J_colind",
                "iefc_J_rowadr",
                "iefc_J_rownnz",
                "iefc_J_rowsuper",
                "iefc_R",
                "iefc_aref",
                "iefc_force",
                "iefc_frictionloss",
                "iefc_id",
                "iefc_state",
                "iefc_type",
                "ifrc_constraint",
                "ifrc_smooth",
                "island_idofadr",
                "island_iefcadr",
                "island_itreeadr",
                "island_ne",
                "island_nefc",
                "island_nf",
                "island_ntree",
                "island_nv",
                "mapM2M",
                "map_dof2idof",
                "map_efc2iefc",
                "map_idof2dof",
                "map_iefc2efc",
                "map_itree2tree",
                "nY",
                "nbody_awake",
                "nidof",
                "nnzJ",
                "nparent_awake",
                "ntree_awake",
                "nv_awake",
                "parent_awake_ind",
                "qLDiagSqrtInv",
                "signature",
                "tree_asleep",
                "tree_awake",
                "tree_island",
            }
        ),
        "3.3.1": frozenset(
            {
                "M",
                "body_awake",
                "body_awake_ind",
                "dof_awake_ind",
                "efc_Y",
                "efc_Y_colind",
                "efc_Y_rowadr",
                "efc_Y_rownnz",
                "efc_diagA",
                "flexvert_J",
                "flexvert_length",
                "flg_energypos",
                "flg_energyvel",
                "flg_rnepost",
                "flg_subtreevel",
                "history",
                "iacc",
                "iacc_smooth",
                "iLD",
                "iLDiagInv",
                "iM",
                "iM_colind",
                "iM_rowadr",
                "iM_rownnz",
                "iefc_D",
                "iefc_J",
                "iefc_J_colind",
                "iefc_J_rowadr",
                "iefc_J_rownnz",
                "iefc_J_rowsuper",
                "iefc_R",
                "iefc_aref",
                "iefc_force",
                "iefc_frictionloss",
                "iefc_id",
                "iefc_state",
                "iefc_type",
                "ifrc_constraint",
                "ifrc_smooth",
                "island_idofadr",
                "island_iefcadr",
                "island_itreeadr",
                "island_ne",
                "island_nefc",
                "island_nf",
                "island_ntree",
                "island_nv",
                "map_dof2idof",
                "map_efc2iefc",
                "map_idof2dof",
                "map_iefc2efc",
                "map_itree2tree",
                "nY",
                "nbody_awake",
                "nidof",
                "nnzJ",
                "nparent_awake",
                "ntree_awake",
                "nv_awake",
                "parent_awake_ind",
                "qLDiagSqrtInv",
                "tree_asleep",
                "tree_awake",
                "tree_island",
            }
        ),
        "3.8.1": frozenset(
            {
                "B_colind",
                "B_rowadr",
                "B_rownnz",
                "C_colind",
                "C_rowadr",
                "C_rownnz",
                "D_colind",
                "D_diag",
                "D_rowadr",
                "D_rownnz",
                "M_colind",
                "M_rowadr",
                "M_rownnz",
                "dof_islandind",
                "efc_JT",
                "efc_JT_colind",
                "efc_JT_rowadr",
                "efc_JT_rownnz",
                "efc_JT_rowsuper",
                "efc_Y",
                "efc_Y_colind",
                "efc_Y_rowadr",
                "efc_Y_rownnz",
                "efc_diagA",
                "flexedge_J_colind",
                "flexedge_J_rowadr",
                "flexedge_J_rownnz",
                "island_dofind",
                "island_dofnum",
                "island_efcadr",
                "island_efcind",
                "island_efcnum",
                "mapD2M",
                "mapM2C",
                "mapM2D",
                "mapM2M",
                "nY",
                "nnzJ",
                "qLDiagSqrtInv",
                "solver_nisland",
                "ten_J_colind",
                "ten_J_rowadr",
                "ten_J_rownnz",
            }
        ),
        "3.10.0": frozenset(
            {
                "B_colind",
                "B_rowadr",
                "B_rownnz",
                "C_colind",
                "C_rowadr",
                "C_rownnz",
                "D_colind",
                "D_diag",
                "D_rowadr",
                "D_rownnz",
                "M_colind",
                "M_rowadr",
                "M_rownnz",
                "dof_islandind",
                "efc_JT",
                "efc_JT_colind",
                "efc_JT_rowadr",
                "efc_JT_rownnz",
                "efc_JT_rowsuper",
                "efc_diagApprox",
                "flexedge_J_colind",
                "flexedge_J_rowadr",
                "flexedge_J_rownnz",
                "island_dofind",
                "island_dofnum",
                "island_efcadr",
                "island_efcind",
                "island_efcnum",
                "iLD",
                "iLDiagInv",
                "iM",
                "iM_colind",
                "iM_rowadr",
                "iM_rownnz",
                "iefc_J",
                "iefc_J_colind",
                "iefc_J_rowadr",
                "iefc_J_rownnz",
                "iefc_J_rowsuper",
                "mapD2M",
                "mapM2C",
                "mapM2D",
                "mapM2M",
                "maxuse_threadstack",
                "nnzJ",
                "qLDiagSqrtInv",
                "solver_nisland",
                "ten_J_colind",
                "ten_J_rowadr",
                "ten_J_rownnz",
            }
        ),
    }
)

KNOWN_RAW_FIELD_SURFACES: Mapping[str, FrozenSet[str]] = MappingProxyType(
    {
        version: frozenset(RAW_PUBLIC_FIELD_UNION - absent)
        for version, absent in _ABSENT_BY_VERSION.items()
    }
)


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_contract_records(
    included: Mapping[str, IncludedFieldRecord],
    excluded: Mapping[str, ExcludedFieldRecord],
) -> None:
    """Validate declaration structure without assuming one engine version."""

    included_names = set(included)
    excluded_names = set(excluded)
    overlap = sorted(included_names & excluded_names)
    if overlap:
        raise FieldContractError(
            "included/excluded field overlap: %s" % ", ".join(overlap)
        )

    for name, record in included.items():
        if name != record.field_name:
            raise FieldContractError(
                "included field key %r disagrees with record %r"
                % (name, record.field_name)
            )
        text_fields = (
            record.field_name,
            record.family_id,
            record.physical_meaning,
            record.owner_resolver_id,
            record.address_resolver_id,
            record.population_prerequisite,
            record.unit_resolver_id,
            record.frame,
            record.group_resolver_id,
            record.property_pattern,
            record.eager_materializer_id,
            record.primary_source,
            record.reader_status,
        )
        if not all(_nonempty_text(value) for value in text_fields):
            raise FieldContractError(
                "included field %s has an empty semantic declaration" % name
            )
        if (
            record.completion_call_id is not None
            and not _nonempty_text(record.completion_call_id)
        ):
            raise FieldContractError(
                "included field %s has an invalid completion call" % name
            )
        if (
            not record.output_sink
            or len(set(record.output_sink)) != len(record.output_sink)
            or any(sink not in _OUTPUT_SINKS for sink in record.output_sink)
        ):
            raise FieldContractError(
                "included field %s has unknown output sink %r"
                % (name, record.output_sink)
            )
        if record.reader_status not in _READER_STATUSES:
            raise FieldContractError(
                "included field %s has unknown reader status %r"
                % (name, record.reader_status)
            )

    for name, record in excluded.items():
        if name != record.field_name:
            raise FieldContractError(
                "excluded field key %r disagrees with record %r"
                % (name, record.field_name)
            )
        if record.reason_code not in ALLOWED_EXCLUSION_REASONS:
            raise FieldContractError(
                "excluded field %s has unknown reason %r"
                % (name, record.reason_code)
            )
        if not _nonempty_text(record.concrete_detail):
            raise FieldContractError(
                "excluded field %s needs concrete detail" % name
            )
        if record.concrete_detail.strip().lower() == "internal":
            raise FieldContractError(
                "excluded field %s cannot use catch-all 'internal'" % name
            )
        if not _nonempty_text(record.primary_source):
            raise FieldContractError(
                "excluded field %s needs a primary source" % name
            )


def _unclassified_error(names: Iterable[str]) -> UnclassifiedMujocoFieldError:
    ordered = ", ".join(sorted(names))
    return UnclassifiedMujocoFieldError(
        "Unclassified public MuJoCo field(s): %s. Review physical meaning, "
        "ownership, population, unit/frame/group, and choose an inclusion "
        "or exclusion reason before accepting this binding." % ordered
    )


def classify_public_field_surface(names: Iterable[str]) -> FieldSurfaceReport:
    """Classify public field names without reading their values."""

    validate_contract_records(INCLUDED_FIELDS, EXCLUDED_FIELDS)
    supplied = tuple(names)
    if any(not _nonempty_text(name) for name in supplied):
        raise FieldContractError("public field names must be nonempty strings")
    if len(set(supplied)) != len(supplied):
        raise FieldContractError("public field surface contains duplicate names")

    supplied_set = set(supplied)
    included = supplied_set & set(INCLUDED_FIELDS)
    excluded = supplied_set & set(EXCLUDED_FIELDS)
    overlap = included & excluded
    unclassified = supplied_set - included - excluded
    if unclassified:
        raise _unclassified_error(unclassified)
    if overlap:
        raise FieldContractError(
            "live included/excluded field overlap: %s"
            % ", ".join(sorted(overlap))
        )
    return FieldSurfaceReport(
        included=tuple(sorted(included)),
        excluded=tuple(sorted(excluded)),
        unclassified=(),
        overlap=(),
    )


def public_noncallable_field_names(data: object) -> Tuple[str, ...]:
    """Enumerate the public field-name denominator without semantic inference."""

    return tuple(
        sorted(
            name
            for name in dir(data)
            if not name.startswith("_") and not callable(getattr(data, name))
        )
    )


def validate_known_version_surface(
    version: str,
    names: Iterable[str],
) -> FieldSurfaceReport:
    """Require exact equality with one mechanically measured raw surface."""

    if version not in KNOWN_RAW_FIELD_SURFACES:
        raise FieldContractError(
            "unsupported MuJoCo field-surface version %r; classify its "
            "public fields before claiming coverage" % version
        )
    supplied = tuple(names)
    report = classify_public_field_surface(supplied)
    supplied_set = set(supplied)
    expected = set(KNOWN_RAW_FIELD_SURFACES[version])
    missing = sorted(expected - supplied_set)
    extra = sorted(supplied_set - expected)
    if missing or extra:
        pieces = []
        if missing:
            pieces.append("missing public fields: %s" % ", ".join(missing))
        if extra:
            pieces.append("unexpected public fields: %s" % ", ".join(extra))
        raise FieldContractError(
            "MuJoCo %s surface mismatch: %s" % (version, "; ".join(pieces))
        )
    return report


def _validate_static_contract() -> None:
    validate_contract_records(INCLUDED_FIELDS, EXCLUDED_FIELDS)
    wrapper_union = set(RAW_PUBLIC_FIELD_UNION) | set(WRAPPER_ALIAS_FIELDS)
    dispositions = set(INCLUDED_FIELDS) | set(EXCLUDED_FIELDS)
    if dispositions != wrapper_union:
        missing = sorted(wrapper_union - dispositions)
        extra = sorted(dispositions - wrapper_union)
        raise FieldContractError(
            "static MuJoCo field partition mismatch; missing=%r extra=%r"
            % (missing, extra)
        )
    if len(INCLUDED_FIELDS) != 63:
        raise FieldContractError(
            "expected 63 included raw fields, found %d" % len(INCLUDED_FIELDS)
        )
    if len(EXCLUDED_FIELDS) != 179:
        raise FieldContractError(
            "expected 179 wrapper-surface exclusions, found %d"
            % len(EXCLUDED_FIELDS)
        )
    for version, absent in _ABSENT_BY_VERSION.items():
        unknown_absent = sorted(set(absent) - set(RAW_PUBLIC_FIELD_UNION))
        if unknown_absent:
            raise FieldContractError(
                "MuJoCo %s absence table contains unknown names: %s"
                % (version, ", ".join(unknown_absent))
            )
    for version, surface in KNOWN_RAW_FIELD_SURFACES.items():
        if not surface <= RAW_PUBLIC_FIELD_UNION:
            raise FieldContractError(
                "MuJoCo %s surface contains names outside the raw union"
                % version
            )


_validate_static_contract()
