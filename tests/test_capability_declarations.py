"""Capability-honesty check: the descoped publication gate.

Per the 2026-08-20 amendment to the adapter-fidelity-certification
decision, the publication criterion for adapter capability claims is this
table plus CORE conformance, not executed CAP/FAC evidence assembly.

Each bundled adapter declares the capability families it supports.
Each declared family must reference at least one existing test that
exercises extraction for that family. An adapter that accepts scenes of
an undeclared family must carry an explicit incompleteness marker rather
than silently omitting the state — the failure class this file exists to
prevent is the Isaac deformable nodal-state omission.

Referenced tests are checked for existence (module imports, attribute
present); their execution is the surrounding suite's job. Simulator-gated
tests count: a declaration backed by a test that skips without its
simulator is still an honest, checkable declaration.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from specola.adapter_enrollment import bundled_adapter_declarations

FAMILIES = frozenset(
    (
        "rigid_body",
        "articulation",
        "deformable_state",
        "contact",
        "physical_sensor",
    )
)

# One reviewed table. Keys are registry entry-point names; values map each
# declared family to (test_module, test_attribute) pairs proving the family
# is exercised somewhere in the suite. NO_PHYSICAL_CAPABILITIES marks an
# adapter that deliberately declares none (signal passthrough).
NO_PHYSICAL_CAPABILITIES = "no_physical_capabilities"

_SHARED_MUJOCO = {
    "articulation": [
        ("tests.test_mujoco_materialization",
         "test_unselected_unnamed_body_and_joint_are_not_owner_omissions"),
    ],
    "rigid_body": [
        ("tests.test_mujoco_materialization",
         "test_classified_mocap_body_and_raw_input_have_distinct_positions"),
    ],
    "contact": [
        ("tests.test_mujoco_contact_entities",
         "test_capabilities_declare_contact_schema_before_first_contact"),
    ],
    "deformable_state": [
        ("tests.test_mujoco_materialization",
         "test_compiled_flex_uses_model_addresses_and_populated_edge_jacobian"),
    ],
}

DECLARATIONS = {
    "mock": {
        "articulation": [
            # Extracts and binds the mock robot's qpos_joint1 signal.
            ("tests.test_capabilities_metadata",
             "test_reset_adapter_routes_metadata_through_discovery_helper"),
        ],
        "physical_sensor": [
            # Exercises the mock gripper's recent_ee_force_* components.
            ("tests.test_check_specs",
             "test_dedicated_unit_overrides_component_note"),
        ],
    },
    "dataset": NO_PHYSICAL_CAPABILITIES,
    "libero": {
        **_SHARED_MUJOCO,
        "contact": [
            ("tests.test_libero_adapter",
             "test_declared_contact_matches_all_native_rows_without_solver_filter"),
        ],
    },
    "robocasa": _SHARED_MUJOCO,
    "metaworld": _SHARED_MUJOCO,
    "dmcontrol": _SHARED_MUJOCO,
    "mamujoco": _SHARED_MUJOCO,
    "pybullet": {
        "rigid_body": [
            ("tests.test_pybullet_adapter",
             "test_base_link_and_contact_keys"),
        ],
        "articulation": [
            ("tests.test_pybullet_adapter",
             "test_base_link_and_contact_keys"),
        ],
        "contact": [
            ("tests.test_pybullet_adapter",
             "test_base_link_and_contact_keys"),
        ],
    },
    "isaaclab": {
        "articulation": [
            ("tests.test_isaaclab_adapter",
             "test_env_index_selects_that_environment"),
        ],
        "rigid_body": [
            ("tests.test_isaaclab_legacy_projection",
             "test_legacy_projection_and_metadata_are_frozen_before_deformable_refactor"),
        ],
        "physical_sensor": [
            ("tests.test_isaaclab_adapter",
             "test_missing_wrench_omits_only_wrench_keys"),
        ],
    },
    "carla": {
        "rigid_body": [
            ("tests.test_carla_adapter",
             "test_dead_actor_is_absent_but_live_read_failure_propagates"),
        ],
    },
    "robodojo": {
        "articulation": [
            ("tests.test_robodojo_adapter",
             "test_articulated_fixtures_report_their_joints"),
        ],
        "rigid_body": [
            ("tests.test_robodojo_adapter",
             "test_rigid_objects_are_not_reported_twice_through_the_dynamic_alias"),
        ],
        "physical_sensor": [
            # The fixture env carries body_incoming_joint_wrench_b; parity
            # covers the link_wrench_* keys it produces.
            ("tests.test_robodojo_adapter",
             "test_metadata_parity_across_every_entity"),
        ],
    },
}

# Families an adapter accepts scenes for WITHOUT declaring support: each
# entry must name the test asserting the explicit incompleteness marker.
# Silent acceptance is the prohibited state.
ACCEPTED_BUT_INCOMPLETE = {
    ("isaaclab", "deformable_state"): (
        "tests.test_isaaclab_legacy_projection",
        "test_legacy_projection_and_metadata_are_frozen_before_deformable_refactor",
    ),
    # Garments are pose-only and fluids report a labelled centroid
    # approximation; no nodal or per-particle state is extracted.
    ("robodojo", "deformable_state"): (
        "tests.test_robodojo_adapter",
        "test_fluid_reports_particles_without_inventing_a_position",
    ),
}


def _registered_names():
    return {item.name for item in bundled_adapter_declarations()}


def test_every_registered_adapter_declares_capabilities():
    registered = _registered_names()
    declared = set(DECLARATIONS)
    assert registered == declared, (
        "capability table out of sync with the adapter registry: "
        f"missing {sorted(registered - declared)}, "
        f"stale {sorted(declared - registered)}"
    )


def test_declared_families_are_known_and_nonempty():
    for name, declaration in DECLARATIONS.items():
        if declaration == NO_PHYSICAL_CAPABILITIES:
            continue
        assert declaration, f"{name}: empty declaration without the marker"
        unknown = set(declaration) - FAMILIES
        assert not unknown, f"{name}: unknown families {sorted(unknown)}"
        for family, references in declaration.items():
            assert references, f"{name}/{family}: no test references"


def _load_test_module(module_name):
    """Load ``tests.<name>`` from this directory by path.

    ``importlib.import_module("tests.x")`` resolves ``tests`` through
    ``sys.path``, and some simulator environments ship a top-level
    ``tests`` package in site-packages (the robocasa env does), which
    shadows this directory and fails every reference here. Loading by
    file path never consults ``sys.path``.
    """
    basename = module_name.rsplit(".", 1)[-1]
    path = Path(__file__).with_name(basename + ".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name, family, module_name, attribute",
    [
        (name, family, module_name, attribute)
        for name, declaration in sorted(DECLARATIONS.items())
        if declaration != NO_PHYSICAL_CAPABILITIES
        for family, references in sorted(declaration.items())
        for module_name, attribute in references
    ],
)
def test_declared_family_references_an_existing_test(
    name, family, module_name, attribute
):
    module = _load_test_module(module_name)
    assert hasattr(module, attribute), (
        f"{name}/{family}: {module_name}.{attribute} does not exist"
    )


@pytest.mark.parametrize(
    "name, family, module_name, attribute",
    [
        (name, family, module_name, attribute)
        for (name, family), (module_name, attribute)
        in sorted(ACCEPTED_BUT_INCOMPLETE.items())
    ],
)
def test_undeclared_acceptance_carries_an_incompleteness_marker(
    name, family, module_name, attribute
):
    declaration = DECLARATIONS[name]
    assert declaration == NO_PHYSICAL_CAPABILITIES or family not in declaration, (
        f"{name}/{family}: declared AND marked incomplete — pick one"
    )
    module = _load_test_module(module_name)
    assert hasattr(module, attribute), (
        f"{name}/{family}: incompleteness marker test "
        f"{module_name}.{attribute} does not exist"
    )


def test_isaac_deformable_incompleteness_marker_is_live():
    # Direct check on the plan compiler, not just on a test's existence:
    # deformable entries must compile to the explicitly incomplete
    # profile with zero semantic fields — silent acceptance is the
    # prohibited state.
    from specola.adapters.isaaclab_extraction import ScalarPlanExtractor

    extractor = ScalarPlanExtractor(lambda value: None, lambda name: name)
    fields, metadata = extractor._fields(object(), "deformable_objects")
    assert fields == ()
    assert metadata == ()
    import inspect

    source = inspect.getsource(ScalarPlanExtractor)
    assert "deformable_legacy_incomplete" in source
