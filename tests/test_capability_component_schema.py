"""Declared component schemas validate specs with zero live instances.

``AdapterCapabilities.components_by_class`` (design section 5.6.1) lets an
adapter declare the component schema of a dynamic entity class whose
population may be empty at reset -- contact pairs are the motivating
case. Compilation must accept a spec whose required components are
covered by a CLOSED declared schema even when the discovered catalog has
no instance of the class, and must still reject components outside that
schema loudly. Without this, ``specola check`` (strict, no capabilities)
and ``specola run`` (open capabilities) disagree about the same spec.
"""

from __future__ import annotations

import pytest

from specola.capabilities import AdapterCapabilities, VocabularyMode
from specola.components import ComponentCatalog
from specola.entities import EntityState, SpecConfigError, TrackState
from specola.stl_parser import compile_specs


_CONTACT_COMPONENTS = frozenset(
    {
        "contact_point_force_max",
        "contact_normal_force_sum",
    }
)


def _catalog_without_contacts() -> ComponentCatalog:
    entity = EntityState(
        entity_id=0,
        entity_class="object",
        position=None,
        velocity=None,
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties={"height": 1.0},
    )
    return ComponentCatalog.from_entities({0: entity})


def _capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        spatial_dims=3,
        class_mode=VocabularyMode.CLOSED,
        supported_classes=frozenset({"object", "robot_contact"}),
        runtime_populated_classes=frozenset({"robot_contact"}),
        components_by_class={
            "robot_contact": _CONTACT_COMPONENTS,
        },
        component_mode_by_class={
            "robot_contact": VocabularyMode.CLOSED,
        },
    )


def _compile(formula: str, capabilities=None):
    return compile_specs(
        formula,
        dt=0.1,
        catalog=_catalog_without_contacts(),
        capabilities=capabilities,
    )


def test_declared_schema_accepts_spec_with_zero_instances():
    formula = "robot_contact_*[contact_point_force_max] < 200"
    _compile(formula, capabilities=_capabilities())


def test_static_declared_class_with_zero_instances_still_rejects():
    from dataclasses import replace

    formula = "robot_contact_*[contact_point_force_max] < 200"
    capabilities = replace(_capabilities(), runtime_populated_classes=frozenset())
    with pytest.raises(SpecConfigError, match="No discovered entity of static class"):
        _compile(formula, capabilities=capabilities)


def test_without_capabilities_zero_instances_still_reject():
    formula = "robot_contact_*[contact_point_force_max] < 200"
    with pytest.raises(SpecConfigError):
        _compile(formula, capabilities=None)


def test_closed_schema_rejects_unknown_component():
    formula = "robot_contact_*[contact_force_typo] < 200"
    with pytest.raises(SpecConfigError) as excinfo:
        _compile(formula, capabilities=_capabilities())
    assert "contact_force_typo" in str(excinfo.value)


def test_declared_class_accepted_by_strict_class_vocabulary():
    # The class is absent from the catalog but declared in
    # supported_classes; strict class validation must accept it.
    formula = "robot_contact_*[contact_point_force_max] < 200"
    compiled = _compile(formula, capabilities=_capabilities())
    assert compiled is not None


def test_check_reports_unresolved_pinned_entity_id():
    # An adapter-declared open entity-ID vocabulary makes an absent
    # pinned ID compile-and-wait in both check and run; check must still
    # name the unresolved pin so a typoed ID is visible.
    from specola.check_specs import _unresolved_pinned_ids

    formula = "object_7[height] < 2"
    compiled = _compile(formula, capabilities=_capabilities())
    unresolved = _unresolved_pinned_ids(
        compiled[0], _catalog_without_contacts()
    )
    assert unresolved == [7]


def test_without_capabilities_pinned_absent_id_still_rejects():
    formula = "object_7[height] < 2"
    with pytest.raises(SpecConfigError):
        _compile(formula, capabilities=None)
