from __future__ import annotations

from specola.adapter import PropertyMeta
from specola.cli_common import _MockAdapter
from specola.discovery_filter import DiscoveryQuery, filter_discovery
from specola.discovery_model import build_discovery_model
from specola.discovery_selector import (
    DiscoveryScope,
    ROLE_ABSENT,
    _resolved_pairs,
    entities_for_scope,
    selector_for_signal,
    selector_preview,
    signal_selections,
)


def test_pinned_single_property_preview_has_one_binding():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    model = build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )
    view = filter_discovery(model, DiscoveryQuery(
        entities=("mock_target_object",),
        signals=("mass",),
    ))

    preview = selector_preview(view)

    assert preview.selector == "object_1[mass]"
    assert preview.binding_count == 1


def _role_view():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    adapter.entity_roles = {1: "target", 2: "distractor"}
    original = adapter.property_metadata

    def metadata(entity_id):
        if entity_id in (1, 2):
            return {
                "mass": PropertyMeta(
                    group="mass",
                    note="kg",
                    tags=("role:" + adapter.entity_roles[entity_id],),
                )
            }
        return original(entity_id)

    adapter.property_metadata = metadata
    model = build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )
    return filter_discovery(model, DiscoveryQuery(include_raw=False))


def test_class_role_scope_preserves_exact_mapping_and_real_tag_resolution():
    view = _role_view()
    scope = DiscoveryScope("object", role="target")
    signal = next(
        item for item in signal_selections(view, scope)
        if item.pattern == "mass"
    )

    preview = selector_for_signal(view, scope, signal)

    assert [(item.entity_id, item.exact_key) for item in signal.mappings] == [
        (1, "mass"),
    ]
    assert preview.selector == "object_*[mass @ role:target]"
    assert preview.binding_count == 1
    assert "constants: {threshold: 0.0}" in preview.example


def test_class_scope_reports_each_exact_entity_key_binding():
    view = _role_view()
    scope = DiscoveryScope("object")
    signal = next(
        item for item in signal_selections(view, scope)
        if item.pattern == "mass"
    )
    preview = selector_for_signal(view, scope, signal)

    assert preview.selector == "object_*[mass]"
    assert preview.binding_count == 2
    assert {item.entity_id for item in signal.mappings} == {1, 2}


def test_declared_family_groups_entity_native_keys_without_name_inference():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    entities[1].properties["xpos_target_2"] = 0.9
    entities[2].properties["xpos_distractor_2"] = 0.8
    original = adapter.property_metadata

    def metadata(entity_id):
        values = dict(original(entity_id))
        if entity_id in (1, 2):
            key = (
                "xpos_target_2"
                if entity_id == 1 else "xpos_distractor_2"
            )
            values[key] = PropertyMeta(
                group="body position",
                note="m",
                family="xpos",
                component="z (axis 2)",
                frame="world",
            )
        return values

    adapter.property_metadata = metadata
    model = build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )
    view = filter_discovery(model, DiscoveryQuery())
    selection = next(
        item for item in signal_selections(
            view, DiscoveryScope("object")
        )
        if item.pattern == "xpos_*_2"
    )

    assert selection.component == "z (axis 2)"
    assert selection.frame == "world"
    assert [item.exact_key for item in selection.mappings] == [
        "xpos_target_2",
        "xpos_distractor_2",
    ]
    assert selector_for_signal(
        view, DiscoveryScope("object"), selection
    ).selector == "object_*[xpos_*_2]"


def test_undeclared_family_keeps_different_exact_keys_separate():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    entities[1].properties["custom_alpha_0"] = 1.0
    entities[2].properties["custom_beta_0"] = 2.0
    original = adapter.property_metadata

    def metadata(entity_id):
        values = dict(original(entity_id))
        key = {
            1: "custom_alpha_0",
            2: "custom_beta_0",
        }.get(entity_id)
        if key is not None:
            values[key] = PropertyMeta(group="custom", note="m")
        return values

    adapter.property_metadata = metadata
    model = build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )
    view = filter_discovery(model, DiscoveryQuery())
    selections = [
        item for item in signal_selections(
            view, DiscoveryScope("object")
        )
        if item.quantity == "custom"
    ]

    assert [item.pattern for item in selections] == [
        "custom_alpha_0",
        "custom_beta_0",
    ]


def test_invalid_third_party_key_does_not_crash_selector_preview():
    view = _role_view()
    assert _resolved_pairs(view.entities, "third.party-key") == ()


def test_roleless_scope_has_exact_population_and_no_invented_selector():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    adapter.entity_roles = {1: "target"}
    model = build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )
    view = filter_discovery(model, DiscoveryQuery(include_raw=False))
    scope = DiscoveryScope("object", role=ROLE_ABSENT)

    scoped = entities_for_scope(view, scope)
    signal = next(
        item for item in signal_selections(view, scope)
        if item.pattern == "mass"
    )
    preview = selector_for_signal(view, scope, signal)

    assert [entity.entity_id for entity in scoped] == [2]
    assert {mapping.entity_id for mapping in signal.mappings} == {2}
    assert preview.selector is None
    assert "cannot express" in preview.explanation


def test_roleless_scope_on_all_roleless_class_keeps_class_selector():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    model = build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )
    view = filter_discovery(model, DiscoveryQuery(include_raw=False))
    scope = DiscoveryScope("object", role=ROLE_ABSENT)
    signal = next(
        item for item in signal_selections(view, scope)
        if item.pattern == "mass"
    )

    preview = selector_for_signal(view, scope, signal)

    assert preview.selector == "object_*[mass]"
    assert preview.binding_count == 2


def test_top_level_role_scope_resolves_single_class_signal():
    view = _role_view()
    scope = DiscoveryScope(None, role="target")
    signal = next(
        item for item in signal_selections(view, scope)
        if item.pattern == "mass"
    )

    preview = selector_for_signal(view, scope, signal)

    assert preview.selector == "object_*[mass @ role:target]"
