"""
test_check_specs.py — specola check CLI (stage-4 Task 2).

specola check explains a spec file against a live sim without running an
episode: selector expansion, parameter slots, referenced component keys,
and plain-English quantifier readings. Exercised end-to-end against
``--sim mock`` (src/specola/adapters/fake.py), subprocess pattern
from ``tests/test_discover_format.py::_run_discover``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from specola.adapter import PropertyMeta
from specola.capabilities import AdapterCapabilities, VocabularyMode
from specola.check_specs import (
    _check_entry,
    _covering_glob,
    _eligible_entities,
    _partial_binding_warnings,
    _uses_legacy_gf,
)
from specola.components import ComponentCatalog
from specola.entities import EntityState, TrackState
from specola.monitor import ParameterSpec, SpecTemplate


def _run_check(specs_path, *extra):
    return subprocess.run(
        [
            sys.executable, "-m", "specola.check_specs",
            "--sim", "mock", "--no-pager", str(specs_path), *extra,
        ],
        capture_output=True, text=True, timeout=60,
    )


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# A valid selector spec expands, reports its monitor count, and renders a
# plain-English ALL/ANY reading for the resulting n-ary min(/max( call. A
# second, non-selector entry exercises the unit column against the mock
# adapter's own PropertyMeta (fake.py: qpos_joint1 -> "rad").
# ---------------------------------------------------------------------------

_VALID_SPECS = """
specs:
- name: gripper_force_axes_bounded
  formula: 'G[0,5s](gripper_*[all: recent_ee_force_* \\ recent_ee_force_norm] <= 5.0)'
- name: joint_within_range
  formula: "G[0,5s](robot_*[qpos_joint1] <= 10.0)"
"""


def test_valid_selector_spec_exits_zero_and_explains_expansion(tmp_path):
    path = _write(tmp_path, "specs.yaml", _VALID_SPECS)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "expands to:" in proc.stdout
    assert "monitors:" in proc.stdout
    assert "rad" in proc.stdout  # unit column entry (qpos_joint1 -> "rad")
    assert "ALL must satisfy" in proc.stdout
    assert "G/F are legacy Specola aliases" in proc.stdout


def test_legacy_alias_warning_does_not_match_component_names():
    assert _uses_legacy_gf("G[0,4](robot_*[x] >= 0)")
    assert _uses_legacy_gf("g[0,4](robot_*[x] >= 0)")
    assert _uses_legacy_gf("not F[0,4](robot_*[x] >= 0)")
    assert _uses_legacy_gf("not f[0,4](robot_*[x] >= 0)")
    assert not _uses_legacy_gf("historically[0,4](robot_*[G_force] >= 0)")


# ---------------------------------------------------------------------------
# An unknown component key fails compilation; the error is printed verbatim
# plus a "did you mean" suggestion from the class's real keys.
# ---------------------------------------------------------------------------

_UNKNOWN_KEY_SPEC = """
specs:
- name: bad_key
  formula: "G[0,5s](robot_*[qppos_joint1] >= 0)"
"""


def test_unknown_component_exits_one_with_did_you_mean(tmp_path):
    path = _write(tmp_path, "specs.yaml", _UNKNOWN_KEY_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "did you mean:" in proc.stdout
    assert "qpos_joint1" in proc.stdout


# ---------------------------------------------------------------------------
# A comparison mixing components with distinct units (rad vs N·m, both on
# the mock robot's joint1) warns but never fails the run.
# ---------------------------------------------------------------------------

_MIXED_UNIT_SPEC = """
specs:
- name: mixed
  formula: "G[0,5s](min(robot_*[qpos_joint1], robot_*[qfrc_applied_joint1]) >= 0)"
"""


def test_mixed_units_warns_but_exits_zero(tmp_path):
    path = _write(tmp_path, "specs.yaml", _MIXED_UNIT_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "warning: mixed units" in proc.stdout
    assert "rad" in proc.stdout
    assert "N·m" in proc.stdout


def test_dedicated_unit_overrides_component_note(capsys):
    entities = {
        1: EntityState(
            entity_id=1,
            entity_class="object",
            track_age=0,
            track_state=TrackState.CONFIRMED,
            properties={"quat_w": 1.0, "quat_x": 0.0},
        ),
    }
    metadata = {
        1: {
            "quat_w": PropertyMeta(
                group="body orientation",
                note="quat w",
                unit="1",
            ),
            "quat_x": PropertyMeta(
                group="body orientation",
                note="quat x",
                unit="1",
            ),
        },
    }
    catalog = ComponentCatalog.from_entities(entities, metadata=metadata)
    entry = SimpleNamespace(
        name="orientation",
        formula_text="min(object_*[quat_w], object_*[quat_x]) >= 0",
        constants={},
        symmetric=False,
        backend="auto",
    )

    assert _check_entry(
        entry,
        0.05,
        catalog,
        metadata,
        entities,
        labels=None,
    )
    output = capsys.readouterr().out
    assert "warning: mixed units" not in output
    assert "quat w" not in output
    assert "quat x" not in output
    assert re.search(r"quat_w\s+1\s+\+1", output)


# ---------------------------------------------------------------------------
# A syntactically broken entry (unbalanced paren) must not swallow the
# report for a valid sibling entry — MonitorEngine.load's all-or-nothing
# contract is deliberately not what backs this CLI (see check_specs.py's
# module docstring and MonitorEngine.iter_spec_entries).
# ---------------------------------------------------------------------------

_ONE_BROKEN_ONE_VALID_SPECS = """
specs:
- name: broken
  formula: "G[0,5s](robot_0[qpos_joint1] <= 1.0"
- name: valid
  formula: "G[0,5s](robot_0[qpos_joint1] <= 1.0)"
"""


def test_broken_entry_does_not_swallow_valid_sibling_report(tmp_path):
    path = _write(tmp_path, "specs.yaml", _ONE_BROKEN_ONE_VALID_SPECS)
    proc = _run_check(path)

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "broken" in proc.stdout
    assert "parse error" in proc.stdout
    assert "valid" in proc.stdout
    assert "slots:" in proc.stdout
    assert "keys:" in proc.stdout
    assert "qpos_joint1" in proc.stdout


# ---------------------------------------------------------------------------
# A compound spec mixing an author-typed pin with a selector-driven pin
# (the whole formula is desugared because of the gripper selector) must
# label each parameter on its own merits: "pinned" for the literally-typed
# robot_0, "expanded" — never "**" — for the gripper slot the selector
# collapsed onto entity 4.
# ---------------------------------------------------------------------------

_COMPOUND_PINNED_AND_EXPANDED_SPEC = """
specs:
- name: compound
  formula: 'G[0,5s](robot_0[qpos_joint1] <= 1.0 and gripper_*[all: recent_ee_force_* \\ recent_ee_force_norm] <= 5.0)'
"""


def test_compound_spec_labels_authored_pin_and_expansion_pin_separately(tmp_path):
    path = _write(tmp_path, "specs.yaml", _COMPOUND_PINNED_AND_EXPANDED_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert re.search(r"robot\s+robot_1\s+pinned\b", proc.stdout)
    assert re.search(r"gripper\s+gripper_1\s+expanded\b", proc.stdout)
    assert "**" not in proc.stdout


# ---------------------------------------------------------------------------
# semantic.py explicitly accepts capitalized Min(/Max( aggregators; the
# reading-line regex must too, not just lowercase min(/max(.
# ---------------------------------------------------------------------------

_CAPITALIZED_MAX_SPEC = """
specs:
- name: maxcap
  formula: "G[0,5s](Max(robot_0[qpos_joint1], robot_0[qvel_joint1]) <= 5.0)"
"""


def test_capitalized_max_prints_reading_line(tmp_path):
    path = _write(tmp_path, "specs.yaml", _CAPITALIZED_MAX_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "reading:" in proc.stdout
    assert "ALL must satisfy" in proc.stdout


# ---------------------------------------------------------------------------
# Passing a directory as the specs path must not raise a raw traceback —
# OSError (IsADirectoryError) is caught alongside the YAML/shape ValueError.
# ---------------------------------------------------------------------------

def test_directory_as_specs_path_exits_nonzero_without_traceback(tmp_path):
    proc = _run_check(tmp_path)

    assert proc.returncode != 0
    assert "Traceback" not in proc.stdout
    assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# A non-numeric constant value (e.g. ``constants: {LIMIT: "five"}``) makes
# ``compile_specs`` raise TypeError, not ValueError — ``_check_entry`` must
# catch both, or the raw traceback crashes the whole run and swallows every
# sibling entry's report (final-review finding 1).
# ---------------------------------------------------------------------------

_STRING_CONSTANT_SPEC = """
specs:
- name: valid_one
  formula: "G[0,5s](robot_0[qpos_joint1] <= 1.0)"
- name: bad_constant
  formula: "G[0,5s](robot_0[qpos_joint1] <= LIMIT)"
  constants: {LIMIT: "five"}
- name: valid_two
  formula: "G[0,5s](robot_0[qpos_joint1] <= 2.0)"
"""


def test_non_numeric_constant_type_error_does_not_swallow_siblings(tmp_path):
    path = _write(tmp_path, "specs.yaml", _STRING_CONSTANT_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stdout
    assert "Traceback" not in proc.stderr
    assert re.search(r"bad_constant[\s\S]*?error:", proc.stdout)
    # Both valid siblings still print their full slots/keys section —
    # the bad entry must not swallow them.
    assert proc.stdout.count("slots:") == 2
    assert proc.stdout.count("keys:") == 2
    assert "valid_one" in proc.stdout
    assert "valid_two" in proc.stdout


# ---------------------------------------------------------------------------
# The ALL/ANY reading must only fire when a whole comparison side IS an
# n-ary min(/max( call — not when min(/max( is merely the first token of
# larger arithmetic on that side (final-review finding 2).
# ---------------------------------------------------------------------------

_COMPOUND_ARITHMETIC_SPEC = """
specs:
- name: compound_arithmetic
  formula: "G[0,5s](min(robot_0[qpos_joint1], robot_0[qvel_joint1]) - max(robot_0[qpos_joint1], robot_0[qvel_joint1]) >= 0)"
"""

_PLAIN_MIN_SPEC = """
specs:
- name: plain_min
  formula: "G[0,5s](min(robot_0[qpos_joint1], robot_0[qvel_joint1]) >= 0)"
"""


def test_min_minus_max_prints_no_reading_line(tmp_path):
    path = _write(tmp_path, "specs.yaml", _COMPOUND_ARITHMETIC_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "reading:" not in proc.stdout


def test_plain_min_call_still_prints_reading_line(tmp_path):
    path = _write(tmp_path, "specs.yaml", _PLAIN_MIN_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "reading:" in proc.stdout
    assert "ALL must satisfy" in proc.stdout


# ---------------------------------------------------------------------------
# Legal ``**`` reductions must report terms, entity count, and components
# per entity.  Equal nested reducers are flattened for presentation because
# they have the same meaning as one reduction.  A one-entity component glob
# remains component-oriented, and repeated terms are not miscounted as
# distinct components.
# ---------------------------------------------------------------------------

_DOUBLE_STAR_READING_SPEC = """
specs:
- name: scene_mass
  formula: "G[0,5s](Max(Max(object_**[mass])) >= 0)"
"""

_PLAIN_COMPONENT_READING_SPEC = """
specs:
- name: force_axes
  formula: 'G[0,5s](Max(gripper_*[recent_ee_force_* \\ recent_ee_force_norm]) <= 5)'
"""

_NONUNIFORM_ENTITY_COMPONENTS_SPEC = """
specs:
- name: nonuniform
  formula: "G[0,5s](Max(object_1[mass], object_1[temperature], object_2[mass]) >= 0)"
"""

_REPEATED_COMPONENT_TERMS_SPEC = """
specs:
- name: repeated
  formula: "G[0,5s](Max(object_1[mass], object_1[mass]) >= 0)"
"""


def test_double_star_reading_counts_terms_entities_and_components(tmp_path):
    path = _write(tmp_path, "specs.yaml", _DOUBLE_STAR_READING_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "reading: max over 2 terms spanning 2 object entities × "
        "1 component per entity >= 0 — ANY must satisfy"
    ) in proc.stdout


def test_plain_component_glob_reading_stays_component_oriented(tmp_path):
    path = _write(tmp_path, "specs.yaml", _PLAIN_COMPONENT_READING_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "reading: max over 3 terms using 3 components <= 5 — "
        "ALL must satisfy"
    ) in proc.stdout
    assert "gripper entities" not in proc.stdout


def test_nonuniform_entity_components_report_per_entity_counts(tmp_path):
    path = _write(tmp_path, "specs.yaml", _NONUNIFORM_ENTITY_COMPONENTS_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "reading: max over 3 terms spanning 2 object entities; "
        "components per entity: object_1=2, object_2=1 >= 0 — "
        "ANY must satisfy"
    ) in proc.stdout


def test_repeated_terms_are_not_counted_as_distinct_components(tmp_path):
    path = _write(tmp_path, "specs.yaml", _REPEATED_COMPONENT_TERMS_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "reading: max over 2 terms using 1 component >= 0 — "
        "ANY must satisfy"
    ) in proc.stdout


# ---------------------------------------------------------------------------
# An author-typed labelled pin (explicit ``class{A}_<id>[`` or legacy
# ``class_A_<id>[``) must report "pinned", not "expanded" — there is no
# expands-to block for either, so "expanded" would be internally
# inconsistent (final-review finding 3).
# ---------------------------------------------------------------------------

_EXPLICIT_LABELLED_PIN_SPEC = """
specs:
- name: labelled_explicit
  formula: "G[0,5s](robot{A}_0[qpos_joint1] <= 1.0)"
"""

_LEGACY_LABELLED_PIN_SPEC = """
specs:
- name: labelled_legacy
  formula: "G[0,5s](robot_A_0[qpos_joint1] <= 1.0)"
"""


def test_explicit_labelled_pin_reports_pinned(tmp_path):
    path = _write(tmp_path, "specs.yaml", _EXPLICIT_LABELLED_PIN_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert re.search(r"robot\s+robot_1\s+pinned\b", proc.stdout)
    assert "expanded" not in proc.stdout


def test_legacy_labelled_pin_reports_pinned(tmp_path):
    path = _write(tmp_path, "specs.yaml", _LEGACY_LABELLED_PIN_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert re.search(r"robot\s+robot_1\s+pinned\b", proc.stdout)
    assert "expanded" not in proc.stdout


# ---------------------------------------------------------------------------
# A non-mapping spec entry (``specs: [hello]``) must not crash engine.py's
# iter_spec_entries with a raw AttributeError — it gets a per-entry error
# tuple instead, and specola check reports it cleanly (final-review finding 5).
# ---------------------------------------------------------------------------

_NON_MAPPING_ENTRY_SPEC = """
specs: [hello]
"""


def test_non_mapping_entry_reports_error_without_traceback(tmp_path):
    path = _write(tmp_path, "specs.yaml", _NON_MAPPING_ENTRY_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stdout
    assert "Traceback" not in proc.stderr
    assert "must be a mapping" in proc.stdout


# ---------------------------------------------------------------------------
# A wildcard slot reads as "every entity of this class" and silently is not
# whenever some class member lacks one of its keys. The mock's two objects
# share only "mass", so object_*[temperature] monitors one of two — that
# warns, without failing the run; object_*[mass] spans the class and stays
# quiet (issue #72).
# ---------------------------------------------------------------------------

_PARTIAL_WILDCARD_SPEC = """
specs:
- name: partial
  formula: "G[0,5s](object_*[temperature] <= 30.0)"
"""

_WHOLE_CLASS_WILDCARD_SPEC = """
specs:
- name: whole_class
  formula: "G[0,5s](object_*[mass] <= 1.0)"
"""

_PINNED_SPEC = """
specs:
- name: pinned
  formula: "G[0,5s](object_1[temperature] <= 30.0)"
"""

_EXPANDED_SPEC = """
specs:
- name: expanded
  formula: "G[0,5s](min(object_**[temperature]) <= 30.0)"
"""


def test_wildcard_binding_subset_of_class_warns_but_exits_zero(tmp_path):
    path = _write(tmp_path, "specs.yaml", _PARTIAL_WILDCARD_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "warning: object_* bound 1 of 2 object entities" in proc.stdout
    assert "not all expose temperature" in proc.stdout


def test_wildcard_binding_whole_class_prints_no_warning(tmp_path):
    path = _write(tmp_path, "specs.yaml", _WHOLE_CLASS_WILDCARD_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "object entities" not in proc.stdout


def test_pinned_slot_binding_one_entity_prints_no_warning(tmp_path):
    path = _write(tmp_path, "specs.yaml", _PINNED_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "object entities" not in proc.stdout


def test_expanded_slot_binding_one_entity_prints_no_warning(tmp_path):
    path = _write(tmp_path, "specs.yaml", _EXPANDED_SPEC)
    proc = _run_check(path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "object entities" not in proc.stdout


# ---------------------------------------------------------------------------
# The warning's remedy on a body-qualified adapter: with each object's keys
# spelled after its own body, no key is shared, but one glob still spans the
# class. The most literal spanning pattern wins (``xpos_*_main_2``, not
# ``xpos_*_2``), and a class that is genuinely heterogeneous gets no
# suggestion rather than the vacuous "*".
# ---------------------------------------------------------------------------

def _object(entity_id, properties):
    return EntityState(
        entity_id=entity_id,
        entity_class="object",
        position=np.zeros(3, dtype=float),
        velocity=np.zeros(3, dtype=float),
        track_age=0,
        track_state=TrackState.CONFIRMED,
        properties=dict(properties),
    )


def _body_qualified_catalog():
    """Three objects, RoboCasa-shaped: every key names its own body."""
    return ComponentCatalog.from_entities({
        entity_id: _object(
            entity_id,
            {f"xpos_{body}_{axis}": 0.0 for axis in range(3)},
        )
        for entity_id, body in enumerate(
            ("obj_main", "distr_counter_main", "distr_cab_main"), start=1
        )
    })


def _heterogeneous_catalog():
    """The mock scene's objects: one shared key, the rest unrelated."""
    return ComponentCatalog.from_entities({
        1: _object(1, {"mass": 0.25, "temperature": 22.0}),
        2: _object(2, {"mass": 0.40, "hinge_angle": 0.15}),
    })


def _wildcard(entity_class, components):
    return SpecTemplate(params=[ParameterSpec(
        param_key=(entity_class, 1),
        is_wildcard=True,
        required_components=list(components),
    )])


def test_covering_glob_prefers_the_most_literal_spanning_pattern():
    catalog = _body_qualified_catalog()

    assert _covering_glob(
        "xpos_obj_main_2", [1, 2, 3], catalog
    ) == "xpos_*_main_2"


def test_covering_glob_never_offers_the_bare_wildcard():
    catalog = _heterogeneous_catalog()

    assert _covering_glob("temperature", [1, 2], catalog) is None
    assert _covering_glob("hinge_angle", [1, 2], catalog) is None


def test_partial_binding_warning_names_a_glob_spanning_the_class():
    warnings = _partial_binding_warnings(
        _wildcard("object", ["xpos_obj_main_2"]), _body_qualified_catalog()
    )

    assert warnings == [
        "object_* bound 1 of 3 object entities (not all expose "
        "xpos_obj_main_2; a glob such as xpos_*_main_2 covers all 3)"
    ]


def test_partial_binding_warning_omits_a_remedy_it_cannot_name():
    warnings = _partial_binding_warnings(
        _wildcard("object", ["temperature"]), _heterogeneous_catalog()
    )

    assert warnings == [
        "object_* bound 1 of 2 object entities (not all expose temperature)"
    ]


def test_eligible_entities_applies_retained_component_tag_filters():
    entities = {
        1: _object(1, {"contact_force": 1.0}),
        2: _object(2, {"contact_force": 2.0}),
    }
    metadata = {
        1: {
            "contact_force": PropertyMeta(
                group="contact",
                tags=("role:target",),
            ),
        },
        2: {
            "contact_force": PropertyMeta(
                group="contact",
                tags=("role:distractor",),
            ),
        },
    }
    catalog = ComponentCatalog.from_entities(entities, metadata=metadata)
    param = ParameterSpec(
        param_key=("object", 1),
        is_wildcard=True,
        required_components=["contact_force"],
        component_tag_filters={
            "contact_force": ("role:distractor",),
        },
    )

    assert _eligible_entities(param, catalog) == [2]


def _dynamic_contact_capabilities():
    return AdapterCapabilities(
        spatial_dims=3,
        class_mode=VocabularyMode.CLOSED,
        supported_classes=frozenset({"object", "contact"}),
        components_by_class={
            "contact": frozenset({"contact_point_force_max"}),
        },
        component_mode_by_class={
            "contact": VocabularyMode.CLOSED,
        },
        runtime_populated_classes=frozenset({"contact"}),
        entity_id_mode=VocabularyMode.OPEN,
    )


def _loaded_spec(formula):
    return SimpleNamespace(
        name="binding_check",
        formula_text=formula,
        constants={},
        symmetric=False,
        backend="auto",
    )


def test_check_entry_rejects_static_class_injective_binding_gap(capsys):
    entities = {1: _object(1, {"mass": 1.0})}
    catalog = ComponentCatalog.from_entities(entities)
    spec = _loaded_spec(
        "object{A}_*[mass] > 0 and object{B}_*[mass] > 0 "
        "and contact_*[contact_point_force_max] > 0"
    )

    assert not _check_entry(
        spec,
        0.1,
        catalog,
        {},
        entities,
        None,
        capabilities=_dynamic_contact_capabilities(),
    )
    output = capsys.readouterr().out
    assert "reset bindings: 0" in output
    assert "static class slot(s)" in output
    assert "object" in output


def test_check_entry_rejects_never_bindable_static_slot(capsys):
    """A static class with no entity exposing the key is a compile error."""
    entities = {1: _object(1, {"mass": 1.0})}
    catalog = ComponentCatalog.from_entities(entities)
    spec = _loaded_spec("object_*[never_extracted] > 0")

    assert not _check_entry(
        spec,
        0.1,
        catalog,
        {},
        entities,
        None,
        capabilities=_dynamic_contact_capabilities(),
    )
    output = capsys.readouterr().out
    assert "error:" in output
    assert "static" in output
    assert "object" in output


def test_check_entry_allows_empty_declared_dynamic_class(capsys):
    catalog = ComponentCatalog.from_entities({})
    spec = _loaded_spec(
        "contact_*[contact_point_force_max] > 0"
    )

    assert _check_entry(
        spec,
        0.1,
        catalog,
        {},
        {},
        None,
        capabilities=_dynamic_contact_capabilities(),
    )
    output = capsys.readouterr().out
    assert "reset bindings: 0" in output
    assert "runtime-populated" in output
