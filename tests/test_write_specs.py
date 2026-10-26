"""Compiler-authoritative tests for the ``specola write-specs`` CLI."""

from __future__ import annotations

from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Dict, List

import pytest
import yaml

from specola.adapter import MonitorObservation, PropertyMeta
from specola.capabilities import AdapterCapabilities, VocabularyMode
from specola.cli_common import _MockAdapter
from specola.components import ComponentCatalog
from specola.discovery_filter import DiscoveryQuery, filter_discovery
from specola.discovery_model import build_discovery_model
from specola.discovery_selector import (
    DiscoveryScope,
    selector_for_signal,
    signal_selections,
)
from specola.engine import MonitorEngine
from specola.monitor import enumerate_bindings
from specola.stl_parser import compile_specs


def _metadata(adapter, entities):
    from specola.discovery_model import property_metadata_for

    return property_metadata_for(adapter, entities)


def _context(adapter=None):
    adapter = adapter or _MockAdapter()
    entities = adapter.reset().perceived_entities
    metadata = _metadata(adapter, entities)
    catalog = ComponentCatalog.from_entities(entities, metadata=metadata)
    return adapter, entities, metadata, catalog


def test_write_specs_context_propagates_property_metadata_failure():
    adapter = _MockAdapter()

    def metadata(entity_id):
        raise ValueError("broken author metadata")

    adapter.property_metadata = metadata

    with pytest.raises(
        RuntimeError,
        match=r"property_metadata\(\) failed for adapter _MockAdapter entity 0",
    ) as exc_info:
        _context(adapter)

    assert isinstance(exc_info.value.__cause__, ValueError)


class _RoleAdapter(_MockAdapter):
    entity_roles: Dict[int, str]
    entity_role_basis: Dict[int, str]
    entity_role_provenance: Dict[int, str]

    def __init__(self):
        super().__init__()
        self.entity_roles = {1: "target", 2: "distractor"}
        self.entity_role_basis = {
            1: "benchmark_declaration",
            2: "benchmark_declaration",
        }
        self.entity_role_provenance = {
            1: "task.objects[0]",
            2: "task.objects[1]",
        }

    def property_metadata(self, entity_id):
        values = dict(super().property_metadata(entity_id))
        if entity_id in (1, 2):
            values["mass"] = PropertyMeta(
                group="mass",
                note="kg",
                tags=("role:" + self.entity_roles[entity_id],),
            )
        return values


def _role_adapter():
    return _RoleAdapter()


def _dynamic_capabilities():
    return AdapterCapabilities(
        spatial_dims=3,
        class_mode=VocabularyMode.CLOSED,
        supported_classes=frozenset({
            "robot", "gripper", "object", "region", "contact",
        }),
        components_by_class={
            "contact": frozenset({"contact_point_force_max"}),
        },
        component_mode_by_class={
            "contact": VocabularyMode.CLOSED,
        },
        runtime_populated_classes=frozenset({"contact"}),
        entity_id_mode=VocabularyMode.OPEN,
    )


class _DynamicAdapter(_MockAdapter):
    capabilities: AdapterCapabilities

    def __init__(self):
        super().__init__()
        self.capabilities = _dynamic_capabilities()


def test_parse_constants_uses_yaml_mapping_and_shared_validation():
    from specola.write_specs import _parse_constants

    assert _parse_constants("") == {}
    assert _parse_constants("{threshold: 0.25}") == {"threshold": 0.25}

    with pytest.raises(ValueError, match="YAML mapping"):
        _parse_constants("[1, 2]")
    with pytest.raises(TypeError, match="must be a real scalar"):
        _parse_constants("{threshold: true}")
    with pytest.raises(ValueError, match="reserved"):
        _parse_constants("{min: 1.0}")


def test_formula_acceptance_compiles_once_through_check_entry(
    monkeypatch,
    capsys,
):
    from specola import check_specs, write_specs

    adapter, entities, metadata, catalog = _context()
    calls = 0
    original = check_specs.compile_specs

    def counted_compile(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(check_specs, "compile_specs", counted_compile)

    accepted = write_specs._check_formula(
        name="joint_limit",
        formula="G[0,1s](robot_*[qpos_joint1] <= limit)",
        constants={"limit": 1.0},
        dt=adapter.dt,
        catalog=catalog,
        metadata=metadata,
        entities=entities,
        labels=adapter.entity_labels,
        capabilities=getattr(adapter, "capabilities", None),
    )

    output = capsys.readouterr().out
    assert accepted
    assert calls == 1
    assert "constants: {limit: 1.0}" in output
    assert "keys:" in output
    assert "qpos_joint1" in output
    assert "rad" in output


def test_role_selector_uses_declared_tag_and_has_one_binding(capsys):
    from specola.write_specs import _check_formula

    adapter, entities, metadata, catalog = _context(_role_adapter())

    accepted = _check_formula(
        name="target_mass",
        formula=(
            "G[0,1s](object_*[mass @ role:target] >= threshold)"
        ),
        constants={"threshold": 0.0},
        dt=adapter.dt,
        catalog=catalog,
        metadata=metadata,
        entities=entities,
        labels=adapter.entity_labels,
        capabilities=getattr(adapter, "capabilities", None),
    )

    output = capsys.readouterr().out
    assert accepted
    assert "role:target" in output
    assert "reset bindings: >=1" in output
    assert "mock_target_object[1]" in output
    assert "mock_distractor_object[2]" not in output


def test_group_filtered_quantifier_uses_selector_compiler(capsys):
    from specola.write_specs import _check_formula

    adapter, entities, metadata, catalog = _context()

    accepted = _check_formula(
        name="joint_angles",
        formula=(
            "G[0,1s]("
            "robot_*[all: qpos_* @ joint angle] <= limit"
            ")"
        ),
        constants={"limit": 1.0},
        dt=adapter.dt,
        catalog=catalog,
        metadata=metadata,
        entities=entities,
        labels=adapter.entity_labels,
        capabilities=getattr(adapter, "capabilities", None),
    )

    output = capsys.readouterr().out
    assert accepted
    assert "expands to:" in output
    assert "qpos_joint1" in output
    assert "qvel_joint1" not in output


def test_declared_runtime_class_compiles_and_waits(capsys):
    from specola.write_specs import _check_formula

    adapter = _DynamicAdapter()
    adapter, entities, metadata, catalog = _context(adapter)

    accepted = _check_formula(
        name="contact_limit",
        formula="contact_*[contact_point_force_max] <= threshold",
        constants={"threshold": 10.0},
        dt=adapter.dt,
        catalog=catalog,
        metadata=metadata,
        entities=entities,
        labels=adapter.entity_labels,
        capabilities=getattr(adapter, "capabilities", None),
    )

    output = capsys.readouterr().out
    assert accepted
    assert "reset bindings: 0" in output
    assert "runtime-populated" in output
    assert "compiles and waits" in output


def test_static_zero_binding_is_rejected(capsys):
    from specola.write_specs import _check_formula

    adapter, entities, metadata, catalog = _context()

    accepted = _check_formula(
        name="impossible_pair",
        formula=(
            "object{A}_*[mass] > 0 and "
            "object{B}_*[mass] > 0 and "
            "object{C}_*[mass] > 0"
        ),
        constants={},
        dt=adapter.dt,
        catalog=catalog,
        metadata=metadata,
        entities=entities,
        labels=adapter.entity_labels,
        capabilities=getattr(adapter, "capabilities", None),
    )

    output = capsys.readouterr().out
    assert not accepted
    assert "reset bindings: 0" in output
    assert "static class slot(s)" in output


def test_unknown_constant_and_component_are_rejected_verbatim(capsys):
    from specola.write_specs import _check_formula

    adapter, entities, metadata, catalog = _context()
    common = {
        "dt": adapter.dt,
        "catalog": catalog,
        "metadata": metadata,
        "entities": entities,
        "labels": adapter.entity_labels,
        "capabilities": getattr(adapter, "capabilities", None),
    }

    assert not _check_formula(
        name="unknown_constant",
        formula="robot_*[qpos_joint1] <= missing_limit",
        constants={},
        **common,
    )
    unknown_constant = capsys.readouterr().out
    assert "Unknown symbol(s) ['missing_limit']" in unknown_constant

    assert not _check_formula(
        name="unknown_component",
        formula="robot_*[qppos_joint1] <= 1.0",
        constants={},
        **common,
    )
    unknown_component = capsys.readouterr().out
    assert "qppos_joint1" in unknown_component
    assert "did you mean: qpos_joint1" in unknown_component


def test_glob_overmatch_is_visible_before_keep_decision(capsys):
    from specola.write_specs import _check_formula

    adapter, entities, metadata, _catalog = _context()
    entities[0].properties.update({
        "qfrc_actuator_robot0_joint1": 0.0,
        "qfrc_constraint_robot0_joint1": 0.0,
        "qfrc_applied_gripper0_right_finger_joint1": 0.0,
    })
    metadata[0] = dict(metadata[0])
    for key in (
        "qfrc_actuator_robot0_joint1",
        "qfrc_constraint_robot0_joint1",
        "qfrc_applied_gripper0_right_finger_joint1",
    ):
        metadata[0][key] = PropertyMeta(
            group="joint force",
            note="N·m",
        )
    catalog = ComponentCatalog.from_entities(entities, metadata=metadata)

    accepted = _check_formula(
        name="joint_force",
        formula="robot_*[qfrc_*_*_joint1] <= limit",
        constants={"limit": 1.0},
        dt=adapter.dt,
        catalog=catalog,
        metadata=metadata,
        entities=entities,
        labels=adapter.entity_labels,
        capabilities=getattr(adapter, "capabilities", None),
    )

    output = capsys.readouterr().out
    assert accepted
    assert "qfrc_actuator_robot0_joint1" in output
    assert "qfrc_constraint_robot0_joint1" in output
    assert "qfrc_applied_gripper0_right_finger_joint1" in output


def test_discovery_selector_round_trips_through_yaml_and_compiler(tmp_path):
    adapter, entities, metadata, catalog = _context()
    model = build_discovery_model(
        adapter,
        entities,
        simulator="mock",
        task=None,
        effective_dt=adapter.dt,
        property_metadata=metadata,
    )
    assert len(model.entities) == 5
    assert sum(len(entity.properties) for entity in model.entities) == 25

    view = filter_discovery(
        model,
        DiscoveryQuery(include_raw=False),
    )
    scope = DiscoveryScope("object")
    signal = next(
        item for item in signal_selections(view, scope)
        if item.pattern == "mass"
    )
    preview = selector_for_signal(view, scope, signal)
    assert preview.selector == "object_*[mass]"
    assert preview.binding_count == 2

    formula = "historically[0,1s]({} >= threshold)".format(preview.selector)
    constants = {"threshold": 0.0}
    compiled = compile_specs(
        formula,
        dt=adapter.dt,
        constants=constants,
        catalog=catalog,
        capabilities=getattr(adapter, "capabilities", None),
    )
    observation = MonitorObservation(0, 0.0, entities)
    initial_count = sum(
        len(enumerate_bindings(item, observation, require_confirmed=True))
        for item in compiled
    )
    assert initial_count == preview.binding_count == 2

    path = tmp_path / "round_trip.yaml"
    path.write_text(yaml.safe_dump({
        "specs": [{
            "name": "object_mass",
            "formula": formula,
            "constants": constants,
        }]
    }, sort_keys=False))
    entries = MonitorEngine.iter_spec_entries(path)
    assert len(entries) == 1
    _name, loaded, error = entries[0]
    assert error is None
    assert loaded is not None
    recompiled = compile_specs(
        loaded.formula_text,
        dt=adapter.dt,
        constants=loaded.constants,
        symmetric=loaded.symmetric,
        backend=loaded.backend,
        catalog=catalog,
        capabilities=getattr(adapter, "capabilities", None),
    )
    reloaded_count = sum(
        len(enumerate_bindings(item, observation, require_confirmed=True))
        for item in recompiled
    )
    assert reloaded_count == preview.binding_count


def test_discovered_reference_reaches_same_pre_step_monitor_without_inference(
    tmp_path, monkeypatch,
):
    from specola import Monitor, check_specs, write_specs
    from specola.discovery_serialize import discovery_document
    adapter, entities, metadata, catalog = _context()
    model = build_discovery_model(
        adapter,
        entities,
        simulator="mock",
        task=None,
        effective_dt=adapter.dt,
        property_metadata=metadata,
    )
    document = discovery_document(filter_discovery(
        model, DiscoveryQuery(include_raw=False)
    ))
    discovered_entity = next(
        entity for entity in document["entities"]
        if entity["id"] == 0 and entity["class"] == "robot"
    )
    discovered_property = next(
        prop for prop in discovered_entity["properties"]
        if prop["key"] == "qpos_joint1"
    )
    assert {
        key: discovered_property[key] for key in ("key", "group", "unit")
    } == {"key": "qpos_joint1", "group": "joint angle", "unit": "rad"}
    reference = "{}_*[{}]".format(
        discovered_entity["class"], discovered_property["key"]
    )
    formula = "historically[0,1s]({} <= limit)".format(reference)
    row = {"name": "joint_limit", "formula": formula,
           "constants": {"limit": 1.0}}
    compiled = []
    original_compile = check_specs.compile_specs
    def capture_compile(*args, **kwargs):
        values = original_compile(*args, **kwargs)
        compiled.append(values)
        return values

    monkeypatch.setattr(check_specs, "compile_specs", capture_compile)
    assert write_specs._check_formula(
        name=row["name"], formula=formula, constants=row["constants"],
        dt=adapter.dt, catalog=catalog, metadata=metadata, entities=entities,
        labels=adapter.entity_labels,
        capabilities=getattr(adapter, "capabilities", None),
    )
    path = tmp_path / "connected.yaml"
    write_specs._write_specs_document([row], str(path))
    assert yaml.safe_load(path.read_text()) == {"specs": [row]}
    entries = MonitorEngine.iter_spec_entries(path)
    assert len(entries) == 1
    name, loaded, error = entries[0]
    assert name == "joint_limit" and error is None and loaded is not None
    assert (loaded.formula_text, loaded.constants, loaded.backend,
            loaded.symmetric) == (formula, row["constants"], "auto", False)
    assert check_specs._check_entry(
        loaded, adapter.dt, catalog, metadata, entities,
        adapter.entity_labels,
        capabilities=getattr(adapter, "capabilities", None),
    )
    assert len(compiled) == 2 and all(len(values) == 1 for values in compiled)
    class NoInferenceAdapter(_MockAdapter):
        def _get_observation_impl(self):
            raise AssertionError("pre-step custody test requested an observation")

    session = Monitor(str(path), results_dir=None).watch(NoInferenceAdapter())
    manager = session.engine.managers["joint_limit"]

    def template_custody(template):
        return (
            template.name, template.desugared_text, template.rtamt_spec,
            template.backend, template.dt, template.window_steps,
            template.window_seconds, template.temporal_op, template.outer_op,
            template.settles_early, template.symmetric, template.grace_steps,
            template.grace_steps_override, template.neutral_values,
            tuple(sorted((template.predicate_fns or {}).keys())),
            tuple((
                param.param_key, param.is_wildcard, param.pinned_entity_id,
                tuple(param.required_components),
                tuple(sorted(param.component_tag_filters.items())),
            ) for param in template.params),
        )

    assert template_custody(compiled[0][0]) == template_custody(compiled[1][0])
    assert template_custody(compiled[1][0]) == template_custody(manager.formula)
    assert session.engine.manager_specs() == {"joint_limit": "joint_limit"}
    assert session.engine.monitor_counts() == {"joint_limit": 1}
    assert session.engine.skipped_specs() == {}
    assert session.engine.extraction_scope() == {
        "robot": frozenset({"qpos_joint1"})
    }
    assert manager.require_confirmed is True
    assert manager.current_bindings == set() and manager.active_instances == []
    assert (manager.n_created, manager.n_pruned, manager.n_recomputes) == (0, 0, 0)
    assert (manager.binding_set.n_full, manager.binding_set.n_incremental) == (0, 0)


def test_declared_role_preview_round_trips_to_one_binding(tmp_path):
    adapter, entities, metadata, catalog = _context(_role_adapter())
    model = build_discovery_model(
        adapter,
        entities,
        simulator="mock",
        task=None,
        effective_dt=adapter.dt,
        property_metadata=metadata,
    )
    view = filter_discovery(
        model,
        DiscoveryQuery(include_raw=False),
    )
    scope = DiscoveryScope("object", role="target")
    signal = next(
        item for item in signal_selections(view, scope)
        if item.pattern == "mass"
    )
    preview = selector_for_signal(view, scope, signal)
    assert preview.selector == "object_*[mass @ role:target]"
    assert preview.binding_count == 1

    formula = "historically[0,1s]({} >= threshold)".format(preview.selector)
    constants = {"threshold": 0.0}
    path = tmp_path / "role_round_trip.yaml"
    path.write_text(yaml.safe_dump({
        "specs": [{
            "name": "target_mass",
            "formula": formula,
            "constants": constants,
        }]
    }, sort_keys=False))
    entries = MonitorEngine.iter_spec_entries(path)
    assert len(entries) == 1
    _name, loaded, error = entries[0]
    assert error is None
    assert loaded is not None

    compiled = compile_specs(
        loaded.formula_text,
        dt=adapter.dt,
        constants=loaded.constants,
        symmetric=loaded.symmetric,
        backend=loaded.backend,
        catalog=catalog,
        capabilities=getattr(adapter, "capabilities", None),
    )
    observation = MonitorObservation(0, 0.0, entities)
    assert sum(
        len(enumerate_bindings(item, observation, require_confirmed=True))
        for item in compiled
    ) == preview.binding_count


def test_cli_writes_checkable_constants_and_model_backed_output(tmp_path):
    output_path = tmp_path / "authored.yaml"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "specola.write_specs",
            "--sim",
            "mock",
            "--output",
            str(output_path),
        ],
        input=(
            "Joint Limit\n"
            "G[0,1s](robot_*[qpos_joint1] <= limit)\n"
            "{limit: 1.0}\n"
            "\n"
            "\n"
            "\n"
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Specola Discovery — mock" in proc.stdout
    assert "Example specification" in proc.stdout
    assert "Dynamic scalar properties" not in proc.stdout
    assert "site_xpos_<site>_*" not in proc.stdout
    assert "Details: --class CLASS" not in proc.stdout
    assert (
        "specola discover --sim mock --class robot --verbose"
        in proc.stdout
    )
    constants_position = proc.stdout.index("constants: {limit: 1.0}")
    assert constants_position < proc.stdout.index("Keep this spec?")
    assert "Formula prompt:\n    historically[0,1s](" in proc.stdout
    assert "qpos_joint1" in proc.stdout
    assert "rad" in proc.stdout
    assert proc.stdout.rindex("specola check") < proc.stdout.rindex("specola run")

    document = yaml.safe_load(output_path.read_text())
    assert document == {
        "specs": [{
            "name": "joint_limit",
            "formula": "G[0,1s](robot_*[qpos_joint1] <= limit)",
            "constants": {"limit": 1.0},
        }]
    }

    checked = subprocess.run(
        [
            sys.executable,
            "-m",
            "specola.check_specs",
            "--sim",
            "mock",
            "--no-pager",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "reset bindings: >=1" in checked.stdout


def test_embedded_summary_redirects_large_family_guidance():
    from specola.discovery_render import render_discovery

    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    original = adapter.property_metadata
    extra = {}
    for index in range(25):
        key = "extra_signal_{}".format(index)
        entities[0].properties[key] = float(index)
        extra[key] = PropertyMeta(
            group="extra family {}".format(index),
            note="1",
        )

    def property_metadata(entity_id):
        values = dict(original(entity_id))
        if entity_id == 0:
            values.update(extra)
        return values

    adapter.property_metadata = property_metadata
    metadata = _metadata(adapter, entities)
    model = build_discovery_model(
        adapter,
        entities,
        simulator="mock",
        task=None,
        effective_dt=adapter.dt,
        property_metadata=metadata,
    )
    view = filter_discovery(
        model,
        DiscoveryQuery(include_raw=False),
    )

    output = render_discovery(view, include_cli_guidance=False)

    assert "use --group GLOB" not in output
    assert "specola discover --group GLOB" not in output
    assert "refine with the discovery command below" in output


def test_verbose_is_bounded_compatibility_notice(monkeypatch, capsys):
    from specola import write_specs

    monkeypatch.setattr(
        sys,
        "argv",
        ["specola write-specs", "--sim", "mock", "--verbose"],
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    write_specs.main()

    output = capsys.readouterr().out
    assert "specola discover" in output
    assert "--verbose" in output
    assert "Run the filtered discovery command above" in output
    assert "specola discover --verbose" not in output
    assert "Dynamic scalar properties" not in output


def test_discovery_command_preserves_adapter_context():
    from specola.write_specs import _discovery_command

    args = SimpleNamespace(
        sim="custom sim",
        adapter_module=["one.adapter", "two.adapter"],
        task="task with spaces",
        robot="robot name",
        all_regions=True,
        dt=0.125,
    )
    model = SimpleNamespace(entities=(
        SimpleNamespace(entity_class="object", raw=False),
    ))

    command = _discovery_command(args, model)

    assert command.startswith("specola discover --sim 'custom sim'")
    assert command.count("--adapter-module") == 2
    assert "--task 'task with spaces'" in command
    assert "--robot 'robot name'" in command
    assert "--all-regions" in command
    assert "--dt 0.125" in command
    assert "--class object --verbose" in command


def test_help_says_output_sets_the_save_prompt_default():
    proc = subprocess.run(
        [sys.executable, "-m", "specola.write_specs", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0
    assert "default at the interactive Save to prompt" in proc.stdout
    assert "Prompted interactively if not given" not in proc.stdout
    assert "specola discover --verbose" not in proc.stdout
    assert "compatibility flag for plain mode" in proc.stdout
    assert "exposes discovery detail interactively" in proc.stdout


def test_main_resets_adapter_once(monkeypatch, capsys):
    from specola import write_specs
    from specola.registry import register

    captured = SimpleNamespace(adapter=None)

    class CountingAdapter(_MockAdapter):
        def reset(self):
            self.reset_calls = getattr(self, "reset_calls", 0) + 1
            return super().reset()

        def property_metadata(self, entity_id):
            self.metadata_calls = getattr(self, "metadata_calls", 0) + 1
            return super().property_metadata(entity_id)

    def factory(task, robot, target_only):
        captured.adapter = CountingAdapter()
        return captured.adapter

    register(
        "write-specs-counting",
        factory,
        description="test adapter",
        replace=True,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["specola write-specs", "--sim", "write-specs-counting"],
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    write_specs.main()
    capsys.readouterr()

    assert captured.adapter.reset_calls == 1
    assert captured.adapter.metadata_calls == 5


def test_write_specs_main_propagates_contextual_metadata_failure(monkeypatch):
    from specola import write_specs
    from specola.registry import register

    class BrokenMetadataAdapter(_MockAdapter):
        def property_metadata(self, entity_id):
            raise KeyError("missing author metadata")

    register(
        "broken-write-metadata",
        lambda task, robot, target_only: BrokenMetadataAdapter(),
        description="test adapter",
        replace=True,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["specola write-specs", "--sim", "broken-write-metadata"],
    )

    with pytest.raises(
        RuntimeError,
        match=r"property_metadata\(\) failed for adapter "
        r"BrokenMetadataAdapter entity 0",
    ) as exc_info:
        write_specs.main()

    assert isinstance(exc_info.value.__cause__, KeyError)


def test_checked_formula_requires_keep_decision(monkeypatch):
    from specola import write_specs

    responses = iter((
        ("Mass Limit", False),
        ("object_*[mass] >= limit", False),
        ("{limit: 0.1}", False),
        ("n", False),
        ("", False),
        ("", False),
    ))
    monkeypatch.setattr(
        write_specs,
        "_prompt",
        lambda *_args, **_kwargs: next(responses),
    )
    monkeypatch.setattr(write_specs, "_check_formula", lambda **_kwargs: True)

    specs = write_specs._interactive_spec_writer(
        dt=0.05,
        catalog=None,
        metadata={},
        entities={},
        labels=None,
        capabilities=None,
    )

    assert specs == []


def test_duplicate_normalized_name_is_reprompted(monkeypatch, capsys):
    from specola import write_specs

    responses = iter((
        ("Mass Limit", False),
        ("object_*[mass] >= limit", False),
        ("{limit: 0.1}", False),
        ("y", False),
        ("mass limit", False),
        ("Temperature Limit", False),
        ("object_*[temperature] <= limit", False),
        ("{limit: 30}", False),
        ("y", False),
        ("", False),
    ))
    monkeypatch.setattr(
        write_specs,
        "_prompt",
        lambda *_args, **_kwargs: next(responses),
    )
    monkeypatch.setattr(write_specs, "_check_formula", lambda **_kwargs: True)

    specs = write_specs._interactive_spec_writer(
        dt=0.05,
        catalog=None,
        metadata={},
        entities={},
        labels=None,
        capabilities=None,
    )

    assert [spec["name"] for spec in specs] == [
        "mass_limit",
        "temperature_limit",
    ]
    assert "already used" in capsys.readouterr().out


def test_invalid_constants_reprompt_without_reimplementing_rules(monkeypatch):
    from specola import write_specs

    responses = iter((
        ("Mass Limit", False),
        ("object_*[mass] >= limit", False),
        ("{limit: true}", False),
        ("{limit: 0.1}", False),
        ("", False),
        ("", False),
    ))
    monkeypatch.setattr(
        write_specs,
        "_prompt",
        lambda *_args, **_kwargs: next(responses),
    )
    accepted = []

    def accept(**kwargs):
        accepted.append(kwargs)
        return True

    monkeypatch.setattr(write_specs, "_check_formula", accept)

    specs = write_specs._interactive_spec_writer(
        dt=0.05,
        catalog=None,
        metadata={},
        entities={},
        labels=None,
        capabilities=None,
    )

    assert specs == [{
        "name": "mass_limit",
        "formula": "object_*[mass] >= limit",
        "constants": {"limit": 0.1},
    }]
    assert accepted[0]["constants"] == {"limit": 0.1}


def test_followup_commands_preserve_compilation_arguments():
    from specola.write_specs import _followup_commands

    args = SimpleNamespace(
        sim="custom sim",
        adapter_module=["one.adapter", "two.adapter"],
        task="task with spaces",
        robot="robot name",
        all_regions=True,
        dt=0.125,
    )

    check, run = _followup_commands(args, "my specs.yaml")

    for command in (check, run):
        assert "--sim 'custom sim'" in command
        assert command.count("--adapter-module") == 2
        assert "--task 'task with spaces'" in command
        assert "--robot 'robot name'" in command
        assert "--all-regions" in command
        assert "--dt 0.125" in command
    assert "--no-pager 'my specs.yaml'" in check
    assert "--specs 'my specs.yaml'" in run


@pytest.mark.parametrize("error", (EOFError, KeyboardInterrupt))
def test_prompt_interrupt_is_clean(monkeypatch, capsys, error):
    from specola.write_specs import _prompt

    def interrupted(_prompt):
        raise error()

    monkeypatch.setattr("builtins.input", interrupted)

    assert _prompt("Formula", default="fallback") == ("fallback", True)
    assert capsys.readouterr().out == "\n"


def test_save_existing_file_skip_and_overwrite(monkeypatch, tmp_path):
    from specola import write_specs

    path = tmp_path / "specs.yaml"
    path.write_text("original\n")
    spec: List[Dict[str, object]] = [
        {"name": "mass", "formula": "object_*[mass] > 0"},
    ]

    responses = iter(((str(path), False), ("s", False)))
    monkeypatch.setattr(write_specs, "_prompt", lambda *_args, **_kwargs: next(responses))
    assert write_specs._save_specs(spec, str(path)) is None
    assert path.read_text() == "original\n"

    responses = iter(((str(path), False), ("o", False)))
    monkeypatch.setattr(write_specs, "_prompt", lambda *_args, **_kwargs: next(responses))
    assert write_specs._save_specs(spec, str(path)) == str(path)
    assert yaml.safe_load(path.read_text()) == {"specs": spec}


def test_save_rename_adds_yaml_extension(monkeypatch, tmp_path):
    from specola import write_specs

    existing = tmp_path / "specs.yaml"
    existing.write_text("original\n")
    renamed = tmp_path / "renamed"
    spec: List[Dict[str, object]] = [
        {"name": "mass", "formula": "object_*[mass] > 0"},
    ]
    responses = iter((
        (str(existing), False),
        ("r", False),
        (str(renamed), False),
    ))
    monkeypatch.setattr(
        write_specs,
        "_prompt",
        lambda *_args, **_kwargs: next(responses),
    )

    saved = write_specs._save_specs(spec, str(existing))

    assert saved == str(renamed) + ".yaml"
    assert yaml.safe_load((tmp_path / "renamed.yaml").read_text()) == {
        "specs": spec,
    }


def test_save_rename_reprompts_when_second_path_exists(monkeypatch, tmp_path):
    from specola import write_specs

    existing = tmp_path / "specs.yaml"
    also_existing = tmp_path / "also.yaml"
    final = tmp_path / "final.yaml"
    existing.write_text("original\n")
    also_existing.write_text("other\n")
    spec: List[Dict[str, object]] = [
        {"name": "mass", "formula": "object_*[mass] > 0"},
    ]
    responses = iter((
        (str(existing), False),
        ("r", False),
        (str(also_existing), False),
        (str(final), False),
    ))
    monkeypatch.setattr(
        write_specs,
        "_prompt",
        lambda *_args, **_kwargs: next(responses),
    )

    saved = write_specs._save_specs(spec, str(existing))

    assert saved == str(final)
    assert yaml.safe_load(final.read_text()) == {"specs": spec}


def test_save_rename_default_is_a_non_colliding_path(monkeypatch, tmp_path):
    from specola import write_specs

    existing = tmp_path / "specs.yaml"
    second = tmp_path / "specs_2.yaml"
    existing.write_text("original\n")
    second.write_text("also original\n")
    spec: List[Dict[str, object]] = [
        {"name": "mass", "formula": "object_*[mass] > 0"},
    ]
    prompted_defaults = []

    def prompt(label, default=""):
        if label == "Save to":
            return str(existing), False
        if "[o]verwrite" in label:
            return "r", False
        assert label == "New filename"
        prompted_defaults.append(default)
        return default, False

    monkeypatch.setattr(write_specs, "_prompt", prompt)

    saved = write_specs._save_specs(spec, str(existing))

    expected = tmp_path / "specs_3.yaml"
    assert prompted_defaults == [str(expected)]
    assert saved == str(expected)
    assert yaml.safe_load(expected.read_text()) == {"specs": spec}


def test_atomic_overwrite_preserves_existing_file_on_yaml_failure(
    monkeypatch,
    tmp_path,
):
    from specola import write_specs

    path = tmp_path / "specs.yaml"
    path.write_text("original\n")
    spec: List[Dict[str, object]] = [
        {"name": "mass", "formula": "object_*[mass] > 0"},
    ]

    def fail_dump(*_args, **_kwargs):
        raise yaml.YAMLError("serialization failed")

    monkeypatch.setattr(write_specs.yaml, "safe_dump", fail_dump)

    with pytest.raises(yaml.YAMLError, match="serialization failed"):
        write_specs._write_specs_document(
            spec,
            str(path),
            overwrite=True,
        )

    assert path.read_text() == "original\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_save_respects_creation_mode_and_preserves_overwrite_mode(
    tmp_path,
):
    from specola.write_specs import _write_specs_document

    reference = tmp_path / "reference.yaml"
    reference.write_text("reference\n")
    created = tmp_path / "created.yaml"
    existing = tmp_path / "existing.yaml"
    existing.write_text("original\n")
    existing.chmod(0o640)
    spec: List[Dict[str, object]] = [
        {"name": "mass", "formula": "object_*[mass] > 0"},
    ]

    _write_specs_document(spec, str(created), overwrite=False)
    _write_specs_document(spec, str(existing), overwrite=True)

    assert stat.S_IMODE(created.stat().st_mode) == stat.S_IMODE(
        reference.stat().st_mode
    )
    assert stat.S_IMODE(existing.stat().st_mode) == 0o640


def test_atomic_save_supports_long_valid_destination_name(tmp_path):
    from specola.write_specs import _write_specs_document

    path = tmp_path / ("s" * 240 + ".yaml")
    spec: List[Dict[str, object]] = [
        {"name": "mass", "formula": "object_*[mass] > 0"},
    ]

    _write_specs_document(spec, str(path), overwrite=False)

    assert yaml.safe_load(path.read_text()) == {"specs": spec}


def test_plain_save_error_can_retry_without_losing_specs(
    monkeypatch,
    tmp_path,
):
    from specola import write_specs

    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    spec: List[Dict[str, object]] = [
        {"name": "mass", "formula": "object_*[mass] > 0"},
    ]
    responses = iter((
        (str(first), False),
        (str(second), False),
    ))
    monkeypatch.setattr(
        write_specs,
        "_prompt",
        lambda *_args, **_kwargs: next(responses),
    )
    original = write_specs._write_specs_document
    calls = []

    def fail_once(specs, path, *, overwrite=True):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("disk unavailable")
        return original(specs, path, overwrite=overwrite)

    monkeypatch.setattr(
        write_specs,
        "_write_specs_document",
        fail_once,
    )

    saved = write_specs._save_specs(spec, str(first))

    assert saved == str(second)
    assert not first.exists()
    assert yaml.safe_load(second.read_text()) == {"specs": spec}


def test_plain_save_race_reopens_collision_choice(
    monkeypatch,
    tmp_path,
):
    from specola import write_specs

    requested = tmp_path / "specs.yaml"
    renamed = tmp_path / "renamed.yaml"
    spec: List[Dict[str, object]] = [
        {"name": "mass", "formula": "object_*[mass] > 0"},
    ]
    responses = iter((
        (str(requested), False),
        ("r", False),
        (str(renamed), False),
    ))
    monkeypatch.setattr(
        write_specs,
        "_prompt",
        lambda *_args, **_kwargs: next(responses),
    )
    original = write_specs._write_specs_document
    raced = {"done": False}

    def create_then_write(specs, path, *, overwrite=True):
        if not overwrite and not raced["done"]:
            raced["done"] = True
            Path(path).write_text("created by another process\n")
        return original(specs, path, overwrite=overwrite)

    monkeypatch.setattr(
        write_specs,
        "_write_specs_document",
        create_then_write,
    )

    saved = write_specs._save_specs(spec, str(requested))

    assert saved == str(renamed)
    assert requested.read_text() == "created by another process\n"
    assert yaml.safe_load(renamed.read_text()) == {"specs": spec}
