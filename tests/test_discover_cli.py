"""specola discover: read-only discovery, no authoring or eval loop."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
import json
import io

import pytest


def test_discovery_renders_adapter_declared_authoring_warnings(capsys):
    from specola.cli_common import _MockAdapter, print_discovery

    adapter = _MockAdapter()
    observation = adapter.reset()
    adapter.discovery_warnings = lambda entities: (
        "raw channel disagrees with the declared authoritative value",
    )

    print_discovery(adapter, observation.perceived_entities)

    out = capsys.readouterr().out
    assert "Authoring warnings" in out
    assert "raw channel disagrees with the declared authoritative value" in out


def test_discover_prints_entities_and_exits():
    proc = subprocess.run(
        [sys.executable, "-m", "specola.discover", "--sim", "mock"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    assert "Entities" in proc.stdout
    assert "Example specification" in proc.stdout
    assert "constants: {threshold: 0.0}" in proc.stdout
    assert "robot" in proc.stdout and "object" in proc.stdout


def test_discover_help_exposes_full_field_contract_view():
    proc = subprocess.run(
        [sys.executable, "-m", "specola.discover", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "--field-contract" in proc.stdout


def test_discover_json_stdout_is_directly_parseable():
    proc = subprocess.run(
        [
            sys.executable, "-m", "specola.discover",
            "--sim", "mock", "--format", "json",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    document = json.loads(proc.stdout)
    assert document["schema_version"] == 1
    assert document["counts"]["before"]["entities"] == 5
    assert "Loading simulator adapter" in proc.stderr


def test_mujoco_field_contract_summary_keeps_classification_distinct(
    capsys,
):
    from specola.adapters.mujoco import MujocoFieldMaterialization
    from specola.adapters.mujoco_contract import FieldSurfaceReport
    from specola.cli_common import _print_mujoco_field_contract

    adapter = SimpleNamespace(
        field_surface_report=FieldSurfaceReport(
            included=("time", "ten_length"),
            excluded=("ncon",),
            unclassified=(),
            overlap=(),
        ),
        field_materialization={
            "time": MujocoFieldMaterialization(
                field_name="time",
                binding_present=True,
                scene_applicable=True,
                emitted_keys=("observation.time",),
            ),
            "ten_length": MujocoFieldMaterialization(
                field_name="ten_length",
                binding_present=True,
                scene_applicable=False,
                emitted_keys=(),
                reason="current reset model has no named tendon",
            ),
        },
    )

    _print_mujoco_field_contract(adapter, full=False)
    out = capsys.readouterr().out
    assert "63 global readers" in out
    assert "2 present on this binding" in out
    assert "1 scene-applicable" in out
    assert "1 materialized into 1 concrete sinks" in out
    assert "1 exclusions present on this binding" in out
    assert "0 unclassified" in out
    assert "Every scene-applicable reader is eagerly materialized" in out


def test_field_contract_reports_a_plan_refusal_on_its_own_line(capsys):
    """A refused gather plan is a performance note, not a capability one.

    The two must stay on separate lines: ``reason`` answers "what can
    this field extract here", the refusal answers "why is it still
    paying the scalar path" (issue #98).
    """
    from specola.adapters.mujoco import MujocoFieldMaterialization
    from specola.adapters.mujoco_contract import FieldSurfaceReport
    from specola.cli_common import _print_mujoco_field_contract

    refusal = "no gather plan: an owner was omitted from the first full pass"
    adapter = SimpleNamespace(
        field_surface_report=FieldSurfaceReport(
            included=("xpos",),
            excluded=(),
            unclassified=(),
            overlap=(),
        ),
        field_materialization={
            "xpos": MujocoFieldMaterialization(
                field_name="xpos",
                binding_present=True,
                scene_applicable=True,
                emitted_keys=("entity[0].properties.xpos_alpha_0",),
                reason="some owners were not materialized: alpha (gated)",
                plan_refusal=refusal,
            ),
        },
    )

    _print_mujoco_field_contract(adapter, full=True)

    out = capsys.readouterr().out
    assert refusal in out
    assert "some owners were not materialized" in out
    capability = [line for line in out.splitlines() if "capability:" in line]
    assert capability and all(
        "no gather plan" not in line for line in capability
    )


def _register_capture_adapter(captured):
    from specola.cli_common import _MockAdapter
    from specola.registry import register

    def factory(task, robot, target_only):
        captured["target_only"] = target_only
        return _MockAdapter()

    register(
        "capture-sim", factory,
        description="test capture", replace=True,
    )


def test_discover_defaults_to_target_only(monkeypatch, capsys):
    from specola import discover

    captured = {}
    _register_capture_adapter(captured)
    monkeypatch.setattr(
        sys, "argv",
        ["specola discover", "--sim", "capture-sim", "--no-pager"],
    )

    discover.main()
    capsys.readouterr()

    assert captured["target_only"] is True


def test_discover_all_regions_restores_full_discovery(monkeypatch, capsys):
    from specola import discover

    captured = {}
    _register_capture_adapter(captured)
    monkeypatch.setattr(
        sys, "argv",
        ["specola discover", "--sim", "capture-sim", "--no-pager",
         "--all-regions"],
    )

    discover.main()
    capsys.readouterr()

    assert captured["target_only"] is False


def test_discover_propagates_contextual_property_metadata_failure(
    monkeypatch,
):
    from specola import discover
    from specola.cli_common import _MockAdapter
    from specola.registry import register

    class BrokenMetadataAdapter(_MockAdapter):
        def property_metadata(self, entity_id):
            raise KeyError("missing discovery metadata")

    register(
        "broken-discovery-metadata",
        lambda task, robot, target_only: BrokenMetadataAdapter(),
        description="test adapter",
        replace=True,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "specola discover",
            "--sim",
            "broken-discovery-metadata",
            "--no-pager",
        ],
    )

    with pytest.raises(
        RuntimeError,
        match=r"property_metadata\(\) failed for adapter "
        r"BrokenMetadataAdapter entity 0",
    ) as exc_info:
        discover.main()

    assert isinstance(exc_info.value.__cause__, KeyError)


def test_plain_output_file_includes_requested_field_contract(
    monkeypatch,
    tmp_path,
    capsys,
):
    from specola import cli_common, discover

    output = tmp_path / "discovery.txt"
    monkeypatch.setattr(
        cli_common,
        "_print_mujoco_field_contract",
        lambda adapter, full: print("FIELD CONTRACT SENTINEL"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "specola discover",
            "--sim", "mock",
            "--field-contract",
            "--output", str(output),
        ],
    )

    discover.main()
    capsys.readouterr()

    assert "FIELD CONTRACT SENTINEL" in output.read_text()


def test_rich_discovery_lists_declared_dynamic_class_without_live_instance():
    from rich.console import Console

    from specola.cli_common import _MockAdapter
    from specola.discovery_filter import DiscoveryQuery, filter_discovery
    from specola.discovery_live import render_rich_discovery
    from specola.discovery_model import build_discovery_model

    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    adapter.capabilities = SimpleNamespace(
        components_by_class={"contact": frozenset({"force"})},
        runtime_populated_classes=frozenset({"contact"}),
        component_tags_by_class={"contact": frozenset({"robot"})},
    )
    model = build_discovery_model(
        adapter,
        entities,
        simulator="mock",
        task=None,
        effective_dt=adapter.dt,
    )
    view = filter_discovery(model, DiscoveryQuery(include_raw=False))
    stream = io.StringIO()

    render_rich_discovery(
        view,
        console=Console(file=stream, force_terminal=False, width=100),
    )

    assert "contact" in stream.getvalue()


@pytest.mark.parametrize(
    "module",
    [
        "specola.discover",
        "specola.run_monitor",
        "specola.check_specs",
        "specola.write_specs",
    ],
)
def test_all_regions_flag_replaces_target_only(module):
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    assert "--all-regions" in proc.stdout
    assert "--target-only" not in proc.stdout
