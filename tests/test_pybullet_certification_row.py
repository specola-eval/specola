"""Adversarial checks for the bounded PyBullet publication-gate evidence row."""

from __future__ import annotations

import copy
import inspect
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from _pins import assert_pin_matches

from specola.conformance import (
    AssuranceLabel,
    CriterionInteractionFact,
    NativeKindMode,
    ObligationResult,
    VariantRole,
    assemble_core_publication_rows,
    validate_refinement_evidence,
)


pytest.importorskip("pybullet", reason="PyBullet row needs the real PyBullet environment")
pytestmark = pytest.mark.pybullet

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "adapter_fidelity" / "pybullet_certification_row.py"
PIN = ROOT / "benchmarks" / "adapter_fidelity" / "evidence_pins" / "pybullet_3.2.7_certification_row_v1.json"
SPEC = importlib.util.spec_from_file_location("pybullet_certification_row", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
row = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = row
SPEC.loader.exec_module(row)


def _pinned():
    return json.loads(PIN.read_text(encoding="utf-8"))


def test_checked_row_is_canonical_and_exactly_regenerates_real_entry_point_evidence():
    pinned = _pinned()
    assert_pin_matches(PIN, row.render_json(pinned) + "\n")
    assert row.render_json(row.build_artifact(ROOT)) == row.render_json(pinned)


@pytest.mark.parametrize("forged_schema", (True, 1.0))
def test_raw_artifact_rejects_non_integer_schema_aliases(forged_schema):
    raw = row.build_raw_artifact(ROOT)
    raw["schema_version"] = forged_schema
    with pytest.raises(row.PyBulletCertificationRowError, match="unexpected raw PyBullet schema"):
        row.normalize_raw_artifact(raw, ROOT)


def test_row_has_exact_four_variant_denominator_and_discovery_metadata_projection_facts():
    durable = row.build_artifact(ROOT)
    variants = durable["projection_variants"]
    assert [item["variant_id"] for item in variants] == [
        "unscoped", "nonempty_scope", "empty_scope", "criterion_poison",
    ]
    assert [len(item["output"]) for item in variants] == [86, 8, 0, 8]
    assert all(item["interactions"] == {"reward": 0, "success": 0, "verdict": 0}
               for item in variants)
    counts = durable["raw_refinement"]["variant_native_read_counts"]
    assert counts[1]["getJointState"] == counts[1]["getLinkState"] == 1
    assert counts[2]["getContactPoints"] == 0
    assert counts[2]["getJointState"] == counts[2]["getLinkState"] == 0


def _old_scope_predicate(reference, actual, scope):
    """The prior row predicate: intentionally ignores non-key semantics."""
    assert {
        entity_id: entity["properties"] for entity_id, entity in actual["entities"].items()
    } == {
        entity_id: {
            key: value for key, value in entity["properties"].items()
            if key in scope.get(entity["entity_class"], frozenset())
        }
        for entity_id, entity in reference["entities"].items()
    }
    for entity_id, entity in actual["entities"].items():
        expected = reference["entities"][entity_id]
        assert entity["entity_class"] == expected["entity_class"]
        assert entity["position"] == expected["position"]
        assert entity["velocity"] == expected["velocity"]
        assert set(entity["properties"]) == set(entity["metadata"])


@pytest.mark.parametrize("mutation, top_level", [
    (lambda item, robot: item["entities"][robot]["metadata"]["basepos_0"].update(group="forged group"), "entities"),
    (lambda item, robot: item["entities"][robot].update(metadata_order=list(reversed(item["entities"][robot]["metadata_order"]))), "entities"),
    (lambda item, robot: item.update(time=item["time"] + 1.0), "time"),
    (lambda item, robot: item.update(timestep=item["timestep"] + 1), "timestep"),
    (lambda item, robot: item["entities"][robot].update(track_age=item["entities"][robot]["track_age"] + 1), "entities"),
    (lambda item, robot: item.update(component_tags={robot: {"basepos_0": ["forged"]}}), "component_tags"),
])
def test_exact_scope_projection_rejects_semantic_deltas_old_key_only_predicate_accepts(mutation, top_level):
    row.build_refinement_facts(ROOT)
    reference = row._ORACLE["get_observation"]
    actual = copy.deepcopy(row._project_observation_record(reference, row._SCOPE))
    robot = next(entity_id for entity_id, entity in actual["entities"].items() if entity["entity_class"] == "robot")
    mutation(actual, robot)
    _old_scope_predicate(reference, actual, row._SCOPE)
    with pytest.raises(row.PyBulletCertificationRowError, match="top-level field {0!r}".format(top_level)):
        row._assert_exact_scope_projection(reference, actual, row._SCOPE)


def test_criterion_poison_traps_truthiness_even_when_factory_does_not_use_it():
    poison = row._CriterionPoison()
    with pytest.raises(AssertionError, match="criterion poison was accessed: __bool__"):
        bool(poison)
    assert poison.interactions == ["__bool__"]


@pytest.mark.parametrize("operator_name", ("__eq__", "__ne__"))
def test_criterion_poison_traps_equality(operator_name):
    poison = row._CriterionPoison()
    comparison_target = object()
    with pytest.raises(AssertionError, match="criterion poison was accessed: {0}".format(operator_name)):
        if operator_name == "__eq__":
            poison == comparison_target
        else:
            poison != comparison_target
    assert poison.interactions == [operator_name]


def test_factory_ignored_parameter_source_audit_rejects_equality_and_identity_before_delete():
    authority = row.validate_entry_point_authority(
        row.resolve_entry_point_authority("pybullet", str(ROOT)), str(ROOT)
    )
    source = inspect.getsource(authority.factory)
    equality = source.replace(
        "    del robot, target_only",
        "    if robot == object():\n        pass\n    del robot, target_only",
    )
    with pytest.raises(row.PyBulletCertificationRowError, match="deleted before semantic code|semantic load"):
        row._audit_ignored_factory_parameter_source(equality)
    identity = source.replace(
        "    del robot, target_only",
        "    if target_only is not None:\n        pass\n    del robot, target_only",
    )
    with pytest.raises(row.PyBulletCertificationRowError, match="deleted before semantic code|semantic load"):
        row._audit_ignored_factory_parameter_source(identity)


def test_factory_ignored_parameter_proof_is_bound_to_current_authority_source():
    authority = row.validate_entry_point_authority(
        row.resolve_entry_point_authority("pybullet", str(ROOT)), str(ROOT)
    )
    audit = row._audit_ignored_factory_parameters(authority, ROOT)
    assert audit == _pinned()["factory_ignored_parameter_audit"]
    assert audit["parameters"] == ["robot", "target_only"]
    assert audit["semantic_load_count"] == {"robot": 0, "target_only": 0}
    assert audit["identity_checks_are_source_audited"] is True


def test_runtime_extension_custody_is_portable_and_rejects_forged_or_different_module(tmp_path):
    import importlib.metadata
    import pybullet

    native, _, _ = row._native_inventory(ROOT)
    custody = row._runtime_extension_custody(
        native, importlib.metadata.version("pybullet"), pybullet.getAPIVersion(), pybullet.__file__
    )
    assert custody == _pinned()["native_accessor_inventory"]["loaded_extension"]
    assert "runtime_module_path" not in json.dumps(_pinned())

    forged = tmp_path / Path(pybullet.__file__).name
    forged.write_bytes(b"forged extension bytes with same package/API declaration")
    with pytest.raises(row.PyBulletCertificationRowError, match="digest differs"):
        row._runtime_extension_custody(
            native, importlib.metadata.version("pybullet"), pybullet.getAPIVersion(), forged
        )
    different = tmp_path / "different-extension.so"
    different.write_bytes(Path(pybullet.__file__).read_bytes())
    with pytest.raises(row.PyBulletCertificationRowError, match="filename differs"):
        row._runtime_extension_custody(
            native, importlib.metadata.version("pybullet"), pybullet.getAPIVersion(), different
        )


def test_adapter_fidelity_gitignore_keeps_local_outputs_ignored_and_durable_log_tracked():
    def ignored(path):
        return subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", "--stdin"],
            cwd=ROOT, input=path + "\n", text=True, check=False,
        ).returncode == 0

    durable_log = "benchmarks/results/adapter_fidelity/logs/pybullet-certification-row-2026-08-16.log"
    assert ignored(durable_log)
    assert ignored("benchmarks/results/adapter_fidelity/raw-custody.json")
    assert subprocess.run(
        ["git", "ls-files", "--error-unmatch", durable_log],
        cwd=ROOT, capture_output=True, text=True, check=False,
    ).returncode == 0


def test_row_does_not_promote_partial_execution_to_certification():
    pinned = _pinned()
    certification = pinned["certification"]
    assert certification["status"] == "uncertified"
    assert certification["adapter_certification_constructed"] is False
    statuses = certification["obligation_status"]
    assert [item["obligation_id"] for item in statuses] == certification["obligation_denominator"]
    assert [item["result"] for item in statuses[:15]] == ["pass"] * 15
    assert [item["result"] for item in statuses[15:]] == ["unexecuted"] * 11
    assert all(item["applicable"] is True for item in statuses)
    assert all(item["assurance_labels"] == ["unverified"] for item in statuses[15:])
    bindability = statuses[14]
    assert bindability["obligation_id"] == "CORE.STATIC_SLOT_BINDABILITY"
    assert pinned["bindability"]["artifact"]["outcome"] == "pass"
    assert pinned["bindability"]["sha256"] in bindability["evidence_provenance"][0]
    assert len(pinned["native_accessor_inventory"]["callsite_candidate_denominator"]) == 27
    assert [item["gap_id"] for item in pinned["unresolved_gaps"]] == [
        "ACCEPTED_URDF_KIND_DENOMINATOR",
        "CONDITIONAL_CAPABILITY_CLOSURE",
        "FULL_PIPELINE_V2",
    ]


def test_raw_refinement_has_exact_fixed_scene_denominators_and_domain():
    inputs = row.build_refinement_facts(ROOT)
    _, inventory, manifest, execution, evidence = inputs
    case = evidence.cases[0]

    assert len(inventory.denominator.candidates) == len(manifest.fields) == 43
    assert case.denominator.candidate_denominator == manifest.field_denominator
    assert len(case.denominator.address_denominator) == 43
    assert all(field.runtime_owner is None for field in case.denominator.address_denominator)
    assert case.denominator.call_denominator == ("reset", "get_observation")
    assert len(case.denominator.witness_denominator) == len(case.witnesses) == 86
    assert len(execution.executions) == len(case.variants) == 4
    assert len({id(item.trace.bound_callables[0].driver)
                for item in execution.executions}) == 4
    native = json.loads(execution.executions[0].hooks[1].artifact.canonical_json)
    assert len(native["payload"]["fixed_field_values"]) == 43
    assert case.domain.native_kind_mode is NativeKindMode.EXTERNALLY_SUPPLIED
    assert case.domain.configuration_denominator == (row._ROW_ID,)
    assert case.domain.native_kind_denominator == case.domain.accepted_native_kinds
    assert case.rejection_denominator == case.rejections == ()
    validate_refinement_evidence(execution, evidence)


def test_raw_refinement_preserves_metadata_scope_reads_and_accessor_counts():
    inputs = row.build_refinement_facts(ROOT)
    case = inputs[4].cases[0]
    variants = case.variants

    assert tuple(item.role for item in variants) == tuple(VariantRole)
    assert tuple(len(item.output) for item in variants) == (86, 8, 0, 8)
    assert tuple(len(item.reads) for item in variants) == (86, 8, 0, 8)
    assert tuple(len(item.transfers) for item in variants) == (0, 0, 0, 0)
    assert {item.interactions for item in variants} == {
        CriterionInteractionFact(0, 0, 0)
    }
    assert variants[3].requested_fields == variants[1].requested_fields
    assert len({item.native_snapshot_sha256 for item in variants}) == 1

    fields = {item.candidate_name: item for item in inputs[2].fields}
    assert len(fields) == 43
    assert sum("Specola consistent-SI declaration" in item.value_transform
               for item in fields.values()) == 27
    assert sum("producer-established semantics" in item.value_transform
               for item in fields.values()) == 16
    assert all(item.unit and item.frame for item in fields.values())

    summary = row._refinement_summary(inputs, assemble_core_publication_rows(*inputs))
    assert summary["variant_native_read_counts"] == [
        row._expected_counts("unscoped"),
        row._expected_counts("nonempty_scope"),
        row._expected_counts("empty_scope"),
        row._expected_counts("nonempty_scope"),
    ]


def test_checkpoint_five_rows_are_exact_without_overall_certification():
    inputs = row.build_refinement_facts(ROOT)
    rows = assemble_core_publication_rows(*inputs)

    assert len(rows) == 26
    assert all(item.result is ObligationResult.PASS for item in rows[:14])
    assert all(item.result is ObligationResult.UNEXECUTED for item in rows[14:])
    assert all(item.applicable is True for item in rows)
    assert all(item.assurance_labels == (AssuranceLabel.UNVERIFIED,) for item in rows[14:])
    durable = row.build_artifact(ROOT)
    assert durable["raw_refinement"]["checkpoint5_row_count"] == 26
    assert durable["certification"]["status"] == "uncertified"
    assert durable["certification"]["adapter_certification_constructed"] is False


def test_validator_rejects_forged_variant_fact_or_forged_certification_status():
    forged = copy.deepcopy(_pinned())
    forged["raw_refinement"]["variant_native_read_counts"][2]["getJointState"] = 1
    with pytest.raises(row.PyBulletCertificationRowError, match="differs from regenerated"):
        row.validate_durable_row(forged, ROOT)
    forged = copy.deepcopy(_pinned())
    forged["certification"]["status"] = "certified"
    with pytest.raises(row.PyBulletCertificationRowError, match="differs from regenerated"):
        row.validate_durable_row(forged, ROOT)
