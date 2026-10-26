"""Check completed-array runtime joins with independent synthetic counts.

These cases exercise conformance schemas and joins. They do not execute a
producer or replace the native fixture's owner-addressed numerical oracle.
"""

from dataclasses import replace

import numpy as np
import pytest

from specola.adapter import MonitorObservation
from specola.adapters import mujoco_completed_population as population_module
from specola.conformance.extraction import ExtractionManifest, manifest_digest
from specola.conformance.instances import RuntimePopulation, InstanceValidationError, validate_materialization
from specola.extraction_observation import ExtractionObservation
from specola.native_values import NativeValue
from specola.query_config import QueryConfiguration

from test_mujoco_completed_fields import fixture, CONFIGURATION, VERSIONS


def joined_fixture(monkeypatch):
    subject, snapshot, _, _ = fixture(monkeypatch, VERSIONS[0])
    # These are the synthetic model declarations, specified before capture.
    counts = {"nbody": 3, "njnt": 2, "ngeom": 4, "nsite": 0, "ncam": 1, "nlight": 0}
    calls = []

    class Storage:
        def __init__(self, model, data, guard, *, _profile):
            self._buffers = {"owned": "unchanged"}
            self._bindings = {}
            for row in _profile["fields"]:
                for plan in row["plans"]:
                    shape = tuple(np.prod([factor if type(factor) is int else
                                           counts[factor["field"].rsplit("@", 1)[1]] for factor in dimension]).item()
                                  for dimension in plan["binding_dimensions"])
                    address = row["field_id"], plan["root"], tuple(plan["native_path"])
                    self._bindings[address] = (1, np.prod(shape) * 8, shape, shape, np.dtype(np.float64).str)

        def _field_regions(self):
            calls.append("layout")
            return dict(self._bindings)

        def _root_integer(self, root, field):
            assert root == "mjModel_"
            return counts[field]

        def _buffer_domains(self):
            return dict(self._buffers)

        def _validate_subject(self):
            calls.append("guard")

        def read(self):
            pytest.fail("population resolution read an observation payload")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(population_module, "MujocoNativeFields", Storage)
    bindings = {quantity.quantity_id: "synthetic.completion." + quantity.quantity_id
                for quantity in subject.quantities}
    resolved = population_module.completed_population(subject._model, snapshot.data, subject._guard,
                                                       subject.quantities, CONFIGURATION, bindings)
    assert calls[-2:] == ["guard", "close"]
    captured = subject.capture(snapshot, CONFIGURATION)
    samples = tuple(replace(sample, binding_id=bindings[sample.quantity_id]) for sample in captured.samples)
    manifest = ExtractionManifest("synthetic completed-array runtime relation", "2" * 64,
                                  (), subject.quantities, ())
    population = RuntimePopulation("2" * 64, manifest_digest(manifest), QueryConfiguration.empty(), 0, 0,
                                   resolved.domains, resolved.instances, query_domains=resolved.query_domains)
    extraction = ExtractionObservation(0, 0.0, 0, 0, population.producer_surface_sha256,
                                        population.manifest_sha256, population.queries,
                                        captured.metadata, samples)
    observation = MonitorObservation(0, 0.0, {}, extraction=extraction)
    validate_materialization(manifest, population, observation)
    return manifest, population, observation


@pytest.mark.parametrize("damage", ["owner_missing", "owner_duplicate", "owner_reordered", "row_missing",
                                    "native_address", "configuration"])
def test_runtime_layout_rejects_owner_address_and_configuration_damage(monkeypatch, damage):
    manifest, population, observation = joined_fixture(monkeypatch)
    assert observation.extraction is not None
    sample = next(sample for sample in observation.extraction.samples if sample.quantity_id.endswith(".xpos"))
    value = sample.value.restore()
    if damage == "owner_missing":
        value["owner_indices"] = (0, 1)
    elif damage == "owner_duplicate":
        value["owner_indices"] = (0, 1, 1)
    elif damage == "owner_reordered":
        value["owner_indices"] = (1, 0, 2)
    elif damage == "row_missing":
        value["value"] = value["value"][:2]
    elif damage == "native_address":
        value["native_path"] = ("wrong_field",)
    else:
        value["configuration_identity"]["sha256"] = "3" * 64
    altered = replace(sample, value=NativeValue.capture(value))
    samples = tuple(altered if item is sample else item for item in observation.extraction.samples)
    observation.extraction = replace(observation.extraction, samples=samples)
    with pytest.raises(InstanceValidationError, match="layout differs"):
        validate_materialization(manifest, population, observation)


@pytest.mark.parametrize("damage", ["missing", "component", "dtype", "prefix"])
def test_runtime_relation_requires_the_intermediate_energy_result(monkeypatch, damage):
    manifest, population, observation = joined_fixture(monkeypatch)
    extraction = observation.extraction
    assert extraction is not None
    sample = next(item for item in extraction.samples if item.quantity_id.endswith(".energy"))
    value = sample.value.restore()
    if damage == "missing":
        del value["stage_results"]
    elif damage == "component":
        value["stage_results"]["mj_energyPos"]["value"] = np.zeros(1, dtype=np.float64)
    elif damage == "dtype":
        value["stage_results"]["mj_energyPos"]["value"] = np.zeros(2, dtype=np.float32)
    else:
        value["stage_results"]["mj_energyPos"]["completion"] = value["completion"]
    altered = replace(sample, value=NativeValue.capture(value))
    observation.extraction = replace(extraction, samples=tuple(
        altered if item is sample else item for item in extraction.samples))
    with pytest.raises(InstanceValidationError, match="layout differs"):
        validate_materialization(manifest, population, observation)


def test_zero_physical_owners_still_require_the_root_quantity(monkeypatch):
    manifest, population, observation = joined_fixture(monkeypatch)
    assert observation.extraction is not None
    empty = next(sample for sample in observation.extraction.samples if sample.quantity_id.endswith(".site_xpos"))
    assert empty.value.restore()["value"].shape == (0, 3)
    assert empty.value.restore()["owner_indices"] == ()
    observation.extraction = replace(observation.extraction,
                                     samples=tuple(sample for sample in observation.extraction.samples if sample is not empty))
    with pytest.raises(InstanceValidationError, match="missing="):
        validate_materialization(manifest, population, observation)


def test_numeric_row_ownership_needs_the_separate_native_oracle(monkeypatch):
    manifest, population, observation = joined_fixture(monkeypatch)
    assert observation.extraction is not None
    sample = next(sample for sample in observation.extraction.samples if sample.quantity_id.endswith(".xpos"))
    value = sample.value.restore()
    value["value"] = value["value"][[1, 0, 2]]
    samples = tuple(replace(sample, value=NativeValue.capture(value)) if item is sample else item
                    for item in observation.extraction.samples)
    observation.extraction = replace(observation.extraction, samples=samples)
    # Layout validation deliberately makes no numerical owner-fidelity claim.
    validate_materialization(manifest, population, observation)
