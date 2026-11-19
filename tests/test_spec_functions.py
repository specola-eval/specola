"""Normal specification loading, scoped evaluation and saved-scene checks."""
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Mapping
import json

import numpy as np
import pytest
import yaml

from specola.adapter import SimulatorAdapter, MonitorObservation, PropertyMeta
from specola.entities import EntityState, TrackState
from specola.engine import MonitorEngine
from specola.session import Monitor
from specola.spec_functions import FunctionInputs, immutable


FUNCTION = '''
from specola.spec_functions import FunctionTarget
calls = []
def prepare(*, inputs, parameters):
    return [FunctionTarget("sample", entity_id=0, label="sample")]
def calculate(*, inputs, parameters, cache, targets):
    calls.append((inputs["sample"][0], cache.get("seen", 0)))
    cache["seen"] = cache.get("seen", 0) + 1
    value = inputs["sample"][0]
    if value == -99:
        return {}
    return {"sample": {"derived": value * parameters["factor"]}}
calculate.prepare = prepare
'''


class InputsAdapter(SimulatorAdapter):
    function_input_schemas = {"test.samples": 1}

    @property
    def dt(self):
        return 0.1

    @property
    def spatial_dims(self):
        return 3

    @property
    def supported_classes(self):
        return frozenset({"object"})

    def __init__(self):
        super().__init__()
        self.value = np.array([2.0])
        self.index = 0
        self.revision = 0
        self.captures = 0

    def _reset_impl(self):
        self.index = 0
        self.revision += 1
        return self._read()

    def _get_observation_impl(self):
        self.index += 1
        return self._read()

    def _read(self):
        inputs = None
        if getattr(self, "_function_input_requests", {}):
            self.captures += 1
            values = immutable({"test.samples": self.value})
            assert isinstance(values, Mapping)
            inputs = FunctionInputs(self.revision, values)
        return MonitorObservation(self.index, self.index * self.dt, {
            0: EntityState(entity_id=0, entity_class="object", position=np.zeros(3),
                           track_state=TrackState.CONFIRMED, track_age=self.index,
                           properties={"native": 1.0})
        }, function_inputs=inputs)

    def property_metadata(self, entity_id):
        return {"native": PropertyMeta(group="native", unit="m")}


def specification(tmp_path, *, formula="object_0[derived] >= 1", functions=True):
    source = tmp_path / "local.py"
    source.write_text(FUNCTION)
    document: dict = {"specs": [{"name": "one", "formula": formula}, {"name": "two", "formula": formula}]}
    if functions:
        document["functions"] = {"twice": {
            "python": "local.py:calculate", "entity_class": "object",
            "inputs": {"sample": {"provider": "test.samples", "schema": 1}},
            "outputs": {"derived": {"unit": "m"}}, "parameters": {"factor": 2},
        }}
    path = tmp_path / "specs.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def test_yaml_session_shares_outputs_preserves_inputs_and_invalidates_cache(tmp_path):
    path = specification(tmp_path)
    adapter = InputsAdapter()
    monitor = Monitor(str(path), results_dir=None)
    session = monitor.watch(adapter)
    implementation = monitor._engine.functions.definitions[0].implementation
    assert implementation.__globals__["calls"] == []  # Preparation never evaluates.
    first = session.step()
    assert [ev.robustness for ev in first.raw.evaluations] == [3.0, 3.0]
    observation = adapter.last_observation
    assert observation is not None
    retained = observation.function_inputs
    assert isinstance(retained, FunctionInputs)
    adapter.value[0] = 3
    assert retained.values["test.samples"][0] == 2
    with pytest.raises(ValueError):
        retained.values["test.samples"].flags.writeable = True
    session.step()
    assert implementation.__globals__["calls"] == [(2, 0), (3, 1)]
    adapter.revision += 1
    session.step()
    assert implementation.__globals__["calls"][-1] == (3, 0)
    monitor.watch(adapter).step()
    assert implementation.__globals__["calls"][-1] == (3, 0)


def test_unavailable_input_suspends_instead_of_passing(tmp_path):
    adapter = InputsAdapter()
    session = Monitor(str(specification(tmp_path, formula="historically[0,0.3s](object_0[derived] >= 1)")), results_dir=None).watch(adapter)
    session.step()
    adapter.value[0] = -99
    unavailable = session.step()
    assert len(unavailable.raw.evaluations) == 2
    assert all(ev.robustness is None for ev in unavailable.raw.evaluations)


def test_nonfinite_function_result_is_a_named_error(tmp_path):
    adapter = InputsAdapter()
    session = Monitor(str(specification(tmp_path)), results_dir=None).watch(adapter)
    adapter.value[0] = np.inf
    with pytest.raises(ValueError, match="function twice failed.*finite scalar"):
        session.step()


def test_unused_declaration_never_requests_native_inputs(tmp_path):
    adapter = InputsAdapter()
    monitor = Monitor(str(specification(tmp_path, formula="object_0[native] >= 0")), results_dir=None)
    session = monitor.watch(adapter)
    session.step()
    assert adapter.captures == 0
    assert monitor._engine.functions.definitions == ()


def test_missing_provider_fails_before_adapter_reset(tmp_path):
    adapter = InputsAdapter()
    adapter.function_input_schemas = {}
    with pytest.raises(ValueError, match="does not provide function input"):
        Monitor(str(specification(tmp_path)), results_dir=None).watch(adapter)
    assert adapter.revision == 0


def test_bare_engine_refuses_unprepared_functions(tmp_path):
    from specola.capabilities import reset_adapter
    from specola.entities import SpecConfigError
    engine = MonitorEngine.load(str(specification(tmp_path)))
    reset = reset_adapter(InputsAdapter())
    with pytest.raises(SpecConfigError, match="never prepared against an adapter"):
        engine.start_episode(dt=0.1, capabilities=reset.capabilities, catalog=reset.discovered_catalog)


def test_output_collision_fails_at_preparation(tmp_path):
    path = specification(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["functions"]["twice"]["outputs"] = {"native": {}}
    document["specs"][0]["formula"] = "object_0[native] > 0"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="output collision"):
        Monitor(str(path), results_dir=None).watch(InputsAdapter())


def test_provenance_retains_formula_and_functions_without_bindings(tmp_path):
    path = specification(tmp_path, formula="object_900[derived] > threshold")
    document = yaml.safe_load(path.read_text())
    for spec in document["specs"]:
        spec["constants"] = {"threshold": 7}
    path.write_text(yaml.safe_dump(document))
    engine = MonitorEngine.load(str(path))
    metadata = engine.spec_provenance
    assert json.loads(metadata["spec_formulas"])[0]["formula"] == document["specs"][0]["formula"]
    functions = json.loads(metadata["spec_functions"])
    assert Path(functions["twice"]["python"].rsplit(":", 1)[0]).is_absolute()
    assert len(functions["twice"]["sha256"]) == 64


def test_recorded_scene_checks_and_saves_functions_without_executing_code(tmp_path):
    from specola.discovery_capture import discovery_payload
    from specola.dashboard.server import DiscoverySnapshot
    from specola.authoring import CheckContext, check_draft, write_specs_document, spec_entry
    adapter = InputsAdapter()
    path = specification(tmp_path)
    captured = []
    monitor = Monitor(str(path), results_dir=None)
    monitor.watch(adapter, _reset_hook=lambda a, r: captured.append(discovery_payload(a, r, simulator="test", task=None, effective_dt=0.1)))
    source = tmp_path / "local.py"
    source.write_text("raise RuntimeError('must not execute during restored check')")
    snapshot = DiscoverySnapshot.from_gzip(captured[0])
    context = CheckContext.from_discovery_model(snapshot.model)
    verdict = check_draft(context, name="restored", formula="object_0[derived] >= 1")
    assert verdict.accepted, verdict.report
    destination = tmp_path / "elsewhere" / "saved.yaml"
    destination.parent.mkdir()
    write_specs_document([spec_entry(name="saved", formula="object_0[derived] >= 1")], str(destination), functions=context.functions)
    with pytest.raises(ValueError, match="source digest changed"):
        MonitorEngine.load(str(destination))


@pytest.mark.parametrize('formula,expected,active', [
    ('object_0[der*] >= 1', 3.0, True),
    ('object_0[all: {native, derived}] >= 0', 1.0, True),
    ('object_0[derived @ authored function] >= 1', 3.0, True),
    ('Min(object_0[{native, derived}]) >= 0', 1.0, True),
    (r'object_0[all: * \ native] >= 1', 3.0, True),
    (r'Min(object_0[* \ derived]) >= 0', 1.0, False),
])
def test_existing_selector_language_requests_only_needed_functions(tmp_path, formula, expected, active):
    adapter = InputsAdapter()
    session = Monitor(str(specification(tmp_path, formula=formula)), results_dir=None).watch(adapter)
    result = session.step()
    assert result._rho_by_spec == {'one': expected, 'two': expected}
    assert bool(session.engine.functions.definitions) is active
    assert bool(adapter.captures) is active
    assert bool(session.engine.functions._caches) is active
    assert isinstance(adapter._authored_function_context, dict)
    assert set(adapter._authored_function_context['declarations']) == {'twice'}
    if active:
        assert len(session.engine.functions.definitions[0].implementation.__globals__['calls']) == 1


def authored_pair_specification(tmp_path):
    path = specification(tmp_path, formula="pair_*[derived] >= 1")
    document = yaml.safe_load(path.read_text())
    document["functions"]["twice"]["entity_class"] = "pair"
    path.write_text(yaml.safe_dump(document))
    source = tmp_path / "local.py"
    source.write_text(FUNCTION.replace('entity_id=0, ', ''))
    return path


class GrowingNativeAdapter(InputsAdapter):
    @property
    def supported_classes(self):
        return frozenset({"object", "pair"})

    def __init__(self):
        super().__init__()
        self.population = 1
        self.added_class = "object"

    def _read(self):
        from dataclasses import replace
        observation = super()._read()
        entities = dict(observation.perceived_entities)
        for entity_id in range(1, self.population):
            entities[entity_id] = replace(entities[0], entity_id=entity_id, entity_class=self.added_class)
        return replace(observation, perceived_entities=entities)


def test_authored_identity_reconciles_native_population_growth_across_resets(tmp_path):
    adapter = GrowingNativeAdapter()
    monitor = Monitor(str(authored_pair_specification(tmp_path)), results_dir=None)
    ids = []
    for population in (1, 2, 2, 4):
        adapter.population = population
        session = monitor.watch(adapter)
        target = session.engine.functions._plans["twice"]["sample"]
        assert target.entity_id not in range(population)
        ids.append(target.entity_id)
        result = session.step()
        assert result._rho_by_spec == {"one": 3.0, "two": 3.0}
        observation = adapter.last_observation
        assert observation is not None
        assert all("derived" not in entity.properties for entity in observation.perceived_entities.values())
        session.end()
    assert ids == [1, 2, 2, 4]


def test_authored_allocator_reserves_unobserved_native_catalog_ids(tmp_path):
    from dataclasses import replace
    from specola.capabilities import reset_adapter
    from specola.components import ComponentCatalog, EntityComponentInfo
    adapter = InputsAdapter()
    engine = MonitorEngine.load(str(authored_pair_specification(tmp_path)))
    adapter.set_function_inputs(engine.functions.requirements)
    reset = reset_adapter(adapter)
    catalog = dict(reset.discovered_catalog.entities)
    catalog[7] = EntityComponentInfo(7, "object", {"native"})
    reset = replace(reset, discovered_catalog=ComponentCatalog(catalog))
    engine.functions.prepare(adapter, reset)
    assert engine.functions._plans["twice"]["sample"].entity_id == 8


@pytest.mark.parametrize("native_class", ["object", "pair"])
def test_overlay_rejects_new_native_owner_even_when_class_matches(tmp_path, native_class):
    adapter = GrowingNativeAdapter()
    session = Monitor(str(authored_pair_specification(tmp_path)), results_dir=None).watch(adapter)
    adapter.population = 2
    adapter.added_class = native_class
    implementation = session.engine.functions.definitions[0].implementation
    with pytest.raises(ValueError, match="function twice failed.*entity ID 1 collides with a native owner"):
        session.step()
    assert implementation.__globals__["calls"] == []


def test_prepare_rejects_custom_allocator_returning_native_owner(tmp_path):
    adapter = InputsAdapter()
    adapter.function_entity_id = lambda key, entity_class: 0
    with pytest.raises(ValueError, match="authored entity ID 0 collides with a native owner"):
        Monitor(str(authored_pair_specification(tmp_path)), results_dir=None).watch(adapter)
