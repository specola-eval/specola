"""New bundled registrations enlarge the extraction denominator by default."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from specola import adapter_enrollment as enrollment
from specola.adapter_enrollment import (
    AdapterDeclaration, EnrollmentError, ProviderEnrollment,
    bundled_adapter_declarations, required_extraction_declarations,
    validate_provider_enrollment,
    bundled_provider_enrollments, extraction_class_membership,
)


def _distribution(monkeypatch, names=("live", "mock", "dataset")):
    entries = tuple(SimpleNamespace(name=name, value="producer." + name, group="specola.adapters") for name in names)
    distribution = SimpleNamespace(entry_points=entries + (
        SimpleNamespace(name="unrelated", value="command", group="console_scripts"),
    ))
    monkeypatch.setattr(enrollment.metadata, "distribution", lambda name: distribution)
    return distribution


def _providers(declarations):
    return tuple(ProviderEnrollment(item, item.entry_point_value + ":Adapter", item.entry_point_value + ":factory")
                 for item in required_extraction_declarations(declarations))


def test_wheel_census_uses_only_distribution_metadata(monkeypatch):
    _distribution(monkeypatch)
    # No filesystem or process-wide registry is part of this census.
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("wheel census read source tree"))
    monkeypatch.setattr(enrollment.metadata, "entry_points", lambda *a, **kw: pytest.fail("used global plugin population"))
    declarations = bundled_adapter_declarations()
    assert {item.name for item in declarations} == {"live", "mock", "dataset"}
    assert {item.name for item in required_extraction_declarations(declarations)} == {"live"}
    validate_provider_enrollment(declarations, _providers(declarations))


def test_new_unused_bundled_adapter_requires_enrollment(monkeypatch):
    distribution = _distribution(monkeypatch)
    providers = _providers(bundled_adapter_declarations())
    distribution.entry_points += (SimpleNamespace(name="new_native", value="producer.new_native", group="specola.adapters"),)
    with pytest.raises(EnrollmentError, match="missing=.*new_native"):
        validate_provider_enrollment(bundled_adapter_declarations(), providers)


@pytest.mark.parametrize("mutation, message", (
    (lambda p: (), "missing="),
    (lambda p: p + p, "duplicate"),
    (lambda p: (replace(p[0], registration=replace(p[0].registration, entry_point_value="foreign.shadow")),), "changed="),
    (lambda p: p + (ProviderEnrollment(AdapterDeclaration("foreign", "foreign.plugin"), "foreign:Adapter", "foreign:factory"),), "extra="),
    (lambda p: (replace(p[0], adapter_class=""),), "adapter class"),
    (lambda p: (replace(p[0], factory=""),), "factory"),
))
def test_missing_stale_duplicate_and_extra_providers_fail(monkeypatch, mutation, message):
    _distribution(monkeypatch)
    declarations = bundled_adapter_declarations()
    with pytest.raises(EnrollmentError, match=message):
        validate_provider_enrollment(declarations, mutation(_providers(declarations)))


def test_duplicate_distribution_registration_fails_before_population_deduplication(monkeypatch):
    _distribution(monkeypatch, ("live", "live"))
    with pytest.raises(EnrollmentError, match="duplicate"):
        bundled_adapter_declarations()


def test_empty_distribution_cannot_pass(monkeypatch):
    _distribution(monkeypatch, ())
    with pytest.raises(EnrollmentError, match="no bundled"):
        bundled_adapter_declarations()


def test_source_declarations_equal_installed_distribution():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with project.open("rb") as source:
        declared = tomllib.load(source)["project"]["entry-points"]["specola.adapters"]
    installed = {item.name: item.entry_point_value for item in bundled_adapter_declarations()}
    assert installed == declared


def test_reviewed_classes_cover_the_distribution_population():
    import importlib
    from specola.adapter import SimulatorAdapter

    declarations = bundled_adapter_declarations()
    providers = bundled_provider_enrollments(declarations)
    validate_provider_enrollment(declarations, providers)
    # Expected registrations come from the installed distribution, not from
    # the reviewed class-name mapping that this check is auditing.
    assert {provider.registration for provider in providers} == set(required_extraction_declarations(declarations))
    for provider in providers:
        module_name, class_name = provider.adapter_class.split(":")
        adapter_class = vars(importlib.import_module(module_name))[class_name]
        assert issubclass(adapter_class, SimulatorAdapter)
        membership = extraction_class_membership(adapter_class, declarations)
        assert membership.status == "bundled_implementation"
        assert membership.registration == provider.registration.name
        factory_module, factory_name = provider.factory.split(":")
        assert callable(vars(importlib.import_module(factory_module))[factory_name])


def test_unused_new_registration_has_no_implicit_class_enrollment():
    declarations = bundled_adapter_declarations() + (AdapterDeclaration("unused_new", "specola.adapters.unused_new"),)
    with pytest.raises(EnrollmentError, match="reviewed implementation.*unused_new"):
        bundled_provider_enrollments(declarations)


def test_class_membership_does_not_depend_on_provider_or_source_equality():
    from specola.adapters.libero import LIBEROAdapter

    class RenamedImplementation:
        _extraction_provider = None

    RenamedImplementation.__module__ = LIBEROAdapter.__module__
    RenamedImplementation.__qualname__ = LIBEROAdapter.__qualname__
    # An edited or missing implementation stays in the bundled audit subject.
    # Its source and behavior are checked by readiness, not by this membership.
    membership = extraction_class_membership(RenamedImplementation)
    assert membership.status == "bundled_implementation"
    assert membership.registration == "libero"


def test_custom_subclass_retains_native_lineage_without_bundled_evidence():
    from specola.adapters.libero import LIBEROAdapter

    class Custom(LIBEROAdapter):
        _extraction_provider = None

    membership = extraction_class_membership(Custom)
    assert membership.status == "unaudited_custom_adapter"
    assert membership.registration is None
    assert "specola.adapters.mujoco:MujocoAdapter" in membership.native_lineage
    assert "specola.adapters.libero:LIBEROAdapter" in membership.native_lineage


def test_mutated_class_name_cannot_hide_distribution_membership():
    from specola.adapters.libero import LIBEROAdapter

    original = LIBEROAdapter.__module__, LIBEROAdapter.__qualname__
    try:
        LIBEROAdapter.__module__ = "foreign_claim"
        LIBEROAdapter.__qualname__ = "Renamed"
        membership = extraction_class_membership(LIBEROAdapter)
        assert membership.status == "bundled_implementation"
        assert membership.registration == "libero"
        assert membership.native_lineage == (
            "specola.adapters.libero:LIBEROAdapter", "specola.adapters.mujoco:MujocoAdapter")
    finally:
        LIBEROAdapter.__module__, LIBEROAdapter.__qualname__ = original


def test_impersonating_class_cannot_supply_its_own_authentication():
    from specola.adapters.fake import _MockAdapter
    from specola.conformance.provider import ExtractionGuardError, ExtractionProvider
    from specola.adapters.libero import LIBEROAdapter

    calls = []

    class Provider(ExtractionProvider):
        def guard_source(self, adapter):
            calls.append("untrusted guard")

    class Impersonator(_MockAdapter):
        _extraction_provider = Provider()

    Impersonator.__module__ = LIBEROAdapter.__module__
    Impersonator.__qualname__ = LIBEROAdapter.__qualname__
    with pytest.raises(ExtractionGuardError, match="implementation_identity_mismatch"):
        Impersonator().reset()
    assert calls == []


def test_source_guard_does_not_execute_an_instance_dictionary_property():
    from specola.adapters.fake import _MockAdapter

    def forbidden_dictionary(self):
        pytest.fail("guard executed the adapter dictionary property")

    custom = type("Custom", (_MockAdapter,), {"__dict__": property(forbidden_dictionary)})
    custom().reset()


def test_distribution_is_resolved_per_reset_not_per_observation(monkeypatch):
    from specola.adapters.fake import _MockAdapter

    distribution = enrollment.metadata.distribution
    calls = []

    def measured(name):
        calls.append(name)
        return distribution(name)

    monkeypatch.setattr(enrollment.metadata, "distribution", measured)
    adapter = _MockAdapter()
    adapter.reset()
    for _ in range(3):
        adapter.get_observation()
    adapter.reset()
    adapter.get_observation()
    assert calls == ["specola", "specola"]


def test_unrelated_plugin_is_unaudited_without_becoming_a_bundled_row():
    from specola.adapter import SimulatorAdapter

    class ThirdParty(SimulatorAdapter):
        pass

    membership = extraction_class_membership(ThirdParty)
    assert membership.status == "unaudited_third_party_provider"
    assert membership.native_lineage == ()


def test_excluded_builtin_classes_keep_the_legacy_capability_wire():
    from specola.adapters.fake import _MockAdapter
    from specola.adapters.dataset import DatasetAdapter
    from specola.capabilities import adapter_capabilities
    from specola.discovery_serialize import capabilities_document

    for adapter_class in (_MockAdapter, DatasetAdapter):
        assert extraction_class_membership(adapter_class).status == "excluded_from_extraction_programme"
    adapter = _MockAdapter()
    adapter.reset()
    document = capabilities_document(adapter_capabilities(adapter))
    assert document is not None
    assert "extraction_status" not in document


@pytest.mark.parametrize("name, module_name, class_name", (
    ("pybullet", "specola.adapters.pybullet", "PyBulletAdapter"),
    ("carla", "specola.adapters.carla", "CarlaAdapter"),
    ("isaaclab", "specola.adapters.isaaclab", "IsaacLabAdapter"),
))
def test_scope_excluded_simulator_adapters_report_the_exclusion(name, module_name, class_name):
    # Maintainer scope decision 2026-10-13: the rows leave the extraction
    # programme the way robodojo did; this is not a certification.
    import importlib
    from specola.adapter import SimulatorAdapter
    from specola.conformance.audit import adapter_surface_audit
    from specola.conformance.provider import adapter_extraction_status, guard_extraction_source

    declarations = bundled_adapter_declarations()
    assert name in {item for item, _ in enrollment.EXTRACTION_SCOPE_EXCLUSIONS}
    assert name not in {item.name for item in required_extraction_declarations(declarations)}
    adapter_class = vars(importlib.import_module(module_name))[class_name]
    membership = extraction_class_membership(adapter_class, declarations)
    assert membership.status == "excluded_from_extraction_programme"
    assert membership.registration == name
    document = adapter_surface_audit(name)
    assert document["status"] == "excluded_from_programme"
    assert document["obligations"] == []
    # The guard no longer refuses the row; no simulator is needed to show it.
    adapter = object.__new__(adapter_class)
    SimulatorAdapter.__init__(adapter)
    assert guard_extraction_source(adapter) is None
    assert adapter_extraction_status(adapter) is None


def test_unaudited_status_survives_discovery_capability_restore():
    from specola.adapters.fake import _MockAdapter
    from specola.capabilities import adapter_capabilities
    from specola.discovery_serialize import capabilities_document
    from specola.discovery_restore import capabilities_from_document

    class ThirdParty(_MockAdapter):
        pass

    adapter = ThirdParty()
    adapter.reset()
    capabilities = adapter_capabilities(adapter)
    assert capabilities.extraction_status == "unaudited_third_party_provider"
    document = capabilities_document(capabilities)
    assert document is not None
    assert document["extraction_status"] == "unaudited_third_party_provider"
    restored = capabilities_from_document({"capabilities": document})
    assert restored.extraction_status == capabilities.extraction_status


def test_native_class_attribute_cannot_replace_provider_registration(monkeypatch):
    from specola.adapter import SimulatorAdapter
    from specola.adapters.mujoco import MujocoAdapter
    from specola.conformance.provider import ExtractionGuardError, ExtractionProvider, guard_extraction_source

    events = []

    class Injected(ExtractionProvider):
        def guard_source(self, adapter):
            events.append("injected source")

    monkeypatch.setattr(MujocoAdapter, "_extraction_provider", Injected(), raising=False)
    adapter = object.__new__(MujocoAdapter)
    SimulatorAdapter.__init__(adapter)
    # The registered bounded provider governs the native class; the class
    # attribute is never consulted, and the fixture has no native subject.
    with pytest.raises(ExtractionGuardError, match="native_subject_unavailable"):
        guard_extraction_source(adapter)
    assert events == []


def test_native_provider_registration_is_unique_and_authenticates_class(monkeypatch):
    from specola.adapter import SimulatorAdapter
    from specola.adapters.mujoco import MujocoAdapter
    from specola.conformance import provider

    monkeypatch.setattr(provider, "_NATIVE_PROVIDERS", {})
    events = []

    class NativeProvider(provider.ExtractionProvider):
        def guard_source(self, adapter):
            events.append(type(adapter))

    native = NativeProvider()
    provider.register_extraction_provider(MujocoAdapter, native)
    adapter = object.__new__(MujocoAdapter)
    SimulatorAdapter.__init__(adapter)
    assert provider.guard_extraction_source(adapter) is native
    assert events == [MujocoAdapter]
    with pytest.raises(provider.ExtractionGuardError, match="provider_already_registered"):
        provider.register_extraction_provider(MujocoAdapter, NativeProvider())

    Impersonator = type("MujocoAdapter", (SimulatorAdapter,), {
        "__module__": MujocoAdapter.__module__,
    })
    with pytest.raises(provider.ExtractionGuardError, match="implementation_identity_mismatch"):
        provider.register_extraction_provider(Impersonator, NativeProvider())


def test_instance_provider_cannot_authenticate_pending_native_adapter():
    from specola.adapter import SimulatorAdapter
    from specola.adapters.mujoco import MujocoAdapter
    from specola.conformance.provider import ExtractionGuardError, ExtractionProvider

    events = []

    class Injected(ExtractionProvider):
        def guard_source(self, adapter):
            events.append("injected source")

    adapter = object.__new__(MujocoAdapter)
    SimulatorAdapter.__init__(adapter)
    setattr(adapter, "_extraction_provider", Injected())
    with pytest.raises(ExtractionGuardError, match="native_subject_unavailable"):
        adapter.reset()
    assert events == []


def test_class_reassignment_invalidates_an_active_observation():
    from specola.adapters.fake import _MockAdapter
    from specola.conformance.provider import ExtractionGuardError

    class Replacement(_MockAdapter):
        pass

    adapter = _MockAdapter()
    adapter.reset()
    adapter.__class__ = Replacement
    with pytest.raises(ExtractionGuardError, match="implementation_identity_mismatch"):
        adapter.get_observation()


def test_replaced_native_base_provider_invalidates_active_observation(monkeypatch):
    from specola.adapter import MonitorObservation, SimulatorAdapter
    from specola.adapters.mujoco import MujocoAdapter
    from specola.conformance.provider import ExtractionGuardError, ExtractionProvider

    events = []

    class EngineGuard(ExtractionProvider):
        def guard_source(self, adapter):
            events.append("source")

        def guard_runtime(self, adapter):
            events.append("runtime")

    class Custom(MujocoAdapter):
        _extraction_provider = None

        def __init__(self):
            SimulatorAdapter.__init__(self)

        @property
        def dt(self):
            return 0.1

        @property
        def spatial_dims(self):
            return 3

        def _reset_impl(self):
            return MonitorObservation(timestep=0, time=0.0, perceived_entities={})

        def _get_observation_impl(self):
            events.append("read")
            return MonitorObservation(timestep=1, time=0.1, perceived_entities={})

    from specola.conformance.provider import _NATIVE_PROVIDERS
    monkeypatch.setitem(_NATIVE_PROVIDERS, MujocoAdapter, EngineGuard())
    adapter = Custom()
    adapter.reset()
    from specola.conformance.provider import _NATIVE_PROVIDERS
    monkeypatch.setitem(_NATIVE_PROVIDERS, MujocoAdapter, EngineGuard())
    with pytest.raises(ExtractionGuardError, match="provider_binding_changed"):
        adapter.get_observation()
    assert events == ["source"]


def test_custom_native_reader_is_labeled_after_inherited_guard_passes(monkeypatch):
    from specola.adapter import MonitorObservation, SimulatorAdapter
    from specola.adapters.mujoco import MujocoAdapter
    from specola.capabilities import AdapterCapabilities, adapter_capabilities
    from specola.conformance.provider import ExtractionProvider

    events = []

    class EngineGuard(ExtractionProvider):
        def guard_source(self, adapter):
            events.append("engine source")

        def guard_runtime(self, adapter):
            events.append("engine runtime")

    class Custom(MujocoAdapter):
        _extraction_provider = None

        def __init__(self):
            SimulatorAdapter.__init__(self)

        @property
        def dt(self):
            return 0.1

        @property
        def spatial_dims(self):
            return 3

        @property
        def capabilities(self):
            return AdapterCapabilities(spatial_dims=3)

        def _reset_impl(self):
            events.append("custom reset")
            return MonitorObservation(timestep=0, time=0.0, perceived_entities={})

        def _get_observation_impl(self):
            events.append("custom read")
            return MonitorObservation(timestep=1, time=0.1, perceived_entities={})

    from specola.conformance.provider import _NATIVE_PROVIDERS
    monkeypatch.setitem(_NATIVE_PROVIDERS, MujocoAdapter, EngineGuard())
    adapter = Custom()
    adapter.reset()
    adapter.get_observation()
    assert events == ["engine source", "custom reset", "engine runtime", "custom read"]
    assert adapter_capabilities(adapter).extraction_status == "unaudited_custom_adapter"


def test_shared_mujoco_implementation_is_an_engine_subject():
    from specola.adapters.mujoco import MujocoAdapter

    membership = extraction_class_membership(MujocoAdapter)
    assert membership.status == "shared_engine_implementation"
    assert membership.registration is None
    assert membership.native_lineage == ("specola.adapters.mujoco:MujocoAdapter",)


@pytest.mark.parametrize("provider", bundled_provider_enrollments(), ids=lambda item: item.registration.name)
def test_direct_bundled_class_without_provider_fails_before_native_access(provider):
    import importlib
    from specola.adapter import SimulatorAdapter
    from specola.conformance.provider import ExtractionGuardError

    module_name, class_name = provider.adapter_class.split(":")
    cls = vars(importlib.import_module(module_name))[class_name]
    adapter = object.__new__(cls)
    SimulatorAdapter.__init__(adapter)
    # No simulator/model exists. A late guard would fail on native metadata
    # access; a registration-name guard would miss this direct construction.
    # Every enrolled row is MuJoCo-family and carries the bounded provider,
    # whose source guard refuses an adapter without its native initializer.
    with pytest.raises(ExtractionGuardError, match="native_subject_unavailable"):
        adapter.reset()
    assert adapter.last_observation is None
    assert not adapter.episode_started


def test_enrolled_row_without_a_registered_provider_is_pending(monkeypatch):
    from specola.adapter import SimulatorAdapter
    from specola.adapters.libero import LIBEROAdapter
    from specola.conformance.provider import _NATIVE_PROVIDERS, ExtractionGuardError

    # A synthetic pending row: the enrolled class with its provider withdrawn.
    monkeypatch.delitem(_NATIVE_PROVIDERS, LIBEROAdapter)
    adapter = object.__new__(LIBEROAdapter)
    SimulatorAdapter.__init__(adapter)
    with pytest.raises(ExtractionGuardError, match="bundled_row_pending: specola.adapters.libero:LIBEROAdapter"):
        adapter.reset()
    assert adapter.last_observation is None


def test_custom_provider_override_cannot_remove_inherited_source_guard(monkeypatch):
    from specola.adapter import SimulatorAdapter
    from specola.adapters.mujoco import MujocoAdapter
    from specola.conformance.provider import ExtractionProvider

    class NativeSourceChanged(RuntimeError):
        pass

    class EngineGuard(ExtractionProvider):
        def guard_source(self, adapter):
            raise NativeSourceChanged("changed native producer")

    class Custom(MujocoAdapter):
        _extraction_provider = None

    from specola.conformance.provider import _NATIVE_PROVIDERS
    monkeypatch.setitem(_NATIVE_PROVIDERS, MujocoAdapter, EngineGuard())
    adapter = object.__new__(Custom)
    SimulatorAdapter.__init__(adapter)
    with pytest.raises(NativeSourceChanged, match="changed native producer"):
        adapter.reset()
