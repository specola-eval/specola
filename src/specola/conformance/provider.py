"""Execute extraction contracts through the shared adapter path.

Providers supply producer guards and independently resolved populations. This
module composes those checks with the existing surface/manifest relations;
it does not infer row readiness from a callback or a valid serializer result.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Callable, Optional, Tuple, TYPE_CHECKING

from ..adapter_enrollment import (
    ExtractionClassMembership, extraction_class_membership, resolved_implementation,
)

from .extraction import ExtractionManifest, ResultSurface
from .instances import RuntimePopulation, validate_materialization, validate_query_population, validate_runtime_population
from .query_bindings import CallableParameterSurface, QueryBindingRecord, validate_query_arguments, validate_query_contract
from .surface import ProducerSurface
from .core_runner import EvidenceIdentity

if TYPE_CHECKING:
    from ..adapter import MonitorObservation, SimulatorAdapter
    from .authoring import PreparedAuthoring


class ExtractionGuardError(RuntimeError):
    """An adapter has no resolved producer contract for ordinary observation."""

    def __init__(self, status: str, missing):
        self.status = status
        self.missing = tuple(missing)
        super().__init__(status + ": " + "; ".join(self.missing))


@dataclass(frozen=True, eq=False)
class _BoundSourceGuard:
    adapter_class: type
    base_classes: Tuple[type, ...]
    membership: ExtractionClassMembership
    guards: Tuple[ExtractionProvider, ...]
    provider: Optional[ExtractionProvider]
    verification_scope: object
    partial: bool = False


def _guard_bindings(adapter_class, membership, *, partial=False):
    class_dict = vars(type)["__dict__"]
    native = any(base is adapter_class for base in membership.native_classes)
    provider = (registered_extraction_provider(adapter_class) if native else
                class_dict.__get__(adapter_class).get("_extraction_provider"))
    guards = []
    for base, identity in reversed(tuple(zip(membership.native_classes, membership.native_lineage))):
        if resolved_implementation(identity) is not base:
            raise ExtractionGuardError("implementation_identity_mismatch", (identity,))
        inherited = registered_extraction_provider(base)
        if inherited is None:
            if partial:
                continue
            status = ("bundled_row_pending" if membership.status in
                      ("bundled_implementation", "shared_engine_implementation")
                      else "unresolved_inherited_producer_contract")
            raise ExtractionGuardError(status, (identity + " has no extraction provider",))
        if not isinstance(inherited, ExtractionProvider):
            raise TypeError("inherited extraction provider has the wrong type")
        if not any(inherited is prior for prior in guards):
            guards.append(inherited)
    if provider is not None:
        if not isinstance(provider, ExtractionProvider):
            raise TypeError("adapter extraction provider has the wrong type")
        if not any(provider is prior for prior in guards):
            guards.append(provider)
    return tuple(guards), provider


def guard_extraction_source(adapter: SimulatorAdapter, *, reset: bool = True,
                            allow_partial: bool = False):
    """Authenticate class and producer guards before metadata or observation.

    Reset resolves distribution enrollment and runs source custody. Following
    observations retain that subject, authenticate its object bindings, and
    run the provider's runtime drift guard. Missing inherited contracts remain
    unresolved even when a custom adapter overrides its own provider.
    """
    from ..adapter import SimulatorAdapter
    from .verification import current_verification

    verification_scope = current_verification(adapter)
    namespace = vars(SimulatorAdapter)["__dict__"].__get__(adapter)
    adapter_class = type(adapter)
    mro = vars(type)["__mro__"].__get__(adapter_class)
    previous = None
    if reset:
        namespace.pop("_extraction_guard_state", None)
        membership = extraction_class_membership(adapter_class)
    else:
        previous = namespace.get("_extraction_guard_state")
        if type(previous) is not _BoundSourceGuard:
            raise ExtractionGuardError("extraction_reset_required", ("no accepted source guard",))
        if previous.verification_scope is not verification_scope:
            raise ExtractionGuardError("verification_scope_changed", ("reset required outside the original candidate scope",))
        if (previous.adapter_class is not adapter_class or len(previous.base_classes) != len(mro)
                or any(before is not after for before, after in zip(previous.base_classes, mro))):
            raise ExtractionGuardError("implementation_identity_mismatch", ("adapter class or bases changed",))
        membership = previous.membership
    guards, provider = _guard_bindings(adapter_class, membership)
    if verification_scope is not None and provider is None:
        raise ExtractionGuardError("verification_provider_required", ("candidate execution requires a declared extraction provider",))
    if previous is not None:
        if (provider is not previous.provider or len(guards) != len(previous.guards)
                or any(before is not after for before, after in zip(previous.guards, guards))):
            raise ExtractionGuardError("provider_binding_changed", ("reset required for a new provider subject",))
    for guard in guards:
        if reset:
            guard.guard_source(adapter)
        else:
            guard.guard_runtime(adapter)
    if reset:
        namespace["_extraction_guard_state"] = _BoundSourceGuard(
            adapter_class, mro, membership, guards, provider, verification_scope,
        )
    return provider


def adapter_extraction_status(adapter) -> Optional[str]:
    """Return unaudited implementation status from the accepted reset subject."""
    from ..adapter import SimulatorAdapter

    if not isinstance(adapter, SimulatorAdapter):
        return None
    namespace = vars(SimulatorAdapter)["__dict__"].__get__(adapter)
    bound = namespace.get("_extraction_guard_state")
    if type(bound) is not _BoundSourceGuard:
        return None
    status = bound.membership.status
    return status if status in ("unaudited_custom_adapter", "unaudited_third_party_provider") else None


@dataclass(frozen=True)
class PreparedExtraction:
    """A producer-guarded subject resolved before invoking observation readers."""

    surface: ProducerSurface
    results: ResultSurface
    manifest: ExtractionManifest
    population: RuntimePopulation
    guard_after_read: Callable[[], None]
    configuration_identity: Optional[EvidenceIdentity] = None
    bindings: Tuple[QueryBindingRecord, ...] = ()
    declarations: Tuple[CallableParameterSurface, ...] = ()

    def validate_before_read(self, adapter=None) -> None:
        from .verification import current_verification

        execution = current_verification(adapter)
        if execution is not None:
            subject = execution.subject
            if (adapter is None or self.surface.producer != subject.producer
                    or self.configuration_identity != subject.configuration
                    or self.population.queries != subject.queries
                    or adapter.query_configuration != subject.queries):
                raise ExtractionGuardError("verification_subject_mismatch", ("prepared producer or configuration differs from candidate subject",))
        if adapter is not None and (self.population.queries != adapter.query_configuration
                or self.population.configuration_epoch != adapter.query_configuration_epoch
                or self.population.episode_epoch != adapter.extraction_episode_epoch):
            raise ExtractionGuardError("prepared_subject_mismatch", ("prepared queries or epochs differ from the adapter before reading",))
        missing = validate_query_contract(self.surface, self.results, self.manifest, bindings=self.bindings,
                                          declarations=self.declarations, require_closed=execution is None)
        if missing:
            raise ExtractionGuardError("unbound_callable_quantities", missing)
        validate_query_arguments(self.population.queries, self.bindings, self.declarations,
                                 tuple(quantity.quantity_id for quantity in self.manifest.quantities))
        validate_query_population(self.surface, self.manifest, self.population, self.bindings)
        validate_runtime_population(self.manifest, self.population)
        if not callable(self.guard_after_read):
            raise TypeError("prepared extraction needs a post-read producer guard")

    def validate_after_read(self, observation: MonitorObservation, adapter=None) -> None:
        from .verification import current_verification

        self.guard_after_read()
        execution = current_verification(adapter)
        if (observation.extraction is not None
                and observation.extraction.verification_only != (execution is not None)):
            raise ExtractionGuardError("verification_mode_mismatch", ("native observation has the wrong candidate-evidence flag",))
        validate_materialization(self.manifest, self.population, observation)


class ExtractionProvider:
    """Optional producer contract; legacy adapters need not implement it.

    Concrete providers authenticate the producer and implementation, enforce
    derived row evidence, resolve query configuration, and capture native
    invariants before returning a prepared subject. Their post-read guard
    checks the same producer/model/configuration identity before the shared
    join accepts output. These requirements are tested separately from this
    orchestration interface.
    """

    def guard_source(self, adapter: SimulatorAdapter) -> None:
        """Authenticate source and safe namespaces without reading observations."""
        raise NotImplementedError

    def guard_runtime(self, adapter: SimulatorAdapter) -> None:
        """Check runtime, model, callback and configuration drift without a new source census."""
        raise NotImplementedError

    def prepare(self, adapter: SimulatorAdapter, *, reset: bool) -> PreparedExtraction:
        raise NotImplementedError

    def prepare_authoring(self, adapter: SimulatorAdapter) -> Optional[PreparedAuthoring]:
        """Optionally return a closed declaration subject without reading values.

        This runs after source authentication and native handle binding. A
        provider that implements schema-only discovery returns PreparedAuthoring
        with native name-table custody and a post-capture guard. Legacy optional
        providers return None and retain their ordinary observation path.
        """
        return None


_NATIVE_PROVIDERS = {}
_NATIVE_PROVIDER_LOCK = RLock()


def register_extraction_provider(adapter_class: type, provider: ExtractionProvider) -> None:
    """Bind an enrolled native implementation once, independently of class hooks.

    Third-party classes retain their own-class provider hook. Registering a
    native provider establishes its binding, not source custody or readiness.
    """
    membership = extraction_class_membership(adapter_class)
    if membership.status not in ("bundled_implementation", "shared_engine_implementation"):
        raise ExtractionGuardError("not_an_enrolled_native_implementation", (str(adapter_class),))
    for base, identity in zip(membership.native_classes, membership.native_lineage):
        if resolved_implementation(identity) is not base:
            raise ExtractionGuardError("implementation_identity_mismatch", (identity,))
    if not isinstance(provider, ExtractionProvider):
        raise TypeError("native extraction provider has the wrong type")
    with _NATIVE_PROVIDER_LOCK:
        if adapter_class in _NATIVE_PROVIDERS:
            raise ExtractionGuardError("provider_already_registered", membership.native_lineage)
        _NATIVE_PROVIDERS[adapter_class] = provider


def registered_extraction_provider(adapter_class: type):
    """Read an existing native binding without importing an adapter or producer."""
    with _NATIVE_PROVIDER_LOCK:
        return _NATIVE_PROVIDERS.get(adapter_class)
