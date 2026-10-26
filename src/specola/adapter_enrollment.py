"""Distribution-owned adapter populations for extraction conformance.

Enrollment is not a completeness result. This module can be imported before
the adapter or conformance packages and does not import simulator modules,
invoke factories, or use the process-wide registry as a bundled denominator.
"""

from dataclasses import dataclass
from importlib import metadata
import sys
from types import ModuleType
from typing import Optional, Tuple


class EnrollmentError(ValueError):
    """Bundled registrations and declared providers do not reconcile."""


@dataclass(frozen=True, order=True)
class AdapterDeclaration:
    """One distribution-owned registration, without importing its target."""

    name: str
    entry_point_value: str


@dataclass(frozen=True)
class ProviderEnrollment:
    """Implementation identities that a runtime provider must later verify."""

    registration: AdapterDeclaration
    adapter_class: str
    factory: str


@dataclass(frozen=True)
class ExtractionClassMembership:
    """Implementation membership, independent of producer or evidence status."""

    status: str
    registration: Optional[str]
    native_lineage: Tuple[str, ...]
    native_classes: Tuple[type, ...]


# These are reviewed implementation bindings, not the adapter denominator.
# The installed distribution supplies that denominator. An added registration
# cannot disappear from it merely because this mapping has no entry yet.
_REVIEWED_IMPLEMENTATIONS = {
    "specola.adapters.libero": ("LIBEROAdapter", "_make_libero"),
    "specola.adapters.robocasa": ("RoboCasaAdapter", "_make_robocasa"),
    "specola.adapters.metaworld": ("MetaworldAdapter", "_make_metaworld"),
    "specola.adapters.dmcontrol": ("DMControlAdapter", "_make_dmcontrol"),
    "specola.adapters.mamujoco": ("MaMujocoAdapter", "_make_mamujoco"),
}
_SHARED_ENGINE_CLASS = "specola.adapters.mujoco:MujocoAdapter"
_EXCLUDED_IMPLEMENTATIONS = {
    "specola.adapters.fake": "_MockAdapter",
    "specola.adapters.dataset": "DatasetAdapter",
    "specola.adapters.robodojo": "RoboDojoAdapter",
    "specola.adapters.pybullet": "PyBulletAdapter",
    "specola.adapters.carla": "CarlaAdapter",
    "specola.adapters.isaaclab": "IsaacLabAdapter",
}


def bundled_provider_enrollments(declarations=None) -> Tuple[ProviderEnrollment, ...]:
    """Reconcile distribution-owned registrations with reviewed class bindings."""
    declarations = bundled_adapter_declarations() if declarations is None else declarations
    required = required_extraction_declarations(declarations)
    records = []
    for declaration in required:
        implementation = _REVIEWED_IMPLEMENTATIONS.get(declaration.entry_point_value)
        if implementation is None:
            raise EnrollmentError("missing reviewed implementation enrollment: " + declaration.name)
        adapter_class, factory = implementation
        records.append(ProviderEnrollment(
            declaration, declaration.entry_point_value + ":" + adapter_class,
            declaration.entry_point_value + ":" + factory,
        ))
    result = tuple(records)
    validate_provider_enrollment(declarations, result)
    return result


def _class_identity(adapter_class: type) -> str:
    # Read the type's native metadata descriptors directly. A third-party
    # metaclass cannot run an overridden accessor during membership checks.
    module = vars(type)["__module__"].__get__(adapter_class)
    qualname = vars(type)["__qualname__"].__get__(adapter_class)
    if type(module) is not str or type(qualname) is not str:
        raise EnrollmentError("adapter class has no declared Python identity")
    return module + ":" + qualname


def resolved_implementation(identity: str):
    """Resolve an enrolled class without importing a module or invoking accessors."""
    module_name, qualname = identity.split(":", 1)
    module = sys.modules.get(module_name)
    if not isinstance(module, ModuleType):
        return None
    namespace = vars(ModuleType)["__dict__"].__get__(module)
    value = namespace.get(qualname)
    return value if isinstance(value, type) else None


def extraction_class_membership(adapter_class: type, declarations=None) -> ExtractionClassMembership:
    """Resolve bundled objects independently of provider and readiness evidence.

    Import-system exports retain class membership when mutable type names
    change. A foreign class claiming a bundled name also enters the guard,
    where export authentication rejects it before its provider can run.
    """
    if not isinstance(adapter_class, type):
        raise EnrollmentError("adapter implementation must be a class")
    declarations = bundled_adapter_declarations() if declarations is None else declarations
    providers = bundled_provider_enrollments(declarations)
    classes = {provider.adapter_class: provider for provider in providers}
    identities = tuple(classes) + (_SHARED_ENGINE_CLASS,)
    resolved = tuple((identity, resolved_implementation(identity)) for identity in identities)
    mro = vars(type)["__mro__"].__get__(adapter_class)
    lineage = []
    native_classes = []
    own_identity = None
    for base in mro:
        identity = next((name for name, declared in resolved if declared is base), None)
        if identity is None:
            claimed = _class_identity(base)
            identity = claimed if claimed in identities else None
        if identity is not None:
            lineage.append(identity)
            native_classes.append(base)
            if base is adapter_class:
                own_identity = identity
    names, objects = tuple(lineage), tuple(native_classes)
    if own_identity in classes:
        return ExtractionClassMembership(
            "bundled_implementation", classes[own_identity].registration.name, names, objects,
        )
    if own_identity == _SHARED_ENGINE_CLASS:
        return ExtractionClassMembership("shared_engine_implementation", None, names, objects)
    if lineage:
        return ExtractionClassMembership("unaudited_custom_adapter", None, names, objects)
    excluded = {name for name, _ in EXTRACTION_SCOPE_EXCLUSIONS}
    for declaration in declarations:
        if declaration.name not in excluded:
            continue
        class_name = _EXCLUDED_IMPLEMENTATIONS.get(declaration.entry_point_value)
        if class_name is not None and resolved_implementation(
            declaration.entry_point_value + ":" + class_name,
        ) is adapter_class:
            return ExtractionClassMembership("excluded_from_extraction_programme", declaration.name, (), ())
    return ExtractionClassMembership("unaudited_third_party_provider", None, (), ())


# These are maintainer scope decisions, not a positive adapter allowlist.
EXTRACTION_SCOPE_EXCLUSIONS = (
    ("mock", "maintainer excludes the synthetic adapter from the new extraction programme"),
    ("dataset", "maintainer excludes CSV/dataset transport from the new extraction programme"),
    ("robodojo", "maintainer decision 2026-09-22: RoboDojo runs on the legacy capability wire "
                 "until an Isaac extraction provider exists; the pending-row gate must not block it"),
    # Maintainer decision 2026-10-13: these three enrolled rows had no
    # extraction provider, so the guard refused every reset with
    # bundled_row_pending. Excluding them is a scope decision, not a
    # certification. They are official under the 2026-08-20 descoped
    # criterion; full certification stays open in #111, #112 and #107.
    ("pybullet", "maintainer decision 2026-10-13: PyBullet runs on the legacy capability wire "
                 "until an extraction provider exists (#111); excluded, not certified"),
    ("carla", "maintainer decision 2026-10-13: CARLA runs on the legacy capability wire "
              "until an extraction provider exists (#112); excluded, not certified"),
    ("isaaclab", "maintainer decision 2026-10-13: Isaac Lab runs on the legacy capability wire "
                 "until an extraction provider exists (#107); excluded, not certified"),
)


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise EnrollmentError(label + " must be nonempty text")


def _declarations(value: Tuple[AdapterDeclaration, ...]) -> None:
    if type(value) is not tuple:
        raise EnrollmentError("adapter declarations must be a tuple")
    names = set()
    for declaration in value:
        if type(declaration) is not AdapterDeclaration:
            raise EnrollmentError("adapter declaration has the wrong type")
        _text(declaration.name, "adapter name")
        _text(declaration.entry_point_value, "adapter entry point")
        if declaration.name in names:
            raise EnrollmentError("duplicate bundled adapter name: " + declaration.name)
        names.add(declaration.name)


def bundled_adapter_declarations() -> Tuple[AdapterDeclaration, ...]:
    """Read the installed specola distribution, including mock and dataset.

    Wheel installations need no source checkout. Source/editable tests must
    separately reconcile this distribution's declarations with pyproject.toml.
    """
    distribution = metadata.distribution("specola")
    declarations = tuple(AdapterDeclaration(entry.name, entry.value)
                         for entry in distribution.entry_points
                         if entry.group == "specola.adapters")
    _declarations(declarations)
    if not declarations:
        raise EnrollmentError("specola distribution has no bundled adapter declarations")
    return tuple(sorted(declarations))


def required_extraction_declarations(declarations: Tuple[AdapterDeclaration, ...]) -> Tuple[AdapterDeclaration, ...]:
    """Apply only the accepted scope exclusions to a bundled population."""
    _declarations(declarations)
    excluded = {name for name, _ in EXTRACTION_SCOPE_EXCLUSIONS}
    return tuple(sorted(item for item in declarations if item.name not in excluded))


def validate_provider_enrollment(declarations: Tuple[AdapterDeclaration, ...], providers: Tuple[ProviderEnrollment, ...]) -> None:
    """Require exact bundled enrollment without claiming runtime custody.

    Class and factory strings identify required bindings. Runtime verification
    must reconcile the actual implementation objects and their source evidence;
    an enrolled string cannot grant a completeness result or certify a subclass.
    """
    required = required_extraction_declarations(declarations)
    if not required:
        raise EnrollmentError("extraction programme has no required adapter declarations")
    if type(providers) is not tuple or any(type(item) is not ProviderEnrollment for item in providers):
        raise EnrollmentError("providers must be a tuple of ProviderEnrollment records")
    registrations = tuple(provider.registration for provider in providers)
    _declarations(registrations)
    for provider in providers:
        _text(provider.adapter_class, "provider adapter class")
        _text(provider.factory, "provider factory")
    expected = {item.name: item.entry_point_value for item in required}
    actual = {item.name: item.entry_point_value for item in registrations}
    if actual != expected:
        missing, extra = set(expected) - set(actual), set(actual) - set(expected)
        changed = {name for name in set(actual) & set(expected) if actual[name] != expected[name]}
        raise EnrollmentError("provider enrollment differs: missing=" + str(sorted(missing))
                              + "; extra=" + str(sorted(extra)) + "; changed=" + str(sorted(changed)))
