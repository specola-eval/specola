"""Combined-reader enrollment order and constructor error retention.

These inert constructor tests exercise the production initializer and namespace
comparison. Native values and effects belong to the separate native matrix.
"""

from types import ModuleType

import pytest

from specola.adapters import mujoco_query_reader as subject
from specola.adapters.mujoco_native import _PythonIdentity


class ConstructorReached(Exception):
    pass


def constructor_fixture(monkeypatch, *, cleanup_error=None):
    producer = ModuleType("mujoco")
    modules = {"mujoco": producer}
    events = []
    intended_error = ConstructorReached("geometry constructor reached")

    def import_module(name):
        if name not in modules:
            parent, _, child = name.rpartition(".")
            owner = import_module(parent)
            modules[name] = ModuleType(name)
            setattr(owner, child, modules[name])
        events.append(("import", name))
        return modules[name]

    class Guard:
        def __init__(self, module):
            # Exercise the actual namespace comparator without enrolling native
            # pointers, loading MuJoCo, or replacing global sys.modules entries.
            self.identity = _PythonIdentity({})
            self.identity.namespaces = [(module, dict(vars(module)))]
            events.append(("enroll", tuple(vars(module))))

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            events.append(("close", "guard"))

        def check(self):
            self.identity.check()

    class Session:
        def __init__(self, model, data, guard):
            self.guard = guard

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            self.guard.check()
            events.append(("close", "session"))
            if cleanup_error is not None:
                raise cleanup_error

    def geometry(model, data, configuration, **shared):
        # The real geometry initializer imports these declarations before its
        # first shared guard check. A cold combined initializer must allow that.
        import_module("mujoco.introspect.functions")
        import_module("mujoco.introspect.structs")
        shared["_guard"].check()
        raise intended_error

    monkeypatch.setattr(subject, "import_module", import_module)
    monkeypatch.setattr(subject, "MujocoNativeGuard", Guard)
    monkeypatch.setattr(subject, "MujocoScratchSession", Session)
    monkeypatch.setattr(subject, "IsolatedGeomDistanceReader", geometry)
    return intended_error, events


def test_cold_geometry_imports_precede_namespace_enrollment(monkeypatch):
    intended_error, events = constructor_fixture(monkeypatch)
    with pytest.raises(ConstructorReached) as error:
        subject.MujocoQueryReader(object(), object(), object(), {})
    assert error.value is intended_error
    enrollment = next(i for i, event in enumerate(events) if event[0] == "enroll")
    for name in ("mujoco.introspect.functions", "mujoco.introspect.structs"):
        assert events.index(("import", name)) < enrollment
    assert events[-2:] == [("close", "session"), ("close", "guard")]


def test_cleanup_failure_preserves_the_initial_constructor_error(monkeypatch):
    cleanup_error = RuntimeError("owned cleanup failure")
    intended_error, events = constructor_fixture(monkeypatch, cleanup_error=cleanup_error)
    with pytest.raises(ConstructorReached) as error:
        subject.MujocoQueryReader(object(), object(), object(), {})
    assert error.value is intended_error
    assert error.value.__cause__ is cleanup_error
    assert events[-1:] == [("close", "guard")]
