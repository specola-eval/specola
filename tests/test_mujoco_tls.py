"""Inert byte-storage controls; native TLS fidelity needs a separate fixture."""

from contextlib import nullcontext
import threading
from types import SimpleNamespace

import pytest

from specola.adapters import mujoco_tls
from specola.conformance.native_images import NativeImageError, NativeMapping
from specola.conformance.query_bindings import QueryBindingError


@pytest.fixture
def storage(monkeypatch):
    monkeypatch.setattr(mujoco_tls, "_ACTIVE", threading.local())
    base = 0x4000
    data = bytearray(range(16))
    mapping = (NativeMapping(base, base + len(data), "rw-p", 0, 0, 0, 0, b""),)
    reads, writes = [], []

    class Reader:
        def __init__(self, mappings):
            assert mappings == mapping

        def __enter__(self):
            return self

        def __exit__(self, *error):
            pass

        def read(self, address, size):
            reads.append((address, size))
            return bytes(data[address - base:address - base + size])

    def copy(address, value, size):
        writes.append((address, bytes(value)))
        assert size == len(value)
        data[address - base:address - base + size] = value

    monkeypatch.setattr(mujoco_tls.native, "NativeMemoryReader", Reader)
    monkeypatch.setattr(mujoco_tls.ctypes, "memmove", copy)
    guard = object.__new__(mujoco_tls.MujocoExistingTls)
    guard._closed = guard._failed = False
    guard._lock = threading.Lock()
    guard._source = {"producer": {
        "symbols": [{"elf_tls_offset": 1, "size": 3}, {"elf_tls_offset": 7, "size": 4}],
        "PT_TLS": {"alignment": 8, "memsz": len(data)}}}
    monkeypatch.setattr(guard, "_lookup", lambda: (7, base, mapping))
    monkeypatch.setattr(guard, "check", lambda: (0x1000, 0x2000, 0x2040, 10))
    return SimpleNamespace(guard=guard, data=data, base=base, mapping=mapping,
                           reads=reads, writes=writes)


@pytest.mark.parametrize("exception", [False, True])
def test_exact_storage_restoration_on_both_exits(storage, exception):
    original = bytes(storage.data)
    failure = RuntimeError("fixture query failure")
    with pytest.raises(RuntimeError, match="fixture query failure") if exception else nullcontext():
        with storage.guard.preserve():
            storage.data[:] = b"x" * len(storage.data)
            if exception:
                raise failure
    expected = bytearray(b"x" * len(original))
    # Expected intervals are authored independently of the guard's resolver.
    expected[1:4], expected[7:11] = original[1:4], original[7:11]
    assert storage.data == expected
    assert storage.writes == [(0x4001, original[1:4]), (0x4007, original[7:11])]
    assert not storage.guard._failed


def test_restore_stub_is_detected_and_poisoned(storage, monkeypatch):
    monkeypatch.setattr(mujoco_tls.ctypes, "memmove", lambda *args: None)
    with pytest.raises(QueryBindingError, match="byte restoration failed"):
        with storage.guard.preserve():
            storage.data[1] ^= 0xff
    assert storage.guard._failed
    previous_reads = tuple(storage.reads)
    with pytest.raises(QueryBindingError, match="restoration is unresolved"):
        with storage.guard.preserve():
            pytest.fail("poisoned guard entered")
    assert tuple(storage.reads) == previous_reads


def test_changed_module_allocation_refuses_before_write(storage, monkeypatch):
    with pytest.raises(QueryBindingError, match="allocation changed"):
        with storage.guard.preserve():
            monkeypatch.setattr(storage.guard, "_lookup", lambda: (8, storage.base, storage.mapping))
    assert storage.writes == []
    assert storage.guard._failed


def test_cross_thread_exit_refuses_before_write(storage, monkeypatch):
    thread = threading.get_ident()
    identities = iter((thread, thread + 1))
    monkeypatch.setattr(mujoco_tls, "threading", SimpleNamespace(get_ident=lambda: next(identities)))
    with pytest.raises(QueryBindingError, match="another thread"):
        with storage.guard.preserve():
            pass
    assert storage.writes == []
    assert storage.guard._failed


def test_reentrant_preservation_retains_outer_scope(storage):
    with storage.guard.preserve():
        for _ in range(2):
            with pytest.raises(QueryBindingError, match="reentrant or concurrent"):
                with storage.guard.preserve():
                    pytest.fail("nested preservation entered")
        storage.data[1] = 99
    assert storage.data[1] == 1


@pytest.mark.parametrize("value", [(0, 0), (7, 0), (None, 0x4000), (7.0, 0x4000), [7, 0x4000]])
def test_missing_tls_never_initializes_native_storage(storage, monkeypatch, value):
    guard = storage.guard
    guard._bridge = SimpleNamespace(existing_tls=lambda *args: value)
    monkeypatch.setattr(mujoco_tls.native, "read_native_mappings",
                        lambda: pytest.fail("absent TLS reached allocation lookup"))
    with pytest.raises(QueryBindingError, match="cannot initialize"):
        mujoco_tls.MujocoExistingTls._lookup(guard)
    assert storage.reads == storage.writes == []


def test_read_only_tls_is_refused_before_storage_access(storage, monkeypatch):
    guard = storage.guard
    guard._bridge = SimpleNamespace(existing_tls=lambda *args: (7, storage.base))
    mapping = (NativeMapping(storage.base, storage.base + len(storage.data),
                             "r--p", 0, 0, 0, 0, b""),)
    monkeypatch.setattr(mujoco_tls.native, "read_native_mappings", lambda: mapping)
    with pytest.raises(NativeImageError, match="incompatible permissions"):
        mujoco_tls.MujocoExistingTls._lookup(guard)
    assert storage.reads == storage.writes == []


def test_removed_bridge_module_is_not_reimported(storage, monkeypatch):
    storage.guard._bridge = SimpleNamespace()
    storage.guard._bridge_identity = ()
    monkeypatch.delitem(mujoco_tls.sys.modules, "specola._native_tls", raising=False)
    monkeypatch.setattr(mujoco_tls, "import_module",
                        lambda *args: pytest.fail("removed bridge was reimported"))
    with pytest.raises(QueryBindingError, match="namespace changed"):
        storage.guard._check_bridge()


def test_foreign_exit_poisons_owner_thread_and_guard(storage):
    context = storage.guard.preserve()
    context.__enter__()
    owner_state = mujoco_tls._ACTIVE.state
    outcomes = []

    def exit_elsewhere():
        foreign = SimpleNamespace(active=True, failed=False)
        mujoco_tls._ACTIVE.state = foreign
        try:
            context.__exit__(None, None, None)
        except BaseException as error:
            outcomes.append(error)
        outcomes.append((foreign.active, foreign.failed))

    worker = threading.Thread(target=exit_elsewhere)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "foreign exit worker did not finish"
    assert storage.writes == [], "TLS bytes were restored on a foreign thread"
    assert storage.guard._failed
    assert not owner_state.active and owner_state.failed
    assert len(outcomes) == 2 and type(outcomes[0]) is QueryBindingError
    assert "moved to another thread" in str(outcomes[0])
    assert outcomes[1] == (True, False)
    replacement = object.__new__(mujoco_tls.MujocoExistingTls)
    replacement.__dict__.update(storage.guard.__dict__)
    replacement._failed = False
    with pytest.raises(QueryBindingError, match="observing thread TLS restoration is unresolved"):
        with replacement.preserve():
            pytest.fail("a new guard accepted the poisoned observing thread")


def test_close_refuses_active_tls_interval(storage):
    with storage.guard.preserve():
        with pytest.raises(QueryBindingError, match="cannot close active"):
            storage.guard.close()
    assert not storage.guard._closed
