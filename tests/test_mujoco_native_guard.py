"""The query guard binds installed declarations to the loaded producer."""

from importlib import metadata
from dataclasses import replace
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from specola.adapters.mujoco_native import MujocoNativeGuard
from specola.conformance.native_images import NativeImageError

pytestmark = pytest.mark.mujoco

_SCENE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.3">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def native_guard():
    mujoco = pytest.importorskip("mujoco")
    with MujocoNativeGuard(mujoco) as guard:
        yield mujoco, guard


@pytest.fixture
def observed_adapter(monkeypatch):
    """A plain MuJoCo adapter one step past reset, counting full guard checks."""
    mujoco = pytest.importorskip("mujoco")
    from specola.adapters.mujoco import MujocoAdapter

    model = mujoco.MjModel.from_xml_string(_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = MujocoAdapter(SimpleNamespace(model=model, data=data))
    adapter.reset()
    mujoco.mj_step(model, data)
    full = MujocoNativeGuard._verify
    verified = []

    def counted(self):
        verified.append(self)
        return full(self)

    monkeypatch.setattr(MujocoNativeGuard, "_verify", counted)
    yield adapter, verified
    adapter.close_native_extraction()


def test_an_observation_verifies_the_native_image_only_at_its_boundaries(observed_adapter):
    # Maintainer decision 2026-10-07 (issue #136): one full check when the
    # observation begins and one when it ends; callers in between reuse it.
    adapter, verified = observed_adapter
    adapter.get_observation()
    assert len(verified) == 2
    adapter.get_observation()
    assert len(verified) == 4


def test_a_failed_end_boundary_check_withdraws_the_observation(observed_adapter, monkeypatch):
    adapter, verified = observed_adapter
    counted = MujocoNativeGuard._verify

    def fail_at_end(self):
        receipt = counted(self)
        if len(verified) == 2:
            raise NativeImageError("MuJoCo native target identity changed")
        return receipt

    monkeypatch.setattr(MujocoNativeGuard, "_verify", fail_at_end)
    with pytest.raises(NativeImageError, match="target identity changed"):
        adapter.get_observation()
    assert len(verified) == 2
    # The end boundary follows the base record, so the recorded observation
    # is withdrawn and the episode requires a reset.
    assert adapter.last_observation is None
    with pytest.raises(RuntimeError, match="reset"):
        adapter.get_observation()


@pytest.fixture
def fresh_adapter(monkeypatch):
    """A plain MuJoCo adapter before its first reset, counting full guard checks."""
    mujoco = pytest.importorskip("mujoco")
    from specola.adapters.mujoco import MujocoAdapter

    model = mujoco.MjModel.from_xml_string(_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = MujocoAdapter(SimpleNamespace(model=model, data=data))
    full = MujocoNativeGuard._verify
    verified = []

    def counted(self):
        verified.append(self)
        return full(self)

    monkeypatch.setattr(MujocoNativeGuard, "_verify", counted)
    yield adapter, verified
    adapter.close_native_extraction()


def test_a_reset_verifies_its_guard_at_construction_and_two_boundaries(fresh_adapter):
    # Maintainer decision 2026-10-12 (issue #136): the guard a reset binds
    # keeps its construction check, then one full check when the reset's
    # reads begin and one when the reset ends; callers in between reuse it.
    adapter, verified = fresh_adapter
    adapter.reset()
    first = adapter._native_projection._guard
    assert verified == [first] * 3
    verified.clear()
    adapter.reset()
    second = adapter._native_projection._guard
    assert second is not first
    # Retiring the first subject is checked on its own guard.
    assert [guard for guard in verified if guard is not first] == [second] * 3


def test_a_failed_end_check_at_reset_requires_another_reset(fresh_adapter, monkeypatch):
    adapter, verified = fresh_adapter
    counted = MujocoNativeGuard._verify

    def fail_at_end(self):
        receipt = counted(self)
        if len(verified) == 3:
            raise NativeImageError("MuJoCo native target identity changed")
        return receipt

    monkeypatch.setattr(MujocoNativeGuard, "_verify", fail_at_end)
    with pytest.raises(NativeImageError, match="target identity changed"):
        adapter.reset()
    assert len(verified) == 3
    # The end boundary follows the recorded reset observation, so it is
    # withdrawn and the episode needs another reset.
    assert adapter.last_observation is None
    with pytest.raises(RuntimeError, match="reset"):
        adapter.get_observation()
    monkeypatch.setattr(MujocoNativeGuard, "_verify", counted)
    adapter.reset()
    adapter.get_observation()


def test_a_binding_failure_at_reset_raises_its_own_error(fresh_adapter, monkeypatch):
    from specola.adapters.mujoco_scratch import MujocoScratchSession

    def refuse(self, *args, **kwargs):
        raise RuntimeError("scratch session refused")

    monkeypatch.setattr(MujocoScratchSession, "__init__", refuse)
    adapter, verified = fresh_adapter
    with pytest.raises(RuntimeError, match="scratch session refused"):
        adapter.reset()
    # Construction and the start boundary only: a raised reset runs no end
    # check on the guard its failed binding already closed.
    assert len(verified) == 2
    assert adapter.last_observation is None


def test_a_receipt_is_held_for_the_opening_thread_only(native_guard, monkeypatch):
    import threading
    mujoco, guard = native_guard
    full = MujocoNativeGuard._verify
    calls = []

    def counted(self):
        calls.append(threading.get_ident())
        return full(self)

    monkeypatch.setattr(MujocoNativeGuard, "_verify", counted)
    with guard.observation():
        before = len(calls)
        guard.check()                       # the opening thread: cached
        assert len(calls) == before
        seen = []
        worker = threading.Thread(target=lambda: seen.append(guard.check()))
        worker.start()
        worker.join()
        assert len(calls) == before + 1     # another thread: verified in full
        assert calls[-1] != threading.get_ident()
        assert seen


def test_observations_do_not_nest(native_guard):
    mujoco, guard = native_guard
    with guard.observation():
        with pytest.raises(RuntimeError, match="already open"):
            with guard.observation():
                pass
    # The outer interval closed cleanly and a new one can open.
    with guard.observation():
        pass


def _readelf(path, option):
    return subprocess.run(
        ["readelf", "--wide", option, str(path)], check=True,
        capture_output=True, text=True).stdout


def _assert_receipt(receipt):
    distribution = metadata.distribution("mujoco")
    mapped_backings = set()
    mapped_rows = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(None, 5)
        major, minor = (int(value, 16) for value in fields[3].split(":"))
        backing = (major, minor, int(fields[4]))
        mapped_backings.add(backing)
        start, end = (int(value, 16) for value in fields[0].split("-"))
        mapped_rows.append((start, end, fields[1], int(fields[2], 16), backing))
    expected_images = set()
    image_identities = {}
    for item in distribution.files:
        if not (item.name.endswith(".so") or ".so." in item.name):
            continue
        stat = Path(distribution.locate_file(item)).stat()
        if (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino) in mapped_backings:
            expected_images.add(item.as_posix())
            backing = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
            loads = [line.split() for line in _readelf(distribution.locate_file(item), "--program-headers").splitlines()
                     if line.split()[:1] == ["LOAD"]]
            first = min(loads, key=lambda fields: int(fields[1], 16))
            # These measured wheel images declare a first LOAD at file offset
            # zero. A different layout fails this bounded oracle explicitly.
            assert int(first[1], 16) == 0
            mapped, = [row for row in mapped_rows if row[4] == backing and row[3] == 0]
            page_size = os.sysconf("SC_PAGE_SIZE")
            bias = mapped[0] - int(first[2], 16) // page_size * page_size
            image_identities[item.as_posix()] = (bias, backing)
    assert {row[0] for row in receipt.images} == expected_images
    assert {name: (bias, backing) for name, bias, backing in receipt.images} == image_identities
    engine = Path(distribution.locate_file(receipt.engine_id))
    symbols = _readelf(engine, "--dyn-syms")
    definitions = {}
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) >= 8 and re.fullmatch(r"\d+:", fields[0]) and fields[6] != "UND":
            definitions[fields[7]] = (int(fields[1], 16), int(fields[2], 16 if fields[2].startswith("0x") else 10))
    table_start, table_size = definitions["mjCOLLISIONFUNC"]
    expected = set()
    for image_id, bias, _ in receipt.images:
        for line in _readelf(distribution.locate_file(image_id), "--relocs").splitlines():
            fields = line.split()
            if len(fields) < 5 or re.fullmatch(r"[0-9a-f]+", fields[0]) is None:
                continue
            offset = int(fields[0], 16)
            if fields[4] in definitions:
                if image_id == receipt.engine_id and table_start <= offset < table_start + table_size:
                    continue
                expected.add((image_id, bias + offset, fields[4]))
    actual = {(image_id, witness.address, witness.symbol)
              for image_id, witness in receipt.imports}
    assert actual == expected
    assert len(actual) == len(receipt.imports)
    # ELF64 ABI establishes the pointer width; this oracle does not call the
    # production table planner or derive its denominator from the receipt.
    engine_bias = next(bias for name, bias, _ in receipt.images if name == receipt.engine_id)
    assert {row.address for row in receipt.table} == {
        engine_bias + offset for offset in range(table_start, table_start + table_size, 8)}
    assert len({row.address for _, row in receipt.imports} | {
        row.address for row in receipt.table}) == len(receipt.imports) + len(receipt.table)
    with Path("/proc/self/mem").open("rb", buffering=0) as memory:
        def pointer(address):
            assert any(start <= address and address + 8 <= end and "r" in permissions
                       for start, end, permissions, _, _ in mapped_rows)
            raw = os.pread(memory.fileno(), 8, address)
            assert len(raw) == 8
            return int.from_bytes(raw, "little")

        for name, witness in receipt.imports:
            assert witness.value == pointer(witness.address)
            target = engine_bias + definitions[witness.symbol][0]
            assert witness.target_role == ("target" if witness.value == target else "importer")
        for witness in receipt.table:
            assert witness.value == pointer(witness.address)
            assert witness.target_role == "target"


def test_loaded_slots_match_independent_gnu_declarations(native_guard):
    _, guard = native_guard
    receipt = guard.check()
    _assert_receipt(receipt)
    assert guard.check() == receipt


@pytest.mark.parametrize("component", ("images", "imports", "table"))
def test_independent_relation_detects_an_omitted_runtime_component(native_guard, component):
    _, guard = native_guard
    receipt = guard.check()
    mutated = replace(receipt, **{component: getattr(receipt, component)[1:]})
    with pytest.raises(AssertionError):
        _assert_receipt(mutated)


@pytest.mark.parametrize("component", ("bias", "backing", "value", "role"))
def test_independent_relation_detects_changed_runtime_identity(native_guard, component):
    _, guard = native_guard
    receipt = guard.check()
    if component in ("bias", "backing"):
        name, bias, backing = receipt.images[0]
        changed = (name, bias + 1, backing) if component == "bias" else (name, bias, (0, 0, 0))
        mutated = replace(receipt, images=(changed,) + receipt.images[1:])
    else:
        name, witness = receipt.imports[0]
        changed = (replace(witness, value=witness.value + 1) if component == "value"
                   else replace(witness, target_role="stubbed-success"))
        mutated = replace(receipt, imports=((name, changed),) + receipt.imports[1:])
    with pytest.raises(AssertionError):
        _assert_receipt(mutated)


@pytest.mark.parametrize("location", ("public_alias", "extension", "descriptor"))
def test_python_replacement_fails_before_a_native_read(native_guard, monkeypatch, location):
    mujoco, guard = native_guard
    if location == "public_alias":
        monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: 0)
    elif location == "extension":
        monkeypatch.setattr(mujoco._functions, "mj_geomDistance", lambda *args: 0)
    else:
        monkeypatch.setattr(mujoco.MjData, "__copy__", lambda *args: None)
    with pytest.raises(NativeImageError, match="Python.*identity"):
        guard.check()


def test_unused_new_native_module_member_changes_the_enrolled_relation(native_guard, monkeypatch):
    mujoco, guard = native_guard
    monkeypatch.setattr(mujoco._functions, "unused_native_observation", len, raising=False)
    with pytest.raises(NativeImageError, match="Python.*population"):
        guard.check()


def test_unregistered_native_module_is_not_hidden_by_distribution_membership(native_guard, monkeypatch):
    mujoco, guard = native_guard
    added = ModuleType("mujoco._new_native_type")
    added.__file__ = str(Path(mujoco.__file__).with_name("_new_native_type.so"))
    monkeypatch.setitem(sys.modules, added.__name__, added)
    with pytest.raises(NativeImageError, match="module population"):
        guard.check()


def test_guard_closure_rejects_subsequent_reads(native_guard):
    _, guard = native_guard
    guard.close()
    with pytest.raises(NativeImageError, match="closed"):
        guard.check()


@pytest.mark.parametrize("location", ("public_alias", "extension", "native_alias"))
def test_enrollment_rejects_ordinary_preexisting_replacements(monkeypatch, location):
    mujoco = pytest.importorskip("mujoco")
    if location == "public_alias":
        monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: 0)
    elif location == "extension":
        monkeypatch.setattr(mujoco._functions, "mj_geomDistance", lambda *args: 0)
    else:
        monkeypatch.setattr(mujoco, "mj_geomDistance", mujoco.mj_contactForce)
        monkeypatch.setattr(mujoco._functions, "mj_geomDistance", mujoco.mj_contactForce)
    with pytest.raises(NativeImageError, match="query.*identity"):
        MujocoNativeGuard(mujoco)


def test_unrecorded_plugin_on_disk_is_an_unresolved_producer_image(monkeypatch, tmp_path):
    mujoco = pytest.importorskip("mujoco")
    distribution = metadata.distribution("mujoco")
    # The producer walks its plugin directory independently of wheel RECORD.
    # An owned package mirror reproduces that discrepancy without modifying
    # the installed simulator or loading the added file.
    for item in distribution.files:
        if item.name.endswith(".so") or ".so." in item.name:
            target = tmp_path / item
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(distribution.locate_file(item))
    overlay = tmp_path / "mujoco/plugin/unrecorded.so"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_bytes(b"owned loader candidate, never executed")

    class MirroredDistribution:
        version = distribution.version
        files = distribution.files

        def locate_file(self, relative):
            return tmp_path / relative

    monkeypatch.setattr(metadata, "distribution", lambda name: MirroredDistribution())
    with pytest.raises(NativeImageError, match="on-disk.*population"):
        MujocoNativeGuard(mujoco)
