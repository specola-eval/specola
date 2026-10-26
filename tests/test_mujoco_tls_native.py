"""Exercise the compiled TLS bridge against independent native declarations.

The dedicated RoboCasa protocol supplies exact compiled-fixture profiles in
SPECOLA_TLS_NATIVE_INPUTS. Each profile has a path and SHA256 in that JSON
manifest. Missing inputs skip ordinary installations; a supplied incomplete
manifest fails. Native setup and the dlinfo oracle run outside observation.
"""

from contextlib import ExitStack
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import BuiltinFunctionType, SimpleNamespace

import pytest


pytestmark = pytest.mark.robocasa


def _carrier(declaration):
    """Use the fixture's compiled ABI, including signedness and pointer width."""
    if declaration is None:
        return None
    if declaration["chain"][0]["tag"] == "DW_TAG_pointer_type":
        assert declaration["size"] == ctypes.sizeof(ctypes.c_void_p)
        return ctypes.c_void_p
    return {(4, 5): ctypes.c_int32, (4, 7): ctypes.c_uint32,
            (8, 5): ctypes.c_int64, (8, 7): ctypes.c_uint64}[
                declaration["size"], declaration["encoding"]]


def _load_image(stack, profile, *, module_name=None):
    from specola.conformance import native_images as native

    row = profile["extension"]
    image = stack.enter_context(native.PinnedNativeFile(row["path"], row["sha256"]))
    layout = native.read_elf_layout(image)
    assert layout.bind_now
    if module_name is None:
        owner = ctypes.PyDLL(str(image.path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    else:
        assert module_name not in sys.modules
        spec = importlib.util.spec_from_file_location(module_name, image.path)
        assert spec is not None and spec.loader is not None
        owner = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = owner
        spec.loader.exec_module(owner)
    image.validate()
    maps = native.read_native_mappings()
    bias = native.resolve_load_bias(layout.segments, image.mappings(maps), image.backing)
    providers = {}
    for name, declared in profile["provider_images"].items():
        identity = declared["identity"]
        file = stack.enter_context(native.PinnedNativeFile(identity["path"], identity["sha256"]))
        elf = native.read_elf_layout(file)
        providers[name] = (elf, native.resolve_load_bias(elf.segments, file.mappings(maps), file.backing))
    with native.NativeMemoryReader(maps) as reader:
        regions = [row["code"] for row in profile["functions"]]
        regions += [profile["literal_region"], profile["PLT_selected_import_stubs"]]
        for region in regions:
            assert reader.read(bias + region["rva"], region["size"]) == bytes.fromhex(region["hex"])
        for relocation in profile["relocations"]:
            disposition = relocation["disposition"]
            if disposition in {
                "CRT_initialization_or_teardown_outside_selected_call_interval",
                "optional_CRT_weak_reference_outside_selected_call_interval",
            }:
                continue
            if disposition == "self_relative":
                expected = bias + relocation["addend"]
            else:
                assert disposition in {"exact_libc_export", "exact_cpython_export", "terminal_stack_corruption_only"}
                provider, provider_bias = providers[relocation["expected_image"]]
                declaration = relocation["expected_symbol"]
                symbol = provider.symbols[declaration["index"]]
                assert (symbol.name, symbol.value, symbol.size, symbol.kind) == (
                    declaration["name"], declaration["value"], declaration["size"], declaration["type"])
                expected = provider_bias + symbol.value
            assert int.from_bytes(reader.read(bias + relocation["slot_rva"], layout.pointer_size), "little") == expected
    functions = {}
    for signature in profile.get("compiled_function_signatures", []):
        if signature["name"].startswith("fixture_"):
            arguments = []
            for parameter in signature["parameters"]:
                carrier = _carrier(parameter["type"])
                assert carrier is not None
                arguments.append(carrier)
            functions[signature["name"]] = ctypes.PYFUNCTYPE(
                _carrier(signature["result"]), *arguments,
            )(bias + signature["rva"])
    return SimpleNamespace(owner=owner, profile=profile, file=image, layout=layout,
                           bias=bias, functions=functions, providers=providers)


@pytest.fixture(scope="module")
def native_tls_inputs():
    from hashlib import sha256

    path = os.environ.get("SPECOLA_TLS_NATIVE_INPUTS")
    if path is None:
        pytest.skip("dedicated compiled TLS fixture inputs are absent")
    assert path is not None
    manifest = json.loads(Path(path).read_text())
    profiles = {}
    for name, row in manifest.items():
        raw = Path(row["path"]).read_bytes()
        assert sha256(raw).hexdigest() == row["sha256"], name
        profiles[name] = json.loads(raw)
    assert {"original", "flags", "bridge", "lookup", "callable", "flags_api"} <= profiles.keys()
    with ExitStack() as stack:
        # Load exact owned fixtures before observation; retain every image owner.
        original = _load_image(stack, profiles["original"])
        bridge = _load_image(stack, profiles["bridge"], module_name="specola._native_tls")
        flags = _load_image(stack, profiles["flags"], module_name="_native_tls_flags_fixture")
        yield SimpleNamespace(profiles=profiles, original=original, bridge=bridge, flags=flags)


def _evidence(tmp_path, name, value):
    (tmp_path / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")


def test_callback_layout_and_declared_modes(native_tls_inputs, tmp_path):
    inputs = native_tls_inputs
    native_abi = inputs.profiles["lookup"]["callback_abi"]
    expected_layout = (native_abi["pointer_bytes"], native_abi["pointer_bytes"], native_abi["info_size"],
                       *((field["offset"], field["size"]) for field in native_abi["fields"]))
    assert inputs.bridge.owner.layout() == expected_layout
    enum, = inputs.profiles["original"]["compiled_enums"]["fixture_mode"]
    modes = {row["name"]: row["value"] for row in enum["members"] if row["name"] != "FIXTURE_MODE_COUNT"}
    dispositions = {"FIXTURE_MATCH": "present", "FIXTURE_NO_TLS": "null", "FIXTURE_SHORT": "short",
                    "FIXTURE_ABSENT": "absent", "FIXTURE_DUPLICATE": "absent",
                    "FIXTURE_OTHER_IMAGE": "absent", "FIXTURE_OTHER_PHDR": "absent",
                    "FIXTURE_OTHER_PHNUM": "absent", "FIXTURE_ITERATOR_ERROR": "absent"}
    assert modes.keys() == dispositions.keys()
    original = inputs.original
    functions = original.functions
    signature, = [row for row in original.profile["compiled_function_signatures"] if row["name"] == "fixture_iterate"]
    callback = original.bias + signature["rva"]
    # These values are opaque fixture identifiers; the callback compares them.
    image, phdr, phnum, module = original.bias, original.bias + original.layout.segments[0].virtual_address, 1, 1
    rows = []
    for name, mode in modes.items():
        assert functions["fixture_configure"](mode, image, phdr, phnum, module) == 0
        try:
            result = inputs.bridge.owner.existing_tls(callback, image, phdr, phnum)
        except RuntimeError as error:
            result = str(error)
        calls = functions["fixture_call_count"]()
        rows.append({"mode": name, "result": result, "iterator_calls": calls})
        _evidence(tmp_path, "callback-modes", rows)
        assert calls == 1
        if dispositions[name] == "present":
            assert result == (module, functions["fixture_storage_address"]())
        elif dispositions[name] == "null":
            assert result == (module, 0)
        elif dispositions[name] == "short":
            assert result == "native TLS callback record is short"
        else:
            assert result == "native TLS image is absent or ambiguous"


@pytest.mark.parametrize("argument", range(4))
@pytest.mark.parametrize("invalid", [True, 1.0, -1, 1 << 64])
def test_invalid_arguments_never_reach_iterator(native_tls_inputs, argument, invalid, tmp_path):
    original = native_tls_inputs.original
    signature, = [row for row in original.profile["compiled_function_signatures"] if row["name"] == "fixture_iterate"]
    arguments = [original.bias + signature["rva"], original.bias, original.bias, 1]
    count = original.functions["fixture_call_count"]()
    arguments[argument] = invalid
    with pytest.raises((TypeError, OverflowError, ValueError)) as error:
        native_tls_inputs.bridge.owner.existing_tls(*arguments)
    after = original.functions["fixture_call_count"]()
    _evidence(tmp_path, "invalid-argument", {"argument": argument, "value": repr(invalid),
                                           "error": str(error.value), "calls": [count, after]})
    assert count == after


@pytest.fixture(scope="module")
def producer_tls(native_tls_inputs):
    import mujoco
    from specola.adapters.mujoco_native import MujocoNativeGuard
    from specola.adapters.mujoco_tls import MujocoExistingTls
    from specola.conformance import native_images as native

    assert mujoco.__version__ == "3.3.1"
    producer = native_tls_inputs.profiles["lookup"]["producer"]
    with ExitStack() as stack:
        image = stack.enter_context(native.PinnedNativeFile(producer["file"]["path"], producer["file"]["sha256"]))
        elf = native.read_elf_layout(image)
        # Reconcile independently supplied source names with this engine's ELF.
        ranges = []
        for row in producer["symbols"]:
            symbol, = [entry for entry in elf.local_symbols if entry.name == row["name"]]
            assert (symbol.kind, symbol.value, symbol.size) == ("STT_TLS", row["elf_tls_offset"], row["size"])
            ranges.append((symbol.name, symbol.value, symbol.size))
        path_buffer = ctypes.create_string_buffer(os.fsencode(image.path))
        functions = native_tls_inputs.original.functions
        handle = functions["fixture_open_existing"](ctypes.addressof(path_buffer))
        assert handle

        def release():
            assert functions["fixture_close_existing"](handle) == 0

        stack.callback(release)
        guard = stack.enter_context(MujocoNativeGuard(mujoco))
        tls = MujocoExistingTls(guard)
        stack.callback(tls.close)
        yield SimpleNamespace(mujoco=mujoco, guard=guard, tls=tls, handle=handle,
                              functions=functions, ranges=tuple(ranges), inputs=native_tls_inputs)


def _oracle(subject):
    status, module, data_status, base = ctypes.c_int(), ctypes.c_size_t(), ctypes.c_int(), ctypes.c_void_p()
    subject.functions["fixture_existing_tls"](subject.handle, ctypes.addressof(status), ctypes.addressof(module),
                                               ctypes.addressof(data_status), ctypes.addressof(base))
    assert status.value == 0 and data_status.value == 0
    return module.value, base.value or 0


def _snapshot(subject, identity=None):
    from specola.conformance import native_images as native

    module, base = _oracle(subject) if identity is None else identity
    assert module > 0 and base > 0
    with native.NativeMemoryReader(native.read_native_mappings()) as reader:
        return {name: reader.read(base + offset, size).hex() for name, offset, size in subject.ranges}


def test_missing_thread_tls_is_not_initialized(producer_tls, tmp_path):
    result = {}

    def worker():
        try:
            from specola.conformance.query_bindings import QueryBindingError
            result["before"] = _oracle(producer_tls)
            with pytest.raises(QueryBindingError, match="producer TLS is absent"):
                with producer_tls.tls.preserve():
                    pytest.fail("missing TLS must refuse before the body")
            result["after"] = _oracle(producer_tls)
        except BaseException as error:
            result["error"] = repr(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=60)
    assert not thread.is_alive()
    _evidence(tmp_path, "absent-tls", result)
    assert "error" not in result
    assert result["before"] == result["after"]
    assert result["before"][0] > 0 and result["before"][1] == 0


def test_same_body_wrong_flags_is_refused_before_invocation(producer_tls, tmp_path):
    from specola.adapters.mujoco_tls import MujocoExistingTls
    from specola.conformance.query_bindings import QueryBindingError

    inputs = producer_tls.inputs
    bridge = inputs.bridge.owner
    original = bridge.existing_tls
    assert type(original) is BuiltinFunctionType
    assert original.__name__ == "existing_tls" and original.__self__ is bridge
    producer_tls.tls.check()
    alternative = inputs.flags.owner.wrong_flags(original)
    assert type(alternative) is BuiltinFunctionType
    assert (alternative.__name__, alternative.__module__, alternative.__self__) == (
        original.__name__, original.__module__, bridge)
    _, python_bias = inputs.flags.providers["cpython"]
    address = inputs.profiles["callable"]["hot_function"]
    flags = inputs.profiles["flags_api"]["function"]
    getter = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)(python_bias + address["value"])
    flag_getter = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object)(python_bias + flags["value"])
    bodies = [getter(original), getter(alternative)]
    conventions = [flag_getter(original), flag_getter(alternative)]
    assert bodies[0] == bodies[1] and conventions[0] != conventions[1]
    try:
        bridge.existing_tls = alternative
        with pytest.raises(QueryBindingError, match="callable argument convention differs") as error:
            MujocoExistingTls(producer_tls.guard)
        _evidence(tmp_path, "wrong-flags", {"body_addresses": bodies, "flags": conventions, "error": str(error.value)})
    finally:
        bridge.existing_tls = original
    producer_tls.tls.check()


@pytest.mark.parametrize("exception_exit", [False, True])
def test_native_declared_ranges_restore_on_both_exits(producer_tls, exception_exit, tmp_path):
    mujoco = producer_tls.mujoco
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    assert data is not None  # Setup creates this thread's native allocation.
    before = _snapshot(producer_tls)
    identity = _oracle(producer_tls)
    assert producer_tls.tls._lookup()[:2] == identity
    inside = {}
    try:
        with producer_tls.tls.preserve():
            for name, offset, size in producer_tls.ranges:
                value = bytes.fromhex(before[name])
                # No producer executes while its opaque scratch bytes are changed.
                changed = bytes([value[0] ^ 1]) + value[1:]
                ctypes.memmove(identity[1] + offset, changed, size)
            inside = _snapshot(producer_tls, identity)
            if exception_exit:
                raise ArithmeticError("owned exception-exit control")
    except ArithmeticError:
        assert exception_exit
    after = _snapshot(producer_tls)
    _evidence(tmp_path, "restoration", {"identity": identity, "before": before, "inside": inside, "after": after})
    assert inside.keys() == before.keys() == after.keys()
    assert all(inside[name] != before[name] for name in before)
    assert after == before
