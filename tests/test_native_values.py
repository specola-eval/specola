"""Detached native value transport cannot substitute Python object behavior."""

import base64
import json
import struct

import numpy as np
import pytest

from specola.native_values import NativeEnumValue, NativeValue, NativeValueError


@pytest.mark.parametrize("value", (None, True, False, 2**64 - 1, -(2**63), "native name", b"\x00\xff", bytearray(b"raw"),
    (1, "x"), [1, "x"], {"count": 1, "labels": ["first", "second"]}, NativeEnumValue("native.Kind", "BOX", 3)))
def test_builtin_and_declared_enum_values_preserve_type_and_content(value):
    encoded = NativeValue.capture(value)
    restored = NativeValue.from_document(encoded.document()).restore()
    assert type(restored) is type(value)
    assert restored == value
    assert NativeValue.capture(restored) == encoded


@pytest.mark.parametrize("bits", ("0000000000000000", "8000000000000000", "7ff0000000000000", "fff0000000000000", "7ff8000000000012"))
def test_python_float_bits_survive_signed_zero_infinity_and_nan_payloads(bits):
    original = bytes.fromhex(bits)
    encoded = NativeValue.capture(struct.unpack(">d", original)[0])
    assert struct.pack(">d", encoded.restore()) == original
    json.dumps(encoded.document(), allow_nan=False)


@pytest.mark.parametrize("dtype", ("<f4", ">f8", "<i8", "<u8", "<c16", "S7", "U3", "datetime64[ns]", "V0"))
@pytest.mark.parametrize("shape", ((), (0, 3), (2, 3)))
def test_plain_arrays_preserve_dtype_shape_and_element_bytes(dtype, shape):
    array = np.zeros(shape, dtype=dtype)
    encoded = NativeValue.capture(array)
    restored = encoded.restore()
    assert restored.dtype == array.dtype
    assert restored.shape == array.shape
    assert restored.tobytes() == array.tobytes()
    assert not restored.flags.writeable


def test_noncontiguous_arrays_detach_without_changing_element_order():
    original = np.arange(24, dtype=">f8").reshape(4, 6)[::-1, ::2]
    before = original.tobytes(order="C")
    encoded = NativeValue.capture(original)
    original[:] = 0
    assert encoded.restore().tobytes(order="C") == before
    assert encoded.restore().shape == (4, 3)
    encoded.document()["shape"] = [100]
    assert encoded.restore().shape == (4, 3)


@pytest.mark.parametrize("value", (np.float32(0.5), np.uint64(2**64 - 1), np.int8(-1), np.complex64(1 + 2j), np.datetime64("2026-09-05"), np.str_(""), np.bytes_(b""), np.str_("native"), np.bytes_(b"native"), np.void(b"")))
def test_numpy_scalars_preserve_their_native_dtype(value):
    restored = NativeValue.capture(value).restore()
    assert type(restored) is type(value)
    assert restored.dtype == value.dtype
    assert restored.tobytes() == value.tobytes()


@pytest.mark.parametrize("dtype", ("S0", "U0", "V0"))
def test_zero_width_array_is_not_promoted_to_one_character(dtype):
    original = np.ndarray((2, 3), dtype=dtype, buffer=b"")
    restored = NativeValue.capture(original).restore()
    assert restored.dtype == original.dtype
    assert restored.dtype.itemsize == 0
    assert restored.shape == original.shape


@pytest.mark.parametrize("dtype", (np.dtype(object), np.dtype([("position", "f8")]),
    np.dtype("i4", metadata={"native_unit": "count"}), np.dtype([("unsafe", object)])))
def test_object_padding_and_metadata_need_a_declared_reader(dtype):
    with pytest.raises(NativeValueError, match="declared field/element reader"):
        NativeValue.capture(np.zeros(1, dtype=dtype))


def test_arbitrary_objects_and_array_subclasses_are_not_executed():
    class Poison:
        def __repr__(self):
            raise AssertionError("repr executed")
        def __array__(self):
            raise AssertionError("implicit array conversion executed")
    class PoisonArray(np.ndarray):
        def tobytes(self, *args, **kwargs):
            raise AssertionError("subclass reader executed")
    for value in (Poison(), np.zeros(1).view(PoisonArray)):
        with pytest.raises(NativeValueError, match="explicit declared reader"):
            NativeValue.capture(value)


@pytest.mark.parametrize("change", (
    lambda d: d.update(extra=True),
    lambda d: d.update(dtype="O"),
    lambda d: d.update(shape=[True]),
    lambda d: d.update(shape=[-1]),
    lambda d: d.update(shape=[2**100]),
    lambda d: d.update(data="not base64"),
    lambda d: d.update(data=base64.b64encode(b"short").decode()),
    lambda d: d.update(kind="arbitrary_constructor"),
))
def test_malformed_or_unsafe_array_documents_fail_before_native_views(change):
    document = NativeValue.capture(np.zeros(1)).document()
    change(document)
    with pytest.raises(NativeValueError):
        NativeValue.from_document(document)


def test_cycles_and_duplicate_record_fields_are_rejected():
    values = []
    values.append(values)
    with pytest.raises(NativeValueError, match="cycle"):
        NativeValue.capture(values)
    document = NativeValue.capture({"value": 1}).document()
    document["fields"] *= 2
    with pytest.raises(NativeValueError, match="duplicate"):
        NativeValue.from_document(document)
