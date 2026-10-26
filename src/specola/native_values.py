"""Detached typed values for native observations, without object execution.

This codec preserves declared values, not arbitrary Python objects, strides,
aliasing or simulator handles. Producers with structured/object NumPy dtypes
must supply a declared field/element reader instead of exposing padding or
object pointers as observations. Unsupported values fail rather than falling
back to repr, pickle, implicit array conversion or descriptor traversal.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any

import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr


class NativeValueError(ValueError):
    """A native value has no supported detached representation."""


@dataclass(frozen=True)
class NativeEnumValue:
    """Declared enum identity without importing or constructing its class."""

    native_type: str
    member: str
    value: Any


def _binary(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unbinary(value: str) -> bytes:
    if type(value) is not str:
        raise NativeValueError("binary payload must be base64 text")
    try:
        result = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise NativeValueError("invalid base64 payload") from error
    if _binary(result) != value:
        raise NativeValueError("binary payload is not canonical base64")
    return result


def _plain_dtype(dtype: np.dtype) -> str:
    if dtype.hasobject or dtype.metadata is not None or dtype.fields is not None or dtype.subdtype is not None:
        raise NativeValueError("structured, metadata-bearing or object dtype needs a declared field/element reader")
    descriptor = dtype_to_descr(dtype)
    if type(descriptor) is not str or descr_to_dtype(descriptor) != dtype:
        raise NativeValueError("NumPy dtype has no lossless plain descriptor")
    return descriptor


def _encode(value: Any, active: set) -> dict:
    kind = type(value)
    if value is None:
        return {"kind": "null"}
    if kind is bool:
        return {"kind": "bool", "value": value}
    if kind is int:
        return {"kind": "int", "value": str(value)}
    if kind is float:
        return {"kind": "float64", "data": _binary(struct.pack(">d", value))}
    if kind is complex:
        return {"kind": "complex128", "data": _binary(struct.pack(">dd", value.real, value.imag))}
    if kind is str:
        return {"kind": "string", "value": value}
    if kind in (bytes, bytearray):
        return {"kind": "bytes" if kind is bytes else "bytearray", "data": _binary(bytes(value))}
    if kind is np.ndarray:
        return {"kind": "ndarray", "dtype": _plain_dtype(value.dtype),
                "shape": list(value.shape), "data": _binary(value.tobytes(order="C"))}
    # Only concrete NumPy scalar classes are accepted. A user subclass must
    # not supply overridden array conversion or value-reading behavior.
    if issubclass(kind, np.generic) and np.dtype(kind).type is kind:
        if value.dtype.kind in ("S", "U"):
            scalar = bytes(value) if value.dtype.kind == "S" else str(value)
            return {"kind": "numpy_text_scalar", "dtype": _plain_dtype(value.dtype), "value": _encode(scalar, active)}
        return {"kind": "numpy_scalar", "dtype": _plain_dtype(value.dtype),
                "shape": [], "data": _binary(value.tobytes())}
    if kind not in (tuple, list, dict, NativeEnumValue):
        raise NativeValueError("native value needs an explicit declared reader")
    if id(value) in active:
        raise NativeValueError("native value contains a cycle")
    active.add(id(value))
    try:
        if kind in (tuple, list):
            return {"kind": "tuple" if kind is tuple else "list", "items": [_encode(item, active) for item in value]}
        if kind is dict:
            if any(type(key) is not str for key in value):
                raise NativeValueError("native record field names must be strings")
            return {"kind": "record", "fields": [[key, _encode(item, active)] for key, item in value.items()]}
        if type(value.native_type) is not str or not value.native_type.strip() or type(value.member) is not str or not value.member.strip():
            raise NativeValueError("native enum needs a type and member identity")
        return {"kind": "enum", "native_type": value.native_type, "member": value.member, "value": _encode(value.value, active)}
    finally:
        active.remove(id(value))


def _keys(document: dict, expected) -> None:
    if any(type(key) is not str for key in document) or set(document) != set(expected):
        raise NativeValueError("typed native value has missing or unknown fields")


def _decode(document: dict) -> Any:
    if type(document) is not dict or type(document.get("kind")) is not str:
        raise NativeValueError("typed native value must be a tagged object")
    kind = document["kind"]
    if kind == "null":
        _keys(document, ("kind",))
        return None
    if kind in ("bool", "int", "string"):
        _keys(document, ("kind", "value"))
        value = document["value"]
        if kind == "bool" and type(value) is bool:
            return value
        if kind == "string" and type(value) is str:
            return value
        if kind == "int" and type(value) is str:
            try:
                integer = int(value)
            except ValueError as error:
                raise NativeValueError("native integer payload is invalid") from error
            if str(integer) == value:
                return integer
        raise NativeValueError("native scalar has the wrong representation")
    if kind in ("bytes", "bytearray", "float64", "complex128"):
        _keys(document, ("kind", "data"))
        data = _unbinary(document["data"])
        if kind == "bytes":
            return data
        if kind == "bytearray":
            return bytearray(data)
        format_string = ">d" if kind == "float64" else ">dd"
        if len(data) != struct.calcsize(format_string):
            raise NativeValueError("native floating-point byte width differs")
        components = struct.unpack(format_string, data)
        return components[0] if kind == "float64" else complex(*components)
    if kind == "numpy_text_scalar":
        _keys(document, ("kind", "dtype", "value"))
        if type(document["dtype"]) is not str:
            raise NativeValueError("NumPy text scalar needs a dtype")
        try:
            dtype = descr_to_dtype(document["dtype"])
            if _plain_dtype(dtype) != document["dtype"] or dtype.kind not in ("S", "U"):
                raise NativeValueError("NumPy text scalar has a non-text dtype")
            value = _decode(document["value"])
            if type(value) is not (bytes if dtype.kind == "S" else str):
                raise NativeValueError("NumPy text scalar has the wrong value type")
            scalar = dtype.type(value)
            if scalar.dtype != dtype:
                raise NativeValueError("NumPy text scalar width differs from its value")
            return scalar
        except (TypeError, ValueError) as error:
            if isinstance(error, NativeValueError):
                raise
            raise NativeValueError("invalid NumPy text scalar: " + str(error)) from error
    if kind in ("ndarray", "numpy_scalar"):
        _keys(document, ("kind", "dtype", "shape", "data"))
        if type(document["dtype"]) is not str or type(document["shape"]) is not list:
            raise NativeValueError("native array needs a dtype and shape")
        shape = document["shape"]
        if any(type(size) is not int or size < 0 or size > np.iinfo(np.intp).max for size in shape):
            raise NativeValueError("native array dimensions are invalid")
        if kind == "numpy_scalar" and shape:
            raise NativeValueError("NumPy scalar has array dimensions")
        try:
            dtype = descr_to_dtype(document["dtype"])
            if _plain_dtype(dtype) != document["dtype"]:
                raise NativeValueError("native dtype descriptor is not canonical")
            data = _unbinary(document["data"])
            if math.prod(shape) * dtype.itemsize != len(data):
                raise NativeValueError("native array byte count differs from shape and dtype")
            if dtype.itemsize:
                array = np.frombuffer(data, dtype=dtype).reshape(tuple(shape))
            else:
                array = np.ndarray(tuple(shape), dtype=dtype, buffer=b"")
            return array[()] if kind == "numpy_scalar" else array
        except (TypeError, ValueError, OverflowError) as error:
            if isinstance(error, NativeValueError):
                raise
            raise NativeValueError("invalid native array encoding: " + str(error)) from error
    if kind in ("tuple", "list"):
        _keys(document, ("kind", "items"))
        if type(document["items"]) is not list:
            raise NativeValueError("native sequence items must be a list")
        items = [_decode(item) for item in document["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "record":
        _keys(document, ("kind", "fields"))
        if type(document["fields"]) is not list:
            raise NativeValueError("native record fields must be a list")
        record = {}
        for field in document["fields"]:
            if type(field) is not list or len(field) != 2 or type(field[0]) is not str or field[0] in record:
                raise NativeValueError("native record has malformed or duplicate fields")
            record[field[0]] = _decode(field[1])
        return record
    if kind == "enum":
        _keys(document, ("kind", "native_type", "member", "value"))
        if any(type(document[name]) is not str or not document[name].strip() for name in ("native_type", "member")):
            raise NativeValueError("native enum needs a type and member identity")
        return NativeEnumValue(document["native_type"], document["member"], _decode(document["value"]))
    raise NativeValueError("unknown native value kind: " + kind)


def _canonical(document: dict) -> str:
    return json.dumps(document, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class NativeValue:
    """An immutable typed snapshot whose public reads cannot alter its bytes."""

    canonical_json: str

    def __post_init__(self) -> None:
        if type(self.canonical_json) is not str:
            raise NativeValueError("native value encoding must be text")
        try:
            document = json.loads(self.canonical_json)
            _decode(document)
            if _canonical(document) != self.canonical_json:
                raise NativeValueError("native value encoding is not canonical")
        except (json.JSONDecodeError, RecursionError) as error:
            raise NativeValueError("invalid native value encoding: " + str(error)) from error

    @classmethod
    def capture(cls, value: Any) -> NativeValue:
        try:
            return cls(_canonical(_encode(value, set())))
        except RecursionError as error:
            raise NativeValueError("native value nesting exceeds parser capacity") from error

    @classmethod
    def from_document(cls, document: dict) -> NativeValue:
        try:
            _decode(document)
            return cls(_canonical(document))
        except (TypeError, ValueError, RecursionError) as error:
            raise NativeValueError("invalid native value document: " + str(error)) from error

    def document(self) -> dict:
        return json.loads(self.canonical_json)

    def restore(self) -> Any:
        return _decode(self.document())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
