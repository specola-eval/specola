"""Read declared initialized fields from owned native MuJoCo allocations.

Field membership and candidate layouts come from the independent producer
census. This reader does not decide semantic closure or authorize an adapter:
its provider must resolve the retained semantic and initialization obligations.
The enclosing adapter owns synchronization and the floating-environment scope.
"""

from contextlib import ExitStack
import ctypes
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np

from ..conformance.native_images import PinnedNativeFile
from ..conformance.query_bindings import QueryBindingError
from ..native_values import NativeValue


@dataclass(frozen=True)
class NativeFieldValue:
    """An owned raw field value with its native containing path and extents."""

    field_id: str
    root: str
    native_path: tuple
    native_shape: tuple
    value: NativeValue


class MujocoNativeFields:
    """Capture candidate main-buffer and embedded-field storage without casts."""

    def __init__(self, model, data, guard, *, _profile=None):
        self._held = ExitStack()
        self._closed = False
        self._model, self._data, self._guard = model, data, guard
        try:
            path = Path(__file__).parent.parent / "data" / ("mujoco-fields-" + guard._version + ".json")
            # Completion uses the same allocation checks with a separately
            # sourced field subset. Its caller retains semantic source custody;
            # supplying a layout does not establish initialization or meaning.
            self.profile = json.loads(path.read_text()) if _profile is None else _profile
            if self.profile["version"] != guard._version:
                raise QueryBindingError("native field declaration version differs")
            self._fields = {row["field_id"]: row for row in self.profile["fields"]}
            if len(self._fields) != len(self.profile["fields"]):
                raise QueryBindingError("duplicate independent native field declaration")
            self._headers = tuple(self._held.enter_context(PinnedNativeFile(
                guard._distribution.locate_file("mujoco/include/mujoco/" + name), digest))
                for name, digest in self.profile["headers"].items())
            self._roots = {"mjModel_": model, "mjData_": data}
            self._validate_subject()
            self._bases = {name: owner._address for name, owner in self._roots.items()}
            for name, address in self._bases.items():
                layout = self.profile["roots"][name]
                if (type(address) is not int or address <= 0 or address % layout["alignment"]
                        or address + layout["size"] > 1 << (8 * ctypes.sizeof(ctypes.c_void_p))):
                    raise QueryBindingError("native field root has an invalid owned address")
            self._buffers = self._buffer_domains()
            self._bindings = self._field_regions()
        except BaseException:
            self._held.close()
            raise

    def _validate_subject(self):
        self._guard.check()
        import mujoco

        if type(self._model) is not mujoco.MjModel or type(self._data) is not mujoco.MjData:
            raise QueryBindingError("native fields require exact owned producer types")
        if self._data.model is not self._model:
            raise QueryBindingError("native fields model and data identities differ")
        if int(self._model.nplugin):
            raise QueryBindingError("native field plugin initialization remains unresolved")
        for header in self._headers:
            header.validate()

    def _root_bytes(self, root, offset, size):
        limit = self.profile["roots"][root]["size"]
        if type(offset) is not int or type(size) is not int or not 0 <= offset <= offset + size <= limit:
            raise QueryBindingError("native field interval lies outside its owned struct")
        return ctypes.string_at(self._bases[root] + offset, size)

    def _root_integer(self, root, field, *, pointer=False):
        identity = "c:@S@" + root + "@FI@" + field
        native = self._fields[identity]["native"]
        kind = native["type"]["canonical_kind"]
        if native["bitfield"] or native["offset_bits"] % 8:
            raise QueryBindingError("native address or count is not byte aligned")
        signed = kind in ("INT", "LONG", "LONGLONG", "SHORT")
        if pointer:
            if kind != "POINTER" or native["type"]["size"] != ctypes.sizeof(ctypes.c_void_p):
                raise QueryBindingError("native allocation pointer width differs")
        elif not signed and kind not in ("UINT", "ULONG", "ULONGLONG", "USHORT"):
            raise QueryBindingError("native count has no declared integer representation")
        raw = self._root_bytes(root, native["offset_bits"] // 8, native["type"]["size"])
        return int.from_bytes(raw, sys.byteorder, signed=signed)

    def _buffer_domains(self):
        result = {}
        for root in self._roots:
            start = self._root_integer(root, "buffer", pointer=True)
            size = self._root_integer(root, "nbuffer")
            if size < 0 or start < 0 or (size and not start) or start + size > 1 << (8 * ctypes.sizeof(ctypes.c_void_p)):
                raise QueryBindingError("native main-buffer extent is invalid")
            result[root] = start, size
        return result

    def _shape(self, dimensions):
        shape = []
        for factors in dimensions:
            values = []
            for factor in factors:
                if type(factor) is int:
                    value = factor
                elif type(factor) is dict and set(factor) == {"field"}:
                    identity = factor["field"]
                    prefix = "c:@S@mjModel_@FI@"
                    if not identity.startswith(prefix):
                        raise QueryBindingError("native extent has an unresolved owner")
                    value = self._root_integer("mjModel_", identity[len(prefix):])
                else:
                    raise QueryBindingError("native extent has an undeclared expression")
                if value < 0:
                    raise QueryBindingError("native extent is negative")
                values.append(value)
            size = math.prod(values)
            if size > sys.maxsize:
                raise QueryBindingError("native extent exceeds the addressable domain")
            shape.append(size)
        return tuple(shape)

    def _field_regions(self):
        """Resolve addresses and header counts without copying array elements."""
        regions = {}
        for row in self.profile["fields"]:
            if row["unresolved"]:
                continue
            for plan in row["plans"]:
                address = row["field_id"], plan["root"], tuple(plan["native_path"])
                if address in regions:
                    raise QueryBindingError("duplicate native field owner address")
                native_shape = self._shape(plan["native_dimensions"])
                binding_shape = self._shape(plan["binding_dimensions"])
                layout = plan["layout"]
                dtype = np.dtype(layout["kind"] + str(layout["itemsize"]))
                count = math.prod(native_shape)
                size = count * dtype.itemsize
                if count != math.prod(binding_shape) or size > sys.maxsize:
                    raise QueryBindingError("native and binding field extents lose components")
                if plan["pointer"]:
                    raw_pointer = self._root_bytes(plan["root"], plan["offset"], plan["field_width"])
                    if len(raw_pointer) != ctypes.sizeof(ctypes.c_void_p):
                        raise QueryBindingError("native field pointer width differs")
                    pointer = int.from_bytes(raw_pointer, sys.byteorder)
                    start, extent = self._buffers[plan["root"]]
                    alignment = layout["native_type"]["alignment"]
                    if size and (not pointer or alignment <= 0 or pointer % alignment
                                 or not start <= pointer <= pointer + size <= start + extent):
                        raise QueryBindingError("native field elements lie outside their declared main buffer")
                else:
                    if size != plan["field_width"]:
                        raise QueryBindingError("inline field extent differs from its native type")
                    if not 0 <= plan["offset"] <= plan["offset"] + size <= self.profile["roots"][plan["root"]]["size"]:
                        raise QueryBindingError("native field interval lies outside its owned struct")
                    pointer = self._bases[plan["root"]] + plan["offset"]
                regions[address] = (pointer, size, native_shape, binding_shape, dtype.str)
        return regions

    def read(self):
        """Return only declared element bytes, retaining stored-stage meaning."""
        if self._closed:
            raise QueryBindingError("native field reader is closed")
        self._validate_subject()
        if any(owner._address != self._bases[root] for root, owner in self._roots.items()):
            raise QueryBindingError("native field root allocation changed")
        if self._buffer_domains() != self._buffers:
            raise QueryBindingError("native field main-buffer allocation changed")
        regions = self._field_regions()
        if regions != self._bindings:
            raise QueryBindingError("native field pointer, layout or owner population changed")
        captured = []
        for address, (pointer, size, native_shape, binding_shape, dtype) in regions.items():
            raw = ctypes.string_at(pointer, size) if size else b""
            value = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(binding_shape)
            captured.append(NativeFieldValue(*address, native_shape, NativeValue.capture(value)))
        if self._buffer_domains() != self._buffers:
            raise QueryBindingError("native field allocation changed during capture")
        if self._field_regions() != regions:
            raise QueryBindingError("native field pointer or layout changed during capture")
        self._validate_subject()
        return tuple(captured)

    def close(self):
        """Retire source custody without closing caller-owned producer state."""
        if not self._closed:
            self._held.close()
            self._closed = True
