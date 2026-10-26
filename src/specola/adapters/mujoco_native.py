"""Bind MuJoCo queries to retained producer images and Python objects.

Enrollment assumes trusted producer initialization and a caller-synchronized
interval without unloading or remapping. It checks current targets and retains
Python identities; it does not attest historical pybind heap contents or private
executable bytes. The independent surface inventory remains a separate guard.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import threading
from types import BuiltinFunctionType, ModuleType
from typing import Tuple, cast

from ..conformance import native_images as native
from ..conformance.core_runner import EvidenceIdentity
from ..conformance.surface import (
    _class_dict, _type_name, _MODULE_DICTIONARY, _TYPE_METADATA, SurfaceValidationError,
)


def _namespace(value):
    if type(value) is ModuleType:
        result = dict(_MODULE_DICTIONARY.__get__(value))
    else:
        result = dict(_class_dict(value))
    if any(type(key) is not str for key in result):
        raise native.NativeImageError("Python namespace identity has non-string keys")
    return result


def _is_class(value):
    return type in _TYPE_METADATA["__mro__"].__get__(type(value))


def _native_file(path):
    if path.name.endswith(".so") or ".so." in path.name:
        return True
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


class _PythonIdentity:
    """Retain raw namespaces without invoking producer descriptors."""

    def __init__(self, modules):
        self.modules = dict(modules)
        self.namespaces = []
        pending = list(modules.values())
        seen = set()
        while pending:
            owner = pending.pop()
            if id(owner) in seen:
                continue
            seen.add(id(owner))
            namespace = _namespace(owner)
            self.namespaces.append((owner, namespace))
            for value in namespace.values():
                if _is_class(value) and _type_name(value).startswith("mujoco."):
                    pending.extend(_TYPE_METADATA["__mro__"].__get__(value))

    def check(self):
        for name, module in self.modules.items():
            if sys.modules.get(name) is not module:
                raise native.NativeImageError("Python module identity changed: " + name)
        for owner, expected in self.namespaces:
            current = _namespace(owner)
            if current.keys() != expected.keys():
                raise native.NativeImageError("Python namespace population changed")
            if any(current[key] is not value for key, value in expected.items()):
                raise native.NativeImageError("Python member identity changed")


@dataclass(frozen=True)
class MujocoNativeReceipt:
    """The complete loaded image, import and collision-table relation."""

    engine_id: str
    images: Tuple[tuple, ...]
    imports: Tuple[tuple, ...]
    table: Tuple[native.NativeSlotWitness, ...]


class MujocoNativeGuard:
    """Authenticate the loaded subset of a pinned distribution's images.

    Construct this guard at subject enrollment, retain it through observations,
    and close it when the subject is retired. Both observation boundaries must
    call :meth:`check`. File ownership does not synchronize external callers.
    """

    def __init__(self, mujoco):
        self._held = ExitStack()
        self._closed = False
        self._receipt = None
        self._observation = None
        try:
            if sys.platform != "linux" or type(mujoco) is not ModuleType:
                raise native.NativeImageError("unresolved MuJoCo native image platform or module")
            self._distribution = metadata.distribution("mujoco")
            self._version = self._distribution.version
            profile_path = Path(__file__).parent.parent / "data" / ("mujoco-native-" + self._version + ".json")
            if not profile_path.is_file():
                raise native.NativeImageError("unresolved MuJoCo native image build: " + self._version)
            self._profile = json.loads(profile_path.read_text())
            if self._profile["version"] != self._version or _namespace(mujoco).get("__version__") != self._version:
                raise native.NativeImageError("MuJoCo module and distribution version differ")
            self._images = {row["relative_path"]: row for row in self._profile["images"]}
            if len(self._images) != len(self._profile["images"]):
                raise native.NativeImageError("duplicate native image declaration")
            self._check_distribution()
            self._files = {
                name: self._held.enter_context(native.PinnedNativeFile(
                    self._distribution.locate_file(name), row["sha256"]))
                for name, row in self._images.items()}
            init = self._profile["package_init"]
            self._init_file = self._held.enter_context(native.PinnedNativeFile(
                self._distribution.locate_file("mujoco/__init__.py"), init["sha256"]))
            if Path(_namespace(mujoco)["__file__"]).resolve() != self._init_file.path.resolve():
                raise native.NativeImageError("MuJoCo package source identity differs")
            self._layouts = {name: native.read_elf_layout(image) for name, image in self._files.items()}
            self._engine_id = self._profile["engine_id"]
            self._engine = self._layouts[self._engine_id]
            tables = [symbol for symbol in self._engine.symbols
                      if symbol.name == "mjCOLLISIONFUNC" and symbol.section != "SHN_UNDEF"]
            if len(tables) != 1:
                raise native.NativeImageError("native collision table declaration is ambiguous")
            table, = tables
            self._initializers = tuple(self._profile["source_initializer"])
            self._table_bytes = native.read_elf_file_bytes(
                self._files[self._engine_id], self._engine, table.value, table.size)
            # Plans are pure functions of the retained frozen layouts and
            # policies, so each check reuses them instead of re-deriving.
            self._table_plan = native.plan_native_pointer_table(
                self._engine, table.name, self._initializers, self._table_bytes)
            self._projected_engine = replace(self._engine, relocations=tuple(
                row for row in self._engine.relocations
                if not table.value <= row.offset < table.value + table.size))
            self._policies = tuple(native.NativeObjectCoalescence(
                row["symbol_name"], row["importer_image_sha256"], row["target_image_sha256"],
                tuple(EvidenceIdentity(**source) for source in row["provenance"]))
                for row in self._profile["object_coalescence"])
            self._binding_plans = {}
            self._page_size = os.sysconf("SC_PAGE_SIZE")
            self._module_names = {row["python_module"]: name for name, row in self._images.items()
                                  if row["python_module"] is not None}
            modules = {name: sys.modules[name] for name in self._module_names if name in sys.modules}
            modules["mujoco"] = mujoco
            if {"mujoco", "mujoco._functions", "mujoco._structs"} - modules.keys():
                raise native.NativeImageError("required MuJoCo query module is absent")
            self._python = _PythonIdentity(modules)
            self._validate_query_identities()
            self._receipt = self.check()
        except BaseException:
            self.close()
            raise

    def _check_distribution(self):
        distribution = metadata.distribution("mujoco")
        # One RECORD read per check: each ``files`` access re-reads it.
        files = distribution.files
        if distribution.version != self._version or files is None:
            raise native.NativeImageError("MuJoCo distribution identity is absent or changed")
        images = {item.as_posix() for item in files
                  if _native_file(Path(str(distribution.locate_file(item))))}
        if images != self._images.keys():
            raise native.NativeImageError("MuJoCo distribution native image population differs")
        package = Path(str(distribution.locate_file("mujoco")))
        on_disk = {"mujoco/" + path.relative_to(package).as_posix() for path in package.rglob("*")
                   if path.is_file() and _native_file(path)}
        if on_disk != images:
            raise native.NativeImageError("MuJoCo on-disk native image population differs from distribution")
        for name in images:
            if Path(str(distribution.locate_file(name))).resolve() != Path(str(self._distribution.locate_file(name))).resolve():
                raise native.NativeImageError("MuJoCo distribution source location changed")

    def _loaded_images(self, mappings):
        package_prefix = os.fsencode(Path(str(self._distribution.locate_file("mujoco"))).resolve()) + b"/"
        known_backings = {image.backing for image in self._files.values()}
        for mapping in mappings:
            if (mapping.pathname.startswith(package_prefix)
                    and (mapping.pathname.endswith(b".so") or b".so." in mapping.pathname
                         or mapping.pathname.endswith(b".so (deleted)"))
                    and mapping.backing not in known_backings):
                raise native.NativeImageError("mapped MuJoCo native image population has an unpinned backing")
        for name, module in tuple(sys.modules.items()):
            if type(name) is not str or not name.startswith("mujoco."):
                continue
            if type(module) is not ModuleType:
                raise native.NativeImageError("Python producer module population is unresolved")
            filename = _namespace(module).get("__file__")
            if type(filename) is str and (filename.endswith(".so") or ".so." in filename):
                if name not in self._module_names:
                    raise native.NativeImageError("Python native module population contains an undeclared image")
        backings = {row.backing for row in mappings if row.backing is not None}
        loaded = tuple((name, native.resolve_load_bias(
            self._layouts[name].segments, image.mappings(mappings), image.backing), image.backing)
            for name, image in self._files.items() if image.backing in backings)
        loaded_ids = {name for name, _, _ in loaded}
        if self._engine_id not in loaded_ids:
            raise native.NativeImageError("reviewed MuJoCo engine is not loaded")
        current = {name for name in self._module_names if name in sys.modules}
        if current != self._python.modules.keys() - {"mujoco"}:
            raise native.NativeImageError("Python native module population changed")
        for name in current:
            module = sys.modules[name]
            if type(module) is not ModuleType or self._module_names[name] not in loaded_ids:
                raise native.NativeImageError("Python extension has no loaded native image identity")
            filename = _namespace(module).get("__file__")
            if type(filename) is not str or Path(filename).resolve() != self._files[self._module_names[name]].path.resolve():
                raise native.NativeImageError("Python extension source identity differs")
        return loaded

    def _validate_query_identities(self):
        public = _namespace(self._python.modules["mujoco"])
        structs = _namespace(self._python.modules["mujoco._structs"])
        for name in ("MjModel", "MjData"):
            cls = structs.get(name)
            if (not _is_class(cls) or _type_name(cast(type, cls)) != "mujoco._structs." + name
                    or _type_name(type(cls)) != "pybind11_builtins.pybind11_type"
                    or public.get(name) is not cls):
                raise native.NativeImageError("MuJoCo query native class identity differs: " + name)
        descriptor = _namespace(structs["MjData"]).get("__copy__")
        if _type_name(type(descriptor)) != "builtins.instancemethod":
            raise native.NativeImageError("MuJoCo query copy descriptor identity differs")
        query = self.function("mj_geomDistance")
        archived = self._profile.get("archived_query_docstring")
        if archived is not None and query.__doc__ != archived["value"]:
            raise native.NativeImageError("MuJoCo query signature identity differs")
        functions = _namespace(self._python.modules["mujoco._functions"])
        if "mj_copyData" in functions:
            self.function("mj_copyData")

    def function(self, name):
        """Require a native function and its matching public export."""
        public = _namespace(self._python.modules["mujoco"])
        functions = _namespace(self._python.modules["mujoco._functions"])
        value = functions.get(name)
        if (type(value) is not BuiltinFunctionType or public.get(name) is not value
                or value.__name__ != name or value.__module__ != "mujoco._functions"):
            raise native.NativeImageError("MuJoCo query callable identity differs: " + name)
        return value

    def check(self) -> MujocoNativeReceipt:
        """Check each declared loaded slot before allowing a native read.

        Inside :meth:`observation` this returns the receipt verified at the
        observation's start instead of repeating the full check.
        """
        if self._closed:
            raise native.NativeImageError("MuJoCo native image guard is closed")
        observation = self._observation
        if observation is not None and observation[0] == threading.get_ident():
            # Only the thread that opened the observation holds its receipt.
            return observation[1]
        return self._verify()

    @contextmanager
    def observation(self):
        """Verify the loaded image at the two observation boundaries only.

        Maintainer decision 2026-10-07 (issue #136) reduced the per-observation
        checks from 25 to these two. The guarantee is that the reviewed image
        was loaded when the observation began and when it ended. A change made
        after the start and reverted before the end is no longer detected; the
        checks that ran between native stages could detect it. The end check
        runs only when the interval exits normally; a raised interval is
        already rejected.

        The receipt is held for the opening thread only; another thread's
        ``check`` verifies in full. Observations do not nest: a second
        ``observation`` while one is open is a programming error, since it
        would reuse a receipt it did not earn.
        """
        if self._closed:
            raise native.NativeImageError("MuJoCo native image guard is closed")
        if self._observation is not None:
            raise RuntimeError("a native observation is already open on this guard")
        receipt = self._verify()
        self._observation = (threading.get_ident(), receipt)
        try:
            yield receipt
        finally:
            self._observation = None
        self._verify()

    def _verify(self) -> MujocoNativeReceipt:
        try:
            self._python.check()
        except SurfaceValidationError as error:
            raise native.NativeImageError("Python identity is unresolved: " + str(error)) from error
        self._check_distribution()
        self._init_file.validate()
        for image in self._files.values():
            image.validate()
        mappings = native.read_native_mappings()
        images = self._loaded_images(mappings)
        if self._receipt is not None and images != self._receipt.images:
            raise native.NativeImageError("loaded MuJoCo native image identity changed")
        biases = {name: bias for name, bias, _ in images}
        engine_bias = biases[self._engine_id]
        engine_file = self._files[self._engine_id]
        imports = []
        with native.NativeMemoryReader(mappings) as reader:
            for name, bias, backing in images:
                importer = self._projected_engine if name == self._engine_id else self._layouts[name]
                target = importer if name == self._engine_id else self._engine
                plan = self._binding_plans.get(name)
                if plan is None:
                    # Planned on first load only: an image that never loads
                    # is never planned, as before.
                    policies = tuple(row for row in self._policies
                                     if row.importer_image_sha256 == importer.image_sha256)
                    plan = self._binding_plans[name] = native.plan_native_bindings(
                        importer, target, coalescence=policies)
                witnesses = native.check_native_bindings(
                    importer, bias, backing, target, engine_bias, engine_file.backing,
                    mappings, reader.read, page_size=self._page_size, plan=plan)
                imports.extend((name, witness) for witness in witnesses)
            table = native.check_native_pointer_table(
                self._engine, engine_bias, engine_file.backing, mappings, "mjCOLLISIONFUNC",
                self._initializers, self._table_bytes, reader.read, page_size=self._page_size,
                plan=self._table_plan)
        receipt = MujocoNativeReceipt(self._engine_id, images, tuple(imports), table)
        if self._receipt is not None and receipt != self._receipt:
            raise native.NativeImageError("MuJoCo native target identity changed")
        return receipt

    def _symbol_address(self, name, kind):
        self.check()
        possible = tuple(symbol for symbol in self._engine.symbols
                         if symbol.name == name and symbol.section != "SHN_UNDEF")
        if len(possible) != 1:
            raise native.NativeImageError("native symbol has no unique definition: " + name)
        symbol, = possible
        native._ordinary_symbol(symbol, definition=True)
        if symbol.kind != kind:
            raise native.NativeImageError("native symbol kind differs: " + name)
        image = self._files[self._engine_id]
        mappings = native.read_native_mappings()
        bias = native.resolve_load_bias(self._engine.segments, image.mappings(mappings), image.backing)
        if self._receipt is None:
            raise native.NativeImageError("native symbol has no enrolled image identity")
        expected_bias = next(value for key, value, _ in self._receipt.images if key == self._engine_id)
        if bias != expected_bias:
            raise native.NativeImageError("native symbol loaded image changed")
        address = bias + symbol.value
        native.require_elf_range(self._engine.segments, bias, mappings, image.backing,
                                 address, symbol.size, "x" if kind == "STT_FUNC" else "r",
                                 page_size=self._page_size)
        return address, symbol, mappings

    def read_global_pointer(self, name):
        """Read an authenticated pointer object with its complete ELF extent."""
        address, symbol, mappings = self._symbol_address(name, "STT_OBJECT")
        if symbol.size != self._engine.pointer_size:
            raise native.NativeImageError("native global is not one declared pointer: " + name)
        with native.NativeMemoryReader(mappings) as reader:
            value = int.from_bytes(reader.read(address, symbol.size), self._engine.byteorder)
        self.check()
        return value

    def _function_address(self, name):
        """Resolve a source-declared private ABI after authenticating its image."""
        address, _, _ = self._symbol_address(name, "STT_FUNC")
        return address

    def _read_declared_local_objects(self, profile):
        """Read only the separately pinned ordinary logging objects.

        TLS offsets are not image addresses. This local-object contract does
        not change the dynamic symbol policy used for native function calls.
        """
        self.check()
        if profile["engine_sha256"] != self._engine.image_sha256:
            raise native.NativeImageError("native local-object source image differs")
        image = self._files[self._engine_id]
        mappings = native.read_native_mappings()
        bias = native.resolve_load_bias(self._engine.segments, image.mappings(mappings), image.backing)
        if self._receipt is None or bias != next(value for key, value, _ in self._receipt.images if key == self._engine_id):
            raise native.NativeImageError("native local-object loaded image changed")
        result = {}
        with native.NativeMemoryReader(mappings) as reader:
            for key, declaration in profile["objects"].items():
                matches = tuple(symbol for symbol in self._engine.local_symbols if symbol.name == declaration["symbol"])
                if len(matches) != 1:
                    raise native.NativeImageError("native local-object definition is ambiguous")
                symbol, = matches
                if (symbol.kind != "STT_OBJECT" or symbol.binding != "STB_LOCAL"
                        or symbol.visibility != "STV_HIDDEN" or symbol.section != int(declaration["section"])
                        or symbol.value != declaration["value"] or symbol.size != declaration["size"]):
                    raise native.NativeImageError("native local-object declaration differs")
                address = bias + symbol.value
                native.require_elf_range(self._engine.segments, bias, mappings, image.backing,
                                         address, symbol.size, "r", page_size=self._page_size)
                result[key] = reader.read(address, symbol.size)
        self.check()
        return result

    def close(self):
        """Release retained read-only files after the subject has retired."""
        self._closed = True
        self._held.close()

    def __enter__(self):
        if self._closed:
            raise native.NativeImageError("MuJoCo native image guard is closed")
        return self

    def __exit__(self, *exception):
        self.close()
