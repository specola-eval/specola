"""Preserve declared MuJoCo CCD storage in already initialized thread TLS.

The public lookup uses a native callback. Python runs outside the loader's
locked callback interval. The caller owns a synchronized producer lifetime;
equal boundary checks do not prevent another thread from unloading an image.
"""

from contextlib import ExitStack, contextmanager
import ctypes
from importlib import import_module
import json
import os
import sys
from pathlib import Path
import threading
from types import BuiltinFunctionType, ModuleType, SimpleNamespace

from specola.conformance import native_images as native
from specola.conformance.query_bindings import QueryBindingError


_ACTIVE = threading.local()


class MujocoExistingTls:
    """Own the source-declared CCD TLS ranges for one guarded producer."""

    def __init__(self, guard):
        self._held = ExitStack()
        self._closed = False
        self._failed = False
        self._lock = threading.Lock()
        self._guard = guard
        self._identity = None
        try:
            self._initialize()
        except BaseException:
            self._held.close()
            self._closed = True
            raise

    def _initialize(self):
        if self._guard._version != "3.3.1":
            raise QueryBindingError("unresolved existing-TLS producer version")
        path = Path(__file__).parent.parent / "data" / "mujoco-tls-3.3.1.json"
        if not path.is_file():
            raise QueryBindingError("unresolved existing-TLS source and bridge evidence")
        self.policy = json.loads(path.read_text())
        if self.policy.get("schema") != "specola.mujoco.existing-tls.v1":
            raise QueryBindingError("unresolved existing-TLS policy schema")
        self._source = self.policy["lookup"]
        self._page_size = os.sysconf("SC_PAGE_SIZE")
        self._files = {
            name: self._held.enter_context(native.PinnedNativeFile(
                row["file"]["path"], row["file"]["sha256"]))
            for name, row in self._source["system_images"].items()
        }
        self._layouts = {name: native.read_elf_layout(file)
                         for name, file in self._files.items()}
        self._engine_file = self._guard._files[self._guard._engine_id]
        self._engine = self._guard._engine
        self._validate_declarations()
        self._compiled = self.policy["bridge"]["compiled"]
        self._callable_api = self.policy["bridge"]["callable_address"]
        self._flags_api = self.policy["bridge"]["callable_flags"]
        self._bridge_images = {
            name: self._held.enter_context(native.PinnedNativeFile(
                row["identity"]["path"], row["identity"]["sha256"]))
            for name, row in self._compiled["provider_images"].items()
        }
        self._bridge_layouts = {name: native.read_elf_layout(file)
                               for name, file in self._bridge_images.items()}
        if (self._callable_api["image"]["sha256"] != self._bridge_images["cpython"].sha256
                or self._flags_api["image"]["sha256"] != self._bridge_images["cpython"].sha256):
            raise QueryBindingError("callable-address API belongs to another interpreter")
        self._bridge = import_module("specola._native_tls")
        if type(self._bridge) is not ModuleType:
            raise QueryBindingError("existing-TLS bridge is not a native module")
        self._bridge_file = self._held.enter_context(native.PinnedNativeFile(
            self._bridge.__file__, self.policy["bridge"]["sha256"]))
        self._bridge_layout = native.read_elf_layout(self._bridge_file)
        if not self._bridge_layout.bind_now:
            raise QueryBindingError("existing-TLS bridge lacks eager native bindings")
        self._bridge_image_identity = None
        for name in ("layout", "existing_tls"):
            function = vars(self._bridge).get(name)
            if (type(function) is not BuiltinFunctionType
                    or function.__module__ != "specola._native_tls"
                    or function.__name__ != name or function.__self__ is not self._bridge):
                raise QueryBindingError("existing-TLS bridge callable identity differs")
        self._bridge_identity = tuple(sorted((key, id(value))
                                             for key, value in vars(self._bridge).items()))
        self._check_bridge()
        abi = self._source["callback_abi"]
        expected = (abi["pointer_bytes"], abi["pointer_bytes"], abi["info_size"],
                    *((field["offset"], field["size"]) for field in abi["fields"]))
        if self._bridge.layout() != expected:
            raise QueryBindingError("compiled existing-TLS callback ABI differs")
        self.check()

    def _validate_symbol(self, image, declaration):
        if declaration["symbol_table"] != "dynamic":
            # The source profile joins a matching debug symbol to exact file
            # bytes. Its index belongs to the debug .symtab, not to .dynsym.
            if declaration["symbol_table"] != "matching_debug_symtab":
                raise QueryBindingError("unresolved TLS lookup symbol table")
            return
        entry = declaration["entry"]
        expected = native.ElfSymbol(
            declaration["index"], declaration["name"], entry["st_info"]["type"],
            entry["st_info"]["bind"], entry["st_other"]["visibility"],
            entry["st_shndx"], entry["st_value"], entry["st_size"],
            declaration["version"]["raw_index"])
        if self._layouts[image].symbols[declaration["index"]] != expected:
            raise QueryBindingError("TLS lookup symbol or GNU version differs")

    def _validate_declarations(self):
        ELFFile = import_module("elftools.elf.elffile").ELFFile

        producer = self._source["producer"]
        if self._engine.image_sha256 != producer["file"]["sha256"]:
            raise QueryBindingError("existing-TLS producer image differs")
        match = producer["callback_image_match"]
        self._engine_file.validate()
        with os.fdopen(os.dup(self._engine_file.fileno()), "rb") as stream:
            elf = ELFFile(stream)
            if ({key: elf.header[key] for key in match["ELF_header"]}
                    != match["ELF_header"]
                    or [dict(segment.header) for segment in elf.iter_segments()
                        if segment["p_type"] == "PT_TLS"] != match["PT_TLS"]):
                raise QueryBindingError("native program-header or TLS declaration differs")
        self._engine_file.validate()
        header = native.read_elf_file_bytes(self._engine_file, self._engine,
            match["program_header_vaddr"], match["program_header_extent"])
        if header != bytes.fromhex(match["program_header_bytes_hex"]):
            raise QueryBindingError("native program-header bytes differ")
        seen = set()
        intervals = []
        for row in producer["symbols"]:
            candidates = [symbol for symbol in self._engine.local_symbols
                          if symbol.name == row["name"]]
            if len(candidates) != 1 or row["name"] in seen:
                raise QueryBindingError("native CCD TLS symbol is absent or ambiguous")
            symbol, = candidates
            start, size = row["elf_tls_offset"], row["size"]
            if (symbol.kind != "STT_TLS" or symbol.binding != "STB_LOCAL"
                    or symbol.value != start or symbol.size != size or start < 0
                    or size <= 0 or start + size > producer["PT_TLS"]["memsz"]
                    or any(start < end and old < start + size for old, end in intervals)):
                raise QueryBindingError("native CCD TLS extent or ownership differs")
            seen.add(row["name"])
            intervals.append((start, start + size))
        if not intervals:
            raise QueryBindingError("native CCD TLS source population is empty")
        for row in self._source["functions"]:
            image = row["image"]
            self._validate_symbol(image, row["symbol"])
            if (row["body_relocations"] or row["symbol"]["entry"]["st_value"] != row["vaddr"]
                    or row["symbol"]["entry"]["st_size"] != row["size"]
                    or native.read_elf_file_bytes(self._files[image], self._layouts[image],
                        row["vaddr"], row["size"]) != bytes.fromhex(row["hex"])):
                raise QueryBindingError("native TLS lookup code declaration differs")
        for row in self._source["binding_slots"]:
            image, entry = row["importer"], row["relocation"]["entry"]
            candidates = [slot for slot in self._layouts[image].relocations
                          if slot.offset == entry["r_offset"]]
            if (len(candidates) != 1 or candidates[0].kind != row["kind"]
                    or candidates[0].symbol_index != entry["r_info_sym"]
                    or candidates[0].addend != entry["r_addend"]):
                raise QueryBindingError("native TLS lookup relocation differs")
            if "imported_symbol" in row:
                self._validate_symbol(image, row["imported_symbol"])
                self._validate_symbol(row["expected_image"], row["expected_symbol"])
            else:
                declaration = row["expected_debug_symbol"]
                if (row["kind"] != "R_X86_64_RELATIVE"
                        or declaration["entry"]["st_value"] != entry["r_addend"]):
                    raise QueryBindingError("native soft TLS initializer differs")

    def _read(self, reader, mappings, image, layout, bias, offset, size, mode):
        address = bias + offset
        native.require_elf_range(layout.segments, bias, mappings, image.backing,
                                 address, size, mode, page_size=self._page_size)
        return reader.read(address, size)

    def _check_bridge(self):
        if (self._bridge is not sys.modules.get("specola._native_tls")
                or self._bridge_identity != tuple(sorted(
                    (key, id(value)) for key, value in vars(self._bridge).items()))):
            raise QueryBindingError("existing-TLS bridge namespace changed")
        self._bridge_file.validate()
        mappings = native.read_native_mappings()
        bias = native.resolve_load_bias(self._bridge_layout.segments,
            self._bridge_file.mappings(mappings), self._bridge_file.backing)
        for file in self._bridge_images.values():
            file.validate()
        biases = {name: native.resolve_load_bias(self._bridge_layouts[name].segments,
            file.mappings(mappings), file.backing) for name, file in self._bridge_images.items()}
        identity = (bias, tuple(sorted(biases.items())))
        if self._bridge_image_identity is not None and self._bridge_image_identity != identity:
            raise QueryBindingError("existing-TLS bridge provider image changed")
        with native.NativeMemoryReader(mappings) as reader:
            self._check_bridge_storage(reader, mappings, bias, biases)
            self._check_callable_addresses(reader, mappings, bias, biases["cpython"])
        self._bridge_image_identity = identity
        return bias

    def _check_bridge_storage(self, reader, mappings, bias, biases):
        def read(offset, size, permission="r"):
            return self._read(reader, mappings, self._bridge_file,
                              self._bridge_layout, bias, offset, size, permission)

        regions = [(row["code"], "x") for row in self._compiled["functions"]]
        regions += [(self._compiled["PLT_selected_import_stubs"], "x"),
                    (self._compiled["literal_region"], "r")]
        for row, permission in regions:
            if read(row["rva"], row["size"], permission) != bytes.fromhex(row["hex"]):
                raise QueryBindingError("existing-TLS bridge code or literal bytes changed")
        width = self._bridge_layout.pointer_size
        for row in self._compiled["relocations"]:
            disposition = row["disposition"]
            if disposition == "CRT_initialization_or_teardown_outside_selected_call_interval":
                continue
            if disposition == "self_relative":
                expected = bias + row["addend"]
            elif disposition in ("exact_cpython_export", "terminal_stack_corruption_only"):
                name, declaration = row["expected_image"], row["expected_symbol"]
                layout, file = self._bridge_layouts[name], self._bridge_images[name]
                symbol = layout.symbols[declaration["index"]]
                if (symbol.name, symbol.value, symbol.size, symbol.kind, symbol.version_index) != (
                        declaration["name"], declaration["value"], declaration["size"],
                        declaration["type"], declaration["version_index_raw"]):
                    raise QueryBindingError("existing-TLS bridge provider symbol differs")
                expected = biases[name] + symbol.value
                native.require_elf_range(layout.segments, biases[name], mappings, file.backing,
                    expected, symbol.size, "x" if symbol.kind == "STT_FUNC" else "r",
                    page_size=self._page_size)
                covered = native.require_elf_range(self._bridge_layout.segments, bias, mappings,
                    self._bridge_file.backing, bias + row["slot_rva"], width, "r",
                    page_size=self._page_size)
                if (any("w" in part.permissions for part in covered)
                        or not any(start <= row["slot_rva"] and row["slot_rva"] + width <= start + size
                                   for start, size in self._bridge_layout.relro)):
                    raise QueryBindingError("existing-TLS bridge import lacks current RELRO protection")
            else:
                raise QueryBindingError("unresolved existing-TLS bridge relocation disposition")
            if int.from_bytes(read(row["slot_rva"], width), "little") != expected:
                raise QueryBindingError("existing-TLS bridge imported or relative pointer changed")
        for row in self._compiled["own_objects"]:
            storage = row["storage"]
            expected = bytearray.fromhex(storage["hex"])
            for relocation in row["relocations"]:
                offset = relocation["slot_rva"] - storage["rva"]
                expected[offset:offset + width] = (bias + relocation["addend"]).to_bytes(width, "little")
            start = 0
            if row["symbol"]["name"] == "specola_tls_module":
                fields = self._compiled["compiled_layouts"]["PyModuleDef"]["fields"]
                base, = [field for field in fields if field["name"] == "m_base"]
                start = base["offset"] + base["size"]
            actual = read(storage["rva"] + start, storage["size"] - start)
            if actual != bytes(expected[start:]):
                raise QueryBindingError("existing-TLS bridge method or module declaration changed")

    def _check_callable_addresses(self, reader, mappings, bridge_bias, python_bias):
        image, layout = self._bridge_images["cpython"], self._bridge_layouts["cpython"]
        rows = [self._callable_api[key] for key in ("hot_function", "cold_function")]
        rows.append(self._flags_api["function"])
        for row in rows:
            if self._read(reader, mappings, image, layout, python_bias,
                row["value"], row["size"], "x") != bytes.fromhex(row["hex"]):
                raise QueryBindingError("native callable-address API code changed")
        exact_type = self._callable_api["exact_type_declaration"]
        if id(BuiltinFunctionType) != python_bias + exact_type["value"]:
            raise QueryBindingError("native builtin type belongs to another interpreter")
        getter = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)(
            python_bias + self._callable_api["hot_function"]["value"])
        flags_getter = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object)(
            python_bias + self._flags_api["function"]["value"])
        methods, = [row for row in self._compiled["own_objects"]
                    if row["symbol"]["name"] == "specola_tls_methods"]
        layout = self._compiled["compiled_layouts"]["PyMethodDef"]
        fields = {row["name"]: row for row in layout["fields"]}
        if ctypes.sizeof(ctypes.c_int) != fields["ml_flags"]["size"]:
            raise QueryBindingError("native builtin flags result width differs")
        table_bytes = bytes.fromhex(methods["storage"]["hex"])

        def field_at(offset, name):
            field = fields[name]
            start = offset + field["offset"]
            return int.from_bytes(table_bytes[start:start + field["size"]], "little",
                                  signed=field["encoding"] == 5)
        # The source-verified public API returns a pointer without invoking it.
        # Exact live builtin inputs keep its no-call branch; the surrounding
        # ctypes carriers allocate outside the native loader callback.
        for public, symbol_name in (("existing_tls", "specola_existing_tls"),
                                    ("layout", "specola_tls_layout")):
            function = vars(self._bridge)[public]
            if type(function) is not BuiltinFunctionType:
                raise QueryBindingError("existing-TLS bridge callable type changed")
            targets = [row["symbol"]["value"] for row in self._compiled["functions"]
                       if row["symbol"]["name"] == symbol_name]
            if len(targets) != 1 or getter(function) != bridge_bias + targets[0]:
                raise QueryBindingError("existing-TLS bridge callable implementation differs")
            flags = [field_at(offset, "ml_flags") for offset in range(0, len(table_bytes), layout["size"])
                     if field_at(offset, "ml_meth") == targets[0]]
            if len(flags) != 1 or flags_getter(function) != flags[0]:
                raise QueryBindingError("existing-TLS bridge callable argument convention differs")

    def check(self):
        """Authenticate the callable route without invoking TLS initialization."""
        if self._closed or self._failed:
            raise QueryBindingError("existing-TLS guard is closed or restoration is unresolved")
        receipt = self._guard.check()
        bridge_bias = self._check_bridge()
        for file in self._files.values():
            file.validate()
        mappings = native.read_native_mappings()
        biases = {name: native.resolve_load_bias(self._layouts[name].segments,
            file.mappings(mappings), file.backing) for name, file in self._files.items()}
        engine_bias = dict((name, bias) for name, bias, _ in receipt.images)[self._guard._engine_id]
        identity = (engine_bias, bridge_bias, tuple(sorted(biases.items())))
        if self._identity is not None and self._identity != identity:
            raise QueryBindingError("native TLS lookup loaded image changed")
        with native.NativeMemoryReader(mappings) as reader:
            for row in self._source["functions"]:
                image = row["image"]
                if self._read(reader, mappings, self._files[image], self._layouts[image],
                    biases[image], row["vaddr"], row["size"], "x") != bytes.fromhex(row["hex"]):
                    raise QueryBindingError("native TLS lookup executable bytes changed")
            for row in self._source["binding_slots"]:
                image, entry = row["importer"], row["relocation"]["entry"]
                expected = (row["expected_symbol"]["entry"]["st_value"]
                            if "expected_symbol" in row else entry["r_addend"])
                value = int.from_bytes(self._read(reader, mappings, self._files[image],
                    self._layouts[image], biases[image], entry["r_offset"],
                    self._layouts[image].pointer_size, "r"), "little")
                if value != biases[row["expected_image"]] + expected:
                    raise QueryBindingError("native TLS lookup binding is unresolved or substituted")
            mutex = self._source["loader_mutex"]
            widths = self.policy["mutex_abi"]
            image = mutex["image"]
            data = self._read(reader, mappings, self._files[image], self._layouts[image],
                biases[image], mutex["native_object_vaddr"] + mutex["member_offset"],
                mutex["size"], "r")
            fields = {name: int.from_bytes(data[mutex[name + "_offset"]:
                mutex[name + "_offset"] + widths[name]["bytes"]], "little",
                signed=widths[name]["signed"]) for name in ("kind", "count")}
            if fields["kind"] != mutex["expected_kind"] or fields["count"] == mutex["count_overflow"]:
                raise QueryBindingError("native loader mutex kind or recursion state is unresolved")
            match = self._source["producer"]["callback_image_match"]
            if self._read(reader, mappings, self._engine_file, self._engine, engine_bias,
                match["program_header_vaddr"], match["program_header_extent"], "r"
            ) != bytes.fromhex(match["program_header_bytes_hex"]):
                raise QueryBindingError("loaded producer program headers changed")
        self._identity = identity
        functions = [row for row in self._source["functions"]
                     if row["symbol"]["name"] == "dl_iterate_phdr"]
        if len(functions) != 1:
            raise QueryBindingError("public TLS iterator declaration is absent or ambiguous")
        function, = functions
        return (biases[function["image"]] + function["vaddr"], engine_bias,
                engine_bias + match["program_header_vaddr"], match["ELF_header"]["e_phnum"])

    def _lookup(self):
        arguments = self.check()
        try:
            value = self._bridge.existing_tls(*arguments)
        except (RuntimeError, ValueError, TypeError) as error:
            raise QueryBindingError("existing native TLS lookup failed: " + str(error)) from error
        if (type(value) is not tuple or len(value) != 2
                or any(type(item) is not int or item <= 0 for item in value)):
            raise QueryBindingError("producer TLS is absent; observation cannot initialize it")
        if self.check() != arguments:
            raise QueryBindingError("native TLS lookup subject changed during callback")
        module_id, base = value
        allocation = self._source["producer"]["PT_TLS"]
        if base % allocation["alignment"]:
            raise QueryBindingError("existing producer TLS allocation alignment differs")
        mappings = native.read_native_mappings()
        native.require_mapped_range(mappings, base, allocation["memsz"], "rw")
        return module_id, base, mappings

    @contextmanager
    def preserve(self):
        """Restore each declared range on the owning thread before returning."""
        if self._closed or self._failed:
            raise QueryBindingError("existing-TLS guard is closed or restoration is unresolved")
        state = getattr(_ACTIVE, "state", None)
        if state is None:
            state = _ACTIVE.state = SimpleNamespace(active=False, failed=False)
        if state.failed:
            raise QueryBindingError("observing thread TLS restoration is unresolved")
        if state.active or not self._lock.acquire(blocking=False):
            raise QueryBindingError("reentrant or concurrent existing-TLS preservation")
        state.active = True
        try:
            owner = threading.get_ident()
            module_id, base, mappings = self._lookup()
            with native.NativeMemoryReader(mappings) as reader:
                saved = tuple((row["elf_tls_offset"], reader.read(
                    base + row["elf_tls_offset"], row["size"]))
                    for row in self._source["producer"]["symbols"])
            try:
                yield
            finally:
                try:
                    if threading.get_ident() != owner:
                        raise QueryBindingError("native TLS restoration moved to another thread")
                    after_module, after_base, after_maps = self._lookup()
                    if (after_module, after_base) != (module_id, base):
                        raise QueryBindingError("existing producer TLS allocation changed")
                    for offset, data in saved:
                        native.require_mapped_range(after_maps, base + offset, len(data), "rw")
                        ctypes.memmove(base + offset, data, len(data))
                    with native.NativeMemoryReader(after_maps) as reader:
                        if any(reader.read(base + offset, len(data)) != data for offset, data in saved):
                            raise QueryBindingError("native CCD TLS byte restoration failed")
                    self.check()
                except BaseException:
                    self._failed = True
                    state.failed = True
                    raise
        finally:
            # Retain the entering thread's record across yield. A foreign
            # exit must not overwrite the exit thread's own TLS scope state.
            state.active = False
            self._lock.release()

    def close(self):
        """Release files without calling any producer TLS accessor."""
        if self._closed:
            return
        if not self._lock.acquire(blocking=False):
            raise QueryBindingError("cannot close active existing-TLS preservation")
        try:
            self._held.close()
            self._closed = True
        finally:
            self._lock.release()
