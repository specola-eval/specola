"""Permit timed scratch stages only with a null or initialized native timer.

This guard does not initialize the producer's default timer. Its successful
system-clock boundary is explicit in the pinned source policy. The caller must
remain outside concurrent native calls throughout the interval.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path

from ..conformance import native_images as native
from ..conformance.query_bindings import QueryBindingError


class MujocoTimerGuard:
    """Bind the actual callback, initialized epoch and clock import target."""

    def __init__(self, guard):
        self._guard = guard
        self._closed = False
        try:
            path = Path(__file__).parent.parent / "data" / ("mujoco-completion-" + guard._version + ".json")
            self.policy = json.loads(path.read_text())
            self._structs_id = guard._module_names["mujoco._structs"]
            self._structs = guard._files[self._structs_id]
            profiles = [row for row in self.policy["images"]
                        if row["relative_path"] == self._structs_id
                        and row["sha256"] == guard._images[self._structs_id]["sha256"]]
            if len(profiles) != 1:
                raise QueryBindingError("native timer source image is absent or ambiguous")
            self._profile = profiles[0]
            self._layout = guard._layouts[self._structs_id]
            self._page_size = os.sysconf("SC_PAGE_SIZE")
        except BaseException:
            self.close()
            raise

    def _read(self, reader, mappings, image, layout, bias, offset, size, permission):
        address = bias + offset
        native.require_elf_range(layout.segments, bias, mappings, image.backing,
                                 address, size, permission, page_size=self._page_size)
        return reader.read(address, size)

    def check(self):
        """Return the current timer identity without executing a callback."""
        if self._closed:
            raise QueryBindingError("native timer guard is closed")
        receipt = self._guard.check()
        callback = self._guard.read_global_pointer("mjcb_time")
        if callback is None or callback == 0:
            return (0, ())
        if "timer" not in self._profile:
            raise QueryBindingError("unresolved nonnull native timer for the pinned producer")
        biases = {name: bias for name, bias, _ in receipt.images}
        bias = biases[self._structs_id]
        if callback != bias + self._profile["timer"]["start"]:
            raise QueryBindingError("unresolved custom native timer callback")
        mappings = native.read_native_mappings()
        with native.NativeMemoryReader(mappings) as reader:
            for name in ("timer", "clock"):
                record = self._profile[name]
                body = self._read(reader, mappings, self._structs, self._layout, bias,
                                  record["start"], record["size"], "x")
                if body != bytes.fromhex(record["hex"]):
                    raise QueryBindingError("native default timer code changed")
            for record in self._profile["additional_ranges"]:
                value = self._read(reader, mappings, self._structs, self._layout, bias,
                    record["start"], record["size"], record["permissions"])
                if value != bytes.fromhex(record["hex"]):
                    raise QueryBindingError("native default timer route or literal changed")
            data = []
            for record in self._profile["data"]:
                data.append((record["name"], self._read(reader, mappings, self._structs,
                    self._layout, bias, record["address"], record["size"], "r")))
            if dict(data)["initialization_guard"][0] != 1:
                raise QueryBindingError("unresolved uninitialized native default timer")
            slots = self._profile["imports"]["clock_gettime"]
            if not slots:
                raise QueryBindingError("native default timer has no declared clock import")
            for slot in slots:
                declarations = [r for r in self._layout.relocations if r.offset == slot["offset"]]
                expected_kind = {6: "R_X86_64_GLOB_DAT", 7: "R_X86_64_JUMP_SLOT"}.get(slot["relocation_type"])
                if (len(declarations) != 1 or declarations[0].symbol_index != slot["symbol_index"]
                        or declarations[0].kind != expected_kind or declarations[0].addend != slot["addend"]
                        or self._layout.symbols[slot["symbol_index"]].name != "clock_gettime"):
                    raise QueryBindingError("native default timer clock relocation differs")
        self._guard.check()
        return (callback, tuple(data))

    @contextmanager
    def preserve(self):
        """Require stable native callback and epoch across timed completion."""
        before = self.check()
        try:
            yield
        finally:
            if self.check() != before:
                raise QueryBindingError("native timer or initialization state changed during observation")

    def close(self):
        """Retire the guard without calling the native timer."""
        if not self._closed:
            self._closed = True
