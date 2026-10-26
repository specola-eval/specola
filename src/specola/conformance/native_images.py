"""Join reviewed native files to Linux mappings before reading their images.

File hashes describe retained descriptors, not resident private pages. These
checks detect recorded identity and mapping drift within a caller-synchronized
loader interval. They do not synchronize a concurrent unload or authenticate
an opaque Python binding's call target. ELF relocation checks supply that
separate join; opening a fresh library handle does not.

The small maps parser retains device/inode fields omitted by psutil 7.2.2.
Its format follows Linux v6.8 fs/proc/task_mmu.c and filesystems/proc.rst;
the source reconciliation is recorded in the extraction evidence record.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from functools import cached_property
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Literal, Optional, Tuple

from .core_runner import EvidenceIdentity
from .extraction import ExtractionValidationError, _provenance


class NativeImageError(ValueError):
    """Native file, mapping or execution-target identity is unresolved."""


@dataclass(frozen=True)
class NativeMapping:
    """One VMA; pathname bytes are diagnostic and are never an identity key."""

    start: int
    end: int
    permissions: str
    offset: int
    device_major: int
    device_minor: int
    inode: int
    pathname: bytes

    @cached_property
    def backing(self) -> Optional[Tuple[int, int, int]]:
        # Derived from frozen fields; cached outside the compared fields.
        return (self.device_major, self.device_minor, self.inode) if self.inode else None


class _MappingCensus(tuple):
    """Records parsed from one maps read: ordered and non-overlapping."""

    ends: Tuple[int, ...]

    def __new__(cls, records):
        census = super().__new__(cls, records)
        # Ascending, since the parser rejects unordered or overlapping ranges.
        census.ends = tuple(record.end for record in census)
        return census


def _parse_mapping(line: bytes) -> NativeMapping:
    fields = line.split(None, 5)
    if len(fields) not in (5, 6):
        raise NativeImageError("malformed native mapping record")
    address, permissions, offset, device, inode = fields[:5]
    if (re.fullmatch(rb"[0-9a-fA-F]+-[0-9a-fA-F]+", address) is None
            or re.fullmatch(rb"[r-][w-][x-][ps]", permissions) is None
            or re.fullmatch(rb"[0-9a-fA-F]+", offset) is None
            or re.fullmatch(rb"[0-9a-fA-F]+:[0-9a-fA-F]+", device) is None
            or re.fullmatch(rb"[0-9]+", inode) is None):
        raise NativeImageError("malformed native mapping fields")
    start, end = (int(value, 16) for value in address.split(b"-"))
    major, minor = (int(value, 16) for value in device.split(b":"))
    if start >= end:
        raise NativeImageError("native mapping ranges are invalid, unordered or overlapping")
    return NativeMapping(start, end, permissions.decode("ascii"), int(offset, 16),
                         major, minor, int(inode), fields[5] if len(fields) == 6 else b"")


# The previous census's records by line. A record is a pure function of its
# line, so a line repeated in the next read reuses its immutable record.
_PARSED_LINES: dict = {}


def parse_native_mappings(content: bytes) -> Tuple[NativeMapping, ...]:
    """Parse ordered Linux maps without normalizing ambiguous path spelling."""
    global _PARSED_LINES
    if type(content) is not bytes or not content:
        raise NativeImageError("native mapping census is absent")
    previous, parsed, records = _PARSED_LINES, {}, []
    lines = content.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    for line in lines:
        record = previous.get(line)
        if record is None:
            record = _parse_mapping(line)
        parsed[line] = record
        if records and record.start < records[-1].end:
            raise NativeImageError("native mapping ranges are invalid, unordered or overlapping")
        records.append(record)
    if not records:
        raise NativeImageError("native mapping census is empty")
    _PARSED_LINES = parsed
    return _MappingCensus(records)


def read_native_mappings() -> Tuple[NativeMapping, ...]:
    """Read procfs without claiming that its buffered reads are atomic."""
    try:
        return parse_native_mappings(Path("/proc/self/maps").read_bytes())
    except OSError as error:
        raise NativeImageError("native mapping census is unavailable: " + str(error)) from error


def require_mapped_range(mappings: Tuple[NativeMapping, ...], address: int, size: int,
                         permissions: str, backing=None) -> Tuple[NativeMapping, ...]:
    """Require the complete byte interval before a caller dereferences it.

    With no backing argument this checks accessibility only. An anonymous BSS
    interval requires a separate unique ELF load-segment ownership join.
    """
    if (type(address) is not int or address < 0 or type(size) is not int or size <= 0
            or type(permissions) is not str or any(mode not in "rwx" for mode in permissions)):
        raise NativeImageError("invalid native memory range or required permissions")
    stop, cursor, covered = address + size, address, []
    # In a parsed census the records ending at or before the address are a
    # prefix; skip it by bisection instead of a scan (same records visited).
    first = bisect_right(mappings.ends, cursor) if type(mappings) is _MappingCensus else 0
    for index in range(first, len(mappings)):
        mapping = mappings[index]
        if mapping.end <= cursor:
            continue
        if mapping.start > cursor:
            break
        if (any(mode not in mapping.permissions for mode in permissions)
                or (backing is not None and mapping.backing != backing)):
            raise NativeImageError("native memory range has incompatible permissions or backing identity")
        covered.append(mapping)
        cursor = min(stop, mapping.end)
        if cursor == stop:
            return tuple(covered)
    raise NativeImageError("native memory range is not completely mapped")


class NativeMemoryReader:
    """Read checked self-process bytes without invoking native pointers.

    The caller owns synchronization and compares mapping/file identity before
    and after the operation. Linux procfs pread errors and short reads fail;
    this class does not fall back to unchecked ctypes dereferences.
    """

    def __init__(self, mappings: Tuple[NativeMapping, ...]):
        self.mappings = mappings
        self._descriptor = None

    def __enter__(self):
        if self._descriptor is not None:
            raise NativeImageError("native memory descriptor is already owned")
        try:
            self._descriptor = os.open("/proc/self/mem", os.O_RDONLY | os.O_CLOEXEC)
        except OSError as error:
            raise NativeImageError("native process memory is unavailable: " + str(error)) from error
        return self

    def read(self, address: int, size: int) -> bytes:
        if self._descriptor is None:
            raise NativeImageError("native memory descriptor is not owned")
        require_mapped_range(self.mappings, address, size, "r")
        try:
            raw = os.pread(self._descriptor, size, address)
        except (OSError, OverflowError) as error:
            raise NativeImageError("native memory read failed: " + str(error)) from error
        if len(raw) != size:
            raise NativeImageError("native memory read returned an incomplete interval")
        return raw

    def __exit__(self, *exception):
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None


def _file_stamp(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_nlink)


@dataclass(frozen=True)
class ElfLoadSegment:
    """Producer image layout from PT_LOAD, retaining file and virtual extents."""

    offset: int
    virtual_address: int
    file_size: int
    memory_size: int
    flags: int
    alignment: int

    def __post_init__(self):
        values = (self.offset, self.virtual_address, self.file_size, self.memory_size,
                  self.flags, self.alignment)
        if (any(type(value) is not int or value < 0 for value in values)
                or self.file_size > self.memory_size or self.flags & ~7
                or (self.alignment > 1 and (self.alignment & (self.alignment - 1)
                    or (self.offset - self.virtual_address) % self.alignment))):
            raise NativeImageError("invalid ELF load segment declaration")


def _file_segment_mappings(segment, bias, mappings, backing):
    if not segment.file_size:
        return ()
    # GNU_RELRO can remove write permission from a writable PT_LOAD segment.
    permissions = ("r" if segment.flags & 4 else "") + ("x" if segment.flags & 1 else "")
    start = bias + segment.virtual_address
    covered = require_mapped_range(mappings, start, segment.file_size, permissions, backing)
    for mapping in covered:
        point = max(start, mapping.start)
        if mapping.offset + point - mapping.start != segment.offset + point - start:
            raise NativeImageError("ELF load segment file offsets disagree with mappings")
    return covered


def resolve_load_bias(segments: Tuple[ElfLoadSegment, ...], mappings: Tuple[NativeMapping, ...],
                      backing, *, anchor: Optional[int] = None) -> int:
    """Join complete file-backed PT_LOAD bytes to a unique mapped instance."""
    if not segments or any(type(segment) is not ElfLoadSegment for segment in segments):
        raise NativeImageError("ELF image has no typed load-segment declarations")
    candidates = set()
    for mapping in mappings:
        if mapping.backing != backing:
            continue
        for segment in segments:
            lower = max(mapping.offset, segment.offset)
            upper = min(mapping.offset + mapping.end - mapping.start, segment.offset + segment.file_size)
            if lower < upper:
                candidates.add(mapping.start - mapping.offset - segment.virtual_address + segment.offset)
    accepted = []
    for bias in sorted(candidates):
        if bias < 0:
            continue
        if anchor is not None and not any(bias + segment.virtual_address <= anchor
                                         < bias + segment.virtual_address + segment.memory_size
                                         for segment in segments):
            continue
        try:
            for segment in segments:
                _file_segment_mappings(segment, bias, mappings, backing)
        except NativeImageError:
            continue
        accepted.append(bias)
    if len(accepted) != 1:
        reason = "ambiguous" if accepted else "no consistent"
        raise NativeImageError(reason + " mapped ELF load instance")
    return accepted[0]


def require_elf_range(segments: Tuple[ElfLoadSegment, ...], bias: int,
                      mappings: Tuple[NativeMapping, ...], backing,
                      address: int, size: int, permissions: str, *, page_size: int):
    """Check an ELF-owned object/function interval, including declared BSS."""
    covered = require_mapped_range(mappings, address, size, permissions)
    if type(page_size) is not int or page_size <= 0 or page_size & (page_size - 1):
        raise NativeImageError("invalid native page-size declaration")
    selected = tuple(segment for segment in segments
                     if bias + segment.virtual_address <= address
                     and address + size <= bias + segment.virtual_address + segment.memory_size)
    if len(selected) != 1:
        raise NativeImageError("native range has no unique ELF segment owner")
    segment, = selected
    if any(not segment.flags & bit for mode, bit in (("r", 4), ("w", 2), ("x", 1))
           if mode in permissions):
        raise NativeImageError("native range permissions differ from ELF declaration")
    start = bias + segment.virtual_address
    file_stop = start + segment.file_size
    anonymous_start = (file_stop + page_size - 1) // page_size * page_size
    for mapping in covered:
        point = max(address, mapping.start)
        stop = min(address + size, mapping.end)
        if point < anonymous_start:
            if (mapping.backing != backing
                    or mapping.offset + point - mapping.start != segment.offset + point - start):
                raise NativeImageError("native range has incompatible ELF file backing")
        if stop > anonymous_start and mapping.backing is not None:
            raise NativeImageError("native BSS range has foreign file backing")
    return covered


@dataclass(frozen=True)
class ElfSymbol:
    """An ELF symbol declaration; this record never resolves a loader name."""

    index: int
    name: str
    kind: str
    binding: str
    visibility: str
    section: object
    value: int
    size: int
    version_index: object = "VER_NDX_GLOBAL"


@dataclass(frozen=True)
class ElfRelocation:
    """One loader write declaration and the tables containing that record."""

    tables: Tuple[str, ...]
    file_offset: int
    offset: int
    kind: str
    symbol_index: int
    addend: Optional[int]


@dataclass(frozen=True)
class NativeElfLayout:
    """Parsed declarations for the reviewed ELF64 little-endian x86-64 ABI."""

    segments: Tuple[ElfLoadSegment, ...]
    symbols: Tuple[ElfSymbol, ...]
    local_symbols: Tuple[ElfSymbol, ...]
    relocations: Tuple[ElfRelocation, ...]
    relro: Tuple[Tuple[int, int], ...]
    bind_now: bool
    needed: Tuple[str, ...]
    pointer_size: int = 8
    byteorder: Literal["little", "big"] = "little"
    image_sha256: str = ""


def read_elf_file_bytes(image: PinnedNativeFile, layout: NativeElfLayout, address: int, size: int) -> bytes:
    """Read declared file bytes without substituting anonymous BSS contents."""
    if type(address) is not int or address < 0 or type(size) is not int or size <= 0:
        raise NativeImageError("invalid native ELF file interval")
    segments = tuple(segment for segment in layout.segments
                     if segment.virtual_address <= address
                     and address + size <= segment.virtual_address + segment.file_size)
    if len(segments) != 1:
        raise NativeImageError("native ELF bytes have no unique file segment")
    segment, = segments
    image.validate()
    raw = os.pread(image.fileno(), size, segment.offset + address - segment.virtual_address)
    image.validate()
    if len(raw) != size:
        raise NativeImageError("native ELF file read returned an incomplete interval")
    return raw


@dataclass(frozen=True)
class NativeBindingExpectation:
    """One native relocation's exact pinned target, independent of its value."""

    relocation: ElfRelocation
    target: ElfSymbol
    importer_alternative: Optional[ElfSymbol] = None


@dataclass(frozen=True)
class NativeObjectCoalescence:
    """A reviewed shared weak object definition in two exact native images.

    This policy does not enumerate relocation slots. The independent ELF join
    still determines every candidate; source custody authorizes two possible
    definitions for this named object and preserves the observed winner.
    """

    symbol_name: str
    importer_image_sha256: str
    target_image_sha256: str
    provenance: Tuple[EvidenceIdentity, ...]


@dataclass(frozen=True)
class NativeSlotWitness:
    """An observed pointer retained with its native declaration address."""

    address: int
    value: int
    symbol: str
    target_role: str = "target"


def _ordinary_symbol(symbol: ElfSymbol, *, definition: bool, weak: bool = False) -> None:
    if (symbol.kind not in ("STT_FUNC", "STT_OBJECT") or symbol.binding != ("STB_WEAK" if weak else "STB_GLOBAL")
            or symbol.visibility != "STV_DEFAULT" or symbol.version_index not in (1, "VER_NDX_GLOBAL")):
        raise NativeImageError("unresolved native symbol binding/type/version: " + symbol.name)
    if definition and (type(symbol.section) is not int or symbol.section <= 0 or symbol.size <= 0):
        raise NativeImageError("native target has no normal defined section and positive extent: " + symbol.name)


def partition_native_relocations(importer: NativeElfLayout, target: NativeElfLayout):
    """Retain the complete declared population outside this target relation.

    The remainder includes both other-image references and relocations without
    named symbols. It is not an exclusion or an assertion of native effects.
    """
    names = {symbol.name for symbol in target.symbols
             if symbol.name and symbol.section != "SHN_UNDEF"}
    selected, other = [], []
    for relocation in importer.relocations:
        if not 0 <= relocation.symbol_index < len(importer.symbols):
            raise NativeImageError("native relocation has a dangling symbol index")
        reference = importer.symbols[relocation.symbol_index]
        (selected if reference.name in names else other).append(relocation)
    return tuple(selected), tuple(other)


def plan_native_bindings(importer: NativeElfLayout, target: NativeElfLayout, *,
                         coalescence: Tuple[NativeObjectCoalescence, ...] = ()):
    """Derive engine-facing slots from declarations, never from reader usage.

    This joins names as candidate references, then requires exact supported
    type, visibility, version and address semantics. It also retains defined
    self-references when the importer is the target engine. Other symbols do
    not acquire custody from this relation.
    """
    if type(coalescence) is not tuple or any(type(row) is not NativeObjectCoalescence for row in coalescence):
        raise NativeImageError("native object coalescence requires typed policy records")
    policies = {}
    for policy in coalescence:
        if (type(policy.symbol_name) is not str or not policy.symbol_name
                or policy.symbol_name in policies):
            raise NativeImageError("missing or duplicate native object coalescence policy")
        for digest in (policy.importer_image_sha256, policy.target_image_sha256):
            if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise NativeImageError("native coalescence policy needs exact image identities")
        if (policy.importer_image_sha256 != importer.image_sha256
                or policy.target_image_sha256 != target.image_sha256):
            raise NativeImageError("native coalescence policy image identity differs")
        try:
            _provenance(policy.provenance, "native object coalescence")
        except ExtractionValidationError as error:
            raise NativeImageError(str(error)) from error
        policies[policy.symbol_name] = policy
    definitions = {}
    for symbol in target.symbols:
        if symbol.name and symbol.section != "SHN_UNDEF":
            definitions.setdefault(symbol.name, []).append(symbol)
    expectations, slots, used_policies = [], set(), set()
    selected, _ = partition_native_relocations(importer, target)
    for relocation in selected:
        reference = importer.symbols[relocation.symbol_index]
        possible = definitions.get(reference.name, ())
        if len(possible) != 1:
            raise NativeImageError("ambiguous native target definition: " + reference.name)
        definition, = possible
        self_weak = (importer is target and reference.index == definition.index
                     and reference.binding == "STB_WEAK")
        alternative = None
        if reference.name in policies:
            if (importer is target or reference.kind != "STT_OBJECT" or definition.kind != "STT_OBJECT"
                    or len([symbol for symbol in importer.symbols
                            if symbol.name == reference.name and symbol.section != "SHN_UNDEF"]) != 1):
                raise NativeImageError("native coalescence is not a unique shared object definition")
            _ordinary_symbol(reference, definition=True, weak=True)
            _ordinary_symbol(definition, definition=True, weak=True)
            alternative = reference
            used_policies.add(reference.name)
        else:
            _ordinary_symbol(reference, definition=self_weak, weak=self_weak)
            _ordinary_symbol(definition, definition=True, weak=self_weak)
        if (reference.kind != definition.kind or relocation.addend != 0
                or relocation.kind not in ("R_X86_64_GLOB_DAT", "R_X86_64_JUMP_SLOT", "R_X86_64_64")):
            raise NativeImageError("unresolved native relocation contract: " + reference.name)
        if relocation.kind == "R_X86_64_JUMP_SLOT" and not importer.bind_now:
            raise NativeImageError("native PLT binding lacks eager-binding declaration")
        if relocation.offset in slots:
            raise NativeImageError("duplicate native binding slot")
        slots.add(relocation.offset)
        expectations.append(NativeBindingExpectation(relocation, definition, alternative))
    if used_policies != policies.keys():
        raise NativeImageError("native coalescence policy has no declared relocation subject")
    ordered = sorted(slots)
    if any(second - first < importer.pointer_size for first, second in zip(ordered, ordered[1:])):
        raise NativeImageError("overlapping native binding slots")
    return tuple(expectations)


def check_native_bindings(importer: NativeElfLayout, importer_bias: int, importer_backing,
                          target: NativeElfLayout, target_bias: int, target_backing,
                          mappings: Tuple[NativeMapping, ...], read_memory, *, page_size: int,
                          coalescence: Tuple[NativeObjectCoalescence, ...] = (), plan=None):
    """Compare each declared target without resolving or invoking a symbol.

    ``plan`` is the caller's retained ``plan_native_bindings`` result for
    these same layouts and coalescence policies; it is derived when absent.
    """
    if plan is None:
        plan = plan_native_bindings(importer, target, coalescence=coalescence)
    witnesses = []
    for expected in plan:
        relocation, symbol = expected.relocation, expected.target
        address = importer_bias + relocation.offset
        covered = require_elf_range(importer.segments, importer_bias, mappings, importer_backing,
                                    address, importer.pointer_size, "r", page_size=page_size)
        if relocation.kind in ("R_X86_64_GLOB_DAT", "R_X86_64_JUMP_SLOT"):
            if (not any(start <= relocation.offset and relocation.offset + importer.pointer_size <= start + size
                        for start, size in importer.relro)
                    or any("w" in mapping.permissions for mapping in covered)):
                raise NativeImageError("native binding slot lacks current RELRO protection")
        target_address = target_bias + symbol.value
        require_elf_range(target.segments, target_bias, mappings, target_backing,
                          target_address, symbol.size, "x" if symbol.kind == "STT_FUNC" else "r",
                          page_size=page_size)
        alternatives = {target_address: "target"}
        if expected.importer_alternative is not None:
            alternative = expected.importer_alternative
            alternative_address = importer_bias + alternative.value
            require_elf_range(importer.segments, importer_bias, mappings, importer_backing,
                              alternative_address, alternative.size, "r", page_size=page_size)
            if alternative_address == target_address:
                raise NativeImageError("coalesced native definitions have an ambiguous address")
            alternatives[alternative_address] = "importer"
        raw = read_memory(address, importer.pointer_size)
        if type(raw) is not bytes or len(raw) != importer.pointer_size:
            raise NativeImageError("native pointer read returned an incomplete byte interval")
        value = int.from_bytes(raw, importer.byteorder)
        if value not in alternatives:
            raise NativeImageError("native binding target differs: " + symbol.name)
        witnesses.append(NativeSlotWitness(address, value, symbol.name, alternatives[value]))
    return tuple(witnesses)


@dataclass(frozen=True)
class NativePointerExpectation:
    """One source-declared table slot and its reconstructed native target."""

    index: int
    offset: int
    target: Optional[ElfSymbol]


def plan_native_pointer_table(layout: NativeElfLayout, table_name: str,
                              initializers: Tuple[Optional[str], ...], file_bytes: bytes):
    """Reconstruct a pointer table from its source initializer and ELF writes.

    Initializer names and population must come from the authenticated producer
    declaration, independently of the adapter and current table contents.
    Source/ELF disagreement or an unreviewed relocation remains unresolved.
    """
    tables = tuple(symbol for symbol in layout.symbols if symbol.name == table_name
                   and symbol.section != "SHN_UNDEF")
    if len(tables) != 1:
        raise NativeImageError("native pointer table has no unique ELF definition")
    table, = tables
    _ordinary_symbol(table, definition=True)
    if (table.kind != "STT_OBJECT" or type(initializers) is not tuple
            or table.size != len(initializers) * layout.pointer_size
            or type(file_bytes) is not bytes or len(file_bytes) != table.size):
        raise NativeImageError("native pointer table source, ELF and file extents disagree")
    slots = {}
    for relocation in layout.relocations:
        if relocation.offset + layout.pointer_size <= table.value or relocation.offset >= table.value + table.size:
            continue
        if (relocation.offset < table.value or relocation.offset + layout.pointer_size > table.value + table.size
                or (relocation.offset - table.value) % layout.pointer_size or relocation.offset in slots):
            raise NativeImageError("overlapping or misaligned native pointer table relocation")
        slots[relocation.offset] = relocation
    result = []
    for index, name in enumerate(initializers):
        offset = table.value + index * layout.pointer_size
        relocation = slots.get(offset)
        raw = file_bytes[index * layout.pointer_size:(index + 1) * layout.pointer_size]
        if name is None:
            if relocation is not None or any(raw):
                raise NativeImageError("native null initializer has a relocation or nonzero file bytes")
            result.append(NativePointerExpectation(index, offset, None))
            continue
        if type(name) is not str or not name or relocation is None:
            raise NativeImageError("native pointer initializer has no declared name or relocation")
        if relocation.kind == "R_X86_64_64":
            if not 0 <= relocation.symbol_index < len(layout.symbols) or relocation.addend != 0:
                raise NativeImageError("unresolved native pointer table absolute relocation")
            target = layout.symbols[relocation.symbol_index]
            _ordinary_symbol(target, definition=True)
            if target.name != name or target.kind != "STT_FUNC":
                raise NativeImageError("native pointer table target differs from source initializer")
        elif relocation.kind == "R_X86_64_RELATIVE":
            if relocation.symbol_index != 0 or type(relocation.addend) is not int:
                raise NativeImageError("unresolved native pointer table relative relocation")
            if not layout.local_symbols:
                raise NativeImageError("native local symbol table absent for relative pointer reconstruction")
            possible = tuple(symbol for symbol in layout.local_symbols
                             if symbol.name == name and symbol.value == relocation.addend)
            if len(possible) != 1:
                raise NativeImageError("native relative pointer has no unique source-named target")
            target, = possible
            if (target.kind != "STT_FUNC" or target.binding != "STB_LOCAL"
                    or target.visibility != "STV_HIDDEN" or type(target.section) is not int
                    or target.section <= 0 or target.size <= 0):
                raise NativeImageError("unresolved native local pointer target")
        else:
            raise NativeImageError("unresolved native pointer table relocation kind")
        result.append(NativePointerExpectation(index, offset, target))
    return tuple(result)


def check_native_pointer_table(layout: NativeElfLayout, bias: int, backing,
                               mappings: Tuple[NativeMapping, ...], table_name: str,
                               initializers: Tuple[Optional[str], ...], file_bytes: bytes,
                               read_memory, *, page_size: int, plan=None):
    """Check every declared table entry, including native null entries.

    ``plan`` is the caller's retained ``plan_native_pointer_table`` result
    for these same arguments; it is derived when absent.
    """
    if plan is None:
        plan = plan_native_pointer_table(layout, table_name, initializers, file_bytes)
    witnesses = []
    for slot in plan:
        address = bias + slot.offset
        require_elf_range(layout.segments, bias, mappings, backing, address, layout.pointer_size,
                          "r", page_size=page_size)
        expected, name = 0, ""
        if slot.target is not None:
            expected, name = bias + slot.target.value, slot.target.name
            require_elf_range(layout.segments, bias, mappings, backing, expected, slot.target.size,
                              "x", page_size=page_size)
        raw = read_memory(address, layout.pointer_size)
        if type(raw) is not bytes or len(raw) != layout.pointer_size:
            raise NativeImageError("native pointer table read returned an incomplete interval")
        value = int.from_bytes(raw, layout.byteorder)
        if value != expected:
            raise NativeImageError("native pointer table differs at index " + str(slot.index))
        witnesses.append(NativeSlotWitness(address, value, name))
    return tuple(witnesses)


# Parsed layouts by reviewed image digest, for every guard in this process.
_ELF_LAYOUTS: dict = {}


def read_elf_layout(image: PinnedNativeFile) -> NativeElfLayout:
    """Parse the retained descriptor with maintained pyelftools 0.33.

    This first supported ABI requires an independently bounded dynamic symbol
    section joined to PT_DYNAMIC. Sectionless/version-table reconstruction,
    RELR and other machine ABIs remain unresolved; no loader call is used as
    a fallback. Relocation kinds outside the execution subset are retained.
    """
    try:
        from elftools import __version__
        from elftools.elf.dynamic import DynamicSegment
        from elftools.elf.elffile import ELFFile
        from elftools.elf.enums import ENUM_RELOC_TYPE_x64
        from elftools.elf.gnuversions import GNUVerSymSection
        from elftools.elf.relocation import RelocationTable
        from elftools.elf.sections import SymbolTableSection
    except ImportError as error:
        raise NativeImageError("native ELF custody requires pyelftools 0.33") from error
    if __version__ != "0.33":
        raise NativeImageError("unreviewed native ELF parser version: " + __version__)
    image.validate()
    # An owned image's bytes matched its digest when it was entered, and the
    # layout is a pure function of those bytes.
    cached = _ELF_LAYOUTS.get(image.sha256)
    if cached is not None:
        return cached
    # The duplicate shares the same open file description and backing inode.
    # Its closure never transfers ownership of the retained descriptor.
    with os.fdopen(os.dup(image.fileno()), "rb") as stream:
        stream.seek(0)
        elf = ELFFile(stream)
        if (elf.elfclass != 64 or not elf.little_endian
                or elf["e_machine"] != "EM_X86_64" or elf["e_type"] != "ET_DYN"):
            raise NativeImageError("unresolved native ELF machine, byte order or image type")
        segments = tuple(ElfLoadSegment(*(int(segment[key]) for key in (
            "p_offset", "p_vaddr", "p_filesz", "p_memsz", "p_flags", "p_align")))
            for segment in elf.iter_segments(type="PT_LOAD"))
        dynamic_segments = tuple(elf.iter_segments(type="PT_DYNAMIC"))
        if len(dynamic_segments) != 1:
            raise NativeImageError("native ELF needs one PT_DYNAMIC declaration")
        dynamic, = dynamic_segments
        if not isinstance(dynamic, DynamicSegment):
            raise NativeImageError("ELF parser did not retain the native dynamic segment type")
        tags = tuple(dynamic.iter_tags())
        tagged = {}
        for tag in tags:
            tagged.setdefault(tag["d_tag"], []).append(tag)

        def unique_tag(name, key="d_val"):
            records = tagged.get(name, ())
            if len(records) != 1:
                raise NativeImageError("native ELF needs one " + name + " declaration")
            return int(records[0][key])

        symbol_address = unique_tag("DT_SYMTAB", "d_ptr")
        symbol_tables = tuple(section for section in elf.iter_sections()
                              if section["sh_type"] == "SHT_DYNSYM"
                              and section["sh_addr"] == symbol_address)
        if len(symbol_tables) != 1:
            raise NativeImageError("native dynamic symbol denominator has no unique declared section")
        symbol_table, = symbol_tables
        if not isinstance(symbol_table, SymbolTableSection):
            raise NativeImageError("ELF parser did not retain the native symbol table type")
        entry_size = unique_tag("DT_SYMENT")
        if (entry_size != symbol_table["sh_entsize"] or entry_size != elf.structs.Elf_Sym.sizeof()
                or symbol_table["sh_size"] % entry_size):
            raise NativeImageError("native dynamic symbol layout differs from its declared extent")
        version_sections = tuple(section for section in elf.iter_sections()
                                 if section["sh_type"] == "SHT_GNU_versym")
        version_section = None
        if "DT_VERSYM" in tagged:
            address = unique_tag("DT_VERSYM", "d_ptr")
            matching = tuple(section for section in version_sections if section["sh_addr"] == address)
            if (len(matching) != 1 or not isinstance(matching[0], GNUVerSymSection)
                    or matching[0].num_symbols() != symbol_table.num_symbols()):
                raise NativeImageError("native symbol versions have no complete declared section")
            version_section = matching[0]
        elif version_sections:
            raise NativeImageError("native symbol version section disagrees with PT_DYNAMIC")

        def symbol_record(index, symbol, version="VER_NDX_GLOBAL"):
            return ElfSymbol(index, symbol.name, symbol["st_info"]["type"],
                             symbol["st_info"]["bind"], symbol["st_other"]["visibility"],
                             symbol["st_shndx"], int(symbol["st_value"]), int(symbol["st_size"]), version)

        symbols = []
        for index, symbol in enumerate(symbol_table.iter_symbols()):
            version = (version_section.get_symbol(index)["ndx"] if version_section is not None
                       else "VER_NDX_GLOBAL")
            record = symbol_record(index, symbol, version)
            if record != symbol_record(index, dynamic.get_symbol(index), version):
                raise NativeImageError("PT_DYNAMIC and dynamic symbol section disagree")
            symbols.append(record)
        if "DT_GNU_HASH" in tagged or "DT_HASH" in tagged:
            if dynamic.num_symbols() != len(symbols):
                raise NativeImageError("native hash-table and section symbol denominators disagree")
        local_tables = tuple(section for section in elf.iter_sections() if section["sh_type"] == "SHT_SYMTAB")
        if len(local_tables) > 1:
            raise NativeImageError("native local symbol table is ambiguous")
        locals_ = []
        for table in local_tables:
            if not isinstance(table, SymbolTableSection):
                raise NativeImageError("ELF parser did not retain the local symbol table type")
            locals_.extend(symbol_record(index, symbol) for index, symbol in enumerate(table.iter_symbols()))
        relocation_names = {value: key for key, value in ENUM_RELOC_TYPE_x64.items() if type(value) is int}
        relocations, locations = [], {}
        for table_name, table in dynamic.get_relocation_tables().items():
            if table_name == "RELR":
                raise NativeImageError("native RELR reconstruction is unresolved")
            if not isinstance(table, RelocationTable):
                raise NativeImageError("ELF parser did not retain the native relocation table type")
            start_tag, size_tag = (("DT_JMPREL", "DT_PLTRELSZ") if table_name == "JMPREL"
                                   else ("DT_" + table_name, "DT_" + table_name + "SZ"))
            address = unique_tag(start_tag, "d_ptr")
            file_offsets = tuple(elf.address_offsets(address, unique_tag(size_tag)))
            if len(file_offsets) != 1 or table.num_relocations() * table.entry_size != unique_tag(size_tag):
                raise NativeImageError("native relocation table has an ambiguous or truncated extent")
            for index, relocation in enumerate(table.iter_relocations()):
                file_offset = file_offsets[0] + index * table.entry_size
                record = ElfRelocation((table_name,), file_offset, int(relocation["r_offset"]),
                                       relocation_names.get(relocation["r_info_type"], str(relocation["r_info_type"])),
                                       int(relocation["r_info_sym"]),
                                       int(relocation["r_addend"]) if relocation.is_RELA() else None)
                if not 0 <= record.symbol_index < len(symbols):
                    raise NativeImageError("native relocation has a dangling symbol index")
                previous = locations.get(file_offset)
                if previous is not None:
                    if replace(record, tables=relocations[previous].tables) != relocations[previous]:
                        raise NativeImageError("overlapping native relocation tables disagree")
                    relocations[previous] = replace(record, tables=relocations[previous].tables + (table_name,))
                else:
                    locations[file_offset] = len(relocations)
                    relocations.append(record)
        ordered = sorted(relocations, key=lambda record: record.offset)
        if any(first.offset == second.offset for first, second in zip(ordered, ordered[1:])):
            raise NativeImageError("different native relocation records write the same address")
        relro = tuple((int(segment["p_vaddr"]), int(segment["p_memsz"]))
                      for segment in elf.iter_segments(type="PT_GNU_RELRO"))
        now = ("DT_BIND_NOW" in tagged
               or any(int(tag["d_val"]) & 8 for tag in tagged.get("DT_FLAGS", ()))
               or any(int(tag["d_val"]) & 1 for tag in tagged.get("DT_FLAGS_1", ())))
        needed = tuple(tag.needed for tag in tagged.get("DT_NEEDED", ()))
        result = NativeElfLayout(segments, tuple(symbols), tuple(locals_), tuple(relocations), relro, now, needed,
                                 image_sha256=image.sha256)
    image.validate()
    _ELF_LAYOUTS[image.sha256] = result
    return result


class PinnedNativeFile:
    """Own one read-only descriptor throughout a native image custody check."""

    def __init__(self, path, sha256: str):
        if type(sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise NativeImageError("native file needs a reviewed SHA-256 digest")
        self.path = Path(path)
        self.sha256 = sha256
        self._stream = None
        self._stamp = None
        self._backing = None

    def __enter__(self):
        if self._stream is not None:
            raise NativeImageError("native file descriptor is already owned")
        try:
            stream = self.path.open("rb")
            self._stream = stream
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink == 0:
                raise NativeImageError("native subject is not a linked regular file")
            self._stamp = _file_stamp(before)
            self._backing = (os.major(before.st_dev), os.minor(before.st_dev), before.st_ino)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            self.validate()
            if digest.hexdigest() != self.sha256:
                raise NativeImageError("native file digest differs from reviewed bytes: " + str(self.path))
            stream.seek(0)
            return self
        except BaseException:
            self.close()
            raise

    def fileno(self) -> int:
        if self._stream is None:
            raise NativeImageError("native file descriptor is not owned")
        return self._stream.fileno()

    @property
    def backing(self) -> Tuple[int, int, int]:
        if self._stream is None or self._backing is None:
            raise NativeImageError("native file descriptor is not owned")
        return self._backing

    def validate(self) -> None:
        """Detect descriptor or installed-path drift since the content hash."""
        try:
            if (_file_stamp(os.fstat(self.fileno())) != self._stamp
                    or _file_stamp(self.path.stat()) != self._stamp):
                raise NativeImageError("native file identity changed: " + str(self.path))
        except OSError as error:
            raise NativeImageError("native file identity changed: " + str(self.path) + ": " + str(error)) from error

    def mappings(self, records: Tuple[NativeMapping, ...]) -> Tuple[NativeMapping, ...]:
        self.validate()
        backing = self.backing
        selected = tuple(record for record in records if record.backing == backing)
        if not selected:
            raise NativeImageError("reviewed native file is not mapped: " + str(self.path))
        return selected

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __exit__(self, *exception):
        self.close()
