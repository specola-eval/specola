"""File identity and memory boundaries precede native image reads."""

import hashlib
import os
from pathlib import Path
from typing import Union

import pytest

from specola.conformance import native_images


def test_maps_preserves_ambiguous_path_bytes_and_split_regions():
    raw = (b"1000-1800 r--p 00000000 08:02 123 /tmp/a b\\012 (deleted)\n"
           b"1800-2000 r-xp 00000800 08:02 123 /tmp/a b\\012 (deleted)\n"
           b"2000-3000 rw-p 00000000 00:00 0\n")
    regions = native_images.parse_native_mappings(raw)
    assert len(regions) == 3
    assert regions[0].pathname == b"/tmp/a b\\012 (deleted)"
    assert regions[0].backing == (8, 2, 123)
    assert regions[1].offset == 0x800
    assert regions[2].pathname == b""
    assert regions[2].backing is None


def test_maps_path_control_bytes_are_not_additional_record_separators():
    path = b"/tmp/a\x0b\x0c\rfile"
    record, = native_images.parse_native_mappings(b"1000-2000 r--p 0 08:02 1 " + path + b"\n")
    assert record.pathname == path


@pytest.mark.parametrize("raw", [
    b"", b"not a map\n", b"2000-1000 r--p 0 08:02 123 x\n",
    b"1000-2000 r-zp 0 08:02 123 x\n", b"1000-2000 r--q 0 08:02 123 x\n",
    b"1000-2000 r--p 0 08:xx 123 x\n", b"1000-2000 r--p 0 08:02 -1 x\n",
    b"1000-2000 r--p 0 08:02 1 x\n1800-3000 r--p 800 08:02 1 x\n",
    b"2000-3000 r--p 0 08:02 1 x\n1000-1800 r--p 800 08:02 1 x\n",
])
def test_malformed_or_overlapping_maps_do_not_become_empty_evidence(raw):
    with pytest.raises(native_images.NativeImageError):
        native_images.parse_native_mappings(raw)


def test_full_range_requires_contiguous_permissions_and_backing_identity():
    regions = native_images.parse_native_mappings(
        b"1000-1800 r--p 0 08:02 123 x\n1800-2000 rw-p 800 08:02 123 x\n"
        b"2000-2800 ---p 1000 08:02 123 x\n3000-3800 r--p 0 08:02 124 y\n")
    covered = native_images.require_mapped_range(regions, 0x1700, 0x200, "r", (8, 2, 123))
    assert covered == regions[:2]
    for start, size, mode, backing in (
        (0x1700, 0x200, "w", (8, 2, 123)),
        (0x1900, 0x800, "r", (8, 2, 123)),
        (0x2700, 0xA00, "", (8, 2, 123)),
        (0x3000, 1, "r", (8, 2, 123)),
        (0x1700, 0, "r", (8, 2, 123)),
        (0x1700, -1, "r", (8, 2, 123)),
    ):
        with pytest.raises(native_images.NativeImageError):
            native_images.require_mapped_range(regions, start, size, mode, backing)


def test_census_bisection_matches_a_scan_of_the_same_records():
    census = native_images.parse_native_mappings(
        b"1000-1800 r--p 0 08:02 123 x\n1800-2000 rw-p 800 08:02 123 x\n"
        b"2000-2800 ---p 1000 08:02 123 x\n3000-3800 r--p 0 08:02 124 y\n4000-5000 r-xp 0 00:00 0\n")
    scan = tuple(census)
    assert type(scan) is tuple and scan == census
    for start in range(0x0F00, 0x5100, 0x80):
        for size in (1, 0x80, 0x800, 0x1800):
            for mode, backing in (("r", None), ("r", (8, 2, 123)), ("", None), ("x", None)):
                outcomes = []
                for mappings in (census, scan):
                    try:
                        outcomes.append(native_images.require_mapped_range(mappings, start, size, mode, backing))
                    except native_images.NativeImageError as error:
                        outcomes.append(str(error))
                assert outcomes[0] == outcomes[1], (hex(start), size, mode, backing)


def test_maps_reread_reuses_records_and_still_checks_every_line_and_order():
    raw = b"1000-1800 r--p 0 08:02 123 x\n2000-2800 rw-p 0 00:00 0\n"
    first = native_images.parse_native_mappings(raw)
    second = native_images.parse_native_mappings(raw)
    assert second == first and all(old is new for old, new in zip(first, second))
    with pytest.raises(native_images.NativeImageError, match="unordered"):
        native_images.parse_native_mappings(b"".join(reversed(raw.splitlines(keepends=True))))
    with pytest.raises(native_images.NativeImageError, match="malformed"):
        native_images.parse_native_mappings(raw + b"not a map\n")


def test_elf_layout_is_reused_for_one_reviewed_digest_and_still_needs_ownership():
    pytest.importorskip("elftools")
    import _ctypes

    path = Path(_ctypes.__file__)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with native_images.PinnedNativeFile(path, digest) as image:
        layout = native_images.read_elf_layout(image)
    with native_images.PinnedNativeFile(path, digest) as again:
        assert native_images.read_elf_layout(again) is layout
    with pytest.raises(native_images.NativeImageError, match="not owned"):
        native_images.read_elf_layout(again)


def test_retained_descriptor_detects_byte_identical_path_replacement(tmp_path):
    path = tmp_path / "engine.so"
    payload = b"fixture file bytes; this file is never loaded"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with native_images.PinnedNativeFile(path, digest) as image:
        original_fd = image.fileno()
        original = os.fstat(original_fd)
        assert image.backing == (os.major(original.st_dev), os.minor(original.st_dev), original.st_ino)
        image.validate()
        replacement = tmp_path / "replacement"
        replacement.write_bytes(payload)
        replacement.replace(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        with pytest.raises(native_images.NativeImageError, match="identity changed"):
            image.validate()
        assert os.fstat(original_fd).st_ino == original.st_ino
    with pytest.raises(OSError):
        os.fstat(original_fd)


def test_hash_failure_and_body_exception_close_owned_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "engine.so"
    path.write_bytes(b"engine")
    opened = []
    original = Path.open

    def track(self, *args, **kwargs):
        stream = original(self, *args, **kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", track)
    with pytest.raises(native_images.NativeImageError, match="digest"):
        with native_images.PinnedNativeFile(path, "0" * 64):
            pytest.fail("incorrect native file identity was accepted")
    assert opened and all(stream.closed for stream in opened)
    with pytest.raises(RuntimeError, match="caller failed"):
        with native_images.PinnedNativeFile(path, hashlib.sha256(b"engine").hexdigest()):
            raise RuntimeError("caller failed")
    assert all(stream.closed for stream in opened)


def test_mapping_match_uses_descriptor_inode_and_retains_all_segments(tmp_path):
    path = tmp_path / "engine.so"
    path.write_bytes(b"engine")
    with native_images.PinnedNativeFile(path, hashlib.sha256(b"engine").hexdigest()) as image:
        major, minor, inode = image.backing
        rows = native_images.parse_native_mappings(
            (f"1000-1800 r--p 0 {major:x}:{minor:x} {inode} unrelated diagnostic path\n"
             f"1800-2000 r-xp 800 {major:x}:{minor:x} {inode} same inode\n"
             f"3000-4000 r-xp 0 {major:x}:{minor:x} {inode + 1} {path}\n").encode())
        assert image.mappings(rows) == rows[:2]
        with pytest.raises(native_images.NativeImageError, match="not mapped"):
            image.mappings(rows[2:])


def test_file_edit_and_unlink_are_identity_drift(tmp_path):
    for mutation in ("edit", "unlink"):
        path = tmp_path / mutation
        path.write_bytes(b"engine")
        with native_images.PinnedNativeFile(path, hashlib.sha256(b"engine").hexdigest()) as image:
            if mutation == "edit":
                path.write_bytes(b"changed engine")
            else:
                path.unlink()
            with pytest.raises(native_images.NativeImageError, match="identity changed"):
                image.validate()


def _loads():
    return (
        native_images.ElfLoadSegment(0, 0, 0x800, 0x800, 4, 0x1000),
        native_images.ElfLoadSegment(0x1000, 0x3000, 0x1800, 0x1800, 5, 0x1000),
        native_images.ElfLoadSegment(0x3000, 0x6000, 0x1000, 0x2800, 6, 0x1000),
    )


def _load_maps(base=0x10000):
    # These native layouts are independent fixture declarations. The ELF
    # virtual addresses differ from file offsets; RELRO splits one segment.
    return native_images.parse_native_mappings((
        f"{base:x}-{base + 0x1000:x} r--p 0 08:02 123 engine\n"
        f"{base + 0x3000:x}-{base + 0x4000:x} r-xp 1000 08:02 123 engine\n"
        f"{base + 0x4000:x}-{base + 0x5000:x} r-xp 2000 08:02 123 engine\n"
        f"{base + 0x6000:x}-{base + 0x6800:x} r--p 3000 08:02 123 engine\n"
        f"{base + 0x6800:x}-{base + 0x7000:x} rw-p 3800 08:02 123 engine\n"
        f"{base + 0x7000:x}-{base + 0x9000:x} rw-p 0 00:00 0\n"
    ).encode())


def test_load_bias_joins_actual_segment_offsets_and_split_mappings():
    assert native_images.resolve_load_bias(_loads(), _load_maps(), (8, 2, 123)) == 0x10000


def test_same_inode_loaded_twice_requires_an_unambiguous_instance():
    mappings = _load_maps() + _load_maps(0x20000)
    with pytest.raises(native_images.NativeImageError, match="ambiguous"):
        native_images.resolve_load_bias(_loads(), mappings, (8, 2, 123))
    assert native_images.resolve_load_bias(_loads(), mappings, (8, 2, 123), anchor=0x23100) == 0x20000


def test_wrong_segment_offset_or_removed_fragment_rejects_load_bias():
    from dataclasses import replace

    maps = _load_maps()
    for changed in (maps[:2] + maps[3:], maps[:2] + (replace(maps[2], offset=0),) + maps[3:]):
        with pytest.raises(native_images.NativeImageError, match="no consistent"):
            native_images.resolve_load_bias(_loads(), changed, (8, 2, 123))


def test_elf_range_requires_correct_segment_and_bss_ownership():
    from dataclasses import replace

    maps = _load_maps()
    assert native_images.require_elf_range(_loads(), 0x10000, maps, (8, 2, 123),
                                           0x13F00, 0x200, "x", page_size=0x1000) == maps[1:3]
    assert native_images.require_elf_range(_loads(), 0x10000, maps, (8, 2, 123),
                                           0x17800, 0x200, "r", page_size=0x1000) == maps[-1:]
    for address, size, mode in ((0x15000, 8, "r"), (0x17800, 8, "x"),
                                (0x187FF, 2, "r"), (0x13F00, 0x200, "w")):
        with pytest.raises(native_images.NativeImageError):
            native_images.require_elf_range(_loads(), 0x10000, maps, (8, 2, 123),
                                            address, size, mode, page_size=0x1000)
    # A readable file-backed mapping cannot impersonate this image's BSS.
    wrong = maps[:-1] + (replace(maps[-1], inode=124, device_major=8, device_minor=2),)
    with pytest.raises(native_images.NativeImageError, match="BSS"):
        native_images.require_elf_range(_loads(), 0x10000, wrong, (8, 2, 123),
                                        0x17800, 8, "r", page_size=0x1000)


def test_one_mapping_cannot_hide_file_backing_across_the_bss_boundary():
    from dataclasses import replace

    maps = _load_maps()
    # This interval starts in declared file data but ends in anonymous BSS.
    # Extending the same inode's mapping over both is not the declared layout.
    extended = maps[:-2] + (replace(maps[-2], end=maps[-1].end),)
    with pytest.raises(native_images.NativeImageError, match="BSS"):
        native_images.require_elf_range(_loads(), 0x10000, extended, (8, 2, 123),
                                        0x16F00, 0x200, "r", page_size=0x1000)


def test_native_relocation_partition_retains_unbound_and_relative_records():
    from dataclasses import replace

    importer, target = _binding_layouts()
    remaining = (
        native_images.ElfRelocation(("RELA",), 0x148, 0x6118, "R_X86_64_GLOB_DAT", 3, 0),
        native_images.ElfRelocation(("RELA",), 0x160, 0x6120, "R_X86_64_RELATIVE", 0, 0x6800),
    )
    extended = replace(importer, symbols=importer.symbols + (_symbol(3, "another_library"),),
                       relocations=importer.relocations + remaining)
    selected, other = native_images.partition_native_relocations(extended, target)
    assert selected == importer.relocations
    assert other == remaining
    assert tuple(item.relocation for item in native_images.plan_native_bindings(extended, target)) == selected
    added = native_images.ElfRelocation(("RELA",), 0x178, 0x6128, "R_X86_64_GLOB_DAT", 3, 0)
    _, changed = native_images.partition_native_relocations(
        replace(extended, relocations=extended.relocations + (added,)), target)
    assert changed == other + (added,)


def test_relative_table_target_requires_retained_local_symbol_declarations():
    from dataclasses import replace

    layout = replace(_table_layout(), local_symbols=())
    with pytest.raises(native_images.NativeImageError, match="local symbol table absent"):
        native_images.plan_native_pointer_table(layout, "dispatch", (None, "query", "hidden_query"), bytes(24))


def _symbol(index, name, *, value=0, section: Union[str, int] = "SHN_UNDEF", kind="STT_FUNC",
            size=16, binding="STB_GLOBAL", visibility="STV_DEFAULT", version="VER_NDX_GLOBAL"):
    return native_images.ElfSymbol(index, name, kind, binding, visibility, section, value, size, version)


def _binding_layouts():
    empty = _symbol(0, "", kind="STT_NOTYPE", size=0, binding="STB_LOCAL", version="VER_NDX_LOCAL")
    symbols = (empty, _symbol(1, "query"), _symbol(2, "state", kind="STT_OBJECT", size=8))
    relocations = tuple(native_images.ElfRelocation(("RELA",), position, offset, kind, symbol, 0)
                        for position, offset, kind, symbol in (
                            (0x100, 0x6100, "R_X86_64_GLOB_DAT", 1),
                            (0x118, 0x6108, "R_X86_64_JUMP_SLOT", 1),
                            (0x130, 0x6110, "R_X86_64_GLOB_DAT", 2)))
    importer = native_images.NativeElfLayout(_loads(), symbols, (), relocations,
                                             ((0x6000, 0x800),), True, ("engine",))
    definitions = (empty, _symbol(1, "query", value=0x3100, section=1),
                   _symbol(2, "state", value=0x7800, section=2, kind="STT_OBJECT", size=8))
    target = native_images.NativeElfLayout(_loads(), definitions, (), (), (), True, ())
    return importer, target


def test_binding_plan_retains_two_native_slots_for_one_function():
    importer, target = _binding_layouts()
    plan = native_images.plan_native_bindings(importer, target)
    assert tuple((item.relocation.offset, item.target.name, item.target.value) for item in plan) == (
        (0x6100, "query", 0x3100), (0x6108, "query", 0x3100), (0x6110, "state", 0x7800))


@pytest.mark.parametrize("damage", ["ifunc", "version", "weak", "duplicate", "addend", "kind", "lazy"])
def test_unsupported_binding_semantics_fail_before_native_reads(damage):
    from dataclasses import replace

    importer, target = _binding_layouts()
    if damage in ("ifunc", "version", "weak"):
        changed = {"ifunc": {"kind": "STT_LOOS"}, "version": {"version_index": 2},
                   "weak": {"binding": "STB_WEAK"}}[damage]
        target = replace(target, symbols=(target.symbols[0], replace(target.symbols[1], **changed), target.symbols[2]))
    elif damage == "duplicate":
        target = replace(target, symbols=target.symbols + (replace(target.symbols[1], index=3),))
    elif damage in ("addend", "kind"):
        changed = {"addend": {"addend": 1}, "kind": {"kind": "R_X86_64_COPY"}}[damage]
        importer = replace(importer, relocations=(replace(importer.relocations[0], **changed),) + importer.relocations[1:])
    else:
        importer = replace(importer, bind_now=False)
    with pytest.raises(native_images.NativeImageError):
        native_images.plan_native_bindings(importer, target)


def test_binding_check_uses_exact_targets_and_complete_slot_population():
    importer, target = _binding_layouts()
    # Both fixture images use the same inode here, but their selected image
    # instances have different independently declared bases.
    maps = _load_maps(0x10000) + _load_maps(0x20000)
    expected = {0x16100: 0x23100, 0x16108: 0x23100, 0x16110: 0x27800}
    read = []

    def memory(address, size):
        read.append(address)
        assert size == 8
        return expected[address].to_bytes(8, "little")

    witnesses = native_images.check_native_bindings(importer, 0x10000, (8, 2, 123),
                                                    target, 0x20000, (8, 2, 123), maps,
                                                    memory, page_size=0x1000)
    assert read == list(expected)
    assert tuple((item.address, item.value) for item in witnesses) == tuple(expected.items())
    expected[0x16108] = 0x23101
    with pytest.raises(native_images.NativeImageError, match="target differs"):
        native_images.check_native_bindings(importer, 0x10000, (8, 2, 123),
                                            target, 0x20000, (8, 2, 123), maps,
                                            memory, page_size=0x1000)


def test_retained_plans_check_exactly_what_derived_plans_check():
    importer, target = _binding_layouts()
    maps = _load_maps(0x10000) + _load_maps(0x20000)
    values = {0x16100: 0x23100, 0x16108: 0x23100, 0x16110: 0x27800,
              0x16800: 0, 0x16808: 0x13100, 0x16810: 0x13200}

    def memory(address, size):
        return values[address].to_bytes(size, "little")

    def bindings(**plan):
        return native_images.check_native_bindings(importer, 0x10000, (8, 2, 123), target, 0x20000, (8, 2, 123),
                                                   maps, memory, page_size=0x1000, **plan)

    layout, names = _table_layout(), (None, "query", "hidden_query")

    def table(**plan):
        return native_images.check_native_pointer_table(layout, 0x10000, (8, 2, 123), _load_maps(), "dispatch",
                                                        names, bytes(24), memory, page_size=0x1000, **plan)

    binding_plan = native_images.plan_native_bindings(importer, target)
    table_plan = native_images.plan_native_pointer_table(layout, "dispatch", names, bytes(24))
    assert bindings(plan=binding_plan) == bindings()
    assert table(plan=table_plan) == table()
    values[0x16108], values[0x16808] = 0x23101, 0x13200
    with pytest.raises(native_images.NativeImageError, match="target differs"):
        bindings(plan=binding_plan)
    with pytest.raises(native_images.NativeImageError, match="pointer table differs"):
        table(plan=table_plan)


def test_binding_check_rejects_writable_relro_before_read():
    from dataclasses import replace

    importer, target = _binding_layouts()
    maps = _load_maps()
    maps = maps[:3] + (replace(maps[3], permissions="rw-p"),) + maps[4:] + _load_maps(0x20000)

    def forbidden_memory(*args):
        pytest.fail("unprotected native slot reached process memory access")

    with pytest.raises(native_images.NativeImageError, match="RELRO"):
        native_images.check_native_bindings(importer, 0x10000, (8, 2, 123),
                                            target, 0x20000, (8, 2, 123), maps,
                                            forbidden_memory, page_size=0x1000)


def _table_layout():
    from dataclasses import replace

    _, layout = _binding_layouts()
    symbols = layout.symbols + (_symbol(3, "dispatch", value=0x6800, section=2, kind="STT_OBJECT", size=24),)
    local = _symbol(0, "hidden_query", value=0x3200, section=1,
                    binding="STB_LOCAL", visibility="STV_HIDDEN")
    relocations = (
        native_images.ElfRelocation(("RELA",), 0x100, 0x6808, "R_X86_64_64", 1, 0),
        native_images.ElfRelocation(("RELA",), 0x118, 0x6810, "R_X86_64_RELATIVE", 0, 0x3200),
    )
    return replace(layout, symbols=symbols, local_symbols=(local,), relocations=relocations)


def test_pointer_table_reconstruction_preserves_null_global_and_local_entries():
    layout = _table_layout()
    expected = native_images.plan_native_pointer_table(layout, "dispatch", (None, "query", "hidden_query"), b"\0" * 24)
    assert tuple((item.index, item.offset, item.target.name if item.target else None) for item in expected) == (
        (0, 0x6800, None), (1, 0x6808, "query"), (2, 0x6810, "hidden_query"))


@pytest.mark.parametrize("damage", ["omit_initializer", "omit_relocation", "unused_relocation", "wrong_name",
                                   "wrong_addend", "unexplained_bytes", "overlap", "missing_local"])
def test_pointer_table_reconstruction_rejects_omissions_and_substitutions(damage):
    from dataclasses import replace

    layout = _table_layout()
    names, raw = (None, "query", "hidden_query"), b"\0" * 24
    if damage == "omit_initializer":
        names = names[:-1]
    elif damage == "omit_relocation":
        layout = replace(layout, relocations=layout.relocations[:-1])
    elif damage == "unused_relocation":
        layout = replace(layout, relocations=layout.relocations + (replace(layout.relocations[0], offset=0x6800),))
    elif damage == "wrong_name":
        names = (None, "hidden_query", "query")
    elif damage == "wrong_addend":
        layout = replace(layout, relocations=layout.relocations[:1] + (replace(layout.relocations[1], addend=0x3201),))
    elif damage == "unexplained_bytes":
        raw = b"\1" + raw[1:]
    elif damage == "overlap":
        layout = replace(layout, relocations=layout.relocations + (replace(layout.relocations[0], offset=0x6804),))
    else:
        layout = replace(layout, local_symbols=())
    with pytest.raises(native_images.NativeImageError):
        native_images.plan_native_pointer_table(layout, "dispatch", names, raw)


def test_pointer_table_check_rejects_mutation_before_a_reader_snapshot():
    layout = _table_layout()
    values = {0x16800: 0, 0x16808: 0x13100, 0x16810: 0x13200}

    def memory(address, size):
        assert size == 8
        return values[address].to_bytes(8, "little")

    def check():
        return native_images.check_native_pointer_table(layout, 0x10000, (8, 2, 123), _load_maps(),
                                                        "dispatch", (None, "query", "hidden_query"), b"\0" * 24,
                                                        memory, page_size=0x1000)

    assert len(check()) == 3
    values[0x16808] = 0x13200
    with pytest.raises(native_images.NativeImageError, match="pointer table differs"):
        check()
    values[0x16808] = 0x13100
    values[0x16800] = 0x13100
    with pytest.raises(native_images.NativeImageError, match="pointer table differs"):
        check()


def test_process_memory_reader_checks_range_before_pread_and_closes_descriptor(monkeypatch):
    maps = _load_maps()
    calls = []

    def open_(path, flags):
        assert path == "/proc/self/mem"
        calls.append(("open", flags))
        return 50

    def pread(fd, size, address):
        calls.append(("pread", fd, size, address))
        return b"\0" * size

    monkeypatch.setattr(native_images.os, "open", open_)
    monkeypatch.setattr(native_images.os, "pread", pread)
    monkeypatch.setattr(native_images.os, "close", lambda fd: calls.append(("close", fd)))
    with native_images.NativeMemoryReader(maps) as memory:
        assert memory.read(0x13100, 8) == b"\0" * 8
        with pytest.raises(native_images.NativeImageError):
            memory.read(0x15000, 8)
    assert [call for call in calls if call[0] == "pread"] == [("pread", 50, 8, 0x13100)]
    assert calls[-1] == ("close", 50)


def test_process_memory_reader_preserves_pread_error_and_short_read(monkeypatch):
    monkeypatch.setattr(native_images.os, "open", lambda *args: 50)
    closed = []
    monkeypatch.setattr(native_images.os, "close", closed.append)
    with native_images.NativeMemoryReader(_load_maps()) as memory:
        monkeypatch.setattr(native_images.os, "pread", lambda *args: b"\0")
        with pytest.raises(native_images.NativeImageError, match="incomplete"):
            memory.read(0x13100, 8)

        def inaccessible(*args):
            raise OSError("fixture inaccessible memory")

        monkeypatch.setattr(native_images.os, "pread", inaccessible)
        with pytest.raises(native_images.NativeImageError, match="fixture inaccessible memory"):
            memory.read(0x13100, 8)
    assert closed == [50]


def test_elf_file_bytes_preserve_offset_mapping_and_refuse_bss(tmp_path):
    payload = bytes(range(128))
    path = tmp_path / "declared-file"
    path.write_bytes(payload)
    layout = native_images.NativeElfLayout(
        (native_images.ElfLoadSegment(32, 0x1020, 64, 96, 6, 32),), (), (), (), (), True, ())
    with native_images.PinnedNativeFile(path, hashlib.sha256(payload).hexdigest()) as image:
        assert native_images.read_elf_file_bytes(image, layout, 0x1028, 4) == bytes((40, 41, 42, 43))
        with pytest.raises(native_images.NativeImageError, match="no unique file segment"):
            native_images.read_elf_file_bytes(image, layout, 0x1060, 4)
