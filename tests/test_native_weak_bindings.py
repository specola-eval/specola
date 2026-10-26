"""Weak definitions need exact native target and image-specific source custody."""

from dataclasses import replace

import pytest

from specola.conformance import native_images as native
from specola.conformance.core_runner import EvidenceIdentity
from test_native_image_custody import _binding_layouts, _load_maps


def _weak_self(kind):
    importer, target = _binding_layouts()
    symbol = replace(target.symbols[1], binding="STB_WEAK", kind=kind)
    return replace(target, symbols=(target.symbols[0], symbol, target.symbols[2]),
                   relocations=importer.relocations[:2], relro=importer.relro)


@pytest.mark.parametrize("kind", ("STT_FUNC", "STT_OBJECT"))
def test_weak_self_reference_retains_every_slot_and_rejects_interposition(kind):
    layout = _weak_self(kind)
    plans = native.plan_native_bindings(layout, layout)
    assert tuple(row.relocation.offset for row in plans) == (0x6100, 0x6108)
    witnesses = native.check_native_bindings(
        layout, 0x10000, (8, 2, 123), layout, 0x10000, (8, 2, 123), _load_maps(),
        lambda address, size: (0x13100).to_bytes(size, "little"), page_size=0x1000)
    assert len(witnesses) == 2
    with pytest.raises(native.NativeImageError, match="target differs"):
        native.check_native_bindings(
            layout, 0x10000, (8, 2, 123), layout, 0x10000, (8, 2, 123), _load_maps(),
            lambda address, size: (0x13101).to_bytes(size, "little"), page_size=0x1000)


def _coalesced():
    importer, target = _binding_layouts()
    reference = replace(importer.symbols[2], binding="STB_WEAK", section=2, value=0x7900)
    definition = replace(target.symbols[2], binding="STB_WEAK")
    importer = replace(importer, symbols=importer.symbols[:2] + (reference,),
                       relocations=importer.relocations[2:], image_sha256="a" * 64)
    target = replace(target, symbols=target.symbols[:2] + (definition,), image_sha256="b" * 64)
    policy = native.NativeObjectCoalescence(
        "state", "a" * 64, "b" * 64, (EvidenceIdentity("fixture shared object definition", "c" * 64),))
    return importer, target, policy


@pytest.mark.parametrize("value,role", ((0x17900, "importer"), (0x27800, "target")))
def test_reviewed_weak_object_coalescence_preserves_which_definition_won(value, role):
    importer, target, policy = _coalesced()
    plan, = native.plan_native_bindings(importer, target, coalescence=(policy,))
    assert plan.importer_alternative == importer.symbols[2]
    witness, = native.check_native_bindings(
        importer, 0x10000, (8, 2, 123), target, 0x20000, (8, 2, 123),
        _load_maps() + _load_maps(0x20000), lambda address, size: value.to_bytes(size, "little"),
        page_size=0x1000, coalescence=(policy,))
    assert (witness.value, witness.target_role) == (value, role)


def test_foreign_coalesced_object_is_not_a_third_accepted_candidate():
    importer, target, policy = _coalesced()
    with pytest.raises(native.NativeImageError, match="target differs"):
        native.check_native_bindings(
            importer, 0x10000, (8, 2, 123), target, 0x20000, (8, 2, 123),
            _load_maps() + _load_maps(0x20000), lambda address, size: (0x27900).to_bytes(size, "little"),
            page_size=0x1000, coalescence=(policy,))


@pytest.mark.parametrize("damage", ("no_policy", "wrong_importer", "wrong_target", "no_source", "duplicate_policy",
                                   "undefined", "strong_importer", "strong_target", "function", "gnu_unique"))
def test_coalescence_is_not_an_unbounded_weak_symbol_exception(damage):
    importer, target, policy = _coalesced()
    policies = (policy,)
    if damage == "no_policy":
        policies = ()
    elif damage == "wrong_importer":
        importer = replace(importer, image_sha256="d" * 64)
    elif damage == "wrong_target":
        target = replace(target, image_sha256="d" * 64)
    elif damage == "no_source":
        policies = (replace(policy, provenance=()),)
    elif damage == "duplicate_policy":
        policies = policies * 2
    elif damage == "undefined":
        importer = replace(importer, symbols=importer.symbols[:2] + (replace(importer.symbols[2], section="SHN_UNDEF"),))
    elif damage in ("strong_importer", "gnu_unique"):
        importer = replace(importer, symbols=importer.symbols[:2] + (replace(importer.symbols[2],
                           binding="STB_GLOBAL" if damage == "strong_importer" else "STB_GNU_UNIQUE"),))
    elif damage == "strong_target":
        target = replace(target, symbols=target.symbols[:2] + (replace(target.symbols[2], binding="STB_GLOBAL"),))
    else:
        importer = replace(importer, symbols=importer.symbols[:2] + (replace(importer.symbols[2], kind="STT_FUNC"),))
        target = replace(target, symbols=target.symbols[:2] + (replace(target.symbols[2], kind="STT_FUNC"),))
    with pytest.raises(native.NativeImageError):
        native.plan_native_bindings(importer, target, coalescence=policies)


def test_both_candidate_ranges_are_checked_before_reading_the_weak_slot():
    importer, target, policy = _coalesced()
    # Only the engine image is mapped. The observed pointer must not hide the
    # missing importer object domain by choosing the other accepted candidate.
    mappings = tuple(row for row in _load_maps() if row.end <= 0x17800) + _load_maps(0x20000)
    with pytest.raises(native.NativeImageError):
        native.check_native_bindings(
            importer, 0x10000, (8, 2, 123), target, 0x20000, (8, 2, 123), mappings,
            lambda *args: pytest.fail("unverified alternative reached a native pointer read"),
            page_size=0x1000, coalescence=(policy,))
