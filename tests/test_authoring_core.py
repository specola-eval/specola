"""An offline check must agree with the live one, or say it cannot.

Checking a specification against a recording is only worth having if it
gives the answer the running simulator would have given. The dangerous
direction is acceptance: a check that rejects too much wastes someone's
afternoon, while a check that accepts too much ships a specification
that binds nothing and reports a clean episode.

These tests pin the agreement itself across a spread of drafts, which is
the property that matters and the one that can actually be checked.

They deliberately do *not* claim that dropping the adapter's
capabilities record flips a verdict. That was assumed while building
this and turned out to be unsupported: the component catalogue and the
reference parser reject an unknown class or key without consulting
capabilities, and ``semantic.py`` treats absent capabilities as closed
rather than open, so losing them would make a check stricter rather than
looser. The record is persisted so the two contexts are given identical
inputs and cannot drift — a fidelity measure — and a document lacking it
is refused rather than restored with substitutes.
"""

from __future__ import annotations

import gzip
import json

import pytest

from specola.authoring import (
    CheckContext,
    check_draft,
    normalise_name,
    spec_entry,
    write_specs_document,
)
from specola.capabilities import reset_adapter
from specola.cli_common import _MockAdapter
from specola.components import ComponentCatalog
from specola.discovery_capture import discovery_payload
from specola.discovery_model import property_metadata_for
from specola.discovery_restore import (
    DiscoveryRestoreError,
    discovery_model_from_document,
)


def _live_and_offline():
    adapter = _MockAdapter()
    reset = reset_adapter(adapter)
    entities = reset.initial_observation.perceived_entities
    metadata = property_metadata_for(adapter, entities)
    live = CheckContext(
        dt=adapter.dt,
        catalog=ComponentCatalog.from_entities(entities, metadata=metadata),
        metadata=metadata,
        entities=entities,
        labels=getattr(adapter, "entity_labels", None),
        capabilities=reset.capabilities,
    )
    document = json.loads(gzip.decompress(discovery_payload(
        adapter, reset, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )))
    offline = CheckContext.from_discovery_model(
        discovery_model_from_document(document)
    )
    return live, offline, document


DRAFTS = [
    ("resolves", "historically[0,1s](object_*[mass] >= 0)"),
    ("glob_over_binds", "historically[0,1s](object_*[xpos_*] >= 0)"),
    ("unknown_class", "historically[0,1s](spaceship_*[mass] >= 0)"),
    ("unknown_key", "historically[0,1s](object_*[no_such_key] >= 0)"),
    ("quantified", "historically[0,1s](object_*[all: mass] >= 0)"),
    ("misplaced", "historically[0,1s](Max(object_*[all: mass]) >= 0)"),
    ("malformed", "historically[0,1s](object_*[mass] >>>= 0)"),
]


@pytest.mark.parametrize(("name", "formula"), DRAFTS)
def test_offline_check_agrees_with_the_live_one(name, formula):
    live, offline, _ = _live_and_offline()
    assert (
        check_draft(live, name=name, formula=formula).accepted
        == check_draft(offline, name=name, formula=formula).accepted
    )


def test_an_unknown_class_is_rejected_offline_too():
    """Rejected by the catalogue and the parser, not by capabilities.

    Worth pinning anyway: it is the verdict an author most needs to be
    the same in both places, whichever layer produces it.
    """
    _, offline, _ = _live_and_offline()
    result = check_draft(
        offline,
        name="unknown_class",
        formula="historically[0,1s](spaceship_*[mass] >= 0)",
    )
    assert not result.accepted


def test_vocabulary_mode_does_not_by_itself_admit_an_unknown_class():
    """Records what was measured, against what was assumed.

    Substituting an OPEN class vocabulary does not make an absent class
    compile: the reference parser resolves the prefix against the
    catalogue first and refuses there. This is pinned because the
    opposite was believed while this module was written, and a future
    reader is otherwise likely to believe it again.
    """
    from dataclasses import replace

    from specola.capabilities import VocabularyMode

    _, offline, _ = _live_and_offline()
    opened = replace(
        offline,
        capabilities=replace(
            offline.capabilities,
            class_mode=VocabularyMode.OPEN,
            supported_classes=frozenset(),
        ),
    )
    result = check_draft(
        opened,
        name="unknown_class",
        formula="historically[0,1s](spaceship_*[mass] >= 0)",
    )
    assert not result.accepted
    assert "spaceship" in result.report


def test_a_document_without_capabilities_refuses_to_restore():
    """A stated limitation, chosen over a silently different authority.

    Recordings written before the sidecar existed are browsable but not
    checkable. That is a real cost, accepted so that a check never
    compiles against substituted capabilities and reports a verdict the
    run would not have given.
    """
    _, _, document = _live_and_offline()
    stripped = dict(document)
    stripped["capabilities"] = None
    with pytest.raises(DiscoveryRestoreError) as excinfo:
        discovery_model_from_document(stripped)
    assert "capabilities" in str(excinfo.value)


def test_the_report_carries_the_checkers_own_explanation():
    """Resolved keys and units are why an author reads a check at all."""
    _, offline, _ = _live_and_offline()
    result = check_draft(
        offline,
        name="resolves",
        formula="historically[0,1s](object_*[mass] >= 0)",
    )
    assert result.accepted
    assert "mass" in result.report
    assert result.error is None


def test_a_checker_crash_rejects_the_draft_rather_than_the_server():
    class _Exploding:
        def __getattr__(self, name):
            raise RuntimeError("poisoned context")

    context = CheckContext(
        dt=0.1, catalog=_Exploding(), metadata={}, entities={},
        labels=None, capabilities=None,
    )
    result = check_draft(
        context, name="x", formula="historically[0,1s](a_*[b] >= 0)"
    )
    assert not result.accepted
    # The checker catches this itself and explains it in the report
    # rather than raising, so the draft is rejected with a reason and
    # check_draft's own exception path stays unused.
    assert "catalog" in result.report


def test_spec_entry_emits_only_schema_keys():
    """``window:`` loads without complaint and does nothing."""
    entry = spec_entry(
        name="x", formula="f", constants={"t": 1.0},
        backend="auto", symmetric=False,
    )
    assert set(entry) == {"name", "formula", "constants"}
    assert "window" not in entry

    full = spec_entry(
        name="x", formula="f", constants={}, backend="rtamt",
        symmetric=True,
    )
    assert set(full) == {"name", "formula", "backend", "symmetric"}


def test_names_normalise_the_way_the_writer_always_has():
    assert normalise_name("Joint Limit") == "joint_limit"


def test_save_refuses_to_clobber_a_file_that_appeared(tmp_path):
    """The collision race, which is why link is used and not replace."""
    target = tmp_path / "specs.yaml"
    target.write_text("specs: []\n")
    with pytest.raises(FileExistsError):
        write_specs_document(
            [{"name": "a", "formula": "f"}], str(target), overwrite=False
        )
    assert target.read_text() == "specs: []\n"

    write_specs_document(
        [{"name": "a", "formula": "f"}], str(target), overwrite=True
    )
    import yaml

    assert yaml.safe_load(target.read_text())["specs"][0]["name"] == "a"


def test_save_leaves_no_temporary_files_behind(tmp_path):
    target = tmp_path / "specs.yaml"
    write_specs_document(
        [{"name": "a", "formula": "f"}], str(target), overwrite=False
    )
    assert [path.name for path in tmp_path.iterdir()] == ["specs.yaml"]
