"""One implementation of the condensation rule, and it is Python's.

The browser used to aggregate and condense signals itself. The two rules
drifted in a way that produced a wrong answer rather than a cosmetic one:
JavaScript kept a shared *prefix*, ``_glob_for_keys`` keeps a shared
prefix **and** a shared suffix, so a family Python names ``a_*_0`` was
named ``a_*`` in the page. The pattern the page then sent back matched
nothing — and the server answered by dropping the signal and returning
the palette for the whole scope, so selecting one axis could hand
somebody a starter for an unrelated signal.

Both halves are pinned here: the rows come from Python, and a pattern
the server does not recognise is refused rather than widened.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("flask_socketio")

from specola.capabilities import reset_adapter        # noqa: E402
from specola.cli_common import _MockAdapter           # noqa: E402
from specola.dashboard.recording import (             # noqa: E402
    RecordingWriter,
)
from specola.dashboard.server import DashboardServer  # noqa: E402
from specola.discovery_capture import (               # noqa: E402
    discovery_payload,
)


def _server(tmp_path):
    recording = tmp_path / "run" / "recording_000"
    recording.mkdir(parents=True)
    writer = RecordingWriter(recording, {"run_id": "r"})
    adapter = _MockAdapter()
    writer.write_discovery(discovery_payload(
        adapter, reset_adapter(adapter), simulator="mock",
        task=None, effective_dt=adapter.dt,
    ))
    return DashboardServer(recording, mode="view")


def test_rows_are_the_terminals_own_aggregation(tmp_path):
    from specola.discovery_filter import DiscoveryQuery, filter_discovery
    from specola.discovery_selector import DiscoveryScope, signal_selections

    server = _server(tmp_path)
    view = filter_discovery(
        server._discovery.model, DiscoveryQuery(include_raw=False)
    )
    expected = signal_selections(view, DiscoveryScope("object"))

    reply = server._resolve_signals("r", {"entity_class": "object"})
    assert [row["pattern"] for row in reply["signals"]] == [
        selection.pattern for selection in expected
    ]


def test_a_row_carries_what_the_list_shows(tmp_path):
    """Signal, unit, frame and one entity's current value — the columns
    the list renders, so the page derives none of them.

    Coverage counts were dropped at the maintainer's request. They were
    the one fact in this pane that said a fan-out binds some entities
    rather than all; that now surfaces at check time instead, where the
    report names the bound keys.
    """
    server = _server(tmp_path)
    row = server._resolve_signals("r", {"entity_class": "object"})[
        "signals"][0]
    assert set(row) == {
        "pattern", "quantity", "component", "unit", "frame",
        "example_key", "example_value",
    }
    assert row["example_key"]


def test_a_glob_pattern_keeps_its_shared_suffix(tmp_path):
    """The exact rule the browser's version dropped."""
    from specola.discovery_selector import _glob_for_keys

    assert _glob_for_keys(["xpos_body_a_0", "xpos_body_b_0"]) == \
        "xpos_body_*_0"
    # A prefix-only condensation would have said ``xpos_body_*``.
    assert _glob_for_keys(["xpos_body_a_0", "xpos_body_b_0"]) != \
        "xpos_body_*"


def test_a_pattern_the_server_does_not_know_is_refused(tmp_path):
    """Never widened. Answering a narrower question with the broader
    scope's forms is how somebody copies a starter for a signal they did
    not select."""
    server = _server(tmp_path)
    reply = server._resolve_palette(
        "r", {"entity_class": "object", "signal": "xpos_body_*"}
    )
    assert reply["forms"] == []
    assert "no signal named" in reply["reason"]


def test_a_pattern_the_server_produced_resolves(tmp_path):
    """The round trip a page actually makes, by position."""
    server = _server(tmp_path)
    rows = server._resolve_signals("r", {"entity_class": "object"})
    pattern = rows["signals"][0]["pattern"]
    reply = server._resolve_palette("r", {
        "entity_class": "object", "signal": pattern, "signal_index": 0,
    })
    assert reply["forms"]
    assert reply["scope"]["signal"] == pattern


def test_an_unambiguous_pattern_still_resolves_without_a_position(tmp_path):
    """So a client that only knows the text keeps working."""
    server = _server(tmp_path)
    rows = server._resolve_signals("r", {"entity_class": "object"})
    pattern = rows["signals"][0]["pattern"]
    assert server._resolve_palette(
        "r", {"entity_class": "object", "signal": pattern}
    )["forms"]


def test_two_rows_with_the_same_text_are_refused_not_guessed(tmp_path):
    """``_glob_for_keys`` returns a bare ``*`` whenever prefix and suffix
    cover the shortest key, so one scope really can print two unrelated
    families identically. Answering about the first is how somebody gets
    a starter for a signal they did not click."""
    from specola.dashboard.server import _pick_signal
    from types import SimpleNamespace

    rows = [SimpleNamespace(pattern="*"), SimpleNamespace(pattern="*")]
    chosen, reason = _pick_signal(rows, "*", None)
    assert chosen is None
    assert "by position" in reason

    chosen, reason = _pick_signal(rows, "*", 1)
    assert chosen is rows[1]
    assert reason is None


def test_a_position_that_no_longer_names_that_row_is_refused(tmp_path):
    """A page whose scope moved under it must not be answered."""
    from specola.dashboard.server import _pick_signal
    from types import SimpleNamespace

    rows = [SimpleNamespace(pattern="mass")]
    chosen, reason = _pick_signal(rows, "temperature", 0)
    assert chosen is None
    assert "no longer at position" in reason


def test_a_role_only_scope_is_a_scope(tmp_path):
    """Role browse mode selects a role across classes, so the absence of
    a class is a question, not an error."""
    server = _server(tmp_path)
    scope = server._scope_from({"role": "target"})
    assert scope is not None
    assert scope.entity_class is None
    assert scope.role == "target"


def test_the_roleless_branch_maps_to_the_python_sentinel(tmp_path):
    """JSON cannot carry Python's sentinel, so a page sends a token that
    no real role name can collide with."""
    from specola.discovery_selector import ROLE_ABSENT

    server = _server(tmp_path)
    scope = server._scope_from({"role": DashboardServer.ROLELESS_TOKEN})
    assert scope.role is ROLE_ABSENT
    # The NUL is what makes it uncollidable with any real role name.
    assert "\x00" in DashboardServer.ROLELESS_TOKEN
    assert server._scope_from({"role": "roleless"}).role == "roleless"


def test_nothing_selected_is_still_nothing_selected(tmp_path):
    server = _server(tmp_path)
    assert server._scope_from({}) is None
    assert server._resolve_signals("r", {})["signals"] == []


def test_the_search_haystack_matches_pythons_field_for_field():
    """The page rebuilds ``entity_search_blob`` from the document rather
    than downloading it (measured +0.184 MiB gzipped on RoboCasa, a 42%
    transfer increase). That duplication is only acceptable while the two
    field lists agree, so this compares them directly rather than merely
    checking that the document carries what Python reads -- a weaker
    check would pass while the JavaScript silently stopped matching on a
    newly added field.
    """
    import re
    from pathlib import Path

    from specola.discovery_model import entity_search_blob

    source = inspect.getsource(entity_search_blob)
    renamed = {"entity_id": "id", "entity_class": "class"}
    # ``properties`` is the container both sides iterate, not a field
    # either of them matches on.
    python_entity = {
        renamed.get(name, name)
        for name in re.findall(r"entity\.(\w+)", source)
    } - {"properties"}
    python_prop = set(re.findall(r"prop\.(\w+)", source))

    script = (Path(__file__).resolve().parents[1] / "src" / "specola"
              / "dashboard" / "static" / "discover.js").read_text()
    body = script[script.index("function haystack("):]
    body = body[:body.index("\n  }")]
    js_entity = set(re.findall(r"entity\.(\w+)", body))
    js_prop = set(re.findall(r"property\.(\w+)", body))
    js_entity.discard("_blob")
    js_entity.discard("properties")

    assert js_entity == python_entity
    assert js_prop == python_prop


def test_every_haystack_field_is_in_the_document(tmp_path):
    """And the fields both sides read are actually serialized."""
    import gzip
    import json
    import re

    from specola.discovery_model import entity_search_blob

    server = _server(tmp_path)
    document = json.loads(gzip.decompress(server._discovery.raw_gzip))
    entity = document["entities"][0]
    prop = entity["properties"][0]

    source = inspect.getsource(entity_search_blob)
    renamed = {"entity_id": "id", "entity_class": "class"}
    for field in re.findall(r"entity\.(\w+)", source):
        assert renamed.get(field, field) in entity, field
    assert "properties" in entity
    for field in re.findall(r"prop\.(\w+)", source):
        assert field in prop, field
