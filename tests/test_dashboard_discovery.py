"""The browser gets the scene, and the scene stays the server's.

Two properties matter here. The document is transferred still
compressed, because it is 1.66 MiB gzipped and 55 MiB decoded on
RoboCasa, and decoding it in this process to re-encode it per request
would put megabytes through the same interpreter that is stepping a
simulator.

And the browser's selection is untrusted. It sends class and signal
names; those are resolved against the immutable Python model and handed
to the same ``selector_palette()`` the terminal calls. Nothing a browser
sends becomes a path, a formula, or an adapter access — and the two
surfaces cannot offer different selectors for one selection, because
there is only one implementation.
"""

from __future__ import annotations

import gzip
import json

import pytest

pytest.importorskip("flask_socketio")

from specola.capabilities import reset_adapter          # noqa: E402
from specola.cli_common import _MockAdapter             # noqa: E402
from specola.dashboard.recording import (               # noqa: E402
    RecordingWriter,
)
from specola.dashboard.server import DashboardServer    # noqa: E402
from specola.discovery_capture import discovery_payload  # noqa: E402


def _recording(tmp_path, *, with_discovery=True):
    writer = RecordingWriter(
        tmp_path, {"run_id": "r", "status": "completed"}
    )
    if with_discovery:
        adapter = _MockAdapter()
        writer.write_discovery(discovery_payload(
            adapter, reset_adapter(adapter), simulator="mock",
            task=None, effective_dt=adapter.dt,
        ))
    return tmp_path


def _server(tmp_path, **kwargs):
    return DashboardServer(_recording(tmp_path, **kwargs), mode="view")


def _client(server):
    return server.app.test_client()


def test_document_is_served_still_compressed(tmp_path):
    server = _server(tmp_path)
    response = _client(server).get(
        "/api/discovery?variant=raw", headers={"Host": "127.0.0.1:8090"}
    )
    assert response.status_code == 200
    assert response.headers["Content-Encoding"] == "gzip"
    body = response.get_data()
    assert body[:2] == b"\x1f\x8b"                  # gzip magic
    document = json.loads(gzip.decompress(body))
    assert document["filters"]["include_raw"] is True


def test_the_digest_identifies_which_episode_the_bytes_are_from(tmp_path):
    server = _server(tmp_path)
    response = _client(server).get(
        "/api/discovery", headers={"Host": "127.0.0.1:8090"}
    )
    import hashlib

    header = response.headers["X-Specola-Discovery-Digest"]
    assert header == server._discovery.digest
    assert header == hashlib.sha256(server._discovery.raw_gzip).hexdigest()


def test_semantic_variant_drops_raw_scene_entities(tmp_path):
    server = _server(tmp_path)
    client = _client(server)
    semantic = json.loads(gzip.decompress(client.get(
        "/api/discovery?variant=semantic",
        headers={"Host": "127.0.0.1:8090"},
    ).get_data()))
    raw = json.loads(gzip.decompress(client.get(
        "/api/discovery?variant=raw", headers={"Host": "127.0.0.1:8090"},
    ).get_data()))
    assert semantic["filters"]["include_raw"] is False
    assert len(semantic["entities"]) <= len(raw["entities"])


def test_an_unknown_variant_is_refused(tmp_path):
    response = _client(_server(tmp_path)).get(
        "/api/discovery?variant=../../etc/passwd",
        headers={"Host": "127.0.0.1:8090"},
    )
    assert response.status_code == 400


def test_a_recording_without_a_snapshot_says_so(tmp_path):
    server = _server(tmp_path, with_discovery=False)
    response = _client(server).get(
        "/api/discovery", headers={"Host": "127.0.0.1:8090"}
    )
    assert response.status_code == 404
    assert server._discovery is None
    assert server._discovery_ready()["available"] is False
    assert server._discovery_ready()["reason"]


def test_the_document_endpoint_is_behind_the_loopback_guard(tmp_path):
    """It carries the whole scene, so it is not an exception."""
    response = _client(_server(tmp_path)).get(
        "/api/discovery", headers={"Host": "evil.example:8090"}
    )
    assert response.status_code == 403


def test_palette_matches_the_terminals_answer_for_one_selection(tmp_path):
    """One implementation, so the surfaces cannot disagree."""
    from specola.discovery_filter import DiscoveryQuery, filter_discovery
    from specola.discovery_selector import DiscoveryScope, signal_selections
    from specola.selector_palette import selector_palette

    server = _server(tmp_path)
    model = server._discovery.model
    view = filter_discovery(model, DiscoveryQuery(include_raw=False))
    scope = DiscoveryScope("object")
    signal = signal_selections(view, scope)[0]
    expected = [
        form.reference
        for form in selector_palette(view, scope, signal=signal)
    ]

    reply = server._resolve_palette(
        "req-1",
        {"entity_class": "object", "signal": signal.pattern},
    )
    assert reply["request_id"] == "req-1"
    assert [form["reference"] for form in reply["forms"]] == expected


def test_palette_rows_carry_no_counts(tmp_path):
    server = _server(tmp_path)
    reply = server._resolve_palette("r", {"entity_class": "object"})
    assert reply["forms"]
    for form in reply["forms"]:
        assert "bindings" not in form
        assert "monitors" not in form


def test_every_offered_form_ships_a_compilable_starter(tmp_path):
    """The browser must never assemble YAML itself."""
    import yaml

    server = _server(tmp_path)
    reply = server._resolve_palette("r", {"entity_class": "object"})
    offered = [form for form in reply["forms"] if form["available"]]
    assert offered
    for form in offered:
        starter = form["starter"]
        assert set(starter) <= {
            "name", "formula", "constants", "backend", "symmetric",
        }
        assert "window" not in starter
        assert "dist(" not in starter["formula"]
        assert "norm(" not in starter["formula"]
        # Round-trips through YAML, which is how it will reach a file.
        assert yaml.safe_load(yaml.safe_dump(starter)) == starter


def test_an_unknown_class_from_the_browser_returns_a_reason(tmp_path):
    """Browser input is looked up, never trusted."""
    server = _server(tmp_path)
    reply = server._resolve_palette(
        "r", {"entity_class": "../../etc/passwd"}
    )
    assert reply["forms"] == []
    assert reply["reason"]

    empty = server._resolve_palette("r", {})
    assert empty["forms"] == []
    assert "no class or role selected" in empty["reason"]


def test_a_stale_request_id_comes_back_untouched(tmp_path):
    """So a client can discard a reply it has already moved past."""
    server = _server(tmp_path)
    reply = server._resolve_palette(
        "stale-42", {"entity_class": "object"}
    )
    assert reply["request_id"] == "stale-42"


def test_publishing_a_snapshot_does_not_disturb_replay(tmp_path):
    """A bad payload costs Discover and nothing else."""
    server = _server(tmp_path, with_discovery=False)
    server.set_discovery_snapshot(b"not a gzip document")
    assert server._discovery is None
    # The recording still reads.
    assert server.reader.manifest["run_id"] == "r"
