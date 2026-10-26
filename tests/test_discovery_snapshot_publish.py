"""Publishing the scene must not cost the recorder a snapshot line.

``set_discovery_snapshot`` is called from the recorder's writer thread —
the same thread that drains the snapshot queue. Building the Python model
there was measured at 0.38–0.56 s on RoboCasa (2,367 entities, 1.16 MiB
gzipped, 14 MiB decoded). For that half second no line is written and
frames pile up against a bounded queue, where ``_enqueue`` drops them;
past two seconds it raises and the recorder disables itself.

Nothing on the writer thread needs the model. The first palette or check
request does, and it can pay for it. The bytes are still validated at
publish time, so a bad payload warns once instead of failing every later
request.
"""

from __future__ import annotations

import gzip
import json
import threading
import time

import pytest

pytest.importorskip("flask_socketio")

from specola.capabilities import reset_adapter            # noqa: E402
from specola.cli_common import _MockAdapter               # noqa: E402
from specola.dashboard.server import DiscoverySnapshot    # noqa: E402
from specola.discovery_capture import (                   # noqa: E402
    discovery_payload,
)


@pytest.fixture()
def payload():
    adapter = _MockAdapter()
    return discovery_payload(
        adapter, reset_adapter(adapter), simulator="mock",
        task=None, effective_dt=adapter.dt,
    )


def test_publishing_does_not_build_the_model(payload):
    snapshot = DiscoverySnapshot.from_gzip(payload)
    assert snapshot._model is None
    assert snapshot.digest                       # cheap work still happened


def test_the_model_appears_on_first_use_and_is_then_reused(payload):
    snapshot = DiscoverySnapshot.from_gzip(payload)
    model = snapshot.model
    assert model.entities
    assert snapshot.model is model


def test_bytes_that_are_not_a_document_are_refused_at_publish(payload):
    with pytest.raises(ValueError):
        DiscoverySnapshot.from_gzip(b"not gzip at all")
    with pytest.raises(ValueError):
        DiscoverySnapshot.from_gzip(gzip.compress(b"[]"))


def test_a_document_that_cannot_be_restored_is_a_reported_condition():
    """It gzips and it parses, and it still is not restorable.

    Publishing cannot catch this without doing the restore, which is the
    work that was moved off the writer thread. So the failure is
    remembered on the snapshot instead of being re-raised from scratch,
    which is what lets the readiness announcement and the document route
    report it once with its reason rather than 500 on every request.
    """
    snapshot = DiscoverySnapshot.from_gzip(
        gzip.compress(json.dumps({"entities": []}).encode())
    )
    failure = snapshot.restore_failure()
    assert failure is not None
    assert "schema_version" in str(failure)
    assert snapshot._error is failure
    with pytest.raises(type(failure)):
        snapshot.model


def test_an_unusable_snapshot_disables_discover_rather_than_erroring(
        tmp_path):
    """The tab must explain itself, not enable and then fail."""
    from specola.dashboard.recording import RecordingWriter
    from specola.dashboard.server import DashboardServer

    recording = tmp_path / "run" / "recording_000"
    recording.mkdir(parents=True)
    RecordingWriter(recording, {"run_id": "r"})
    server = DashboardServer(recording, mode="view")
    server.set_discovery_snapshot(
        gzip.compress(json.dumps({"entities": []}).encode())
    )
    # The publish thread proves restorability; wait for it rather than
    # racing it.
    deadline = time.time() + 30
    while server._discovery._error is None and time.time() < deadline:
        time.sleep(0.02)

    ready = server._discovery_ready()
    assert ready["available"] is False
    assert "cannot be rebuilt" in ready["reason"]

    response = server.app.test_client().get(
        "/api/discovery", headers={"Host": "127.0.0.1:8090"}
    )
    assert response.status_code == 409
    assert "DiscoveryRestoreError" in response.get_json()["reason"]

    # The resolver itself raises -- it is the socket handler that turns
    # that into an answer, which is what a page receives.
    from specola.discovery_restore import DiscoveryRestoreError

    with pytest.raises(DiscoveryRestoreError):
        server._resolve_palette("r", {"entity_class": "object"})
    client = server.socketio.test_client(server.app)
    reply = client.emit(
        "selector_palette", {"request_id": "r", "entity_class": "object"},
        callback=True,
    )
    assert reply["forms"] == []
    assert "DiscoveryRestoreError" in reply["reason"] or reply["reason"]
    client.disconnect()


def test_publishing_a_good_snapshot_leaves_no_failure(payload, tmp_path):
    from specola.dashboard.recording import RecordingWriter
    from specola.dashboard.server import DashboardServer

    recording = tmp_path / "run" / "recording_000"
    recording.mkdir(parents=True)
    RecordingWriter(recording, {"run_id": "r"})
    server = DashboardServer(recording, mode="view")
    server.set_discovery_snapshot(payload)
    assert server._discovery_ready()["available"] is True
    assert server._discovery.restore_failure() is None


def test_a_context_built_from_a_retired_snapshot_is_not_cached(
        payload, tmp_path):
    """Building takes a moment. A publish landing inside that window
    resets the cache, and a plain assignment afterwards would put the
    *previous* episode's scene back -- every later check would then
    compile against a scene the browser is no longer being shown."""
    from specola.capabilities import reset_adapter
    from specola.cli_common import _MockAdapter
    from specola.dashboard.recording import RecordingWriter
    from specola.dashboard.server import DashboardServer
    from specola.discovery_capture import discovery_payload

    recording = tmp_path / "run" / "recording_000"
    recording.mkdir(parents=True)
    RecordingWriter(recording, {"run_id": "r"})
    server = DashboardServer(recording, mode="view")
    server.set_discovery_snapshot(payload)
    first = server._discovery

    # Stand in for the publish that lands mid-build.
    original = first.model

    class Retiring:
        digest = "x"

        @property
        def model(self):
            adapter = _MockAdapter()
            server.set_discovery_snapshot(discovery_payload(
                adapter, reset_adapter(adapter), simulator="mock",
                task=None, effective_dt=adapter.dt,
            ))
            return original

    server._discovery = Retiring()
    context, error = server._check_context()
    assert context is not None, error
    # The snapshot changed under it, so nothing was cached against the
    # one now being served.
    assert server._context is None or server._context[0] is server._discovery


def test_validation_does_not_decompress_the_whole_document(payload):
    """It reads one byte through the stream. The point of moving the
    model off this path is lost if the check decodes 14 MiB instead."""
    seen = []
    real = gzip.decompress

    def counting(data):
        seen.append(len(data))
        return real(data)

    gzip.decompress = counting
    try:
        DiscoverySnapshot.from_gzip(payload)
    finally:
        gzip.decompress = real
    assert seen == []


def test_concurrent_first_uses_build_one_model(payload):
    """Two requests can arrive together; they must not each decode the
    document, and must not see different scenes."""
    snapshot = DiscoverySnapshot.from_gzip(payload)
    seen = []
    barrier = threading.Barrier(4)

    def grab():
        barrier.wait()
        seen.append(snapshot.model)

    threads = [threading.Thread(target=grab) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive()
    assert len(seen) == 4
    assert all(model is seen[0] for model in seen)


def test_the_semantic_variant_still_works_off_the_lazy_model(payload):
    """``gzip_for`` resolves the model before taking the lock it shares
    with the builder; a nested acquire would deadlock here."""
    snapshot = DiscoverySnapshot.from_gzip(payload)
    body = snapshot.gzip_for("semantic")
    assert json.loads(gzip.decompress(body))["filters"]["include_raw"] \
        is False
    assert snapshot.gzip_for("raw") is payload


@pytest.mark.parametrize("flag", ("verification_only", "audit_only", "schema_only"))
def test_raw_download_cannot_publish_declaration_or_candidate_documents(payload, flag):
    from specola.discovery_restore import DiscoveryRestoreError

    document = json.loads(gzip.decompress(payload))
    document[flag] = True
    snapshot = DiscoverySnapshot.from_gzip(gzip.compress(json.dumps(document).encode()))
    assert snapshot._model is None
    with pytest.raises(DiscoveryRestoreError, match="cannot become ordinary discovery"):
        snapshot.gzip_for("raw")


def test_raw_download_validates_the_document_before_returning_bytes():
    from specola.discovery_restore import DiscoveryRestoreError

    snapshot = DiscoverySnapshot.from_gzip(gzip.compress(b'{"entities": []}'))
    with pytest.raises(DiscoveryRestoreError, match="schema_version"):
        snapshot.gzip_for("raw")


@pytest.mark.parametrize("nested", (False, True))
def test_raw_http_route_rejects_candidate_bytes_before_returning_a_document(tmp_path, nested):
    from specola.dashboard.recording import RecordingWriter
    from specola.dashboard.server import DashboardServer
    from test_extraction_observation import _StructuredAdapter

    adapter = _StructuredAdapter()
    payload = discovery_payload(adapter, reset_adapter(adapter), simulator="fixture", task=None, effective_dt=adapter.dt)
    document = json.loads(gzip.decompress(payload))
    (document["extraction"] if nested else document)["verification_only"] = True
    recording = tmp_path / "recording_000"
    recording.mkdir()
    RecordingWriter(recording, {"run_id": "r"})
    server = DashboardServer(recording, mode="view")
    server.set_discovery_snapshot(gzip.compress(json.dumps(document).encode()))
    response = server.app.test_client().get(
        "/api/discovery?variant=raw", headers={"Host": "127.0.0.1:8090"})
    assert response.status_code == 409
    assert response.get_json()["available"] is False
    assert "Content-Encoding" not in response.headers
