"""Validate discovery documents before ordinary recording publication."""

import gzip
import json
import queue
import threading

import pytest

from specola.dashboard.recording import DISCOVERY_FILE, RecordingWriter
from specola.discovery_capture import discovery_payload
from specola.discovery_restore import DiscoveryRestoreError


def _payload():
    from test_extraction_observation import _StructuredAdapter
    from specola.capabilities import reset_adapter

    adapter = _StructuredAdapter()
    return discovery_payload(adapter, reset_adapter(adapter), simulator="fixture",
                             task=None, effective_dt=adapter.dt)


def _invalid(payload, mutation):
    document = json.loads(gzip.decompress(payload))
    if mutation == "nested_candidate":
        document["extraction"]["verification_only"] = True
    elif mutation == "malformed":
        del document["schema_version"]
    else:
        document[mutation] = True
    return gzip.compress(json.dumps(document).encode())


@pytest.mark.parametrize("mutation", ["verification_only", "audit_only", "schema_only",
                                       "nested_candidate", "malformed"])
def test_writer_rejects_ineligible_sidecar_before_creating_any_payload_file(tmp_path, mutation):
    writer = RecordingWriter(tmp_path, {"run_id": "fixture"})
    manifest = (tmp_path / "manifest.json").read_bytes()
    try:
        with pytest.raises(DiscoveryRestoreError):
            writer.write_discovery(_invalid(_payload(), mutation))
        assert not (tmp_path / DISCOVERY_FILE).exists()
        assert not (tmp_path / (DISCOVERY_FILE + ".tmp")).exists()
        assert (tmp_path / "manifest.json").read_bytes() == manifest
        assert "discovery" not in writer.manifest
    finally:
        writer._fh.close()


def test_invalid_sidecar_cannot_replace_an_existing_ordinary_document(tmp_path):
    writer = RecordingWriter(tmp_path, {"run_id": "fixture"})
    payload = _payload()
    try:
        entry = writer.write_discovery(payload)
        with pytest.raises(DiscoveryRestoreError):
            writer.write_discovery(_invalid(payload, "nested_candidate"))
        assert (tmp_path / DISCOVERY_FILE).read_bytes() == payload
        assert writer.manifest["discovery"] == entry
    finally:
        writer._fh.close()


def _sink(tmp_path):
    from specola.dashboard.recorder import WebRecorderSink

    return WebRecorderSink(tmp_path, run_id="fixture", simulator="fixture",
                           task="fixture", mode="live", cameras=(), video="jpeg")


@pytest.mark.parametrize("mutation", ["audit_only", "nested_candidate"])
def test_sink_refuses_ineligible_payload_before_queueing_or_notifying(tmp_path, mutation):
    sink = _sink(tmp_path)
    notifications = []
    sink._on_discovery = notifications.append
    try:
        with pytest.raises(DiscoveryRestoreError):
            sink.write_discovery(_invalid(_payload(), mutation))
    finally:
        sink.close()
    assert notifications == []
    assert not (sink.recording_dir / DISCOVERY_FILE).exists()


def test_sink_validates_once_on_the_caller_and_writer_preserves_the_exact_bytes(tmp_path, monkeypatch):
    from specola import discovery_restore

    original = discovery_restore.discovery_model_from_document
    threads = []

    def traced(document):
        threads.append(threading.get_ident())
        return original(document)

    monkeypatch.setattr(discovery_restore, "discovery_model_from_document", traced)
    sink = _sink(tmp_path)
    payload = _payload()
    notifications = []
    sink._on_discovery = notifications.append
    try:
        sink.write_discovery(payload)
    finally:
        sink.close()
    assert threads == [threading.get_ident()]
    assert notifications == [payload]
    assert (sink.recording_dir / DISCOVERY_FILE).read_bytes() == payload


def test_queue_item_detaches_mutable_input_before_the_caller_can_change_it():
    from specola.dashboard.recorder import _DiscoveryItem

    payload = _payload()
    mutable = bytearray(payload)
    item = _DiscoveryItem(mutable)
    mutable[:] = b"changed after queue preparation"
    assert type(item.payload) is bytes
    assert item.payload == payload


def test_full_queue_reports_a_warning_after_discovery_validation(capsys, monkeypatch):
    from specola.dashboard.recorder import WebRecorderSink

    def full(*args, **kwargs):
        raise queue.Full

    sink = object.__new__(WebRecorderSink)
    sink.active = True
    sink._queue = queue.Queue()
    monkeypatch.setattr(sink._queue, "put", full)
    sink.write_discovery(_payload())
    assert "discovery sidecar dropped: writer queue stalled" in capsys.readouterr().err


def test_failed_discovery_callback_does_not_stop_the_recording_worker(tmp_path, capsys):
    sink = _sink(tmp_path)

    def fail(_payload):
        raise RuntimeError("intentional publication failure")

    sink._on_discovery = fail
    try:
        sink.write_discovery(_payload())
        sink._queue.put(({"step": 0}, {}, None, None))
    finally:
        sink.close()
    assert "intentional publication failure" in capsys.readouterr().err
    from specola.dashboard.recording import RecordingReader
    reader = RecordingReader(sink.recording_dir)
    assert reader.read_step(0)["step"] == 0
