import json

import numpy as np
import pytest

from specola import state_recording as sr


def _writer(tmp_path, **kw):
    args = dict(simulator="mock", task=None, signature={"engine": "fake", "length": 3}, dt=0.05)
    args.update(kw)
    return sr.StateWriter(tmp_path / "ep0_states.npz", **args)


def test_round_trip_keeps_rows_and_header(tmp_path):
    w = _writer(tmp_path)
    buffer = np.array([1.0, 2.0, 3.0])
    w.append(buffer)
    buffer[:] = [4.0, 5.0, 6.0]      # the adapter reuses its buffer
    w.append(buffer)
    path = w.close("completed")
    assert not (tmp_path / "ep0_states.rows").exists()
    rec = sr.StateRecording.load(path)
    assert rec.header["format_version"] == sr.FORMAT_VERSION
    assert rec.header["steps"] == 2 and rec.header["status"] == "completed"
    assert rec.header["signature"] == {"engine": "fake", "length": 3}
    assert rec.states.shape == (2, 3) and rec.states.dtype == np.float64
    assert rec.states[0].tolist() == [1.0, 2.0, 3.0]
    assert sr.is_state_recording(path)
    sr.check_compatible(rec, simulator="mock", task=None, signature={"engine": "fake", "length": 3}, dt=0.05)


def test_writer_rejects_a_row_of_another_length(tmp_path):
    w = _writer(tmp_path)
    w.append(np.zeros(3))
    with pytest.raises(sr.StateRecordingError, match="length 2"):
        w.append(np.zeros(2))


def test_close_is_idempotent_and_records_partial_status(tmp_path):
    w = _writer(tmp_path)
    w.append(np.zeros(3))
    path = w.close("truncated")
    assert w.close("completed") == path
    assert sr.StateRecording.load(path).header["status"] == "truncated"


@pytest.mark.parametrize("field, value, message", [
    ("simulator", "libero", "simulator"),
    ("task", "other", "task"),
    ("signature", {"engine": "fake", "length": 3, "model": "b"}, "signature"),
    ("dt", 0.1, "pass --dt 0.05"),
])
def test_check_compatible_names_the_differing_field(tmp_path, field, value, message):
    w = _writer(tmp_path)
    w.append(np.zeros(3))
    rec = sr.StateRecording.load(w.close("completed"))
    expected = dict(simulator="mock", task=None, signature={"engine": "fake", "length": 3}, dt=0.05)
    expected[field] = value
    with pytest.raises(sr.StateRecordingError, match=message):
        sr.check_compatible(rec, **expected)


def test_load_rejects_an_empty_or_malformed_recording(tmp_path):
    empty = _writer(tmp_path).close("completed")
    with pytest.raises(sr.StateRecordingError, match="no rows"):
        sr.StateRecording.load(empty)
    def saved(name, header, states):
        target = tmp_path / name
        np.savez(target, header=json.dumps(header), states=states)
        return target

    good = {"format_version": 1, "simulator": "mock", "task": None,
            "signature": {"length": 3}, "dt": 0.05, "steps": 1, "status": "completed"}
    with pytest.raises(sr.StateRecordingError, match="4 wide, the signature says 3"):
        sr.StateRecording.load(saved("wide_states.npz", good, np.zeros((1, 4))))
    with pytest.raises(sr.StateRecordingError, match="not an object"):
        sr.StateRecording.load(saved("list_states.npz", [1, 2], np.zeros((1, 3))))
    with pytest.raises(sr.StateRecordingError, match="positive integer"):
        sr.StateRecording.load(saved("frac_states.npz", {**good, "signature": {"length": 3.9}}, np.zeros((1, 3))))
    with pytest.raises(sr.StateRecordingError, match="positive finite"):
        sr.StateRecording.load(saved("nan_states.npz", {**good, "dt": float("nan")}, np.zeros((1, 3))))
    with pytest.raises(sr.StateRecordingError, match="non-finite"):
        sr.StateRecording.load(saved("inf_states.npz", good, np.full((1, 3), np.inf)))


def test_robomimic_file_is_not_a_state_recording_and_a_bad_npz_is_an_error(tmp_path):
    other = tmp_path / "demo.hdf5"
    other.write_bytes(b"\x89HDF\r\n\x1a\n")
    assert not sr.is_state_recording(other)
    bad = tmp_path / "bad.npz"
    np.savez(bad, x=np.zeros(2))
    with pytest.raises(sr.StateRecordingError, match="not a state recording"):
        sr.is_state_recording(bad)


def test_base_adapter_has_no_state_tier():
    from specola.adapters.fake import _MockAdapter
    adapter = _MockAdapter()
    assert adapter.state_signature() is None
    assert adapter.capture_state() is None
    with pytest.raises(NotImplementedError, match="_MockAdapter"):
        adapter.restore_state(np.zeros(1))


def test_writer_refuses_a_signature_without_a_length(tmp_path):
    with pytest.raises(sr.StateRecordingError, match="positive integer 'length'"):
        sr.StateWriter(tmp_path / "x_states.npz", simulator="mock", task=None,
                       signature={"engine": "fake"}, dt=0.05)


def test_check_compatible_reports_identity_before_dt(tmp_path):
    w = _writer(tmp_path)
    w.append(np.zeros(3))
    rec = sr.StateRecording.load(w.close("completed"))
    with pytest.raises(sr.StateRecordingError, match="simulator"):
        sr.check_compatible(rec, simulator="libero", task=None,
                            signature={"engine": "fake", "length": 3}, dt=0.5)


def test_an_empty_npz_is_reported_not_raised(tmp_path):
    empty = tmp_path / "empty_states.npz"
    empty.write_bytes(b"")
    with pytest.raises(sr.StateRecordingError, match="not a state recording"):
        sr.is_state_recording(empty)
