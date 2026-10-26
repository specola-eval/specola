"""Protocol: envelope fields, run_meta, decimation min/max."""

from typing import Any, Dict

from specola.dashboard.protocol import (
    PROTOCOL_VERSION,
    Bundle,
    decimate,
    run_meta,
)


def _bundle(**kwargs):
    # Annotated: the values are deliberately heterogeneous, and without
    # it the unpack below reports one type error per Bundle field.
    defaults: Dict[str, Any] = dict(
        run_id="r", epoch=2, seq=5,
        line={"step": 7, "sim_time": 0.1, "wall_time": 1.0,
              "specs": [], "transitions": []},
        frames={}, max_step=9, live=True,
        drops={"frames": 0, "delivery": 0})
    defaults.update(kwargs)
    return Bundle(**defaults)


def test_bundle_wire_has_envelope():
    wire = _bundle().to_wire()
    assert (wire["protocol_version"], wire["epoch"], wire["seq"]) == (
        PROTOCOL_VERSION, 2, 5)
    assert wire["step"] == 7 and wire["max_step"] == 9
    assert wire["new_specs"] == [] and wire["new_bindings"] == []


def test_bundle_wire_passes_frames_and_deltas_through():
    line = {"step": 0, "sim_time": 0.0, "wall_time": 1.0,
            "specs": [{"name": "s"}], "transitions": [{"step": 0}],
            "new_specs": [{"name": "s"}], "new_bindings": [{"key": "k"}]}
    wire = _bundle(line=line,
                   frames={"cam": {"bytes": b"jpeg", "step": 0}}).to_wire()
    assert wire["frames"]["cam"]["bytes"] == b"jpeg"
    assert wire["new_bindings"] == [{"key": "k"}]


def test_bundle_wire_omits_poses_when_absent():
    assert "poses" not in _bundle().to_wire()


def test_bundle_wire_carries_poses_when_present():
    bundle = _bundle(poses={"bytes": b"\x00" * 28, "step": 4})
    wire = bundle.to_wire()
    assert wire["poses"]["step"] == 4 and len(wire["poses"]["bytes"]) == 28


def test_run_meta_envelope():
    meta = run_meta({"run_id": "r1"}, mode="view", specs=[], bindings=[],
                    max_step=3, view_command="specola view d")
    assert meta["protocol_version"] == PROTOCOL_VERSION
    assert meta["run_id"] == "r1" and meta["mode"] == "view"
    assert meta["max_step"] == 3


def test_decimate_min_max_per_column():
    steps = list(range(100))
    values = [float(i % 10) for i in steps]
    out = decimate(steps, values, 0, 99, px=10)
    assert len(out) == 10
    assert out[0][1] == 0.0 and out[0][2] == 9.0    # min and max survive
    assert [entry[0] for entry in out] == [i * 10 for i in range(10)]


def test_decimate_small_window_returns_raw():
    out = decimate([0, 1, 2], [1.0, None, 3.0], 0, 2, px=50)
    assert out == [[0, 1.0, 1.0], [2, 3.0, 3.0]]


def test_decimate_clamps_to_window():
    # Window [1, 2] with one pixel column: both in-window points fold
    # into a single [first_step, min, max] entry; out-of-window dropped.
    out = decimate([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0], 1, 2, px=1)
    assert out == [[1, 2.0, 3.0]]


def test_decimate_empty_and_degenerate():
    assert decimate([], [], 0, 10, px=5) == []
    assert decimate([0], [1.0], 5, 4, px=5) == []
    assert decimate([0], [1.0], 0, 0, px=0) == []
