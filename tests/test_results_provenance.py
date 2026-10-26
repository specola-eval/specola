"""
Results must say which spec set produced them.

A results file and a spec file are otherwise two artifacts paired on
trust. The digest recorded alongside results covers parsed spec content,
so it survives reformatting, moves when a threshold moves, and can be
recomputed from any candidate spec file.

Two contracts are load-bearing and easy to break by accident:

- ``load()`` must not validate constant values. That is deferred to
  ``start_episode``, where the error names the offending constant.
  Computing provenance must not quietly move that check earlier.
- Writing provenance must never cost a results file. The rows are on
  disk before the metadata is, so a metadata failure is reported, not
  raised.
"""

from __future__ import annotations

import json
import sys

import pytest

from specola.engine import MonitorEngine
from specola.results_io import PROVENANCE_KEYS, RESERVED_META_KEYS, save_rows
from specola.session import _validate_meta

SPEC = """specs:
- name: settles
  formula: "G[0,5](object_*[xpos_a] - limit >= 0)"
  constants: {limit: 0.5}
"""


def _write(tmp_path, text, name="specs.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_digest_is_stable_across_loads(tmp_path):
    path = _write(tmp_path, SPEC)
    first = MonitorEngine.load(path).spec_provenance
    second = MonitorEngine.load(path).spec_provenance
    assert first["spec_digest"] == second["spec_digest"]
    assert first["spec_source"] == path


def test_digest_ignores_comments_and_reformatting(tmp_path):
    plain = MonitorEngine.load(_write(tmp_path, SPEC, "a.yaml"))
    annotated = MonitorEngine.load(_write(
        tmp_path,
        "# provenance note\n" + SPEC + "\n# trailing comment\n",
        "b.yaml",
    ))
    assert (
        plain.spec_provenance["spec_digest"]
        == annotated.spec_provenance["spec_digest"]
    )


def test_digest_moves_when_a_threshold_moves(tmp_path):
    before = MonitorEngine.load(_write(tmp_path, SPEC, "a.yaml"))
    after = MonitorEngine.load(_write(
        tmp_path, SPEC.replace("0.5", "0.6"), "b.yaml",
    ))
    assert (
        before.spec_provenance["spec_digest"]
        != after.spec_provenance["spec_digest"]
    )


def test_constants_are_recorded_as_authored(tmp_path):
    engine = MonitorEngine.load(_write(tmp_path, SPEC))
    recorded = json.loads(engine.spec_provenance["spec_constants"])
    assert recorded == {"settles": {"limit": 0.5}}


@pytest.mark.parametrize("value", ['"close"', "[1, 2, 3]", "true", "null"])
def test_load_does_not_validate_constant_values(tmp_path, value):
    """A bad constant is start_episode's error to raise, not load's.

    Computing the digest must not coerce values: doing so rejected specs
    at load that the loader deliberately accepts, and turned the
    friendly "must be a real scalar" message into a raw ValueError from
    inside provenance computation.
    """
    path = _write(tmp_path, SPEC.replace("0.5", value))
    engine = MonitorEngine.load(path)
    assert engine.spec_provenance["spec_digest"]


def test_boolean_constant_is_not_recorded_as_one(tmp_path):
    """``float(True)`` is 1.0, and a digest must not say so.

    ``validate_constants`` rejects booleans outright, so a spec with one
    never runs. Recording it as 1.0 would give a never-run spec the same
    provenance as a real threshold of 1.0.
    """
    boolean = MonitorEngine.load(_write(tmp_path, SPEC.replace("0.5", "true"), "a.yaml"))
    numeric = MonitorEngine.load(_write(tmp_path, SPEC.replace("0.5", "1.0"), "b.yaml"))
    assert (
        boolean.spec_provenance["spec_digest"]
        != numeric.spec_provenance["spec_digest"]
    )
    assert json.loads(boolean.spec_provenance["spec_constants"]) == {
        "settles": {"limit": True}
    }


def test_provenance_is_empty_without_a_spec_file():
    assert MonitorEngine([]).spec_provenance == {}


def test_provenance_survives_construction_without_init():
    """Tests build engines with __new__ to drive a hand-made manager.

    Reading provenance off one of those must report "none" rather than
    raising while a results file is being written.
    """
    engine = MonitorEngine.__new__(MonitorEngine)
    assert engine.spec_provenance == {}


def test_provenance_keys_cannot_be_shadowed_by_caller_meta():
    assert PROVENANCE_KEYS <= RESERVED_META_KEYS
    for key in sorted(PROVENANCE_KEYS):
        with pytest.raises(ValueError):
            _validate_meta({key: "forged"})


def test_csv_fallback_writes_provenance_to_a_sidecar(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    written = save_rows(
        [{"step": 0, "rho": 1.0}],
        tmp_path / "ep0.parquet",
        {"episode_id": "e0", "spec_digest": "abc123"},
        quiet=True,
    )
    assert written == tmp_path / "ep0.csv"
    sidecar = json.loads((tmp_path / "ep0.meta.json").read_text())
    assert sidecar["spec_digest"] == "abc123"


def test_sidecar_renders_metadata_the_way_parquet_does(tmp_path, monkeypatch):
    """Same metadata must read back the same with or without pyarrow."""
    monkeypatch.setitem(sys.modules, "pandas", None)
    save_rows(
        [{"step": 0, "rho": 1.0}],
        tmp_path / "ep0.parquet",
        {"success": True, "steps": 150},
        quiet=True,
    )
    sidecar = json.loads((tmp_path / "ep0.meta.json").read_text())
    assert sidecar["success"] == "True"
    assert sidecar["steps"] == "150"


def test_a_failed_sidecar_does_not_cost_the_results(tmp_path, monkeypatch, capsys):
    """The rows are already on disk when the sidecar is written.

    Raising here would have session.end() drop the record's path, leaving
    a real results file nobody holds a reference to.
    """
    monkeypatch.setitem(sys.modules, "pandas", None)
    (tmp_path / "ep0.meta.json").mkdir()  # open() for write fails on a directory

    written = save_rows(
        [{"step": 0, "rho": 1.0}],
        tmp_path / "ep0.parquet",
        {"spec_digest": "abc123"},
        quiet=True,
    )

    assert written == tmp_path / "ep0.csv"
    assert written.exists()
    assert "provenance sidecar failed" in capsys.readouterr().out
