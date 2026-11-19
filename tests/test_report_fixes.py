"""Regressions for the reviewed CSV, membership and page-layout defects."""

import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import specola
from specola.report import load
from specola.report_export import _text_pages, _trace_pages, export_report
from specola.results_io import save_rows
from specola.session import _validate_meta


def _fixture(directory, *, episode="modern", task="modern", legacy=False,
             score=0.30000000000000004, display_task=None, iteration=2**53 + 1):
    fixture = json.loads((Path(specola.__file__).parent / "data/report-fixture.json").read_text())
    rows = copy.deepcopy(fixture["rows"])
    metadata = copy.deepcopy(fixture["metadata"])
    caller_meta = _validate_meta({"score": score, "nan_score": float("nan"), "iteration": iteration})
    metadata.update(caller_meta, episode_id=episode)
    if legacy:
        metadata.pop("task_id")
        metadata.pop("task")
    else:
        metadata.update(task=task if display_task is None else display_task, task_id=task)
    for row in rows:
        row.update(caller_meta, episode_id=episode, task=task if display_task is None else display_task)
        if legacy:
            row.pop("task_id")
        else:
            row["task_id"] = task
    rows[0].update(time=0.12345678912345678, rho=0.30000000000000004)
    directory.mkdir(parents=True, exist_ok=True)
    path = save_rows(rows, directory / "episode.parquet", metadata, quiet=True)
    return path, rows


def _load_route(directory, path, route):
    if route == "parquet" and path.suffix != ".parquet":
        pytest.skip("Explicit Parquet regression requires optional PyArrow")
    if route == "directory":
        return load(str(directory))
    return load(str(path if route == "parquet" else path.with_suffix(".csv")))


@pytest.mark.parametrize("route", ["directory", "csv", "parquet"])
@pytest.mark.parametrize("score", [0.30000000000000004, float("nan"), float("inf"), -0.0])
def test_writer_loader_exact_floats_nan_and_large_integer_metadata(tmp_path, route, score):
    path, rows = _fixture(tmp_path, score=score)
    frame = _load_route(tmp_path, path, route)
    for column in ("time", "rho"):
        assert float(frame[column].iloc[0]).hex() == rows[0][column].hex()
    assert frame["nan_score"].isna().all()
    if math.isnan(score):
        assert frame["score"].isna().all()
    else:
        assert all(float(value).hex() == score.hex() for value in frame["score"])
    assert set(frame["iteration"]) == {2**53 + 1}
    assert str(frame["step"].dtype) in ("Int64", "int64")


@pytest.mark.parametrize("route", ["directory", "csv", "parquet"])
@pytest.mark.parametrize("iteration", [-2**63, -17, 0, 2**53 + 1, 2**63 - 1, 2**63 + 7, 2**64 - 1])
def test_integer_metadata_signed_and_unsigned_range(tmp_path, route, iteration):
    path, _ = _fixture(tmp_path, iteration=iteration)
    frame = _load_route(tmp_path, path, route)
    assert all(int(value) == iteration for value in frame["iteration"])
    if route != "parquet":
        assert all(isinstance(value, int) for value in frame["iteration"])


@pytest.mark.parametrize("route", ["directory", "csv", "parquet"])
def test_nullable_integer_cells_keep_missingness_and_values(tmp_path, route):
    path, rows = _fixture(tmp_path, iteration=-17)
    # save_rows supports missing row cells; None is not passed as caller
    # metadata to _validate_meta, whose scalar contract excludes it.
    rows[1]["iteration"] = None
    rows[1]["window_steps"] = None
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    path = save_rows(rows, path.with_suffix(".parquet"), metadata, quiet=True)
    frame = _load_route(tmp_path, path, route)
    assert frame["iteration"].isna().tolist() == [False, True, False, False]
    assert frame["window_steps"].isna().tolist() == [False, True, False, False]
    assert all(int(value) == -17 for value in frame["iteration"].dropna())
    assert all(int(value) == 2 for value in frame["window_steps"].dropna())
    if route != "parquet":
        assert all(isinstance(value, int) for value in frame["iteration"].dropna())


@pytest.mark.parametrize("column", ["rho", "score", "iteration", "step"])
def test_csv_rejects_malformed_numeric_values(tmp_path, column):
    path, _ = _fixture(tmp_path)
    path = path.with_suffix(".csv")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0][column] = "not-a-number"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        load(str(tmp_path))


@pytest.mark.parametrize("route", ["directory", "parquet"])
def test_report_plot_csv_preserves_saved_float_values(tmp_path, route):
    path, rows = _fixture(tmp_path / "raw")
    if route == "parquet" and path.suffix != ".parquet":
        pytest.skip("Explicit Parquet regression requires optional PyArrow")
    result = export_report(tmp_path / "raw", output_dir=tmp_path / "report",
                           paths=[path] if route == "parquet" else None)
    with (tmp_path / "report/report-plot-data.csv").open(newline="") as stream:
        plotted = list(csv.DictReader(stream))
    for column in ("time", "rho", "score"):
        assert float(plotted[0][column]).hex() == rows[0][column].hex()
    assert plotted[0]["nan_score"] in ("", "nan")
    assert pd.read_csv(tmp_path / "report/report-plot-data.csv")["nan_score"].isna().all()
    assert int(plotted[0]["iteration"]) == 2**53 + 1
    metrics = result["metrics"]
    assert "export_seconds" not in metrics
    assert metrics["report_build_seconds"] > 0
    assert "artifact hashing" in metrics["measurement_scope"]
    assert "publication" in metrics["measurement_scope"]


@pytest.mark.parametrize("route", ["directory", "csv", "parquet"])
def test_mixed_legacy_task_fallback_and_declared_population(tmp_path, route):
    modern, _ = _fixture(tmp_path / "raw/modern", task="modern")
    legacy, _ = _fixture(tmp_path / "raw/legacy", episode="legacy", task="legacy", legacy=True)
    legacy.with_suffix(".meta.json").unlink()
    if route == "parquet":
        if modern.suffix != ".parquet" or legacy.suffix != ".parquet":
            pytest.skip("Explicit Parquet regression requires optional PyArrow")
        result = export_report(tmp_path / "raw", output_dir=tmp_path / "report",
                               paths=[modern, legacy], batch_tasks=["modern", "legacy"])
    else:
        with pytest.warns(RuntimeWarning, match="Saved provenance unavailable"):
            result = export_report(tmp_path / "raw", output_dir=tmp_path / "report",
                                   paths=[modern.with_suffix(".csv"), legacy.with_suffix(".csv")] if route == "csv" else None,
                                   batch_tasks=["modern", "legacy"] if route == "csv" else None)
    assert result["task_count"] == 2
    assert {task for source in result["sources"] for task in source["task_ids"]} == {"modern", "legacy"}
    assert result["unavailable_task_identity_sources"] == 0
    summary = pd.read_csv(tmp_path / "report/report-task-spec-summary.csv")
    assert set(summary["task"]) == {"modern", "legacy"}
    assert summary["satisfaction_denominator"].sum() == 2
    assert summary["fail"].sum() == 2


@pytest.mark.parametrize("route", ["directory", "parquet"])
def test_explicitly_empty_modern_identity_stays_unavailable(tmp_path, route):
    path, _ = _fixture(tmp_path, task="", display_task="descriptive task name without an identity")
    frame = _load_route(tmp_path, path, route)
    assert frame.attrs["sources"][0]["task_ids"] == []
    assert frame.attrs["sources"][0]["task_identity_unavailable"] is True


class _CapturedPages:
    def __init__(self):
        self.text = []
        self.axes = []

    def savefig(self, figure, **kwargs):
        self.text.append([text.get_text() for text in figure.texts])
        for axis in figure.axes:
            renderer = figure.canvas.get_renderer()
            legend = axis.get_legend()
            self.axes.append({"xlim": axis.get_xlim(),
                              "legend_overlaps_panel": legend.get_window_extent(renderer).overlaps(axis.get_window_extent(renderer)) if legend else None})


def test_bounded_records_stay_together_and_long_records_repeat_identity():
    pdf = _CapturedPages()
    inventory = []
    records = ["Spec: bounded-{}\n".format(index) + "\n".join("value {}".format(line) for line in range(12)) for index in range(5)]
    _text_pages(pdf, inventory, "Records", records)
    for index in range(5):
        assert sum(any("Spec: bounded-{}".format(index) in text for text in page) for page in pdf.text) == 1
    assert all(not record["continued"] for page in inventory for record in page["records"])
    pdf = _CapturedPages()
    inventory = []
    _text_pages(pdf, inventory, "Long records", ["Spec: long-record\n" + "\n".join("formula part {}".format(line) for line in range(85))])
    assert len(inventory) >= 2
    assert all("Long records" in page[0] for page in pdf.text)
    for page, entry in zip(pdf.text[1:], inventory[1:]):
        assert any("Spec: long-record" in text and "continued" in text for text in page)
        assert entry["records"][0]["continued"] is True
        assert entry["records"][0]["first_page"] == 1


def test_trace_endpoints_and_legend_leave_data_visible(tmp_path):
    path, _ = _fixture(tmp_path)
    pdf = _CapturedPages()
    inventory = []
    frame = load(str(path))
    _trace_pages(pdf, inventory, frame)
    assert len(pdf.axes) == 1
    assert pdf.axes[0]["xlim"][0] < frame["time"].min()
    assert pdf.axes[0]["xlim"][1] > frame["time"].max()
    assert not pdf.axes[0]["legend_overlaps_panel"]
    assert inventory[-1]["trace"]["finite_pairs"] == 3


def test_never_bound_groups_keep_identity_and_reason_without_empty_plots(tmp_path):
    from types import SimpleNamespace
    from specola.session import Session

    path, rows = _fixture(tmp_path / "raw")
    session: Any = Session.__new__(Session)
    session.engine = SimpleNamespace(spec_names=["synthetic_margin", "never-bound-a", "never-bound-b", "skipped"])
    session._rows = rows
    session._next_step = 4
    session._skipped = {"skipped": "synthetic absent class"}
    session._episode_id = "synthetic"
    session._meta = {}
    full = Session._build_full_rows(session, status="truncated", success=None)
    markers = {row["spec"]: row for row in full if row["spec"] != "synthetic_margin"}
    assert markers["never-bound-a"]["step"] == markers["never-bound-b"]["step"] == 3
    assert markers["skipped"]["step"] == -1
    pdf = _CapturedPages()
    inventory = []
    traces = _trace_pages(pdf, inventory, pd.DataFrame(full))
    assert len(traces) == 3
    assert len(pdf.axes) == 1
    unavailable = [trace for trace in traces if trace["status"] == "unavailable"]
    assert {trace["spec"] for trace in unavailable} == {"never-bound-a", "never-bound-b"}
    assert all(trace["finite_pairs"] == 0 and trace["saved_rows"] == 1
               and trace["reason"] == "No finite robustness/time pairs available." for trace in unavailable)
    assert any(page["title"] == "Unavailable traces" for page in inventory)
    assert all(trace["spec"] != "skipped" for trace in traces)
