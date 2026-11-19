"""Generate and render bounded synthetic offline reports for visual review.

Run this script in the installed package's environment. Poppler tools belong
to this acceptance path; ordinary report export does not require them.
"""

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from PIL import Image

import specola
from specola.results_io import save_rows
from specola.report import load, episode_table
from specola.report_export import export_report, report_preflight
from specola.dashboard.recording import RecordingWriter


def generate(output):
    fixture_path = Path(specola.__file__).parent / "data" / "report-fixture.json"
    fixture = json.loads(fixture_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    print("Synthetic fixture:", fixture_path, fixture["purpose"], flush=True)
    print("Readiness:", report_preflight(output), flush=True)
    members = []
    tasks = []
    long_name = "long_spec_" + "unbroken_identifier_" * 12
    for index, status in enumerate(("completed", "truncated", "failed")):
        task = "synthetic-task-{}".format(index)
        tasks.append(task)
        episode = "synthetic-episode-{:03d}".format(index)
        rows = copy.deepcopy(fixture["rows"])
        metadata = copy.deepcopy(fixture["metadata"])
        metadata.update(episode_id=episode, status=status, task=task, task_id=task,
                        run_id="synthetic-report-fixture")
        long_binding = "right_edge_annotation_" * 10
        for row in rows:
            row.update(episode_id=episode, status=status, task=task, task_id=task,
                       binding=long_binding)
            if index == 0 and row["verdict_final"]:
                row["rho"] = 0.5
            if index == 2:
                row["verdict_final"] = False
        # Use the production builder: never-bound specs carry the last real
        # step, while skipped specs carry -1. Do not imitate those branches.
        from types import SimpleNamespace
        from specola.session import Session
        session: Any = Session.__new__(Session)
        session.engine = SimpleNamespace(spec_names=[
            "synthetic_margin", "synthetic-missing", "synthetic-skipped", long_name,
        ])
        session._rows = rows
        session._next_step = metadata["steps"]
        session._skipped = {"synthetic-skipped": "synthetic absent class"}
        session._episode_id = episode
        session._meta = {"task": task, "task_id": task, "seed": 7}
        rows = Session._build_full_rows(session, status=status, success=None)
        formulas = json.loads(metadata["spec_formulas"])
        formulas.extend([
            {"name": "synthetic-missing", "formula": "G[0,1s] ({missing.mass} > 0)", "constants": {}},
            {"name": "synthetic-skipped", "formula": "G[0,1s] ({absent.mass} > 0)", "constants": {}},
            {"name": long_name, "formula": "G[0,1s] (" + "literal_very_long_signal_name_" * 22 + " ≥ ρ) ∧ ¬ false $literal$", "constants": {"limit": 0.25}},
        ])
        metadata["spec_formulas"] = json.dumps(formulas, ensure_ascii=False)
        directory = output / "dense" / "raw" / task
        directory.mkdir(parents=True, exist_ok=True)
        members.append(save_rows(rows, directory / (episode + ".parquet"), metadata, quiet=True))
    recording = output / "dense" / "synthetic-recording"
    writer = RecordingWriter(recording, {"run_id": "synthetic-report-fixture", "task": tasks[0],
                                         "fixture_origin": "Synthetic color grid; no simulator image"})
    pixels = np.zeros((180, 320, 3), dtype="uint8")
    pixels[:90, :160] = (230, 40, 40)
    pixels[:90, 160:] = (40, 180, 40)
    pixels[90:, :160] = (40, 80, 230)
    pixels[90:, 160:] = (230, 200, 40)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG")
    writer.write_frame("synthetic-color-grid-camera", 3, buffer.getvalue())
    writer.finalize("completed")
    batch = {"members": [str(path.resolve()) for path in members], "tasks": tasks}
    (output / "dense" / "batch.json").write_text(json.dumps(batch, indent=2))
    dense = export_report(output / "dense" / "raw", output_dir=output / "dense" / "report",
                          paths=members, batch_tasks=tasks,
                          recordings={str(members[0]): str(recording)})
    outcomes = episode_table(load(str(output / "dense" / "raw"), paths=members))
    assert outcomes["verdict"].value_counts().to_dict() == {
        "undetermined": 7, "not_applicable": 3, "pass": 1, "fail": 1,
    }
    assert dense["episode_count"] == 3 and dense["task_count"] == 3
    never_bound = outcomes[outcomes["spec"].isin(["synthetic-missing", long_name])]
    assert len(never_bound) == 6 and (never_bound["steps"] == 4).all()
    assert len(dense["traces"]) == 9
    unavailable = [trace for trace in dense["traces"] if trace["status"] == "unavailable"]
    assert len(unavailable) == 6
    assert all(trace["finite_pairs"] == 0 and trace["saved_rows"] == 1 and trace["reason"] for trace in unavailable)
    assert all(trace["finite_pairs"] > 0 for trace in dense["traces"] if trace["status"] == "plotted")
    assert any(row["status"] == "rendered" for row in dense["scene_images"])
    assert sum(row["status"] == "unavailable" for row in dense["scene_images"]) == 2
    reports = [("dense", output / "dense" / "report", dense)]
    for name, status in (("empty", "truncated"), ("zero-no-specs", "truncated"), ("historical-failure", "failed")):
        directory = output / name / "raw"
        directory.mkdir(parents=True, exist_ok=True)
        metadata = copy.deepcopy(fixture["metadata"])
        no_rows = name in ("empty", "zero-no-specs")
        metadata.update(status=status, steps=0 if no_rows else 4)
        rows = [] if no_rows else copy.deepcopy(fixture["rows"])
        if name in ("historical-failure", "zero-no-specs"):
            metadata.pop("spec_formulas")
            metadata.pop("spec_constants")
            for row in rows:
                row["status"] = status
        path = save_rows(rows, directory / "synthetic.parquet", metadata, quiet=True)
        manifest = export_report(directory, output_dir=output / name / "report", paths=[path])
        reports.append((name, output / name / "report", manifest))
    return reports


def render(reports, output):
    inventory = []
    for name, directory, manifest in reports:
        pdf = directory / "report-run.pdf"
        rendered = directory / "rendered"
        rendered.mkdir(exist_ok=True)
        for program in ("pdfinfo", "pdffonts", "pdftotext"):
            command = [program, str(pdf)]
            if program == "pdftotext":
                command.extend([str(directory / "report-text.txt")])
            with (directory / (program + ".log")).open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        with (directory / "render.log").open("w") as log:
            subprocess.run(["pdftoppm", "-scale-to", "1700", "-png", str(pdf), str(rendered / "page")],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        pages = sorted(rendered.glob("page-*.png"))
        assert len(pages) == len(manifest["pages"])
        text = (directory / "report-text.txt").read_text()
        assert "Native success: 0 / 0" in text
        if name in ("empty", "zero-no-specs"):
            assert "No observations available." in text
            assert len(pages) == (2 if name == "empty" else 1)
        if name == "historical-failure":
            assert "Historical monitored formula unavailable" in text
        if name == "dense":
            assert "Actual sampled step: 3" in text
        for page, declared in zip(pages, manifest["pages"]):
            inventory.append({"fixture": name, "pdf": str(pdf.resolve()),
                              "page": declared["page"], "title": declared["title"],
                              "png": str(page.resolve()), "png_sha256": hashlib.sha256(page.read_bytes()).hexdigest(),
                              "visual_reviewed": False})
        print(name, "pages:", len(pages), "metrics:", manifest["metrics"], flush=True)
    (output / "rendered-page-inventory.json").write_text(json.dumps(inventory, indent=2))
    print("Rendered pages:", len(inventory), "visual review pending", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    render(generate(arguments.output), arguments.output)
