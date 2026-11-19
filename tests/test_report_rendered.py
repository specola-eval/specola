"""Poppler regression lane for installed report dependencies and page output."""

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import specola
from specola.results_io import save_rows
from specola.report_export import export_report


def test_rendered_synthetic_report_preserves_values_and_embeds_fonts(tmp_path):
    for program in ("pdfinfo", "pdffonts", "pdftotext", "pdftoppm"):
        if shutil.which(program) is None:
            pytest.skip("Rendered report regression requires Poppler tool: " + program)
    fixture_path = Path(specola.__file__).parent / "data" / "report-fixture.json"
    fixture = json.loads(fixture_path.read_text())
    path = save_rows(fixture["rows"], tmp_path / "synthetic.parquet", fixture["metadata"], quiet=True)
    manifest = export_report(tmp_path, paths=[path])
    pdf = tmp_path / "report-run.pdf"
    text_path = tmp_path / "text.txt"
    commands = {
        "pdfinfo": ["pdfinfo", str(pdf)],
        "pdffonts": ["pdffonts", str(pdf)],
        "pdftotext": ["pdftotext", str(pdf), str(text_path)],
        "pdftoppm": ["pdftoppm", "-scale-to", "1000", "-png", str(pdf), str(tmp_path / "page")],
    }
    logs = {}
    for name, command in commands.items():
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (tmp_path / (name + ".log")).write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        logs[name] = result.stdout
    assert len(list(tmp_path.glob("page-*.png"))) == len(manifest["pages"])
    assert re.search(r"CID TrueType\s+Identity-H\s+yes\s+yes\s+yes", logs["pdffonts"])
    text = re.sub(r"\s+", " ", text_path.read_text())
    assert "Satisfaction: 0 / 1 = 0.0" in text
    assert "Native success: 0 / 0" in text
    assert "synthetic; no simulator or policy evaluation" in text
    assert "Historical monitored formula unavailable" not in text
    assert "ρ ≥ −∞" in text
