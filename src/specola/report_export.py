"""Offline CSV and fixed-page PDF reports from finalized result artifacts.

Matplotlib supplies rendering and font embedding; pandas supplies the existing
outcome tables. No simulator, GUI, browser or TeX process participates here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Optional


MIN_FONT_PT = 10
PAGE_INCHES = (8.27, 11.69)
PDF_STYLE = {
    "font.family": "DejaVu Sans", "font.size": MIN_FONT_PT,
    "pdf.fonttype": 42, "pdf.use14corefonts": False,
    "text.usetex": False, "text.parse_math": False, "savefig.bbox": None,
}


class ReportPreflightError(RuntimeError):
    """A report dependency, destination or PDF rendering prerequisite failed."""

    def __init__(self, category, detail):
        self.category = category
        super().__init__(category + ": " + detail)


def report_preflight(output_dir) -> dict:
    """Check dependencies and destination before adapter construction.

    Temporary directories created for this check are removed when empty, so
    rejected simulator startup leaves no published run directory.
    """
    try:
        import pandas as pd
        import matplotlib as mpl
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.font_manager import FontProperties, findfont
    except ImportError as exc:
        raise ReportPreflightError("dependency", str(exc)) from exc
    directory = Path(output_dir)
    created = []
    cursor = directory
    while not cursor.exists():
        created.append(cursor)
        cursor = cursor.parent

    def remove_created():
        for path in created:
            try:
                path.rmdir()
            except OSError:
                pass

    try:
        directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".report-preflight-", suffix=".pdf", dir=directory)
        os.close(fd)
    except OSError as exc:
        remove_created()
        raise ReportPreflightError("output-location", str(exc)) from exc
    try:
        with mpl.rc_context(PDF_STYLE):
            font = findfont(FontProperties(family="DejaVu Sans"), fallback_to_default=False)
            figure = Figure(figsize=(2, 2))
            FigureCanvasAgg(figure)
            figure.text(0.1, 0.5, "Report readiness: ρ ≥ 0", parse_math=False)
            with PdfPages(temporary) as pdf:
                pdf.savefig(figure, bbox_inches=None)
        return {"pandas": pd.__version__, "matplotlib": mpl.__version__, "font": font}
    except OSError as exc:
        raise ReportPreflightError("output-location", str(exc)) from exc
    except Exception as exc:
        raise ReportPreflightError("pdf-backend-font", str(exc)) from exc
    finally:
        Path(temporary).unlink(missing_ok=True)
        remove_created()


def _figure(title, number):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=PAGE_INCHES, dpi=100)
    FigureCanvasAgg(figure)
    figure.text(0.07, 0.95, title, size=14, weight="bold", va="top", parse_math=False)
    figure.text(0.07, 0.035, "Specola | saved results | minimum text 10 pt", size=MIN_FONT_PT)
    figure.text(0.91, 0.035, str(number), size=MIN_FONT_PT, ha="right")
    return figure


def _wrap(figure, text, width, size=MIN_FONT_PT):
    """Measure literal characters, preserving newlines and breaking long IDs."""
    from matplotlib.font_manager import FontProperties

    renderer = figure.canvas.get_renderer()
    properties = FontProperties(family="DejaVu Sans", size=size)
    lines = []
    for paragraph in str(text).split("\n"):
        line = ""
        for char in paragraph:
            candidate = line + char
            pixels = renderer.get_text_width_height_descent(candidate, properties, ismath=False)[0]
            if pixels > width and line:
                lines.append(line)
                line = char
            else:
                line = candidate
        lines.append(line)
    return lines


def _check_page(figure, title, number):
    """Check unclipped text extents against the fixed physical page."""
    from matplotlib.text import Text
    from matplotlib.ft2font import FT2Font
    from matplotlib.font_manager import FontProperties, findfont

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.bbox.width, figure.bbox.height
    checked = 0
    glyphs = set()
    for artist in figure.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        if artist.get_fontsize() < MIN_FONT_PT:
            raise ValueError("Report text falls below minimum font size")
        box = artist.get_window_extent(renderer)
        if box.x0 < 15 or box.y0 < 15 or box.x1 > width - 15 or box.y1 > height - 15:
            raise ValueError("Report text exceeds page bounds: " + repr(artist.get_text()))
        glyphs.update(ord(char) for char in artist.get_text() if not char.isspace())
        checked += 1
    font = FT2Font(findfont(FontProperties(family="DejaVu Sans"), fallback_to_default=False))
    missing = glyphs - set(font.get_charmap())
    if missing:
        raise ValueError("Report font lacks authored glyphs: " + repr(sorted(missing)))
    return {"page": number, "title": title, "text_artists_checked": checked,
            "minimum_font_pt": MIN_FONT_PT, "glyphs": sorted(glyphs)}


def _text_pages(pdf, inventory, title, paragraphs, *, identities=None):
    """Keep bounded records together and identify longer continuations."""
    measure = _figure(title, len(inventory) + 1)
    capacity = 44
    lines = []
    records = []
    section_pages = 0

    def publish():
        nonlocal lines, records, section_pages
        heading = title + (" (continued)" if section_pages else "")
        figure = _figure(heading, len(inventory) + 1)
        for index, line in enumerate(lines):
            figure.text(0.07, 0.895 - index * (15 / 72 / PAGE_INCHES[1]),
                        line, size=MIN_FONT_PT, va="top", parse_math=False)
        page = _check_page(figure, title, len(inventory) + 1)
        page["records"] = records
        inventory.append(page)
        pdf.savefig(figure, bbox_inches=None)
        figure.clear()
        lines = []
        records = []
        section_pages += 1

    for index, paragraph in enumerate(paragraphs):
        record_lines = _wrap(measure, paragraph, measure.bbox.width * 0.84)
        identity = str(identities[index] if identities is not None else str(paragraph).split("\n", 1)[0])
        if len(record_lines) <= capacity:
            if lines and len(lines) + len(record_lines) + 1 > capacity:
                publish()
            first_page = len(inventory) + 1
            records.append({"record": index + 1, "identity": identity,
                            "continued": False, "first_page": first_page})
            lines.extend(record_lines)
            if len(lines) < capacity:
                lines.append("")
            continue
        if lines:
            publish()
        first_page = len(inventory) + 1
        continued = False
        while record_lines:
            prefix = []
            if continued:
                prefix = _wrap(measure, identity + " (continued; record {} begins on page {})".format(index + 1, first_page),
                               measure.bbox.width * 0.84)
                if len(prefix) > 8:
                    prefix = ["Record {} continued; identity begins on page {}.".format(index + 1, first_page)]
                prefix.append("")
            take = capacity - len(prefix)
            lines = prefix + record_lines[:take]
            record_lines = record_lines[take:]
            records = [{"record": index + 1, "identity": identity,
                        "continued": continued, "first_page": first_page}]
            publish()
            continued = True
    if lines or not section_pages:
        publish()
    measure.clear()


def _summary_table(episodes, group_columns):
    import pandas as pd

    columns = group_columns + ["episode_specs", "pass", "fail", "undetermined",
                              "not_applicable", "satisfaction_denominator", "sat_rate",
                              "known_native_success_episodes", "native_successes", "native_success_rate",
                              "completed", "truncated", "failed"]
    records = []
    for keys, group in episodes.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        counts = group["verdict"].value_counts()
        row.update({status: int(counts.get(status, 0)) for status in
                    ("pass", "fail", "undetermined", "not_applicable")})
        denominator = row["pass"] + row["fail"]
        row.update(episode_specs=len(group), satisfaction_denominator=denominator,
                   sat_rate=row["pass"] / denominator if denominator else float("nan"))
        known = group["success"].notna()
        row["known_native_success_episodes"] = int(known.sum())
        row["native_successes"] = int((group.loc[known, "success"] == True).sum())
        row["native_success_rate"] = row["native_successes"] / int(known.sum()) if known.any() else float("nan")
        row.update({status: int((group["status"] == status).sum())
                    for status in ("completed", "truncated", "failed")})
        records.append(row)
    return pd.DataFrame(records, columns=columns)


def _trace_pages(pdf, inventory, frame):
    import numpy as np

    traces = []
    unavailable = []
    unavailable_identities = []
    group_columns = ["episode_id", "spec", "binding_key"]
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        samples = group[group["step"] >= 0].sort_values("step", kind="stable")
        if samples.empty:
            continue
        label = "Episode: {}\nCompiled spec: {}\nBinding key: {}\nDisplay binding: {}".format(
            *keys, group["binding"].iloc[0])
        finite = np.isfinite(samples["rho"].astype(float)) & np.isfinite(samples["time"].astype(float))
        trace: dict = dict(zip(group_columns, (str(key) for key in keys)))
        trace.update(binding=str(group["binding"].iloc[0]), saved_rows=len(samples),
                     finite_pairs=int(finite.sum()), missing_nonfinite_pairs=int((~finite).sum()),
                     episode_status=str(group["status"].iloc[0]))
        traces.append(trace)
        if not finite.any():
            trace.update(status="unavailable", reason="No finite robustness/time pairs available.", page=None)
            unavailable.append(label + "\nReason: " + trace["reason"]
                               + "\n{} saved rows; 0 finite pairs; {} missing/nonfinite pairs. Episode status: {}.".format(
                                   len(samples), int((~finite).sum()), trace["episode_status"]))
            unavailable_identities.append("Episode: {}; spec: {}; binding key: {}".format(*keys))
            continue
        figure = _figure("Robustness trace", len(inventory) + 1)
        identity_lines = _wrap(figure, label, figure.bbox.width * 0.84)
        if len(identity_lines) > 4:
            _text_pages(pdf, inventory, "Trace identity", [label])
            figure = _figure("Robustness trace", len(inventory) + 1)
            identity_lines = ["Identity: preceding page(s). Values are not pooled across bindings."]
        for index, line in enumerate(identity_lines):
            figure.text(0.07, 0.895 - index * 0.02, line, size=MIN_FONT_PT, va="top")
        axis = figure.add_axes((0.14, 0.27, 0.76, 0.53))
        categories = (
            ("settled", (samples["verdict_final"] == True) & (samples["suspended"] == False), "#146b3a", "o"),
            ("provisional", (samples["verdict_final"] == False) & (samples["suspended"] == False), "#285c96", "."),
            ("suspended", samples["suspended"] == True, "#965900", "x"),
        )
        # Lines retain gaps for missing/nonfinite values and category changes.
        for name, mask, color, marker in categories:
            values = samples["rho"].astype(float).where(mask & finite, np.nan)
            axis.plot(samples["time"], values, color=color, marker=marker,
                      markersize=3, linewidth=0.8, label=name)
        axis.axhline(0, color="#555555", linewidth=0.7)
        times = samples.loc[np.isfinite(samples["time"].astype(float)), "time"]
        lower, upper = float(times.min()), float(times.max())
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        axis.set_xticks(np.linspace(lower, upper, 5))
        padding = (upper - lower) * 0.04
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_yticks(np.linspace(*axis.get_ylim(), 5))
        axis.minorticks_off()
        axis.set_xlabel("Saved simulation time (s)")
        axis.set_ylabel("Robustness (formula units)")
        axis.tick_params(labelsize=MIN_FONT_PT)
        legend = axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                             borderaxespad=0, ncol=3, frameon=False, fontsize=MIN_FONT_PT)
        note = (
            "Status: {}. {} saved rows; {} finite plotted pairs; {} missing/nonfinite pairs.\n"
            "No display reduction. Full values, including infinity, remain in report-plot-data.csv.\n"
            "Component units and signals are unavailable in the raw observation schema."
        ).format(group["status"].iloc[0], len(samples), int(finite.sum()), int((~finite).sum()))
        for index, line in enumerate(_wrap(figure, note, figure.bbox.width * 0.84)):
            figure.text(0.07, 0.13 - index * 0.02, line, size=MIN_FONT_PT, va="top")
        page = _check_page(figure, "Robustness trace", len(inventory) + 1)
        renderer = figure.canvas.get_renderer()
        panel = axis.get_window_extent(renderer)
        if legend.get_window_extent(renderer).overlaps(panel):
            raise ValueError("Robustness legend overlaps the data panel")
        positions = axis.transData.transform(samples.loc[finite, ["time", "rho"]].astype(float).to_numpy())
        # A 3 pt marker and its 0.8 pt stroke fit inside this 4 pt
        # clearance. Check before clipping, not the clipped artist box.
        clearance = 4 * figure.dpi / 72
        if any(x - panel.x0 < clearance or panel.x1 - x < clearance for x, _ in positions):
            raise ValueError("Robustness endpoint marker exceeds the data panel")
        trace.update(status="plotted", reason="", page=len(inventory) + 1)
        page["trace"] = {**trace, "x_limits": list(axis.get_xlim()), "legend_outside_panel": True}
        inventory.append(page)
        pdf.savefig(figure, bbox_inches=None)
        figure.clear()
    if unavailable:
        _text_pages(pdf, inventory, "Unavailable traces", unavailable,
                    identities=unavailable_identities)
    return traces


def _image_pages(pdf, inventory, sources, recordings):
    """Render explicitly associated finalized recordings through RecordingReader."""
    import hashlib
    import io
    from matplotlib.image import imread
    from .dashboard.recording import RecordingReader
    from .conformance.verification import require_ordinary_execution

    associations = {str(Path(raw).resolve()): Path(directory)
                    for raw, directory in (recordings or {}).items()}
    source_paths = {source["path"] for source in sources}
    if not set(associations) <= source_paths:
        raise ValueError("Recording association does not name a selected raw artifact")
    rows = []
    for source in sources:
        raw = source["path"]
        if raw not in associations:
            rows.append({"raw_source": raw, "recording": "", "camera": "", "sampled_step": None,
                         "status": "unavailable", "reason": "No finalized recording explicitly associated"})
            continue
        directory = associations[raw]
        if not (directory / "manifest.json").is_file():
            rows.append({"raw_source": raw, "recording": str(directory), "camera": "", "sampled_step": None,
                         "status": "unavailable", "reason": "Associated recording manifest missing"})
            continue
        reader = RecordingReader(directory)
        manifest = reader.manifest
        require_ordinary_execution(manifest)
        metadata = source["metadata"]
        if (not metadata.get("run_id") or manifest.get("run_id") != metadata["run_id"]
                or manifest.get("task") != metadata.get("task_id", metadata.get("task"))
                or manifest.get("status") != metadata.get("status")):
            raise ValueError("Recording run/task/status identity differs from saved raw metadata: " + str(directory))
        if manifest.get("status") not in ("completed", "truncated", "failed"):
            raise ValueError("Associated recording is not finalized: " + str(directory))
        cameras = reader.cameras()
        if not cameras:
            rows.append({"raw_source": raw, "recording": str(directory), "camera": "", "sampled_step": None,
                         "status": "unavailable", "reason": "Finalized recording has no saved camera frames"})
        for camera in cameras:
            steps = reader.frame_steps(camera)
            entry = reader.frame_entry(camera, steps[-1]) if steps else None
            row = {"raw_source": raw, "recording": str(directory), "camera": camera,
                   "sampled_step": entry[2] if entry is not None else None}
            if entry is None:
                row.update(status="unavailable", reason="No indexed frame available")
            elif entry[1] != "jpeg":
                row.update(status="unsupported", reason="Saved encoded video; existing offline reader provides encoded bytes without a Python image decoder")
            else:
                image = imread(io.BytesIO(entry[0]), format="jpeg")
                figure = _figure("Saved scene image", len(inventory) + 1)
                label = "Camera: {}\nActual sampled step: {}\nRaw source: {}".format(camera, entry[2], raw)
                lines = _wrap(figure, label, figure.bbox.width * 0.84)
                if len(lines) > 7:
                    _text_pages(pdf, inventory, "Scene image identity", [label])
                    lines = ["Identity: preceding page(s). Actual sampled step: " + str(entry[2])]
                    figure = _figure("Saved scene image", len(inventory) + 1)
                for index, line in enumerate(lines):
                    figure.text(0.07, 0.895 - index * 0.02, line, size=MIN_FONT_PT, va="top")
                axis = figure.add_axes((0.08, 0.16, 0.84, 0.53))
                axis.imshow(image, origin="upper", aspect="equal", interpolation="nearest")
                axis.set_axis_off()
                figure.text(0.07, 0.1, "Representative saved frame; no claim about unsampled steps.", size=MIN_FONT_PT)
                inventory.append(_check_page(figure, "Saved scene image", len(inventory) + 1))
                pdf.savefig(figure, bbox_inches=None)
                figure.clear()
                row.update(status="rendered", reason="", codec="jpeg",
                           sha256=hashlib.sha256(entry[0]).hexdigest())
            rows.append(row)
    unavailable = [json.dumps(row, ensure_ascii=False) for row in rows if row["status"] != "rendered"]
    if unavailable:
        _text_pages(pdf, inventory, "Scene image availability", unavailable)
    return rows


def export_report(results_dir, *, output_dir=None, paths=None, batch_tasks=None, recordings=None,
                  trace_allocations: bool = False) -> dict:
    """Publish CSV/PDF artifacts from saved rows with explicit batch membership.

    ``paths`` lists raw episodes. ``batch_tasks`` declares the exact task
    population of a combined report; it must equal the saved task population.
    A directory scan alone never labels a report as a declared batch.
    Allocation tracing is opt-in; caller-owned tracing is never reset or stopped.
    """
    import hashlib
    import resource
    import tracemalloc
    import pandas as pd
    import matplotlib as mpl
    from matplotlib.backends.backend_pdf import PdfPages
    from .report import load, episode_table

    started = time.perf_counter()
    own_tracing = trace_allocations and not tracemalloc.is_tracing()
    if own_tracing:
        tracemalloc.start()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    destination = Path(output_dir if output_dir is not None else results_dir)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        frame = load(str(results_dir), paths=paths)
        sources = frame.attrs.get("sources", [])
        if not sources:
            raise ValueError("No raw result artifacts selected; supply saved episode paths")
        if paths is not None:
            episode_sources = {}
            for source in sources:
                identities = set(source.get("episode_ids", []))
                if source["metadata"].get("episode_id"):
                    identities.add(source["metadata"]["episode_id"])
                for identity in identities:
                    if identity in episode_sources:
                        raise ValueError("Duplicate episode identity in explicit report membership: " + identity)
                    episode_sources[identity] = source["path"]
        tasks = {task for source in sources for task in source["task_ids"]}
        unavailable_task_sources = sum(source["task_identity_unavailable"] for source in sources)
        if batch_tasks is not None:
            if paths is None or not batch_tasks or len(set(batch_tasks)) != len(batch_tasks):
                raise ValueError("A batch requires explicit raw paths and unique declared tasks")
            if tasks != set(batch_tasks):
                raise ValueError("Declared batch tasks differ from saved tasks: " + repr(sorted(tasks)))
        episodes = episode_table(frame)
        groups = [column for column in ("task_id", "task", "policy", "seed") if column in episodes] + ["spec"]
        summary = _summary_table(episodes, groups)
        specs = []
        provenance = []
        for source in sources:
            metadata = source["metadata"]
            compact_metadata = {key: value for key, value in metadata.items() if key not in ("spec_formulas", "spec_constants", "raw_column_types")}
            provenance.append("Raw source: " + source["path"] + "\nSaved metadata (formulas and constants appear on specification pages): " + json.dumps(compact_metadata, ensure_ascii=False, sort_keys=True))
            formulas = json.loads(metadata.get("spec_formulas", "[]"))
            if not formulas:
                specs.append({"source": source["path"], "spec": "unavailable", "formula": "Historical monitored formula unavailable", "constants": metadata.get("spec_constants", "unavailable")})
            for item in formulas:
                specs.append({"source": source["path"], "spec": item["name"], "formula": item["formula"], "constants": json.dumps(item.get("constants", {}), ensure_ascii=False, sort_keys=True)})
        spec_frame = pd.DataFrame(specs, columns=["source", "spec", "formula", "constants"])
        episode_ids = set(frame["episode_id"].dropna().astype(str))
        episode_ids.update(source["metadata"]["episode_id"] for source in sources if "episode_id" in source["metadata"])
        status_records = frame[["episode_id", "status", "success"]].drop_duplicates("episode_id")
        extra_statuses = []
        known_ids = set(status_records["episode_id"])
        for source in sources:
            metadata = source["metadata"]
            identity = metadata.get("episode_id")
            if identity and identity not in known_ids:
                saved_success = metadata.get("success")
                if saved_success not in (None, "None", "", "True", "False"):
                    raise ValueError("Malformed saved native success: " + repr(saved_success))
                extra_statuses.append({
                    "episode_id": identity, "status": metadata.get("status", "unavailable"),
                    "success": True if saved_success == "True" else False if saved_success == "False" else None,
                })
                known_ids.add(identity)
        if extra_statuses:
            status_records = pd.concat([status_records, pd.DataFrame(extra_statuses)], ignore_index=True)
        known_success = status_records["success"].notna()
        native_successes = int((status_records.loc[known_success, "success"] == True).sum())
        known_count = int(known_success.sum())
        status_counts = status_records["status"].value_counts().to_dict()
        inventory = []
        with tempfile.TemporaryDirectory(prefix=".report-export-", dir=destination) as staging:
            staging = Path(staging)
            frames = {"report-episode-summary.csv": episodes, "report-task-spec-summary.csv": summary,
                      "report-plot-data.csv": frame, "report-specifications.csv": spec_frame}
            for name, table in frames.items():
                table.to_csv(staging / name, index=False)
            with mpl.rc_context(PDF_STYLE), PdfPages(staging / "report-run.pdf", metadata={"Title": "Specola saved run report"}) as pdf:
                fixture_notes = sorted({source["metadata"]["fixture_origin"] for source in sources if "fixture_origin" in source["metadata"]})
                overview = [
                    *("Data origin: " + note for note in fixture_notes),
                    "Scope: " + ("declared batch" if batch_tasks is not None else "selected saved run artifacts; benchmark completeness is not asserted"),
                    "{} raw artifacts; {} tasks with saved identity; {} episodes; {} episode/spec outcomes; {} raw rows.".format(len(sources), len(tasks), len(episode_ids), len(episodes), len(frame)),
                    "Tasks: " + (", ".join(sorted(tasks)) or "unavailable"),
                    "{} raw sources contain unavailable task identity.".format(unavailable_task_sources),
                    "Episode statuses: " + json.dumps(status_counts, sort_keys=True),
                    "Native success: {} / {} episodes with known success ({} selected episodes). Specification satisfaction is separate.".format(native_successes, known_count, len(episode_ids)),
                    "Outcome denominators: satisfaction = pass / (pass + fail). Undetermined and not applicable outcomes are excluded from this denominator and retained separately.",
                    "No observations available." if frame.empty or not (frame["step"] >= 0).any() else "Raw traces retain settled, provisional and suspended values. No plot reduction is applied.",
                    "Saved component time series are unavailable in the raw schema. Explicitly associated finalized recordings supply scene images when available. Policy/seed identities appear in saved provenance when present.",
                    "Layout: A4 portrait; minimum font 10 pt; literal spec text; no TeX. Per-binding robustness has formula-dependent units; units are not inferred.",
                ]
                if frame.empty:
                    overview.extend("Raw source: " + source["path"] for source in sources)
                    overview.append("Saved provenance is retained in report-manifest.json. No observed outcome, trace or image is inferred from empty rows.")
                    if all(row["spec"] == "unavailable" for row in specs):
                        overview.append("Historical monitored formula unavailable; no saved specification snapshot.")
                _text_pages(pdf, inventory, "Run overview", overview)
                summaries = []
                for row in summary.to_dict("records"):
                    identity = "; ".join(key + ": " + str(row[key]) for key in groups)
                    statistics = (
                        "Outcomes: pass {pass}; fail {fail}; undetermined {undetermined}; not applicable {not_applicable}; total {episode_specs}.\n"
                        "Satisfaction: {pass} / {satisfaction_denominator} = {sat_rate}.\n"
                        "Native success: {native_successes} / {known_native_success_episodes} known episodes = {native_success_rate}.\n"
                        "Episode statuses: completed {completed}; truncated {truncated}; failed {failed}."
                    ).format(**row)
                    summaries.append(identity + "\n" + statistics)
                if summaries:
                    _text_pages(pdf, inventory, "Task and specification summaries", summaries)
                if not episodes.empty:
                    episode_records = episodes.to_dict("records")
                    _text_pages(pdf, inventory, "Episode outcomes",
                                [json.dumps(row, default=str, ensure_ascii=False) for row in episode_records],
                                identities=["Episode: {}; spec: {}".format(row["episode_id"], row["spec"]) for row in episode_records])
                if not frame.empty or any(row["spec"] != "unavailable" for row in specs):
                    _text_pages(pdf, inventory, "Monitored specifications", ["Spec: {spec}\nFormula: {formula}\nConstants: {constants}\nRaw source: {source}".format(**row) for row in specs])
                traces = _trace_pages(pdf, inventory, frame)
                pd.DataFrame(traces, columns=["episode_id", "spec", "binding_key", "binding", "saved_rows",
                                              "finite_pairs", "missing_nonfinite_pairs", "episode_status",
                                              "status", "reason", "page"]).to_csv(staging / "report-trace-summary.csv", index=False)
                if not frame.empty or recordings:
                    scene_images = _image_pages(pdf, inventory, sources, recordings)
                else:
                    scene_images = [{"raw_source": source["path"], "recording": "", "camera": "", "sampled_step": None,
                                     "status": "unavailable", "reason": "No finalized recording explicitly associated"} for source in sources]
                pd.DataFrame(scene_images, columns=["raw_source", "recording", "camera", "sampled_step", "status", "reason", "codec", "sha256"]).to_csv(staging / "report-scene-images.csv", index=False)
                if not frame.empty:
                    _text_pages(pdf, inventory, "Saved provenance", provenance)
            python_peak = tracemalloc.get_traced_memory()[1] if own_tracing else None
            manifest = {
                "schema": 1, "sources": sources, "declared_batch_tasks": batch_tasks,
                "task_count": len(tasks), "unavailable_task_identity_sources": unavailable_task_sources,
                "episode_count": len(episode_ids),
                "raw_rows": len(frame), "episode_spec_outcomes": len(episodes),
                "pages": inventory, "minimum_font_pt": MIN_FONT_PT, "scene_images": scene_images,
                "traces": traces,
                "metrics": {"report_build_seconds": time.perf_counter() - started,
                            "report_build_python_tracemalloc_peak_bytes": python_peak,
                            "process_high_water_rss_before_build_kib": rss_before,
                            "process_high_water_rss_after_build_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            "measurement_scope": "Result loading and CSV/PDF construction after initial export_report imports; excludes artifact hashing, manifest serialization and publication",
                            "memory_scope": ("Python allocations during report build" if own_tracing else
                                             "Python allocation peak unavailable: tracing disabled or caller-owned") +
                                            "; RSS is whole-process lifetime high-water, not report-exclusive memory"},
                "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in staging.iterdir()},
            }
            (staging / "report-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            # Each artifact is replaced atomically. The manifest publishes last
            # and records hashes so interrupted multi-file replacement is detectable.
            for name in list(manifest["files"]) + ["report-manifest.json"]:
                (staging / name).replace(destination / name)
        return manifest
    finally:
        if own_tracing:
            tracemalloc.stop()


def export_finalized_session(session, output_dir, *, recording_dir=None) -> Optional[dict]:
    """Export a retained finalized record after its caller releases the adapter."""
    import traceback

    record = session._record
    if not session.ended or record is None or record.path is None:
        return None
    try:
        recordings = {str(record.path.resolve()): str(recording_dir)} if recording_dir is not None else None
        manifest = export_report(output_dir, paths=[record.path], recordings=recordings)
    except Exception:
        error = Path(output_dir) / "report-error.log"
        failure = traceback.format_exc()
        try:
            error.write_text(failure, encoding="utf-8")
        except OSError:
            import sys
            print(failure, file=sys.stderr)
            print("  Report diagnostics could not be saved at " + str(error), file=sys.stderr)
        print("  Report export failed; raw episode status remains {}. Details: {}".format(record.status, error))
        return {"error": str(error)}
    print("  Report PDF: {}".format(Path(output_dir) / "report-run.pdf"))
    for name in manifest["files"]:
        if name.endswith(".csv"):
            print("  Report CSV: {}".format(Path(output_dir) / name))
    peak = manifest["metrics"]["report_build_python_tracemalloc_peak_bytes"]
    peak_text = "unavailable" if peak is None else "{} bytes".format(peak)
    print("  Report build before hashing/publication: {:.3f} s; Python allocation peak {} (whole-process RSS recorded in the manifest).".format(manifest["metrics"]["report_build_seconds"], peak_text))
    return manifest
