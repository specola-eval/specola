"""report.py — aggregation over episode result parquets (`specola report` CLI).

Reads the parquet files written by `results_io.save_rows` (Task 5's schema:
one row per monitor evaluation, `verdict_final` rows carrying the settled
robustness value for a `(monitor spec, binding)` pair) and rolls them up
into per-episode verdicts and per-group satisfaction statistics.

pandas is a baseline dependency. Imports remain local so discovery and ordinary
doctor do not load plotting or table libraries. PyArrow is optional for
additional Parquet output and historical Parquet input.

Verdict rule (kept identical to `Session._compute_outcomes` in session.py,
so a report built from parquet files agrees with the verdicts an in-process
`Session` would have computed for the same run): for a given
`(episode_id, authored_spec)` group, restrict to rows where
`verdict_final` is True and `not_applicable` is False. Within that subset,
take each `(spec, binding_key)` pair's LAST row by step (a binding that
dipped negative and recovered is judged by where it ended up, not where it
dipped). Keying on `binding_key` — the entity-ID identity — rather than on
the display `binding` label matters: two same-category entities can share a
label, and keying on the label would let a passing binding absorb a
co-labeled failing one. `Session._compute_outcomes` keys the same way.
If any of those last-per-binding rho values is negative, the verdict is
"fail"; if there is at least one such value and all are non-negative, it is
"pass". If the subset is empty because every row is `not_applicable`, the
verdict is "not_applicable"; if it is empty for any other reason (no final
rows at all — e.g. the episode was truncated before any monitor settled),
the verdict is "undetermined".

`worst_rho` is the min over that same last-per-binding rho set (mirrors
`SpecOutcome.worst_rho`). `first_violation_step` is the min step over ALL
final, non-`not_applicable` rows with rho < 0 in the group (not just the
last-per-binding set) — matching `SpecOutcome.first_violation_step`, so a
binding that violated early and later recovered still reports when the
violation first happened.

Episode length (`steps`) is not part of Task 5's row schema — it is not
stamped anywhere. We recover it as `max(step) + 1` over the group's
non-`not_applicable` rows. This under-reports the true episode length if
the last row written for that (episode, spec) happens to be an earlier
final row than the episode's actual last step (e.g. nothing more ever got
evaluated after the last violation) — callers that need an exact episode
length should stamp one explicitly in a future schema revision; today's
callers (the `mean_detection_lead` diagnostic) only need a lower bound on
how many steps elapsed after the first violation was detected.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, TYPE_CHECKING

from .results_io import CORE_COLUMNS, RESERVED_META_KEYS

# The columns this module derives below (verdict, worst_rho, steps,
# success_rate, ...) are reserved against caller meta keys by
# ``results_io.RESERVED_META_KEYS`` — the single authority for that set,
# imported rather than restated here so the reserved list and the columns
# actually produced can never drift. ``session.watch()`` enforces it.
assert {
    "verdict", "worst_rho", "first_violation_step", "steps",
    "episodes", "sat_rate", "success_rate", "mean_worst_rho",
    "min_worst_rho", "mean_detection_lead",
    "n_success_but_violated", "n_failed_but_satisfied",
} <= RESERVED_META_KEYS

if TYPE_CHECKING:
    # Runtime imports stay inside the functions below. This import is never executed at
    # runtime — it exists so type checkers can resolve the "pd.DataFrame"
    # annotations used throughout this module.
    import pandas as pd



def _require_pandas():
    try:
        import pandas  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Reporting requires pandas; reinstall specola with its baseline dependencies") from exc


def load(results_dir: str, *, paths=None) -> "pd.DataFrame":
    """Read raw CSV or Parquet once per basename, retaining saved metadata.

    Explicit paths define membership. Compatibility scans accept observation
    headers and exclude report-* exports. CSV wins over a sibling Parquet.
    Identity and authored-text columns remain strings, including empty bindings.
    """
    import csv
    import json
    import warnings
    import pandas as pd

    root = Path(results_dir)
    explicit = paths is not None or root.is_file()
    candidates = [Path(path) for path in paths] if paths is not None else (
        [root] if root.is_file() else sorted(
            list(root.rglob("*.csv")) + list(root.rglob("*.parquet"))
        )
    )
    selected = {}
    for path in candidates:
        if path.suffix not in (".csv", ".parquet"):
            raise ValueError("Raw result path must end in .csv or .parquet: " + str(path))
        if path.name.startswith("report-"):
            if explicit:
                raise ValueError("A report export is not a raw episode: " + str(path))
            continue
        key = str(path.resolve().with_suffix(""))
        if key not in selected or path.suffix == ".csv":
            selected[key] = path
    frames = []
    sources = []
    required = {"step", "time", "spec", "binding", "rho", "verdict_final"}
    for path in sorted(selected.values()):
        metadata = {}
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames or []
                if not required <= set(columns):
                    if explicit:
                        raise ValueError("Not an observation CSV: " + str(path))
                    continue
                records = list(reader)
            frame = pd.DataFrame(records, columns=columns)
            for column in ("step", "window_steps"):
                if column in frame:
                    frame[column] = pd.array([int(value) if value != "" else None for value in frame[column]], dtype="Int64")
            for column in ("time", "rho"):
                frame[column] = frame[column].map(lambda value: float(value) if value != "" else float("nan")).astype("float64")
            for column in ("suspended", "verdict_final", "not_applicable", "success"):
                if column not in frame:
                    continue
                nullable = column == "success"
                def boolean(value):
                    if value == "True":
                        return True
                    if value == "False":
                        return False
                    if nullable and value == "":
                        return None
                    raise ValueError("Invalid boolean in " + column + ": " + repr(value))
                frame[column] = frame[column].map(boolean).astype("boolean")
            sidecar = path.with_suffix(".meta.json")
            if sidecar.exists():
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict) or any(
                    not isinstance(value, str) for value in metadata.values()
                ):
                    raise ValueError("Result metadata must contain string values: " + str(sidecar))
            else:
                warnings.warn("Saved provenance unavailable: " + str(sidecar), RuntimeWarning)
            column_types = json.loads(metadata.get("raw_column_types", "{}"))
            for column, kind in column_types.items():
                if column not in frame or column in CORE_COLUMNS:
                    continue
                if kind == "float":
                    frame[column] = frame[column].map(lambda value: float(value) if value != "" else float("nan")).astype("float64")
                elif kind == "int":
                    frame[column] = pd.Series([int(value) if value != "" else None for value in frame[column]], index=frame.index, dtype=object)
                elif kind == "bool":
                    if not frame[column].isin(["True", "False"]).all():
                        raise ValueError("Invalid metadata boolean column: " + column)
                    frame[column] = frame[column].map({"True": True, "False": False})
        else:
            import pyarrow.parquet as pq
            schema = pq.read_schema(path)
            metadata = {
                key.decode(): value.decode()
                for key, value in (schema.metadata or {}).items() if key != b"pandas"
            }
            if not required <= set(schema.names):
                if not schema.names and "episode_id" in metadata:
                    frame = pd.DataFrame(columns=sorted(CORE_COLUMNS))
                elif explicit:
                    raise ValueError("Not an observation Parquet: " + str(path))
                else:
                    continue
            else:
                frame = pd.read_parquet(path)
        from .conformance.verification import require_ordinary_execution
        require_ordinary_execution(metadata)
        sources.append({"path": str(path.resolve()), "metadata": metadata})
        if "episode_id" not in frame:
            frame["episode_id"] = metadata.get("episode_id", str(path))
        if "authored_spec" not in frame:
            frame["authored_spec"] = frame["spec"]
        else:
            frame["authored_spec"] = frame["authored_spec"].fillna(frame["spec"])
        if "binding_key" not in frame:
            frame["binding_key"] = frame["binding"]
        else:
            frame["binding_key"] = frame["binding_key"].fillna(frame["binding"])
        for column, default in (("success", None), ("status", "completed"),
                                ("not_applicable", False), ("suspended", False),
                                ("window_steps", None)):
            if column not in frame:
                frame[column] = default
            elif default is not None:
                frame[column] = frame[column].fillna(default)
        sources[-1]["episode_ids"] = list(frame["episode_id"].dropna().astype(str).unique())
        if not frame.empty and ("task_id" in frame or "task" in frame):
            task_values = frame["task_id"] if "task_id" in frame else frame["task"]
            if "task_id" in frame and "task" in frame:
                task_values = task_values.where(task_values.notna(), frame["task"])
            task_ids = {str(value) for value in task_values.dropna() if str(value) != ""}
            unavailable_task = bool((task_values.isna() | (task_values == "")).any())
        else:
            task = metadata.get("task_id", metadata.get("task", ""))
            task_ids = {task} if task != "" else set()
            unavailable_task = not task_ids
        sources[-1]["task_ids"] = sorted(task_ids)
        sources[-1]["task_identity_unavailable"] = unavailable_task
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=sorted(CORE_COLUMNS))
    df.attrs["sources"] = sources
    return df


def _meta_columns(df: "pd.DataFrame") -> List[str]:
    return [c for c in df.columns if c not in CORE_COLUMNS]


def _episode_spec_row(episode_id, spec, group, meta_cols) -> dict:
    import numpy as np

    final_mask = (group["verdict_final"] == True) & (group["not_applicable"] == False)  # noqa: E712
    filtered = group[final_mask]

    if not filtered.empty:
        idx = filtered.groupby(["spec", "binding_key"])["step"].idxmax()
        last_rows = filtered.loc[idx]
        rhos = last_rows["rho"]
        verdict = "fail" if (rhos < 0).any() else "pass"
        worst_rho = rhos.min()
        violations = filtered[filtered["rho"] < 0]
        first_violation_step = violations["step"].min() if not violations.empty else np.nan
    elif group["not_applicable"].any():
        verdict = "not_applicable"
        worst_rho = np.nan
        first_violation_step = np.nan
    else:
        verdict = "undetermined"
        worst_rho = np.nan
        first_violation_step = np.nan

    applicable_rows = group[group["not_applicable"] == False]  # noqa: E712
    steps = applicable_rows["step"].max() + 1 if not applicable_rows.empty else np.nan

    row = {
        "episode_id": episode_id,
        "spec": spec,
        "verdict": verdict,
        "worst_rho": worst_rho,
        "first_violation_step": first_violation_step,
        "steps": steps,
        "success": group["success"].iloc[0],
        "status": group["status"].iloc[0],
    }
    for col in meta_cols:
        row[col] = group[col].iloc[0]
    return row


def episode_table(df: "pd.DataFrame") -> "pd.DataFrame":
    """One row per (episode_id, authored_spec): verdict + diagnostics.

    Columns: episode_id, spec (the authored spec name), verdict, worst_rho,
    first_violation_step, steps, success, status, plus any meta columns
    (columns of `df` outside `results_io.CORE_COLUMNS`, e.g. train_step).
    """
    import pandas as pd

    base_cols = ["episode_id", "spec", "verdict", "worst_rho",
                 "first_violation_step", "steps", "success", "status"]
    if df.empty:
        return pd.DataFrame(columns=base_cols + _meta_columns(df))

    meta_cols = _meta_columns(df)
    records = []
    for (episode_id, spec), group in df.groupby(["episode_id", "authored_spec"], sort=False):
        records.append(_episode_spec_row(episode_id, spec, group, meta_cols))

    out = pd.DataFrame.from_records(records, columns=base_cols + meta_cols)
    return out.sort_values(["episode_id", "spec"], ignore_index=True)


def _summarize_group(group: "pd.DataFrame") -> dict:
    determinable = group[group["verdict"].isin(["pass", "fail"])]
    n_pass = int((determinable["verdict"] == "pass").sum())
    n_fail = int((determinable["verdict"] == "fail").sum())
    sat_rate = n_pass / (n_pass + n_fail) if (n_pass + n_fail) > 0 else float("nan")

    success_numeric = group["success"].dropna().astype("float64")
    success_rate = success_numeric.mean()

    lead_mask = (group["success"] == False) & group["first_violation_step"].notna()  # noqa: E712
    lead_values = group.loc[lead_mask, "steps"] - group.loc[lead_mask, "first_violation_step"]
    mean_detection_lead = lead_values.mean() if not lead_values.empty else float("nan")

    n_success_but_violated = int(((group["success"] == True) & (group["verdict"] == "fail")).sum())  # noqa: E712
    n_failed_but_satisfied = int(((group["success"] == False) & (group["verdict"] == "pass")).sum())  # noqa: E712

    return {
        "episodes": len(group),
        "success_rate": success_rate,
        "sat_rate": sat_rate,
        "mean_worst_rho": group["worst_rho"].mean(),
        "min_worst_rho": group["worst_rho"].min(),
        "mean_detection_lead": mean_detection_lead,
        "n_success_but_violated": n_success_but_violated,
        "n_failed_but_satisfied": n_failed_but_satisfied,
    }


def summarize(df: "pd.DataFrame", by: List[str]) -> "pd.DataFrame":
    """One row per (group values..., spec): satisfaction + agreement stats.

    `sat_rate` excludes `not_applicable`/`undetermined` episodes from its
    denominator (pass / (pass + fail)). `mean_detection_lead` averages
    `steps - first_violation_step` over episodes with `success == False`
    and a recorded first violation. `n_success_but_violated` counts
    episodes flagged `success == True` where the spec verdict is "fail";
    `n_failed_but_satisfied` counts the reverse disagreement.
    """
    import pandas as pd

    ep = episode_table(df)
    missing = [k for k in by if k not in ep.columns]
    if missing:
        raise SystemExit(
            "--by column{} not in the results: {}; the meta columns present "
            "are {}".format(
                "s" if len(missing) > 1 else "", ", ".join(missing),
                ", ".join(_meta_columns(df)) or "none",
            )
        )
    group_cols = list(by) + ["spec"]

    if ep.empty:
        cols = group_cols + ["episodes", "success_rate", "sat_rate",
                              "mean_worst_rho", "min_worst_rho",
                              "mean_detection_lead", "n_success_but_violated",
                              "n_failed_but_satisfied"]
        return pd.DataFrame(columns=cols)

    records = []
    for keys, group in ep.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        record.update(_summarize_group(group))
        records.append(record)

    out = pd.DataFrame.from_records(records)
    return out.sort_values(group_cols, ignore_index=True)


def main() -> None:
    import argparse
    import json
    from .report_export import export_report, report_preflight, ReportPreflightError

    ap = argparse.ArgumentParser(prog="specola report")
    ap.add_argument("results_dir")
    ap.add_argument("--by", default="", help="comma-separated meta columns")
    ap.add_argument("--csv", default=None, help="legacy aggregate-summary CSV path")
    ap.add_argument("--parquet", default=None, help="legacy aggregate-summary Parquet path")
    ap.add_argument("--output-dir", default=None, help="directory for regenerated PDF and report CSV files")
    ap.add_argument("--batch-manifest", default=None,
                    help="JSON with explicit members (raw paths) and tasks; paths resolve beside the manifest")
    ap.add_argument("--recording", action="append", default=[], metavar="RAW_PATH=RECORDING_DIR",
                    help="explicit raw episode to finalized recording association (repeatable)")
    args = ap.parse_args()
    _require_pandas()
    root = Path(args.results_dir)
    if not root.exists():
        ap.error("results path not found: {}".format(root))
    output = Path(args.output_dir) if args.output_dir else (root.parent if root.is_file() else root)
    paths = None
    tasks = None
    recordings = {}
    for association in args.recording:
        if "=" not in association:
            ap.error("--recording requires RAW_PATH=RECORDING_DIR")
        raw_path, directory = association.split("=", 1)
        recordings[str(Path(raw_path).resolve())] = str(Path(directory).resolve())
    if args.batch_manifest:
        manifest_path = Path(args.batch_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or set(manifest) != {"members", "tasks"}:
            ap.error("batch manifest requires exactly members and tasks")
        if (not isinstance(manifest["members"], list) or not manifest["members"]
                or not all(isinstance(item, str) and item for item in manifest["members"])
                or not isinstance(manifest["tasks"], list)
                or not all(isinstance(item, str) for item in manifest["tasks"])):
            ap.error("batch members and tasks must be explicit string lists")
        paths = [(manifest_path.parent / item).resolve() for item in manifest["members"]]
        if len(set(path.with_suffix("") for path in paths)) != len(paths):
            ap.error("batch members must name each raw episode once")
        tasks = manifest["tasks"]
    try:
        report_preflight(output)
    except ReportPreflightError as exc:
        ap.error("Report readiness failed ({}); install specola's declared dependencies".format(exc))
    df = load(str(root), paths=paths)
    by = [key.strip() for key in args.by.split(",") if key.strip()]
    out = summarize(df, by=by)
    try:
        result = export_report(str(root), output_dir=output, paths=paths, batch_tasks=tasks, recordings=recordings)
    except ValueError as exc:
        ap.error(str(exc))
    if args.csv:
        out.to_csv(args.csv, index=False)
    if args.parquet:
        out.to_parquet(args.parquet, index=False)
    print(out.to_string(index=False))
    for name in result["files"]:
        print("  Report artifact: " + str(output / name))


if __name__ == "__main__":
    main()
