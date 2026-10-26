"""report.py — aggregation over episode result parquets (`specola report` CLI).

Reads the parquet files written by `results_io.save_rows` (Task 5's schema:
one row per monitor evaluation, `verdict_final` rows carrying the settled
robustness value for a `(monitor spec, binding)` pair) and rolls them up
into per-episode verdicts and per-group satisfaction statistics.

pandas/pyarrow are optional for the `specola` package as a whole; they
are only required to actually run this module. All pandas imports here are
local to functions so importing `specola.report` never forces the
dependency — only calling into it does. The CLI entry point (`main`) turns
a missing pandas/pyarrow into a friendly `SystemExit` pointing at the
`report` extra.

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
    # pandas/pyarrow are optional for the package; only imported for real
    # inside the functions below. This import is never executed at
    # runtime — it exists so type checkers can resolve the "pd.DataFrame"
    # annotations used throughout this module.
    import pandas as pd

_REPORT_HINT = "specola report requires pandas/pyarrow: pip install 'specola[report]'"


def _require_pandas():
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit(_REPORT_HINT) from exc


def load(results_dir: str) -> "pd.DataFrame":
    """Recursively glob `*.parquet` under `results_dir` and concatenate.

    Tolerates old-format files (missing Task 5 columns): `authored_spec`
    falls back to `spec`; `binding_key` falls back to the display `binding`
    label; `episode_id` falls back to the source file path;
    `success`/`status`/`not_applicable` become null/"completed"/False where
    absent. Never touches simulators — this is a pure file reader.
    """
    import pandas as pd

    paths = sorted(Path(results_dir).rglob("*.parquet"))
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        if "episode_id" not in frame.columns:
            frame = frame.copy()
            frame["episode_id"] = str(path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=sorted(CORE_COLUMNS))

    df = pd.concat(frames, ignore_index=True, sort=False)

    if "authored_spec" not in df.columns:
        df["authored_spec"] = float("nan")
    df["authored_spec"] = df["authored_spec"].fillna(df["spec"])

    if "success" not in df.columns:
        df["success"] = float("nan")
    # No fallback value for success — missing stays null, per spec.

    if "status" not in df.columns:
        df["status"] = "completed"
    else:
        df["status"] = df["status"].fillna("completed")

    if "not_applicable" not in df.columns:
        df["not_applicable"] = False
    else:
        df["not_applicable"] = df["not_applicable"].fillna(False)

    # binding_key is the entity-ID identity the verdict rule keys on. Old
    # files (written before it existed) fall back to the display binding
    # label — safe there, since those runs predate the co-labeled-binding
    # collision the key was added to fix.
    if "binding_key" not in df.columns:
        df["binding_key"] = df["binding"]
    else:
        df["binding_key"] = df["binding_key"].fillna(df["binding"])

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

    success_numeric = group["success"].astype("float64")
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

    ap = argparse.ArgumentParser(prog="specola report")
    ap.add_argument("results_dir")
    ap.add_argument("--by", default="", help="comma-separated meta columns")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--parquet", default=None)
    args = ap.parse_args()

    _require_pandas()

    df = load(args.results_dir)
    by = [k.strip() for k in args.by.split(",") if k.strip()]
    out = summarize(df, by=by)
    if args.csv:
        out.to_csv(args.csv, index=False)
    if args.parquet:
        out.to_parquet(args.parquet, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
