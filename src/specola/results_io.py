"""results_io.py — episode parquet writing shared by the facade and CLI.

One step-level file per episode; filenames are collision-proof across
parallel jobs (timestamp + random suffix), so no process ever appends
to a shared file. Falls back to CSV when pyarrow is missing — results
are never silently dropped.
"""

from __future__ import annotations

import csv
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cli_common import bold, dim, green, amber


def _meta_strings(schema_meta: Dict[str, object]) -> Dict[str, str]:
    """Render metadata exactly as the parquet path does.

    The parquet writer stringifies every value on its way into the
    schema, so the CSV sidecar does too: the same metadata must read back
    identically whether or not pyarrow was installed when it was written.
    Keeping native JSON types here instead would make ``success`` a
    boolean in one environment and the string ``"True"`` in the other.
    """
    return {k: str(v) for k, v in schema_meta.items()}

CORE_COLUMNS = frozenset({
    "step", "time", "spec", "authored_spec", "binding", "binding_key", "rho",
    "suspended", "verdict_final", "window_steps",
    "episode_id", "status", "success", "not_applicable",
})

# Columns report.py derives at aggregation time (per-episode verdicts and
# per-group statistics) rather than reading from the row schema. A caller
# meta key colliding with one of these would not surface until report time,
# where it corrupts episode_table / summarize (pandas raises a
# duplicate-label ValueError on the doubled column). This is the single
# authority for that name set; report.py imports it rather than repeating
# the list, so the two can never drift.
DERIVED_REPORT_COLUMNS = frozenset({
    "verdict", "worst_rho", "first_violation_step", "steps",
    "episodes", "sat_rate", "success_rate",
    "mean_worst_rho", "min_worst_rho", "mean_detection_lead",
    "n_success_but_violated", "n_failed_but_satisfied",
})

# Spec-set identity attached to every results file (engine.spec_provenance).
# Reserved like the column names below: caller meta shadowing one of these
# would leave results claiming a provenance that did not produce them, which
# is worse than carrying none at all.
PROVENANCE_KEYS = frozenset({
    "spec_source", "spec_digest", "spec_constants", "extraction_status", "verification_only",
    "audit_only", "schema_only",
})

# Every key a caller-supplied meta entry must not shadow: the row schema's
# own columns, every report-derived column, and the spec-set identity
# attached to the results file. The first two guard the row schema and the
# report; the third guards a claim about what produced the rows, and is not
# a column at all. watch() validates incoming meta against this superset.
RESERVED_META_KEYS = CORE_COLUMNS | DERIVED_REPORT_COLUMNS | PROVENANCE_KEYS


def episode_filename(episode_index: int) -> str:
    """Generate a collision-resistant episode filename.

    Format: <YYYYMMDD_HHMMSS>_ep<N>_<8 hex>.parquet

    The suffix is drawn per call, so same-second calls differ. It is 32
    bits rather than 16 because a colliding filename means one episode's
    results silently overwrite another's, and 16 bits does not carry the
    "no collisions" claim at any realistic scale: 50 draws collide about
    1.8% of the time, which is a birthday bound, not a tail risk. At 32
    bits the same 50 draws collide with probability about 3e-7.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_ep{episode_index}_{secrets.token_hex(4)}.parquet"


def save_rows(
    rows: List[dict],
    path: Path,
    schema_meta: Dict[str, object],
    quiet: bool = False,
) -> Optional[Path]:
    """Write rows to parquet (schema metadata attached), CSV fallback.

    Args:
        rows: List of row dictionaries to save.
        path: Target path (will write .parquet or .csv depending on pyarrow availability).
        schema_meta: Metadata to attach to the parquet schema, or to the
            sidecar described below on the CSV fallback.
        quiet: If True, suppress routine output messages. A provenance
            sidecar that could not be written is reported regardless.

    Returns:
        Path to the written results file (.parquet or .csv), or None when
        there was nothing to write. The CSV fallback cannot write a
        zero-row file — with no rows there are no field names to put in a
        header — so it writes nothing and says so. Returning the path
        anyway would name a file that does not exist, or, if an earlier
        episode's file happens to occupy that name, one holding somebody
        else's results. Callers must treat the return value as "the file
        I wrote", never as "the file I would have written".

        The CSV fallback writes a second file: ``<name>.meta.json``
        beside the CSV, carrying ``schema_meta`` because CSV has nowhere
        to put it. Only the results path is returned, so a caller moving
        or copying "the file I wrote" as a self-contained artifact must
        take the sidecar with it. Its absence is reported, never silent,
        but it is not an error: the rows are already on disk and are
        worth more than the annotation.
    """
    from .conformance.verification import require_ordinary_execution
    require_ordinary_execution(schema_meta)
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        df    = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df)

        # Store run metadata in the Parquet schema so it survives round-trips.
        # Access later with: pq.read_table(path).schema.metadata
        existing = table.schema.metadata or {}
        combined = {**existing,
                    **{k.encode(): str(v).encode() for k, v in schema_meta.items()}}
        table = table.replace_schema_metadata(combined)

        pq.write_table(table, path, compression="snappy")
        if not quiet:
            print(f"  {green('✓')} results → {bold(str(path))}")
            print(dim(f"      {len(rows)} rows  |  "
                      f"pd.read_parquet('{path}')"))
        return path

    except ImportError:
        # Fall back to CSV — same data, larger file. CSV carries no schema
        # metadata, so the run metadata goes to a sidecar rather than being
        # dropped: results whose spec set cannot be identified are not
        # usable as evidence, and losing that silently is the failure this
        # avoids.
        csv_path = path.with_suffix(".csv")

        if not rows:
            if not quiet:
                print(f"  {amber('⚠')}  no rows to write — no results file created")
            return None

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # The rows are already on disk. A metadata failure must not turn a
        # written results file into a raised exception, because the caller
        # (session.end) responds to one by dropping the record's path —
        # leaving a real file nobody holds a reference to. Report it
        # loudly instead, quiet or not: results whose spec set cannot be
        # identified are still results, but they are not evidence.
        meta_path = csv_path.with_suffix(".meta.json")
        try:
            with open(meta_path, "w") as f:
                json.dump(
                    _meta_strings(schema_meta), f, indent=2, sort_keys=True,
                )
        except Exception as exc:
            print(f"  {amber('⚠')}  results written, but their provenance "
                  f"sidecar failed: {exc}")
            print(f"      {bold(str(csv_path))} has no {meta_path.name}")

        if not quiet:
            print(f"  {amber('⚠')}  pyarrow not installed — saved as CSV instead:")
            print(f"      {bold(str(csv_path))}")
            print(dim("      pip install pyarrow  to get Parquet output"))
        return csv_path
