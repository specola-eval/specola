"""Write episode CSV with provenance and optional Parquet.

Collision-resistant episode filenames keep parallel jobs separate. Each raw
representation is published atomically; metadata accompanies the CSV.
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
    "audit_only", "schema_only", "spec_formulas", "spec_functions", "spec_functions_active",
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
) -> Path:
    """Write required CSV and metadata, plus Parquet when available.

    The returned path identifies the preferred raw representation actually
    written. CSV and its metadata sidecar remain available beside Parquet.
    Empty episodes carry the core headers and episode metadata.
    """
    import os
    import tempfile

    from .conformance.verification import require_ordinary_execution
    require_ordinary_execution(schema_meta)
    path = Path(path)
    csv_path = path.with_suffix(".csv")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if not columns:
        columns = sorted(CORE_COLUMNS)
    metadata = _meta_strings(schema_meta)
    metadata["raw_results_schema"] = "1"
    metadata["raw_column_types"] = json.dumps({
        column: type(next((row[column] for row in rows
                           if row.get(column) is not None), "")).__name__
        for column in columns
    }, sort_keys=True)

    def publish(destination, write):
        fd, name = tempfile.mkstemp(prefix="." + destination.name, dir=destination.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            write(temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def write_csv(temporary):
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    publish(csv_path, write_csv)
    meta_path = csv_path.with_suffix(".meta.json")
    try:
        publish(meta_path, lambda temporary: temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8",
        ))
    except Exception as exc:
        print(f"  {amber('⚠')} results written, but their provenance sidecar failed: {exc}")
        print(f"      {bold(str(csv_path))} has no {meta_path.name}")

    preferred = csv_path
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pass
    else:
        try:
            table = pa.Table.from_pandas(pd.DataFrame(rows, columns=columns))
            table = table.replace_schema_metadata({
                **(table.schema.metadata or {}),
                **{key.encode(): value.encode() for key, value in metadata.items()},
            })
            publish(path, lambda temporary: pq.write_table(
                table, temporary, compression="snappy",
            ))
            preferred = path
        except Exception as exc:
            print(f"  {amber('⚠')} optional Parquet write failed: {exc}; CSV preserved at {csv_path}")
    if not quiet:
        print(f"  {green('✓')} results → {bold(str(preferred))}")
        print(dim(f"      {len(rows)} rows; required CSV: {csv_path}"))
    return preferred
