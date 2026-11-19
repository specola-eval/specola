# Read and regenerate run reports

You receive CSV and PDF reports when `specola run` finalizes an episode and
releases its adapter. Completed, truncated and failed episodes with finalized rows
keep their recorded status. An abandoned worker does not publish a report from
buffers it can still change. RoboDojo releases its environment at this boundary;
its Kit application remains alive until process exit.

Reporting uses baseline dependencies `pandas>=2.0.3,<4` and
`matplotlib>=3.7.5,<4`. PyArrow remains optional for additional raw Parquet files
and historical Parquet input. Before adapter construction, a bounded preflight
checks the table library, PDF backend, bundled DejaVu Sans font and destination.
It removes its temporary PDF and any empty directories it creates. Ordinary
`specola doctor` does not import the plotting libraries.

## Find the artifacts

Your run directory contains a required raw episode CSV and its `.meta.json`
sidecar, plus a matching Parquet file when PyArrow is available. The sidecar
keeps saved formula text, constants, function provenance and column types. Copy
it with the CSV. Empty episodes still receive CSV headers and metadata.

The offline exporter writes these artifacts:

| File | Contents |
| --- | --- |
| `report-run.pdf` | Overview, available outcome summaries, formulas, traces, scene images and provenance |
| `report-episode-summary.csv` | One outcome per episode and authored spec |
| `report-task-spec-summary.csv` | Counts and denominators by task/spec and available policy/seed identity |
| `report-plot-data.csv` | Full raw rows underlying the traces, including missing and infinite values |
| `report-trace-summary.csv` | Eligible binding groups, finite-pair counts, plot pages and unavailable reasons |
| `report-specifications.csv` | Saved monitored formulas and constants, with their raw source paths |
| `report-scene-images.csv` | Image availability, camera, actual sampled step and rendered frame hash |
| `report-manifest.json` | Explicit raw sources, saved metadata, page checks, artifact hashes and export metrics |

Each output replaces its predecessor atomically. The manifest publishes last;
its hashes expose an interrupted replacement across several files. Raw results
remain separate from these derived outputs. A report error preserves the raw
episode's status, writes `report-error.log` when possible and gives an otherwise
successful run exit status 2. A controlled run failure keeps its original error.

## Regenerate a report

Use a raw episode file or a directory of raw results:

```bash
specola report results/my-run
specola report results/my-run/episode.csv --output-dir regenerated-report
```

Directory loading validates observation columns, excludes `report-*` exports
and reads a CSV/Parquet sibling file stem once, preferring CSV. It preserves
identity strings such as `001`, `NA`, `null` and empty display bindings. Invalid
boolean values fail loading. Historical missing formula text remains labeled
unavailable; the exporter never reopens a mutable specification source file to
guess what a past run monitored.

The existing `--csv` and `--parquet` options still write aggregate summaries.
Use `--by task,seed` to group that legacy summary. The automatic PDF/CSV bundle
uses the same episode outcome rules.

## Declare a combined batch

Use a JSON file with explicit raw members and the exact saved task population:

```json
{
  "members": ["task-a/episode.csv", "task-b/episode.csv"],
  "tasks": ["task-a", "task-b"]
}
```

```bash
specola report results --batch-manifest results/batch.json --output-dir batch-report
```

Member paths resolve beside the manifest. Duplicate raw file stems or episode
identities fail validation. Declared task identities must match saved `task_id`
values, or legacy `task` values when `task_id` is absent. A directory scan alone
does not establish a benchmark or batch population.

## Include a saved scene image

Automatic reports associate the current finalized dashboard recording
explicitly. For regeneration, pass the association yourself:

```bash
specola report episode.csv --recording episode.csv=recording_000
```

The existing recording reader validates saved run, task and episode status
against the raw metadata. A JPEG camera contributes its latest indexed frame,
labeled with camera and actual sampled step. A missing recording remains
unavailable. Saved encoded video remains labeled unsupported because the
existing offline reader supplies encoded bytes without a Python image decoder.
You do not need a video dependency for ordinary report generation. Component
signal plots remain unavailable because the current raw/recording schema does
not save those signal series or their units.

## Interpret the values and checks

Specification satisfaction uses `pass / (pass + fail)`. Undetermined and not
applicable outcomes keep separate counts. Native success uses episodes with
known success flags as its denominator. The exporter never substitutes a spec
verdict for native success. Traces keep settled, provisional and suspended
states separate and do not pool robustness across formulas or bindings.
Groups without a finite time/robustness pair appear in a concise unavailable
section. Their identities, row counts and reasons remain in the trace summary
CSV and manifest; an empty axis does not stand in for an observed trace.

PDF pages use A4 portrait and a minimum 10 pt font. Long text wraps and
paginates; it does not shrink. Literal formula text disables math parsing and
TeX. Local PDF settings request embedded TrueType subsets. Bounds and glyph
checks complement rendered visual review; they do not establish readability for
an unmeasured future label population.

The `report_build_seconds` and `report_build_python_tracemalloc_peak_bytes`
metrics cover result loading and CSV/PDF construction. They exclude the initial
imports in `export_report`, artifact hashing, manifest serialization and publication. These fields
do not measure the complete export call. The manifest distinguishes this build
interval from whole-process lifetime RSS. RSS includes prior monitoring and any native pools
that remain alive. It is not report-exclusive allocation. Python allocations
exclude native allocations; a caller already tracing allocations receives a
null report peak rather than having its tracing state reset.

You can exercise the packaged synthetic fixture through `tests/report_acceptance.py`
in a source checkout. The JSON resource lives at
`specola/data/report-fixture.json` inside the installed package. It contains
synthetic values and no native success claim. The acceptance script renders
each PDF page with `pdftoppm` and retains a page inventory for independent review:

```bash
python tests/report_acceptance.py --output /absolute/writable/report-evidence
```

The implementation uses public `matplotlib` PDF/Agg, text extent and image APIs
and pandas table operations. The acceptance run uses pandas 3.0.5 and
`matplotlib` 3.11.1 on Python 3.12.13. The image API research also exercises
`matplotlib` 3.10.3 with Pillow 11.3.0. The
[PDF API](https://matplotlib.org/3.10.3/api/backend_pdf_api.html),
[text API](https://matplotlib.org/3.10.3/api/text_api.html) and
[image API](https://matplotlib.org/3.10.3/api/image_api.html) document the rendering
interfaces. The declared Python 3.8 dependency floor has not been exercised by
those acceptance runs.
