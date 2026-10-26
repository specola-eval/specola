"""
X1: the CLI summary, the Session record, and specola report must derive
the SAME verdict for the same episode.

Every number in the paper's evaluation flows through one of these three
paths, and the reduction rule was implemented once (CLI) and re-implemented
twice (Session._compute_outcomes, report.episode_table) with docstring
promises of agreement. A divergence found after results exist invalidates
tables, so this suite drives all three over identical episodes through the
dataset adapter, where every case is exact.

The structural cases: a verdict settled long before the window would fill,
in both directions; a binding that dipped and recovered; mixed
final/pending bindings; a spec whose class is absent; and no final verdict
at all. Each case asserts the three paths agree AND that the agreed
verdict is the semantically correct one.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import pytest

from specola.adapters.dataset import DatasetAdapter
from specola.session import Monitor

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from specola import report as report_module

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Summary lines look like "  name   VERDICT   ..." after ANSI stripping.
_VERDICT_LINE = re.compile(
    r"^\s{2}(?P<name>\S+)\s+(?P<verdict>PASS|FAIL|UNDETERMINED|NOT APPLICABLE)\b"
)


def _write_csv(path: Path, per_step_rows) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["time", "entity_id", "entity_class", "px", "py", "sig"]
        )
        for step, rows in enumerate(per_step_rows):
            for entity_id, entity_class, sig in rows:
                writer.writerow(
                    [step * 0.1, entity_id, entity_class, 0.0, 0.0, sig]
                )


def _cli_verdicts(tmp_path: Path, csv_path: Path, specs_path: Path, steps: int):
    """Run the real CLI; return {spec: verdict} parsed from the summary."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "specola.run_monitor",
            "--sim", "dataset", "--task", str(csv_path),
            "--specs", str(specs_path), "--steps", str(steps),
            "--no-live",
        ],
        capture_output=True, text=True, cwd=tmp_path, timeout=120,
    )
    # The CLI exits 0 here; 1 would mean a spec-load failure or an
    # uncaught exception, and the split below would then die with an
    # IndexError that hides stderr.
    assert proc.returncode == 0, proc.stderr[-2000:]
    text = _ANSI.sub("", proc.stdout)
    summary = text.split("Episode summary", 1)[1]
    verdicts = {}
    for line in summary.splitlines():
        match = _VERDICT_LINE.match(line)
        if match:
            verdicts[match.group("name")] = match.group("verdict")
    assert verdicts, f"no summary verdicts parsed:\n{text[-2000:]}"
    return verdicts


def _session_verdicts(csv_path: Path, specs_path: Path, steps: int, results_dir):
    monitor = Monitor(str(specs_path), results_dir=str(results_dir))
    session = monitor.watch(DatasetAdapter(str(csv_path)))
    for _ in range(steps):
        session.step()
    record = session.end(success=True)
    return {name: outcome.verdict for name, outcome in record.specs.items()}


def _report_verdicts(results_dir) -> dict:
    table = report_module.episode_table(report_module.load(str(results_dir)))
    assert len(table["episode_id"].unique()) == 1
    return dict(zip(table["spec"], table["verdict"]))


# Map the three vocabularies onto one for comparison.
_CANON = {
    "PASS": "pass", "FAIL": "fail", "UNDETERMINED": "undetermined",
    "NOT APPLICABLE": "not_applicable",
    "pass": "pass", "fail": "fail", "undetermined": "undetermined",
    "not_applicable": "not_applicable",
}


def _run_all_three(tmp_path, per_step_rows, specs_yaml, steps=None):
    # The adapter's reset consumes the first frame, so N frames drive
    # N - 1 monitored steps.
    steps = len(per_step_rows) - 1 if steps is None else steps
    csv_path = tmp_path / "trace.csv"
    specs_path = tmp_path / "specs.yaml"
    _write_csv(csv_path, per_step_rows)
    specs_path.write_text(specs_yaml)

    cli = _cli_verdicts(tmp_path, csv_path, specs_path, steps)
    session_dir = tmp_path / "session_results"
    ses = _session_verdicts(csv_path, specs_path, steps, session_dir)
    rep = _report_verdicts(session_dir)

    canon = {}
    for name in ses:
        canon[name] = {
            "cli": _CANON[cli[name]],
            "session": _CANON[ses[name]],
            "report": _CANON.get(rep.get(name), rep.get(name)),
        }
    return canon


def _assert_agree(canon, spec, expected):
    got = canon[spec]
    assert got["cli"] == got["session"] == got["report"] == expected, (
        f"{spec}: divergence or wrong verdict: {got}, expected {expected!r}"
    )


STEPS = 6


def test_settled_fail_beats_the_unfilled_window(tmp_path):
    """A violated G settles at step 1 against a window far longer than the
    episode. The verdict is decided; reporting it as undetermined because
    the episode is shorter than the window discards a settled answer."""
    rows = [[(1, "object", 1.0)]] + [[(1, "object", -1.0)]] * (STEPS - 1)
    specs = 'specs:\n- name: g_fail\n  formula: "G[0,50](object_*[sig] >= 0)"\n'
    canon = _run_all_three(tmp_path, rows, specs)
    _assert_agree(canon, "g_fail", "fail")


def test_settled_pass_beats_the_unfilled_window(tmp_path):
    rows = [[(1, "object", -1.0)], [(1, "object", 1.0)]] + [
        [(1, "object", -1.0)]
    ] * (STEPS - 2)
    specs = 'specs:\n- name: f_pass\n  formula: "F[0,50](object_*[sig] >= 0)"\n'
    canon = _run_all_three(tmp_path, rows, specs)
    _assert_agree(canon, "f_pass", "pass")


def test_one_failing_binding_fails_the_spec_despite_pending_siblings(tmp_path):
    # Two objects: one violates (settles), the other stays positive with
    # the window still filling. Any final failure fails the spec.
    rows = [
        [(1, "object", -1.0), (2, "object", 1.0)]
        for _ in range(STEPS)
    ]
    specs = 'specs:\n- name: g_mixed\n  formula: "G[0,50](object_*[sig] >= 0)"\n'
    canon = _run_all_three(tmp_path, rows, specs)
    _assert_agree(canon, "g_mixed", "fail")


def test_recovered_after_dip_is_judged_by_the_last_final_value(tmp_path):
    # Window of 3: the dip at step 1 settles a FAIL early, but by the last
    # step the window has slid past it and the final value is positive.
    # All three paths judge the episode by where the binding ended up.
    rows = (
        [[(1, "object", 1.0)], [(1, "object", -1.0)]]
        + [[(1, "object", 1.0)]] * (STEPS - 2)
    )
    specs = 'specs:\n- name: g_recover\n  formula: "G[0,3](object_*[sig] >= 0)"\n'
    canon = _run_all_three(tmp_path, rows, specs)
    _assert_agree(canon, "g_recover", "pass")


def test_never_bound_class_is_undetermined_everywhere(tmp_path):
    """The dataset adapter's vocabulary is open, so a spec over an absent
    class compiles a manager that never binds. It must show as
    undetermined in all three paths, not vanish: before the marker-row
    fix it produced no rows at all and was silently missing from the
    report."""
    rows = [[(1, "object", 1.0)] for _ in range(STEPS)]
    specs = (
        "specs:\n"
        '- name: g_ok\n  formula: "G[0,3](object_*[sig] >= 0)"\n'
        '- name: micro\n  formula: "G[0,3](microwave_*[sig] >= 0)"\n'
    )
    canon = _run_all_three(tmp_path, rows, specs)
    _assert_agree(canon, "g_ok", "pass")
    _assert_agree(canon, "micro", "undetermined")


def test_skipped_class_is_not_applicable_everywhere(tmp_path):
    """The mock adapter's vocabulary is closed, so the same absent-class
    spec is skipped as unsatisfiable and recorded not_applicable, in all
    three paths."""
    from specola.adapters.fake import _MockAdapter

    specs_path = tmp_path / "specs.yaml"
    specs_path.write_text(
        "specs:\n"
        '- name: mass_ok\n  formula: "G[0,3](object_*[mass] >= 0.05)"\n'
        '- name: micro\n  formula: "G[0,3](microwave_*[door] <= 0.1)"\n'
    )

    proc = subprocess.run(
        [
            sys.executable, "-m", "specola.run_monitor",
            "--sim", "mock", "--specs", str(specs_path),
            "--steps", "6", "--no-live",
        ],
        capture_output=True, text=True, cwd=tmp_path, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    text = _ANSI.sub("", proc.stdout)
    summary = text.split("Episode summary", 1)[1]
    cli = {}
    for line in summary.splitlines():
        match = _VERDICT_LINE.match(line)
        if match:
            cli[match.group("name")] = match.group("verdict")

    session_dir = tmp_path / "session_results"
    monitor = Monitor(str(specs_path), results_dir=str(session_dir))
    session = monitor.watch(_MockAdapter())
    for _ in range(6):
        session.step()
    record = session.end(success=True)
    ses = {n: o.verdict for n, o in record.specs.items()}
    rep = _report_verdicts(session_dir)

    for name, expected in (("mass_ok", "pass"), ("micro", "not_applicable")):
        got = {
            "cli": _CANON[cli[name]],
            "session": _CANON[ses[name]],
            "report": _CANON.get(rep.get(name), rep.get(name)),
        }
        assert got["cli"] == got["session"] == got["report"] == expected, (
            f"{name}: {got}, expected {expected!r}"
        )


def test_fanout_cli_is_per_monitor_while_session_reduces_per_spec(tmp_path):
    """Signal fan-out (``sig_*``) is the one place the paths differ in
    granularity by design: the CLI summary prints one line per fanned-out
    monitor and no roll-up line named after the authored spec, while
    Session and report reduce over all of the spec's monitors. Pinned so
    a future change to either side is a conscious one (internal review
    finding: this suite's name-keyed comparison cannot express the case,
    so without this test the divergence would be invisible here)."""
    csv_path = tmp_path / "trace.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["time", "entity_id", "entity_class", "px", "py", "sig_a", "sig_b"]
        )
        for step in range(STEPS):
            writer.writerow([step * 0.1, 1, "object", 0.0, 0.0, 1.0, -1.0])
    specs_path = tmp_path / "specs.yaml"
    specs_path.write_text(
        'specs:\n- name: fan\n  formula: "G[0,3](object_*[sig_*] >= 0)"\n'
    )

    proc = subprocess.run(
        [
            sys.executable, "-m", "specola.run_monitor",
            "--sim", "dataset", "--task", str(csv_path),
            "--specs", str(specs_path), "--steps", str(STEPS - 1),
            "--no-live",
        ],
        capture_output=True, text=True, cwd=tmp_path, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    summary = _ANSI.sub("", proc.stdout).split("Episode summary", 1)[1]

    fan_lines = [
        line for line in summary.splitlines()
        if line.lstrip().startswith("fan")
    ]
    assert any("sig_a" in line and "PASS" in line for line in fan_lines)
    assert any("sig_b" in line and "FAIL" in line for line in fan_lines)
    # No roll-up: every fan line names a monitor, none is the bare spec.
    assert fan_lines and all("sig_" in line for line in fan_lines)

    session_dir = tmp_path / "session_results"
    monitor = Monitor(str(specs_path), results_dir=str(session_dir))
    session = monitor.watch(DatasetAdapter(str(csv_path)))
    for _ in range(STEPS - 1):
        session.step()
    record = session.end(success=True)
    assert record.specs["fan"].verdict == "fail"
    assert _report_verdicts(session_dir)["fan"] == "fail"


def test_never_final_is_undetermined_everywhere(tmp_path):
    # Positive throughout, window longer than the episode, G never settles.
    rows = [[(1, "object", 1.0)] for _ in range(STEPS)]
    specs = 'specs:\n- name: g_pend\n  formula: "G[0,50](object_*[sig] >= 0)"\n'
    canon = _run_all_three(tmp_path, rows, specs)
    _assert_agree(canon, "g_pend", "undetermined")
