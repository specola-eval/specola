"""specola report aggregation over synthetic episode parquets."""
import pytest

pd = pytest.importorskip("pandas")

from specola import report, results_io


def _write_episode(dirpath, episode_id, spec, finals, success, steps,
                   meta=None, not_applicable=False):
    """finals: list of (step, rho) final-verdict rows for one binding."""
    rows = []
    if not_applicable:
        rows.append({
            "step": -1, "time": float("nan"), "spec": spec,
            "authored_spec": spec, "binding": "", "rho": float("nan"),
            "suspended": False, "verdict_final": False, "window_steps": 0,
            "not_applicable": True,
        })
    else:
        for step, rho in finals:
            rows.append({
                "step": step, "time": step * 0.05, "spec": spec,
                "authored_spec": spec, "binding": "obj", "rho": rho,
                "suspended": False, "verdict_final": True,
                "window_steps": 20, "not_applicable": False,
            })
        # Make episode length visible to episode_table: one extra ordinary
        # (non-final) row at the last step of the episode. episode_table
        # derives steps = max(step) + 1 over non-not_applicable rows.
        rows.append({
            "step": steps - 1, "time": (steps - 1) * 0.05, "spec": spec,
            "authored_spec": spec, "binding": "obj", "rho": float("nan"),
            "suspended": True, "verdict_final": False, "window_steps": 20,
            "not_applicable": False,
        })
    for row in rows:
        row.update({"episode_id": episode_id, "status": "completed",
                    "success": success, **(meta or {})})
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / results_io.episode_filename(0)
    results_io.save_rows(rows, path, {}, quiet=True)
    return path


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results"
    m1 = {"train_step": 1000, "task": "t"}
    m2 = {"train_step": 2000, "task": "t"}
    # ckpt 1000: one failing episode (viol at step 25 of 100), success=False
    _write_episode(d / "a", "ep1", "s1", [(20, 0.5), (25, -0.2)], False, meta=m1, steps=100)
    # ckpt 1000: success=True but spec violated -> disagreement
    _write_episode(d / "a", "ep2", "s1", [(30, -0.1)], True, meta=m1, steps=100)
    # ckpt 2000: clean pass
    _write_episode(d / "b", "ep3", "s1", [(90, 0.9)], True, meta=m2, steps=100)
    # not_applicable spec on one episode
    _write_episode(d / "b", "ep3b", "s2", [], True, meta=m2, steps=100,
                   not_applicable=True)
    return str(d)


def test_load_recursive(results_dir):
    df = report.load(results_dir)
    assert set(df["episode_id"]) == {"ep1", "ep2", "ep3", "ep3b"}


def test_episode_table_verdicts(results_dir):
    ep = report.episode_table(report.load(results_dir))
    v = dict(zip(ep["episode_id"] + "/" + ep["spec"], ep["verdict"]))
    assert v["ep1/s1"] == "fail"      # last final rho for the binding is -0.2
    assert v["ep2/s1"] == "fail"
    assert v["ep3/s1"] == "pass"
    assert v["ep3b/s2"] == "not_applicable"


def test_summarize_by_train_step(results_dir):
    out = report.summarize(report.load(results_dir), by=["train_step"])
    row1000 = out[(out["train_step"] == 1000) & (out["spec"] == "s1")].iloc[0]
    assert row1000["episodes"] == 2
    assert row1000["sat_rate"] == 0.0
    assert row1000["n_success_but_violated"] == 1
    row2000 = out[(out["train_step"] == 2000) & (out["spec"] == "s1")].iloc[0]
    assert row2000["sat_rate"] == 1.0
    # not_applicable excluded from rates
    assert "s2" not in set(out[out["sat_rate"].notna()]["spec"])


def test_old_format_files_tolerated(tmp_path):
    """Files missing the new columns aggregate with nulls, no crash."""
    rows = [{"step": 0, "time": 0.0, "spec": "s", "binding": "b",
             "rho": 0.1, "suspended": False, "verdict_final": True,
             "window_steps": 5}]
    p = tmp_path / "old.parquet"
    results_io.save_rows(rows, p, {}, quiet=True)
    df = report.load(str(tmp_path))
    assert df["success"].isna().all()
    # No binding_key column in an old file -> falls back to `binding`, so a
    # single-binding episode still aggregates correctly.
    ep = report.episode_table(df)
    assert list(ep["verdict"]) == ["pass"]


def _final_row(spec, binding, binding_key, step, rho):
    return {
        "step": step, "time": step * 0.05, "spec": spec,
        "authored_spec": spec, "binding": binding, "binding_key": binding_key,
        "rho": rho, "suspended": False, "verdict_final": True,
        "window_steps": 20, "not_applicable": False,
    }


def test_colliding_binding_labels_keep_fail_verdict(tmp_path):
    """Finding C2 (report side): two bindings sharing one display label —
    one ending negative, the other positive at a later step — must yield
    'fail'. Pre-fix (groupby on `binding`) the shared label keeps only the
    later positive row and reports 'pass'; keyed on `binding_key` the
    negative binding survives and the verdict is 'fail'.
    """
    d = tmp_path / "results"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    # Failing binding: last final rho negative, at an EARLIER step.
    for step, rho in [(19, -0.1), (23, -0.1)]:
        rows.append(_final_row("s", "widget", "1", step, rho))
    # Passing binding: last final rho positive, at a LATER step (would win a
    # label-keyed idxmax and hide the failure).
    for step, rho in [(19, 0.05), (39, 0.05)]:
        rows.append(_final_row("s", "widget", "2", step, rho))
    for row in rows:
        row.update({"episode_id": "ep", "status": "completed", "success": True})
    results_io.save_rows(rows, d / results_io.episode_filename(0), {}, quiet=True)

    ep = report.episode_table(report.load(str(d)))
    verdict = dict(zip(ep["episode_id"] + "/" + ep["spec"], ep["verdict"]))
    assert verdict["ep/s"] == "fail"


def test_same_run_id_episodes_stay_distinct(tmp_path):
    """Finding C1: two Monitors sharing one run_id (parallel same-second
    invocations, or specola run called twice in the same second) must
    still produce distinct episode_ids. Pre-fix both episode_ids were
    '<run_id>_ep0' and report.episode_table silently merged the two
    episodes into one (episodes=1, wrong success_rate); the episode_id salt
    keeps them separate.
    """
    from specola.adapters.fake import _MockAdapter
    from specola.session import Monitor

    specs = tmp_path / "specs.yaml"
    specs.write_text(
        'specs:\n'
        '  - name: mass_cap\n'
        '    formula: "G[0,1s](object_*[mass] <= 0.3)"\n'
    )
    rdir = tmp_path / "results"
    run_id = "20260101_000000"   # identical for both -> exercises episode_id salt
    for success in (True, False):
        monitor = Monitor(str(specs), results_dir=str(rdir), run_id=run_id)
        session = monitor.watch(_MockAdapter())
        for _ in range(30):
            session.step()
        session.end(success=success)

    df = report.load(str(rdir))
    ep = report.episode_table(df)
    assert len(set(ep["episode_id"])) == 2

    out = report.summarize(df, by=[])
    row = out[out["spec"] == "mass_cap"].iloc[0]
    assert row["episodes"] == 2
    assert row["success_rate"] == 0.5


def test_summarize_refuses_a_missing_by_column(results_dir):
    df = report.load(results_dir)
    with pytest.raises(SystemExit, match="--by column not in the results: nope"):
        report.summarize(df, by=["nope"])
