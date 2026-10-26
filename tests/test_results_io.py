import re

from specola import results_io


def test_episode_filename_shape():
    name = results_io.episode_filename(7)
    assert re.fullmatch(r"\d{8}_\d{6}_ep7_[0-9a-f]{8}\.parquet", name)


def test_episode_filename_draws_a_fresh_suffix_per_call(monkeypatch):
    """
    The property that makes same-second calls differ: the suffix is drawn
    on every call rather than once per process or per second.

    Asserted by driving the entropy source, not by drawing samples and
    hoping they differ. The previous version of this test took 50 real
    draws and required all 50 distinct, which is a birthday trial: at the
    16-bit suffix it then used, that failed about 1.8% of runs, and even
    at 32 bits an N-distinct assertion stays probabilistic in principle.
    """
    suffixes = iter(["aaaaaaaa", "bbbbbbbb"])
    monkeypatch.setattr(
        results_io.secrets, "token_hex", lambda _n: next(suffixes),
    )

    first = results_io.episode_filename(0)
    second = results_io.episode_filename(0)

    assert first.endswith("_aaaaaaaa.parquet")
    assert second.endswith("_bbbbbbbb.parquet")
    assert first != second


def test_episode_filename_suffix_is_wide_enough_to_carry_the_claim():
    # A colliding filename overwrites another episode's results, so the
    # suffix width is the guarantee. 32 bits puts 50 same-second draws at
    # a collision probability of roughly 3e-7; 16 bits put it at 1.8%.
    name = results_io.episode_filename(0)
    assert name.endswith(".parquet")
    # str.removesuffix needs 3.9; pyproject declares >=3.8.
    suffix = name[: -len(".parquet")].rsplit("_", 1)[1]

    assert len(suffix) == 8


def test_save_rows_roundtrip(tmp_path):
    pytest_pd = __import__("pytest").importorskip("pandas")
    rows = [
        {"step": 0, "time": 0.0, "spec": "s", "rho": 1.0},
        {"step": 1, "time": 0.05, "spec": "s", "rho": -0.5},
    ]
    path = tmp_path / "ep.parquet"
    written = results_io.save_rows(rows, path, {"task": "t1"}, quiet=True)
    df = pytest_pd.read_parquet(written)
    assert list(df["step"]) == [0, 1]


def test_csv_fallback_reports_nothing_written_for_an_empty_row_set(
    tmp_path, monkeypatch,
):
    """
    The CSV fallback cannot write a zero-row file: with no rows there are
    no field names for a header. It used to return the path anyway, which
    named a file that did not exist -- or, if an earlier episode's file
    happened to occupy that name, one holding somebody else's results.
    """
    import sys

    from specola import results_io

    monkeypatch.setitem(sys.modules, "pyarrow", None)

    target = tmp_path / "episode.parquet"
    stale = target.with_suffix(".csv")
    stale.write_text("old,results\n1,2\n")

    written = results_io.save_rows([], target, {}, quiet=True)

    assert written is None
    # the earlier file is left exactly as it was, not truncated or claimed
    assert stale.read_text() == "old,results\n1,2\n"


def test_csv_fallback_still_writes_and_reports_a_non_empty_row_set(
    tmp_path, monkeypatch,
):
    import sys

    from specola import results_io

    monkeypatch.setitem(sys.modules, "pyarrow", None)

    target = tmp_path / "episode.parquet"
    written = results_io.save_rows(
        [{"step": 0, "rho": 1.5}], target, {}, quiet=True,
    )

    assert written == target.with_suffix(".csv")
    assert written.exists()
    assert "rho" in written.read_text()
