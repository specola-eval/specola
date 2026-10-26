"""Authoring in the browser is authoring in the terminal.

The terminal writer's workflow is draft, validate, keep, accumulate,
save, and three properties make it trustworthy rather than merely
convenient: a validation belongs to the revision it validated, the
report is the checker's own text, and the destination is chosen
deliberately rather than defaulted into. A second authoring surface that
dropped any of those would accept specifications the first one refuses.

The destination is the one deliberate difference. The terminal asks for
a path; the browser has none to give, because a page that could name a
file could name any file. The server derives it from the recording it
was started on.
"""

from __future__ import annotations

import pytest

pytest.importorskip("flask_socketio")

import yaml                                          # noqa: E402

from specola.capabilities import reset_adapter        # noqa: E402
from specola.cli_common import _MockAdapter           # noqa: E402
from specola.dashboard.recording import (             # noqa: E402
    RecordingWriter,
)
from specola.dashboard.server import DashboardServer  # noqa: E402
from specola.discovery_capture import (               # noqa: E402
    discovery_payload,
)


def _server(tmp_path, *, with_discovery=True):
    recording = tmp_path / "run" / "recording_000"
    recording.mkdir(parents=True)
    writer = RecordingWriter(
        recording, {"run_id": "r", "status": "completed"}
    )
    if with_discovery:
        adapter = _MockAdapter()
        writer.write_discovery(discovery_payload(
            adapter, reset_adapter(adapter), simulator="mock",
            task=None, effective_dt=adapter.dt,
        ))
    return DashboardServer(recording, mode="view")


def _accepted_draft(server, name="held"):
    """One draft this scene really accepts, found the same way a person
    would: from the palette's own starter."""
    palette = server._resolve_palette("r", {"entity_class": "object"})
    starter = next(
        form["starter"] for form in palette["forms"] if form["available"]
    )
    return {"name": name, "formula": starter["formula"],
            "constants": starter.get("constants") or {},
            "symmetric": bool(starter.get("symmetric"))}


# -- the check ---------------------------------------------------------------


def test_the_starter_the_palette_offers_is_one_this_scene_accepts(tmp_path):
    """The whole point of generating it server-side."""
    server = _server(tmp_path)
    draft = _accepted_draft(server)
    context, error = server._check_context()
    assert context is not None, error

    from specola.authoring import check_draft

    result = check_draft(
        context, name=draft["name"], formula=draft["formula"],
        constants=draft["constants"], symmetric=draft["symmetric"],
    )
    assert result.accepted, result.report


def test_the_report_is_the_checkers_own_text(tmp_path):
    """Not a summary. It says which keys bound and in what units, which
    is the part an author reads to learn why a formula binds what it
    binds."""
    server = _server(tmp_path)
    context, _ = server._check_context()

    from specola.authoring import check_draft

    draft = _accepted_draft(server)
    result = check_draft(
        context, name="r", formula=draft["formula"],
        constants=draft["constants"], symmetric=draft["symmetric"],
    )
    assert result.report.strip()
    assert len(result.report) > 40


def test_a_run_without_a_snapshot_says_so_instead_of_guessing(tmp_path):
    server = _server(tmp_path, with_discovery=False)
    context, error = server._check_context()
    assert context is None
    assert "discovery snapshot" in error


# -- the save ----------------------------------------------------------------


def test_the_browser_cannot_name_the_file(tmp_path):
    """Derived from the recording, so there is nothing to traverse."""
    server = _server(tmp_path)
    assert server.authored_dir == tmp_path / "run" / "authored"

    context, _ = server._check_context()
    reply = server._save_specs(context, [_accepted_draft(server)], "ask")
    assert reply["ok"], reply
    assert reply["path"] == str(tmp_path / "run" / "authored" / "specs.yaml")


def test_what_lands_on_disk_is_a_loadable_specification_document(tmp_path):
    server = _server(tmp_path)
    context, _ = server._check_context()
    reply = server._save_specs(context, [_accepted_draft(server)], "ask")

    document = yaml.safe_load(open(reply["path"]))
    assert list(document) == ["specs"]
    assert document["specs"][0]["name"] == "held"
    # ``window`` is the key the design mock invented; it is not in the
    # schema and would load as configured while doing nothing.
    assert "window" not in document["specs"][0]


def test_every_entry_is_rechecked_immediately_before_writing(tmp_path):
    """A page can accumulate for an hour. Nothing guarantees a verdict it
    recorded then still holds now."""
    server = _server(tmp_path)
    context, _ = server._check_context()
    good = _accepted_draft(server)
    bad = {"name": "rot", "formula": "historically[0,1s](nosuch_*[x] >= 1)",
           "constants": {}, "symmetric": False}

    reply = server._save_specs(context, [good, bad], "ask")
    assert reply["ok"] is False
    assert "rot" in reply["error"]
    # Nothing partial reached disk.
    assert not (server.authored_dir / "specs.yaml").exists()


def test_an_existing_file_is_reported_not_clobbered(tmp_path):
    server = _server(tmp_path)
    context, _ = server._check_context()
    first = server._save_specs(context, [_accepted_draft(server)], "ask")
    assert first["ok"]
    original = open(first["path"]).read()

    again = server._save_specs(
        context, [_accepted_draft(server, "second")], "ask"
    )
    assert again["ok"] is False
    assert again["collision"] is True
    assert open(first["path"]).read() == original


def test_rename_saves_alongside_and_overwrite_replaces(tmp_path):
    server = _server(tmp_path)
    context, _ = server._check_context()
    first = server._save_specs(context, [_accepted_draft(server)], "ask")

    renamed = server._save_specs(
        context, [_accepted_draft(server, "second")], "rename"
    )
    assert renamed["ok"]
    assert renamed["path"] != first["path"]
    assert yaml.safe_load(open(first["path"]))["specs"][0]["name"] == "held"

    replaced = server._save_specs(
        context, [_accepted_draft(server, "third")], "overwrite"
    )
    assert replaced["path"] == first["path"]
    assert yaml.safe_load(open(first["path"]))["specs"][0]["name"] == "third"


def test_two_entries_may_not_share_a_name(tmp_path):
    server = _server(tmp_path)
    context, _ = server._check_context()
    reply = server._save_specs(
        context,
        [_accepted_draft(server), _accepted_draft(server)],
        "ask",
    )
    assert reply["ok"] is False
    assert "duplicate" in reply["error"]


def test_constants_that_are_not_a_mapping_are_refused(tmp_path):
    server = _server(tmp_path)
    context, _ = server._check_context()
    draft = _accepted_draft(server)
    draft["constants"] = "threshold: 1"          # a string, not a mapping
    reply = server._save_specs(context, [draft], "ask")
    assert reply["ok"] is False


def test_an_entry_that_is_not_an_object_is_refused(tmp_path):
    server = _server(tmp_path)
    context, _ = server._check_context()
    assert server._save_specs(context, ["../../etc/passwd"], "ask")[
        "ok"] is False


def test_a_new_snapshot_retires_the_compiled_scene(tmp_path):
    """A draft checked against the old one would report on an episode
    that is no longer the one being served."""
    server = _server(tmp_path)
    first, _ = server._check_context()
    assert first is not None

    adapter = _MockAdapter()
    server.set_discovery_snapshot(discovery_payload(
        adapter, reset_adapter(adapter), simulator="mock",
        task=None, effective_dt=adapter.dt,
    ))
    assert server._context is None
    second, _ = server._check_context()
    assert second is not first
