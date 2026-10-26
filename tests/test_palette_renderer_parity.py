"""One selection means one palette, whichever surface renders it.

specola discovers through five surfaces: Textual, Rich, plain, JSON, and
the browser. They are separate renderers over one model, and the failure
mode that matters is drift -- the same filter offering different
selectors depending on how you looked at it, so a specification written
from one surface monitors something else than the reader was shown.

These tests compare each non-interactive renderer against the shared
projection rather than against a fixed string, so they keep holding when
the palette's contents legitimately change.
"""

from __future__ import annotations

import io

import pytest

from specola.cli_common import _MockAdapter
from specola.discovery_filter import DiscoveryQuery, filter_discovery
from specola.discovery_model import build_discovery_model
from specola.discovery_render import render_discovery
from specola.discovery_serialize import discovery_document
from specola.selector_palette import palette_for_view


@pytest.fixture(scope="module")
def model():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    return build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )


def _view(model, **query):
    query.setdefault("include_raw", False)
    return filter_discovery(model, DiscoveryQuery(**query))


def _plain_references(text):
    body = text.split("Ways to select this")[1]
    out = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            # The split leaves the header's own line-end first; only a
            # blank *after* rows have started ends the block.
            if out:
                break
            continue
        if not stripped[0].isalpha():
            break
        out.append(stripped.split("  ↔")[0].split("    —")[0])
    return out


def _rich_references(view):
    from rich.console import Console

    from specola.discovery_live import render_rich_discovery

    buffer = io.StringIO()
    render_rich_discovery(view, console=Console(file=buffer, width=200))
    body = buffer.getvalue().split("Ways to select this")[1]
    out = []
    for line in body.splitlines():
        stripped = line.strip().strip("│").strip()
        if not stripped:
            if out:
                break
            continue
        if stripped.startswith("Filter with") or stripped.startswith("╰"):
            break
        out.append(stripped.split("  ↔")[0].split("    —")[0])
    return out


SELECTIONS = [
    {"classes": ("object",), "signals": ("mass",)},
    {"classes": ("object",)},
    {"classes": ("gripper",)},
    {"classes": ("robot",), "signals": ("qpos_joint1",)},
]


@pytest.mark.parametrize("query", SELECTIONS)
def test_every_surface_offers_the_same_ordered_forms(model, query):
    view = _view(model, **query)
    expected = [form.reference for form in palette_for_view(view)[0]]
    assert expected, query

    document = discovery_document(view)["selector_palette"]
    assert [form["reference"] for form in document["forms"]] == expected

    plain = _plain_references(render_discovery(view, detailed=True))
    assert plain == expected

    assert _rich_references(view) == expected


def test_rich_does_not_eat_the_component_bracket(model):
    """Rich parses ``[...]`` as style markup.

    Rendered as a plain string, ``object_*[mass]`` reaches the terminal
    as ``object_*`` — the component silently removed, which is the part
    of a reference that decides what gets monitored. The renderer passes
    Text for exactly this reason, and this test is why it must keep
    doing so.
    """
    view = _view(model, classes=("object",), signals=("mass",))
    references = _rich_references(view)
    assert references
    for reference in references:
        assert "[" in reference and reference.endswith("]"), reference
    assert "object_*[mass]" in references


def test_rich_no_longer_redirects_to_textual(model):
    """Rich used to tell readers to switch UI for selector previews."""
    from rich.console import Console

    from specola.discovery_live import render_rich_discovery

    buffer = io.StringIO()
    render_rich_discovery(
        _view(model, classes=("object",)),
        console=Console(file=buffer, width=200),
    )
    assert "--ui textual" not in buffer.getvalue()


@pytest.mark.parametrize("query", SELECTIONS)
def test_no_surface_shows_a_count_or_a_scope_header(model, query):
    """Rows are forms. Coverage lives in the signals pane, not here."""
    view = _view(model, **query)
    document = discovery_document(view)["selector_palette"]
    for form in document["forms"]:
        assert "bindings" not in form
        assert "monitors" not in form

    for reference in _plain_references(render_discovery(view, detailed=True)):
        assert "binding" not in reference
        assert "monitor" not in reference


def test_a_selection_spanning_classes_says_why_it_has_no_forms(model):
    """A reference names one class, so several classes name none."""
    view = _view(model)
    forms, reason = palette_for_view(view)
    assert forms == ()
    assert "multiple classes" in reason

    document = discovery_document(view)["selector_palette"]
    assert document["forms"] == []
    assert document["reason"] == reason
    assert reason in render_discovery(view, detailed=True)


def test_json_stays_one_parseable_document(model):
    """The palette must not break --format json's single-document rule."""
    import json

    from specola.discovery_serialize import discovery_json

    text = discovery_json(_view(model, classes=("object",)))
    parsed = json.loads(text)
    assert parsed["selector_palette"]["forms"]
    assert text.count("\n") > 1


def test_plain_reports_an_unavailable_form_with_its_reason(model):
    """A shape the grammar cannot express here keeps its place."""
    view = _view(model, classes=("object",), signals=("hinge_angle",))
    forms, _ = palette_for_view(view)
    unavailable = [form for form in forms if not form.available]
    if not unavailable:
        pytest.skip("this selection expresses every shape")
    text = render_discovery(view, detailed=True)
    for form in unavailable:
        assert form.unavailable_reason in text


def test_every_surface_checks_before_it_offers_a_starter(model):
    """Bracket availability is not the compiler's answer.

    ``palette_for_view`` feeds the plain renderer, the Rich overview and
    the JSON document, and the first two emit a starter for any row they
    call available. It used to hand back bracket-level answers, so all
    ten of RoboCasa's illegal ``contact`` forms were listed as offerable
    on those surfaces while the browser refused them.

    The catalog is derived from the view's own model rather than asked
    for, so a caller cannot forget to pass it.
    """
    from specola.selector_palette import palette_for_view

    view = _view(model, classes=("object",))
    forms, reason = palette_for_view(view)
    assert forms, reason
    assert all(form.verified for form in forms)

    unchecked, _ = palette_for_view(view, verify=False)
    assert not any(form.verified for form in unchecked)


def test_the_json_document_says_whether_a_row_was_checked(model):
    """Additive and nullable, so the schema version stays 1."""
    import json

    from specola.discovery_serialize import discovery_json

    document = json.loads(discovery_json(_view(model, classes=("object",))))
    rows = document["selector_palette"]["forms"]
    assert rows
    for row in rows:
        assert "verified" in row
        assert "expansions" in row
