"""Headless interaction contract for the write-specs Textual workspace."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Dict, List
import sys

import pytest
from textual.widgets import Button, DataTable, Input, TextArea

from specola.cli_common import _MockAdapter
from specola.components import ComponentCatalog
from specola.discovery_filter import DiscoveryQuery, filter_discovery
from specola.discovery_model import (
    build_discovery_model,
    property_metadata_for,
)
from specola.discovery_selector import DiscoveryScope
from specola.engine import MonitorEngine


MINIMUM_SIZE = (80, 30)
TEST_SIZE = (110, 55)


def _build_app(suggested_output: Path):
    from specola.write_specs import build_write_specs_app

    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    metadata = property_metadata_for(adapter, entities)
    model = build_discovery_model(
        adapter,
        entities,
        simulator="mock",
        task=None,
        effective_dt=adapter.dt,
        property_metadata=metadata,
    )
    view = filter_discovery(
        model,
        DiscoveryQuery(include_raw=False),
    )
    catalog = ComponentCatalog.from_entities(
        entities,
        metadata=metadata,
    )
    return build_write_specs_app(
        view,
        dt=adapter.dt,
        catalog=catalog,
        metadata=metadata,
        entities=entities,
        labels=adapter.entity_labels,
        capabilities=getattr(adapter, "capabilities", None),
        suggested_output=str(suggested_output),
    )


def _set_field(app, selector: str, value: str) -> None:
    widget = app.query_one(selector)
    if hasattr(widget, "value"):
        widget.value = value
    else:
        widget.text = value


def _field_value(app, selector: str) -> str:
    widget = app.query_one(selector)
    if hasattr(widget, "value"):
        return widget.value
    return widget.text


def _rendered(app, selector: str) -> str:
    widget = app.query_one(selector)
    if isinstance(widget, DataTable):
        return "\n".join(
            " ".join(str(cell) for cell in widget.get_row_at(index))
            for index in range(widget.row_count)
        )
    return str(widget.render())


def _button_containing(app, text: str) -> Button:
    expected = text.lower()
    for button in app.screen.query(Button):
        if expected in str(button.label).lower():
            return button
    raise AssertionError("No button contains {!r}".format(text))


def _mass_row(table: DataTable) -> int:
    for index in range(table.row_count):
        # The signal cell may carry a dim " · component" suffix.
        if any(
            str(cell) == "mass" or str(cell).startswith("mass · ")
            for cell in table.get_row_at(index)
        ):
            return index
    raise AssertionError("No mass signal row")


async def _submit_save_path(app, pilot, path: Path) -> None:
    await pilot.click("#save")
    await pilot.pause()
    path_inputs = list(app.screen.query(Input))
    assert len(path_inputs) == 1
    path_inputs[0].value = str(path)
    await pilot.press("enter")
    await pilot.pause()


async def _select_mass(app, pilot, scope: DiscoveryScope) -> DataTable:
    app._select_scope(scope)
    await pilot.pause()
    signals = app.query_one("#signals", DataTable)
    signals.move_cursor(row=_mass_row(signals))
    signals.focus()
    for _ in range(10):
        if (
            app.selected_signal is not None
            and app.selected_signal.pattern == "mass"
            and app.selected_form is not None
        ):
            break
        await pilot.pause(0.05)
    assert app.selected_signal is not None
    assert app.selected_signal.pattern == "mass"
    assert app.selected_form is not None
    return signals


async def _enter_valid_draft_and_keep(app, pilot) -> None:
    _set_field(app, "#spec_name", "Mass Limit")
    _set_field(
        app,
        "#formula",
        "G[0,1s](object_*[mass] >= limit)",
    )
    _set_field(app, "#constants", "{limit: 0.1}")
    await pilot.pause()
    await pilot.click("#validate")
    await pilot.pause()
    assert not app.query_one("#keep", Button).disabled
    await pilot.click("#keep")
    await pilot.pause()


def test_workspace_launches_and_quits_headless(tmp_path):
    app = _build_app(tmp_path / "specs.yaml")

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            for selector in (
                "#browse",
                "#signals",
                "#palette",
                "#spec_name",
                "#formula",
                "#constants",
                "#validation",
                "#kept_specs",
                "#validate",
                "#keep",
                "#clear",
                "#save",
                "#size_warning",
            ):
                assert app.query_one(selector)
            assert len(
                app.query_one("#signals", DataTable).columns
            ) == 3
            # One column, holding the whole reference: the palette
            # replaced a four-column entity/key mapping table.
            assert len(
                app.query_one("#palette", DataTable).columns
            ) == 1
            assert app.query_one(
                "#selected_signal"
            ).max_scroll_y == 0
            assert app.query_one("#spec_name").border_title == "Name"
            assert app.query_one("#formula").border_title == "Formula"
            assert app.query_one("#constants").border_title == (
                "Constants (YAML)"
            )
            binding_keys = {
                binding[0] for binding in app.BINDINGS
            }
            assert "ctrl+i" not in binding_keys
            assert "ctrl+v" not in binding_keys
            await pilot.press("q")

    asyncio.run(exercise())


def test_writer_inherits_tree_cursor_and_viewport_restoration(tmp_path):
    app = _build_app(tmp_path / "specs.yaml")
    view = app.current_view
    template = next(
        entity for entity in view.entities
        if entity.entity_class == "object"
    )
    large_view = replace(
        view,
        entities=tuple(
            replace(
                template,
                entity_id=index,
                label="mock_object_{}".format(index),
            )
            for index in range(100)
        ),
    )
    app.base_view = large_view
    app.current_view = large_view

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            tree = app.query_one("#browse")
            scope = DiscoveryScope("object", entity_id=50)
            node = app._nodes_by_scope[scope]
            node.parent.expand()
            tree.select_node(node)
            app._select_scope(scope)
            tree.scroll_to(
                y=40, animate=False, force=True, immediate=True
            )
            _set_field(app, "#spec_name", "Draft survives")
            await pilot.pause()
            offset = tree.cursor_line - int(tree.scroll_y)

            app._refresh_browser()
            await pilot.pause()

            assert tree.cursor_node.data == scope
            assert int(tree.scroll_y) == 40
            assert tree.cursor_line - int(tree.scroll_y) == offset
            assert app.query_one("#spec_name").value == "Draft survives"

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("requested", "is_tty", "expected"),
    (
        (None, True, "textual"),
        (None, False, "plain"),
        ("textual", True, "textual"),
        ("plain", True, "plain"),
        ("plain", False, "plain"),
    ),
)
def test_select_ui_routes_auto_textual_and_plain(
    monkeypatch,
    requested,
    is_tty,
    expected,
):
    from specola import write_specs

    monkeypatch.setattr(write_specs, "_is_tty", lambda: is_tty)

    assert write_specs._select_ui(requested) == expected


def test_explicit_textual_non_tty_rejects_before_adapter_reset(
    monkeypatch,
):
    from specola import write_specs

    reset_attempts = []

    def forbidden_load(*_args, **_kwargs):
        reset_attempts.append(True)
        raise AssertionError("adapter reset happened before UI rejection")

    monkeypatch.setattr(write_specs, "_is_tty", lambda: False)
    monkeypatch.setattr(write_specs, "load_sim", forbidden_load)
    monkeypatch.setattr(
        sys,
        "argv",
        ["specola write-specs", "--sim", "mock", "--ui", "textual"],
    )

    with pytest.raises(SystemExit) as raised:
        write_specs.main()

    assert str(raised.value) == (
        "--ui textual requires interactive stdin and stdout; "
        "use --ui plain for redirected input or output"
    )
    assert reset_attempts == []


@pytest.mark.parametrize(
    ("scope", "expected_selector"),
    (
        (DiscoveryScope("object"), "object_*[mass]"),
        (
            DiscoveryScope("object", entity_id=1),
            "object_1[mass]",
        ),
    ),
)
def test_class_and_entity_scopes_insert_authority_selector(
    tmp_path,
    scope,
    expected_selector,
):
    app = _build_app(tmp_path / "specs.yaml")

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            signals = await _select_mass(app, pilot, scope)
            assert signals.has_focus
            # The palette offers every shape, and the one it lands on
            # is the authority selector the previous panel produced --
            # widening forms sit below it, never in front of it.
            palette = app.query_one("#palette", DataTable)
            assert palette.row_count == len(app.palette_forms) > 1
            assert app.selected_form.reference == expected_selector

            await pilot.press("i")
            await pilot.pause()

            assert _field_value(app, "#formula") == expected_selector

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("scope", "expected_selector"),
    (
        (DiscoveryScope("object"), "object_*[mass]"),
        (
            DiscoveryScope("object", entity_id=1),
            "object_1[mass]",
        ),
    ),
)
def test_copy_selector_preserves_authority_text(
    monkeypatch,
    tmp_path,
    scope,
    expected_selector,
):
    app = _build_app(tmp_path / "specs.yaml")
    copied = []
    monkeypatch.setattr(
        app,
        "copy_to_clipboard",
        lambda text: copied.append(text),
    )

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _select_mass(app, pilot, scope)

            await pilot.click("#copy_selector")
            await pilot.pause()

            assert copied == [expected_selector]
            assert _field_value(app, "#formula") == ""

    asyncio.run(exercise())


def test_selector_inserts_at_formula_cursor(tmp_path):
    app = _build_app(tmp_path / "specs.yaml")

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _select_mass(
                app,
                pilot,
                DiscoveryScope("object"),
            )
            formula = app.query_one("#formula", TextArea)
            formula.text = "G[0,1s]( >= limit)"
            formula.cursor_location = (0, len("G[0,1s]("))
            app.query_one("#signals", DataTable).focus()

            await pilot.press("i")
            await pilot.pause()

            assert formula.text == (
                "G[0,1s](object_*[mass] >= limit)"
            )

    asyncio.run(exercise())


def test_formula_accepts_literal_insert_binding_key(tmp_path):
    app = _build_app(tmp_path / "specs.yaml")

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            formula = app.query_one("#formula", TextArea)
            formula.focus()
            await pilot.press("i")
            await pilot.pause()

            assert formula.text == "i"

    asyncio.run(exercise())


def test_scope_change_cannot_insert_the_previous_preview(tmp_path):
    app = _build_app(tmp_path / "specs.yaml")

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _select_mass(
                app,
                pilot,
                DiscoveryScope("object"),
            )
            assert app.selected_form is not None
            assert app.selected_form.reference == "object_*[mass]"

            app._select_scope(
                DiscoveryScope("object", entity_id=1)
            )
            app._insert_selector()

            assert _field_value(app, "#formula") == ""

    asyncio.run(exercise())


def test_draft_survives_browsing_scope_changes(tmp_path):
    app = _build_app(tmp_path / "specs.yaml")
    expected = {
        "#spec_name": "Object Mass Limit",
        "#formula": "G[0,1s](object_*[mass] >= threshold)",
        "#constants": "{threshold: 0.3}",
    }

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            for selector, value in expected.items():
                _set_field(app, selector, value)
            await pilot.pause()

            for scope in (
                DiscoveryScope("robot"),
                DiscoveryScope("object", entity_id=1),
                DiscoveryScope("object"),
            ):
                app._select_scope(scope)
                await pilot.pause()

            assert {
                selector: _field_value(app, selector)
                for selector in expected
            } == expected

    asyncio.run(exercise())


def test_writer_search_submit_applies_final_filter_once(
    tmp_path,
    monkeypatch,
):
    import specola.explore as explore_module

    calls = []
    original = explore_module.filter_discovery

    def counting_filter(model, query):
        calls.append(query)
        return original(model, query)

    monkeypatch.setattr(
        explore_module,
        "filter_discovery",
        counting_filter,
    )
    app = _build_app(tmp_path / "specs.yaml")

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await pilot.press("/")
            await pilot.press("m", "a", "s", "s", "enter")
            await pilot.pause()

            final_search = ("mass",)
            assert calls[-1].search == final_search
            assert sum(
                query.search == final_search for query in calls
            ) == 1

    asyncio.run(exercise())


def test_successful_validation_edit_stale_and_keep_gating(tmp_path):
    app = _build_app(tmp_path / "specs.yaml")

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            keep = app.query_one("#keep", Button)
            assert keep.disabled

            _set_field(app, "#spec_name", "Joint Limit")
            _set_field(
                app,
                "#formula",
                "G[0,1s](robot_*[qpos_joint1] <= limit)",
            )
            _set_field(app, "#constants", "{limit: 1.0}")
            await pilot.pause()
            assert keep.disabled

            await pilot.click("#validate")
            await pilot.pause()
            report = _rendered(app, "#validation")
            assert "constants: {limit: 1.0}" in report
            assert "qpos_joint1" in report
            assert "reset bindings: >=1" in report
            assert not keep.disabled
            workspace = app.query_one("#writer_workspace")
            validation = app.query_one("#validation_scroll")
            assert workspace.scroll_y > 0
            assert validation.region.y < app.screen.size.height

            _set_field(app, "#constants", "{limit: 2.0}")
            await pilot.pause()
            assert "stale" in _rendered(app, "#validation").lower()
            assert keep.disabled

            await pilot.click("#validate")
            await pilot.pause()
            assert not keep.disabled
            await pilot.click("#keep")
            await pilot.pause()

            assert "joint_limit" in _rendered(app, "#kept_specs")
            assert _field_value(app, "#spec_name") == ""
            assert _field_value(app, "#formula") == ""
            assert _field_value(app, "#constants") in ("", "{}")
            assert keep.disabled
            assert "Not validated." in _rendered(
                app, "#validation"
            )

    asyncio.run(exercise())


def test_draft_survives_search_raw_and_browse_mode_rebuilds(tmp_path):
    app = _build_app(tmp_path / "specs.yaml")
    expected = {
        "#spec_name": "Persistent Draft",
        "#formula": "object_*[mass] > 0",
        "#constants": "{threshold: 0.3}",
    }

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            for selector, value in expected.items():
                _set_field(app, selector, value)

            await pilot.press("/")
            await pilot.press("o", "b", "j", "e", "c", "t", "enter")
            await pilot.pause()
            app.query_one("#browse").focus()
            await pilot.press("r")
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()

            assert {
                selector: _field_value(app, selector)
                for selector in expected
            } == expected

    asyncio.run(exercise())


def test_clear_preserves_kept_entries_and_duplicate_name_gate(tmp_path):
    app = _build_app(tmp_path / "specs.yaml")

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _enter_valid_draft_and_keep(app, pilot)

            _set_field(app, "#spec_name", "MASS LIMIT")
            _set_field(app, "#formula", "object_*[mass] > 0")
            _set_field(app, "#constants", "{}")
            await pilot.click("#validate")
            await pilot.pause()
            assert "already used" in _rendered(
                app, "#validation"
            )

            await pilot.click("#clear")
            await pilot.pause()
            assert "mass_limit" in _rendered(app, "#kept_specs")
            assert len(app.specs) == 1

            await pilot.press("q")
            await pilot.pause()
            await pilot.click(_button_containing(app, "discard"))

    asyncio.run(exercise())


def test_real_validation_error_is_ansi_aware_and_readable(tmp_path):
    app = _build_app(tmp_path / "specs.yaml")

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            _set_field(app, "#spec_name", "Missing Limit")
            _set_field(
                app,
                "#formula",
                "robot_*[qpos_joint1] <= missing_limit",
            )
            _set_field(app, "#constants", "{}")

            await pilot.click("#validate")
            await pilot.pause()

            report = _rendered(app, "#validation")
            assert "Unknown symbol(s) ['missing_limit']" in report
            assert "\x1b" not in report
            assert "\\x1b" not in report
            assert "[31m" not in report
            assert app.query_one("#keep", Button).disabled

    asyncio.run(exercise())


def test_edit_after_rejected_validation_uses_neutral_stale_wording(
    tmp_path,
):
    app = _build_app(tmp_path / "specs.yaml")

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await pilot.click("#validate")
            await pilot.pause()

            _set_field(app, "#spec_name", "Now Editing")
            await pilot.pause()

            report = _rendered(app, "#validation")
            assert "Draft edited since the last validation." in report
            assert "last successful validation" not in report

    asyncio.run(exercise())


def test_save_collision_rename_uses_first_free_sibling(tmp_path):
    existing = tmp_path / "specs.yaml"
    second = tmp_path / "specs_2.yaml"
    existing.write_text("original\n")
    second.write_text("also original\n")
    app = _build_app(existing)

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _enter_valid_draft_and_keep(app, pilot)

            await pilot.click("#save")
            await pilot.pause()
            path_inputs = list(app.screen.query(Input))
            assert len(path_inputs) == 1
            assert path_inputs[0].value == str(existing)
            await pilot.press("enter")
            await pilot.pause()

            await pilot.click(_button_containing(app, "rename"))

    asyncio.run(exercise())

    renamed = tmp_path / "specs_3.yaml"
    assert existing.read_text() == "original\n"
    assert second.read_text() == "also original\n"
    assert renamed.exists()
    entries = MonitorEngine.iter_spec_entries(renamed)
    assert len(entries) == 1
    assert entries[0][2] is None


def test_save_to_new_path_sets_only_saved_path(tmp_path):
    requested = tmp_path / "new_specs"
    app = _build_app(requested)

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _enter_valid_draft_and_keep(app, pilot)
            await _submit_save_path(app, pilot, requested)

            assert app.saved_path == str(requested) + ".yaml"
            assert not app.kept_dirty

    asyncio.run(exercise())

    saved = tmp_path / "new_specs.yaml"
    assert saved.exists()
    entries = MonitorEngine.iter_spec_entries(saved)
    assert len(entries) == 1
    assert entries[0][2] is None


def test_new_file_write_is_exclusive(tmp_path):
    from specola.write_specs import _write_specs_document

    path = tmp_path / "specs.yaml"
    path.write_text("created by another process\n")
    specs: List[Dict[str, object]] = [{
        "name": "mass_limit",
        "formula": "object_*[mass] >= 0",
    }]

    with pytest.raises(FileExistsError):
        _write_specs_document(
            specs,
            str(path),
            overwrite=False,
        )

    assert path.read_text() == "created by another process\n"


def test_save_error_keeps_the_dirty_workspace_alive(tmp_path):
    output = tmp_path / "missing-parent" / "specs.yaml"
    app = _build_app(output)

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _enter_valid_draft_and_keep(app, pilot)

            app._complete_save(str(output))
            await pilot.pause()

            assert app.is_running
            assert app.kept_dirty
            assert app.saved_path is None
            assert "Could not save" in _rendered(
                app, "#save_status"
            )

            await pilot.press("q")
            await pilot.pause()
            await pilot.click(_button_containing(app, "discard"))

    asyncio.run(exercise())
    assert not output.exists()


def test_failed_overwrite_preserves_previous_saved_file(
    tmp_path,
    monkeypatch,
):
    from specola import write_specs

    output = tmp_path / "specs.yaml"
    output.write_text("previous valid document\n")
    app = _build_app(output)
    app.saved_path = str(output)

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _enter_valid_draft_and_keep(app, pilot)

            def fail_dump(*_args, **_kwargs):
                raise write_specs.yaml.YAMLError(
                    "serialization failed"
                )

            monkeypatch.setattr(
                write_specs.yaml,
                "safe_dump",
                fail_dump,
            )
            app._complete_save(str(output), overwrite=True)
            await pilot.pause()

            assert output.read_text() == "previous valid document\n"
            assert app.saved_path == str(output)
            assert app.kept_dirty
            assert "Could not save" in _rendered(
                app, "#save_status"
            )

            await pilot.press("q")
            await pilot.pause()
            await pilot.click(_button_containing(app, "discard"))

    asyncio.run(exercise())
    assert output.read_text() == "previous valid document\n"


def test_save_collision_overwrite_is_explicit(tmp_path):
    existing = tmp_path / "specs.yaml"
    existing.write_text("original\n")
    app = _build_app(existing)

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _enter_valid_draft_and_keep(app, pilot)
            await _submit_save_path(app, pilot, existing)

            assert existing.read_text() == "original\n"
            await pilot.click(_button_containing(app, "overwrite"))
            await pilot.pause()
            assert app.saved_path == str(existing)
            assert not app.kept_dirty

    asyncio.run(exercise())

    entries = MonitorEngine.iter_spec_entries(existing)
    assert len(entries) == 1
    assert entries[0][2] is None


def test_file_appearing_after_availability_check_is_not_overwritten(
    tmp_path,
    monkeypatch,
):
    from specola import write_specs

    requested = tmp_path / "specs.yaml"
    app = _build_app(requested)
    original = write_specs._write_specs_document
    raced = {"done": False}

    def create_then_write(specs, path, *, overwrite=True):
        if not overwrite and not raced["done"]:
            raced["done"] = True
            Path(path).write_text("created by another process\n")
        return original(specs, path, overwrite=overwrite)

    monkeypatch.setattr(
        write_specs,
        "_write_specs_document",
        create_then_write,
    )

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _enter_valid_draft_and_keep(app, pilot)
            await _submit_save_path(app, pilot, requested)

            assert requested.read_text() == (
                "created by another process\n"
            )
            assert app.saved_path is None
            assert app.kept_dirty
            assert _button_containing(app, "rename")

            await pilot.click(_button_containing(app, "rename"))
            await pilot.pause()

    asyncio.run(exercise())
    assert requested.read_text() == "created by another process\n"
    assert (tmp_path / "specs_2.yaml").exists()


def test_rename_reopens_collision_if_free_sibling_appears(
    tmp_path,
    monkeypatch,
):
    from specola import write_specs

    existing = tmp_path / "specs.yaml"
    existing.write_text("original\n")
    app = _build_app(existing)
    original = write_specs._write_specs_document
    raced = {"done": False, "path": None}

    def create_then_write(specs, path, *, overwrite=True):
        if not overwrite and not raced["done"]:
            raced["done"] = True
            raced["path"] = path
            Path(path).write_text("created by another process\n")
        return original(specs, path, overwrite=overwrite)

    monkeypatch.setattr(
        write_specs,
        "_write_specs_document",
        create_then_write,
    )

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _enter_valid_draft_and_keep(app, pilot)
            await _submit_save_path(app, pilot, existing)
            await pilot.click(_button_containing(app, "rename"))
            await pilot.pause()

            raced_path = Path(raced["path"])
            assert raced_path.read_text() == (
                "created by another process\n"
            )
            assert app.saved_path is None
            assert app.kept_dirty
            assert _button_containing(app, "rename")

            await pilot.click(_button_containing(app, "rename"))
            await pilot.pause()

    asyncio.run(exercise())
    raced_path = Path(raced["path"])
    assert raced_path.read_text() == "created by another process\n"
    assert app.saved_path is not None
    saved_path = Path(app.saved_path)
    assert saved_path.exists()
    assert saved_path != raced_path


@pytest.mark.parametrize("choice", ("cancel", "skip"))
def test_save_collision_cancel_and_skip_preserve_dirty_work(
    tmp_path,
    choice,
):
    existing = tmp_path / "specs.yaml"
    existing.write_text("original\n")
    app = _build_app(existing)

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _enter_valid_draft_and_keep(app, pilot)
            await _submit_save_path(app, pilot, existing)

            await pilot.click(_button_containing(app, choice))
            await pilot.pause()
            assert app.saved_path is None
            assert app.kept_dirty
            assert "mass_limit" in _rendered(app, "#kept_specs")
            assert app.is_running

    asyncio.run(exercise())
    assert existing.read_text() == "original\n"


def test_unsaved_kept_entry_requires_return_or_discard(tmp_path):
    output = tmp_path / "specs.yaml"
    app = _build_app(output)

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _enter_valid_draft_and_keep(app, pilot)

            await pilot.press("q")
            await pilot.pause()
            await pilot.click(_button_containing(app, "return"))
            await pilot.pause()
            assert app.is_running
            assert "mass_limit" in _rendered(app, "#kept_specs")

            await pilot.press("q")
            await pilot.pause()
            await pilot.click(_button_containing(app, "discard"))

    asyncio.run(exercise())
    assert not output.exists()


def test_exit_save_successfully_writes_and_closes(tmp_path):
    output = tmp_path / "specs.yaml"
    app = _build_app(output)

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            await _enter_valid_draft_and_keep(app, pilot)

            await pilot.press("q")
            await pilot.pause()
            await pilot.click(_button_containing(app, "save file"))
            await pilot.pause()
            path_inputs = list(app.screen.query(Input))
            assert len(path_inputs) == 1
            path_inputs[0].value = str(output)
            await pilot.press("enter")

    asyncio.run(exercise())
    assert not app.is_running
    entries = MonitorEngine.iter_spec_entries(output)
    assert len(entries) == 1
    assert entries[0][0] == "mass_limit"


def test_unkept_draft_requires_return_or_discard(tmp_path):
    output = tmp_path / "specs.yaml"
    app = _build_app(output)
    expected = {
        "#spec_name": "Draft Only",
        "#formula": "object_*[mass] > 0",
        "#constants": "{}",
    }

    async def exercise():
        async with app.run_test(size=TEST_SIZE) as pilot:
            await pilot.pause()
            for selector, value in expected.items():
                _set_field(app, selector, value)
            await pilot.pause()

            await pilot.press("q")
            await pilot.pause()
            assert "current draft not kept" in str(
                app.screen.query_one("Static").render()
            )
            assert not list(app.screen.query("#unsaved_save"))
            await pilot.click(_button_containing(app, "return"))
            await pilot.pause()
            assert {
                selector: _field_value(app, selector)
                for selector in expected
            } == expected

            await pilot.press("q")
            await pilot.pause()
            await pilot.click(_button_containing(app, "discard"))

    asyncio.run(exercise())
    assert not output.exists()


def test_exact_minimum_is_usable_and_below_it_requests_resize(tmp_path):
    app = _build_app(tmp_path / "specs.yaml")

    async def exercise():
        async with app.run_test(size=MINIMUM_SIZE) as pilot:
            await pilot.pause()
            warning = app.query_one("#size_warning")
            assert not warning.display
            for selector in (
                "#browser_row",
                "#browse",
                "#signals",
                "#formula",
                "#validate",
                "#keep",
                "#save",
            ):
                region = app.query_one(selector).region
                assert region.width > 0
                assert region.height > 0
            assert app.query_one("#browser_row").region.height >= 8

            await pilot.resize_terminal(79, 30)
            await pilot.pause()
            warning = app.query_one("#size_warning")
            assert warning.display
            message = str(warning.render())
            assert "Write Specs requires at least 80 × 30." in message
            assert "--ui plain" in message

            await pilot.resize_terminal(80, 29)
            await pilot.pause()
            warning = app.query_one("#size_warning")
            assert warning.display
            assert "Write Specs requires at least 80 × 30." in str(
                warning.render()
            )

    asyncio.run(exercise())


def test_keyboard_traversal_validates_and_keeps_at_minimum_size(
    tmp_path,
):
    app = _build_app(tmp_path / "specs.yaml")

    async def exercise():
        async with app.run_test(size=MINIMUM_SIZE) as pilot:
            await pilot.pause()
            _set_field(app, "#spec_name", "Keyboard Spec")
            _set_field(app, "#formula", "object_*[mass] > 0")
            _set_field(app, "#constants", "{}")

            app.query_one("#spec_name").focus()
            await pilot.press("tab")
            assert app.query_one("#formula").has_focus
            await pilot.press("tab")
            assert app.query_one("#constants").has_focus
            await pilot.press("tab")
            assert app.query_one("#validate").has_focus
            await pilot.press("enter")
            await pilot.pause()
            assert not app.query_one("#keep", Button).disabled

            await pilot.press("tab", "enter")
            await pilot.pause()
            assert len(app.specs) == 1
            assert app.specs[0]["name"] == "keyboard_spec"

            await pilot.press("q")
            await pilot.pause()
            await pilot.click(_button_containing(app, "discard"))

    asyncio.run(exercise())


def test_textual_main_prints_followups_only_from_existing_helper(
    monkeypatch,
    capsys,
    tmp_path,
):
    from specola import write_specs

    saved = tmp_path / "saved.yaml"
    calls = []

    class SavedPathOnlyApp:
        saved_path = str(saved)

        def run(self):
            return None

    def followups(args, path):
        calls.append((args, path))
        return "CHECK FROM HELPER", "RUN FROM HELPER"

    monkeypatch.setattr(write_specs, "_is_tty", lambda: True)
    monkeypatch.setattr(
        write_specs,
        "build_write_specs_app",
        lambda *_args, **_kwargs: SavedPathOnlyApp(),
    )
    monkeypatch.setattr(write_specs, "_followup_commands", followups)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "specola write-specs",
            "--sim",
            "mock",
            "--ui",
            "textual",
        ],
    )

    write_specs.main()

    output = capsys.readouterr().out
    assert len(calls) == 1
    assert calls[0][1] == str(saved)
    assert output.count("CHECK FROM HELPER") == 1
    assert output.count("RUN FROM HELPER") == 1
    assert output.index("CHECK FROM HELPER") < output.index(
        "RUN FROM HELPER"
    )
