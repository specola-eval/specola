from __future__ import annotations

import asyncio
from dataclasses import replace

from specola.cli_common import _MockAdapter
from specola.discovery_filter import DiscoveryQuery, filter_discovery
from specola.discovery_model import build_discovery_model
from specola.discovery_selector import DiscoveryScope, ROLE_ABSENT
from specola.explore import build_explorer, discovery_explorer_class


def _build_mock_view():
    adapter = _MockAdapter()
    entities = adapter.reset().perceived_entities
    model = build_discovery_model(
        adapter, entities, simulator="mock", task=None,
        effective_dt=adapter.dt,
    )
    return filter_discovery(
        model, DiscoveryQuery(include_raw=False),
    )


def _build_large_mock_view(count=100):
    view = _build_mock_view()
    template = next(
        entity for entity in view.entities
        if entity.entity_class == "object"
    )
    entities = tuple(
        replace(
            template,
            entity_id=index,
            label="mock_object_{}".format(index),
        )
        for index in range(count)
    )
    return replace(view, entities=entities)


def _build_role_mock_view():
    view = _build_mock_view()
    entities = []
    for entity in view.entities:
        role = (
            "target" if entity.entity_id == 1 else
            "distractor" if entity.entity_id == 2 else
            entity.role
        )
        properties = tuple(
            replace(
                prop,
                tags=prop.tags + (("role:" + role,) if role else ()),
            )
            for prop in entity.properties
        )
        entities.append(replace(
            entity,
            role=role,
            properties=properties,
        ))
    tagged = tuple(entities)
    # The model is retagged too, not only the view. A view is always a
    # subset of the model it was filtered from -- `filter_discovery` only
    # ever drops -- and the palette now checks forms against a catalog
    # built from that model. A view carrying tags its own model lacks is
    # a state that cannot arise in production, and it made a legal role
    # selector look uncompilable.
    return replace(
        view,
        entities=tagged,
        model=replace(view.model, entities=tagged),
    )


def _build_mock_explorer():
    return build_explorer(_build_mock_view())


def test_explorer_launches_and_quits_headless():
    app = _build_mock_explorer()

    async def exercise():
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("Header")
            assert app.query_one("Footer")
            assert app.query_one("#browse").root.children
            assert app.query_one("#signals").row_count > 0
            # The palette resolves on a debounce, so it is one tick
            # behind the signal table it follows.
            await pilot.pause(0.1)
            assert app.query_one("#palette").row_count > 0
            await pilot.press("c")
            await pilot.pause()
            assert app.browse_mode == "role"
            await pilot.press("/")
            await pilot.press("t", "a", "r", "g", "e", "t")
            await pilot.pause()
            await pilot.press("ctrl+c")

    asyncio.run(exercise())


def test_discovery_selection_preserves_identity_detail_and_selector_authority():
    app = _build_mock_explorer()

    async def exercise():
        async with app.run_test() as pilot:
            await pilot.pause()
            entity_scope = DiscoveryScope("object", entity_id=1)
            assert str(
                app._nodes_by_scope[entity_scope].label
            ) == "object_1  mock_target_object"

            class_scope = DiscoveryScope("object")
            app._select_scope(class_scope)
            await pilot.pause()
            class_signal = next(
                signal for signal in app.signal_by_key.values()
                if signal.pattern == "mass"
            )
            app._show_signal(class_signal)
            await pilot.pause(0.1)
            assert app.selected_form is not None

            detail = str(
                app.query_one("#selected_signal_content").render()
            )
            assert "Component    —" in detail
            assert "Frame        —" in detail
            assert app.selected_form.reference == "object_*[mass]"
            authoring = str(
                app.query_one("#authoring_content").render()
            )
            # The panel names the reference and what the shape means.
            # It deliberately carries no binding or monitor count: the
            # signals pane above already reports coverage, and a count
            # per palette row was the thing this redesign removed.
            assert "Reference        object_*[mass]" in authoring
            assert "Initial bindings" not in authoring
            assert "binding" not in authoring.split("Shape")[0]

            # Every alternative the language offers for this selection
            # is on screen, not just the authority one.
            references = [
                form.reference for form in app.palette_forms
            ]
            assert "object_*[mass]" in references
            assert any(
                reference.startswith("object_**[") for reference in references
            )

            app._select_scope(entity_scope)
            await pilot.pause()
            entity_signal = next(
                signal for signal in app.signal_by_key.values()
                if signal.pattern == "mass"
            )
            app._show_signal(entity_signal)
            await pilot.pause(0.1)
            assert app.selected_form is not None

            assert app.selected_form.reference == "object_1[mass]"

    asyncio.run(exercise())


def test_scope_change_clears_the_previous_preview_synchronously():
    app = _build_mock_explorer()

    async def exercise():
        async with app.run_test() as pilot:
            await pilot.pause()
            app._select_scope(DiscoveryScope("object"))
            await pilot.pause()
            class_signal = next(
                signal for signal in app.signal_by_key.values()
                if signal.pattern == "mass"
            )
            app._show_signal(class_signal)
            app._resolve_selected_signal()
            assert app.selected_form is not None
            assert app.selected_form.reference == "object_*[mass]"

            app._select_scope(
                DiscoveryScope("object", entity_id=1)
            )

            assert app.selected_signal is None
            assert app.selected_form is None

    asyncio.run(exercise())


def test_composed_explorer_ignores_non_search_inputs(monkeypatch):
    import specola.explore as explore_module
    from textual.widgets import Input

    view = _build_mock_view()
    calls = []
    original = explore_module.filter_discovery

    def counting_filter(model, query):
        calls.append(query)
        return original(model, query)

    monkeypatch.setattr(
        explore_module, "filter_discovery", counting_filter
    )
    DiscoveryExplorer = discovery_explorer_class(view)

    class WriterExplorer(DiscoveryExplorer):
        def compose(self):
            yield from self.compose_discovery()
            yield Input(id="writer_formula")

    app = WriterExplorer()

    async def exercise():
        async with app.run_test() as pilot:
            writer = app.query_one("#writer_formula", Input)
            writer.focus()
            writer.value = "object_*[mass]"
            await pilot.press("enter")
            await pilot.pause(0.25)

            assert calls == []
            assert app.query_one("#search", Input).value == ""
            assert app.current_view is view

    asyncio.run(exercise())


def test_partial_bracket_is_rendered_as_an_error_and_escape_leaves_search():
    app = _build_mock_explorer()

    async def exercise():
        async with app.run_test() as pilot:
            await pilot.press("/")
            await pilot.press("[")
            await pilot.pause(0.25)

            authoring = str(
                app.query_one("#authoring_content").render()
            )
            assert "unclosed bracket in filter pattern '['" in authoring

            app.action_toggle_raw()
            assert "unclosed bracket" in str(
                app.query_one("#authoring_content").render()
            )

            await pilot.press("escape")
            await pilot.pause()
            assert app.query_one("#browse").has_focus

    asyncio.run(exercise())


def test_bracketed_reference_renders_verbatim_and_help_restores_it():
    app = _build_mock_explorer()

    async def exercise():
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            before = str(app.query_one("#authoring_content").render())
            assert "[" in before
            assert "]" in before

            await pilot.press("?")
            await pilot.pause()
            assert "navigate" in str(
                app.query_one("#authoring_content").render()
            )
            await pilot.press("escape")
            await pilot.pause()
            assert str(
                app.query_one("#authoring_content").render()
            ) == before

    asyncio.run(exercise())


def test_search_is_debounced_to_one_filter_pass(monkeypatch):
    import specola.explore as explore_module

    app = _build_mock_explorer()
    calls = []
    original = explore_module.filter_discovery

    def counting_filter(model, query):
        calls.append(query)
        return original(model, query)

    monkeypatch.setattr(
        explore_module, "filter_discovery", counting_filter
    )

    async def exercise():
        async with app.run_test() as pilot:
            await pilot.press("/")
            await pilot.press("t", "a", "r", "g", "e", "t")
            await pilot.pause(0.4)
            assert len(calls) == 1

    asyncio.run(exercise())


def test_role_root_is_selectable_and_tree_state_survives_rebuild():
    app = _build_mock_explorer()

    async def exercise():
        async with app.run_test() as pilot:
            await pilot.press("c")
            await pilot.pause()
            tree = app.query_one("#browse")
            role_node = tree.root.children[0]
            assert role_node.data is not None
            assert role_node.data.entity_class is None

            role_node.expand()
            class_scope = role_node.children[0].data
            app._select_scope(class_scope)
            app._refresh_browser()

            restored_role = app._nodes_by_scope[role_node.data]
            assert restored_role.is_expanded
            assert app.selected_scope == class_scope

    asyncio.run(exercise())


def test_tree_rebuild_restores_cursor_identity_and_viewport_anchor():
    view = _build_large_mock_view()
    app = build_explorer(view)

    async def exercise():
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            tree = app.query_one("#browse")
            scope = DiscoveryScope("object", entity_id=50)
            node = app._nodes_by_scope[scope]
            node.parent.expand()
            await pilot.pause()
            tree.move_cursor(node)
            selected_scope = DiscoveryScope("object")
            app._select_scope(selected_scope)
            tree.scroll_to(
                y=40, animate=False, force=True, immediate=True
            )
            await pilot.pause()
            assert tree.cursor_node.data == scope
            offset = tree.cursor_line - int(tree.scroll_y)

            app._refresh_browser()
            await pilot.pause()

            assert tree.cursor_node.data == scope
            assert app.selected_scope == selected_scope
            assert int(tree.scroll_y) == 40
            assert tree.cursor_line - int(tree.scroll_y) == offset

            reordered = view.entities[10:] + view.entities[:10]
            app.current_view = replace(view, entities=reordered)
            app._refresh_browser()
            await pilot.pause()

            assert tree.cursor_node.data == scope
            assert app.selected_scope == selected_scope
            assert tree.cursor_line - int(tree.scroll_y) == offset

            without_scope = tuple(
                entity for entity in reordered if entity.entity_id != 50
            )
            app.current_view = replace(view, entities=without_scope)
            app._refresh_browser()
            await pilot.pause()

            assert tree.cursor_node.data == DiscoveryScope("object")
            assert app.selected_scope == selected_scope
            assert int(tree.scroll_y) <= tree.cursor_line
            assert tree.cursor_line < int(tree.scroll_y) + tree.size.height

    asyncio.run(exercise())


def test_signal_rows_group_by_quantity_and_class_only_when_needed():
    app = _build_mock_explorer()

    async def exercise():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._select_scope(DiscoveryScope("object"))
            await pilot.pause()
            signals = app.query_one("#signals")
            labels = [str(column.label) for column in signals.columns.values()]
            assert labels == ["Signal", "Unit / frame"]
            assert str(signals.get_row_at(0)[0]) == "▼ ungrouped (3)"
            assert signals.row_count == 4
            assert all(
                "object" not in str(signals.get_row_at(index)[0])
                for index in range(1, signals.row_count)
            )

            signals.move_cursor(row=0)
            signals.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert signals.row_count == 1
            assert str(signals.get_row_at(0)[0]) == "▶ ungrouped (3)"

            assert app._apply_filter("mass", app.show_raw)
            await pilot.pause()
            assert signals.row_count == 4
            assert "mass" in {
                str(signals.get_row_at(index)[0])
                for index in range(1, signals.row_count)
            }

            assert app._apply_filter("", app.show_raw)
            await pilot.pause()
            assert signals.row_count == 1
            assert str(signals.get_row_at(0)[0]) == "▶ ungrouped (3)"

            await pilot.press("enter")
            await pilot.pause()
            assert signals.row_count == 4

            app._select_scope(DiscoveryScope(None, role=ROLE_ABSENT))
            await pilot.pause()
            group_labels = {
                str(signals.get_row_at(index)[0])
                for index in range(signals.row_count)
                if str(signals.get_row_at(index)[0]).startswith(("▼", "▶"))
            }
            assert any("object / ungrouped" in label for label in group_labels)
            assert any(
                "robot / applied torques" in label
                for label in group_labels
            )
            last_group_row = max(
                index
                for index in range(signals.row_count)
                if str(signals.get_row_at(index)[0]).startswith(
                    ("▼", "▶")
                )
            )
            collapsed_label = str(signals.get_row_at(last_group_row)[0])
            assert collapsed_label.startswith("▶")
            signals.move_cursor(row=last_group_row)
            await pilot.press("enter")
            await pilot.pause()
            assert str(
                signals.get_row_at(signals.cursor_coordinate.row)[0]
            ) == collapsed_label.replace("▶", "▼", 1)

            mass = next(
                signal for signal in app.signal_by_key.values()
                if signal.pattern == "mass"
            )
            app._select_scope(DiscoveryScope("object"))
            app._show_signal(mass)
            app._resolve_selected_signal()
            authoring = str(
                app.query_one("#authoring_content").render()
            )
            assert "* wildcard" in authoring
            assert "@ group/tag" in authoring

    asyncio.run(exercise())


def test_group_row_toggles_on_a_single_click_without_holding_the_cursor():
    # Textual dispatches _on_click once per MRO class that defines it, so
    # the stock DataTable handler runs alongside the ClickSelect one; a
    # correct implementation yields exactly one toggle per click.
    app = _build_mock_explorer()

    async def exercise():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._select_scope(DiscoveryScope("object"))
            await pilot.pause()
            signals = app.query_one("#signals")
            assert signals.row_count == 4  # group row + 3 expanded signals
            # After a scope change the cursor sits on the first signal,
            # not on the group row.
            assert signals.cursor_coordinate.row != 0
            # Border (1) + column header (1) put data row 0 at y=2.
            await pilot.click("#signals", offset=(4, 2))
            await pilot.pause()
            assert signals.row_count == 1  # one click collapsed the group
            # The same click on the now-cursored row is the stock path;
            # it must toggle exactly once too, not twice.
            await pilot.click("#signals", offset=(4, 2))
            await pilot.pause()
            assert signals.row_count == 4

    asyncio.run(exercise())


def test_palette_column_grows_and_shrinks_across_clears():
    # Issue #97: DataTable.clear() never resets Column.content_width, so a
    # wide cell inflated the column permanently and squeezed the rest of
    # the row out of view. The palette inherits the same hazard through
    # _reset_table; here the width comes from the reference text, which a
    # long simulator key stretches (RoboCasa keys run past 40 chars).
    base_view = _build_mock_view()
    scope = DiscoveryScope("object", entity_id=1)

    def _view_with_long_key(key):
        entities = []
        for entity in base_view.entities:
            if not (
                entity.entity_class == "object" and entity.entity_id == 1
            ):
                entities.append(entity)
                continue
            entities.append(replace(
                entity,
                properties=tuple(
                    replace(prop, key=key) if prop.key == "mass" else prop
                    for prop in entity.properties
                ),
            ))
        return replace(base_view, entities=tuple(entities))

    def _form_column(table):
        return next(
            column for column in table.columns.values()
            if str(column.label) == "Selector form"
        )

    long_label = "x" * 52  # a key name, so it lands in the reference
    app = build_explorer(_view_with_long_key(long_label))

    async def exercise():
        async with app.run_test() as pilot:
            await pilot.pause()
            palette = app.query_one("#palette")

            app._select_scope(scope)
            await pilot.pause()
            mass = next(iter(app.signal_by_key.values()))
            app._show_signal(mass)
            await pilot.pause(0.1)
            # GROWS: a long key still widens the column (no regression).
            assert _form_column(palette).content_width >= len(long_label)

            app.current_view = _view_with_long_key("s")
            app._select_scope(scope)
            await pilot.pause()
            short_mass = next(iter(app.signal_by_key.values()))
            app._show_signal(short_mass)
            await pilot.pause(0.1)
            # SHRINKS: a short key narrows the column back down, and the
            # row fits the panel without clipping.
            assert _form_column(palette).content_width < len(long_label)
            total_width = sum(
                column.get_render_width(palette)
                for column in palette.columns.values()
            )
            assert total_width <= palette.size.width

    asyncio.run(exercise())


def test_role_scope_and_exact_entity_explain_distinct_selector_meanings():
    app = build_explorer(_build_role_mock_view())

    async def exercise():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            role_scope = DiscoveryScope("object", role="target")
            app._select_scope(role_scope)
            role_signal = next(
                signal for signal in app.signal_by_key.values()
                if signal.pattern == "mass"
            )
            app._show_signal(role_signal)
            app._resolve_selected_signal()
            role_authoring = str(
                app.query_one("#authoring_content").render()
            )
            assert app.query_one("#signals").border_title == (
                "Signals · object ∩ target (1)"
            )
            # A role scope must land on the role-filtered form. The
            # plain object_*[mass] binds the whole class, so defaulting
            # to it would monitor the distractor the scope excluded.
            assert app.selected_form.reference == (
                "object_*[mass @ role:target]"
            )
            assert "object_*[mass @ role:target]" in role_authoring
            assert "binds only entities declared with this role" in (
                role_authoring
            )

            entity_scope = DiscoveryScope("object", entity_id=1)
            app._select_scope(entity_scope)
            entity_signal = next(
                signal for signal in app.signal_by_key.values()
                if signal.pattern == "mass"
            )
            app._show_signal(entity_signal)
            app._resolve_selected_signal()
            entity_authoring = str(
                app.query_one("#authoring_content").render()
            )
            assert "object_1[mass]" in entity_authoring
            assert "role:target is informational" in entity_authoring
            assert "@ role:target" not in (
                entity_authoring.split("Scope")[0]
            )

    asyncio.run(exercise())
