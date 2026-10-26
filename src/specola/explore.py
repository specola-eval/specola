"""Read-only Textual explorer over the shared discovery model."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from rich.text import Text

from .discovery_filter import DiscoveryView, filter_discovery
from .run_present import OverlayPanel
from .discovery_selector import (
    DiscoveryScope,
    ROLE_ABSENT,
    entities_for_scope,
    signal_selections,
)
from .selector_palette import (
    _verified_against_model,
    default_form,
    selector_palette,
    starter_yaml_for_form,
)
from .branding import MARK as _BRAND_MARK


def _summary(view: DiscoveryView) -> Text:
    model = view.model
    counts = []
    for record in model.classes:
        if record.raw and not view.query.include_raw:
            continue
        counts.append("{} {}".format(record.live_count, record.entity_class))
    text = Text()
    text.append(model.simulator.upper(), style="bold")
    text.append(" · {}".format(model.task or model.adapter_type))
    text.append(
        " · raw {} · reset snapshot".format(
            "on" if view.query.include_raw else "off"
        ),
        style="dim",
    )
    text.append("\n")
    text.append(" · ".join(counts))
    text.append(
        " · {} signals · dt {} s".format(
            sum(len(entity.properties) for entity in view.entities),
            model.effective_dt,
        ),
        style="dim",
    )
    return text


def click_select_data_table_class(DataTable, Coordinate):
    """Build a DataTable subclass whose rows select on a single click.

    Stock Textual (8.2.8) posts RowSelected only when the clicked row
    already holds the cursor, so a click on any other row needs a second
    click to select. Textual dispatches ``_on_click`` once per MRO class
    that defines it, so the stock handler always runs as well — this
    handler must NOT call ``super()``; it posts for the clicked row and
    leaves the cursor move (and the already-on-cursor case) to stock.
    Overrides private API (_on_click, coordinate_to_cell_key message
    shape); re-check when the textual pin in pyproject.toml moves past
    8.x. Takes the textual classes as arguments because callers import
    textual lazily inside their UI entry points.
    """

    class ClickSelectDataTable(DataTable):
        async def _on_click(self, event) -> None:
            meta = event.style.meta
            if (
                "row" not in meta
                or "column" not in meta
                or meta["row"] == -1
                or meta["column"] == -1
                or not self.show_cursor
                or self.cursor_type != "row"
                or not self._data
            ):
                return
            clicked = Coordinate(meta["row"], meta["column"])
            if clicked == self.cursor_coordinate:
                return  # the stock handler posts for this case
            row_key, _ = self.coordinate_to_cell_key(clicked)
            self.post_message(
                DataTable.RowSelected(self, clicked.row, row_key)
            )

    return ClickSelectDataTable


def discovery_explorer_class(view: DiscoveryView):
    """Return the shared discovery app class bound to ``view``."""
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, VerticalScroll
        from textual.timer import Timer
        from textual.widgets import (
            DataTable, Footer, Header, Input, Static, Tree,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Textual discovery requires the bounded UI dependencies"
        ) from exc

    from textual.coordinate import Coordinate

    ClickSelectDataTable = click_select_data_table_class(
        DataTable, Coordinate
    )

    class DiscoveryExplorer(App):
        TITLE = "Specola Discover"
        BINDINGS = [
            ("q", "quit", "Exit"),
            ("ctrl+c", "quit", "Exit"),
            ("c", "toggle_browse_mode", "Class/role"),
            ("/", "focus_search", "Search"),
            ("r", "toggle_raw", "Raw"),
            ("y", "show_reference", "Reference"),
            ("?", "show_help", "Help"),
            ("escape", "close_panel", "Back"),
        ]
        CSS = """
        #summary { height: 2; padding: 0 1; }
        #search { height: 3; }
        #browser_row { height: 1fr; min-height: 8; }
        #browse { width: 43%; border: round $primary; }
        #signals { width: 57%; border: round $primary; }
        #selected_signal { height: 7; border: round $primary; padding: 0 1; }
        #palette { height: 8; border: round $primary; }
        #authoring { height: 7; border: round $primary; padding: 0 1; }
        #selected_signal_content, #authoring_content { height: auto; }
        #browse, #signals, #selected_signal, #palette, #authoring {
            border-title-color: $text-muted;
        }
        #browse:focus, #signals:focus, #palette:focus,
        #selected_signal:focus, #authoring:focus {
            border: round $accent;
            border-title-color: $accent;
            border-title-style: bold;
        }
        """

        # Coverage counts were removed at the maintainer's request
        # (2026-09-08): what a signal binds to is the selector palette's
        # job, and a per-row count invited reading it as a promise.
        _SIGNAL_COLUMNS = ("Signal", "Unit / frame")
        # One column, and it holds the whole reference. The panel this
        # replaced listed the entity/key mapping a single reference
        # resolves to; the palette lists the references themselves, and
        # what each one means belongs beside the selected row rather
        # than in a column every row has to pay for.
        _PALETTE_COLUMNS = ("Selector form",)

        def __init__(self):
            super().__init__()
            self.base_view = view
            self.current_view = view
            self.show_raw = view.query.include_raw
            self.browse_mode = "class"
            self.signal_by_key = {}
            self.signal_group_by_key = {}
            self.expanded_signal_groups = set()
            self._signal_group_scope = None
            self.selected_scope = None
            self.selected_signal = None
            self._rebuilding = False
            self._filter_timer: Optional[Timer] = None
            self._selector_timer: Optional[Timer] = None
            self._nodes_by_scope = {}
            self._authoring_panel = None
            self.palette_forms = ()
            self.form_by_key = {}
            self.selected_form = None

        def compose(self) -> ComposeResult:
            yield Header(icon=_BRAND_MARK)
            yield from self.compose_discovery()
            yield Footer()

        def compose_discovery(self) -> ComposeResult:
            """Compose the discovery surface without the app chrome."""
            yield Static(
                _summary(self.current_view), id="summary", markup=False
            )
            yield Input(
                placeholder="Search declared names, roles, signals, tags…",
                id="search",
            )
            with Horizontal(id="browser_row"):
                yield Tree(Text("Class"), id="browse")
                yield ClickSelectDataTable(id="signals", cursor_type="row")
            with VerticalScroll(id="selected_signal"):
                yield Static(
                    "Select a class, role, or entity.",
                    id="selected_signal_content",
                    markup=False,
                )
            yield ClickSelectDataTable(id="palette", cursor_type="row")
            with VerticalScroll(id="authoring"):
                yield Static(
                    "Select a signal to create a reference.",
                    id="authoring_content",
                    markup=False,
                )

        def on_mount(self) -> None:
            self.theme = "nord"
            self._authoring_panel = OverlayPanel(
                self.query_one("#authoring_content", Static),
                "Select a signal to create a reference.",
            )
            self.query_one("#browse").border_title = "Browse"
            self.query_one("#signals").border_title = "Signals"
            self.query_one("#selected_signal").border_title = (
                "Selected signal"
            )
            self.query_one("#palette").border_title = (
                "Ways to select this"
            )
            self.query_one("#authoring").border_title = (
                "Use in a specification"
            )
            signals = self.query_one("#signals", DataTable)
            signals.zebra_stripes = True
            signals.cursor_foreground_priority = "renderable"
            signals.add_columns(*self._SIGNAL_COLUMNS)
            palette = self.query_one("#palette", DataTable)
            palette.zebra_stripes = True
            palette.cursor_foreground_priority = "renderable"
            palette.add_columns(*self._PALETTE_COLUMNS)
            self._refresh_browser()
            self.query_one("#browse", Tree).focus()

        def _entity_label(self, entity) -> str:
            identity = entity.label or entity.declared_name or ""
            return "{}_{}  {}".format(
                entity.entity_class, entity.entity_id, identity
            ).rstrip()

        def _reset_table(self, table: DataTable, *columns: str) -> None:
            """Clear a table's rows *and* columns, then rebuild the columns.

            ``DataTable.clear()`` alone never resets ``Column.content_width``,
            so widths only ratchet up across repopulations (issue #97).
            ``clear(columns=True)`` drops the Column objects entirely, so
            the re-added columns start fresh.
            """
            table.clear(columns=True)
            table.add_columns(*columns)

        def _debounce(self, attr: str, delay: float, callback) -> None:
            timer = getattr(self, attr)
            if timer is None:
                setattr(self, attr, self.set_timer(delay, callback))
            else:
                timer.reset()

        def _add_branch(self, parent, scope, label, expanded):
            node = parent.add(Text(label), data=scope)
            self._nodes_by_scope[scope] = node
            if scope in expanded:
                node.expand()
            return node

        def _add_leaf(self, parent, class_name, entity):
            scope = DiscoveryScope(class_name, entity_id=entity.entity_id)
            node = parent.add_leaf(
                Text(self._entity_label(entity)), data=scope
            )
            self._nodes_by_scope[scope] = node

        def _expanded_scopes(self, tree) -> set:
            expanded = set()

            def visit(node) -> None:
                if node.data is not None and node.is_expanded:
                    expanded.add(node.data)
                for child in node.children:
                    visit(child)

            for child in tree.root.children:
                visit(child)
            return expanded

        def _refresh_browser(self) -> None:
            tree = self.query_one("#browse", Tree)
            expanded = self._expanded_scopes(tree)
            initial_build = not tree.root.children
            previous_scope = self.selected_scope
            cursor_scope = (
                tree.cursor_node.data
                if tree.cursor_node is not None else None
            )
            cursor_offset = tree.cursor_line - int(tree.scroll_y)
            tree.clear()
            tree.root.label = Text(
                "Class" if self.browse_mode == "class" else "Role"
            )
            tree.root.expand()
            entities = self.current_view.entities
            first_scope = None
            self._nodes_by_scope = {}
            if self.browse_mode == "class":
                class_names = sorted({
                    entity.entity_class for entity in entities
                } | {
                    record.entity_class
                    for record in self.current_view.model.declared_classes
                    if record.live_count == 0
                })
                for class_name in class_names:
                    members = tuple(
                        entity for entity in entities
                        if entity.entity_class == class_name
                    )
                    declared = any(
                        record.entity_class == class_name
                        for record in self.current_view.model.declared_classes
                    )
                    suffix = "{}".format(len(members))
                    if not members and declared:
                        suffix = "0 live · declared"
                    scope = DiscoveryScope(class_name)
                    node = self._add_branch(
                        tree.root, scope,
                        "{}  {}".format(class_name, suffix), expanded,
                    )
                    if first_scope is None and members:
                        first_scope = scope
                    roles = sorted({
                        entity.role for entity in members if entity.role
                    })
                    role_ids = set()
                    for role in roles:
                        role_members = tuple(
                            entity for entity in members
                            if entity.role == role
                        )
                        role_node = self._add_branch(
                            node, DiscoveryScope(class_name, role=role),
                            "{}  {}".format(role, len(role_members)),
                            expanded,
                        )
                        for entity in role_members:
                            role_ids.add(entity.entity_id)
                            self._add_leaf(role_node, class_name, entity)
                    for entity in members:
                        if entity.entity_id not in role_ids:
                            self._add_leaf(node, class_name, entity)
            else:
                roles = sorted({
                    entity.role for entity in entities if entity.role
                })
                if any(entity.role is None for entity in entities):
                    roles.append(ROLE_ABSENT)
                for role in roles:
                    role_label = (
                        "(no declared role)"
                        if role is ROLE_ABSENT else role
                    )
                    members = tuple(
                        entity for entity in entities
                        if (
                            entity.role is None
                            if role is ROLE_ABSENT
                            else entity.role == role
                        )
                    )
                    role_scope = DiscoveryScope(None, role=role)
                    role_node = self._add_branch(
                        tree.root, role_scope,
                        "{}  {}".format(role_label, len(members)),
                        expanded,
                    )
                    if first_scope is None and members:
                        first_scope = role_scope
                    for class_name in sorted({
                        entity.entity_class for entity in members
                    }):
                        class_members = tuple(
                            entity for entity in members
                            if entity.entity_class == class_name
                        )
                        class_node = self._add_branch(
                            role_node, DiscoveryScope(class_name, role=role),
                            "{}  {}".format(
                                class_name, len(class_members)
                            ),
                            expanded,
                        )
                        for entity in class_members:
                            self._add_leaf(class_node, class_name, entity)
            self.query_one("#summary", Static).update(
                _summary(self.current_view)
            )
            selected_scope = (
                previous_scope
                if previous_scope in self._nodes_by_scope
                else first_scope
            )
            if selected_scope is not None:
                selected_node = self._nodes_by_scope[selected_scope]
                if initial_build and not expanded:
                    parent = selected_node.parent
                    if parent is not None:
                        parent.expand()
                    selected_node.expand()
                self._select_scope(selected_scope)
                cursor_node = self._nodes_by_scope.get(cursor_scope)
                if cursor_node is not None:
                    tree.move_cursor(cursor_node)

                    def restore_cursor_and_viewport() -> None:
                        restored = self._nodes_by_scope.get(cursor_scope)
                        if restored is None:
                            return
                        tree.move_cursor(restored)
                        tree.scroll_to(
                            y=max(0, tree.cursor_line - cursor_offset),
                            animate=False,
                            force=True,
                            immediate=True,
                        )

                    self.call_after_refresh(restore_cursor_and_viewport)
                else:
                    tree.select_node(selected_node)

                    def restore_fallback() -> None:
                        fallback = self._nodes_by_scope.get(selected_scope)
                        if fallback is None:
                            return
                        tree.move_cursor(fallback)
                        tree.scroll_to_node(fallback, animate=False)

                    self.call_after_refresh(restore_fallback)
            else:
                self._clear_signal_details()

        def _select_scope(self, scope) -> None:
            if self._selector_timer is not None:
                self._selector_timer.stop()
                self._selector_timer = None
            self.selected_signal = None
            self._reset_palette()
            self.query_one("#selected_signal_content", Static).update(
                "Select a signal in {}.".format(scope.label)
            )
            self._set_authoring(
                "Select a signal to create a reference."
            )
            self.selected_scope = scope
            table = self.query_one("#signals", DataTable)
            scope_count = len(entities_for_scope(self.current_view, scope))
            table.border_title = "Signals · {} ({})".format(
                scope.label, scope_count
            )
            selections = signal_selections(self.current_view, scope)
            groups = {}
            for signal in selections:
                entity_class = signal.mappings[0].entity_class
                quantity = signal.quantity or "ungrouped"
                group_id = (scope, entity_class, quantity)
                groups.setdefault(group_id, []).append(signal)
            if self._signal_group_scope != scope:
                self.expanded_signal_groups.clear()
                self._signal_group_scope = scope
                if groups:
                    self.expanded_signal_groups.add(next(iter(groups)))

            self._rebuilding = True
            visible_signals = []
            try:
                self._reset_table(table, *self._SIGNAL_COLUMNS)
                self.signal_by_key = {
                    "signal:{}".format(index): signal
                    for index, signal in enumerate(selections)
                }
                keys_by_signal = {
                    id(signal): key
                    for key, signal in self.signal_by_key.items()
                }
                self.signal_group_by_key = {}
                row_index = 0
                for group_index, (group_id, group_signals) in enumerate(
                    groups.items()
                ):
                    _scope, entity_class, quantity = group_id
                    expanded = (
                        bool(self.current_view.query.search)
                        or group_id in self.expanded_signal_groups
                    )
                    group_key = "group:{}".format(group_index)
                    self.signal_group_by_key[group_key] = group_id
                    label = quantity
                    if scope.entity_class is None:
                        label = "{} / {}".format(entity_class, quantity)
                    marker = "▼" if expanded else "▶"
                    table.add_row(
                        Text(
                            "{} {} ({})".format(
                                marker, label, len(group_signals)
                            ),
                            style="bold",
                        ),
                        Text(""),
                        key=group_key,
                    )
                    row_index += 1
                    if not expanded:
                        continue
                    for signal in group_signals:
                        unit_frame = " / ".join(
                            item for item in (signal.unit, signal.frame)
                            if item
                        ) or "—"
                        pattern_cell = Text(signal.pattern)
                        if signal.component:
                            pattern_cell.append(
                                " · {}".format(signal.component),
                                style="dim",
                            )
                        table.add_row(
                            pattern_cell,
                            Text(unit_frame, style="dim"),
                            key=keys_by_signal[id(signal)],
                        )
                        visible_signals.append((row_index, signal))
                        row_index += 1
            finally:
                self._rebuilding = False
            if visible_signals:
                row, signal = visible_signals[0]

                def select_first_visible_signal() -> None:
                    if self.selected_scope != scope:
                        return
                    table.move_cursor(row=row)
                    self._show_signal(signal)

                self.call_after_refresh(select_first_visible_signal)
            elif not selections:
                self._clear_signal_details(
                    "No live signals in {}.".format(scope.label)
                )

        def _clear_signal_details(self, message="No matching entities.") -> None:
            self.selected_signal = None
            self._rebuilding = True
            try:
                self._reset_table(
                    self.query_one("#signals", DataTable),
                    *self._SIGNAL_COLUMNS,
                )
            finally:
                self._rebuilding = False
            self._reset_palette()
            self.query_one("#selected_signal_content", Static).update(
                message
            )
            self._set_authoring("No selector preview is available.")

        def _show_signal(self, signal) -> None:
            self.selected_signal = signal
            self.query_one("#selected_signal_content", Static).update(
                "Pattern      {}\nQuantity     {}\nComponent    {}\n"
                "Unit         {}\nFrame        {}".format(
                    signal.pattern,
                    signal.quantity or "—",
                    signal.component or "—",
                    signal.unit or "—",
                    signal.frame or "—",
                )
            )
            self._reset_palette()
            self._set_authoring("Resolving selector forms…")
            self._debounce(
                "_selector_timer", 0.05, self._resolve_selected_signal
            )

        def _reset_palette(self) -> None:
            self.palette_forms = ()
            self.form_by_key = {}
            self.selected_form = None
            self._reset_table(
                self.query_one("#palette", DataTable),
                *self._PALETTE_COLUMNS,
            )

        def _resolve_selected_signal(self) -> None:
            """Fill the palette for the current selection.

            Debounced rather than lazy: measured worst case on RoboCasa
            is 0.105 s for a signal-scoped palette, which is inside the
            budget for a keystroke-driven panel, and resolving every row
            up front is what lets an unavailable shape be shown with its
            reason instead of appearing only once it is focused.
            """
            self._selector_timer = None
            if not self.is_running:
                return
            signal = self.selected_signal
            scope = self.selected_scope
            if scope is None:
                return
            forms = selector_palette(
                self.current_view, scope, signal=signal
            )
            # Checked, because this surface offers a pasteable starter
            # for any row it calls available and bracket availability is
            # not the compiler's answer: ten of RoboCasa's sixteen
            # `contact` forms resolve cleanly and do not compile. The
            # catalog comes from the view's own model, so there is
            # nothing for a caller to forget to pass. The writer
            # overrides this method and checks again with its own
            # catalog, which is the same verdict reached twice, not a
            # different one.
            forms = _verified_against_model(self.current_view, forms)
            self.palette_forms = forms
            table = self.query_one("#palette", DataTable)
            self._rebuilding = True
            try:
                self._reset_table(table, *self._PALETTE_COLUMNS)
                self.form_by_key = {}
                for index, form in enumerate(forms):
                    key = "form:{}".format(index)
                    self.form_by_key[key] = form
                    text = Text(form.reference)
                    if form.partner_reference:
                        text.append(
                            "  ↔  " + form.partner_reference, style="dim"
                        )
                    if not form.available:
                        # Dimmed rather than hidden: a shape the grammar
                        # cannot express here is a fact about the scene
                        # that an author otherwise discovers at compile
                        # time.
                        text.stylize("dim")
                    table.add_row(text, key=key)
            finally:
                self._rebuilding = False
            if not forms:
                self._set_authoring(
                    "No selector form is available for this selection."
                )
                return
            default = default_form(forms, scope)
            first = forms.index(default) if default is not None else 0
            # Selected synchronously, cursor moved after the refresh: a
            # caller that resolves the palette and immediately inserts
            # (the keyboard path, and every test) must not depend on a
            # repaint having happened first.
            self._show_form(forms[first])

            def move_cursor_to_first_form() -> None:
                if self.selected_scope != scope:
                    return
                table.move_cursor(row=first)

            self.call_after_refresh(move_cursor_to_first_form)

        def _show_form(self, form) -> None:
            """Explain one palette row beside it."""
            self.selected_form = form
            lines = ["Reference        {}".format(form.reference)]
            if form.partner_reference:
                lines.append(
                    "Second slot      {}".format(form.partner_reference)
                )
            lines.append(
                "Shape            {} binding · {} component".format(
                    form.binding_kind, form.component_kind
                )
            )
            if form.available:
                if form.note:
                    lines.append("Means            {}".format(form.note))
                lines.append("")
                # Generated by the shared starter authority and quoted by
                # the YAML writer, so what is shown here is exactly what
                # compiles -- including the quoting a difference form's
                # backslash needs.
                lines.extend(
                    starter_yaml_for_form(form).rstrip().splitlines()
                )
            else:
                lines.append(
                    "Unavailable      {}".format(form.unavailable_reason)
                )
            scope = self.selected_scope
            if scope is not None and scope.entity_id is not None:
                # Pinning already names the entity, so its declared role
                # constrains nothing here. Saying so keeps the pinned
                # and role-scoped cases visibly different, which is the
                # distinction that decides how many monitors you get.
                entity = next(
                    iter(entities_for_scope(self.current_view, scope)),
                    None,
                )
                if entity is not None and entity.role:
                    lines.append(
                        "Scope            exact entity; role:{} is "
                        "informational".format(entity.role)
                    )
            lines.append("")
            lines.append(
                "Selector syntax  * wildcard · {a, b} set · @ group/tag "
                "· \\ difference · any:/all: quantifier"
            )
            if self.current_view.model.warnings:
                lines.append(
                    "Warning          {}".format(
                        self.current_view.model.warnings[0]
                    )
                )
            self._set_authoring("\n".join(lines))

        def _set_authoring(self, text: str) -> None:
            self._authoring_panel.set_text(text)

        def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
            if not self.is_running:
                return
            if (
                event.node.data is not None
                and event.node.data != self.selected_scope
            ):
                self._select_scope(event.node.data)

        def on_data_table_row_highlighted(
            self, event: DataTable.RowHighlighted
        ) -> None:
            if not self.is_running or self._rebuilding:
                return
            if event.data_table.id == "palette":
                form = self.form_by_key.get(str(event.row_key.value))
                if form is not None:
                    self._show_form(form)
                return
            if event.data_table.id != "signals":
                return
            signal = self.signal_by_key.get(str(event.row_key.value))
            if signal is not None:
                self._show_signal(signal)

        def on_data_table_row_selected(
            self, event: DataTable.RowSelected
        ) -> None:
            if event.data_table.id != "signals" or self._rebuilding:
                return
            if self.current_view.query.search:
                self.notify(
                    "Signal groups stay expanded while search is active.",
                    markup=False,
                )
                return
            group_id = self.signal_group_by_key.get(
                str(event.row_key.value)
            )
            if group_id is None:
                return
            if group_id in self.expanded_signal_groups:
                self.expanded_signal_groups.remove(group_id)
            else:
                self.expanded_signal_groups.add(group_id)
            if self.selected_scope is not None:
                scope = self.selected_scope
                table = event.data_table
                self._select_scope(scope)

                def restore_group_cursor() -> None:
                    if self.selected_scope != scope:
                        return
                    key = next((
                        row_key
                        for row_key, candidate
                        in self.signal_group_by_key.items()
                        if candidate == group_id
                    ), None)
                    if key is None:
                        return
                    table.move_cursor(row=table.get_row_index(key))

                self.call_after_refresh(restore_group_cursor)

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id != "search":
                return
            self._debounce("_filter_timer", 0.2, self._apply_pending_filter)

        def _apply_pending_filter(self) -> None:
            self._filter_timer = None
            if not self.is_running:
                return
            value = self.query_one("#search", Input).value
            self._apply_filter(value, self.show_raw)

        def _apply_filter(self, value: str, include_raw: bool) -> bool:
            search = (value,) if value else ()
            query = replace(
                self.base_view.query,
                search=search,
                include_raw=include_raw,
            )
            try:
                current_view = filter_discovery(
                    self.base_view.model, query
                )
            except ValueError as exc:
                self._authoring_panel.dismiss_overlay()
                self._set_authoring(str(exc))
                return False
            self.show_raw = include_raw
            self.current_view = current_view
            self._refresh_browser()
            return True

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id != "search":
                return
            if self._filter_timer is not None:
                self._filter_timer.stop()
                self._filter_timer = None
            self._apply_filter(event.value, self.show_raw)
            self.query_one("#browse", Tree).focus()

        def action_toggle_raw(self) -> None:
            search_widget = self.query_one("#search", Input)
            previous_ids = {
                (entity.entity_class, entity.entity_id)
                for entity in self.current_view.entities
            }
            proposed = not self.show_raw
            if not self._apply_filter(search_widget.value, proposed):
                return
            current_ids = {
                (entity.entity_class, entity.entity_id)
                for entity in self.current_view.entities
            }
            changed = (
                current_ids - previous_ids
                if proposed else previous_ids - current_ids
            )
            class_count = len({entity_class for entity_class, _ in changed})
            self.notify(
                "Raw {}: {} {} across {} {} {}.".format(
                    "on" if proposed else "off",
                    "added" if proposed else "hidden",
                    len(changed),
                    class_count,
                    "class" if class_count == 1 else "classes",
                    "" if changed else "(no matching raw entities)",
                ).rstrip(),
                markup=False,
            )

        def action_toggle_browse_mode(self) -> None:
            self.browse_mode = (
                "role" if self.browse_mode == "class" else "class"
            )
            self._refresh_browser()

        def action_focus_search(self) -> None:
            self.query_one("#search", Input).focus()

        def action_show_reference(self) -> None:
            if self.selected_form is not None:
                self._show_form(self.selected_form)
            elif self.selected_scope is not None:
                self._resolve_selected_signal()

        def action_show_help(self) -> None:
            self._authoring_panel.show_overlay(
                "↑/↓ navigate · Enter signal group · Tab panel · "
                "C class/role · / search · R raw · y reference · "
                "* wildcard · {a, b} set · @ group/tag · \\ difference · "
                "any:/all: quantifier · ? help · q exit"
            )

        def action_close_panel(self) -> None:
            search = self.query_one("#search", Input)
            if search.has_focus:
                self.query_one("#browse", Tree).focus()
                return
            self._authoring_panel.dismiss_overlay()

    return DiscoveryExplorer


def build_explorer(view: DiscoveryView):
    return discovery_explorer_class(view)()


def run_explorer(view: DiscoveryView) -> None:
    build_explorer(view).run()
