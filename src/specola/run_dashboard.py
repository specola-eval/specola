"""Textual dashboard shell consuming immutable run snapshots."""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import math
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Event, Thread
import time
from typing import Any, Callable, Iterable, Optional, TextIO

from rich.text import Text

from .cli_common import STATE_COLOR_NAMES
from .run_present import OverlayPanel
from .run_state import (
    RunSnapshot,
    SpecificationSnapshot,
    recovery_transitions,
)
from .branding import MARK as _BRAND_MARK


@dataclass(frozen=True)
class DashboardResult:
    snapshot: Optional[RunSnapshot]
    stopped: bool


class WorkerAbandoned(RuntimeError):
    """Ctrl-C abandoned a simulator worker that may still be running.

    The session may be mid-mutation; callers must not finalize results.
    """


@dataclass(frozen=True)
class _DashboardUpdate:
    snapshot: RunSnapshot
    transitions: tuple


# Canonical mapping shared with the plain renderer; Rich has no "amber".
_STATE_STYLES = {
    state: "yellow" if name == "amber" else name
    for state, name in STATE_COLOR_NAMES.items()
}


def _state_cell(value: str, _row_style=None) -> Text:
    """Style ' · '-separated chunks that are exactly a state, or a count
    followed by a state. Free-text tails (skip reasons, population
    summaries) stay unstyled even when they contain state words."""
    text = Text()
    for index, part in enumerate(value.split(" · ")):
        if index:
            text.append(" · ")
        words = part.split(" ")
        style = None
        if len(words) == 1:
            style = _STATE_STYLES.get(words[0])
        elif len(words) == 2 and words[0].isdigit():
            style = _STATE_STYLES.get(words[1])
        if style is not None and words[-1] == "violated":
            style = "bold red"
        text.append(part, style=style)
    return text


def _row_style(state_value: str):
    """Whole-row color for a binding or specification state."""
    style = _STATE_STYLES.get(state_value)
    if state_value == "violated":
        return "bold red"
    return style


# Worst binding state wins the specification row's color.
_SPEC_STYLE_PRECEDENCE = (
    "violated", "error", "missing_data", "pending",
    "rewarming", "suspended", "satisfied",
)


def _spec_row_style(specification):
    """Aggregate row color for a specification header row.

    SpecificationState values (active/waiting/…) are not binding states
    and mostly have no color; the row takes the color of the worst
    binding state present instead.
    """
    present = {
        state.value for state, _count in specification.state_counts
    }
    for name in _SPEC_STYLE_PRECEDENCE:
        if name in present:
            return _row_style(name)
    return _STATE_STYLES.get(specification.state.value)


_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(trace, tail=None, annotate=True) -> str:
    """Render a robustness trace as unicode blocks.

    None and non-finite samples are gaps: the trace is appended unfiltered
    (run_state.py), so nan/inf can appear in it. ``tail`` keeps only the
    last N samples; ``annotate`` appends the [lo … hi] range.
    """
    if tail is not None:
        trace = trace[-tail:]
    values = [
        value for value in trace
        if value is not None and math.isfinite(value)
    ]
    if len(values) < 2:
        return "—" if annotate else ""
    lo, hi = min(values), max(values)
    span = hi - lo
    chars = []
    for value in trace:
        if value is None or not math.isfinite(value):
            chars.append(" ")
        elif span == 0.0:
            chars.append(_SPARK_BLOCKS[3])
        else:
            chars.append(
                _SPARK_BLOCKS[int((value - lo) / span * 7)]
            )
    line = "".join(chars)
    if annotate:
        line += "  [{:+.3g} … {:+.3g}]".format(lo, hi)
    return line


def _sign_cell(value: str, row_style=None) -> Text:
    """Robustness numbers follow their row's state color when it has one;
    sign coloring is the fallback. The sign itself stays textual."""
    if row_style:
        return Text(value, style=row_style)
    if value.startswith("+"):
        return Text(value, style="green")
    if value.startswith("-"):
        return Text(value, style="red")
    return Text(value)




def _specification_status(
    specification: SpecificationSnapshot,
    *,
    expanded: bool,
) -> str:
    status = " · ".join(
        "{} {}".format(count, state.value)
        for state, count in specification.state_counts
    ) or specification.state.value.replace("_", " ")
    if specification.skip_reason:
        status += " · " + specification.skip_reason
    if (
        expanded
        and specification.active_count != specification.seen_count
    ):
        status += " · {} active / {} seen".format(
            specification.active_count,
            specification.seen_count,
        )
    return status


def _simulator_log_excerpt(path: Optional[str], limit: int = 16384) -> str:
    if path is None:
        return "No durable simulator log was enabled."
    log_path = Path(path)
    try:
        with log_path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            data = stream.read()
    except OSError as exc:
        return "Unable to read simulator log: {}: {}".format(
            type(exc).__name__, exc
        )
    text = data.decode("utf-8", "replace")
    if size > limit:
        text = "… preceding log content omitted …\n" + text.split("\n", 1)[-1]
    return "{}\n\nFull text: {}".format(
        text.rstrip() or "No simulator messages were captured.",
        str(log_path.resolve()),
    )


def run_textual_dashboard(
    snapshots: Callable[[Event], Iterable[RunSnapshot]],
    *,
    headless: bool = False,
    simulator_log: Optional[str] = None,
    simulator_stream: Optional[TextIO] = None,
    build_only: bool = False,
    refresh_rate: float = 10.0,
    step_on_main_thread: bool = False,
) -> Any:
    """Consume a monitoring stream in a worker while Textual owns the TTY.

    ``step_on_main_thread`` advances the stream on the thread running
    Textual's event loop instead, one step per callback, for simulators
    that can only be stepped on the main thread.
    """
    if refresh_rate <= 0:
        raise ValueError("refresh_rate must be positive")
    try:
        from textual.app import App, ComposeResult
        from textual.containers import VerticalScroll
        from textual.coordinate import Coordinate
        from textual.widgets import DataTable, Footer, Header, Input, Static

        from .explore import click_select_data_table_class

        ClickSelectDataTable = click_select_data_table_class(
            DataTable, Coordinate
        )
    except ImportError as exc:
        raise RuntimeError(
            "Textual run UI requires the bounded UI dependencies"
        ) from exc

    stop = Event()
    done = Event()
    queue = Queue(maxsize=1)
    result = {"snapshot": None, "error": None, "stopped": False}

    def mark_stopped() -> None:
        result["stopped"] = True
        stop.set()

    def publish(snapshot: RunSnapshot) -> None:
        # Coalesce into the one-slot queue without losing transitions.
        result["snapshot"] = snapshot
        carry = list(snapshot.transitions)
        while True:
            try:
                queue.put_nowait(_DashboardUpdate(snapshot, tuple(carry)))
                return
            except Full:
                try:
                    previous = queue.get_nowait()
                except Empty:
                    continue
                carry[:0] = previous.transitions

    class RunDashboard(App):
        TITLE = "Specola Run"
        BINDINGS = [
            ("q", "stop_run", "Stop"),
            ("ctrl+c", "stop_run", "Stop"),
            ("ctrl+q", "stop_run", "Stop"),
            ("space", "toggle_expand", "Expand"),
            ("enter", "inspect", "Inspect"),
            ("j", "navigate_down", "Down"),
            ("k", "navigate_up", "Up"),
            ("shift+j", "inspect_json", "JSON"),
            ("t", "transitions", "Transitions"),
            ("l", "logs", "Logs"),
            ("/", "focus_search", "Search"),
            ("f", "filter", "Filter"),
            ("s", "sort", "Sort"),
            ("y", "expose", "Expose"),
            ("?", "help", "Help"),
            ("escape", "close_inspector", "Back"),
        ]
        CSS = """
        #status { height: 2; padding: 0 1; }
        #search { height: 3; }
        #run { height: 1fr; border: round $primary; }
        #transitions { height: 5; border: round $primary; padding: 0 1; }
        #detail {
            height: 35%; min-height: 8; max-height: 18;
            padding: 0 1; border: round $primary;
        }
        #detail_content { height: auto; }
        #run, #transitions, #detail { border-title-color: $text-muted; }
        #run:focus, #transitions:focus, #detail:focus {
            border: round $accent;
            border-title-color: $accent;
            border-title-style: bold;
        }
        """

        def compose(self) -> ComposeResult:
            yield Header(icon=_BRAND_MARK)
            yield Static(
                "Loading monitor state…", id="status", markup=False
            )
            yield Input(
                placeholder="Search specification or binding…", id="search"
            )
            yield ClickSelectDataTable(id="run", cursor_type="row")
            yield Static(
                "No transitions yet.", id="transitions", markup=False
            )
            with VerticalScroll(id="detail"):
                yield Static(
                    "Enter inspect · Space expand · / search · "
                    "f filter · s sort · q stop",
                    id="detail_content",
                    markup=False,
                )
            yield Footer()

        def on_mount(self) -> None:
            self.theme = "nord"
            if simulator_stream is not None:
                self.begin_capture_print(self)
            self.query_one("#run").border_title = "Specifications"
            self.query_one("#transitions").border_title = (
                "Recent transitions"
            )
            self.query_one("#detail").border_title = "Inspector"
            table = self.query_one("#run", DataTable)
            table.zebra_stripes = True
            # State colors must survive the cursor row; the cursor keeps
            # its background but not its foreground override.
            table.cursor_foreground_priority = "renderable"
            self.column_keys = table.add_columns(
                "Specification / binding",
                "State",
                "Now",
                "Worst",
                "Trace",
            )
            (
                label_key, state_key, now_key, worst_key, _trace_key,
            ) = self.column_keys
            self.cell_stylers = {
                state_key: _state_cell,
                now_key: _sign_cell,
                worst_key: _sign_cell,
            }
            self.rendered_row_styles = {}
            self.bindings_by_key = {}
            self.specifications_by_key = {}
            self.expanded = set()
            self.latest_snapshot = None
            self.search_text = ""
            self.state_filter = None
            self.sort_mode = "authored"
            self.renderer_errors = deque(maxlen=20)
            self.rendered_rows = {}
            self.binding_labels = {}
            # Matches the 3 content rows the 5-row bordered panel shows;
            # a larger buffer would clip its newest entries.
            self.recent_transitions = deque(maxlen=3)
            self.exit_scheduled = False
            self.detail_panel = OverlayPanel(
                self.query_one("#detail_content", Static),
                "Enter inspect · Space expand · / search · "
                "f filter · s sort · q stop",
            )
            # Wall clock starts when monitoring starts, not at the first
            # rendered snapshot, so a slow first simulator step is visible.
            self.wall_start = time.monotonic()
            table.focus()
            if step_on_main_thread:
                self.snapshot_iter = iter(snapshots(stop))
                self.call_after_refresh(self.drive_one_snapshot)
            else:
                self.driver_thread = Thread(
                    target=self.drive_snapshots,
                    name="specola-run-driver",
                    daemon=True,
                )
                self.driver_thread.start()
            self.set_interval(1.0 / refresh_rate, self.poll_snapshots)
            if headless:
                # A deterministic test escape hatch if a backend suppresses
                # timers while no terminal driver is attached.
                self.set_timer(1.0, self._headless_stop)

        def drive_snapshots(self) -> None:
            try:
                for snapshot in snapshots(stop):
                    publish(snapshot)
            except BaseException as exc:
                result["error"] = exc
            finally:
                done.set()

        def drive_one_snapshot(self) -> None:
            # One simulator step per callback on the event loop's thread;
            # the UI renders and handles keys between steps, not during one.
            try:
                snapshot = next(self.snapshot_iter, None)
            except BaseException as exc:
                result["error"] = exc
                snapshot = None
            if snapshot is None:
                done.set()
                return
            publish(snapshot)
            self.call_after_refresh(self.drive_one_snapshot)

        def on_print(self, event) -> None:
            if simulator_stream is not None:
                simulator_stream.write(event.text)
                simulator_stream.flush()

        def poll_snapshots(self) -> None:
            latest = None
            transitions = []
            while True:
                try:
                    update = queue.get_nowait()
                except Empty:
                    break
                latest = update.snapshot
                transitions.extend(update.transitions)
            if latest is not None:
                try:
                    self._show(latest)
                except Exception as exc:
                    self.renderer_errors.append(
                        "{}: {}".format(type(exc).__name__, exc)
                    )
                    try:
                        self._set_detail(
                            "Renderer warning: {}\n"
                            "Monitoring and result recording continue.".format(
                                self.renderer_errors[-1]
                            )
                        )
                    except Exception:
                        pass
                # Drained deltas must survive a rendering failure; only a
                # failure in the transition panel itself may drop them.
                if transitions:
                    try:
                        self._record_transition_deltas(transitions)
                    except Exception as exc:
                        self.renderer_errors.append(
                            "{}: {}".format(type(exc).__name__, exc)
                        )
            if (
                done.is_set()
                and queue.empty()
                and not self.exit_scheduled
            ):
                self.exit_scheduled = True
                # Let the last coalesced snapshot reach the terminal before
                # closing the application.  This matters when a producer
                # completes inside a single presentation interval.
                self.call_after_refresh(self.exit)

        def _show(
            self,
            snapshot: RunSnapshot,
            *,
            rebuild: bool = False,
        ) -> None:
            self.latest_snapshot = snapshot
            table = self.query_one("#run", DataTable)
            desired_rows = []
            desired_styles = {}
            bindings_by_key = {}
            specifications_by_key = {}
            specifications = list(snapshot.specifications)
            if self.sort_mode == "entity":
                specifications.sort(key=lambda item: item.name.casefold())
            active = sum(
                specification.active_count
                for specification in specifications
            )
            seen = sum(
                specification.seen_count
                for specification in specifications
            )
            population = (
                "{} active / {} seen monitors".format(active, seen)
                if active != seen
                else "{} active monitors".format(active)
            )
            status = Text()
            status.append("step {}".format(snapshot.step), style="bold")
            status.append(
                " · t={:.3f} s · wall={:.1f} s".format(
                    snapshot.simulator_time,
                    time.monotonic() - self.wall_start,
                ),
                style="dim",
            )
            status.append(
                " · {} specs · {}".format(len(specifications), population)
            )
            self.query_one("#status", Static).update(status)
            for specification in specifications:
                searchable = specification.name.casefold()
                specification_search_hit = (
                    not self.search_text
                    or self.search_text in searchable
                )
                binding_search_hit = (
                    not self.search_text
                    or any(
                        self.search_text in binding.label.casefold()
                        for binding in specification.bindings
                    )
                )
                if not (
                    specification_search_hit or binding_search_hit
                ):
                    continue
                specification_state_hit = (
                    self.state_filter is None
                    or specification.state.value == self.state_filter
                )
                binding_state_hit = (
                    self.state_filter is None
                    or any(
                        binding.state.value == self.state_filter
                        for binding in specification.bindings
                    )
                )
                if not (
                    specification_state_hit or binding_state_hit
                ):
                    continue
                is_expanded = specification.name in self.expanded
                counts = _specification_status(
                    specification,
                    expanded=is_expanded,
                )
                marker = (
                    "▼" if is_expanded else "▶"
                )
                spec_key = "spec:{}".format(specification.name)
                specifications_by_key[spec_key] = specification
                desired_styles[spec_key] = _spec_row_style(specification)
                desired_rows.append((
                    spec_key,
                    (
                        "{} {}".format(marker, specification.name),
                        counts,
                        "",
                        "",
                        "",
                    ),
                ))
                for binding in specification.bindings:
                    label_key = (specification.name, binding.binding_key)
                    if label_key not in self.binding_labels:
                        self.binding_labels[label_key] = binding.label
                if not is_expanded:
                    continue
                bindings = list(specification.bindings)
                if self.sort_mode == "entity":
                    bindings.sort(key=lambda item: item.label.casefold())
                elif self.sort_mode == "recent":
                    bindings.sort(
                        key=lambda item: (
                            item.transitions[-1].step
                            if item.transitions else -1
                        ),
                        reverse=True,
                    )
                for binding in bindings:
                    if (
                        self.search_text
                        and not specification_search_hit
                        and self.search_text not in binding.label.casefold()
                    ):
                        continue
                    if (
                        self.state_filter is not None
                        and not specification_state_hit
                        and binding.state.value != self.state_filter
                    ):
                        continue
                    row_key = "binding:{}:{}".format(
                        binding.manager_name, binding.binding_key
                    )
                    bindings_by_key[row_key] = binding
                    state = binding.state.value
                    if not binding.active:
                        state += " · inactive"
                    unit = " " + binding.unit if binding.unit else ""
                    desired_styles[row_key] = _row_style(
                        binding.state.value
                    )
                    desired_rows.append((
                        row_key,
                        (
                            "    {}".format(binding.label),
                            state,
                            "—" if binding.current_robustness is None
                            # +0.0 normalizes IEEE -0.0 so a satisfied
                            # boundary value never renders as red -0.000.
                            else "{:+.3f}{}".format(
                                binding.current_robustness + 0.0,
                                unit,
                            ),
                            "—" if binding.worst_robustness is None
                            else "{:+.3f}{}".format(
                                binding.worst_robustness + 0.0,
                                unit,
                            ),
                            # ljust keeps the trace column at a constant
                            # 24, sparing the O(rows) column re-measure
                            # Textual runs when a cell narrows.
                            _sparkline(
                                binding.trace, tail=24, annotate=False
                            ).ljust(24),
                        ),
                    ))
            self.bindings_by_key = bindings_by_key
            self.specifications_by_key = specifications_by_key
            self._apply_rows(
                table, desired_rows, desired_styles, rebuild=rebuild
            )

        def _styled(self, column_key, value: str, row_style=None) -> Text:
            styler = self.cell_stylers.get(column_key)
            if styler is not None:
                return styler(value, row_style)
            return Text(value, style=row_style or "")

        def _apply_rows(
            self, table, desired_rows, desired_styles, *, rebuild: bool
        ) -> None:
            selected_key = None
            selected_row = table.cursor_coordinate.row
            if table.row_count:
                try:
                    row_key, _ = table.coordinate_to_cell_key(
                        table.cursor_coordinate
                    )
                    selected_key = str(row_key.value)
                except Exception:
                    selected_key = None
            scroll_y = table.scroll_y
            desired = dict(desired_rows)
            desired_order = [key for key, _cells in desired_rows]

            if not rebuild:
                # In-place updates can only remove rows and append new rows
                # at the end. A new row that belongs mid-table (a dynamic
                # binding under an earlier expanded specification) or a
                # changed relative order needs the rebuild path, which
                # restores cursor and scroll itself.
                kept = [
                    key for key in self.rendered_rows if key in desired
                ]
                if (
                    desired_order[:len(kept)] != kept
                    or any(
                        key in self.rendered_rows
                        for key in desired_order[len(kept):]
                    )
                ):
                    rebuild = True

            if rebuild:
                table.clear()
                self.rendered_rows = {}
                self.rendered_row_styles = {}
            else:
                for key in tuple(self.rendered_rows):
                    if key not in desired:
                        table.remove_row(key)
                        self.rendered_rows.pop(key, None)
                        self.rendered_row_styles.pop(key, None)

            for key, cells in desired_rows:
                row_style = desired_styles.get(key)
                previous = self.rendered_rows.get(key)
                if previous is None:
                    table.add_row(
                        *(
                            self._styled(column_key, cell, row_style)
                            for column_key, cell in zip(
                                self.column_keys, cells
                            )
                        ),
                        key=key,
                    )
                    continue
                # A row whose state color changed must repaint every cell,
                # even those whose text is unchanged.
                style_changed = (
                    self.rendered_row_styles.get(key) != row_style
                )
                for column_key, old, new in zip(
                    self.column_keys, previous, cells
                ):
                    if style_changed or old != new:
                        table.update_cell(
                            key,
                            column_key,
                            # update_width=True: columns are sized on
                            # add_row, so a state that later widens
                            # ("3 pending" -> "3 satisfied") would clip
                            # without a re-measure.
                            self._styled(column_key, new, row_style),
                            update_width=True,
                        )

            self.rendered_rows = desired
            self.rendered_row_styles = desired_styles
            if not desired_order:
                return
            if selected_key in desired:
                target_row = table.get_row_index(selected_key)
            else:
                target_row = min(selected_row, len(desired_order) - 1)
            table.move_cursor(row=target_row, scroll=False)
            if rebuild:
                table.scroll_to(
                    y=scroll_y,
                    animate=False,
                    force=True,
                    immediate=True,
                )
                self.call_after_refresh(
                    table.scroll_to,
                    y=scroll_y,
                    animate=False,
                    force=True,
                    immediate=True,
                )

        def _record_transition_deltas(self, transitions) -> None:
            for specification, binding_key, transition in transitions:
                self.recent_transitions.append((
                    specification,
                    self.binding_labels.get(
                        (specification, binding_key), binding_key
                    ),
                    transition,
                ))
            text = "\n".join(
                "step {} · {} · {} · {} → {}".format(
                    transition.step,
                    specification,
                    label,
                    (
                        transition.from_state.value
                        if transition.from_state is not None else "—"
                    ),
                    transition.to_state.value,
                )
                for specification, label, transition
                in self.recent_transitions
            )
            self.query_one("#transitions", Static).update(
                text or "No transitions yet."
            )

        def _set_detail(self, text: str) -> None:
            self.detail_panel.set_text(text)

        def _selected_binding(self):
            return self.bindings_by_key.get(self._selected_key())

        def _selected_key(self):
            table = self.query_one("#run", DataTable)
            if table.row_count == 0:
                return None
            row_key, _column_key = table.coordinate_to_cell_key(
                table.cursor_coordinate
            )
            return str(row_key.value)

        def action_inspect(self) -> None:
            binding = self._selected_binding()
            if binding is None:
                self.action_toggle_expand()
                return
            recoveries = recovery_transitions(binding.transitions)
            unit = " " + binding.unit if binding.unit else ""
            lines = [
                "{} · {}".format(binding.specification, binding.label),
                "Current state       {}".format(binding.state.value),
                "Active population   {}".format(
                    "yes" if binding.active else "no"
                ),
                "Current robustness  {}".format(
                    "—" if binding.current_robustness is None
                    else "{:+.6g}{}".format(
                        binding.current_robustness, unit
                    )
                ),
                "Last final          {}".format(
                    "—"
                    if binding.final_robustness is None
                    else "{:+.6g}{} at step {}".format(
                        binding.final_robustness,
                        unit,
                        binding.final_step,
                    )
                ),
                "Worst robustness    {}".format(
                    "—" if binding.worst_robustness is None
                    else "{:+.6g}{} at step {}".format(
                        binding.worst_robustness,
                        unit,
                        binding.worst_step,
                    )
                ),
                "Robustness trace    {}".format(_sparkline(binding.trace)),
                "First violation     {}".format(
                    "—"
                    if binding.first_violation_step is None
                    else "step {} · t={:.3f} s".format(
                        binding.first_violation_step,
                        binding.first_violation_time,
                    )
                ),
                "Recovery count      {}".format(len(recoveries)),
                "Last recovery       {}".format(
                    "—"
                    if not recoveries
                    else "step {} · t={:.3f} s".format(
                        recoveries[-1].step,
                        recoveries[-1].simulator_time,
                    )
                ),
                "Current verdict     {}".format(
                    "final" if binding.final else "provisional"
                ),
                "Window              {}/{} samples".format(
                    binding.segment_samples
                    if binding.segment_samples is not None else "—",
                    binding.window_steps
                    if binding.window_steps is not None else "—",
                ),
                "Backend             {}".format(binding.backend or "—"),
                "",
                "Binding key: {}".format(binding.binding_key),
                "Entities: {}".format(
                    " ↔ ".join(
                        "{}_{}{}{}".format(
                            entity.entity_class,
                            entity.entity_id,
                            " · " + entity.label if entity.label else "",
                            " · role:" + entity.role if entity.role else "",
                        )
                        for entity in binding.entities
                    ) or "global"
                ),
                "",
                "Authored: {}".format(binding.authored_formula or "—"),
                "Resolved: {}".format(binding.resolved_formula or "—"),
            ]
            if binding.components:
                lines.append("")
                lines.append("Signals:")
                lines.extend(
                    "  {}  {}{}".format(
                        component.key,
                        (
                            "—" if component.value is None
                            else "{:+.6g}".format(component.value)
                        ),
                        " " + component.unit if component.unit else "",
                    )
                    for component in binding.components
                )
            if binding.error_message:
                lines.append("Error: " + binding.error_message)
            self._set_detail("\n".join(lines))

        def action_inspect_json(self) -> None:
            binding = self._selected_binding()
            if binding is None:
                return
            from .run_inspect import binding_json

            self._set_detail(binding_json(binding))

        def action_navigate_down(self) -> None:
            self.query_one("#run", DataTable).action_cursor_down()

        def action_navigate_up(self) -> None:
            self.query_one("#run", DataTable).action_cursor_up()

        def action_toggle_expand(self) -> None:
            key = self._selected_key()
            specification = self.specifications_by_key.get(key)
            if specification is None:
                return
            if specification.name in self.expanded:
                self.expanded.remove(specification.name)
            else:
                self.expanded.add(specification.name)
            if self.latest_snapshot is not None:
                self._show(self.latest_snapshot, rebuild=True)

        def action_transitions(self) -> None:
            binding = self._selected_binding()
            if binding is None:
                return
            self._set_detail("\n".join(
                "step {} · t={:.3f} s · {} → {}".format(
                    transition.step,
                    transition.simulator_time,
                    (
                        transition.from_state.value
                        if transition.from_state is not None else "—"
                    ),
                    transition.to_state.value,
                )
                for transition in binding.transitions
            ) or "No transitions.")

        def action_logs(self) -> None:
            self._set_detail(
                "Simulator messages\n{}".format(
                    _simulator_log_excerpt(simulator_log)
                )
            )

        def action_focus_search(self) -> None:
            self.query_one("#search", Input).focus()

        def on_input_changed(self, event: Input.Changed) -> None:
            self.search_text = event.value.casefold().strip()
            if self.latest_snapshot is not None:
                # No forced rebuild: narrowing only removes rows in place;
                # _apply_rows escalates itself if the order truly changes.
                self._show(self.latest_snapshot)

        def on_input_submitted(self, _event: Input.Submitted) -> None:
            self.query_one("#run", DataTable).focus()

        def action_filter(self) -> None:
            options = [
                None, "satisfied", "violated", "pending", "suspended",
                "rewarming", "missing_data", "error", "active",
                "waiting", "undetermined", "not_applicable",
            ]
            index = options.index(self.state_filter)
            self.state_filter = options[(index + 1) % len(options)]
            if self.latest_snapshot is not None:
                self._show(self.latest_snapshot, rebuild=True)
            self._set_detail(
                "Filter: {}".format(self.state_filter or "all")
            )

        def action_sort(self) -> None:
            options = ["authored", "recent", "entity"]
            index = options.index(self.sort_mode)
            self.sort_mode = options[(index + 1) % len(options)]
            if self.latest_snapshot is not None:
                self._show(self.latest_snapshot, rebuild=True)
            self._set_detail(
                "Sort: {}".format(self.sort_mode)
            )

        def action_expose(self) -> None:
            binding = self._selected_binding()
            if binding is not None:
                self._set_detail(
                    binding.resolved_formula or binding.binding_key
                )

        def action_help(self) -> None:
            self.detail_panel.show_overlay(
                "↑/↓ or j/k navigate · Space expand · Enter inspect · "
                "J JSON · T transitions · L logs · / search · f filter · "
                "s sort · y expose · Esc back · q stop"
            )

        def action_close_inspector(self) -> None:
            search = self.query_one("#search", Input)
            if search.has_focus:
                self.query_one("#run", DataTable).focus()
                return
            if self.detail_panel.dismiss_overlay():
                return
            self._set_detail(
                "Select a binding and press J for its JSON projection."
            )

        def on_data_table_row_selected(
            self, _event: DataTable.RowSelected
        ) -> None:
            self.action_inspect()

        def action_stop_run(self) -> None:
            if stop.is_set():
                # Second request escalates: leave the TUI immediately and
                # let the caller's join loop wait for (or report on) the
                # simulator worker.
                self.exit()
                return
            mark_stopped()
            try:
                self.query_one("#status", Static).update(
                    "stopping · waiting for the current simulator step "
                    "(press again to leave the dashboard now)"
                )
            except Exception:
                pass

        def _headless_stop(self) -> None:
            if not done.is_set():
                mark_stopped()
                self.exit()

    application = RunDashboard()
    if build_only:
        return application
    try:
        application.run(headless=headless)
    finally:
        if not done.is_set():
            mark_stopped()
        stop.set()
        from .run_monitor import tee_process_streams

        # Entered by at most one branch below: a run has either a driver
        # thread or a main-thread stream, never both.
        tee_context = (
            tee_process_streams(simulator_stream)
            if simulator_stream is not None else nullcontext()
        )
        driver_thread = getattr(application, "driver_thread", None)
        if driver_thread is not None:
            # A slow final simulator step is normal; never turn it into a
            # failed episode. Wait as long as it takes, telling the user
            # what is happening (the terminal is restored at this point).
            # While waiting, simulator stdout/stderr must keep reaching the
            # durable log: the App's print capture is gone, so tee the
            # process streams to the log stream for the join's duration.
            # A truly hung step is abandoned with Ctrl-C, which raises
            # WorkerAbandoned so the caller knows finalization is unsafe
            # while the worker may still be mutating the session.
            waited = 0.0
            try:
                with tee_context:
                    while driver_thread.is_alive():
                        driver_thread.join(timeout=1.0)
                        waited += 1.0
                        if driver_thread.is_alive() and waited == 5.0:
                            print(
                                "Waiting for the simulator to finish its "
                                "current step before finalizing results "
                                "(Ctrl-C to abandon)…",
                                file=sys.stderr,
                            )
            except KeyboardInterrupt:
                raise WorkerAbandoned(
                    "simulator worker abandoned mid-step; session state "
                    "must not be finalized"
                )
        close = getattr(getattr(application, "snapshot_iter", None), "close", None)
        if close is not None:
            # Main-thread stepping leaves the stream suspended between
            # steps, never mid-step; closing it runs the stream's cleanup.
            with tee_context:
                try:
                    close()
                except BaseException as exc:
                    result["error"] = exc
    if result["error"] is not None:
        raise result["error"]
    return DashboardResult(
        result["snapshot"], bool(result["stopped"])
    )
