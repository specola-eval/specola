"""
write_specs.py — spec template authoring CLI
===================================================
Discover the entities a simulator exposes, write STL specs against them
with catalogue-aware check validation, and save to a YAML file.

Usage
-----
    specola write-specs --sim mock
    specola write-specs --sim robocasa --task PickPlaceCounterToCabinet --queries examples/geometry_queries.yaml
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
import os
import secrets
import shlex
import shutil
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Tuple

import yaml

from .entities import EntityState
from .cli_common import (
    bold, dim, cyan, green, amber, red,
    load_sim,
)
from .branding import MARK as _BRAND_MARK


def _parse_constants(source: str) -> Dict[str, float]:
    """Parse the existing YAML mapping shape and reuse compiler validation."""
    if not source.strip():
        return {}
    try:
        value = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise ValueError("Invalid constants YAML: {}".format(exc)) from exc
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            "Constants must be a YAML mapping such as {threshold: 0.25}"
        )

    from .spec.semantic import validate_constants

    return validate_constants(value)


def _check_formula(
    *,
    name: str,
    formula: str,
    constants: Mapping[str, float],
    dt: float,
    catalog,
    metadata,
    entities: Dict[int, EntityState],
    labels,
    capabilities,
) -> bool:
    """Compile once through the existing per-entry checker."""
    from .check_specs import _check_entry

    entry = SimpleNamespace(
        name=name,
        formula_text=formula,
        constants=dict(constants),
        symmetric=False,
        backend="auto",
    )
    return _check_entry(
        entry,
        dt,
        catalog,
        metadata,
        entities,
        labels,
        capabilities=capabilities,
    )


def _normalise_name(name: str) -> str:
    """Apply the writer's existing saved-name normalization."""
    return name.lower().replace(" ", "_")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive spec writer
# ─────────────────────────────────────────────────────────────────────────────

def _prompt(label: str, default: str = "") -> Tuple[str, bool]:
    suffix = f" [{dim(default)}]" if default else ""
    try:
        val = input(f"  {bold(label)}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default, True
    return (val if val else default), False


def _read_constants() -> Tuple[Optional[Dict[str, float]], bool]:
    while True:
        source, interrupted = _prompt(
            "Constants (YAML mapping)",
            default="{}",
        )
        if interrupted:
            return None, True
        try:
            return _parse_constants(source), False
        except (TypeError, ValueError) as exc:
            print(f"  {red('✗')} {exc}")
            print(dim("  Enter a YAML mapping, for example {limit: 1.0}."))


def _interactive_spec_writer(
    *,
    dt: float,
    catalog,
    metadata,
    entities: Dict[int, EntityState],
    labels,
    capabilities,
) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    print(dim("  Blank name → done   |   Blank formula → skip spec"))
    print()

    while True:
        name, interrupted = _prompt("Spec name")
        if interrupted or not name:
            break
        name = _normalise_name(name)
        if any(spec["name"] == name for spec in specs):
            print(
                f"  {red('✗')} Spec name {name!r} is already used. "
                "Choose a unique name."
            )
            continue

        while True:
            formula, interrupted = _prompt("Formula")
            if interrupted:
                return specs
            if not formula:
                break

            constants, interrupted = _read_constants()
            if interrupted:
                return specs
            assert constants is not None

            accepted = _check_formula(
                name=name,
                formula=formula,
                constants=constants,
                dt=dt,
                catalog=catalog,
                metadata=metadata,
                entities=entities,
                labels=labels,
                capabilities=capabilities,
            )
            if accepted:
                keep, interrupted = _prompt(
                    "Keep this spec? [y/n]",
                    default="y",
                )
                if interrupted:
                    return specs
                if keep.strip().lower() in ("y", "yes", ""):
                    spec: Dict[str, object] = {
                        "name": name,
                        "formula": formula,
                    }
                    if constants:
                        spec["constants"] = constants
                    specs.append(spec)
                    break
                print(dim(
                    "  Retype the formula, or press Enter to skip this spec."
                ))
                continue

            print(dim(
                "  Try the formula again, or press Enter to skip this spec."
            ))
        print()

    return specs


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def _save_specs(
    specs: List[Dict[str, object]],
    suggested: str,
) -> Optional[str]:
    if not specs:
        print(dim("  No specs to save."))
        return None

    # Ask for the filename
    path, interrupted = _prompt("Save to", default=suggested)
    if interrupted or not path:
        print(dim("  Skipping save."))
        return None
    path = _yaml_path(path)

    overwrite = False

    # Handle existing file
    if Path(path).exists():
        print(f"  {amber('!')} {bold(path)} already exists.")
        choice, _ = _prompt("  [o]verwrite  [r]ename  [s]kip", default="r")
        choice = choice.strip().lower()

        if choice in ("s", "skip", ""):
            print(dim("  Skipping save."))
            return None

        if choice in ("r", "rename"):
            new_path, interrupted = _prompt(
                "New filename",
                default=_available_renamed_path(path),
            )
            if interrupted or not new_path:
                print(dim("  Skipping save."))
                return None
            new_path = _yaml_path(new_path)
            # If the renamed path also exists, keep asking
            while Path(new_path).exists():
                print(f"  {amber('!')} {bold(new_path)} also exists.")
                new_path, interrupted = _prompt(
                    "Try another name",
                    default=_available_renamed_path(new_path),
                )
                if interrupted or not new_path:
                    print(dim("  Skipping save."))
                    return None
                new_path = _yaml_path(new_path)
            path = new_path

        elif choice not in ("o", "overwrite"):
            print(dim("  Unrecognised choice — skipping save."))
            return None
        else:
            overwrite = True

    while True:
        try:
            _write_specs_document(specs, path, overwrite=overwrite)
        except FileExistsError:
            print(
                "  {} {} appeared before save.".format(
                    amber("!"), bold(path)
                )
            )
            choice, _ = _prompt(
                "  [o]verwrite  [r]ename  [s]kip",
                default="r",
            )
            choice = choice.strip().lower()
            if choice in ("s", "skip", ""):
                print(dim("  Skipping save."))
                return None
            if choice in ("o", "overwrite"):
                overwrite = True
                continue
            if choice not in ("r", "rename"):
                print(dim("  Unrecognised choice — skipping save."))
                return None
            path, interrupted = _prompt(
                "New filename",
                default=_available_renamed_path(path),
            )
            if interrupted or not path:
                print(dim("  Skipping save."))
                return None
            path = _yaml_path(path)
            overwrite = False
            continue
        except (OSError, yaml.YAMLError) as exc:
            print("  {} Could not save {}: {}".format(
                red("✗"), bold(path), exc
            ))
            path, interrupted = _prompt(
                "Try another filename",
                default=path,
            )
            if interrupted or not path:
                print(dim("  Skipping save."))
                return None
            path = _yaml_path(path)
            overwrite = False
            continue
        print(f"  {green('Saved →')} {bold(path)}")
        return path


def _yaml_path(path: str) -> str:
    """Return ``path`` with an existing Specola YAML extension."""
    if path.endswith(".yaml") or path.endswith(".yml"):
        return path
    return path + ".yaml"


def _write_specs_document(
    specs: List[Dict[str, object]],
    path: str,
    *,
    overwrite: bool = True,
) -> None:
    """Atomically publish the existing specification document shape."""
    document = yaml.safe_dump(
        {"specs": specs},
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    destination = Path(path)
    temporary_path: Optional[str] = None
    descriptor: Optional[int] = None
    try:
        while descriptor is None:
            temporary_path = str(destination.with_name(
                ".specola-{}.tmp".format(secrets.token_hex(8))
            ))
            try:
                descriptor = os.open(
                    temporary_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o666,
                )
            except FileExistsError:
                continue
        assert temporary_path is not None
        if overwrite and destination.exists():
            shutil.copystat(path, temporary_path)
        with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
            descriptor = None
            fh.write(document)
            fh.flush()
            os.fsync(fh.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _available_renamed_path(path: str) -> str:
    """Return the first numbered sibling path that does not exist."""
    source = Path(path)
    suffix = source.suffix
    stem = source.stem
    index = 2
    while True:
        candidate = source.with_name(
            "{}_{}{}".format(stem, index, suffix)
        )
        if not candidate.exists():
            return str(candidate)
        index += 1


def _adapter_arguments(args) -> List[str]:
    common = ["--sim", args.sim]
    for module_name in args.adapter_module:
        common.extend(("--adapter-module", module_name))
    if args.task:
        common.extend(("--task", args.task))
    if args.robot:
        common.extend(("--robot", args.robot))
    if args.all_regions:
        common.append("--all-regions")
    if args.dt is not None:
        common.extend(("--dt", str(args.dt)))
    return common


def _followup_commands(args, saved: str) -> Tuple[str, str]:
    common = _adapter_arguments(args)

    check = [
        "specola", "check", *common, "--no-pager", saved,
    ]
    run = [
        "specola", "run", *common, "--specs", saved,
    ]
    return shlex.join(check), shlex.join(run)


def _discovery_command(args, model) -> str:
    common = _adapter_arguments(args)
    entity_class = next(
        (
            entity.entity_class
            for entity in model.entities
            if not entity.raw
        ),
        None,
    )
    detail = (
        ["--class", entity_class, "--verbose"]
        if entity_class is not None
        else ["--raw", "--verbose"]
    )
    return shlex.join(["specola", "discover", *common, *detail])


def build_write_specs_app(
    view,
    *,
    dt: float,
    catalog,
    metadata,
    entities: Dict[int, EntityState],
    labels,
    capabilities,
    suggested_output: str,
):
    """Build the Textual discovery workspace with persistent authoring."""
    try:
        from rich.text import Text
        from textual.app import ComposeResult
        from textual.containers import Horizontal, VerticalScroll
        from textual.events import Resize
        from textual.screen import ModalScreen
        from textual.widgets import (
            Button,
            Footer,
            Header,
            Input,
            Static,
            TextArea,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Textual write-specs requires the bounded UI dependencies"
        ) from exc

    from .explore import discovery_explorer_class
    from .selector_palette import verify_forms

    DiscoveryExplorer = discovery_explorer_class(view)

    class SavePathScreen(ModalScreen):
        CSS = """
        SavePathScreen {
            align: center middle;
            background: $background 70%;
        }
        #save_path_dialog {
            width: 72;
            height: 11;
            padding: 1 2;
            border: round $primary;
            background: $surface;
        }
        #save_path_actions { height: 3; }
        """

        def __init__(self, initial: str):
            super().__init__()
            self.initial = initial

        def compose(self) -> ComposeResult:
            with VerticalScroll(id="save_path_dialog"):
                yield Static("Save kept specifications")
                yield Input(
                    value=self.initial,
                    placeholder="specs.yaml",
                    id="save_path",
                )
                with Horizontal(id="save_path_actions"):
                    yield Button(
                        "Save",
                        id="save_path_confirm",
                        variant="primary",
                    )
                    yield Button("Cancel", id="save_path_cancel")

        def on_mount(self) -> None:
            self.query_one("#save_path", Input).focus()

        def on_input_submitted(
            self,
            event: Input.Submitted,
        ) -> None:
            event.stop()
            self.dismiss(event.value)

        def on_button_pressed(
            self,
            event: Button.Pressed,
        ) -> None:
            event.stop()
            if event.button.id == "save_path_confirm":
                self.dismiss(
                    self.query_one("#save_path", Input).value
                )
            elif event.button.id == "save_path_cancel":
                self.dismiss(None)

    class CollisionScreen(ModalScreen):
        CSS = """
        CollisionScreen {
            align: center middle;
            background: $background 70%;
        }
        #collision_dialog {
            width: 76;
            height: 13;
            padding: 1 2;
            border: round $warning;
            background: $surface;
        }
        #collision_actions { height: 3; }
        """

        def __init__(self, path: str):
            super().__init__()
            self.path = path

        def compose(self) -> ComposeResult:
            with VerticalScroll(id="collision_dialog"):
                yield Static(
                    "{} already exists.\nRename uses {}.".format(
                        self.path,
                        _available_renamed_path(self.path),
                    ),
                    markup=False,
                )
                with Horizontal(id="collision_actions"):
                    yield Button(
                        "Rename",
                        id="collision_rename",
                        variant="primary",
                    )
                    yield Button(
                        "Overwrite",
                        id="collision_overwrite",
                        variant="warning",
                    )
                    yield Button("Skip", id="collision_skip")
                    yield Button("Cancel", id="collision_cancel")

        def on_button_pressed(
            self,
            event: Button.Pressed,
        ) -> None:
            event.stop()
            choices = {
                "collision_rename": "rename",
                "collision_overwrite": "overwrite",
                "collision_skip": "skip",
                "collision_cancel": None,
            }
            if event.button.id in choices:
                self.dismiss(choices[event.button.id])

    class UnsavedScreen(ModalScreen):
        CSS = """
        UnsavedScreen {
            align: center middle;
            background: $background 70%;
        }
        #unsaved_dialog {
            width: 72;
            height: 11;
            padding: 1 2;
            border: round $warning;
            background: $surface;
        }
        #unsaved_actions { height: 3; }
        """

        def __init__(self, kept_count: int, has_draft: bool):
            super().__init__()
            self.kept_count = kept_count
            self.has_draft = has_draft

        def compose(self) -> ComposeResult:
            detail = []
            if self.kept_count:
                detail.append(
                    "{} kept specification{} not saved".format(
                        self.kept_count,
                        "" if self.kept_count == 1 else "s",
                    )
                )
            if self.has_draft:
                detail.append("current draft not kept")
            with VerticalScroll(id="unsaved_dialog"):
                yield Static(
                    "Unsaved work\n" + " · ".join(detail),
                    markup=False,
                )
                with Horizontal(id="unsaved_actions"):
                    if self.kept_count:
                        yield Button(
                            "Save file…",
                            id="unsaved_save",
                            variant="primary",
                        )
                    yield Button("Return to writer", id="unsaved_return")
                    yield Button(
                        "Discard and quit",
                        id="unsaved_discard",
                        variant="error",
                    )

        def on_button_pressed(
            self,
            event: Button.Pressed,
        ) -> None:
            event.stop()
            choices = {
                "unsaved_save": "save",
                "unsaved_return": "return",
                "unsaved_discard": "discard",
            }
            if event.button.id in choices:
                self.dismiss(choices[event.button.id])

    class WriteSpecsApp(DiscoveryExplorer):
        TITLE = "Specola Write Specs"
        MINIMUM_SIZE = (80, 30)
        BINDINGS = [
            binding
            for binding in DiscoveryExplorer.BINDINGS
            if binding[1] != "quit"
        ] + [
            ("q", "request_exit", "Exit"),
            ("ctrl+c", "request_exit", "Exit"),
            ("i", "insert_selector", "Insert"),
        ]
        CSS = DiscoveryExplorer.CSS + """
        #writer_workspace { height: 1fr; }
        #size_warning {
            display: none;
            height: 1fr;
            content-align: center middle;
            text-align: center;
            border: round $warning;
        }
        #draft_title, #kept_specs, #save_status { padding: 0 1; }
        #writer_workspace #browser_row {
            height: 9;
            min-height: 9;
        }
        #writer_workspace #selected_signal { height: 8; }
        #writer_workspace #palette { height: 5; }
        #writer_workspace #authoring { height: 5; }
        #spec_name, #constants { height: 3; }
        #formula { height: 4; border: round $primary; }
        #selector_actions, #draft_actions { height: 3; }
        #validation_scroll {
            height: 10;
            border: round $primary;
            padding: 0 1;
        }
        #validation { height: auto; }
        #insert_selector, #copy_selector { width: 1fr; }
        #validate, #keep, #clear, #save { width: 1fr; }
        """

        def __init__(self):
            super().__init__()
            self.specs: List[Dict[str, object]] = []
            self.saved_path: Optional[str] = None
            self.kept_dirty = False
            self.draft_revision = 0
            self.validated_revision: Optional[int] = None
            self.validation_ok = False
            self._validated_entry: Optional[
                Dict[str, object]
            ] = None
            self._last_report = Text("Not validated.")
            self._programmatic_edit = False
            self._pending_save_path: Optional[str] = None
            self._exit_after_save = False
            self.last_save_error: Optional[str] = None

        def compose(self) -> ComposeResult:
            yield Header(icon=_BRAND_MARK)
            yield Static(
                "Write Specs requires at least 80 × 30.\n"
                "Resize the terminal or rerun with --ui plain.",
                id="size_warning",
                markup=False,
            )
            with VerticalScroll(id="writer_workspace"):
                yield from self.compose_discovery()
                with Horizontal(id="selector_actions"):
                    yield Button(
                        "Insert selector",
                        id="insert_selector",
                    )
                    yield Button(
                        "Copy selector",
                        id="copy_selector",
                    )
                yield Static("Draft", id="draft_title")
                yield Input(
                    placeholder="Specification name",
                    id="spec_name",
                )
                yield TextArea("", id="formula")
                yield Input(
                    value="{}",
                    placeholder="Constants YAML mapping",
                    id="constants",
                )
                with Horizontal(id="draft_actions"):
                    yield Button(
                        "Validate",
                        id="validate",
                        variant="primary",
                    )
                    yield Button("Keep spec", id="keep", disabled=True)
                    yield Button("Clear draft", id="clear")
                    yield Button("Save file…", id="save", disabled=True)
                yield Static(
                    "Kept specs: 0",
                    id="kept_specs",
                    markup=False,
                )
                yield Static("", id="save_status", markup=False)
                with VerticalScroll(id="validation_scroll"):
                    yield Static(
                        self._last_report,
                        id="validation",
                        markup=False,
                    )
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#spec_name").border_title = "Name"
            self.query_one("#formula").border_title = "Formula"
            self.query_one("#constants").border_title = (
                "Constants (YAML)"
            )
            self.query_one("#validation_scroll").border_title = (
                "Validation — current draft"
            )
            self._apply_size(
                self.screen.size.width,
                self.screen.size.height,
            )

        def _apply_size(self, width: int, height: int) -> None:
            too_small = (
                width < self.MINIMUM_SIZE[0]
                or height < self.MINIMUM_SIZE[1]
            )
            self.query_one("#size_warning").display = too_small
            self.query_one("#writer_workspace").display = not too_small

        def on_resize(self, event: Resize) -> None:
            self._apply_size(event.size.width, event.size.height)

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id == "search":
                return
            if event.input.id in ("spec_name", "constants"):
                self._draft_edited()

        def on_input_submitted(
            self,
            event: Input.Submitted,
        ) -> None:
            if event.input.id == "search":
                return

        def on_text_area_changed(
            self,
            event: TextArea.Changed,
        ) -> None:
            if event.text_area.id == "formula":
                self._draft_edited()

        def _draft_edited(self) -> None:
            if self._programmatic_edit:
                return
            # The STALE banner only needs one paint, on the first edit after
            # a validation; later keystrokes would rewrite identical content.
            already_stale = (
                self.validated_revision is not None
                and self.draft_revision != self.validated_revision
            )
            self.draft_revision += 1
            self.validation_ok = False
            self.query_one("#keep", Button).disabled = True
            if self.validated_revision is not None and not already_stale:
                stale = Text(
                    "Validation — STALE\n"
                    "Draft edited since the last validation.\n\n"
                )
                stale.append_text(self._last_report.copy())
                self.query_one("#validation", Static).update(
                    stale
                )

        def _formula(self) -> str:
            return self.query_one("#formula", TextArea).text.strip()

        def _has_draft(self) -> bool:
            return bool(
                self.query_one("#spec_name", Input).value.strip()
                or self._formula()
                or self.query_one("#constants", Input).value.strip()
                not in ("", "{}")
            )

        def _resolve_selected_signal(self) -> None:
            """Fill the palette, then check all of it against the scene.

            The explorer can only resolve a form's bracket against the
            reset snapshot; the writer holds a catalog and is where a
            form becomes a specification, so it checks that each one is
            legal. On RoboCasa's contact class that rejects ten of
            sixteen shapes which resolve cleanly -- precisely the gap
            bracket-level verification cannot see, and precisely the
            rows this surface would otherwise offer a starter for.

            The whole palette, not the focused row. Verifying one row at
            a time was right while a single form cost seconds to
            compile; expansion-only checking made a whole palette 0.030 s
            on `contact` and 1.94 s on the widest class, `fixture`. It
            also meant an unchecked row looked exactly like a checked
            one, and the per-reference cache it needed was corrupted by
            two forms legitimately sharing a reference.
            """
            super()._resolve_selected_signal()
            if not self.palette_forms:
                return
            verified = verify_forms(
                self.palette_forms,
                catalog=catalog,
                dt=dt,
                capabilities=capabilities,
            )
            replacement = dict(zip(
                (form.reference for form in self.palette_forms), verified
            ))
            self.palette_forms = verified
            self.form_by_key = {
                key: replacement.get(form.reference, form)
                for key, form in self.form_by_key.items()
            }
            if self.selected_form is not None:
                self.selected_form = replacement.get(
                    self.selected_form.reference, self.selected_form
                )
            self._repaint_palette()
            if self.selected_form is not None:
                super()._show_form(self.selected_form)

        def _repaint_palette(self) -> None:
            """Redraw the rows so a refusal is visible, not just stored."""
            # No type argument: DataTable is not among the widgets this
            # module imports, and the base class already owns the table's
            # column handling.
            table = self.query_one("#palette")
            self._rebuilding = True
            try:
                self._reset_table(table, *self._PALETTE_COLUMNS)
                for key, form in self.form_by_key.items():
                    text = Text(form.reference)
                    if form.partner_reference:
                        text.append(
                            "  \u2194  " + form.partner_reference, style="dim"
                        )
                    if not form.available:
                        text.stylize("dim")
                    table.add_row(text, key=key)
            finally:
                self._rebuilding = False

        def _ensure_form(self):
            """The highlighted palette row, resolving the palette first.

            The palette is filled on a debounce, so a keyboard user who
            selects a signal and immediately presses ``i`` can arrive
            before it exists; resolving synchronously here costs the same
            work the timer was about to do.
            """
            if (
                self.selected_form is None
                and self.selected_signal is not None
                and self.selected_scope is not None
            ):
                self._resolve_selected_signal()
            return self.selected_form

        def _usable_form(self):
            form = self._ensure_form()
            if form is None or not form.available:
                self.notify(
                    "Highlight an available selector form first."
                    if form is None
                    else form.unavailable_reason,
                    severity="warning",
                    markup=False,
                )
                return None
            return form

        def _insert_selector(self) -> None:
            form = self._usable_form()
            if form is None:
                return
            formula = self.query_one("#formula", TextArea)
            formula.insert(form.reference)
            formula.focus()
            if form.partner_reference:
                self.notify(
                    "Two-slot form: insert {} as well, and set "
                    "symmetric: true when the expression is "
                    "order-independent.".format(form.partner_reference),
                    markup=False,
                )

        def action_insert_selector(self) -> None:
            if (
                self.query_one("#signals").has_focus
                or self.query_one("#palette").has_focus
            ):
                self._insert_selector()

        def _copy_selector(self) -> None:
            form = self._usable_form()
            if form is None:
                return
            self.copy_to_clipboard(form.reference)
            self.notify("Selector copied.", markup=False)

        def _validate_draft(self) -> None:
            raw_name = self.query_one("#spec_name", Input).value.strip()
            name = _normalise_name(raw_name)
            formula = self._formula()
            constants_source = self.query_one(
                "#constants", Input
            ).value

            if not name:
                self._validation_failure(
                    "Specification name is required."
                )
                return
            if any(spec["name"] == name for spec in self.specs):
                self._validation_failure(
                    "Spec name {!r} is already used.".format(name)
                )
                return
            if not formula:
                self._validation_failure("Formula is required.")
                return
            try:
                constants = _parse_constants(constants_source)
            except (TypeError, ValueError) as exc:
                self._validation_failure(str(exc))
                return

            output = StringIO()
            with redirect_stdout(output):
                accepted = _check_formula(
                    name=name,
                    formula=formula,
                    constants=constants,
                    dt=dt,
                    catalog=catalog,
                    metadata=metadata,
                    entities=entities,
                    labels=labels,
                    capabilities=capabilities,
                )
            body = Text.from_ansi(output.getvalue())
            header = Text(
                "✓ Current draft accepted\n"
                if accepted
                else "✗ Current draft rejected\n"
            )
            header.append_text(body)
            self._last_report = header
            self.query_one("#validation", Static).update(
                header
            )
            self.validated_revision = self.draft_revision
            self.validation_ok = accepted
            self.query_one("#keep", Button).disabled = not accepted
            validation_scroll = self.query_one(
                "#validation_scroll", VerticalScroll
            )
            validation_scroll.scroll_home(animate=False)
            self.query_one(
                "#writer_workspace", VerticalScroll
            ).scroll_to_widget(
                validation_scroll,
                animate=False,
                force=True,
            )
            if accepted:
                entry: Dict[str, object] = {
                    "name": name,
                    "formula": formula,
                }
                if constants:
                    entry["constants"] = constants
                self._validated_entry = entry
            else:
                self._validated_entry = None

        def _validation_failure(self, message: str) -> None:
            report = Text("✗ Current draft rejected\n" + message)
            self._last_report = report
            self.query_one("#validation", Static).update(
                report
            )
            self.validated_revision = self.draft_revision
            self.validation_ok = False
            self._validated_entry = None
            self.query_one("#keep", Button).disabled = True
            validation_scroll = self.query_one(
                "#validation_scroll", VerticalScroll
            )
            validation_scroll.scroll_home(animate=False)
            self.query_one(
                "#writer_workspace", VerticalScroll
            ).scroll_to_widget(
                validation_scroll,
                animate=False,
                force=True,
            )

        def _keep_spec(self) -> None:
            if (
                not self.validation_ok
                or self.validated_revision != self.draft_revision
                or self._validated_entry is None
            ):
                return
            self.specs.append(dict(self._validated_entry))
            self.kept_dirty = True
            self.query_one("#save", Button).disabled = False
            self._refresh_kept()
            self._clear_draft()

        def _refresh_kept(self) -> None:
            lines = ["Kept specs: {}".format(len(self.specs))]
            lines.extend(
                "{} ✓ {}".format(index, spec["name"])
                for index, spec in enumerate(self.specs, start=1)
            )
            self.query_one("#kept_specs", Static).update(
                "\n".join(lines)
            )

        def _clear_draft(self) -> None:
            self._programmatic_edit = True
            try:
                self.query_one("#spec_name", Input).value = ""
                self.query_one("#formula", TextArea).load_text("")
                self.query_one("#constants", Input).value = "{}"
            finally:
                self._programmatic_edit = False
            self.draft_revision += 1
            self.validated_revision = None
            self.validation_ok = False
            self._validated_entry = None
            self._last_report = Text("Not validated.")
            self.query_one("#validation", Static).update(
                self._last_report
            )
            self.query_one("#keep", Button).disabled = True

        def _open_save(self, *, exit_after: bool = False) -> None:
            if not self.specs:
                self.notify(
                    "Keep at least one validated specification first.",
                    severity="warning",
                )
                return
            self._exit_after_save = exit_after
            initial = self.saved_path or suggested_output
            self.push_screen(
                SavePathScreen(initial),
                self._save_path_chosen,
            )

        def _save_path_chosen(self, value) -> None:
            if value is None or not str(value).strip():
                self._exit_after_save = False
                return
            path = _yaml_path(str(value).strip())
            if Path(path).exists():
                self._pending_save_path = path
                self.push_screen(
                    CollisionScreen(path),
                    self._collision_chosen,
                )
                return
            self._complete_save(path, overwrite=False)

        def _collision_chosen(self, choice) -> None:
            path = self._pending_save_path
            self._pending_save_path = None
            if path is None or choice in (None, "skip"):
                self._exit_after_save = False
                return
            if choice == "rename":
                path = _available_renamed_path(path)
                overwrite = False
            elif choice != "overwrite":
                self._exit_after_save = False
                return
            else:
                overwrite = True
            self._complete_save(path, overwrite=overwrite)

        def _complete_save(
            self,
            path: str,
            *,
            overwrite: bool = False,
        ) -> None:
            try:
                _write_specs_document(
                    self.specs,
                    path,
                    overwrite=overwrite,
                )
            except FileExistsError:
                self._pending_save_path = path
                self.query_one("#save_status", Static).update(
                    "{} appeared before save; choose rename or "
                    "explicit overwrite.".format(path)
                )
                self.push_screen(
                    CollisionScreen(path),
                    self._collision_chosen,
                )
                return
            except (OSError, yaml.YAMLError) as exc:
                message = "Could not save {}: {}".format(path, exc)
                self.last_save_error = message
                self.query_one("#save_status", Static).update(
                    message
                )
                self.notify(
                    message,
                    severity="error",
                    markup=False,
                )
                self._exit_after_save = False
                return
            self.saved_path = path
            self.kept_dirty = False
            self.last_save_error = None
            self.query_one("#save_status", Static).update(
                "Saved {}".format(path)
            )
            self.notify("Saved {}".format(path), markup=False)
            if self._exit_after_save and not self._has_draft():
                self.exit()
            self._exit_after_save = False

        def action_request_exit(self) -> None:
            if self.kept_dirty or self._has_draft():
                self.push_screen(
                    UnsavedScreen(
                        len(self.specs) if self.kept_dirty else 0,
                        self._has_draft(),
                    ),
                    self._unsaved_chosen,
                )
                return
            self.exit()

        def _unsaved_chosen(self, choice) -> None:
            if choice == "discard":
                self.exit()
            elif choice == "save":
                self._open_save(exit_after=True)

        def on_button_pressed(
            self,
            event: Button.Pressed,
        ) -> None:
            button_id = event.button.id
            if button_id == "insert_selector":
                self._insert_selector()
            elif button_id == "copy_selector":
                self._copy_selector()
            elif button_id == "validate":
                self._validate_draft()
            elif button_id == "keep":
                self._keep_spec()
            elif button_id == "clear":
                self._clear_draft()
            elif button_id == "save":
                self._open_save()

    return WriteSpecsApp()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="specola write-specs",
        description=(
            "Browse the discovery workspace while authoring and validating "
            "specifications."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
          specola write-specs --sim mock
          specola write-specs --sim robocasa --task PickPlaceCounterToCabinet --queries examples/geometry_queries.yaml
          ROBOCASA_TASK=PickPlaceCounterToCabinet specola write-specs --sim robocasa --queries examples/geometry_queries.yaml
        """),
    )
    ap.add_argument("--sim",     default="mock")
    ap.add_argument("--adapter-module", action="append", default=[],
                    metavar="MODULE",
                    help="Import MODULE so its adapter registration runs "
                         "(repeatable); use with --sim <registered-name>")
    ap.add_argument("--task",    default=None)
    ap.add_argument("--robot",   default=None)
    ap.add_argument("--output",  default=None,
                    help="Set the default at the interactive Save to prompt "
                         "(default: <sim>_specs.yaml).")
    ap.add_argument("--all-regions", action="store_true",
                    help="Discover regions from every fixture, not just "
                         "the task's target regions (adapters without "
                         "the distinction ignore this)")
    ap.add_argument("--verbose", action="store_true",
                    help="Deprecated compatibility flag for plain mode; "
                         "Textual exposes discovery detail interactively")
    ap.add_argument(
        "--ui",
        choices=("textual", "plain"),
        default=None,
        help="Authoring UI; defaults to Textual on a TTY and plain otherwise",
    )
    ap.add_argument("--dt",      type=float, default=None)
    ap.add_argument("--queries", type=Path, default=None, metavar="PATH",
                    help="Native query configuration YAML or JSON, separate from "
                         "the specification (required by the MuJoCo-family adapters)")
    return ap


def _is_tty() -> bool:
    return bool(
        getattr(sys.stdin, "isatty", lambda: False)()
        and getattr(sys.stdout, "isatty", lambda: False)()
    )


def _select_ui(requested: Optional[str]) -> str:
    """Resolve automatic UI selection before simulator startup."""
    interactive = _is_tty()
    if requested == "textual" and not interactive:
        raise SystemExit(
            "--ui textual requires interactive stdin and stdout; "
            "use --ui plain for redirected input or output"
        )
    return requested or ("textual" if interactive else "plain")


def main():
    args = _parser().parse_args()
    ui = _select_ui(args.ui)

    from .registry import load_adapter_module
    for module_name in args.adapter_module:
        load_adapter_module(module_name)

    from .query_config import load_query_configuration
    queries = load_query_configuration(args.queries) if args.queries is not None else None
    adapter, dt, _dims, ents, _eclasses = load_sim(
        args.sim, task=args.task, robot=args.robot,
        target_only=not args.all_regions, dt_override=args.dt,
        header="STL Spec Authoring", queries=queries,
    )

    from .cli_common import _task_description
    from .components import ComponentCatalog
    from .discovery_filter import DiscoveryQuery, filter_discovery
    from .discovery_model import (
        build_discovery_model,
        property_metadata_for,
    )
    from .discovery_render import render_discovery

    metadata = property_metadata_for(adapter, ents)
    model = build_discovery_model(
        adapter,
        ents,
        simulator=args.sim,
        task=args.task,
        effective_dt=dt,
        task_description=_task_description(adapter),
        property_metadata=metadata,
        extraction=adapter.last_observation.extraction if adapter.last_observation is not None else None,
    )
    view = filter_discovery(
        model,
        DiscoveryQuery(include_raw=False),
    )
    catalog = ComponentCatalog.from_entities(ents, metadata=metadata)
    labels = getattr(adapter, "entity_labels", None)
    capabilities = getattr(adapter, "capabilities", None)
    suggested = args.output or f"{args.sim}_specs.yaml"

    if ui == "textual":
        app = build_write_specs_app(
            view,
            dt=dt,
            catalog=catalog,
            metadata=metadata,
            entities=ents,
            labels=labels,
            capabilities=capabilities,
            suggested_output=suggested,
        )
        app.run()
        saved = app.saved_path
        if saved:
            check_command, run_command = _followup_commands(
                args, saved
            )
            print(f"  {green('Saved →')} {bold(saved)}")
            print()
            print(dim("  Check the saved file with:"))
            print(f"  {cyan(check_command)}")
            print()
            print(dim("  Run evaluation with:"))
            print(f"  {cyan(run_command)}")
            print()
        return

    print()
    print(
        render_discovery(view, include_cli_guidance=False),
        end="",
    )
    discovery_command = _discovery_command(args, model)
    print()
    print(dim("  Explore this adapter in detail with:"))
    print(f"  {cyan(discovery_command)}")
    print()
    if args.verbose:
        print(amber(
            "  --verbose no longer expands write-specs output. "
            "Run the filtered discovery command above for detailed values."
        ))
        print()

    print(bold("  Write your specs"))
    print(dim(
        "  Compose STL with the discovered references above; "
        "resolved keys and units are checked before acceptance."
    ))
    print()
    specs = _interactive_spec_writer(
        dt=dt,
        catalog=catalog,
        metadata=metadata,
        entities=ents,
        labels=labels,
        capabilities=capabilities,
    )

    if not specs:
        print(dim("  No specs written."))
        print(dim("  Tip: write at least one spec, then evaluate with specola run."))
        print()
        return

    print(bold("  Specs to save:"))
    for s in specs:
        print(f"  {green(s['name'])}")
        print(f"    {dim(s['formula'])}")
    print()

    saved = _save_specs(specs, suggested=suggested)
    if saved:
        check_command, run_command = _followup_commands(args, saved)
        print()
        print(dim("  Check the saved file with:"))
        print(f"  {cyan(check_command)}")
        print()
        print(dim("  Run evaluation with:"))
        print(f"  {cyan(run_command)}")
    print()


if __name__ == "__main__":
    main()
