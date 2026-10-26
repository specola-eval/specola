"""
registry.py — SimulatorAdapter registry and discovery
=====================================================
The single mechanism through which adapters become available to the CLI
and to library users. Adding a new simulator requires only a new adapter
module plus registration — no shared registry, CLI, parser, monitor, or
runner code changes.

Discovery paths
---------------
1. Direct registration. An adapter module calls ``register()`` at module
   level, so importing the module makes the adapter available:

       from specola.registry import register

       def _make(task, robot, target_only):
           ...
           return MyAdapter(env, ...)

       register(
           name        = "mysim",
           factory     = _make,
           description = "My simulator (MyLib >= 1.0)",
           install_hint= "pip install mysimlib",
           requires    = ("mysimlib",),
       )

   ``requires`` names the top-level packages the simulator needs.
   ``specola doctor`` checks that they can be located, without importing
   them, to report which adapters find their simulator packages.

2. Entry points. An installed distribution declares the adapter module
   under the ``specola.adapters`` entry-point group; ``resolve()``
   imports it on first use:

       [project.entry-points."specola.adapters"]
       mysim = "my_package.my_adapter"

   The entry-point value is the module whose import performs the
   ``register()`` call above. specola's own bundled adapters are
   declared exactly this way in its pyproject.toml.

3. Explicit module loading (development). ``--adapter-module
   my_package.my_adapter`` on the CLIs calls ``load_adapter_module()``,
   which imports the module so its registration runs.

Duplicate names are rejected unless ``replace=True`` is passed.

Factory signature
-----------------
    factory(task: Optional[str], robot: Optional[str], target_only: bool)
        -> SimulatorAdapter

    task / robot interpretation is adapter-specific:
      - RoboCasa: task = robosuite task name, robot = robot model name
      - LIBERO:   task = "suite:task_id" e.g. "libero_object:0"
      - mock:     both ignored

    A factory may also declare ``capture_frames: bool = True``. ``specola
    run`` passes ``capture_frames=False`` when it records no camera frame
    and opens no viewer (neither ``--web`` nor ``--render``), so the factory
    can skip building a renderer. A factory without the parameter never
    receives it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
import importlib.util
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .adapter import SimulatorAdapter


ENTRY_POINT_GROUP = "specola.adapters"


@dataclass
class _AdapterEntry:
    name:         str
    factory:      Callable[[Optional[str], Optional[str], bool], SimulatorAdapter]
    description:  str
    install_hint: str = ""
    handles:      Optional[Callable[[Any], bool]] = None
    wrap:         Optional[Callable[..., SimulatorAdapter]] = None
    requires:     Tuple[str, ...] = ()


_REGISTRY: Dict[str, _AdapterEntry] = {}


def register(
    name:         str,
    factory:      Callable[[Optional[str], Optional[str], bool], SimulatorAdapter],
    description:  str = "",
    install_hint: str = "",
    *,
    replace:      bool = False,
    handles:      Optional[Callable[[Any], bool]] = None,
    wrap:         Optional[Callable[..., SimulatorAdapter]] = None,
    requires:     Tuple[str, ...] = (),
) -> None:
    """Register a simulator adapter under the given name.

    Call this at module level in your adapter file so it self-registers
    on import. The name is what users pass to --sim.

    Raises
    ------
    ValueError
        If the name is already registered and ``replace`` is not True.
    """
    if not replace and name in _REGISTRY:
        raise ValueError(
            f"Adapter name {name!r} is already registered. Pass "
            f"replace=True to replace it deliberately."
        )

    _REGISTRY[name] = _AdapterEntry(
        name=name,
        factory=factory,
        description=description,
        install_hint=install_hint,
        handles=handles,
        wrap=wrap,
        requires=tuple(requires),
    )


def get(name: str) -> Optional[_AdapterEntry]:
    """Return an already-registered entry, without triggering discovery."""
    return _REGISTRY.get(name)


def load_adapter_module(module_name: str) -> None:
    """Import an adapter module so its module-level registration runs.

    This backs the ``--adapter-module`` CLI option and is also useful in
    tests and notebooks.
    """
    importlib.import_module(module_name)


def _iter_adapter_entry_points() -> Iterable:
    """Yield entry points in the ``specola.adapters`` group.

    Compatible with Python 3.8 (dict-style importlib.metadata API) through
    3.12+ (``EntryPoints.select``).
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # Python < 3.8
        from importlib_metadata import entry_points  # type: ignore

    all_entry_points = entry_points()

    if hasattr(all_entry_points, "select"):
        return all_entry_points.select(group=ENTRY_POINT_GROUP)

    return all_entry_points.get(ENTRY_POINT_GROUP, [])


def resolve(name: str) -> _AdapterEntry:
    """Return the adapter entry for a name, discovering it if necessary.

    Lookup order:

    1. adapters already registered in this process;
    2. installed ``specola.adapters`` entry points, whose modules are
       imported on demand so their registration runs.

    Raises
    ------
    ValueError
        If no adapter with this name can be found. The message lists every
        discoverable adapter name.
    """
    entry = _REGISTRY.get(name)
    if entry is not None:
        return entry

    for entry_point in _iter_adapter_entry_points():
        if entry_point.name == name:
            entry_point.load()
            break

    entry = _REGISTRY.get(name)
    if entry is not None:
        return entry

    options = " | ".join(sorted(available())) or "<none>"
    raise ValueError(
        f"Unknown simulator adapter {name!r}. Available adapters: "
        f"{options}. Third-party adapters can be exposed through the "
        f"{ENTRY_POINT_GROUP!r} entry-point group or loaded explicitly "
        f"with --adapter-module."
    )


def available() -> Dict[str, str]:
    """Return {name: description} for every discoverable adapter.

    Includes adapters registered in this process and entry-point adapters
    that have not been imported yet (shown with a generic description, since
    describing them would require importing them).
    """
    names = {e.name: e.description for e in _REGISTRY.values()}

    for entry_point in _iter_adapter_entry_points():
        names.setdefault(
            entry_point.name,
            f"entry point ({entry_point.value})",
        )

    return names


class NoAdapterMatch(ValueError):
    """No registered adapter recognizes this environment object."""


class AmbiguousAdapterMatch(ValueError):
    """More than one registered adapter claims this environment object."""


def _env_chain(env: Any) -> Iterable[Any]:
    """Yield env, then each `.env` inner wrapper, cycle-safe (outermost first)."""
    seen = set()
    current = env
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = getattr(current, "env", None)


def _import_all_entry_points() -> None:
    """Best-effort import of every entry-point adapter module so its
    registration (including handles/wrap) is visible to detection.
    Modules that fail to import (missing sim deps) are skipped: an
    adapter that cannot import cannot own the researcher's live env."""
    for entry_point in _iter_adapter_entry_points():
        try:
            entry_point.load()
        except Exception:
            continue


@dataclass(frozen=True)
class AdapterStatus:
    """What ``specola doctor`` reports for one adapter."""

    name:         str
    description:  str
    install_hint: str
    requires:     Tuple[str, ...]
    missing:      Tuple[str, ...]
    load_error:   str = ""
    checked:      bool = True

    @property
    def ready(self) -> bool:
        return self.checked and not self.load_error and not self.missing


def adapter_status(load_third_party: bool = True) -> List[AdapterStatus]:
    """Status of every discoverable adapter, sorted by name.

    Imports each entry-point adapter module (specola's own adapter modules
    import their simulators lazily, inside the factory) and checks each
    declared requirement with ``importlib.util.find_spec``, which locates a
    package without importing it. An adapter whose module fails to import
    is reported with the error rather than skipped.

    ``load_third_party=False`` imports only specola's bundled adapters and
    lists other entry points as unchecked, for callers that must not run
    third-party import code (the banner printed by bare ``specola``).
    """
    errors: Dict[str, str] = {}
    unchecked: List[str] = []
    for entry_point in _iter_adapter_entry_points():
        if entry_point.name in _REGISTRY:
            continue
        if not load_third_party and not str(entry_point.value).startswith("specola."):
            unchecked.append(entry_point.name)
            continue
        try:
            entry_point.load()
        except Exception as exc:
            errors[entry_point.name] = f"{type(exc).__name__}: {exc}"

    statuses = []
    for name, entry in _REGISTRY.items():
        missing = tuple(m for m in entry.requires if not _module_found(m))
        statuses.append(AdapterStatus(
            name=name, description=entry.description,
            install_hint=entry.install_hint, requires=entry.requires,
            missing=missing,
        ))
    for name, error in errors.items():
        if name not in _REGISTRY:
            statuses.append(AdapterStatus(
                name=name, description="", install_hint="", requires=(),
                missing=(), load_error=error,
            ))
    for name in unchecked:
        if name not in _REGISTRY:
            statuses.append(AdapterStatus(
                name=name, description="", install_hint="", requires=(),
                missing=(), checked=False,
            ))
    return sorted(statuses, key=lambda s: s.name)


def _module_found(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detect(env: Any) -> _AdapterEntry:
    """Return the unique adapter entry whose ``handles()`` claims ``env``.

    Walks the wrapper chain outermost-first; the first link with exactly
    one match wins. Zero matches anywhere raises NoAdapterMatch; more
    than one match at a link raises AmbiguousAdapterMatch. Both errors
    name the candidates and the explicit-adapter escape hatch.
    """
    _import_all_entry_points()
    candidates = [
        (e, e.handles) for e in _REGISTRY.values() if e.handles is not None
    ]

    for link in _env_chain(env):
        matches = [e for e, handles in candidates if handles(link)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(sorted(m.name for m in matches))
            raise AmbiguousAdapterMatch(
                f"Multiple adapters claim {type(link).__name__} "
                f"({type(link).__module__}): {names}. Pass an explicit "
                f"adapter instance instead of the raw env."
            )

    tried = ", ".join(sorted(e.name for e, _ in candidates)) or "<none>"
    raise NoAdapterMatch(
        f"No adapter recognizes {type(env).__name__} "
        f"(module {type(env).__module__!r}). Adapters with detection "
        f"support: {tried}. Construct the adapter yourself and pass it "
        f"explicitly, e.g. Monitor(...).watch(MyAdapter(env))."
    )


def wrap_env(env: Any, **kwargs: Any) -> SimulatorAdapter:
    """Detect the adapter for ``env`` and construct it around the env."""
    entry = detect(env)
    if entry.wrap is None:
        raise ValueError(
            f"Adapter {entry.name!r} matched this env but does not "
            f"support wrapping an existing env; construct it explicitly."
        )
    return entry.wrap(env, **kwargs)
