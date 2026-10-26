"""
run_monitor.py — STL monitor evaluation CLI
=======================================================
Load a specs YAML, run the monitor in lockstep with a simulator, stream
STL robustness to the terminal, and save every per-binding robustness
value to a Parquet file for post-processing.

The output file is always saved automatically — no flag needed.  Each
invocation gets its own run directory; the (single) episode it drives
gets its own Parquet file inside it:
    results/<run_id>/<YYYYMMDD_HHMMSS>_ep<N>_<4-hex>.parquet
where run_id = <YYYYMMDD_HHMMSS>_<sim>_<task-slug>. This CLI always
watches exactly one episode per run, so N is always 0.

Drive modes
-----------
  --mode random            random actions (proves the loop runs)
  --mode hold              zero action, physics settles (sanity check)
  --mode replay --demo X   replay a Specola state recording
                           (<episode>_states.npz, written by every run on
                           an adapter with a state tier) or a
                           RoboCasa/robomimic HDF5 demonstration
                           (the constructed env's MuJoCo model must match
                           the recording env — pass the same --task; see
                           docs/ROADMAP.md on honoring env_args metadata)

Usage
-----
    specola run --sim mock --specs examples/mock_specs.yaml --steps 150
    specola run --sim robocasa --specs my_specs.yaml --queries examples/geometry_queries.yaml --mode random
    specola run --sim libero --task libero_object:0 --specs my_specs.yaml --queries examples/geometry_queries.yaml --mode replay --demo demo.hdf5

Post-processing
---------------
    import pandas as pd
    df = pd.read_parquet(
        "results/20260709_153042_robocasa_PickPlace/20260709_153042_ep0_a1b2.parquet"
    )
    # columns: step, time, spec, authored_spec, binding, binding_key, rho,
    #          suspended, verdict_final, window_steps, episode_id, status,
    #          success, not_applicable, plus one column per meta key (this
    #          CLI passes sim, task, mode, seed)
    df[df.spec == "test"].pivot(index="time", columns="binding", values="rho").plot()
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import logging
import re
import shutil
import sys
import tempfile
import textwrap
import traceback
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from threading import Event, RLock
from typing import Callable, Dict, Iterator, Optional, TextIO, Tuple, cast

import numpy as np

from .adapter import SimulatorAdapter
from .conformance.provider import ExtractionGuardError
from .entities import SpecConfigError
from .live_display import supports_live
from .session import Monitor, Session, _LiveEntityLabels
from .state_recording import (
    SUFFIX as STATE_SUFFIX, StateRecording, StateRecordingError, StateWriter,
    check_compatible, is_state_recording,
)

from .cli_common import (
    bold, dim, green, amber, red,
    _validate_dt_override, construct_sim, print_ready,
)
from .run_dashboard import WorkerAbandoned


class _TeeStream:
    """Mirror direct simulator output to its terminal and durable log."""

    def __init__(self, terminal: TextIO, log: io.TextIOBase) -> None:
        self._terminal = terminal
        self._log = log

    def write(self, value: str) -> int:
        self._terminal.write(value)
        self._log.write(value)
        self._log.flush()
        return len(value)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._terminal.isatty()

    def fileno(self) -> int:
        # faulthandler.enable(), which Isaac Sim's SimulationApp calls at
        # construction, wants the descriptor behind sys.stderr; the tee
        # answers with its terminal's so the C-level trace still lands
        # there. The log copy is not mirrored for that path.
        return self._terminal.fileno()

    @property
    def encoding(self):
        return self._terminal.encoding


@contextmanager
def tee_process_streams(log_stream: io.TextIOBase):
    """Mirror process stdout/stderr to a durable log stream."""
    with redirect_stdout(_TeeStream(sys.stdout, log_stream)):
        with redirect_stderr(_TeeStream(sys.stderr, log_stream)):
            yield


class _SimulatorOutput(io.TextIOBase):
    """Own a stable, locked log stream across startup and run publication."""

    encoding = "utf-8"

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._lock = RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("a", encoding="utf-8", buffering=1)

    def write(self, value: str) -> int:
        with self._lock:
            return self._stream.write(value)

    def flush(self) -> None:
        with self._lock:
            stream = getattr(self, "_stream", None)
            if stream is not None and not stream.closed:
                stream.flush()

    def writable(self) -> bool:
        return True

    @contextmanager
    def capture(self, *, enabled: bool = True):
        if not enabled:
            yield self
            return
        with tee_process_streams(self):
            yield

    @property
    def stream(self) -> TextIO:
        return cast(TextIO, self)

    def promote(self, path: Path) -> None:
        """Append startup bytes once, retaining the original on copy failure."""
        with self._lock:
            if path.resolve() == self.path.resolve():
                return
            self._stream.flush()
            path.parent.mkdir(parents=True, exist_ok=True)
            destination = path.open("a", encoding="utf-8", buffering=1)
            try:
                with self.path.open("r", encoding="utf-8") as source:
                    shutil.copyfileobj(source, destination)
                destination.flush()
            except BaseException:
                destination.close()
                raise
            previous = self._stream
            self._stream = destination
            self.path = path
            previous.close()

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            try:
                super().close()
            finally:
                stream = getattr(self, "_stream", None)
                if stream is not None:
                    stream.close()


def _write_failure(
    *,
    phase: str,
    adapter: str,
    task: Optional[str],
    exc: BaseException,
    diagnostics_path: Path,
    result_path: Optional[Path] = None,
) -> None:
    """Persist the active traceback and render one truthful failure record."""
    from .run_failures import RunFailure, render_run_failure

    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(traceback.format_exc(), encoding="utf-8")
    failure = RunFailure(
        phase=phase,
        adapter=adapter,
        task=task,
        error="{}: {}".format(type(exc).__name__, exc),
        diagnostics_path=str(diagnostics_path.resolve()),
        extraction_status=exc.status if isinstance(exc, ExtractionGuardError) else None,
        result_path=(
            str(result_path.resolve()) if result_path is not None else None
        ),
    )
    print(red(render_run_failure(failure)), file=sys.stderr)


def _record_state(
    state_writer: Optional[StateWriter], adapter, step: int
) -> Optional[StateWriter]:
    """Append this step's state, or stop recording and keep monitoring.

    A state the writer cannot take (a diverging simulator's NaN, an
    adapter that returned none) ends the state file as ``failed`` with
    a printed reason; the run itself goes on, as it did before state
    recording existed. Returns the writer to keep using, or None.
    """
    if state_writer is None:
        return None
    try:
        state = adapter.capture_state()
        if state is None:
            raise StateRecordingError(
                f"{type(adapter).__name__} returned no state at step {step}"
            )
        state_writer.append(state)
        return state_writer
    except StateRecordingError as exc:
        path = state_writer.close("failed")
        print("  {}".format(dim(
            f"State recording stopped at step {step}: {exc}; "
            f"{path} is marked failed, monitoring continues"
        )), flush=True)
        return None


def _close_state_writer_failed(
    state_writer: Optional[StateWriter], exc: BaseException
) -> None:
    """Close the episode's state recording as failed without hiding ``exc``."""
    if state_writer is None:
        return
    try:
        state_writer.close("failed")
    except BaseException as close_error:
        exc.add_note(
            "State recording close also failed: {}: {}".format(
                type(close_error).__name__, close_error
            )
        ) if hasattr(exc, "add_note") else None


# ─────────────────────────────────────────────────────────────────────────────
# Monitor-count visibility
#
# Stage-3 selector fan-out has no cap (explicit user decision): a spec
# that resolves to N monitors gets N monitors, always. This line is pure
# visibility into that fan-out, not a gate — no warning tone, nothing
# ever refused for producing "too many" monitors.
# ─────────────────────────────────────────────────────────────────────────────

def _monitor_count_line(counts: Dict[str, int]) -> str:
    """Format the episode-start monitor-count line.

    ``counts`` maps authored spec name -> number of compiled monitors
    (``MonitorEngine.monitor_counts()``). Always states the total; adds a
    parenthetical only for specs that actually fanned out (n > 1), in
    authored order.
    """
    total = sum(counts.values())
    fanned = ", ".join(
        f"{name}: {n}" for name, n in counts.items() if n > 1
    )
    line = f"monitors: {total}"
    if fanned:
        line += f"  ({fanned})"
    return line


# ─────────────────────────────────────────────────────────────────────────────
# Drive modes
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_demo(demo_path: str) -> Tuple[str, int]:
    """Split ``path.hdf5#N``, validate the fragment and check the path exists.

    Shared by state iteration and scene loading so both always resolve
    the same demo group. Scene loading runs during startup, so a missing
    file is rejected before the run directory exists.
    """
    demo_index = 0
    if "#" in demo_path:
        demo_path, fragment = demo_path.rsplit("#", 1)
        try:
            demo_index = int(fragment)
        except ValueError:
            raise SystemExit(
                f"--demo fragment must be an integer demo index, "
                f"got {fragment!r}"
            )
    if Path(demo_path).suffix == STATE_SUFFIX:
        # Never read a .npz as a robomimic demonstration (a fragment kept
        # it from the state-recording check in prime_replay_env).
        raise SystemExit(
            f"--demo {demo_path} is a Specola state recording; it holds one "
            "episode, so pass it without a #N fragment"
        )
    if not Path(demo_path).exists():
        raise SystemExit(f"--demo path does not exist: {demo_path}")
    return demo_path, demo_index


def _demo_group_name(f, demo_path: str, demo_index: int) -> str:
    def _numeric(key: str) -> Tuple[int, str]:
        match = re.search(r"(\d+)$", key)
        return (int(match.group(1)) if match else -1, key)

    demos = sorted(f["data"].keys(), key=_numeric)
    if not demos:
        raise SystemExit(f"No demos in {demo_path}")
    if not 0 <= demo_index < len(demos):
        raise SystemExit(
            f"--demo index {demo_index} out of range: "
            f"{demo_path} holds {len(demos)} demos"
        )
    return demos[demo_index]


def _iter_replay_states(demo_path: str) -> Iterator[np.ndarray]:
    """Yield recorded flattened MuJoCo states from a robomimic HDF5.

    ``--demo path.hdf5`` replays the file's first demonstration by
    numeric index; ``--demo path.hdf5#12`` selects ``demo_12``. The
    fragment form exists so batch drivers can iterate a multi-demo file
    without extracting per-demo copies.
    """
    demo_path, demo_index = _resolve_demo(demo_path)
    import h5py

    with h5py.File(demo_path, "r") as f:
        demo = _demo_group_name(f, demo_path, demo_index)
        states = f["data"][demo]["states"][:]
        print(dim(f"  replaying demo '{demo}' ({len(states)} states)"))
        for state in states:
            yield state


def _load_replay_scene(demo_path: str) -> dict:
    """Optional per-episode scene records beside a demo's ``states``.

    Flattened MuJoCo state carries qpos/qvel only, which misses two
    things a faithful replay needs:

    - ``model_body_pos`` / ``model_body_quat``: scene layout the env
      bakes into the *model* at reset (LIBERO fixture placement, and
      with it every fixture-anchored region site).
    - ``reset_state``: the flattened state at the recorder's
      adapter-reset moment (pre ``set_init_state``). The adapter's
      first observation — and every derived quantity seeded by it, e.g.
      finite-difference velocities — comes from that scene; without it
      a replay's episode-start rows diverge.

    When present, ``prime_replay_env`` applies them BEFORE the monitor
    compiles, so the reset observation matches the recorded episode.
    """
    demo_path, demo_index = _resolve_demo(demo_path)
    import h5py

    with h5py.File(demo_path, "r") as f:
        group = f["data"][_demo_group_name(f, demo_path, demo_index)]
        keys = ("model_body_pos", "model_body_quat", "reset_state")
        present = [k for k in keys if k in group]
        if not present:
            return {}  # pre-fix corpus: legacy replay, no priming
        if len(present) != len(keys):
            raise SystemExit(
                f"--demo group carries a partial scene record "
                f"({present}); a corrupted or hand-edited demo must not "
                f"silently replay with degraded fidelity"
            )
        return {
            "body_pos": np.asarray(group["model_body_pos"][:], dtype=float),
            "body_quat": np.asarray(group["model_body_quat"][:], dtype=float),
            "reset_state": np.asarray(group["reset_state"][:], dtype=float),
        }


def prime_replay_env(
    adapter,
    demo: Optional[str],
    *,
    simulator: str,
    task: Optional[str],
    dt: Optional[float] = None,
) -> Optional[StateRecording]:
    """Apply a demo's recorded scene to the sim before compilation.

    Must run before ``Monitor.watch()``: adapter reset deliberately does
    not touch the env, so a primed sim is exactly what the reset
    observation reads. No-op when the demo carries no scene records
    (pre-fix corpora) or the env exposes no ``.sim``.

    A Specola state recording is checked against this run (simulator,
    task, the adapter's state signature, and ``dt``, the run's ``--dt``
    or else ``adapter.dt``) and only then is its reset row restored. The
    loaded recording is returned for ``_make_advancer``; any other demo
    returns None. A refusal raises ``StateRecordingError``.
    """
    if demo is None:
        return None
    if is_state_recording(demo):
        signature = adapter.state_signature()
        if signature is None:
            raise StateRecordingError(
                "--mode replay from a Specola state recording is not "
                f"supported by the {simulator} adapter: it has no state tier"
            )
        recording = StateRecording.load(demo)
        check_compatible(
            recording, simulator=simulator, task=task, signature=signature,
            dt=dt if dt is not None else float(adapter.dt),
        )
        adapter.restore_state(recording.states[0])
        print(dim(
            f"  restored the recorded reset state ({len(recording.states)} states)"
        ))
        return recording
    if callable(getattr(adapter, "make_advancer", None)):
        return  # adapter owns replay; its format is not ours to open
    env = getattr(adapter, "env", None)
    sim_owner = _find_in_env_chain(env, "sim") if env is not None else None
    if sim_owner is None:
        return
    scene = _load_replay_scene(demo)
    if not scene:
        return
    sim = sim_owner.sim
    if (sim.model.body_pos.shape != scene["body_pos"].shape
            or sim.model.body_quat.shape != scene["body_quat"].shape):
        raise SystemExit(
            "--demo model snapshot shape mismatch: recorded "
            f"{scene['body_pos'].shape} vs model "
            f"{sim.model.body_pos.shape} — the demo was recorded on "
            "a different scene/model"
        )
    expected = sim.get_state().flatten().shape
    if scene["reset_state"].shape != expected:
        raise SystemExit(
            "--demo reset_state length mismatch: recorded "
            f"{scene['reset_state'].shape} vs live model {expected} — "
            "the demo was recorded on a different model"
        )
    sim.model.body_pos[:] = scene["body_pos"]
    sim.model.body_quat[:] = scene["body_quat"]
    sim.set_state_from_flattened(scene["reset_state"])
    sim.forward()
    print(dim("  applied recorded scene (placement + reset state)"))


def _find_in_env_chain(env, attribute: str):
    """Return the first object in an env wrapper chain exposing an attribute.

    Wrappers such as LIBERO's ControlEnv hold the underlying robosuite
    environment at ``.env`` and do not forward action metadata like
    ``action_spec`` or ``action_dim``. Actions are still fed to the
    outermost wrapper so its bookkeeping runs.
    """
    seen = set()
    current = env
    while current is not None and id(current) not in seen:
        if hasattr(current, attribute):
            return current
        seen.add(id(current))
        current = getattr(current, "env", None)
    return None


def _step_signals_episode_end(result) -> bool:
    """True if a gym/gymnasium-style step() return signals episode end.

    Envs like Meta-World raise if step() is called again once their
    internal step budget is exhausted, instead of just returning stale
    obs — so the advancer must stop itself at truncation rather than
    relying on the caller to notice next time around.
    """
    if not isinstance(result, tuple):
        return False
    if len(result) == 5:      # gymnasium: obs, reward, terminated, truncated, info
        _, _, terminated, truncated, _ = result
        return bool(terminated or truncated)
    if len(result) == 4:      # legacy gym: obs, reward, done, info
        _, _, done, _ = result
        return bool(done)
    return False


def _make_advancer(adapter: SimulatorAdapter, mode: str,
                   demo: Optional[str], seed: int, *,
                   recording: Optional[StateRecording] = None) -> Callable[[int], bool]:
    if recording is not None:
        # A Specola state recording replays on any adapter with a state
        # tier, ahead of the adapter's own drive and the robomimic path:
        # step k restores row k + 1, the state observation k + 1 saw.
        # ``exhausted`` (read by _typed_eval_loop) marks a recording that
        # ran out, as opposed to a step limit.
        def advance(step: int) -> bool:
            if step + 1 >= len(recording.states):
                setattr(advance, "exhausted", True)
                return False
            adapter.restore_state(recording.states[step + 1])
            return True
        return advance

    # Simulator control is adapter/experiment territory (design
    # decision 7): an adapter that exposes make_advancer(mode, demo,
    # seed) -> (step -> bool) owns its drive behavior entirely, and new
    # simulators (CARLA, ROS bridges, ...) need no shared-CLI changes.
    # The duck-typed robosuite/gymnasium fallbacks below exist only for
    # envs that fit those conventions.
    adapter_advancer = getattr(adapter, "make_advancer", None)
    if callable(adapter_advancer):
        return cast(Callable[[int], bool], adapter_advancer(mode, demo, seed))

    if demo is not None and mode != "replay":
        raise SystemExit(f"--demo is only read by --mode replay, not --mode {mode}")

    env = getattr(adapter, "env", None)

    if mode == "replay":
        if env is None:
            raise SystemExit("--mode replay needs a real sim env (not 'mock')")
        if demo is None:
            raise SystemExit("--mode replay requires --demo <path.hdf5>")
        sim_owner = _find_in_env_chain(env, "sim")
        if sim_owner is None:
            raise SystemExit("--mode replay needs an env exposing .sim")
        states = _iter_replay_states(demo)
        def advance(step: int) -> bool:
            try:
                state = next(states)
            except StopIteration:
                setattr(advance, "exhausted", True)
                return False
            sim_owner.sim.set_state_from_flattened(state)
            sim_owner.sim.forward()
            return True
        return advance

    # Action metadata: robosuite-style action_spec/action_dim, or a
    # gymnasium-style Box action_space.
    space_owner = _find_in_env_chain(env, "action_space")

    if mode == "hold":
        dim_owner = _find_in_env_chain(env, "action_dim")
        if env is not None and dim_owner is None and space_owner is None:
            raise SystemExit(
                "--mode hold needs an env exposing action_dim or "
                "action_space"
            )
        done = False
        def advance(step: int) -> bool:
            nonlocal done
            if env is not None:
                if done:
                    return False
                if dim_owner is not None:
                    result = env.step(np.zeros(dim_owner.action_dim))
                else:
                    result = env.step(np.zeros(space_owner.action_space.shape))
                done = _step_signals_episode_end(result)
            return True
        return advance

    rng = np.random.default_rng(seed)
    spec_owner = _find_in_env_chain(env, "action_spec")
    if env is not None and spec_owner is None and space_owner is None:
        raise SystemExit(
            "--mode random needs an env exposing action_spec or "
            "action_space"
        )
    done = False
    def advance(step: int) -> bool:
        nonlocal done
        if env is not None:
            if done:
                return False
            if spec_owner is not None:
                lo, hi = spec_owner.action_spec
                result = env.step(rng.uniform(lo, hi))
            else:
                space = space_owner.action_space
                result = env.step(
                    rng.uniform(space.low, space.high).astype(np.float32)
                )
            done = _step_signals_episode_end(result)
        return True
    return advance


def _snapshot_stream(
    session: Session,
    tracker,
    *,
    advance: Callable[[int], bool],
    max_steps: int,
    render: bool,
    stop_event=None,
    simulator_output: Optional[_SimulatorOutput] = None,
    capture_simulator_output: bool = True,
    web_sink=None,
    state_writer: Optional[StateWriter] = None,
):
    """Drive monitoring once and yield immutable presentation snapshots.

    ``advance`` is built at startup by ``_make_advancer``, so a drive mode
    the adapter refuses never reaches the stream. ``state_writer``
    receives the simulator state after every advance, before the
    observation that describes it.
    """
    adapter = session.adapter
    step = 0
    try:
        while max_steps == 0 or step < max_steps:
            if stop_event is not None and stop_event.is_set():
                break
            if simulator_output is None:
                if not advance(step):
                    break
                state_writer = _record_state(state_writer, adapter, step)
                if render:
                    adapter.render()
                raw = session.step().raw
            else:
                with simulator_output.capture(
                    enabled=capture_simulator_output
                ):
                    if not advance(step):
                        break
                    state_writer = _record_state(state_writer, adapter, step)
                    if render:
                        adapter.render()
                    raw = session.step().raw
            snapshot = tracker.update(raw)
            if web_sink is not None:
                web_sink.on_step(adapter, snapshot)
            yield snapshot
            step += 1
    finally:
        if render or web_sink is not None:
            if simulator_output is None:
                adapter.close_render()
            else:
                with simulator_output.capture(
                    enabled=capture_simulator_output
                ):
                    adapter.close_render()


def _print_typed_summary(record, snapshot, engine, state_path=None) -> None:
    from .run_summary import (
        build_run_summary,
        render_run_summary_sections,
        undetermined_reason,
    )

    summary = build_run_summary(record, snapshot, engine)
    sections = render_run_summary_sections(summary)
    print()
    print("Episode summary · " + sections.heading)
    print(sections.progress)
    print(sections.results_heading)
    manager_specs = engine.manager_specs()
    for name, outcome in record.specs.items():
        manager_names = [
            manager_name
            for manager_name, authored_name in manager_specs.items()
            if authored_name == name
        ]
        if (
            len(manager_names) > 1
            and any(manager_name != name for manager_name in manager_names)
        ):
            # Selector fan-out remains per monitor in the CLI summary.
            # The durable Session/report projection retains the authored
            # specification aggregate.
            continue
        verdict = outcome.verdict.replace("_", " ").upper()
        verdict_style = {
            "pass": green, "fail": red, "undetermined": amber,
            "not_applicable": dim,
        }.get(outcome.verdict)
        if verdict_style is not None:
            verdict = verdict_style(verdict)
        reason = ""
        if outcome.verdict == "undetermined":
            reason = undetermined_reason(summary, name)
        print("  {}  {}{}".format(name, verdict, reason))
    for line in sections.binding_lines + sections.identity_lines:
        print(line)
    if state_path is not None:
        # Listed under the Results line: the episode's state recording.
        print(" " * len("Results     ") + str(Path(state_path).resolve()))
    if snapshot is not None:
        for specification in snapshot.specifications:
            if specification.bindings:
                continue
            explanation = (
                specification.skip_reason
                or specification.state.value.replace("_", " ")
            )
            print(
                "  {} · {} · {}".format(
                    specification.name,
                    specification.state.value.replace("_", " "),
                    explanation,
                )
            )
    print()


def _typed_eval_loop(
    session: Session,
    *,
    advance: Callable[[int], bool],
    mode: str,
    max_steps: int,
    entity_labels: dict,
    render: bool,
    ui: str,
    refresh_rate: float,
    summary_only: bool,
    adapter_name: str = "",
    task: Optional[str] = None,
    diagnostics_path: Optional[Path] = None,
    simulator_log: Optional[Path] = None,
    simulator_output: Optional[_SimulatorOutput] = None,
    phase_state: Optional[dict] = None,
    web_sink=None,
    web_server=None,
    state_writer: Optional[StateWriter] = None,
) -> None:
    from .run_state import (
        BindingEntity, MonitorDefinition, RunStateTracker,
    )

    engine = session.engine
    authored = {
        record["name"]: record
        for record in engine.specification_metadata()
    }
    definitions = {}
    for manager_name, manager in engine.managers.items():
        specification = engine.manager_specs().get(
            manager_name, manager_name
        )
        spec_record = authored.get(specification, {})
        formula = manager.formula
        definitions[manager_name] = MonitorDefinition(
            manager_name=manager_name,
            specification=specification,
            authored_formula=str(spec_record.get("formula", "")),
            resolved_formula=(
                formula.desugared_text or formula.name
            ),
            backend=formula.backend,
            window_steps=formula.window_steps,
            entity_classes=tuple(
                str(getattr(parameter.entity_class, "value",
                            parameter.entity_class))
                for parameter in formula.params
            ),
            components_by_slot=tuple(
                tuple(parameter.required_components)
                for parameter in formula.params
            ),
        )
    adapter = session.adapter
    roles = getattr(adapter, "entity_roles", {}) or {}
    role_basis = getattr(adapter, "entity_role_basis", {}) or {}
    role_provenance = getattr(
        adapter, "entity_role_provenance", {}
    ) or {}
    declared_names = getattr(adapter, "entity_declared_names", {}) or {}
    detail_ids = set(entity_labels) | set(roles) | set(declared_names)
    entity_details = {
        entity_id: BindingEntity(
            entity_id=entity_id,
            entity_class="",
            label=entity_labels.get(entity_id),
            role=roles.get(entity_id),
            role_basis=role_basis.get(entity_id),
            role_provenance=role_provenance.get(entity_id),
            declared_name=declared_names.get(entity_id),
        )
        for entity_id in detail_ids
    }
    metadata_callback = getattr(adapter, "property_metadata", None)
    component_metadata = {}
    if callable(metadata_callback):
        for entity_id in detail_ids:
            try:
                value = metadata_callback(entity_id) or {}
            except Exception:
                value = {}
            if isinstance(value, Mapping):
                component_metadata[entity_id] = value
    tracker = RunStateTracker(
        engine.manager_specs(),
        skipped_specs=engine.skipped_specs(),
        labels=entity_labels,
        definitions=definitions,
        entity_details=entity_details,
        component_metadata=component_metadata,
    )
    snapshot = None
    interrupted = False

    print()
    print(
        bold("  Run")
        + dim(
            "  {} authored spec(s) · {} initial monitor(s) · mode={}".format(
                len(engine.spec_names), len(engine.managers), mode
            )
        )
    )
    print(dim("  Monitoring and result recording run at simulator cadence."))
    print()

    # Cooperative stop for the rich/plain paths (the textual path owns
    # its own Event): the web dashboard's stop_run sets it.
    stop_event = Event()
    if web_server is not None:
        web_server.on_stop_run(stop_event.set)

    def _textual_stream(stop):
        # Share Textual's stop Event with the dashboard's stop_run.
        if web_server is not None:
            web_server.on_stop_run(stop.set)
        return _snapshot_stream(
            session,
            tracker,
            advance=advance,
            max_steps=max_steps,
            render=render,
            stop_event=stop,
            simulator_output=simulator_output,
            capture_simulator_output=False,
            web_sink=web_sink,
            state_writer=state_writer,
        )

    try:
        if ui == "textual":
            from .run_dashboard import run_textual_dashboard

            dashboard = run_textual_dashboard(
                _textual_stream,
                simulator_log=(
                    str(simulator_log.resolve())
                    if simulator_log is not None else None
                ),
                refresh_rate=refresh_rate,
                simulator_stream=(
                    simulator_output.stream
                    if simulator_output is not None else None
                ),
                step_on_main_thread=bool(
                    getattr(adapter, "step_on_main_thread", False)
                ),
            )
            snapshot = dashboard.snapshot
            interrupted = dashboard.stopped
        elif ui == "rich":
            from .run_live import RichRunLive

            display = RichRunLive(refresh_rate)
            renderer_active = False
            renderer_entered = False
            try:
                try:
                    display.__enter__()
                    renderer_active = True
                    renderer_entered = True
                except Exception as exc:
                    print(
                        "  Renderer warning: {}: {}; continuing in plain "
                        "mode.".format(type(exc).__name__, exc),
                        file=sys.stderr,
                    )
                for current in _snapshot_stream(
                    session,
                    tracker,
                    advance=advance,
                    max_steps=max_steps,
                    render=render,
                    stop_event=stop_event,
                    simulator_output=simulator_output,
                    web_sink=web_sink,
                    state_writer=state_writer,
                ):
                    snapshot = current
                    if not summary_only:
                        from .run_log import render_plain_transitions
                        for line in render_plain_transitions(current):
                            if renderer_active:
                                try:
                                    display.emit("  " + line)
                                except Exception as exc:
                                    renderer_active = False
                                    print(
                                        "  Renderer warning: {}: {}; "
                                        "monitoring continues.".format(
                                            type(exc).__name__, exc
                                        ),
                                        file=sys.stderr,
                                    )
                            else:
                                print("  " + line)
                    if renderer_active:
                        try:
                            display.update(current)
                        except Exception as exc:
                            renderer_active = False
                            print(
                                "  Renderer warning: {}: {}; monitoring "
                                "continues.".format(type(exc).__name__, exc),
                                file=sys.stderr,
                            )
            finally:
                if renderer_entered:
                    try:
                        display.__exit__(None, None, None)
                    except Exception as exc:
                        print(
                            "  Renderer shutdown warning: {}: {}".format(
                                type(exc).__name__, exc
                            ),
                            file=sys.stderr,
                        )
        else:
            for current in _snapshot_stream(
                session,
                tracker,
                advance=advance,
                max_steps=max_steps,
                render=render,
                stop_event=stop_event,
                simulator_output=simulator_output,
                web_sink=web_sink,
                state_writer=state_writer,
            ):
                snapshot = current
                if not summary_only:
                    from .run_log import render_plain_transitions
                    for line in render_plain_transitions(current):
                        print("  " + line)
    except KeyboardInterrupt:
        interrupted = True
        print("\n  {}".format(dim("Stopped by user")))
    except WorkerAbandoned as exc:
        # The simulator worker may still be mutating the session; calling
        # session.end() here would race it. Report honestly and leave no
        # result record rather than a corrupt one. The state writer is
        # left open for the same reason (the worker may still append), so
        # its .rows spool stays on disk and no .npz claims a status.
        if web_sink is not None:
            web_sink.finalize_episode("failed")
        print(
            "\n  Simulator step abandoned mid-execution; results were NOT "
            "finalized ({}).".format(exc),
            file=sys.stderr,
        )
        if diagnostics_path is not None:
            _write_failure(
                phase="monitor evaluation",
                adapter=adapter_name or type(session.adapter).__name__,
                task=task,
                exc=exc,
                diagnostics_path=diagnostics_path,
                result_path=None,
            )
        raise
    except BaseException as exc:
        result_path = None
        try:
            record = session.end(status="failed")
            result_path = record.path
        except BaseException as finalization_error:
            exc.add_note(
                "Result finalization also failed: {}: {}".format(
                    type(finalization_error).__name__,
                    finalization_error,
                )
            ) if hasattr(exc, "add_note") else None
        _close_state_writer_failed(state_writer, exc)
        # After session.end, matching the normal path's ordering: the
        # recording never claims an outcome the results don't have.
        if web_sink is not None:
            web_sink.finalize_episode("failed")
        if diagnostics_path is not None:
            _write_failure(
                phase="monitor evaluation",
                adapter=adapter_name or type(session.adapter).__name__,
                task=task,
                exc=exc,
                diagnostics_path=diagnostics_path,
                result_path=result_path,
            )
        raise

    # A dashboard stop_run ends the stream cleanly; without this the run
    # (and the recording manifest) would be mislabeled "completed" — the
    # same mislabel class as the absorbed-Ctrl-C case below.
    if stop_event.is_set():
        interrupted = True
    # A replay whose recording ran out before a positive --steps did not
    # monitor what was asked; --steps 0 asks for the whole recording.
    exhausted = bool(getattr(advance, "exhausted", False)) and max_steps > 0
    status = "truncated" if interrupted or exhausted else "completed"
    try:
        record = session.end(status=status)
        state_path = (
            state_writer.close(status) if state_writer is not None else None
        )
    except BaseException as exc:
        # Result finalization failed: the run did not complete; the
        # recording must not claim it did.
        _close_state_writer_failed(state_writer, exc)
        if web_sink is not None:
            web_sink.finalize_episode("failed")
        if diagnostics_path is not None:
            _write_failure(
                phase=(
                    str(phase_state.get("phase"))
                    if phase_state is not None else "result finalization"
                ),
                adapter=adapter_name or type(session.adapter).__name__,
                task=task,
                exc=exc,
                diagnostics_path=diagnostics_path,
            )
        raise
    # Finalize the recording manifest HERE, beside session.end(), where
    # the terminal status is actually known: the outer finally never
    # sees the absorbed Ctrl-C, and finalizing before session.end()
    # would freeze "completed" into a run whose results then failed to
    # save.
    if web_sink is not None:
        web_sink.finalize_episode(status)
    if exhausted:
        print("\n  {}".format(dim(
            "recording exhausted after {} steps; --steps asked for {}".format(
                record.steps, max_steps
            )
        )))
    _print_typed_summary(record, snapshot, engine, state_path)

    unmatched = engine.unmatched_slots()
    for name, params in unmatched.items():
        for parameter in params:
            print(
                "  {} {}: parameter {} matched no entity at any step".format(
                    amber("⚠"),
                    name,
                    parameter.param_key,
                )
            )


def _resolve_ui_and_summary(
    args: argparse.Namespace, compatible_tty: bool, web_active: bool
) -> Tuple[str, bool, bool]:
    """Resolve the terminal UI and summary-only mode from parsed args.

    An explicit --ui, --summary-only, or --no-live always wins exactly as
    given. Otherwise, a *successfully serving* --web dashboard (issue
    #102) implies a summary-only plain terminal — a one-line startup
    notice plus the final summary, since the browser dashboard is the
    viewport and the Textual worker-thread overhead is otherwise paid
    for a terminal nobody watches. ``web_active`` must reflect whether
    the dashboard actually came up (the same check the end-of-run
    "still serving" message uses) — a --web that failed to start (bad
    port, missing extra) falls back to today's normal default instead,
    since there is then neither a browser nor a terminal viewport to
    lose. Absent --web, today's TTY-conditional default is untouched.
    Returns ``(ui, summary_only, web_default_applied)``.
    """
    web_default_applies = (
        web_active
        and args.ui is None
        and not args.summary_only
        and not args.no_live
    )
    if args.no_live:
        ui = "plain"
    elif args.ui:
        ui = args.ui
    elif web_default_applies:
        ui = "plain"
    elif compatible_tty and args.render:
        ui = "rich"
    elif compatible_tty:
        ui = "textual"
    else:
        ui = "plain"
    summary_only = args.summary_only or web_default_applies
    return ui, summary_only, web_default_applies


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _discovery_reset_hook(
    consume,
    *,
    simulator: str,
    task,
    dt_override,
    task_description,
):
    """Serialize the episode's full reset before scope narrows the adapter.

    The consumer receives finished bytes. The CLI stages them until startup
    guards pass, then queues them before the first recording snapshot.
    Discovery failures remain browsing failures and do not stop monitoring.
    Without ``dt_override`` the effective dt is ``adapter.dt`` after reset.
    """
    if consume is None:
        return None

    def hook(adapter, reset) -> None:
        try:
            from .discovery_capture import discovery_payload

            consume(discovery_payload(
                adapter,
                reset,
                simulator=simulator,
                task=task,
                effective_dt=(dt_override if dt_override is not None
                              else float(adapter.dt)),
                task_description=task_description,
            ))
        except Exception as exc:
            from .dashboard.recorder import warn as _recorder_warn

            _recorder_warn(
                "{}: {}; discovery snapshot skipped, monitoring and "
                "recording continue.".format(type(exc).__name__, exc)
            )

    return hook


def main():
    from .conformance.verification import require_ordinary_execution
    require_ordinary_execution()
    ap = argparse.ArgumentParser(
        prog="specola run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
          specola run --sim mock --specs examples/mock_specs.yaml --steps 150
          specola run --sim robocasa --specs my_specs.yaml --queries examples/geometry_queries.yaml --mode random
          specola run --sim libero --task libero_object:0 --specs my_specs.yaml --queries examples/geometry_queries.yaml --mode replay --demo demo.hdf5

          Results are always saved automatically, one Parquet file per episode
          (CSV when pyarrow is not installed):
              results/<run_id>/<timestamp>_ep<N>_<hex>.parquet
          Post-processing:
              import pandas as pd
              df = pd.read_parquet("results/...")
              # columns: step, time, spec, authored_spec, binding,
              #          binding_key, rho, suspended, verdict_final,
              #          window_steps, episode_id, status, success,
              #          not_applicable, sim, task, mode, seed

          Textual controls:
              Space expand/collapse   Enter inspect   J JSON   T transitions
              L simulator log   / search   f filter   s sort   q stop
        """),
    )
    ap.add_argument("--sim",     default="mock")
    ap.add_argument("--queries", type=Path, default=None, metavar="PATH",
                    help="Native query configuration YAML or JSON, separate from specs")
    ap.add_argument("--adapter-module", action="append", default=[],
                    metavar="MODULE",
                    help="Import MODULE so its adapter registration runs "
                         "(repeatable); use with --sim <registered-name>")
    ap.add_argument("--task",    default=None,
                    help="Task name; its form depends on --sim (see the adapter table in README.md)")
    ap.add_argument("--robot",   default=None,
                    help="Robot name (default: PandaOmron)")
    ap.add_argument("--specs",   required=True,
                    help="Path to a specs YAML (author one with specola write-specs)")
    ap.add_argument("--mode",    default="random",
                    choices=["random", "hold", "replay"],
                    help="How to drive the sim (mock and dataset run "
                         "random and hold alike)")
    ap.add_argument("--demo",    default=None,
                    help="What --mode replay replays: a Specola state "
                         "recording (<episode>_states.npz) or a robomimic "
                         "HDF5 demonstration; append #N to an HDF5 path "
                         "to select demo N of a multi-demo file "
                         "(default: the first by numeric index)")
    ap.add_argument("--steps",   type=int, default=0,
                    help="Eval steps (0 = until Ctrl-C or replay end)")
    ap.add_argument("--seed",    type=int, default=0,
                    help="RNG seed for --mode random, the robocasa kitchen, "
                         "the metaworld task variant and the isaaclab "
                         "environment seed")
    ap.add_argument("--all-regions", action="store_true",
                    help="Discover regions from every fixture, not just "
                         "the task's target regions (adapters without "
                         "the distinction ignore this)")
    ap.add_argument("--dt",      type=float, default=None)
    ap.add_argument("--render",  action="store_true",
                    help="open a live viewer window and refresh it every "
                         "step; interactive TTYs use the Rich terminal UI "
                         "so the simulator viewer remains on the main thread "
                         "(adapters without a live viewer refuse this)")
    ap.add_argument(
        "--ui", choices=("textual", "rich", "plain"), default=None,
        help="Terminal UI; defaults to Textual on a TTY and plain otherwise",
    )
    ap.add_argument(
        "--web", nargs="?", const=8090, type=int, default=None,
        metavar="PORT",
        help="Serve a live browser dashboard on 127.0.0.1 (default port "
             "8090, 0 picks a free port; needs the 'web' extra) and "
             "record the run for later replay with 'specola view'. "
             "Composable with every --ui mode. The process keeps "
             "serving after the run ends until Ctrl-C. Without an "
             "explicit --ui, the terminal defaults to a one-line notice "
             "plus the final summary (pass --ui textual/rich to also "
             "keep a live terminal UI).",
    )
    ap.add_argument(
        "--web-cameras", default=None, metavar="NAME[,NAME...]",
        help="Cameras to record for --web: comma-separated names, or "
             "'all' for every named camera (default: the adapter's "
             "default camera)",
    )
    ap.add_argument(
        "--web-size", default="1280x720", metavar="WxH",
        help="JPEG capture size for --web (default: 1280x720; larger "
             "sizes and multi-camera setups may need --web-stride)",
    )
    ap.add_argument(
        "--web-stride", type=int, default=1, metavar="N",
        help="Record every Nth camera frame for --web; data is recorded "
             "every step regardless (default: 1)",
    )
    ap.add_argument(
        "--web-video", choices=("auto", "nvenc", "jpeg"), default="auto",
        help="How --web records camera frames: 'nvenc' for hardware "
             "h264 video segments, 'jpeg' for one JPEG per step, or "
             "'auto' (default) to use the hardware encoder when this "
             "machine has a working one. Both 'auto' and 'nvenc' fall "
             "back to JPEG when the encoder cannot be started.",
    )
    ap.add_argument("--no-live", action="store_true",
                    help="alias for --ui plain; emit literal transitions")
    ap.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Root directory for automatic result saving",
    )
    ap.add_argument(
        "--run-name", default=None,
        help="Run directory name (default: timestamp, simulator, and task)",
    )
    ap.add_argument(
        "--no-state-recording", action="store_true",
        help="Do not write <episode>_states.npz beside the results; "
             "--mode replay from a state recording still works",
    )
    ap.add_argument(
        "--refresh-rate", type=float, default=10.0, metavar="HZ",
        help="Presentation refresh rate only (default: 10)",
    )
    ap.add_argument(
        "--summary-only", action="store_true",
        help="Suppress live transition lines and print the final summary",
    )
    ap.add_argument(
        "--simulator-log-level", "--sim-log-level",
        dest="sim_log_level", default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Python simulator logger level (default: WARNING)",
    )
    args = ap.parse_args()
    if args.no_live and args.ui not in (None, "plain"):
        ap.error("--no-live cannot be combined with --ui textual or rich")
    if not (0.1 <= args.refresh_rate <= 60.0):
        ap.error("--refresh-rate must be between 0.1 and 60 Hz")
    if args.web_stride < 1:
        ap.error("--web-stride must be >= 1")
    if args.web is not None and not (0 <= args.web <= 65535):
        ap.error("--web PORT must be 0-65535")
    try:
        web_width, web_height = (
            int(piece) for piece in args.web_size.lower().split("x")
        )
        if web_width < 1 or web_height < 1:
            raise ValueError
    except ValueError:
        ap.error("--web-size must be WxH, e.g. 640x480")
    if args.web is None and (
        args.web_cameras is not None
        or args.web_stride != 1
        or args.web_size != "1280x720"
        or args.web_video != "auto"
    ):
        ap.error("--web-cameras/--web-stride/--web-size/--web-video "
                 "require --web")
    web_cameras = None
    if args.web_cameras is not None:
        names = [piece.strip() for piece in args.web_cameras.split(",")]
        names = [name for name in names if name]
        if not names:
            ap.error("--web-cameras must name at least one camera")
        if "all" in names and names != ["all"]:
            ap.error("--web-cameras 'all' cannot be combined with names")
        web_cameras = ("all",) if names == ["all"] else tuple(names)
    if args.run_name is not None and (
        not args.run_name.strip()
        or args.run_name in (".", "..")
        or "/" in args.run_name
        or "\\" in args.run_name
    ):
        ap.error("--run-name must be one non-empty path component")

    logging.getLogger().setLevel(getattr(logging, args.sim_log_level))
    compatible_tty = bool(
        supports_live()
        and getattr(sys.stdin, "isatty", lambda: False)()
    )
    if args.ui == "textual" and not compatible_tty:
        ap.error("--ui textual requires interactive stdin and stdout")
    if args.render and args.ui == "textual":
        ap.error(
            "--render cannot be combined with --ui textual because the "
            "simulator viewer must remain on the main thread; use "
            "--ui rich or --ui plain"
        )

    from .registry import load_adapter_module
    for module_name in args.adapter_module:
        load_adapter_module(module_name)

    # run_id groups this invocation's results under results/<run_id>/. This
    # CLI drives exactly one episode per invocation, but the Monitor/Session
    # facade underneath is built for many (e.g. a training callback calling
    # watch() once per rollout), hence Monitor's run-level constructor arg.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", args.task or "notask")
    task_slug = task_slug.strip("_")[:40] or "notask"
    run_id = args.run_name or f"{ts}_{args.sim}_{task_slug}"
    run_directory = args.output_dir / run_id
    startup_directory = Path(tempfile.mkdtemp(prefix="specola-startup-"))
    diagnostics_path = startup_directory / "run-error.log"
    simulator_log = run_directory / "simulator.log"
    simulator_output = _SimulatorOutput(startup_directory / "simulator.log")
    staged_discovery = []

    web_sink = None
    web_server = None
    phase_state = {"phase": "adapter loading"}

    def set_phase(phase: str) -> None:
        phase_state["phase"] = phase

    log_handler = logging.StreamHandler(simulator_output)
    log_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    logging.getLogger().addHandler(log_handler)
    try:
        from .query_config import load_query_configuration

        queries = load_query_configuration(args.queries) if args.queries is not None else None
        with simulator_output.capture():
            # No reset here: Monitor.watch performs the episode's only one
            # and the ready line reports it (maintainer decision 2026-10-12).
            adapter = construct_sim(
                args.sim, task=args.task, robot=args.robot,
                target_only=not args.all_regions,
                header="Specola Evaluation", queries=queries,
                # Frames are captured for --web and a viewer opens for
                # --render; otherwise a factory may skip its renderer.
                capture_frames=args.web is not None or args.render,
                # A recording replays only on the model it was made on, so
                # a factory that draws its scene draws it from --seed.
                seed=args.seed,
            )
            dt_override = _validate_dt_override(args.dt)
        # An adapter has a live viewer only when it overrides the base no-op.
        if args.render and type(adapter).render is SimulatorAdapter.render:
            raise ValueError(
                "--render is not supported by the {} adapter: it has no "
                "live viewer".format(args.sim)
            )

        # Grab the backend's human task description without using it as the
        # directory identity: explicit --task/--run-name remain reproducible.
        task_desc = args.task
        _env = getattr(adapter, "env", None)
        if _env is not None and hasattr(_env, "get_ep_meta"):
            try:
                task_desc = (
                    _env.get_ep_meta().get("lang", args.task) or args.task
                )
            except Exception:
                pass

        set_phase("specification compilation")
        if not Path(args.specs).exists():
            raise FileNotFoundError(args.specs)
        print(dim(f"  Loading specs: {args.specs} …"))
        monitor = Monitor(
            args.specs,
            results_dir=str(args.output_dir),
            run_id=run_id,
        )
        with simulator_output.capture():
            recording = None
            if args.mode == "replay":
                set_phase("replay priming")
                recording = prime_replay_env(
                    adapter, args.demo, simulator=args.sim, task=args.task,
                    dt=dt_override,
                )
            session = monitor.watch(
                adapter,
                meta={
                    "sim":  args.sim,
                    "task": task_desc or "",
                    "mode": args.mode,
                    "seed": args.seed,
                },
                # --dt reaches the compiled spec windows only through
                # this argument; None keeps adapter.dt read after reset.
                dt=dt_override,
                _phase_hook=set_phase,
                _reset_hook=_discovery_reset_hook(
                    staged_discovery.append if args.web is not None else None,
                    simulator=args.sim,
                    task=args.task,
                    dt_override=dt_override,
                    task_description=task_desc,
                ),
            )
            # Row 0 of the state recording is the reset the episode
            # compiled against; it is held here and written only once
            # startup has passed, so a rejected startup writes no file.
            set_phase("state capture")
            state_writer = None
            state_signature = (
                None if args.no_state_recording else adapter.state_signature()
            )
            reset_state = (
                None if state_signature is None else adapter.capture_state()
            )
            result_path = session._path
            if (state_signature is not None and reset_state is not None
                    and result_path is not None):
                state_writer = StateWriter(
                    result_path.with_name(
                        result_path.stem + "_states" + STATE_SUFFIX
                    ),
                    simulator=args.sim, task=args.task,
                    signature=state_signature,
                    dt=(dt_override if dt_override is not None
                        else float(adapter.dt)),
                )
                # Owned: the drive setup below may move the simulator.
                reset_state = np.array(reset_state, dtype=np.float64)
            print_ready(
                adapter,
                dt_override if dt_override is not None else float(adapter.dt),
                len(adapter.last_observation.perceived_entities),
            )
            # Built here, once, so a mode or --demo the adapter refuses is a
            # startup rejection like a missing --demo (#135). Not before the
            # reset: building can act on the simulator (CARLA's random mode
            # spawns its fleet), and the reset must not observe that.
            set_phase("drive mode setup")
            advance = _make_advancer(
                adapter, args.mode, args.demo, args.seed, recording=recording
            )
        set_phase("startup log publication to " + str(simulator_log))
        simulator_output.promote(simulator_log)
        diagnostics_path = run_directory / "run-error.log"
        try:
            shutil.rmtree(startup_directory)
        except OSError:
            logging.getLogger(__name__).warning(
                "Startup diagnostics retained at %s", startup_directory,
                exc_info=True,
            )
    except (ValueError, SpecConfigError, FileNotFoundError, ExtractionGuardError) as exc:
        with simulator_output.capture():
            _write_failure(
                phase=phase_state["phase"],
                adapter=args.sim,
                task=args.task,
                exc=exc,
                diagnostics_path=diagnostics_path,
            )
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()
        simulator_output.close()
        raise SystemExit(1)
    except BaseException as exc:
        with simulator_output.capture():
            _write_failure(
                phase=phase_state["phase"],
                adapter=args.sim,
                task=args.task,
                exc=exc,
                diagnostics_path=diagnostics_path,
            )
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()
        simulator_output.close()
        raise

    try:
        if args.web is not None:
            try:
                from .dashboard.recorder import WebRecorderSink
                from .dashboard.server import DashboardServer

                web_sink = WebRecorderSink(
                    run_directory, run_id=run_id, simulator=args.sim,
                    task=args.task or "", mode=args.mode,
                    cameras=web_cameras, size=(web_width, web_height),
                    stride=args.web_stride, video=args.web_video,
                )
                web_server = DashboardServer(web_sink.recording_dir, mode="live")
                web_sink.set_callbacks(
                    web_server.on_line, web_server.on_episode_end,
                    web_server.set_discovery_snapshot,
                )
                url = web_server.start(args.web)
                web_sink.set_dashboard_url(url)
                for payload in staged_discovery:
                    web_sink.write_discovery(payload)
                # Flushed: scripts read this address from a pipe or file.
                print("  Web dashboard: {}".format(url), flush=True)
            except Exception as exc:
                print(
                    "  Web warning: {}: {}; continuing without dashboard.".format(
                        type(exc).__name__, exc,
                    ),
                    file=sys.stderr,
                )
                if web_sink is not None:
                    web_sink.close("failed")
                web_sink = None
                web_server = None

        engine = session.engine
        print(
            "  {} loaded {} specification(s) → {} initial monitor(s)".format(
                green("✓"), len(engine.spec_names), len(engine.managers)
            )
        )
        print("  Specifications and bindings")
        manager_specs = engine.manager_specs()
        for specification, count in engine.monitor_counts().items():
            print(
                "    ▼ {:<28} {} initial monitor{}".format(
                    bold(specification), count, "" if count == 1 else "s"
                )
            )
            manager_names = [
                name for name, authored in manager_specs.items()
                if authored == specification
            ]
            for name in manager_names:
                print("        {}".format(dim(name)))

        # An unsatisfiable spec is visible at startup and in the summary.
        for name, reason in engine.skipped_specs().items():
            print(f"  {amber('⚠')} {bold(name)} skipped: {dim(reason)}")

        web_active = (
            web_server is not None and web_sink is not None
            and web_server.active
        )
        ui, summary_only, web_default_applies = _resolve_ui_and_summary(
            args, compatible_tty, web_active
        )
        if (
            compatible_tty and args.render and args.ui is None
            and not args.no_live and not web_default_applies
        ):
            print(
                "note: defaulting to the Rich terminal UI because --render "
                "keeps the simulator viewer on the main thread",
                file=sys.stderr,
            )
        if web_default_applies:
            print(
                dim(
                    "  Terminal UI is off while --web serves the browser "
                    "dashboard; pass --ui textual to restore it."
                )
            )
        set_phase("monitor evaluation")
        if state_writer is not None and reset_state is not None:
            state_writer.append(reset_state)
        _typed_eval_loop(
            session,
            advance=advance,
            mode=args.mode, max_steps=args.steps,
            entity_labels=_LiveEntityLabels(adapter),
            render=args.render,
            ui=ui,
            refresh_rate=args.refresh_rate,
            summary_only=summary_only,
            adapter_name=args.sim,
            task=task_desc or args.task,
            diagnostics_path=run_directory / "run-error.log",
            simulator_log=simulator_log,
            simulator_output=simulator_output,
            phase_state=phase_state,
            web_sink=web_sink,
            web_server=web_server,
            state_writer=state_writer,
        )
        if (web_server is not None and web_sink is not None
                and web_server.active):
            # The server keeps serving the finished recording until the
            # process exits (dashboard spec); batch runs should simply
            # not pass --web.
            print(
                "  Dashboard still serving at {} — Ctrl-C to exit; "
                "reopen later: specola view {}".format(
                    web_server.url, web_sink.recording_dir
                ),
                flush=True,
            )
            from .dashboard.server import wait_for_interrupt

            # The recording is already finalized, so the heap the wait
            # leaves frozen holds nothing that needs a finalizer.
            wait_for_interrupt()
            print("\n  {}".format(dim("Dashboard closed")))
    finally:
        if web_sink is not None:
            # Failed-path backstop only: the true status was finalized
            # beside session.end() inside _typed_eval_loop; this is a
            # no-op there.
            web_sink.close("failed")
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()
        simulator_output.close()


if __name__ == "__main__":
    main()
