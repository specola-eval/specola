import asyncio
import time
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from queue import Empty, Queue
from threading import Event

from specola.results import BindingEvaluation, EngineResult, EvaluationStatus
from specola.run_dashboard import (
    _specification_status,
    run_textual_dashboard,
)
from specola.run_state import (
    BindingState,
    MonitorDefinition,
    RunSnapshot,
    RunTransition,
    RunStateTracker,
    SpecificationSnapshot,
    SpecificationState,
)


def _live_snapshots(binding_count=20):
    managers = {
        "manager_{}".format(index): "authored"
        for index in range(binding_count)
    }
    definitions = {
        manager: MonitorDefinition(
            manager_name=manager,
            specification="authored",
            authored_formula="G[0,1s](object_*[mass] >= 0)",
            resolved_formula=(
                "G[0,1s](object_{}[mass] >= 0)".format(index)
            ),
            entity_classes=("object",),
            components_by_slot=(("mass",),),
        )
        for index, manager in enumerate(managers)
    }
    labels = {
        index: (
            "fixture_10[cacc_signal_2]"
            if index == 10 else "entity_{}".format(index)
        )
        for index in range(binding_count)
    }
    tracker = RunStateTracker(
        managers, definitions=definitions, labels=labels
    )

    def snapshot(step):
        return tracker.update(EngineResult(
            timestep=step,
            time=step * 0.1,
            evaluations=tuple(
                BindingEvaluation(
                    spec_name=manager,
                    binding=(index,),
                    robustness=0.1 + step * 0.01,
                    status=EvaluationStatus.ACTIVE,
                    verdict_final=True,
                    timestep=step,
                    time=step * 0.1,
                )
                for index, manager in enumerate(managers)
            ),
        ))

    return snapshot


def _mailbox_stream():
    mailbox = Queue()
    worker_done = Event()

    def snapshots(stop):
        try:
            while not stop.is_set():
                try:
                    snapshot = mailbox.get(timeout=0.02)
                except Empty:
                    continue
                yield snapshot
        finally:
            worker_done.set()

    return mailbox, worker_done, snapshots


def test_textual_dashboard_consumes_snapshot_at_requested_refresh_rate(
    monkeypatch,
):
    from textual.app import App

    intervals = []
    original = App.set_interval

    def capture_interval(self, interval, *args, **kwargs):
        intervals.append(interval)
        return original(self, interval, *args, **kwargs)

    monkeypatch.setattr(App, "set_interval", capture_interval)
    snapshot = RunSnapshot(
        step=3,
        simulator_time=0.3,
        specifications=(),
        transitions=(),
    )

    result = run_textual_dashboard(
        lambda stop: iter((snapshot,)),
        headless=True,
        refresh_rate=4.0,
    )

    assert result.snapshot == snapshot
    assert result.stopped is False
    assert 0.25 in intervals


def test_textual_dashboard_can_step_on_the_main_thread():
    # Kit-hosted simulators deadlock when stepped off the main thread (#130).
    import threading

    threads = []
    closed = []

    def snapshots(stop):
        try:
            step = 0
            while True:
                threads.append(threading.current_thread())
                time.sleep(0.005)
                yield RunSnapshot(
                    step=step, simulator_time=0.0,
                    specifications=(), transitions=(),
                )
                step += 1
        finally:
            closed.append(True)

    result = run_textual_dashboard(
        snapshots, headless=True, step_on_main_thread=True
    )

    # The headless stop timer firing proves steps leave the loop free.
    assert result.stopped is True
    assert len(threads) > 1
    assert all(thread is threading.main_thread() for thread in threads)
    assert closed == [True]


def test_specification_status_hides_population_history_until_expanded():
    specification = SpecificationSnapshot(
        name="dynamic",
        state=SpecificationState.ACTIVE,
        bindings=(),
        state_counts=((BindingState.SATISFIED, 1),),
        active_count=1,
        seen_count=3,
    )

    assert "active /" not in _specification_status(
        specification,
        expanded=False,
    )
    assert "1 active / 3 seen" in _specification_status(
        specification,
        expanded=True,
    )


def test_specification_status_keeps_exact_skip_reason_visible():
    specification = SpecificationSnapshot(
        name="missing",
        state=SpecificationState.NOT_APPLICABLE,
        bindings=(),
        state_counts=(),
        skip_reason="UnknownClass: microwave is not declared",
    )

    status = _specification_status(specification, expanded=False)

    assert "not applicable" in status
    assert "UnknownClass: microwave is not declared" in status


def test_state_color_repaints_unchanged_cells_on_transition():
    # Whole-row state colors: when only the state changes, cells whose
    # text is identical (the label) must still repaint in the new color.
    managers = {"manager_0": "authored"}
    definitions = {
        "manager_0": MonitorDefinition(
            manager_name="manager_0",
            specification="authored",
            authored_formula="G[0,1s](object_*[mass] >= 0)",
            resolved_formula="G[0,1s](object_0[mass] >= 0)",
            entity_classes=("object",),
            components_by_slot=(("mass",),),
        )
    }
    tracker = RunStateTracker(
        managers, definitions=definitions, labels={0: "entity_0"}
    )

    def snap(step, rho):
        return tracker.update(EngineResult(
            timestep=step,
            time=step * 0.1,
            evaluations=(BindingEvaluation(
                spec_name="manager_0",
                binding=(0,),
                robustness=rho,
                status=EvaluationStatus.ACTIVE,
                verdict_final=True,
                timestep=step,
                time=step * 0.1,
            ),),
        ))

    mailbox, worker_done, snapshots = _mailbox_stream()
    mailbox.put(snap(0, 0.5))
    app = run_textual_dashboard(snapshots, build_only=True)

    async def exercise():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("space")
            await pilot.pause()
            table = app.query_one("#run")
            label = table.get_row_at(1)[0]
            assert str(label) == "    entity_0"
            style_before = str(label.style)
            assert "red" not in style_before

            mailbox.put(snap(1, -0.5))
            await pilot.pause(0.3)
            label = table.get_row_at(1)[0]
            assert str(label) == "    entity_0"  # text unchanged
            assert str(label.style) == "bold red"  # color repainted
            # Robustness numbers follow the row color, not their sign.
            now_cell = table.get_row_at(1)[2]
            assert str(now_cell.style) == "bold red"
            await pilot.press("ctrl+q")

        assert worker_done.wait(1)

    asyncio.run(exercise())


def test_textual_dashboard_expands_inspects_filters_logs_and_stops(tmp_path):
    snapshot = RunStateTracker(
        {"manager": "authored"}, labels={2: "onion"}
    ).update(EngineResult(
        timestep=3,
        time=0.3,
        evaluations=(BindingEvaluation(
            spec_name="manager",
            binding=(2,),
            robustness=-0.2,
            status=EvaluationStatus.ACTIVE,
            verdict_final=True,
            timestep=3,
            time=0.3,
        ),),
    ))

    def snapshots(stop):
        yield snapshot
        stop.wait(5)

    simulator_log = tmp_path / "simulator.log"
    simulator_log.write_text("WARN retained simulator warning\n")
    app = run_textual_dashboard(
        snapshots,
        simulator_log=str(simulator_log),
        build_only=True,
    )

    async def exercise():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app.query_one("#run").row_count == 1
            await pilot.press("space")
            await pilot.pause()
            assert app.query_one("#run").row_count == 2
            await pilot.press("j")
            await pilot.press("enter")
            await pilot.pause()
            assert "Current state" in str(
                app.query_one("#detail_content").render()
            )
            await pilot.press("shift+j")
            await pilot.pause()
            assert "schema_version" in str(
                app.query_one("#detail_content").render()
            )
            await pilot.press("f")
            await pilot.press("s")
            await pilot.press("l")
            await pilot.pause()
            assert "retained simulator warning" in str(
                app.query_one("#detail_content").render()
            )
            await pilot.press("q")

    asyncio.run(exercise())


def test_live_updates_preserve_cursor_scroll_and_allow_inspection():
    make_snapshot = _live_snapshots()
    mailbox, worker_done, snapshots = _mailbox_stream()
    mailbox.put(make_snapshot(0))
    app = run_textual_dashboard(snapshots, build_only=True)

    async def exercise():
        async with app.run_test(size=(100, 22)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("space")
            await pilot.pause()
            table = app.query_one("#run")
            table.move_cursor(row=11)
            table.scroll_to(
                y=8, animate=False, force=True, immediate=True
            )
            selected_before = app._selected_key()
            scroll_before = table.scroll_y

            for step in range(1, 8):
                mailbox.put(make_snapshot(step))
            await pilot.pause(0.3)

            assert app._selected_key() == selected_before
            assert table.cursor_coordinate.row == 11
            assert table.scroll_y == scroll_before

            await pilot.press("enter")
            await pilot.pause()
            detail = str(app.query_one("#detail_content").render())
            assert "fixture_10[cacc_signal_2]" in detail
            assert "object_10[mass]" in detail
            assert app.query_one("#detail").can_focus
            await pilot.press("ctrl+q")

        assert worker_done.wait(1)
        assert not app.driver_thread.is_alive()

    asyncio.run(exercise())


def test_single_click_inspects_a_row_the_cursor_is_not_on():
    # Stock DataTable posts RowSelected only when the clicked row already
    # holds the cursor; ClickSelectDataTable makes one click select. The
    # stock handler still runs (Textual dispatches _on_click per MRO
    # class), so this also pins that a click inspects exactly once.
    make_snapshot = _live_snapshots()
    mailbox, worker_done, snapshots = _mailbox_stream()
    mailbox.put(make_snapshot(0))
    app = run_textual_dashboard(snapshots, build_only=True)

    async def exercise():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("space")
            await pilot.pause()
            table = app.query_one("#run")
            assert table.cursor_coordinate.row != 3
            detail_before = str(app.query_one("#detail_content").render())
            # Border (1) + column header (1) put data row 3 at y=5.
            await pilot.click("#run", offset=(4, 5))
            await pilot.pause()
            assert table.cursor_coordinate.row == 3
            detail_after = str(app.query_one("#detail_content").render())
            assert detail_after != detail_before
            await pilot.press("ctrl+q")

        assert worker_done.wait(1)
        assert not app.driver_thread.is_alive()

    asyncio.run(exercise())


def test_latest_snapshot_mailbox_preserves_replaced_transition_deltas():
    make_snapshot = _live_snapshots(binding_count=1)
    produced = Event()

    def snapshots(stop):
        try:
            for step in range(100):
                transition = RunTransition(
                    step=step,
                    simulator_time=step * 0.1,
                    from_state=None,
                    to_state=BindingState.SATISFIED,
                )
                yield replace(
                    make_snapshot(step),
                    transitions=(("authored", "binding_0", transition),),
                )
        finally:
            produced.set()

    app = run_textual_dashboard(
        snapshots,
        build_only=True,
        refresh_rate=2.0,
    )
    publish = dict(zip(
        app.drive_snapshots.__func__.__code__.co_freevars,
        app.drive_snapshots.__func__.__closure__,
    ))["publish"].cell_contents
    closure = dict(zip(publish.__code__.co_freevars, publish.__closure__))
    assert closure["queue"].cell_contents.maxsize == 1

    shown_steps = []
    delivered_steps = []
    original_show = app._show
    original_record = app._record_transition_deltas

    def record_show(snapshot, **kwargs):
        shown_steps.append(snapshot.step)
        return original_show(snapshot, **kwargs)

    def record_transitions(transitions):
        delivered_steps.extend(
            transition.step for _, _, transition in transitions
        )
        return original_record(transitions)

    app._show = record_show
    app._record_transition_deltas = record_transitions
    drive_snapshots = app.drive_snapshots
    drive_snapshots()
    assert produced.wait(1)
    assert closure["queue"].cell_contents.qsize() == 1
    app.drive_snapshots = lambda: None
    exit_callbacks = []
    exit_calls = []

    def record_exit():
        exit_calls.append(True)

    def record_after_refresh(callback, *args, **kwargs):
        if callback is record_exit:
            exit_callbacks.append(callback)

    app.exit = record_exit
    app.call_after_refresh = record_after_refresh

    async def exercise():
        async with app.run_test(size=(100, 22)) as pilot:
            app.poll_snapshots()
            app.poll_snapshots()
            await pilot.pause()

        assert shown_steps == [99]
        assert exit_calls == []
        assert exit_callbacks == [record_exit]
        exit_callbacks[0]()
        assert exit_calls == [True]
        assert delivered_steps == list(range(100))
        assert [
            transition.step for _, _, transition in app.recent_transitions
        ] == list(range(97, 100))

    asyncio.run(exercise())


def test_forced_rebuild_preserves_stable_cursor_and_scroll():
    make_snapshot = _live_snapshots()
    mailbox, worker_done, snapshots = _mailbox_stream()
    mailbox.put(make_snapshot(0))
    app = run_textual_dashboard(snapshots, build_only=True)

    async def exercise():
        async with app.run_test(size=(100, 22)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("space")
            await pilot.pause()
            table = app.query_one("#run")
            table.move_cursor(row=11)
            table.scroll_to(
                y=8, animate=False, force=True, immediate=True
            )
            selected_before = app._selected_key()
            scroll_before = table.scroll_y

            app._show(make_snapshot(1), rebuild=True)
            await pilot.pause()

            assert app._selected_key() == selected_before
            assert table.cursor_coordinate.row == 11
            assert table.scroll_y == scroll_before
            await pilot.press("ctrl+q")

        assert worker_done.wait(1)
        assert not app.driver_thread.is_alive()

    asyncio.run(exercise())


def test_transition_panel_reads_only_per_snapshot_deltas():
    make_snapshot = _live_snapshots(binding_count=1)
    snapshot = make_snapshot(0)
    binding = snapshot.specifications[0].bindings[0]

    class PoisonHistory(tuple):
        def __iter__(self):
            raise AssertionError("full transition history was scanned")

    poisoned_binding = replace(
        binding,
        transitions=PoisonHistory(binding.transitions),
    )
    poisoned_specification = replace(
        snapshot.specifications[0],
        bindings=(poisoned_binding,),
    )
    delta = RunTransition(
        step=4,
        simulator_time=0.4,
        from_state=BindingState.SATISFIED,
        to_state=BindingState.VIOLATED,
    )
    poisoned_snapshot = replace(
        snapshot,
        specifications=(poisoned_specification,),
        transitions=(("authored", binding.binding_key, delta),),
    )
    mailbox, _worker_done, snapshots = _mailbox_stream()
    app = run_textual_dashboard(snapshots, build_only=True)

    async def exercise():
        async with app.run_test() as pilot:
            app._show(poisoned_snapshot)
            app._record_transition_deltas(
                poisoned_snapshot.transitions
            )
            await pilot.pause()
            panel = str(app.query_one("#transitions").render())
            assert "step 4" in panel
            assert "satisfied → violated" in panel
            await pilot.press("ctrl+q")

    asyncio.run(exercise())


def test_escape_leaves_dashboard_search_and_help_restores_inspector():
    make_snapshot = _live_snapshots(binding_count=1)
    mailbox, _worker_done, snapshots = _mailbox_stream()
    mailbox.put(make_snapshot(0))
    app = run_textual_dashboard(snapshots, build_only=True)

    async def exercise():
        async with app.run_test() as pilot:
            await pilot.pause(0.2)
            await pilot.press("/")
            assert app.query_one("#search").has_focus
            await pilot.press("escape")
            await pilot.pause()
            assert app.query_one("#run").has_focus

            await pilot.press("space", "j", "enter")
            await pilot.pause()
            before = str(app.query_one("#detail_content").render())
            await pilot.press("?")
            await pilot.press("escape")
            await pilot.pause()
            assert str(
                app.query_one("#detail_content").render()
            ) == before
            await pilot.press("ctrl+q")

    asyncio.run(exercise())


def test_dynamic_binding_inserts_under_its_specification():
    managers = {"m_a": "alpha", "m_b": "beta"}
    tracker = RunStateTracker(managers, labels={0: "e0", 1: "e1", 2: "e2"})

    def result(step, bindings_by_manager):
        return EngineResult(
            timestep=step,
            time=step * 0.1,
            evaluations=tuple(
                BindingEvaluation(
                    spec_name=manager,
                    binding=binding,
                    robustness=0.1,
                    status=EvaluationStatus.ACTIVE,
                    verdict_final=True,
                    timestep=step,
                    time=step * 0.1,
                )
                for manager, bindings in bindings_by_manager.items()
                for binding in bindings
            ),
        )

    first = tracker.update(result(0, {"m_a": [(0,)], "m_b": [(2,)]}))
    second = tracker.update(
        result(1, {"m_a": [(0,), (1,)], "m_b": [(2,)]})
    )
    _mailbox, _worker_done, snapshots = _mailbox_stream()
    app = run_textual_dashboard(snapshots, build_only=True)

    async def exercise():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.expanded.add("alpha")
            app._show(first, rebuild=True)
            app._show(second)
            await pilot.pause()
            order = list(app.rendered_rows)
            assert (
                order.index("binding:m_a:m_a::entity:1")
                < order.index("spec:beta")
            )
            table_keys = [
                str(row_key.value)
                for row_key in app.query_one("#run").rows
            ]
            assert table_keys == order
            await pilot.press("ctrl+q")

    asyncio.run(exercise())


def test_specification_level_active_filter_keeps_expanded_bindings():
    make_snapshot = _live_snapshots(binding_count=3)
    mailbox, _worker_done, snapshots = _mailbox_stream()
    mailbox.put(make_snapshot(0))
    app = run_textual_dashboard(snapshots, build_only=True)

    async def exercise():
        async with app.run_test() as pilot:
            await pilot.pause(0.2)
            await pilot.press("space")
            app.state_filter = "active"
            app._show(app.latest_snapshot, rebuild=True)
            assert app.query_one("#run").row_count == 4
            await pilot.press("ctrl+q")

    asyncio.run(exercise())


def test_headless_escape_reports_stopped_and_joins_driver():
    make_snapshot = _live_snapshots(binding_count=1)
    worker_done = Event()

    def snapshots(stop):
        step = 0
        try:
            while not stop.is_set():
                yield make_snapshot(step)
                step += 1
                stop.wait(0.01)
        finally:
            worker_done.set()

    result = run_textual_dashboard(snapshots, headless=True)

    assert result.stopped is True
    assert worker_done.is_set()


def test_slow_final_step_is_waited_out_not_failed():
    make_snapshot = _live_snapshots(binding_count=1)
    worker_done = Event()

    def snapshots(stop):
        step = 0
        try:
            while not stop.is_set():
                yield make_snapshot(step)
                step += 1
                stop.wait(0.01)
            time.sleep(1.5)
        finally:
            worker_done.set()

    result = run_textual_dashboard(snapshots, headless=True)

    assert result.stopped is True
    assert worker_done.is_set()


def test_second_stop_press_escalates_to_immediate_exit():
    make_snapshot = _live_snapshots(binding_count=1)

    def snapshots(stop):
        yield make_snapshot(0)
        stop.wait(10)
        # Simulate a slow final simulator step: the app must stay alive
        # after the first stop press and exit only on the second.
        time.sleep(2.0)

    app = run_textual_dashboard(snapshots, build_only=True)

    async def exercise():
        async with app.run_test() as pilot:
            await pilot.pause(0.2)
            await pilot.press("q")
            await pilot.pause()
            status = str(app.query_one("#status").render())
            assert "stopping" in status
            assert app.is_running
            await pilot.press("q")
            await pilot.pause()
            assert not app.is_running

    asyncio.run(exercise())


def test_post_exit_simulator_output_reaches_durable_log():
    make_snapshot = _live_snapshots(binding_count=1)
    simulator_log = StringIO()

    def snapshots(stop):
        yield make_snapshot(0)
        stop.wait(10)
        # Worker output after the App exits, during the join window,
        # must still reach the durable log.
        time.sleep(0.5)
        print("late simulator chatter")

    with redirect_stdout(StringIO()):
        run_textual_dashboard(
            snapshots,
            headless=True,
            simulator_stream=simulator_log,
        )

    assert "late simulator chatter" in simulator_log.getvalue()


def test_textual_exit_restores_stdout_and_copies_prints_to_simulator_log():
    make_snapshot = _live_snapshots(binding_count=1)
    simulator_log = StringIO()
    terminal = StringIO()

    def snapshots(stop):
        print("simulator notice")
        step = 0
        while not stop.is_set():
            yield make_snapshot(step)
            step += 1
            stop.wait(0.02)

    with redirect_stdout(terminal):
        result = run_textual_dashboard(
            snapshots,
            headless=True,
            simulator_stream=simulator_log,
        )
        print("episode summary")

    assert result.stopped is True
    assert "simulator notice" in simulator_log.getvalue()
    assert terminal.getvalue().endswith("episode summary\n")
