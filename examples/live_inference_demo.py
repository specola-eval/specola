"""Live policy inference on LIBERO with specola monitoring — demo script.

The two-process setup the docs describe (docs/python-api.md): a policy
server owns the GPU and the model; this process owns the simulator and
the monitor. Terminal shows per-step robustness margins; --web serves
the dashboard (camera feed, binding tree, robustness plots) while the
policy drives the robot.

Start the policy server first (openvla-oft env, one GPU):

    cd /path/to/specola
    conda run -n openvla-oft python benchmarks/x7_oft_server.py \
        --suite libero_goal --port 8001 \
        --checkpoint <path> --policy-repo <openvla-oft repo>

Then this script (libero env):

    env -u PYTHONPATH PYTHONNOUSERSITE=1 \
      ~/miniconda3/envs/libero/bin/python examples/live_inference_demo.py \
      --suite libero_goal --task 0 --port 8001 \
      --specs <spec file for the task> --web 0

The r8 corrected spec pack (libero-v2-freeze-2026-08-17-r8/core/) binds
only against the campaign branch's adapter — prepend
PYTHONPATH=<workspace>/specola-worktrees/libero-v2-faithfulness/src for
those specs. Simpler spec files run on the current branch as-is.

Episode semantics mirror benchmarks/x7_rollout.py (the campaign client):
env.seed(0), init state --trial, 10 dummy-action settle steps, OFT's
full 8-action chunk open-loop.
"""

import argparse
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280,
             "libero_goal": 300, "libero_10": 520}
NUM_STEPS_WAIT = 10
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def build_oft_element(obs, prompt):
    # benchmarks/x7_oft_server.py contract: raw LIBERO obs keys; the
    # server does its own flipping, resizing, and state assembly.
    return {
        "agentview_image": obs["agentview_image"],
        "eye_in_hand_image": obs["robot0_eye_in_hand_image"],
        "eef_pos": obs["robot0_eef_pos"],
        "eef_quat": obs["robot0_eef_quat"],
        "gripper_qpos": obs["robot0_gripper_qpos"],
        "prompt": prompt,
    }


def make_dashboard(session, run_dir, run_id, task_name, port,
                   simulator="libero"):
    """Wire the CLI's dashboard onto an API-driven session.

    Returns (sink, server, tracker); feed tracker.update(result.raw) to
    sink.on_step every step. Construction mirrors run_monitor.py.
    """
    # ponytail: tracker construction copied from run_monitor.py rather
    # than refactored into a shared helper; fold into core if a second
    # script needs it.
    from specola.dashboard.recorder import WebRecorderSink
    from specola.dashboard.server import DashboardServer
    from specola.run_state import (
        BindingEntity, MonitorDefinition, RunStateTracker,
    )

    engine = session.engine
    authored = {r["name"]: r for r in engine.specification_metadata()}
    definitions = {}
    for manager_name, manager in engine.managers.items():
        specification = engine.manager_specs().get(manager_name, manager_name)
        spec_record = authored.get(specification, {})
        formula = manager.formula
        definitions[manager_name] = MonitorDefinition(
            manager_name=manager_name,
            specification=specification,
            authored_formula=str(spec_record.get("formula", "")),
            resolved_formula=formula.desugared_text or formula.name,
            backend=formula.backend,
            window_steps=formula.window_steps,
            entity_classes=tuple(
                str(getattr(p.entity_class, "value", p.entity_class))
                for p in formula.params
            ),
            components_by_slot=tuple(
                tuple(p.required_components) for p in formula.params
            ),
        )
    adapter = session.adapter
    entity_labels = getattr(adapter, "entity_labels", {}) or {}
    roles = getattr(adapter, "entity_roles", {}) or {}
    declared = getattr(adapter, "entity_declared_names", {}) or {}
    detail_ids = set(entity_labels) | set(roles) | set(declared)
    entity_details = {
        eid: BindingEntity(
            entity_id=eid, entity_class="",
            label=entity_labels.get(eid), role=roles.get(eid),
            role_basis=(getattr(adapter, "entity_role_basis", {}) or {}).get(eid),
            role_provenance=(getattr(adapter, "entity_role_provenance", {})
                             or {}).get(eid),
            declared_name=declared.get(eid),
        )
        for eid in detail_ids
    }
    meta_cb = getattr(adapter, "property_metadata", None)
    component_metadata = {}
    if callable(meta_cb):
        for eid in detail_ids:
            try:
                component_metadata[eid] = meta_cb(eid) or {}
            except Exception:
                component_metadata[eid] = {}
    tracker = RunStateTracker(
        engine.manager_specs(),
        skipped_specs=engine.skipped_specs(),
        labels=entity_labels,
        definitions=definitions,
        entity_details=entity_details,
        component_metadata=component_metadata,
    )
    sink = WebRecorderSink(
        run_dir, run_id=run_id, simulator=simulator, task=task_name,
        mode="live-inference", cameras=None, size=(640, 480),
        stride=1, video="auto",
    )
    server = DashboardServer(sink.recording_dir, mode="live")
    # older checkouts predate the discovery-snapshot callback
    discovery_cb = getattr(server, "set_discovery_snapshot", None)
    try:
        sink.set_callbacks(server.on_line, server.on_episode_end,
                           discovery_cb)
    except TypeError:
        sink.set_callbacks(server.on_line, server.on_episode_end)
    url = server.start(port)
    set_url = getattr(sink, "set_dashboard_url", None)
    if callable(set_url):
        set_url(url)
    print(f"  Web dashboard: {url}", flush=True)
    return sink, server, tracker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(MAX_STEPS),
                    default="libero_goal")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--trial", type=int, default=0,
                    help="init-state index")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--web", type=int, default=None,
                    help="dashboard port; 0 picks a free one")
    ap.add_argument("--out", default=None,
                    help="run directory (default: a temp dir)")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    from specola import Monitor

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(args.task)
    prompt = str(task.language)
    bddl = str(Path(get_libero_path("bddl_files"))
               / task.problem_folder / task.bddl_file)

    client = websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port)
    print(f"policy server: {client.get_server_metadata()}", flush=True)

    env = OffScreenRenderEnv(bddl_file_name=bddl,
                             camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(args.task)
    obs = env.set_init_state(init_states[args.trial])
    control_freq = env.env.control_freq

    run_id = f"demo_{args.suite}_task{args.task:02d}"
    run_dir = Path(args.out) if args.out else \
        Path(tempfile.mkdtemp(prefix="specola_demo_")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    monitor = Monitor(specs=args.specs, results_dir=str(run_dir),
                      run_id=run_id)
    session = monitor.watch(env, task=f"{args.suite}:{args.task}",
                            dt=1.0 / control_freq)

    sink = server = tracker = None
    if args.web is not None:
        sink, server, tracker = make_dashboard(
            session, run_dir, run_id, prompt, args.web)

    print(f'task: "{prompt}"', flush=True)
    queue = deque()
    success, t = False, 0
    try:
        while t < MAX_STEPS[args.suite] + NUM_STEPS_WAIT:
            if t < NUM_STEPS_WAIT:
                obs, _r, done, _i = env.step(DUMMY_ACTION)
            else:
                if not queue:
                    chunk = client.infer(build_oft_element(obs, prompt))
                    queue.extend(np.asarray(chunk["actions"]))
                obs, _r, done, _i = env.step(
                    np.asarray(queue.popleft()).tolist())
            result = session.step()
            if tracker is not None:
                sink.on_step(session.adapter, tracker.update(result.raw))
            worst = sorted(
                ((ev.spec_name, ev.robustness)
                 for ev in result.raw.evaluations
                 if ev.robustness is not None),
                key=lambda x: x[1])[:3]
            line = "  ".join(f"{n}: {r:+.4f}" for n, r in worst)
            print(f"step {t:3d}  {line}", flush=True)
            if done:
                success = True
                break
            t += 1
    finally:
        session.end(success=success, status="completed")
        if sink is not None:
            sink.finalize_episode("completed")
        print(f"success={success}  results in {run_dir}", flush=True)
        if server is not None:
            try:
                input("dashboard still serving — press Enter to exit ")
            except EOFError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
