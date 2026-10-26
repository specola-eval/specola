"""Run RoboDojo's policy evaluation client with every episode monitored.

RoboDojo evaluates a policy through a websocket client (XPolicyLab): the
policy server owns the model, the client (``src/eval_client/main.py`` in the
RoboDojo checkout) owns Isaac Sim and drives each policy's own
``eval_one_episode`` loop. This script is that client with specola
attached. It runs ``main.py`` unchanged in this process and wraps the
environment it builds at three points: after ``reset`` each parallel
environment gets a monitored session, every N physics steps each running
session is stepped, and after ``run_eval`` each session ends with the
success flag RoboDojo itself computed and a head-camera snapshot of the
terminal state. RoboDojo's own result files are written as usual; specola's
result files land beside them under ``--results-dir``.

Start the policy server first (its own conda env, see the policy README):

    cd <RoboDojo checkout>
    bash scripts/robodojo.sh server --policy-dir XPolicyLab/policy/demo_policy \\
        --task stack_bowls --ckpt external --policy-env demo-policy \\
        --policy-port 9999

Then this script, from the RoboDojo checkout, in the RoboDojo env:

    ROBODOJO_ROOT=$PWD EVAL_NUM=2 OMNI_KIT_ACCEPT_EULA=YES \\
      env -u PYTHONPATH PYTHONNOUSERSITE=1 \\
      ~/miniconda3/envs/RoboDojo/bin/python \\
      <specola>/examples/robodojo_monitored_eval.py \\
      --specs <specola>/examples/robodojo_specs.yaml \\
      --results-dir <specola>/results/robodojo-policies/demo_policy \\
      --task_name stack_bowls --policy_name demo_policy --port 9999

Every flag this script does not know is passed to ``main.py``; the ones
``scripts/eval_policy.sh`` would set are defaulted here the same way.

Monitoring cadence: by default each monitor update is taken where
RoboDojo evaluates its own success check -- inside ``RewardManager.step``,
once per policy action (10 physics steps of 4 ms), for exactly the
environments that call evaluates (still running, success flag set, check
list not yet passed) -- so every sample is the state the native checker
saw, and a session ends where RoboDojo stops checking. ``--stride N``
instead updates every N physics steps (the 2026-09-23/24 campaigns used
``--stride 10``); those samples are not tied to the native check.

Evidence: in the default cadence every native check also writes, per
environment and episode, ``evidence/ep<index>_env<env>_layout<id>.npz``
under ``--results-dir``: the action count, physics step count and wall
time of each check, the outcome of each native primitive
(``RewardManager.call_func_parser`` on each registered check, before
RoboDojo's own call on the same state), whether the check stage passed,
the RoboDojo-frame pose of every layout instance and every arm's
end-effector pose, and a JSON ``meta`` record (asset identities and
oriented-bbox vertices, the reset poses RoboDojo captured, the registered
check list, the entity labels, the specification file). Enough to
re-evaluate a corrected predicate offline without the simulator.

``--expect-labels`` names a JSON file of the entity bindings the
specification assumes, ``{"rigid_2": "basket", "rigid_*": ["doll0", ...]}``:
an id key must carry that label, a ``<class>_*`` key must hold exactly
those labels. An episode whose scene breaks the assumption is not
monitored (its session ends truncated at reset, with the reason printed);
its evidence is still written.

Known limits: RoboDojo's in-process restart on a PhysX fatal error
re-executes ``main.py`` directly, without this wrapper, so a restarted run
continues unmonitored; and ``--mode``-style specola drive options do not
apply, the policy drives.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from specola.session import Session


def _parse() -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--specs", required=True, help="specola spec YAML")
    ap.add_argument("--results-dir", required=True,
                    help="Where specola result files and snapshots go")
    ap.add_argument("--stride", type=int, default=None,
                    help="Physics steps between monitor updates; default: "
                         "one update at each native success check")
    ap.add_argument("--expect-labels", default=None,
                    help="JSON file of the entity labels the specs assume")
    ap.add_argument("--distmax", type=float, default=None,
                    help="Enable sampled pair distances with this cutoff (m)")
    ap.add_argument("--queries", default=None,
                    help="Query configuration YAML/JSON (robodojo.mesh_pair: "
                         "cutoff, robot links and the pairs to compute)")
    ap.add_argument("--snapshot-camera", default="cam_head")
    ap.add_argument("--run-id", default=None, help="specola run id (dir name)")
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--policy_name", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--env_cfg_type", default="arx_x5")
    ap.add_argument("--num_envs", type=int, default=None,
                    help="Override the sim config's parallel env count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--ckpt_name", default="external")
    ap.add_argument("--action_type", default="joint")
    return ap.parse_known_args()


def _robodojo_root() -> Path:
    root = os.environ.get("ROBODOJO_ROOT") or os.getcwd()
    if not (Path(root) / "src" / "eval_client" / "main.py").is_file():
        raise SystemExit(
            "Run from the RoboDojo checkout or set ROBODOJO_ROOT to it")
    return Path(root).resolve()


def _sim_num_envs(root: Path, env_cfg_type: str) -> int:
    import yaml

    cfg = yaml.safe_load((root / "env_cfg" / f"{env_cfg_type}.yml").read_text())
    sim = yaml.safe_load(
        (root / "env_cfg" / "sim" / f"{cfg['config']['sim']}.yml").read_text())
    return int(sim.get("scene", {}).get("num_envs", 1))


def _main_argv(args: argparse.Namespace, passthrough: list[str], root: Path) -> list[str]:
    kit_args = (
        "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera "
        "--/app/hangDetector/enabled=false"
    )
    num_envs = args.num_envs or _sim_num_envs(root, args.env_cfg_type)
    return [
        str(root / "src" / "eval_client" / "main.py"),
        "--task_name", args.task_name,
        "--env_cfg_type", args.env_cfg_type,
        "--num_envs", str(num_envs),
        "--enable_cameras",
        "--kit_args", kit_args,
        "--device_id", str(args.device_id),
        "--policy_name", args.policy_name,
        "--port", str(args.port),
        "--protocol", "ws",
        "--policy_server_url", f"ws://{args.host}:{args.port}",
        "--additional_info",
        f"ckpt_name={args.ckpt_name},action_type={args.action_type}",
        "--seed", str(args.seed),
        "--host", args.host,
        "--headless",
        *passthrough,
    ]


def _jsonable(value):
    """Registered check arguments as JSON: callables by name."""
    if callable(value):
        return f"<callable {getattr(value, '__name__', type(value).__name__)}>"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _label_problem(adapter, expected: dict) -> str:
    """Why this scene breaks the specification's entity assumptions, or ''."""
    observation = adapter.last_observation
    entities = observation.perceived_entities if observation is not None else {}
    labels = adapter.entity_labels
    for key, want in expected.items():
        cls, _, binding = key.rpartition("_")
        if binding == "*":
            got = sorted(labels.get(i, "") for i, e in entities.items()
                         if e.entity_class == cls)
            if got != sorted(want):
                return f"{key} holds {got}, specs assume {sorted(want)}"
            continue
        entity = entities.get(int(binding))
        if entity is None or entity.entity_class != cls or labels.get(int(binding)) != want:
            got = None if entity is None else (entity.entity_class, labels.get(int(binding)))
            return f"{key} is {got}, specs assume {want!r}"
    return ""


class _Evidence:
    """Raw inputs and native outcomes at each native check of one episode
    of one environment (see the module docstring, "Evidence")."""

    _LAYOUT_TYPES = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment")

    # Func_Parser predicates re-evaluated for the evidence: only ones that
    # read state and write none (checked against checkout 726e9aa; the
    # label callable select_label_by_zmin is pure too). Others -- is_moved
    # and is_not_moved with update=True, is_functional_point_moved,
    # update_object_state -- change RoboDojo's reward state when called,
    # so calling them again would change RoboDojo's own verdict; they,
    # and OR-groups (lists), are recorded as NaN.
    _PURE = frozenset({
        "is_axis_up", "is_stacked", "all_robot_back_to_origin", "is_robot_back_to_origin",
        "is_labels_axis_difference_in_range", "is_A_on_B_left", "is_A_on_B_right",
        "is_A_on_B_bottom", "is_A_cover_B", "is_lift", "is_not_lift",
    })

    def __init__(self, env, idx: int, meta: dict) -> None:
        import numpy as np

        self.env, self.idx, self.rows = env, idx, []
        layout = env.scene_manager.layout_manager
        parser = env.reward_manager.func_parser
        self.instances = []
        objects = {}
        for layout_type in self._LAYOUT_TYPES:
            for record in layout.get_layout_records(idx, layout_type) or ():
                name = record.get("inst_name")
                if not name:
                    continue
                self.instances.append(name)
                data = layout.get_instance_metadata(idx, inst_name=name) or {}
                try:
                    box = np.asarray(layout.get_instance_bbox_vertices(inst_name=name, env_idx=idx),
                                     dtype=float).reshape(-1, 3).tolist()
                except Exception:
                    box = None
                objects[name] = {
                    "label": record.get("label"), "layout_type": layout_type,
                    "instance_type": layout.instance_type_by_env[idx].get(name),
                    "model_name": data.get("model_name"), "model_id": data.get("model_id"),
                    "oriented_bbox_vertices": box,
                }
        self.arms = [r for r in env.robot_manager.robot_list]
        pre = parser.pre_state[idx]
        self.meta = dict(meta)
        self.meta.update({
            "objects": objects,
            "reset_pose": {n: np.asarray(v["pose"], dtype=float).tolist()
                           for n, v in pre.items() if v.get("pose") is not None},
            "ee_home": {a: np.asarray(v, dtype=float).tolist()
                        for a, v in parser.robot_origin_endpose[idx].items()},
            "robots": [{"arm_name": r.arm_name, "type": getattr(r, "type", None),
                        "ee_link_name": getattr(r, "ee_link_name", None)} for r in self.arms],
            "physics_dt": float(env.dt),
            "collect_interval": float(env.obs_manager.collect_interval),
            "instances": self.instances,
            "errors": [],
        })

    def _pose(self, name):
        import numpy as np

        result = self.env.scene_manager.layout_manager.get_instance_pose(self.idx, inst_name=name)
        if not isinstance(result, tuple) or result[0] is None or result[1] is None:
            return [float("nan")] * 7
        return np.concatenate([np.asarray(r.cpu() if hasattr(r, "cpu") else r, dtype=float).reshape(-1)[:n]
                               for r, n in zip(result, (3, 4))]).tolist()

    def outcomes(self):
        """Each registered check's native value, and the label a callable
        label resolved to, on the current state (before RoboDojo's call)."""
        rm = self.env.reward_manager
        checks = rm.check_list[self.idx][0]
        if "checks" not in self.meta:
            self.meta["checks"] = [[c[0], _jsonable(c[1])] if isinstance(c, tuple) else ["<or-group>", _jsonable(c)]
                                   for c in checks]
            # RewardManager.step runs its query stage first, which can fail
            # an environment before the check stage; with queries present
            # the sampled set can include an environment the check skipped.
            self.meta["queries_registered"] = bool(rm.query_list[self.idx] or rm.trigger_query_list[self.idx])
        values, selected = [], []
        for check in checks:
            if not isinstance(check, tuple) or check[0] not in self._PURE:
                values.append(float("nan"))
                selected.append(None)
                continue
            func_name, args = check
            values.append(float(rm.call_func_parser((func_name, args), self.idx)))
            label = args.get("label") if isinstance(args, dict) else None
            selected.append(rm.func_parser._select_label({
                "env_idx": self.idx, "label": label, "label_args": args.get("label_args"),
            }) if callable(label) else None)
        return values, selected

    def record(self, values, selected, passed, physics_steps, monitor_step):
        import time

        import numpy as np

        endposes = [np.asarray(self.env.robot_manager.get_real_endpose(r, env_idx_list=[self.idx])[self.idx],
                               dtype=float).tolist() for r in self.arms]
        self.rows.append({
            "action": int(self.env.take_action_cnt[self.idx]),
            "physics_steps": int(physics_steps), "wall_time": time.time(),
            "monitor_step": int(monitor_step), "stage_pass": bool(passed),
            "primitive": values, "selected_label": selected,
            "pose": [self._pose(n) for n in self.instances], "ee_pose": endposes,
        })

    def save(self, path, **final) -> None:
        import json

        import numpy as np

        self.meta.update(final)
        rows = self.rows
        n_checks = len(self.meta.get("checks", []))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            meta=np.array(json.dumps(self.meta)),
            action=np.array([r["action"] for r in rows], dtype=np.int64),
            physics_steps=np.array([r["physics_steps"] for r in rows], dtype=np.int64),
            wall_time=np.array([r["wall_time"] for r in rows], dtype=float),
            monitor_step=np.array([r["monitor_step"] for r in rows], dtype=np.int64),
            stage_pass=np.array([r["stage_pass"] for r in rows], dtype=bool),
            primitive=np.array([r["primitive"] for r in rows], dtype=float).reshape(len(rows), n_checks),
            selected_label=np.array(json.dumps([r["selected_label"] for r in rows])),
            pose=np.array([r["pose"] for r in rows], dtype=float).reshape(len(rows), len(self.instances), 7),
            ee_pose=np.array([r["ee_pose"] for r in rows], dtype=float).reshape(len(rows), len(self.arms), 7),
        )


def _attach(env, args: argparse.Namespace, monitors, results_dir: Path) -> None:
    """Hook one built EvalEnv: sessions per parallel env, stepped at each
    native success check (or every ``--stride`` physics steps).

    ``monitors(idx)`` returns the Monitor for parallel env ``idx``: a
    Monitor holds one open session and ends it when another is watched,
    so the N environments RoboDojo steps together each need their own.

    Nothing here may end RoboDojo's evaluation: the native calls run
    unguarded and exactly once; every monitor and evidence call is
    contained, ending that environment's session as truncated (or
    recording the error in its evidence) and letting RoboDojo continue.
    The exceptions are reading ``--expect-labels`` and ``--queries`` when
    the environment is built, which fail the run before any episode.
    """
    import json
    import traceback

    from specola.adapters.robodojo import RoboDojoAdapter

    queries = None
    if args.queries:
        from specola.query_config import load_query_configuration

        queries = load_query_configuration(args.queries)
    expected = json.loads(Path(args.expect_labels).read_text()) if args.expect_labels else {}
    native = args.stride is None

    sessions: dict[int, "Session"] = {}
    adapters: dict[int, RoboDojoAdapter] = {}
    evidence: dict[int, _Evidence] = {}
    counters: dict[int, int] = {}
    physics: dict[int, int] = {}
    monitor_steps: dict[int, int] = {}
    layouts: dict[int, int] = {}
    episode = {"n": 0}
    physics_dt = float(getattr(env, "dt", 0.004))
    if native:
        dt = physics_dt * float(env.obs_manager.collect_interval)
    else:
        dt = physics_dt * args.stride
    rm = env.reward_manager

    orig_reset, orig_step, orig_run_eval, orig_close = (
        env.reset, env.step, env.run_eval, env.close)
    orig_reward_step = rm.step

    def contained(idx: int, what: str, exc: BaseException) -> None:
        # A predicate meeting a non-finite value (2026-09-24, a bowl's
        # quaternion rounding acos's argument below -1) or any other
        # monitor failure: this environment's session ends as truncated
        # with the reason; RoboDojo keeps scoring it its own way.
        print(f"[monitor] env {idx}: {what} failed: {exc!r}; session ends truncated")
        session = sessions.get(idx)
        if session is not None and not session.ended:
            try:
                session.end(success=None, status="truncated")
            except Exception:
                traceback.print_exc()

    def end_all(status: str) -> None:
        for idx, session in list(sessions.items()):
            if not session.ended:
                try:
                    session.end(success=None, status=status)
                except Exception as exc:
                    traceback.print_exc()
                    print(f"[monitor] env {idx}: session end failed: {exc!r}")
        if evidence:
            # A batch RoboDojo abandoned (reset or close mid-episode):
            # RoboDojo does not score it, so its evidence is not kept.
            print(f"[monitor] {len(evidence)} unfinished evidence records dropped ({status})")
        sessions.clear()
        evidence.clear()

    def reset(seed=None, options=None):
        end_all("truncated")
        out = orig_reset(seed=seed, options=options)
        for idx, layout in env.current_env_seed_map.items():
            if env.end_flag[idx]:
                # Unstable at settle or an invalid layout: RoboDojo ends it
                # before the episode and never scores it (run_eval's
                # exist_envs), so it is not an episode here either.
                print(f"[monitor] env {idx}: layout {layout} ended by RoboDojo at reset; not monitored")
                continue
            layouts[idx] = int(layout)
            counters[idx] = physics[idx] = monitor_steps[idx] = 0
            meta = {"task": args.task_name, "policy": args.policy_name,
                    "layout_id": int(layout), "env_idx": int(idx),
                    "run_id": os.environ.get("ROBODOJO_RUN_ID", "")}
            try:
                # dt on the adapter too, so the result file's time column and
                # the spec windows share the monitor cadence.
                adapter = RoboDojoAdapter(env, env_index=idx, dt=dt, distmax=args.distmax)
                adapters[idx] = adapter
                sessions[idx] = monitors(idx).watch(adapter, dt=dt, queries=queries, meta=meta)
            except Exception as exc:
                traceback.print_exc()
                contained(idx, "session start", exc)
                continue
            try:
                problem = _label_problem(adapter, expected) if expected else ""
            except Exception as exc:
                problem = f"label check failed: {exc!r}"
            if native:
                try:
                    import specola

                    evidence[idx] = _Evidence(env, idx, {
                        **meta, "specs": str(Path(args.specs).resolve()),
                        "specola_file": specola.__file__,
                        "entity_labels": {str(k): v for k, v in adapter.entity_labels.items()},
                        "entity_classes": {str(k): e.entity_class for k, e in
                                           (adapter.last_observation.perceived_entities
                                            if adapter.last_observation is not None else {}).items()},
                        "label_problem": problem,
                    })
                except Exception as exc:
                    traceback.print_exc()
                    print(f"[monitor] env {idx}: evidence start failed: {exc!r}")
            if problem:
                contained(idx, "entity label check", ValueError(problem))
        return out

    def step(env_idx_list, decimation=1):
        out = orig_step(env_idx_list, decimation=decimation)
        for idx in env_idx_list:
            if idx in physics:
                physics[idx] += decimation
            if native:
                continue
            session = sessions.get(idx)
            if session is None or session.ended:
                continue
            counters[idx] += decimation
            if counters[idx] >= args.stride:
                counters[idx] = 0
                try:
                    session.step()
                except Exception as exc:
                    contained(idx, "monitor step", exc)
        return out

    def reward_step(env_idx_list=None):
        ids = range(env.num_envs) if env_idx_list is None else env_idx_list
        # Exactly the environments RewardManager.step evaluates.
        checked = [i for i in ids if rm.func_parser._check_env_success(i)
                   and len(rm.check_list[i]) > 0]
        stages = {i: len(rm.check_list[i]) for i in checked}
        before = {}
        for idx in checked:
            if idx in evidence:
                try:
                    before[idx] = evidence[idx].outcomes()
                except Exception as exc:
                    evidence[idx].meta["errors"].append(f"outcomes: {exc!r}")
        out = orig_reward_step(env_idx_list)
        for idx in checked:
            monitor_step = -1
            session = sessions.get(idx)
            if session is not None and not session.ended:
                try:
                    session.step()
                    monitor_step = monitor_steps[idx]
                    monitor_steps[idx] += 1
                except Exception as exc:
                    contained(idx, "monitor step", exc)
            if idx in evidence and idx in before:
                values, selected = before[idx]
                try:
                    evidence[idx].record(values, selected, len(rm.check_list[idx]) < stages[idx],
                                         physics[idx], monitor_step)
                except Exception as exc:
                    evidence[idx].meta["errors"].append(f"record: {exc!r}")
        return out

    def run_eval():
        out = orig_run_eval()
        index = episode["n"]
        for idx, session in list(sessions.items()):
            unstable = idx in env.unstable_envs
            if not session.ended:
                try:
                    # The snapshot is a courtesy; the verdict below is not.
                    frame = adapters[idx].capture_frame(args.snapshot_camera)
                    if frame is not None:
                        from PIL import Image

                        name = f"episode_{index:04d}_env{idx}_layout{layouts[idx]}.png"
                        Image.fromarray(frame).save(results_dir / name)
                except Exception as exc:
                    print(f"[monitor] snapshot failed for env {idx}: {exc}")
            # end() on an ended session returns its record unchanged; one
            # whose own end() failed inside contained() has none and
            # raises, which must not reach RoboDojo's run_eval.
            try:
                record = session.end(
                    success=None if unstable else bool(env.success[idx]),
                    status="truncated" if unstable else "completed")
            except Exception as exc:
                traceback.print_exc()
                print(f"[monitor] env {idx}: session end failed: {exc!r}")
                record = None
            if idx in evidence:
                try:
                    evidence[idx].save(
                        results_dir / "evidence" / f"ep{index:04d}_env{idx}_layout{layouts[idx]}.npz",
                        episode_index=index, native_success=bool(env.success[idx]),
                        unstable=unstable,
                        session_status=None if record is None else record.status,
                        episode_id=None if record is None else record.episode_id,
                        result_file=None if record is None or record.path is None else str(record.path),
                    )
                except Exception as exc:
                    traceback.print_exc()
                    print(f"[monitor] env {idx}: evidence save failed: {exc!r}")
            index += 1
        episode["n"] = index
        sessions.clear()
        evidence.clear()
        return out

    def close():
        end_all("truncated")
        orig_close()

    env.reset, env.step, env.run_eval, env.close = reset, step, run_eval, close
    if native:
        rm.step = reward_step


def main() -> None:
    args, passthrough = _parse()
    root = _robodojo_root()
    os.chdir(root)
    # What scripts/eval_policy.sh puts on PYTHONPATH: the checkout for
    # env/task/utils, and XPolicyLab for the websocket client.
    for entry in (root / "XPolicyLab", root):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    os.environ.setdefault("ROBODOJO_ROOT", str(root))

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    main_py = _main_argv(args, passthrough, root)
    sys.argv = main_py
    # main.py launches the Omniverse app and parses argv at import; run it
    # under a name that is not __main__ so its main() waits for us.
    ns = runpy.run_path(main_py[0], run_name="robodojo_eval_client")

    from specola import Monitor

    run_id = args.run_id or results_dir.name
    per_env: dict[int, Monitor] = {}

    def monitors(idx: int) -> Monitor:
        # All write into one directory; result file names carry the
        # episode and a random salt, so parallel envs never collide.
        if idx not in per_env:
            per_env[idx] = Monitor(args.specs, results_dir=str(results_dir.parent),
                                   run_id=run_id)
        return per_env[idx]

    # run_path hands back a copy of the module globals; main() resolves
    # names through its own __globals__, so that is where the swap goes.
    module_globals = ns["main"].__globals__
    build = module_globals["create_eval_env"]

    def create_eval_env(config, app, resume_state=None, **kwargs):
        env = build(config, app, resume_state=resume_state, **kwargs)
        _attach(env, args, monitors, results_dir)
        return env

    module_globals["create_eval_env"] = create_eval_env
    ns["main"]()


if __name__ == "__main__":
    main()
