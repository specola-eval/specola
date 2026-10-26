"""
isaaclab.py — adapter for NVIDIA Isaac Lab environments (PhysX and Newton).

Isaac Lab is not MuJoCo-based, so this implements ``SimulatorAdapter``
directly rather than subclassing ``MujocoAdapter``: state comes from Isaac
Lab's backend-neutral asset API (``Articulation.data``) rather than from
``MjData``. The same code path serves both physics backends, since Isaac Lab
3.0 dispatches ``Articulation``/``RigidObject`` to per-backend
implementations behind one interface.

Two properties of Isaac Lab shape this adapter.

**It is vectorized.** An Isaac Lab environment is ``num_envs`` independent
copies of a scene, and every state array is shaped ``(num_envs, ...)``.
Specola monitors one scene, so an adapter instance is bound to one
``env_index`` and reports that environment only. Monitoring N environments
means N adapter instances and N engines. This is not a limitation to
apologise for: task randomisation is per-environment (FORGE randomises its
contact-force threshold per env), so a single monitor over env 0 would
silently describe one arbitrary sample rather than the population.

**Its arrays live on the GPU.** ``Articulation.data.*`` returns a
``ProxyArray`` over device memory. Measured on an RTX 5090, moving one to
numpy costs about 0.009 ms and does not differ between backends, so the
crossing is done eagerly here rather than being deferred behind a lazy
wrapper the monitor would have to understand.

Watching a vectorized run
-------------------------
Live visualization does not survive vectorization, and pretending otherwise
would mislabel data. Parallel environments share one viewport, so a frame
cannot be attributed to any single environment; ``camera_names`` therefore
reports nothing when ``num_envs > 1``. Nor would watching help: nobody reads
a thousand robustness traces as they scroll past.

The workflow a vectorized run actually supports is record-then-investigate.
Run the sweep with a monitor per environment and let every one write its
trace; find the episode worth looking at from those traces -- the violation,
the near-miss, the disagreement between backends -- then reproduce that one
configuration at ``num_envs=1``, where the viewport does correspond to the
scene and the dashboard's live view and 3D panel mean what they show.

One caveat belongs with that: Isaac Lab publishes no determinism guarantee
for the Newton backend, and the PhysX reproducibility documentation does not
cover this path either. A single-environment reproduction of a
vectorized episode should be treated as a close re-run rather than a replay
of the same trajectory, unless equality has been checked for the
configuration in question.

Semantic boundary
-----------------
Only declared engine state is extracted. Raw incoming joint wrenches are
emitted from ``root_view.get_link_incoming_joint_force()`` — the engine's
own quantity. Their reset-lifecycle freshness, the reason this surface was
withheld as issue #105, was probed on 2026-08-20 and the staleness
hypothesis refuted on the pinned certification configuration
(benchmarks/adapter_fidelity/evidence_pins/probes/). They must not be
substituted with task-processed sensor observations, which would couple
this adapter to task internals — FORGE, for instance, smooths and
perturbs its sensor wrench for sim-to-real transfer.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

import numpy as np

from ..adapter import MonitorObservation, PropertyMeta, SimulatorAdapter
from ..registry import register as _register
from . import isaac_state
from .isaaclab_extraction import (
    ExtractionPlan,
    IsaacLabExtractionStrategy,
    ScalarPlanExtractor,
)


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    """
    Materialise one Isaac Lab state array as numpy.

    Isaac Lab 3.0 hands back a ``ProxyArray`` over device memory on both the
    PhysX and Newton backends. ``torch`` is imported lazily because this
    module is imported by the registry at startup, and importing torch there
    would cost every ``specola`` invocation whether or not Isaac is used.
    """
    if value is None:
        return None

    try:
        import torch
    except ImportError:  # pragma: no cover - torch ships with Isaac Lab
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    # ``isaaclab.utils.warp.proxy_array.ProxyArray`` publishes the underlying
    # tensor as ``.torch`` (and the warp array as ``.warp``). Reach for that
    # explicitly rather than duck-typing a converter: ProxyArray forwards
    # unknown attribute access to the tensor, so a generic
    # ``getattr(value, "numpy")`` silently resolves to ``Tensor.numpy`` and
    # raises "can't convert cuda:0 device type tensor to numpy" -- the host
    # copy has to be arranged on this side.
    tensor = getattr(value, "torch", None)
    if (
        torch is not None
        and tensor is not None
        and isinstance(tensor, torch.Tensor)
    ):
        return tensor.detach().cpu().numpy()

    if torch is not None:
        # Covers anything exposing __torch_function__ or the CUDA array
        # interface without a ``.torch`` accessor.
        try:
            return torch.as_tensor(value).detach().cpu().numpy()
        except Exception:  # noqa: BLE001 - fall through to the numpy path
            pass

    return np.asarray(value)


def _token(name: str) -> str:
    """
    Sanitise a simulator name into a property-key token.

    Mirrors ``MujocoAdapter._property_token``'s contract -- lowercase,
    punctuation collapsed to underscores, a leading digit made legal -- so
    keys from the two adapter families are shaped the same way even though
    the field names they prefix differ.
    """
    cleaned: List[str] = []
    previous_underscore = False

    for character in name.strip().lower():
        if character.isalnum():
            cleaned.append(character)
            previous_underscore = False
        elif not previous_underscore:
            cleaned.append("_")
            previous_underscore = True

    token = "".join(cleaned).strip("_")

    if not token:
        raise ValueError(f"Simulator name {name!r} sanitises to an empty token")

    if token[0].isdigit():
        token = f"n_{token}"

    return token



from ._usd_scene import (
    MeshCollector,
    SCENE_COLLISION_GROUP as _SCENE_COLLISION_GROUP,
    SCENE_VISUAL_GROUP as _SCENE_VISUAL_GROUP,
    bake_transform as _bake_transform,
    count_meshes_outside,
    fan_triangulate as _fan_triangulate,
    link_prim_paths,
    pack_scene,
    stage_is_metres_z_up,
)


class IsaacLabAdapter(SimulatorAdapter):
    """
    Monitor one environment of a vectorized Isaac Lab scene.

    Args:
        env: An Isaac Lab environment. Either the ``gym`` wrapper or the
            unwrapped ``DirectRLEnv``/``ManagerBasedRLEnv`` is accepted; the
            wrapper is unwrapped here so callers need not care.
        env_index: Which parallel environment to report. Must be in
            ``range(num_envs)``.
        robot_class: Entity class reported for the articulated robot.
        object_class: Entity class reported for every other articulation.
    """

    # Renders to pump before accepting a black frame. Five sufficed in
    # measurement; the margin costs nothing because it runs once.
    _WARMUP_RENDERS = 12

    # The robodojo adapter, on the same Kit stack, deadlocked when a worker
    # thread stepped the stage while the main thread held the GIL (USD's
    # stage lock against the interpreter lock). Not reproduced on Isaac Lab;
    # stepping on the main thread costs nothing and rules it out.
    step_on_main_thread = True

    def __init__(
        self,
        env: Any,
        env_index: int = 0,
        robot_class: str = "robot",
        object_class: str = "object",
        extractor_strategy: Optional[IsaacLabExtractionStrategy] = None,
    ) -> None:
        super().__init__()

        self._env = getattr(env, "unwrapped", env)
        self.robot_class = robot_class
        self.object_class = object_class

        num_envs = int(getattr(self._env, "num_envs", 1))
        if not 0 <= env_index < num_envs:
            raise ValueError(
                f"env_index {env_index} is outside range(num_envs)="
                f"range({num_envs})"
            )
        self.env_index = int(env_index)

        self._step = 0
        self._render_warmed = False
        self._layout: Optional[List[list]] = None

        # Row plan for scene_geometry/scene_poses, built lazily by
        # _ensure_scene_plan and fixed for the adapter's lifetime.
        self._scene_plan: Optional[Dict[str, Any]] = None
        self._scene_plan_built = False

        # Scene key -> entity id, assigned once so ids are stable across the
        # episode. The base class rejects an id that changes class mid-episode.
        self._assets: Dict[str, Any] = {}
        self._entity_id_of: Dict[str, int] = {}
        self._class_of: Dict[str, str] = {}
        self._extractor_strategy = (
            ScalarPlanExtractor(_to_numpy, _token)
            if extractor_strategy is None
            else extractor_strategy
        )
        self._extraction_plan: ExtractionPlan
        self._collect_assets()

    # -- scene inventory ---------------------------------------------------

    def _collect_assets(self) -> None:
        """Compile documented scene-family classification and static layout once."""
        self._extraction_plan = self._extractor_strategy.compile_plan(
            self._env.scene, self.robot_class, self.object_class
        )
        self._assets = {
            entity.key: entity.asset for entity in self._extraction_plan.entities
        }
        self._entity_id_of = {
            entity.key: entity.entity_id for entity in self._extraction_plan.entities
        }
        self._class_of = {
            entity.key: entity.entity_class
            for entity in self._extraction_plan.entities
        }

    # -- adapter interface -------------------------------------------------

    @property
    def dt(self) -> float:
        """
        Seconds between observations.

        This is the environment step, not the physics step: Isaac Lab
        decimates physics inside one env step (FORGE runs physics at 120 Hz
        and steps the environment at 15 Hz), and observations arrive at the
        latter rate.
        """
        step_dt = getattr(self._env, "step_dt", None)
        if step_dt is not None:
            return float(step_dt)
        return float(self._env.physics_dt)

    @property
    def spatial_dims(self) -> int:
        return 3

    @property
    def supported_classes(self) -> FrozenSet[str]:
        return frozenset({self.robot_class, self.object_class})

    def make_advancer(self, mode: str, demo: Optional[str], seed: int):
        """
        Drive one episode for ``specola run``.

        Isaac Lab's ``step`` takes a ``torch`` tensor of shape
        ``(num_envs, action_dim)`` on the environment's device, so the
        shared robosuite/gymnasium fallback (which feeds NumPy arrays)
        cannot drive it. ``hold`` steps with zero actions; ``random``
        draws uniformly over ``single_action_space`` for every parallel
        environment and is refused when that box is unbounded, since a
        uniform sample over it does not exist. ``replay`` of a Specola
        state recording never reaches this method (the shared runner
        restores the recorded states itself); any other ``--demo`` is
        refused, since Isaac Lab has no demonstration format of its own
        that this adapter reads.
        """
        if demo is not None:
            raise SystemExit(
                f"--demo {demo}: the isaaclab adapter replays only a Specola "
                "state recording (<episode>_states.npz written by specola run)."
            )
        if mode == "replay":
            raise SystemExit(
                "--mode replay on the isaaclab adapter needs --demo with a "
                "Specola state recording (<episode>_states.npz)."
            )
        if mode not in ("hold", "random"):
            raise SystemExit(
                f"--mode {mode} is not supported by the isaaclab adapter"
            )
        import torch

        env = self._env
        space = getattr(env, "single_action_space", None)
        if space is None or getattr(space, "shape", None) is None:
            raise SystemExit(
                f"--mode {mode} needs an env exposing single_action_space"
            )
        num_envs = int(getattr(env, "num_envs", 1))
        shape = (num_envs,) + tuple(space.shape)
        device = getattr(env, "device", "cpu")
        if mode == "hold":
            def sample() -> np.ndarray:
                return np.zeros(shape, dtype=np.float32)
        else:
            low, high = np.asarray(space.low), np.asarray(space.high)
            if not (np.isfinite(low).all() and np.isfinite(high).all()):
                raise SystemExit(
                    "--mode random needs a bounded action space; this "
                    "task's single_action_space is unbounded, use --mode hold"
                )
            rng = np.random.default_rng(seed)

            def sample() -> np.ndarray:
                return rng.uniform(low, high, shape).astype(np.float32)

        done = False

        def advance(step: int) -> bool:
            nonlocal done
            if done:
                return False
            action = torch.as_tensor(sample(), device=device)
            _, _, terminated, truncated, _ = env.step(action)
            done = bool(
                torch.as_tensor(terminated).flatten()[self.env_index]
                or torch.as_tensor(truncated).flatten()[self.env_index]
            )
            return True

        return advance

    def _reset_impl(self) -> MonitorObservation:
        self._step = 0
        return self._observation()

    def _get_observation_impl(self) -> MonitorObservation:
        self._step += 1
        return self._observation()

    # -- state tier (replay) ------------------------------------------------

    def _state_refusal(self) -> Optional[str]:
        return isaac_state.gripper_refusal(
            self._env.scene, int(getattr(self._env, "num_envs", 1))
        )

    def _state_layout(self) -> List[list]:
        # The scene's assets are fixed once it is built, so is the layout.
        if self._layout is None:
            self._layout = isaac_state.layout(
                self._env.scene.get_state(is_relative=False)
            )
        return self._layout

    def state_signature(self) -> Optional[dict]:
        """The scene's ``get_state`` layout, configuration, layers and physics.

        Built by ``isaac_state.signature``: the leaf layout for one
        environment, a digest of ``env.cfg.to_dict()``, a record of every
        USD layer the stage uses and a digest of the monitored
        environment's realised masses, inertia tensors and shape
        materials, so a recording made on another task configuration,
        asset file or randomised draw is refused before a state is
        restored.

        None, with a warning naming the reason, when the scene has surface
        grippers and more than one environment: ``reset_to`` applies a
        gripper state to every environment, not only ``env_index``
        (``isaac_state.gripper_refusal``).
        """
        reason = self._state_refusal()
        if reason is not None:
            warnings.warn(f"isaaclab adapter: {reason}", RuntimeWarning, stacklevel=2)
            return None
        task = getattr(getattr(self._env, "spec", None), "id", None)
        return isaac_state.signature(
            self._env.scene, self._env.cfg.to_dict(), self.env_index, task
        )

    def capture_state(self) -> Optional[np.ndarray]:
        """``scene.get_state(is_relative=False)`` for ``env_index``, flat float64."""
        if self._state_refusal() is not None:
            return None
        return isaac_state.flatten(
            self._env.scene.get_state(is_relative=False), self.env_index
        )

    def restore_state(self, state: np.ndarray) -> None:
        """``scene.reset_to`` for ``env_index``, then a forward pass and a buffer update.

        ``reset_to`` writes the root poses and velocities and the joint
        positions and velocities, and also sets the recorded joint
        positions and velocities as the joint targets (Isaac Lab 6.1.17,
        ``interactive_scene.py:689``). ``sim.forward()`` updates the
        articulation kinematics without a physics step, and
        ``scene.update(0.0)`` refreshes the asset buffers, so the next
        observation reads the restored state.
        """
        reason = self._state_refusal()
        if reason is not None:
            raise NotImplementedError(f"isaaclab adapter: {reason}")
        scene = self._env.scene
        scene.reset_to(
            isaac_state.unflatten(self._state_layout(), state),
            env_ids=[self.env_index],
            is_relative=False,
        )
        self._env.sim.forward()
        scene.update(0.0)

    # -- extraction --------------------------------------------------------

    def _body_names(self, asset: Any) -> Tuple[str, ...]:
        return tuple(getattr(asset, "body_names", ()) or ())

    def property_metadata(
        self,
        entity_id: int,
    ) -> Dict[str, PropertyMeta]:
        metadata = self._extractor_strategy.metadata(
            self._extraction_plan, self.env_index, entity_id
        )
        accepted = self.last_observation
        if accepted is not None:
            entity = accepted.perceived_entities.get(entity_id)
            if entity is None:
                return {}
            metadata = {
                key: value
                for key, value in metadata.items()
                if key in entity.properties
            }
        return dict(self._project_property_metadata(entity_id, metadata))

    def set_extraction_scope(
        self, scope: Optional[Mapping[str, FrozenSet[str]]]
    ) -> None:
        """Prune Isaac native reads using the same scope as base projection."""
        super().set_extraction_scope(scope)
        setter = getattr(
            self._extractor_strategy, "set_plan_extraction_scope", None
        )
        if callable(setter):
            setter(self._extraction_plan, self._extraction_scope)
            return
        setter = getattr(self._extractor_strategy, "set_extraction_scope", None)
        if callable(setter):
            setter(self._extraction_scope)

    def _observation(self) -> MonitorObservation:
        return self._extractor_strategy.extract(
            self._extraction_plan,
            self.env_index,
            self._step,
            self._step * self.dt,
        )

    # -- visualization -----------------------------------------------------

    def _scene_cameras(self) -> Dict[str, Any]:
        """
        Scene sensors that render per environment.

        Isaac Lab's ``Camera`` is constructed with ``num_instances=num_envs``
        and renders every environment's view in one vectorized pass, so its
        output is indexed by environment. That is what makes a frame
        attributable here where the shared viewport is not: slicing
        ``output["rgb"][env_index]`` yields this environment's pixels rather
        than a picture of the whole grid.

        A camera has to be declared in the environment config before the
        scene is built, so this only finds one the caller asked for. The
        adapter does not add sensors: doing so would change what the
        simulator computes, which is outside what an adapter may do.
        """
        sensors = getattr(getattr(self._env, "scene", None), "sensors", None)
        if not sensors:
            return {}

        cameras: Dict[str, Any] = {}
        for name, sensor in dict(sensors).items():
            data = getattr(sensor, "data", None)
            if data is None:
                continue
            output = getattr(data, "output", None)
            if output is None:
                continue
            try:
                has_rgb = "rgb" in output
            except TypeError:  # pragma: no cover - defensive
                continue
            if has_rgb:
                cameras[str(name)] = sensor

        return cameras

    def camera_names(self) -> Tuple[str, ...]:
        """
        Named frame sources, empty unless a frame can honestly be attributed
        to *this* environment.

        Two conditions, and the second is about correctness rather than
        availability.

        The environment must be able to render at all: that needs
        ``enable_cameras`` at app launch *and* ``render_mode="rgb_array"`` at
        ``gym.make``, neither of which an adapter can switch on afterwards.
        Advertising a camera the caller did not enable would promise frames
        that never arrive.

        And the scene must hold a single environment. Isaac Lab lays parallel
        environments out in a grid sharing **one** viewport, so under
        ``num_envs > 1`` every adapter would return the same whole-grid image
        and the dashboard would file it against each monitor's timeline as
        though it were that environment's camera. Silently mislabelling other
        environments' pixels is worse than having no camera, so the viewport
        is offered only for a single-environment scene.

        A scene camera lifts that restriction, because Isaac Lab renders one
        per environment in a single vectorized pass and indexes the result by
        environment. Declare a ``CameraCfg`` in the environment config and
        every monitor gets its own correctly attributed view; scene cameras
        are therefore listed first, and the viewport appears only as a
        single-environment fallback.
        """
        names = tuple(sorted(self._scene_cameras()))

        if getattr(self._env, "render_mode", None) == "rgb_array" and (
            int(getattr(self._env, "num_envs", 1)) == 1
        ):
            names += ("viewport",)

        return names

    def capture_frame(
        self,
        camera: Optional[str] = None,
        *,
        width: int = 320,
        height: int = 240,
    ) -> Optional[np.ndarray]:
        """
        Render one HxWx3 uint8 RGB frame of the current scene.

        ``width``/``height`` are ignored: the Omniverse viewport renders at
        the resolution configured at app launch (1280x720 by default), and
        the contract allows returning a simulator's native size.

        **The first frames are black.** Omniverse's render pipeline needs
        several passes before the viewport produces pixels -- measured here,
        a frame taken after one step is uniformly zero while one taken after
        five is fully populated. Returning those would hand the dashboard a
        black video that looks like a dead camera rather than a warming one,
        so the first call pumps the renderer until pixels appear. The retry
        runs once per adapter; afterwards frames are taken as they come, and
        a genuinely black scene is reported honestly.
        """
        del width, height

        available = self.camera_names()
        if not available:
            return None

        selected = camera or available[0]

        cameras = self._scene_cameras()
        if selected in cameras:
            return self._capture_scene_camera(cameras[selected])

        if selected != "viewport":
            return None

        frame = self._render_once()

        if frame is not None and not self._render_warmed:
            for _ in range(self._WARMUP_RENDERS):
                if frame is not None and frame.any():
                    break
                frame = self._render_once()
            self._render_warmed = True

        return frame

    def _capture_scene_camera(self, sensor: Any) -> Optional[np.ndarray]:
        """
        This environment's slice of a per-environment camera.

        The leading axis is the environment, which is the whole reason a
        scene camera is attributable where the viewport is not. An output
        that does not carry that axis is refused rather than guessed at: a
        3-D array here would mean the camera is not per-environment, and
        indexing it would silently return a row of pixels as an image.
        """
        output = getattr(getattr(sensor, "data", None), "output", None)
        if output is None:
            return None

        try:
            rgb = output["rgb"]
        except (KeyError, TypeError):  # pragma: no cover - defensive
            return None

        frame = _to_numpy(rgb)
        if frame is None or frame.ndim != 4:
            return None

        if self.env_index >= frame.shape[0]:
            return None

        image = frame[self.env_index]
        if image.ndim != 3 or image.shape[2] < 3:
            return None

        image = image[:, :, :3]
        if image.dtype != np.uint8:
            # Isaac hands back float rgb in 0..1 for some configurations.
            scale = 255.0 if float(np.nanmax(image, initial=0.0)) <= 1.0 else 1.0
            image = np.clip(image * scale, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(image)

    def _render_once(self) -> Optional[np.ndarray]:
        try:
            rendered = self._env.render()
        except Exception:  # pragma: no cover - render config dependent
            return None

        if rendered is None:
            return None

        frame = np.asarray(rendered)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None

        return np.ascontiguousarray(frame[:, :, :3].astype(np.uint8))

    # -- introspection helpers --------------------------------------------

    @property
    def entity_labels(self) -> Dict[int, str]:
        """Scene key behind each entity id, for discovery output."""
        return {eid: key for key, eid in self._entity_id_of.items()}

    # ------------------------------------------------------------------
    # 3D pose panel (scene_geometry / scene_poses)
    # ------------------------------------------------------------------

    def _ensure_scene_plan(self) -> Optional[Dict[str, Any]]:
        """
        Build, once, the body-row plan shared by ``scene_geometry`` and
        ``scene_poses``.

        The plan fixes the table layout for the adapter's lifetime: row 0
        is a world anchor (as in the MuJoCo export), then every asset's
        links in ``_assets`` order, then the environment's static
        fixtures. Deciding viability exactly once is what upholds the
        base-class contract that the hooks never alternate between None
        and arrays within an episode: a scene that cannot support the
        panel is None from the first call and stays None.
        """
        if not self._scene_plan_built:
            self._scene_plan_built = True
            self._scene_plan = self._build_scene_plan()
        return self._scene_plan

    def _build_scene_plan(self) -> Optional[Dict[str, Any]]:
        stage = getattr(getattr(self._env, "scene", None), "stage", None)
        if stage is None:
            return None

        sources: List[Tuple[str, Any, Tuple[str, ...], Tuple[Optional[str], ...]]] = []
        for key, asset in self._assets.items():
            names = self._body_names(asset)
            positions = _to_numpy(getattr(asset.data, "body_pos_w", None))
            orientations = _to_numpy(getattr(asset.data, "body_quat_w", None))
            if (
                not names
                or positions is None
                or orientations is None
                or positions.ndim != 3
                or orientations.ndim != 3
                or positions.shape[1] != len(names)
                or orientations.shape[1] != len(names)
                or orientations.shape[2] != 4
            ):
                # Not every monitored scene entry carries body poses — a
                # declared CameraCfg lands in scene.keys() with a .data
                # of its own. Such entries simply have no rows in the 3D
                # scene; only a scene with nothing poseable refuses.
                continue
            sources.append(
                (key, asset, names, self._link_prim_paths(stage, asset, names))
            )
        if not sources:
            return None

        # A pose-bearing asset whose prims cannot be attributed at all —
        # no resolved link paths AND no configured root to exclude from
        # the statics sweep (rigid-object collections keep their roots
        # under cfg.rigid_objects, not cfg.prim_path) — would leak its
        # meshes into the statics as world-baked ghosts frozen at reset.
        # Refusing the whole panel is the honest degradation.
        for _, asset, _, paths in sources:
            if any(path is not None for path in paths):
                continue
            if not self._asset_root_patterns(asset):
                return None

        try:
            if not stage_is_metres_z_up(stage):
                return None
        except ImportError:  # pragma: no cover - pxr ships with Isaac Sim
            return None

        # Static fixtures: children of this environment's prim owned by
        # no asset (FORGE's Table). Their meshes are exported in world
        # coordinates against constant identity pose rows.
        statics: List[Tuple[str, str]] = []
        env_paths = getattr(self._env.scene, "env_prim_paths", None)
        env_path: Optional[str] = None
        if env_paths is not None and 0 <= self.env_index < len(env_paths):
            env_path = str(env_paths[self.env_index])
        if env_path is not None:
            env_prim = stage.GetPrimAtPath(env_path)
            if env_prim and env_prim.IsValid():
                # A prim belongs to a scene asset if any resolved link
                # path sits under it, or if the asset's own configured
                # root names it. The second condition matters twice: an
                # asset whose links all failed to resolve must degrade
                # to "no geometry", not reappear as a frozen static
                # ghost, and a pose-less scene entry (a declared
                # camera) must not become an empty static row.
                owned = [
                    path
                    for _, _, _, paths in sources
                    for path in paths
                    if path is not None
                ]
                for asset in self._assets.values():
                    owned.extend(self._asset_root_patterns(asset))
                for child in env_prim.GetChildren():
                    prefix = str(child.GetPath())
                    if any(
                        path == prefix or path.startswith(prefix + "/")
                        for path in owned
                    ):
                        continue
                    statics.append((child.GetName(), prefix))

        nbody = 1 + sum(len(names) for _, _, names, _ in sources) + len(statics)
        return {
            "sources": sources,
            "statics": statics,
            "nbody": nbody,
            "env_path": env_path,
        }

    def _asset_root_patterns(self, asset: Any) -> List[str]:
        """Env-resolved prim roots an asset's config claims.

        Ordinary assets declare one ``cfg.prim_path``; rigid-object
        collections declare one per member under ``cfg.rigid_objects``.
        These roots anchor the statics exclusion — a prim under any of
        them belongs to a monitored asset even when its per-body link
        paths could not be resolved.
        """
        cfg = getattr(asset, "cfg", None)
        patterns: List[str] = []
        root = getattr(cfg, "prim_path", None)
        if root:
            patterns.append(str(root))
        members = getattr(cfg, "rigid_objects", None)
        if members:
            values = (
                members.values() if hasattr(members, "values") else members
            )
            for member in values:
                member_root = getattr(member, "prim_path", None)
                if member_root:
                    patterns.append(str(member_root))
        return [
            pattern.replace("env_.*", f"env_{self.env_index}")
            for pattern in patterns
        ]

    def _link_prim_paths(
        self,
        stage: Any,
        asset: Any,
        names: Tuple[str, ...],
    ) -> Tuple[Optional[str], ...]:
        """Per-body USD prim paths, in body order (see ``_usd_scene``)."""
        return link_prim_paths(stage, asset, names, self.env_index)

    # ------------------------------------------------------------------
    # 3D pose panel (scene_geometry / scene_poses)
    # ------------------------------------------------------------------

    def _ensure_scene_plan(self) -> Optional[Dict[str, Any]]:
        """
        Build, once, the body-row plan shared by ``scene_geometry`` and
        ``scene_poses``.

        The plan fixes the table layout for the adapter's lifetime: row 0
        is a world anchor (as in the MuJoCo export), then every asset's
        links in ``_assets`` order, then the environment's static
        fixtures. Deciding viability exactly once is what upholds the
        base-class contract that the hooks never alternate between None
        and arrays within an episode: a scene that cannot support the
        panel is None from the first call and stays None.
        """
        if not self._scene_plan_built:
            self._scene_plan_built = True
            self._scene_plan = self._build_scene_plan()
        return self._scene_plan

    def _build_scene_plan(self) -> Optional[Dict[str, Any]]:
        stage = getattr(getattr(self._env, "scene", None), "stage", None)
        if stage is None:
            return None

        sources: List[Tuple[str, Any, Tuple[str, ...], Tuple[Optional[str], ...]]] = []
        for key, asset in self._assets.items():
            names = self._body_names(asset)
            positions = _to_numpy(getattr(asset.data, "body_pos_w", None))
            orientations = _to_numpy(getattr(asset.data, "body_quat_w", None))
            if (
                not names
                or positions is None
                or orientations is None
                or positions.ndim != 3
                or orientations.ndim != 3
                or positions.shape[1] != len(names)
                or orientations.shape[1] != len(names)
                or orientations.shape[2] != 4
            ):
                # Not every monitored scene entry carries body poses — a
                # declared CameraCfg lands in scene.keys() with a .data
                # of its own. Such entries simply have no rows in the 3D
                # scene; only a scene with nothing poseable refuses.
                continue
            sources.append(
                (key, asset, names, self._link_prim_paths(stage, asset, names))
            )
        if not sources:
            return None

        # A pose-bearing asset whose prims cannot be attributed at all —
        # no resolved link paths AND no configured root to exclude from
        # the statics sweep (rigid-object collections keep their roots
        # under cfg.rigid_objects, not cfg.prim_path) — would leak its
        # meshes into the statics as world-baked ghosts frozen at reset.
        # Refusing the whole panel is the honest degradation.
        for _, asset, _, paths in sources:
            if any(path is not None for path in paths):
                continue
            if not self._asset_root_patterns(asset):
                return None

        try:
            if not stage_is_metres_z_up(stage):
                return None
        except ImportError:  # pragma: no cover - pxr ships with Isaac Sim
            return None

        # Static fixtures: children of this environment's prim owned by
        # no asset (FORGE's Table). Their meshes are exported in world
        # coordinates against constant identity pose rows.
        statics: List[Tuple[str, str]] = []
        env_paths = getattr(self._env.scene, "env_prim_paths", None)
        env_path: Optional[str] = None
        if env_paths is not None and 0 <= self.env_index < len(env_paths):
            env_path = str(env_paths[self.env_index])
        if env_path is not None:
            env_prim = stage.GetPrimAtPath(env_path)
            if env_prim and env_prim.IsValid():
                # A prim belongs to a scene asset if any resolved link
                # path sits under it, or if the asset's own configured
                # root names it. The second condition matters twice: an
                # asset whose links all failed to resolve must degrade
                # to "no geometry", not reappear as a frozen static
                # ghost, and a pose-less scene entry (a declared
                # camera) must not become an empty static row.
                owned = [
                    path
                    for _, _, _, paths in sources
                    for path in paths
                    if path is not None
                ]
                for asset in self._assets.values():
                    owned.extend(self._asset_root_patterns(asset))
                for child in env_prim.GetChildren():
                    prefix = str(child.GetPath())
                    if any(
                        path == prefix or path.startswith(prefix + "/")
                        for path in owned
                    ):
                        continue
                    statics.append((child.GetName(), prefix))

        nbody = 1 + sum(len(names) for _, _, names, _ in sources) + len(statics)
        return {
            "sources": sources,
            "statics": statics,
            "nbody": nbody,
            "env_path": env_path,
        }

    def _asset_root_patterns(self, asset: Any) -> List[str]:
        """Env-resolved prim roots an asset's config claims.

        Ordinary assets declare one ``cfg.prim_path``; rigid-object
        collections declare one per member under ``cfg.rigid_objects``.
        These roots anchor the statics exclusion — a prim under any of
        them belongs to a monitored asset even when its per-body link
        paths could not be resolved.
        """
        cfg = getattr(asset, "cfg", None)
        patterns: List[str] = []
        root = getattr(cfg, "prim_path", None)
        if root:
            patterns.append(str(root))
        members = getattr(cfg, "rigid_objects", None)
        if members:
            values = (
                members.values() if hasattr(members, "values") else members
            )
            for member in values:
                member_root = getattr(member, "prim_path", None)
                if member_root:
                    patterns.append(str(member_root))
        return [
            pattern.replace("env_.*", f"env_{self.env_index}")
            for pattern in patterns
        ]

    def _link_prim_paths(
        self,
        stage: Any,
        asset: Any,
        names: Tuple[str, ...],
    ) -> Tuple[Optional[str], ...]:
        """
        Per-body USD prim paths, in body order.

        The physics view's ``link_paths`` is authoritative: it is in body
        order and disambiguates nested same-name prims (FORGE's fixed
        asset holds ``forge_hole_8mm/forge_hole_8mm``). The fallback
        matches body names against the asset's subtree and resolves only
        unique hits — a body whose prim cannot be identified gets no
        geometry rather than somebody else's.
        """
        for view_name in ("root_view", "root_physx_view"):
            view = getattr(asset, view_name, None)
            paths = getattr(view, "link_paths", None) if view is not None else None
            if not paths:
                continue
            if isinstance(paths[0], (list, tuple)):
                if 0 <= self.env_index < len(paths):
                    per_env = list(paths[self.env_index])
                    if len(per_env) == len(names):
                        return tuple(str(path) for path in per_env)
            break

        root_path = getattr(getattr(asset, "cfg", None), "prim_path", None)
        if not root_path:
            return tuple(None for _ in names)
        root_prim = stage.GetPrimAtPath(
            root_path.replace("env_.*", f"env_{self.env_index}")
        )
        if not (root_prim and root_prim.IsValid()):
            return tuple(None for _ in names)

        from pxr import Usd

        subtree = list(
            Usd.PrimRange(
                root_prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
            )
        )
        resolved: List[Optional[str]] = []
        for name in names:
            hits = [str(prim.GetPath()) for prim in subtree if prim.GetName() == name]
            resolved.append(hits[0] if len(hits) == 1 else None)
        return tuple(resolved)

    def scene_geometry(self) -> Optional[dict]:
        """
        Export this environment's USD meshes for the 3D panel.

        Walked once per episode from the plan's link prims (instance
        proxies included, so instanceable assets export like inlined
        ones). Each mesh's prim-to-link transform is baked into its
        vertices — the MuJoCo export does the same via the compiler — so
        geom offsets are identity and only body poses move per step.
        Static fixtures are baked in world coordinates against constant
        identity pose rows. Faces are fan-triangulated the way Hydra
        draws them; subdivision-surface prims export their base cage,
        which is the usual preview approximation. Collision hulls
        (purpose "guide") land in hidden group 0, visual meshes
        (purpose "default"/"proxy") in visible group 1; final-quality
        "render" prims, invisible prims, non-mesh gprims and scenery
        outside this environment's prim are counted in ``skipped``.
        """
        plan = self._ensure_scene_plan()
        if plan is None:
            return None

        stage = self._env.scene.stage
        bodies: List[Dict[str, Any]] = [{"name": "world", "parent": 0}]
        link_roots = {
            path
            for _, _, _, paths in plan["sources"]
            for path in paths
            if path is not None
        }
        collector = MeshCollector(link_roots)

        row = 1
        for key, _, names, paths in plan["sources"]:
            for name, path in zip(names, paths):
                bodies.append({"name": f"{key}/{name}", "parent": 0})
                link_prim = stage.GetPrimAtPath(path) if path else None
                if link_prim and link_prim.IsValid():
                    collector.collect(link_prim, row, link_prim)
                else:
                    collector.skipped["unresolved_link"] += 1
                row += 1
        for name, path in plan["statics"]:
            bodies.append({"name": name, "parent": 0})
            static_prim = stage.GetPrimAtPath(path)
            if static_prim and static_prim.IsValid():
                collector.collect(static_prim, row, None)
            row += 1

        env_path = plan["env_path"]
        if env_path is not None:
            outside = count_meshes_outside(stage, env_path)
            if outside:
                collector.skipped["outside_env"] = outside

        assert len(bodies) == plan["nbody"]
        return pack_scene(bodies, collector.collected, collector.skipped)

    def scene_poses(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Copy this step's body poses for the 3D panel.

        Rows follow the plan's table: world anchor, each asset's links
        from ``body_pos_w``/``body_quat_w`` sliced to this environment,
        then identity rows for static fixtures (their world transforms
        are baked into the exported vertices). Isaac Lab 6.x reports
        quaternions as ``(x, y, z, w)`` (``wp.quatf``); they are
        reordered to the ``(w, x, y, z)`` the recording format wants.
        ``np.concatenate`` allocates fresh float32 arrays, satisfying the
        owning-copy requirement on the simulator thread.
        """
        plan = self._ensure_scene_plan()
        if plan is None:
            return None

        index = self.env_index
        positions = [np.zeros((1, 3), dtype=np.float32)]
        orientations = [np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)]
        for key, asset, _, _ in plan["sources"]:
            # One device access per asset: body_link_pose_w is
            # wp.transformf, (num_envs, n, 7) as (px, py, pz, qx, qy,
            # qz, qw). The two-field read is the fallback for data
            # objects that do not expose the combined pose.
            pose = _to_numpy(getattr(asset.data, "body_link_pose_w", None))
            if pose is not None and pose.ndim == 3 and pose.shape[2] == 7:
                env_pose = pose[index]
                positions.append(np.asarray(env_pose[:, :3], dtype=np.float32))
                orientations.append(
                    np.asarray(env_pose[:, (6, 3, 4, 5)], dtype=np.float32)
                )
                continue
            body_pos = _to_numpy(asset.data.body_pos_w)
            body_quat = _to_numpy(asset.data.body_quat_w)
            if body_pos is None or body_quat is None:
                # Viability was established at plan time; a vanished pose
                # field is a broken invariant. Failing loudly lets the
                # recorder's fence disable the sink instead of this
                # adapter shifting every later pose ordinal.
                raise RuntimeError(
                    f"Isaac Lab asset {key!r} lost body_pos_w/body_quat_w "
                    "mid-episode"
                )
            positions.append(np.asarray(body_pos[index], dtype=np.float32))
            # Isaac Lab 6.x stores quaternions as wp.quatf (x, y, z, w);
            # the recording format wants (w, x, y, z).
            orientations.append(
                np.asarray(body_quat[index][:, (3, 0, 1, 2)], dtype=np.float32)
            )
        n_static = len(plan["statics"])
        if n_static:
            positions.append(np.zeros((n_static, 3), dtype=np.float32))
            static_quats = np.zeros((n_static, 4), dtype=np.float32)
            static_quats[:, 0] = 1.0
            orientations.append(static_quats)
        return np.concatenate(positions), np.concatenate(orientations)


def _kit_app_running() -> bool:
    """Whether this process already hosts a running Omniverse Kit app."""
    try:
        import omni.kit.app  # type: ignore

        app = omni.kit.app.get_app()
        return app is not None and app.is_running()
    except Exception:  # noqa: BLE001 - no importable Kit means no live app
        return False


def _launch_kit_app() -> Any:
    """
    Launch a headless Kit app the way Isaac Lab's scripts do.

    A process that hosts Kit owns three things an interactive host would
    otherwise provide (MISTAKES.md, "A process that hosts a Kit app must
    close it and keep it awake"): the app's close before the interpreter
    tears down, the hang detector off while the CLI idles Kit (``--web``
    serving after the episode), and the SIGINT handler, which Kit replaces
    with one that exits without closing.

    Kit parses the host's ``sys.argv`` as its own command line, and the
    CLI's (``discover --sim isaaclab --task ...``) are not Kit's, so the
    launch sees the program name alone. ``AppLauncher`` also resets
    ``sys.stdout`` to ``sys.__stdout__``, which would undo a caller's
    redirect (``discover --format json`` keeps stdout for the document),
    so both are restored afterwards.
    """
    import atexit
    import signal
    import sys

    from isaaclab.app import AppLauncher  # type: ignore

    argv, stdout = sys.argv, sys.stdout
    sys.argv = argv[:1]
    try:
        app = AppLauncher(
            headless=True, kit_args="--/app/hangDetector/enabled=false"
        ).app
    finally:
        sys.argv, sys.stdout = argv, stdout

    atexit.register(app.close)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return app


def _make_isaaclab(
    task: Optional[str],
    robot: Optional[str],
    target_only: bool,
    seed: Optional[int] = None,
) -> IsaacLabAdapter:
    """
    Build an Isaac Lab adapter for ``task``.

    ``seed`` is written to ``env_cfg.seed`` before the environment is
    built, where Isaac Lab seeds the generators its startup and reset
    events draw from (FORGE's shape friction and peg mass), so two
    processes with one seed realise the same physical parameters and a
    state recording from one replays in the other. ``None`` keeps the
    task's own seed setting.

    ``task`` is a registered Isaac Lab gym id, e.g.
    ``Isaac-Forge-PegInsert-Direct-v0``. ``robot`` is unused: Isaac Lab task
    ids already name their embodiment, so overriding it here would contradict
    the task registration rather than configure it.

    When no Kit app is running, the factory launches one headless and
    closes it at exit, so every CLI surface works without a launch script.
    A caller that already launched the app (library use) keeps it
    untouched. ``isaaclab_tasks`` is imported only after the app is up,
    because importing it registers the gym ids and needs Kit's modules.

    The environment is created with ``num_envs=1`` and env 0 is monitored.
    Monitoring several parallel environments is done by constructing one
    adapter per index against an environment the caller already owns, rather
    than through this factory, which exists for the single-scene CLI paths.
    """
    del robot, target_only

    if not task:
        raise ValueError(
            "The isaaclab adapter requires --task naming a registered Isaac "
            "Lab environment, e.g. Isaac-Forge-PegInsert-Direct-v0"
        )

    if not _kit_app_running():
        _launch_kit_app()

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401  - registers the task ids
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task, num_envs=1)
    if seed is not None:
        env_cfg.seed = seed
    env = gym.make(task, cfg=env_cfg)
    env.reset()

    return IsaacLabAdapter(env, env_index=0)


_register(
    name="isaaclab",
    factory=_make_isaaclab,
    description="NVIDIA Isaac Lab environments (PhysX and Newton backends)",
    install_hint=(
        "Install Isaac Sim and Isaac Lab into their own conda env; see "
        "docs."
    ),
    requires=("isaaclab",),
)
