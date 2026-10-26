"""
pybullet.py — PyBullet physics engine adapter
=============================================
The first non-MuJoCo engine integration (issue #33): a direct
``SimulatorAdapter`` implementation against the ``pybullet`` client API,
not a ``MujocoAdapter`` subclass.

One entity per PyBullet body. Per-entity quantities that exist exactly
once per body use entity-generic keys — the body's identity is already
carried by the entity, so embedding its name in the key would only
fragment the vocabulary across instances:

    basepos_0..2, basequat_0..3
        Base pose from ``getBasePositionAndOrientation``. Quaternion
        components use PyBullet's (x, y, z, w) order.
    basevel_0..2, baseangvel_0..2
        Base velocity from ``getBaseVelocity``.
    contactforce
        Sum of contact normal forces involving the body, from
        ``getContactPoints``. Zero until the engine has stepped —
        PyBullet reports contacts computed by the most recent
        ``stepSimulation``.

Because these keys are identical across instances, one spec like
``G[0,5s](robot_*[basepos_2] >= 0.1)`` binds every robot in the scene,
including several copies of the same URDF — no per-instance
configuration. Quantities that occur several times per body keep the
sim-native ``<field>_<token>[_<index>]`` shape, with tokens sanitized
from the names PyBullet reports (``getJointInfo`` bytes fields):

    jointpos_<joint>, jointvel_<joint>, jointtorque_<joint>
        Single-DOF joint state from ``getJointState`` (revolute and
        prismatic; fixed joints carry no state, spherical/planar
        multi-DOF joints are not extracted).
    linkpos_<link>_0..2
        World link center-of-mass positions from ``getLinkState`` with
        ``computeForwardKinematics=1`` for every named link, including
        children of fixed joints, so they describe the current joint
        positions.

Entity classification maps body names through ``body_rules`` (first
matching ``fnmatch`` pattern wins). Bodies matching no rule default to
``"robot"`` when articulated (at least one joint) and ``"object"``
otherwise.

Stepping ownership: the adapter never steps the simulation. Wrap an
existing client and keep calling ``stepSimulation()`` yourself; the CLI
factory owns its client and drives it through ``make_advancer``.

Fixed-step assumption: ``dt`` is read once from the engine's
``fixedTimeStep`` at construction. Changing the timestep mid-episode is
not supported.

Metadata units below use Specola's explicit consistent-SI declaration for
this adapter surface; PyBullet does not globally mandate SI. Frames,
quaternion convention, angular units, and contact scalar sign come from the
version-matched PyBullet producer contract.

Schema: bodies, joints, and links are enumerated once per ``reset()``;
bodies added to the client afterwards are picked up at the next reset,
not mid-episode (property-key schema is fixed once an episode begins).
Bodies removed mid-episode (``removeBody``) simply disappear from
observations — the binding lifecycle treats that as entity absence. A
removed uid that PyBullet reuses for a *differently named* body stays
absent (the adapter re-checks the body name every snapshot); reuse by a
body with the identical name is indistinguishable and reported under
the original identity.

Usage
-----
    specola discover    --sim pybullet --task path/to/scene.urdf
    specola run --sim pybullet --task "arm.urdf@fixed,box.urdf" \\
                    --specs my_specs.yaml --mode hold

In your own loop, wrap the client you already stepped::

    adapter = PyBulletAdapter(client_id)

Hardware rendering (``PYBULLET_EGL=1``) is wired up by the CLI factory,
which loads the EGL plugin before any body — the plugin only sees
bodies loaded after it. When wrapping your own client, call
``_maybe_load_egl(pybullet, client_id)`` right after ``connect`` (before
any ``loadURDF``) and pass its result as ``egl_plugin``; otherwise the
adapter renders with the CPU TinyRenderer.
"""

from __future__ import annotations

import ctypes.util
import os
import pkgutil
import sys
from fnmatch import fnmatchcase
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, cast

import numpy as np

from ..adapter import MonitorObservation, PropertyMeta, SimulatorAdapter
from ..entities import EntityState, TrackState
from ..registry import register as _register
from .mujoco import _property_token

_QUAT_NOTES = ("quat x", "quat y", "quat z", "quat w")
_WORLD_FRAME = "world Cartesian"
_QUAT_FRAME = "active base-local-to-world (x, y, z, w)"
_CONTACT_FRAME = "frame-free scalar per-body aggregate"
_JOINT_FRAME = "parent-local joint axis"

# Vertical FOV shared by _frame_view's distance fit and capture_frame's
# projection — they must move together.
_CAPTURE_FOV_DEG = 60.0

# Where the CLI factory loads every URDF (PyBullet's defaults, passed
# explicitly so the state signature records the pose actually used).
_LOAD_POSITION = (0.0, 0.0, 0.0)
_LOAD_ORIENTATION = (0.0, 0.0, 0.0, 1.0)


class PyBulletAdapter(SimulatorAdapter):
    """
    Expose one PyBullet physics client as monitor observations.

    Parameters
    ----------
    client_id:
        An already-connected PyBullet physics client id (DIRECT or GUI).
        The caller keeps ownership of stepping and disconnecting unless
        the adapter was built by the CLI factory.

    body_rules:
        Optional ``(pattern, entity_class)`` pairs matched against body
        names with ``fnmatch`` (first match wins). Unmatched articulated
        bodies become ``"robot"``, jointless bodies ``"object"``.

    body_labels:
        Optional ``{body_uid: label}`` display-name overrides. Purely
        presentational (discovery tables, live-monitor binding names):
        labels never appear in property keys, and classification
        (``body_rules``) always matches the sim-reported body name.
        Without an override, duplicate body names are suffixed
        ``<name>#<uid>`` for display so bindings stay tellable apart.

    egl_plugin:
        Plugin id from ``_maybe_load_egl`` (see the module docstring),
        or -1 (default) to render with the CPU TinyRenderer. The plugin
        must have been loaded before any body.
    """

    def __init__(
        self,
        client_id: int,
        body_rules: Optional[Sequence[Tuple[str, str]]] = None,
        body_labels: Optional[Dict[int, str]] = None,
        egl_plugin: int = -1,
    ) -> None:
        super().__init__()
        import pybullet

        self._p = pybullet
        self._client = client_id
        self._body_rules: Tuple[Tuple[str, str], ...] = tuple(
            body_rules or ()
        )
        self._body_labels: Dict[int, str] = dict(body_labels or {})
        self._owns_client = False
        self._step = 0

        params = self._p.getPhysicsEngineParameters(
            physicsClientId=self._client
        )
        self._dt = float(params["fixedTimeStep"])
        self._egl_plugin = int(egl_plugin)

        # Filled by _discover_schema() at reset.
        self.entity_labels: Dict[int, str] = {}
        self._names: Dict[int, str] = {}
        self._classes: Dict[int, str] = {}
        self._movable_joints: Dict[int, List[Tuple[int, str, int]]] = {}
        self._links: Dict[int, List[Tuple[int, str]]] = {}
        self._meta: Dict[int, Dict[str, PropertyMeta]] = {}
        # Filled by the CLI factory: per body uid, the file digests, load
        # pose and fixed-base flag that ``state_signature`` carries.
        self._loaded: Dict[int, dict] = {}

    # ------------------------------------------------------------------
    @property
    def dt(self) -> float:
        return self._dt

    @property
    def spatial_dims(self) -> int:
        return 3

    @property
    def supported_classes(self) -> FrozenSet[str]:
        return frozenset(
            {entity_class for _, entity_class in self._body_rules}
            | {"robot", "object"}
        )

    def property_metadata(
        self,
        entity_id: int,
    ) -> Dict[str, PropertyMeta]:
        metadata = self._meta.get(entity_id, {})
        projected = self._project_property_metadata(entity_id, metadata)
        if projected is metadata:
            return metadata
        return cast(Dict[str, PropertyMeta], dict(projected))

    # ------------------------------------------------------------------
    def _reset_impl(self) -> MonitorObservation:
        self._step = 0
        self._discover_schema()
        self._frame_view()
        return self._snapshot()

    def _frame_view(self) -> None:
        """Aim the capture camera at the scene: union the body AABBs
        (ground planes excluded — a 60m plane would zoom the camera out
        to a dot) and back the orbit off far enough for the bounding
        sphere to fit the vertical FOV. Computed once per reset so the
        view stays steady while bodies move."""
        pb = self._p
        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        for uid in self._names:
            # getAABB(uid) alone is the base link only; articulated
            # bodies need the union over links or a KUKA frames as a
            # 20cm base.
            try:
                aabbs = [pb.getAABB(uid, physicsClientId=self._client)]
                joints = pb.getNumJoints(
                    uid, physicsClientId=self._client)
                for link in range(joints):
                    aabbs.append(pb.getAABB(
                        uid, link, physicsClientId=self._client))
            except pb.error:
                continue                  # body removed since discovery
            lo = np.min([a[0] for a in aabbs], axis=0)
            hi = np.max([a[1] for a in aabbs], axis=0)
            if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
                continue                  # degenerate collision shape
            size = hi - lo
            if max(size[0], size[1]) > 20.0:
                continue
            low = np.minimum(low, lo)
            high = np.maximum(high, hi)
        if not np.isfinite(low).all():
            return                        # plane-only scene: keep default
        center = (low + high) / 2.0
        radius = float(np.linalg.norm(high - low)) / 2.0
        self._view_target = tuple(float(c) for c in center)
        # Sphere-in-frustum fit (distance = r / sin(fov/2)) with a 15%
        # margin. Assumes landscape output: the vertical FOV is the
        # binding constraint, true for every current caller.
        half_fov = np.radians(_CAPTURE_FOV_DEG / 2.0)
        self._view_distance = float(
            max(0.75, radius / np.sin(half_fov) * 1.15))

    def _get_observation_impl(self) -> MonitorObservation:
        self._step += 1
        return self._snapshot()

    # ------------------------------------------------------------------
    def _discover_schema(self) -> None:
        """
        Enumerate bodies, joints, and links once for the episode and
        build the matching property metadata, guarding loudly against
        joint or link names that sanitize to the same token within one
        body. Body names never enter property keys (per-entity
        quantities use entity-generic keys), so duplicate body names
        are legal and only disambiguated for display.
        """
        pb = self._p
        self.entity_labels = {}
        self._names = {}
        self._classes = {}
        self._movable_joints = {}
        self._links = {}
        self._meta = {}

        seen_labels: set = set()

        for index in range(pb.getNumBodies(physicsClientId=self._client)):
            uid = pb.getBodyUniqueId(index, physicsClientId=self._client)
            info = pb.getBodyInfo(uid, physicsClientId=self._client)
            name: str = info[1].decode()
            label = self._body_labels.get(uid, name)
            while label in seen_labels:
                label = f"{label}#{uid}"
            seen_labels.add(label)

            self.entity_labels[uid] = label
            self._names[uid] = name
            # Rules match the sim-reported name so a display label
            # cannot change an entity's class.
            self._classes[uid] = self._classify(
                name,
                articulated=pb.getNumJoints(
                    uid, physicsClientId=self._client
                ) > 0,
            )

            self._discover_body_schema(uid)

    def _classify(self, label: str, articulated: bool) -> str:
        for pattern, entity_class in self._body_rules:
            if fnmatchcase(label, pattern):
                return entity_class
        return "robot" if articulated else "object"

    def _discover_body_schema(self, uid: int) -> None:
        """Enumerate one body's joints/links and build its metadata."""
        pb = self._p
        movable: List[Tuple[int, str, int]] = []
        links: List[Tuple[int, str]] = []
        joint_token_owners: Dict[str, str] = {}
        link_token_owners: Dict[str, str] = {}

        # Per-entity quantities use entity-generic keys (no body token):
        # identical instances expose identical vocabularies, so one
        # wildcard spec binds every instance.
        meta: Dict[str, PropertyMeta] = {}
        for axis in range(3):
            meta[f"basepos_{axis}"] = PropertyMeta(
                group="body position", note="m", unit="m", frame=_WORLD_FRAME
            )
            meta[f"basevel_{axis}"] = PropertyMeta(
                group="body linear velocity", note="m/s",
                unit="m/s", frame=_WORLD_FRAME,
            )
            meta[f"baseangvel_{axis}"] = PropertyMeta(
                group="body angular velocity", note="rad/s",
                unit="rad/s", frame=_WORLD_FRAME,
            )
        for axis in range(4):
            meta[f"basequat_{axis}"] = PropertyMeta(
                group="body orientation", note=_QUAT_NOTES[axis],
                unit="1", frame=_QUAT_FRAME,
            )
        meta["contactforce"] = PropertyMeta(
            group="contact normal force", note="N",
            unit="N", frame=_CONTACT_FRAME,
        )

        for joint_index in range(
            pb.getNumJoints(uid, physicsClientId=self._client)
        ):
            info = pb.getJointInfo(
                uid, joint_index, physicsClientId=self._client
            )
            joint_name = info[1].decode()
            joint_type = info[2]
            link_name = info[12].decode()

            link_token = _property_token(link_name)
            if link_token in link_token_owners:
                raise ValueError(
                    f"Link name {link_name!r} and link name "
                    f"{link_token_owners[link_token]!r} on body "
                    f"{self.entity_labels[uid]!r} both sanitize to "
                    f"property token {link_token!r}."
                )
            link_token_owners[link_token] = link_name
            links.append((joint_index, link_token))
            for axis in range(3):
                meta[f"linkpos_{link_token}_{axis}"] = PropertyMeta(
                    group="link position",
                    note="m (world COM)", unit="m", frame=_WORLD_FRAME,
                )

            if joint_type not in (
                pb.JOINT_REVOLUTE,
                pb.JOINT_PRISMATIC,
            ):
                continue

            joint_token = _property_token(joint_name)
            if joint_token in joint_token_owners:
                raise ValueError(
                    f"Joint name {joint_name!r} and joint name "
                    f"{joint_token_owners[joint_token]!r} on body "
                    f"{self.entity_labels[uid]!r} both sanitize to "
                    f"property token {joint_token!r}."
                )
            joint_token_owners[joint_token] = joint_name
            movable.append((joint_index, joint_token, joint_type))

            # Units come from the joint type, never from the name.
            revolute = joint_type == pb.JOINT_REVOLUTE
            meta[f"jointpos_{joint_token}"] = PropertyMeta(
                group="joint angle" if revolute else "joint displacement",
                note="rad" if revolute else "m",
                unit="rad" if revolute else "m", frame=_JOINT_FRAME,
            )
            meta[f"jointvel_{joint_token}"] = PropertyMeta(
                group=(
                    "joint angular velocity"
                    if revolute
                    else "joint linear velocity"
                ),
                note="rad/s" if revolute else "m/s",
                unit="rad/s" if revolute else "m/s", frame=_JOINT_FRAME,
            )
            meta[f"jointtorque_{joint_token}"] = PropertyMeta(
                group="applied torques" if revolute else "applied forces",
                note="N·m" if revolute else "N",
                unit="N·m" if revolute else "N", frame=_JOINT_FRAME,
            )

        self._movable_joints[uid] = movable
        self._links[uid] = links
        self._meta[uid] = meta

    # ------------------------------------------------------------------
    def _snapshot(self) -> MonitorObservation:
        pb = self._p
        entities: Dict[int, EntityState] = {}
        live_bodies: List[int] = []

        for uid, name in self._names.items():
            # Liveness check: a body removed mid-episode is entity
            # absence, and a reused uid whose body name changed must not
            # be reported under the old identity.
            try:
                info = pb.getBodyInfo(uid, physicsClientId=self._client)
            except pb.error:
                continue
            if info[1].decode() != name:
                continue

            live_bodies.append(uid)

        # Contact aggregation is global in PyBullet, so defer it until
        # liveness is known and skip it entirely when no live class needs
        # contactforce. The unscoped path still requests the full surface.
        contact_requested = False
        for uid in live_bodies:
            if self._property_is_requested(
                self._classes[uid], "contactforce"
            ):
                contact_requested = True
                break
        contact_totals = self._contact_totals() if contact_requested else {}

        for uid in live_bodies:
            position, quaternion = pb.getBasePositionAndOrientation(
                uid, physicsClientId=self._client
            )
            linear, angular = pb.getBaseVelocity(
                uid, physicsClientId=self._client
            )

            properties: Dict[str, float] = {}
            for axis in range(3):
                key = f"basepos_{axis}"
                if self._property_is_requested(self._classes[uid], key):
                    properties[key] = float(position[axis])
                key = f"basevel_{axis}"
                if self._property_is_requested(self._classes[uid], key):
                    properties[key] = float(linear[axis])
                key = f"baseangvel_{axis}"
                if self._property_is_requested(self._classes[uid], key):
                    properties[key] = float(angular[axis])
            for axis in range(4):
                key = f"basequat_{axis}"
                if self._property_is_requested(self._classes[uid], key):
                    properties[key] = float(quaternion[axis])
            if self._property_is_requested(self._classes[uid], "contactforce"):
                properties["contactforce"] = contact_totals.get(uid, 0.0)

            for joint_index, joint_token, _ in self._movable_joints[uid]:
                position_key = f"jointpos_{joint_token}"
                velocity_key = f"jointvel_{joint_token}"
                torque_key = f"jointtorque_{joint_token}"
                if not any(
                    self._property_is_requested(self._classes[uid], key)
                    for key in (position_key, velocity_key, torque_key)
                ):
                    continue
                state = pb.getJointState(
                    uid, joint_index, physicsClientId=self._client
                )
                if self._property_is_requested(self._classes[uid], position_key):
                    properties[position_key] = float(state[0])
                if self._property_is_requested(self._classes[uid], velocity_key):
                    properties[velocity_key] = float(state[1])
                if self._property_is_requested(self._classes[uid], torque_key):
                    properties[torque_key] = float(state[3])

            for joint_index, link_token in self._links[uid]:
                keys = tuple(
                    f"linkpos_{link_token}_{axis}" for axis in range(3)
                )
                if not any(
                    self._property_is_requested(self._classes[uid], key)
                    for key in keys
                ):
                    continue
                # computeForwardKinematics=1: after stepSimulation the cached
                # link frames are those of the joint positions before the
                # step; the flag recomputes them at the current positions,
                # as resetJointState does, without stepping or detecting
                # collisions (maintainer decision 2026-10-14).
                link_state = pb.getLinkState(
                    uid, joint_index, computeForwardKinematics=1,
                    physicsClientId=self._client
                )
                for axis, key in enumerate(keys):
                    if self._property_is_requested(self._classes[uid], key):
                        properties[key] = float(link_state[0][axis])

            entities[uid] = EntityState(
                entity_id=uid,
                entity_class=self._classes[uid],
                position=np.asarray(position, dtype=float),
                velocity=np.asarray(linear, dtype=float),
                track_age=self._step,
                track_state=TrackState.CONFIRMED,
                properties=properties,
            )

        return MonitorObservation(
            timestep=self._step,
            time=self._step * self._dt,
            perceived_entities=entities,
        )

    def _contact_totals(self) -> Dict[int, float]:
        """Sum contact normal forces per body over all current contacts."""
        totals: Dict[int, float] = {}
        for contact in self._p.getContactPoints(
            physicsClientId=self._client
        ):
            body_a, body_b = contact[1], contact[2]
            force = float(contact[9])
            totals[body_a] = totals.get(body_a, 0.0) + force
            if body_b != body_a:
                totals[body_b] = totals.get(body_b, 0.0) + force
        return totals

    # ------------------------------------------------------------------
    def make_advancer(self, mode, demo, seed):
        """
        Drive behavior for the CLI: one stepSimulation() per step.

        Only ``hold`` is supported: physics advances under no new command,
        so each joint keeps whatever motor setting it already has
        (PyBullet's default velocity motor holds it at zero velocity).
        ``random`` is refused because a URDF scene declares no action
        space to sample, and ``replay`` and ``--demo`` because the adapter
        reads no demonstration format (#132). A Specola state recording
        is not read here: the shared runner restores its rows through
        ``restore_state`` and never calls this hook.
        """
        if demo is not None:
            raise SystemExit(
                "The pybullet adapter has no --demo replay: it reads no "
                "demonstration format."
            )
        if mode != "hold":
            raise SystemExit(
                f"--mode {mode} is not supported by the pybullet adapter: "
                "a URDF scene declares no action space to sample and there "
                "is no replay path; use --mode hold to step physics under "
                "no new command."
            )
        del seed

        def advance(step: int) -> bool:
            self._p.stepSimulation(physicsClientId=self._client)
            return True

        return advance

    # -- state tier (replay) ------------------------------------------------

    def _state_layout(self) -> Optional[List[Tuple[int, List[Tuple[int, int, int, int]]]]]:
        """Every body in id order with its joints that carry state.

        Each joint is ``(index, type, positions, velocities)``: one and
        one for revolute and prismatic joints, the lengths
        ``getJointStateMultiDof`` reports for spherical ones; fixed joints
        carry none. None when a body has a planar joint: PyBullet 3.2.7
        reports two values each for its three degrees of freedom, and a
        restore from them leaves the link elsewhere (measured 2026-10-14),
        so such a scene has no state tier.
        """
        # One plain ``pb = self._p`` and no backend call inside a
        # comprehension: the grammar benchmarks/adapter_fidelity/
        # pybullet_source_inventory.py audits.
        pb = self._p
        uids = []
        for index in range(pb.getNumBodies(physicsClientId=self._client)):
            uids.append(pb.getBodyUniqueId(index, physicsClientId=self._client))
        layout = []
        for uid in sorted(uids):
            joints = []
            for index in range(pb.getNumJoints(uid, physicsClientId=self._client)):
                kind = pb.getJointInfo(uid, index, physicsClientId=self._client)[2]
                if kind == pb.JOINT_PLANAR:
                    return None
                if kind == pb.JOINT_SPHERICAL:
                    position, velocity = pb.getJointStateMultiDof(
                        uid, index, physicsClientId=self._client)[:2]
                    joints.append((index, kind, len(position), len(velocity)))
                elif kind in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC):
                    joints.append((index, kind, 1, 1))
            layout.append((uid, joints))
        return layout

    def state_signature(self) -> Optional[dict]:
        """Engine build, every body's model, and the state length.

        Per body: what the CLI factory loaded (SHA-256 of the URDF file and
        of every mesh it references, the load pose and ``useFixedBase``),
        then per link, base first, the collision shapes (geometry type,
        dimensions, local frame) and dynamics (mass, lateral friction,
        restitution, local inertia diagonal, inertial frame), and per joint
        its name, type, limits, child link, axis, parent frame and state
        lengths. A body the caller loaded into a wrapped client has no file
        digests; its engine reads still fingerprint it. Mesh file paths are
        left out, so a recording replays where the same files live
        elsewhere. None when the scene has a planar joint (see
        ``_state_layout``).
        """
        import importlib.metadata
        import json

        layout = self._state_layout()
        if layout is None:
            return None
        pb = self._p
        bodies = []
        for uid, stateful in layout:
            lengths = {index: [npos, nvel] for index, _, npos, nvel in stateful}
            links, joints = [], []
            for link in range(-1, pb.getNumJoints(uid, physicsClientId=self._client)):
                collision = []
                for shape in pb.getCollisionShapeData(uid, link, physicsClientId=self._client):
                    collision.append([shape[2], shape[3], shape[5], shape[6]])
                dynamics = pb.getDynamicsInfo(uid, link, physicsClientId=self._client)
                links.append({
                    "collision": collision,
                    "mass": dynamics[0], "lateral_friction": dynamics[1],
                    "inertia_diagonal": dynamics[2], "inertial_frame": dynamics[3:5],
                    "restitution": dynamics[5],
                })
                if link < 0:
                    continue
                info = pb.getJointInfo(uid, link, physicsClientId=self._client)
                joints.append({
                    "name": info[1].decode(), "type": info[2], "limits": info[8:12],
                    "link": info[12].decode(), "axis": info[13],
                    "parent_frame": info[14:16], "parent": info[16],
                    "state": lengths.get(link, [0, 0]),
                })
            bodies.append({**self._loaded.get(uid, {}), "links": links, "joints": joints})
        signature = {
            "engine": "pybullet",
            "api": pb.getAPIVersion(physicsClientId=self._client),
            "package": importlib.metadata.version("pybullet"),
            "bodies": bodies,
            "length": _state_length(layout),
        }
        # The recording keeps the signature as JSON; compare in that form.
        return json.loads(json.dumps(signature))

    def capture_state(self) -> Optional[np.ndarray]:
        """Per body in id order: base position (3), quaternion (4), linear
        and angular velocity (3 and 3), then each joint's positions and
        velocities. Contact forces and motor torques are step outputs and
        are not part of it."""
        layout = self._state_layout()
        if layout is None:
            return None
        pb = self._p
        values: List[float] = []
        for uid, stateful in layout:
            position, orientation = pb.getBasePositionAndOrientation(
                uid, physicsClientId=self._client)
            linear, angular = pb.getBaseVelocity(uid, physicsClientId=self._client)
            values.extend((*position, *orientation, *linear, *angular))
            for index, kind, _, _ in stateful:
                if kind == pb.JOINT_SPHERICAL:
                    joint_position, joint_velocity = pb.getJointStateMultiDof(
                        uid, index, physicsClientId=self._client)[:2]
                    values.extend((*joint_position, *joint_velocity))
                else:
                    values.extend(pb.getJointState(uid, index, physicsClientId=self._client)[:2])
        return np.array(values, dtype=np.float64)

    def restore_state(self, state: np.ndarray) -> None:
        """Reset every base and joint to ``state``, then detect collisions.

        ``performCollisionDetection`` brings the contact list to the
        restored configuration. No physics step: contact normal forces and
        joint motor torques keep the values the last step left.
        """
        layout = self._state_layout()
        if layout is None:
            raise NotImplementedError(
                "PyBulletAdapter cannot restore a scene with a planar joint: "
                "PyBullet reports too few of its state values to restore it"
            )
        expected = _state_length(layout)
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (expected,):
            raise ValueError(
                f"state of shape {state.shape}; this scene's state has length {expected}"
            )
        pb = self._p
        cursor = 0
        for uid, stateful in layout:
            base = state[cursor:cursor + 13].tolist()
            cursor += 13
            pb.resetBasePositionAndOrientation(
                uid, base[0:3], base[3:7], physicsClientId=self._client)
            pb.resetBaseVelocity(uid, base[7:10], base[10:13], physicsClientId=self._client)
            for index, kind, npos, nvel in stateful:
                joint = state[cursor:cursor + npos + nvel].tolist()
                cursor += npos + nvel
                if kind == pb.JOINT_SPHERICAL:
                    pb.resetJointStateMultiDof(
                        uid, index, targetValue=joint[:npos], targetVelocity=joint[npos:],
                        physicsClientId=self._client)
                else:
                    pb.resetJointState(
                        uid, index, targetValue=joint[0], targetVelocity=joint[1],
                        physicsClientId=self._client)
        pb.performCollisionDetection(physicsClientId=self._client)

    def capture_frame(
        self,
        camera: Optional[str] = None,
        *,
        width: int = 320,
        height: int = 240,
    ) -> Optional[np.ndarray]:
        """
        Render one offscreen RGB frame from the monitors' physics client.

        Works in DIRECT (headless) connections. TinyRenderer (CPU) by
        default; ``PYBULLET_EGL=1`` switches to the hardware EGL
        renderer (lighting and shadows, needs a GPU with libEGL).
        PyBullet has no named model cameras, so ``camera`` must be
        ``None``.
        """
        if camera is not None:
            raise ValueError(
                "PyBullet has no named cameras; use the default view"
            )
        # ponytail: fixed orbit angle, auto-framed target/distance from
        # reset-time scene bounds; expose yaw/pitch knobs when a task
        # actually needs a different vantage.
        view = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=getattr(
                self, "_view_target", (0.0, 0.0, 0.0)),
            distance=getattr(self, "_view_distance", 2.0),
            yaw=45.0,
            pitch=-30.0,
            roll=0.0,
            upAxisIndex=2,
        )
        projection = self._p.computeProjectionMatrixFOV(
            fov=_CAPTURE_FOV_DEG,
            aspect=width / float(height),
            nearVal=0.01,
            farVal=100.0,
        )
        _, _, rgba, _, _ = self._p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=(self._p.ER_BULLET_HARDWARE_OPENGL
                      if self._egl_plugin >= 0
                      else self._p.ER_TINY_RENDERER),
            physicsClientId=self._client,
        )
        array = np.asarray(rgba, dtype=np.uint8).reshape(
            height, width, 4
        )
        return array[:, :, :3]

    def close(self) -> None:
        """
        Disconnect the client if this adapter created it (factory mode).

        Nothing in the CLI calls this — one-shot CLI processes rely on
        process exit to tear the DIRECT client down. Embedding code that
        builds adapters through the registry factory and outlives them
        should call it. Safe to call repeatedly, and a no-op for
        wrapped caller-owned clients.
        """
        if self._owns_client and self._client >= 0:
            self._p.disconnect(physicsClientId=self._client)
            self._client = -1


def _state_length(layout) -> int:
    """13 base values per body plus each joint's positions and velocities."""
    return sum(13 + sum(npos + nvel for _, _, npos, nvel in joints)
               for _, joints in layout)


def _urdf_digests(path: str) -> dict:
    """SHA-256 of a URDF file and of every mesh file it references.

    Mesh names (``package://`` stripped) resolve against the URDF's
    directory, then ``pybullet_data``; a name that resolves nowhere is
    recorded with a null digest. Keys are the names as the URDF writes them.
    """
    import hashlib
    import xml.etree.ElementTree as ElementTree

    def digest(file: str) -> str:
        with open(file, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    roots = [os.path.dirname(os.path.abspath(path))]
    try:
        import pybullet_data
        roots.append(pybullet_data.getDataPath())
    except ImportError:
        pass
    meshes: Dict[str, Optional[str]] = {}
    for element in ElementTree.parse(path).iter("mesh"):
        name = element.get("filename", "")
        relative = name[len("package://"):] if name.startswith("package://") else name
        found = next((candidate for candidate in (os.path.join(root, relative) for root in roots)
                      if os.path.isfile(candidate)), None)
        meshes[name] = digest(found) if found else None
    return {"urdf_sha256": digest(path), "mesh_sha256": dict(sorted(meshes.items()))}


def _resolve_urdf(path: str) -> str:
    if os.path.exists(path):
        return path
    try:
        import pybullet_data
        candidate = os.path.join(pybullet_data.getDataPath(), path)
        if os.path.exists(candidate):
            return candidate
    except ImportError:
        pass
    raise ValueError(
        f"URDF path {path!r} not found (checked the filesystem and "
        f"pybullet_data)."
    )


def _maybe_load_egl(pb, client: int) -> int:
    """Load PyBullet's EGL hardware-render plugin when opted in.

    Opt-in via ``PYBULLET_EGL=1`` (PyBullet's own convention). Never the
    default: with the plugin loaded but no working libEGL, the first
    render calls ``exit(EXIT_FAILURE)`` inside the plugin — it kills the
    process rather than raising. The libEGL probe below narrows that
    window; a truly broken EGL stack past the probe still crashes, which
    is why the flag is explicit.

    MUST run before any body is loaded: the plugin keeps its own scene
    copy synced on body load, so bodies that predate it render as an
    empty background. Returns the plugin id, or -1 when disabled or
    unavailable.
    """
    if os.environ.get("PYBULLET_EGL", "").lower() not in (
            "1", "true", "yes"):
        return -1
    if ctypes.util.find_library("EGL") is None:
        print("  Web warning: PYBULLET_EGL set but libEGL not found; "
              "using TinyRenderer.", file=sys.stderr)
        return -1
    egl = pkgutil.get_loader("eglRenderer")
    if egl is not None:
        return pb.loadPlugin(
            egl.get_filename(), "_eglRendererPlugin",
            physicsClientId=client)
    return pb.loadPlugin("eglRendererPlugin", physicsClientId=client)


def _make_pybullet(
    task: Optional[str],
    robot: Optional[str],
    target_only: bool,
) -> PyBulletAdapter:
    """
    Registry factory. ``task`` is a comma-separated list of URDF paths,
    each resolved against the filesystem and then ``pybullet_data``;
    append ``@fixed`` to a path to load it with a fixed base
    (e.g. ``--task "kuka_iiwa/model.urdf@fixed,cube.urdf"``).
    ``robot`` is ignored.
    """
    del robot, target_only
    if not task:
        raise ValueError(
            "The pybullet adapter needs --task "
            "<scene.urdf[@fixed][,more.urdf...]>"
        )

    import pybullet

    client = pybullet.connect(pybullet.DIRECT)
    loaded: Dict[int, dict] = {}
    try:
        # Before any loadURDF: the EGL renderer only sees bodies loaded
        # after it (see _maybe_load_egl).
        egl_plugin = _maybe_load_egl(pybullet, client)
        pybullet.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
        for spec in task.split(","):
            spec = spec.strip()
            fixed = spec.endswith("@fixed")
            if fixed:
                spec = spec[: -len("@fixed")]
            path = _resolve_urdf(spec)
            uid = pybullet.loadURDF(
                path,
                basePosition=_LOAD_POSITION,
                baseOrientation=_LOAD_ORIENTATION,
                useFixedBase=fixed,
                physicsClientId=client,
            )
            loaded[uid] = {
                **_urdf_digests(path),
                "load_position": list(_LOAD_POSITION),
                "load_orientation": list(_LOAD_ORIENTATION),
                "fixed_base": fixed,
            }
    except Exception:
        pybullet.disconnect(physicsClientId=client)
        raise

    adapter = PyBulletAdapter(client, egl_plugin=egl_plugin)
    adapter._owns_client = True
    adapter._loaded = loaded
    return adapter


_register(
    name="pybullet",
    factory=_make_pybullet,
    description="PyBullet physics engine (URDF scenes)",
    install_hint="pip install pybullet",
    requires=("pybullet",),
)
