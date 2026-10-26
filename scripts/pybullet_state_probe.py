"""Measure what PyBullet's reset calls reproduce, for the PyBullet state tier.

Task 6 of ``docs/superpowers/plans/2026-10-13-state-replay.md``; the
numbers are on ``docs/evidence/state-replay-pybullet-2026-10-14.md``.
Rerun it on another PyBullet build before trusting the declared
tolerance in ``benchmarks/replay_fidelity.py`` (``RESTORE_TOLERANCE``) or
the ``LIVE_KINEMATIC_LAG`` entry there.

1. Base quaternion: how far does a read of the base orientation move
   after ``resetBasePositionAndOrientation`` with that same value? The
   server stores the quaternion it is given and reports
   ``btTransform::getRotation()`` of a matrix built from it
   (``examples/SharedMemory/PhysicsServerCommandProcessor.cpp`` lines
   7933 to 7935 and 7959 in the 3.2.7 source distribution). In
   units of the last place of the component and of 1.0, on the audited
   scene (30 ``hold`` steps, every row restored through the adapter), on
   20,000 random orientations, and on a free cube spinning for 2,000
   steps. Base positions, base velocities and joint states are checked
   for bit equality on the audited scene.
2. Live kinematic lag: does ``getLinkState`` after ``stepSimulation``
   (``computeForwardKinematics=0``, what the adapter reads) equal the
   forward kinematics of the joint positions after the step or before
   it? A twin client evaluates both.
3. Planar joints: does ``getJointStateMultiDof`` carry all three of a
   planar joint's degrees of freedom, and does a restore from it bring
   the link back?

    PYTHONNOUSERSITE=1 PYTHONPATH=$PWD/src \\
        ~/miniconda3/envs/pybullet/bin/python scripts/pybullet_state_probe.py
"""

from __future__ import annotations

import importlib.metadata
import os
import tempfile

import numpy as np
import pybullet as pb
import pybullet_data

from specola.adapters.pybullet import _make_pybullet

SCENE = "plane.urdf@fixed,kuka_iiwa/model.urdf@fixed,cube_small.urdf"


def ulps(a, b):
    """Largest |a - b| in ulps of the larger component and in ulps of 1.0."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    gap = np.abs(a - b)
    return (float(np.max(gap / np.spacing(np.maximum(np.abs(a), np.abs(b))))),
            float(np.max(gap / np.spacing(1.0))))


def audited_scene() -> None:
    adapter = _make_pybullet(SCENE, None, False)

    def capture() -> np.ndarray:
        state = adapter.capture_state()
        assert state is not None, "the audited scene has a state tier"
        return state

    adapter.reset()
    advance = adapter.make_advancer("hold", None, 0)
    rows = [capture()]
    for step in range(30):
        advance(step)
        rows.append(capture())
    # The cube is the third body; its quaternion is the fourth to seventh
    # of its 13 base values (the plane has 13, the KUKA 13 + 7 x 2).
    cube = 13 + 27
    quaternion = list(range(cube + 3, cube + 7))
    worst, changed = (0.0, 0.0), 0
    for row in rows:
        adapter.restore_state(row)
        again = capture()
        assert np.array_equal(np.delete(row, quaternion), np.delete(again, quaternion)), \
            "a base position, base velocity or joint value did not come back bit for bit"
        if not np.array_equal(row[quaternion], again[quaternion]):
            changed += 1
            worst = tuple(max(w, u) for w, u in zip(worst, ulps(row[quaternion], again[quaternion])))
    adapter.close()
    print(f"1a. audited scene, {len(rows)} rows: quaternion changed on {changed}; largest change "
          f"{worst[0]:.3g} ulps of the component, {worst[1]:.3g} ulps of 1.0; every other value bit-equal")


def random_and_spinning() -> None:
    client = pb.connect(pb.DIRECT)
    cube = pb.loadURDF(os.path.join(pybullet_data.getDataPath(), "cube_small.urdf"),
                       physicsClientId=client)
    rng = np.random.default_rng(0)
    worst, changed = (0.0, 0.0), 0
    for _ in range(20000):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        pb.resetBasePositionAndOrientation(cube, [0, 0, 1], q.tolist(), physicsClientId=client)
        q0 = pb.getBasePositionAndOrientation(cube, physicsClientId=client)[1]
        pb.resetBasePositionAndOrientation(cube, [0, 0, 1], q0, physicsClientId=client)
        q1 = pb.getBasePositionAndOrientation(cube, physicsClientId=client)[1]
        if q0 != q1:
            changed += 1
            worst = tuple(max(w, u) for w, u in zip(worst, ulps(q0, q1)))
    print(f"1b. 20,000 random orientations: {changed} changed; largest change "
          f"{worst[0]:.3g} ulps of the component, {worst[1]:.3g} ulps of 1.0")

    pb.resetBasePositionAndOrientation(cube, [0, 0, 1], [0, 0, 0, 1], physicsClientId=client)
    pb.resetBaseVelocity(cube, [0, 0, 0], [3.0, -2.0, 7.0], physicsClientId=client)
    worst, changed = (0.0, 0.0), 0
    for _ in range(2000):
        pb.stepSimulation(physicsClientId=client)
        position, q0 = pb.getBasePositionAndOrientation(cube, physicsClientId=client)
        linear, angular = pb.getBaseVelocity(cube, physicsClientId=client)
        pb.resetBasePositionAndOrientation(cube, position, q0, physicsClientId=client)
        pb.resetBaseVelocity(cube, linear, angular, physicsClientId=client)
        q1 = pb.getBasePositionAndOrientation(cube, physicsClientId=client)[1]
        if q0 != q1:
            changed += 1
            worst = tuple(max(w, u) for w, u in zip(worst, ulps(q0, q1)))
    pb.disconnect(physicsClientId=client)
    print(f"1c. free cube spinning, 2,000 steps, no gravity: {changed} changed; largest change "
          f"{worst[0]:.3g} ulps of the component, {worst[1]:.3g} ulps of 1.0")


def kinematic_lag() -> None:
    live, twin = _make_pybullet(SCENE, None, False), _make_pybullet(SCENE, None, False)
    kuka, joints = 1, range(7)

    def joint_state(client):
        return [pb.getJointState(kuka, j, physicsClientId=client)[:2] for j in joints]

    def forward_kinematics(state):
        for j, (position, velocity) in enumerate(state):
            pb.resetJointState(kuka, j, position, velocity, physicsClientId=twin._client)
        return np.array([pb.getLinkState(kuka, j, computeForwardKinematics=1,
                                         physicsClientId=twin._client)[0] for j in joints])

    before = joint_state(live._client)
    gap_after = gap_before = 0.0
    for _ in range(30):
        pb.stepSimulation(physicsClientId=live._client)
        read = np.array([pb.getLinkState(kuka, j, physicsClientId=live._client)[0] for j in joints])
        after = joint_state(live._client)
        gap_after = max(gap_after, float(np.max(np.abs(read - forward_kinematics(after)))))
        gap_before = max(gap_before, float(np.max(np.abs(read - forward_kinematics(before)))))
        before = after
    live.close()
    twin.close()
    print(f"2. getLinkState after a step, 30 hold steps: largest difference from forward kinematics "
          f"of the joint positions after the step {gap_after:.3g} m, before the step {gap_before:.3g} m")


PLANAR = """
<robot name="planar">
  <link name="base"><inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
  <link name="slab"><inertial><origin xyz="0.2 0.1 0"/><mass value="0.5"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <joint name="slab_joint" type="planar"><parent link="base"/><child link="slab"/><origin xyz="0.3 0 0"/><axis xyz="0 0 1"/></joint>
</robot>
"""


def planar_joint() -> None:
    path = os.path.join(tempfile.mkdtemp(), "planar.urdf")
    with open(path, "w") as handle:
        handle.write(PLANAR)
    client = pb.connect(pb.DIRECT)
    uid = pb.loadURDF(path, useFixedBase=True, physicsClientId=client)
    pb.setGravity(-9.81, 3.0, 0.0, physicsClientId=client)    # in the joint's plane
    for _ in range(10):
        pb.stepSimulation(physicsClientId=client)
    position, velocity = pb.getJointStateMultiDof(uid, 0, physicsClientId=client)[:2]
    pose = pb.getLinkState(uid, 0, computeForwardKinematics=1, physicsClientId=client)[4]
    for _ in range(10):
        pb.stepSimulation(physicsClientId=client)
    pb.resetJointStateMultiDof(uid, 0, targetValue=position, targetVelocity=velocity,
                               physicsClientId=client)
    again = pb.getLinkState(uid, 0, computeForwardKinematics=1, physicsClientId=client)[4]
    axis = pb.getJointInfo(uid, 0, physicsClientId=client)[13]
    pb.disconnect(physicsClientId=client)
    print(f"3. planar joint: getJointInfo axis {axis}; getJointStateMultiDof gives {len(position)} positions "
          f"and {len(velocity)} velocities; link position after the restore is "
          f"{float(np.linalg.norm(np.subtract(again, pose))):.3g} m from the captured one")


if __name__ == "__main__":
    print(f"pybullet {importlib.metadata.version('pybullet')}, API {pb.getAPIVersion()}")
    audited_scene()
    random_and_spinning()
    kinematic_lag()
    planar_joint()
