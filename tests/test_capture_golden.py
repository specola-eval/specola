"""Golden-frame checks: adapter capture vs the simulator's native render.

Each test renders the same simulator state through the simulator's own
pipeline and through ``capture_frame``, then compares. This is the
mechanical tripwire for the failure class found 2026-07-31: RoboCasa
uploaded reset-time textures through its live render context only, and
a fresh ``mujoco.Renderer`` sampled stale texture data — every frame
"worked" while every frame was wrong (green-spotted robot, wrong
fixture colors). Per-env smoke checks missed it because "produces
frames" was the bar; comparing against the native renderer is the bar
that catches it, along with RGB/BGR swaps and upside-down cameras.

Procedure notes: .claude/skills/adding-an-adapter, "Camera capture
obligations". Every new adapter with ``capture_frame`` gets a test
here.
"""

from __future__ import annotations

import numpy as np
import pytest


def _assert_matches_native(native, ours, *, tolerance=8.0):
    """Same state, two pipelines: near-identical and not flipped."""
    assert native.shape == ours.shape
    diff = float(np.abs(native.astype(int) - ours.astype(int)).mean())
    flipped_diff = float(
        np.abs(native[::-1].astype(int) - ours.astype(int)).mean()
    )
    assert diff < tolerance, (
        "adapter capture diverges from the native render "
        f"(mean abs diff {diff:.1f}); texture/color corruption or a "
        "channel swap"
    )
    assert flipped_diff > diff, (
        "adapter capture matches the FLIPPED native render better than "
        "the upright one; orientation bug"
    )


@pytest.mark.robocasa
def test_robocasa_capture_matches_native_render():
    robosuite = pytest.importorskip("robosuite")
    pytest.importorskip("robocasa")
    from specola.adapters.robocasa import _wrap

    env = robosuite.make(
        "PickPlaceCounterToCabinet",
        robots=["PandaOmron"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_center"],
        camera_heights=240,
        camera_widths=320,
        ignore_done=True,
    )
    try:
        observation = env.reset()
        native = np.flipud(
            observation["robot0_agentview_center_image"]
        )
        adapter = _wrap(env)
        adapter.reset()
        ours = adapter.capture_frame(
            camera="robot0_agentview_center", width=320, height=240
        )
        _assert_matches_native(native, ours)
    finally:
        env.close()


@pytest.mark.libero
def test_libero_capture_matches_native_render():
    pytest.importorskip("libero")
    import os

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(1)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=240, camera_widths=320
    )
    try:
        observation = env.reset()
        native = np.flipud(observation["agentview_image"])
        from specola.adapters.libero import LIBEROAdapter

        adapter = LIBEROAdapter(env)
        adapter.reset()
        ours = adapter.capture_frame(
            camera="agentview", width=320, height=240
        )
        _assert_matches_native(native, ours)
    finally:
        env.close()


@pytest.mark.dmcontrol
def test_dmcontrol_capture_matches_native_render():
    pytest.importorskip("dm_control")
    from specola.adapters.dmcontrol import _make_dmcontrol

    adapter = _make_dmcontrol("walker:walk", None, False)
    adapter.reset()
    # Pin both pipelines to the same camera by name — camera 0 being
    # the adapter's default is incidental, not contractual.
    camera = adapter.camera_names()[0]
    ours = adapter.capture_frame(camera=camera, width=320, height=240)
    native = adapter.env._dm_env.physics.render(
        camera_id=camera, width=320, height=240
    )
    _assert_matches_native(native, ours, tolerance=1.0)


@pytest.mark.metaworld
def test_metaworld_capture_matches_native_render():
    pytest.importorskip("metaworld")
    import mujoco

    from specola.adapters.metaworld import _make_metaworld
    from specola.adapters.mujoco import _raw_struct

    adapter = _make_metaworld(None, None, False)
    adapter.reset()
    model = _raw_struct(adapter._model, "_model")
    data = _raw_struct(adapter._data, "_data")
    renderer = mujoco.Renderer(model, height=240, width=320)
    try:
        # behindGripper is correctly authored (no flip expected), so
        # the raw renderer IS the native reference for color/texture;
        # corner's deliberate flip has its own test in
        # test_metaworld_env.py.
        renderer.update_scene(data, camera="behindGripper")
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        native = renderer.render()
        ours = adapter.capture_frame(
            camera="behindGripper", width=320, height=240
        )
        _assert_matches_native(native, ours, tolerance=1.0)
    finally:
        renderer.close()
