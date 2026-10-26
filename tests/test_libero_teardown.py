"""Interpreter-teardown regression for the non-rendering LIBERO helper."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.libero
def test_repeated_nonrendering_envs_exit_cleanly():
    pytest.importorskip("libero")
    code = textwrap.dedent(
        """
        from specola.adapters.libero import make_libero_env

        for task_id in range(3):
            env, _task = make_libero_env(
                suite="libero_object",
                task_id=task_id,
            )
            env.close()
        """
    )
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["MUJOCO_GL"] = "egl"
    environment["NUMBA_CACHE_DIR"] = (
        "/tmp/diagnostic-catalogue-numba"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        "stdout:\n{}\nstderr:\n{}".format(
            result.stdout,
            result.stderr,
        )
    )
    assert "free(): invalid pointer" not in result.stderr
