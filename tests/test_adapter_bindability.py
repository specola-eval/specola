"""Static-slot bindability guard, run by every registered adapter.

This is the contract test C5 of
``docs/superpowers/plans/2026-09-02-never-bound-static-slot.md``: the guard
a new adapter runs under before it ships. A fail means a key the adapter
declares on one of its static classes cannot bind on any discovered entity
of that class, or compiles but is dropped once a real session installs its
extraction scope (the X15 never-bound gripper slot failure mode).

The ``robocasa`` parameter is a strict xfail (issue #122): RoboCasa builds
a different kitchen per construction from an OS-entropy-seeded generator,
so the probe cannot decide it until the factory takes a seed.

A newly registered adapter is covered here automatically, probed with
``scene=None``. It needs a ``_SCENES`` entry only when its factory demands
a task argument, and a ``_PRODUCER_MODULES`` entry only when its producer
library is optional in this environment. Producer absence is the only
thing that skips: once the producer imports, ``setup_unavailable`` is a
failure of the adapter, its factory, its reset, or one of the probe legs,
and is reported with the probe's own detail rather than skipped.

The two Isaac rows have a precondition importability cannot express, so
``_PRECONDITIONS`` names it: ``isaaclab``'s factory builds a gym task
inside an Omniverse app the caller has already launched, and
``robodojo``'s launches the app itself but needs a RoboDojo checkout
(``ROBODOJO_ROOT`` or the working directory). Without those the probe
would report a missing launcher or checkout as ``setup_unavailable``,
which is not a bindability verdict, so the row skips with that reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import warnings

import pytest

from specola.conformance.bindability import probe_for_registry_name
from specola.registry import available

_FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks" / "adapter_fidelity"

# Scenes matching the repo's certification-row fixtures, so these adapters
# reach the probe instead of skipping on "no --task given". Every other
# adapter keeps scene=None (its default fixture, or none at all).
_SCENES = {
    "dataset": str(_FIXTURES / "dataset_certification_fixture.csv"),
    "pybullet": "{}@fixed,{}".format(
        _FIXTURES / "pybullet_certification_robot.urdf",
        _FIXTURES / "pybullet_certification_box.urdf",
    ),
    # The tasks the two Isaac adapters were verified live against
    # (docs/ROADMAP.md, Isaac Sim and RoboDojo adapter entries).
    "isaaclab": "Isaac-Forge-PegInsert-Direct-v0",
    "robodojo": "stack_bowls",
}

# Module(s) whose absence means the adapter's producer library is not
# installed in this environment, and the only reason a row skips. Names
# left out (mock, dataset) need no external producer.
_PRODUCER_MODULES = {
    "pybullet": ("pybullet",),
    "dmcontrol": ("dm_control",),
    "mamujoco": ("gymnasium_robotics",),
    "libero": ("libero",),
    "robocasa": ("robocasa", "robosuite"),
    "metaworld": ("metaworld",),
    "isaaclab": ("isaaclab_tasks",),
    "carla": ("carla",),
    "robodojo": ("isaaclab",),
}


def _omniverse_app_running() -> Optional[str]:
    try:
        import omni.kit.app  # type: ignore
    except Exception:
        return "no Omniverse app in this process"
    app = omni.kit.app.get_app()
    if app is None or not app.is_running():
        return "the isaaclab factory needs an Omniverse app launched by the caller"
    return None


def _robodojo_checkout() -> Optional[str]:
    from specola.adapters.robodojo import _robodojo_root

    try:
        _robodojo_root()
    except ValueError as exc:
        return str(exc)
    return None


# A reason to skip, or None to probe. Checked after the producer imports.
_PRECONDITIONS = {
    "isaaclab": _omniverse_app_running,
    "robodojo": _robodojo_checkout,
}

# Only registry names with an identically named marker in pyproject.toml's
# [tool.pytest.ini_options] get one.
_MARKERS = {
    "robocasa": pytest.mark.robocasa,
    "libero": pytest.mark.libero,
    "metaworld": pytest.mark.metaworld,
    "pybullet": pytest.mark.pybullet,
    "dmcontrol": pytest.mark.dmcontrol,
    "robodojo": pytest.mark.robodojo,
}


# RoboCasa builds a different kitchen per construction (robosuite seeds a
# per-environment RNG from OS entropy), so the probe cannot decide it until
# the factory takes a seed; strict so the marker must be removed once #122
# lands and the probe starts deciding RoboCasa.
_XFAIL = {
    "robocasa": pytest.mark.xfail(
        strict=True,
        reason=(
            "RoboCasa samples a new kitchen per construction (robosuite "
            "per-env RNG from OS entropy); the probe cannot decide it "
            "until the factory takes a seed, issue #122"
        ),
    ),
}


def _params():
    for name in sorted(available()):
        marks = tuple(
            m[name] for m in (_MARKERS, _XFAIL) if name in m
        )
        yield pytest.param(name, id=name, marks=marks)


@pytest.mark.parametrize("name", list(_params()))
def test_registered_adapter_binds_every_static_slot(name):
    for module_name in _PRODUCER_MODULES.get(name, ()):
        pytest.importorskip(module_name)
    precondition = _PRECONDITIONS.get(name)
    reason = precondition() if precondition is not None else None
    if reason is not None:
        pytest.skip("{}: {}".format(name, reason))

    artifact = probe_for_registry_name(name, _SCENES.get(name))

    assert artifact.outcome != "setup_unavailable", (
        "{}: the producer imports, so the probe should decide; it reported "
        "{}".format(name, artifact.detail or "no detail")
    )

    if artifact.outcome == "not_applicable":
        warnings.warn(
            "{}: bindability not covered (open class vocabulary); "
            "declare supported_classes to get this guard".format(name)
        )
        return

    failures = [
        "{}/{}: {}".format(
            result.case.entity_class, result.case.component, result.reason
        )
        for result in artifact.results
        if result.status == "fail"
    ]
    assert artifact.outcome != "fail", "\n".join(failures)
