"""The state tier of an Isaac Lab scene, shared by the Isaac adapters.

Pure functions over an ``InteractiveScene`` (or an object shaped like
one) and the state dictionary its ``get_state`` returns, so the
``isaaclab`` and ``robodojo`` adapters build one state and one signature
the same way. A state is what ``get_state(is_relative=False)`` reports
for one environment: root poses and velocities and joint positions and
velocities of articulations, root poses and velocities of rigid
objects, nodal positions and velocities of deformable objects, and the
state of each surface gripper. Rigid-object collections are not in
``get_state`` (Isaac Lab 6.1.17, ``interactive_scene.py:720``), so a
scene that holds one is not covered by this state.

The signature records what decides whether a state row means the same
thing in another process: the leaf layout, a digest of the environment
configuration, the USD layers the stage uses and the realised physical
parameters of the monitored environment (masses, inertia tensors and shape
materials, where task events randomise them). See
``docs/superpowers/plans/2026-10-13-state-replay.md``, Task 1.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Iterator, List, Mapping, Optional, Tuple

import numpy as np

# get_state kind -> the InteractiveScene property holding those assets.
_FAMILIES = {
    "articulation": "articulations",
    "deformable_object": "deformable_objects",
    "rigid_object": "rigid_objects",
    "gripper": "surface_grippers",
}

# A memory address in a repr, which differs between processes.
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _host(value: Any) -> np.ndarray:
    """One Isaac array (tensor, ProxyArray, warp array or NumPy) as a host NumPy array."""
    # ProxyArray publishes its tensor as ``.torch`` and forwards ``.numpy``
    # to a tensor method that raises on CUDA memory, so ``.torch`` goes first.
    tensor = getattr(value, "torch", value)
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    to_numpy = getattr(value, "numpy", None)   # warp arrays
    return np.asarray(to_numpy() if callable(to_numpy) else value)


def _dtype_name(value: Any) -> str:
    return str(getattr(getattr(value, "torch", value), "dtype")).removeprefix("torch.")


def _leaves(state: Mapping) -> Iterator[Tuple[str, str, str, Any]]:
    """``(asset, kind, key, value)`` for every leaf, in ``get_state`` order."""
    for kind, assets in state.items():
        for asset, leaves in assets.items():
            if isinstance(leaves, Mapping):
                for key, value in leaves.items():
                    yield asset, kind, key, value
            else:
                # A surface gripper's state is one tensor, not a dictionary.
                yield asset, kind, "", leaves


def layout(state: Mapping) -> List[list]:
    """``[asset, kind, key, shape, dtype]`` per leaf for one environment, in ``get_state`` order."""
    return [
        [asset, kind, key, [int(n) for n in tuple(value.shape)[1:]], _dtype_name(value)]
        for asset, kind, key, value in _leaves(state)
    ]


def state_length(rows: List[list]) -> int:
    return int(sum(int(np.prod(shape)) for _, _, _, shape, _ in rows))


def flatten(state: Mapping, env_index: int) -> np.ndarray:
    """The ``env_index`` row of every leaf, as float64, concatenated in layout order."""
    parts = [
        _host(value)[env_index].astype(np.float64).ravel()
        for _, _, _, value in _leaves(state)
    ]
    return np.concatenate(parts) if parts else np.zeros(0)


def unflatten(rows: List[list], flat: np.ndarray) -> dict:
    """The ``get_state`` dictionary for one environment, leading dimension 1.

    Each leaf gets its recorded data type: the articulation writers assert
    float32 (Isaac Lab 6.1.17, ``write_joint_position_to_sim_index``).
    """
    import torch

    flat = np.asarray(flat, dtype=np.float64)
    expected = state_length(rows)
    if flat.shape != (expected,):
        raise ValueError(
            f"state of shape {flat.shape}; this scene's state has length {expected}"
        )
    state: dict = {}
    offset = 0
    for asset, kind, key, shape, dtype in rows:
        size = int(np.prod(shape))
        values = torch.as_tensor(
            flat[offset:offset + size].reshape([1, *shape]), dtype=getattr(torch, dtype)
        )
        offset += size
        bucket = state.setdefault(kind, {})
        if key == "":
            bucket[asset] = values
        else:
            bucket.setdefault(asset, {})[key] = values
    return state


def gripper_refusal(scene: Any, num_envs: int) -> Optional[str]:
    """Why this scene has no state tier, or None.

    ``InteractiveScene.reset_to`` passes a surface gripper's state to
    ``set_grippers_command`` without ``env_ids`` (Isaac Lab 6.1.17,
    ``interactive_scene.py:712``), so a one-environment row would set
    every environment's gripper. With one environment that is the same
    gripper; with more it is not.
    """
    if getattr(scene, "surface_grippers", None) and num_envs > 1:
        return (
            "no state tier: the scene has surface grippers and "
            f"{num_envs} environments, and Isaac Lab's reset_to applies a "
            "gripper state to every environment, not only the monitored one"
        )
    return None


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(value: Any) -> Any:
    """A configuration value in a form whose canonical JSON is stable across processes.

    Callables become ``module.qualname``, an absolute path naming a local
    file becomes that file's SHA-256, tuples become lists, floats their
    ``repr`` (JSON has no infinity), array-likes their lists, and any other
    object its ``repr`` without memory addresses.
    """
    if isinstance(value, Mapping):
        return {str(key): normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise(item) for item in value]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        if os.path.isabs(value) and os.path.isfile(value):
            return _file_sha256(value)
        return value
    if callable(value):
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{getattr(value, '__module__', None)}.{qualname}"
    if hasattr(value, "tolist"):
        return normalise(value.tolist())
    return _ADDRESS.sub("", repr(value))


def config_sha256(config: Mapping) -> str:
    """SHA-256 of the canonical JSON of a normalised configuration dictionary."""
    text = json.dumps(normalise(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def layer_records(stage: Any) -> List[list]:
    """One record per layer the stage uses, sorted, with no process-specific identifier.

    A local file is ``["file", path, sha256]``; an anonymous (in-memory)
    layer, whose ``anon:0x...`` identifier differs between equivalent
    stages, is ``["anon", sha256 of its exported text]``; anything else
    (a Nucleus or other remote identifier) is ``["remote", identifier]``,
    identified by name only because its bytes are not fetched.
    """
    records = []
    for layer in stage.GetUsedLayers():
        if layer.anonymous:
            text = layer.ExportToString()
            records.append(["anon", hashlib.sha256(text.encode()).hexdigest()])
        elif layer.realPath and os.path.isfile(layer.realPath):
            records.append(["file", layer.realPath, _file_sha256(layer.realPath)])
        else:
            records.append(["remote", layer.identifier])
    return sorted(records)


def physics_sha256(scene: Any, rows: List[list], env_index: int) -> str:
    """SHA-256 over the realised physical parameters of the monitored environment.

    For every asset in the layout: ``data.body_mass``, ``data.body_inertia``
    and ``root_view.get_material_properties()`` (static friction, dynamic
    friction, restitution per shape), each where the asset has it. These
    are where startup and reset events write randomised masses and
    friction (FORGE: ``forge_env_cfg.py`` events).
    """
    digest = hashlib.sha256()
    seen = set()
    for asset, kind, _, _, _ in rows:
        if (kind, asset) in seen:
            continue
        seen.add((kind, asset))
        obj = getattr(scene, _FAMILIES[kind])[asset]
        data = getattr(obj, "data", None)
        view = getattr(obj, "root_view", None) or getattr(obj, "root_physx_view", None)
        materials = getattr(view, "get_material_properties", None)
        sources = (
            ("body_mass", getattr(data, "body_mass", None)),
            ("body_inertia", getattr(data, "body_inertia", None)),
            ("material_properties", materials() if callable(materials) else None),
        )
        for name, value in sources:
            if value is None:
                continue
            array = np.ascontiguousarray(_host(value)[env_index])
            digest.update(f"{kind}/{asset}/{name}/{array.dtype}/{array.shape};".encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def signature(scene: Any, config: Mapping, env_index: int, task: Optional[str]) -> dict:
    """The state signature of one environment of ``scene`` (plan Task 1 contract)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        engine_version: Optional[str] = version("isaaclab")
    except PackageNotFoundError:   # a stand-in scene in a test
        engine_version = None
    rows = layout(scene.get_state(is_relative=False))
    stage = getattr(scene, "stage", None)
    return {
        "engine": "isaaclab",
        "version": engine_version,
        "task": task,
        "layout": rows,
        "cfg_sha256": config_sha256(config),
        "layers": layer_records(stage) if stage is not None else [],
        "physics_sha256": physics_sha256(scene, rows, env_index),
        "length": state_length(rows),
    }
