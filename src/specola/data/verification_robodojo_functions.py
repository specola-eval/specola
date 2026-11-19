"""Authored RoboDojo approximations over immutable producer operands.

The sampled estimator preserves Task 9A's vertex population, collision-mesh
preference, link selection, bounding-sphere rejection and float32 cutoff.
It is an overestimate off-vertex and does not measure penetration depth.
"""
import numpy as np

from specola.spec_functions import FunctionTarget, FunctionOutput


def _merge(collected, row):
    # Relocated from adapters._usd_scene.merge_meshes, without changing the
    # collision-hull preference or row-local vertex/index order.
    from specola.adapters._usd_scene import SCENE_COLLISION_GROUP
    mine = [entry for entry in collected if entry[0] == row]
    if not mine:
        return None
    chosen = [entry for entry in mine if entry[1] == SCENE_COLLISION_GROUP] or mine
    vertices, triangles, offset = [], [], 0
    for _, _, _, verts, tris in chosen:
        vertices.append(verts)
        triangles.append(tris.astype(np.int64) + offset)
        offset += int(verts.shape[0])
    return np.concatenate(vertices), np.concatenate(triangles)


def _mesh_validate(parameters):
    unknown = set(parameters) - {"distmax", "robot_links", "pairs", "device"}
    if unknown:
        raise ValueError(f"mesh_clearance: unknown parameters {sorted(unknown)}")
    cutoff = parameters.get("distmax")
    if type(cutoff) not in (int, float) or not (0 < cutoff < float("inf")):
        raise ValueError("mesh_clearance: distmax must be finite and positive, in metres")
    links = parameters.get("robot_links")
    if links is not None and (not isinstance(links, (tuple, list)) or any(not isinstance(link, str) for link in links)):
        raise ValueError("mesh_clearance: robot_links must be a list of names")
    pairs = parameters.get("pairs")
    if pairs is not None and any(len(pair) != 2 or any(not isinstance(label, str) for label in pair) for pair in pairs):
        raise ValueError("mesh_clearance: pairs must contain two labels each")


def _mesh_plan(scene, parameters):
    geometry = scene["geometry"]
    links = parameters.get("robot_links")
    known = {name for _, names, _, _ in geometry["robots"] for name in names}
    if links is not None and set(links) - known:
        raise ValueError(f"mesh_clearance: unknown robot links {sorted(set(links) - known)}; known: {sorted(known)}")
    rows_of, labels = {}, {}
    present = {entry[0] for entry in geometry["collected"]}
    for key, names, rows, label in geometry["robots"]:
        selected = [row for name, row in zip(names, rows) if row in present and (links is None or name in links)]
        if selected:
            rows_of[key] = selected
            labels[key] = label
    for key, _name, row, label in geometry["objects"]:
        if row in present:
            rows_of[key] = [row]
            labels[key] = label
    wanted = parameters.get("pairs")
    wanted = None if wanted is None else {frozenset(pair) for pair in wanted}
    pairs, matched = {}, set()
    keys = list(rows_of)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pair_labels = frozenset((labels[a], labels[b]))
            if wanted is not None and pair_labels not in wanted:
                continue
            matched.add(pair_labels)
            pairs[f"pair:{a}|{b}"] = (a, b)
    if wanted is not None and wanted - matched:
        raise ValueError(f"mesh_clearance: pairs have no mesh on both sides: {sorted(sorted(pair) for pair in wanted - matched)}")
    return rows_of, labels, pairs


def _mesh_prepare(*, inputs, parameters):
    scene = inputs["scene"]
    if scene is None:
        raise ValueError("mesh_clearance: native scene geometry is unavailable")
    rows_of, labels, pairs = _mesh_plan(scene, parameters)
    result = []
    for key, (a, b) in pairs.items():
        kind = "robot_pair" if a.startswith("robot:") and b.startswith("robot:") else "robot" if a.startswith("robot:") else "environment"
        result.append(FunctionTarget(
            key=key, label=f"{labels[a]}|{labels[b]}",
            tags=(f"a:{labels[a]}", f"b:{labels[b]}", kind, "method:vertex_sample"),
            position=0.5 * (scene["positions"][rows_of[a][0]] + scene["positions"][rows_of[b][0]]),
        ))
    return result


def mesh_clearance(*, inputs, parameters, cache, targets):
    from specola.functions_mesh import SampledMeshDistance
    scene = inputs["scene"]
    if "plan" not in cache:
        cache["plan"] = _mesh_plan(scene, parameters)
    rows_of, _labels, pairs = cache["plan"]
    if "query" not in cache:
        cache["query"] = SampledMeshDistance(device=parameters.get("device"))
        cache["radius"] = {}
    query, radius = cache["query"], cache["radius"]
    required_rows = {row for key in targets if key in pairs for owner in pairs[key] for row in rows_of[owner]}
    for row in sorted(required_rows - set(radius)):
        merged = _merge(scene["geometry"]["collected"], row)
        if merged is None:
            continue
        query.add(str(row), merged[0], merged[1])
        vertices = np.asarray(merged[0], dtype=float).reshape(-1, 3)
        radius[row] = float(np.linalg.norm(vertices, axis=1).max()) if len(vertices) else 0.0
    xpos, xquat = scene["positions"], scene["quaternions"]
    distmax = float(parameters["distmax"])
    centres64 = xpos.astype(np.float64)
    eps = 8.0 * float(np.finfo(np.float32).eps)
    updated, result = set(), {}
    for key in targets:
        if key not in pairs:
            continue
        a, b = pairs[key]
        best, clipped = distmax, True
        for row_a in rows_of[a]:
            for row_b in rows_of[b]:
                reach = radius.get(row_a, 0.0) + radius.get(row_b, 0.0)
                pa, pb = centres64[row_a], centres64[row_b]
                margin = eps * (float(np.abs(pa).max()) + float(np.abs(pb).max()) + reach)
                if float(np.linalg.norm(pa - pb)) > reach + distmax + margin:
                    continue
                for row in (row_a, row_b):
                    if row not in updated:
                        query.update(str(row), xpos[row], xquat[row])
                        updated.add(row)
                value, hit_cap = query.distance(str(row_a), str(row_b), distmax)
                if value < best:
                    best = value
                clipped = clipped and hit_cap
        result[key] = FunctionOutput(
            {"mesh_distance_sampled": float(best), "mesh_distance_clipped": 1.0 if clipped else 0.0},
            position=0.5 * (xpos[rows_of[a][0]] + xpos[rows_of[b][0]]),
        )
    return result


setattr(mesh_clearance, "prepare", _mesh_prepare)
setattr(mesh_clearance, "validate", _mesh_validate)
