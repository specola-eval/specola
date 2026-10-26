"""
USD mesh export shared by the Isaac-family adapters' 3D-panel hooks.

``scene_geometry`` on any USD-backed adapter is the same job: walk mesh
prims under a set of body roots, bake each mesh's prim-to-body transform
into its vertices, fan-triangulate, sort collision hulls from visual
meshes, and pack everything into the dashboard's geometry pair. The
adapters differ only in how they find the body roots and read poses, so
that part stays with them and this module holds the rest.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Dashboard 3D-panel geom groups. The client shows a checkbox per group;
# mapping collision hulls to a hidden group 0 and visual meshes to a
# visible group 1 mirrors the robosuite convention the MuJoCo adapters
# already export, so the panel behaves the same across simulators.
SCENE_COLLISION_GROUP = 0
SCENE_VISUAL_GROUP = 1

# USD render purposes drawn as visual geometry. Renderers draw "default"
# and "proxy" by default and omit "render" (final-quality) and "guide";
# Isaac authors collision hulls with purpose "guide"
# (isaaclab.sim.schemas sets it explicitly to hide simulation meshes).
SCENE_VISUAL_PURPOSES = frozenset({"default", "proxy"})

SCENE_FALLBACK_RGBA = (0.7, 0.7, 0.7, 1.0)

# Shader inputs probed for a base colour when no displayColor is
# authored, in order: Omniverse MDL (OmniPBR) convention first, then
# UsdPreviewSurface, then the common PBR spelling.
SCENE_COLOR_INPUTS = ("diffuse_color_constant", "diffuseColor", "baseColor")

# (body_row, group, rgba, vertices (N, 3) float32, triangles (T, 3) uint32)
CollectedMesh = Tuple[int, int, List[float], np.ndarray, np.ndarray]


def fan_triangulate(
    face_vertex_counts: Any,
    face_vertex_indices: Any,
) -> np.ndarray:
    """
    Flatten USD polygon faces into ``(T, 3)`` uint32 triangles.

    Triangle ``j`` of an n-gon is ``(v0, v[j+1], v[j+2])`` -- the same fan
    Hydra's ``HdMeshUtil`` applies, so the export matches what USD's own
    renderer would draw for convex faces. Faces with fewer than three
    vertices contribute nothing.
    """
    counts = np.asarray(face_vertex_counts, dtype=np.int64)
    indices = np.asarray(face_vertex_indices, dtype=np.int64)
    if counts.size == 0 or indices.size == 0:
        return np.zeros((0, 3), dtype=np.uint32)
    # Offsets into the flat index list are defined by every face,
    # including degenerate ones that are then dropped.
    offsets = np.cumsum(counts) - counts
    triangles_per_face = counts - 2
    keep = triangles_per_face > 0
    offsets = offsets[keep]
    triangles_per_face = triangles_per_face[keep]
    if triangles_per_face.size == 0:
        return np.zeros((0, 3), dtype=np.uint32)
    total = int(triangles_per_face.sum())
    face_of_triangle = np.repeat(
        np.arange(triangles_per_face.size), triangles_per_face
    )
    first_triangle = np.cumsum(triangles_per_face) - triangles_per_face
    corner = np.arange(total) - np.repeat(first_triangle, triangles_per_face)
    base = offsets[face_of_triangle]
    return np.stack(
        (indices[base], indices[base + corner + 1], indices[base + corner + 2]),
        axis=1,
    ).astype(np.uint32)


def bake_transform(
    points: np.ndarray,
    matrix: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    Apply a Gf 4x4 (row-vector convention: translation in row 3) to
    ``(N, 3)`` points; returns float32 vertices and whether the linear
    part mirrors (negative determinant), in which case triangle winding
    must be reversed to keep front faces outward.
    """
    linear = matrix[:3, :3]
    baked = points @ linear + matrix[3, :3]
    return baked.astype(np.float32), bool(np.linalg.det(linear) < 0)


def mesh_rgba(prim: Any) -> List[float]:
    """
    Best-effort colour for one mesh prim: authored ``displayColor``, else
    the bound material's base colour, else neutral grey.

    Colour is presentation-only -- a wrong colour cannot mislabel state --
    so unresolved material graphs degrade to the fallback rather than
    failing the export. The MDL render context is probed before the
    universal one because Omniverse assets bind MDL shaders.
    """
    from pxr import UsdGeom, UsdShade

    gprim = UsdGeom.Gprim(prim)
    alpha = 1.0
    opacity = gprim.GetDisplayOpacityAttr().Get()
    if opacity is not None and len(opacity):
        alpha = float(opacity[0])
    color = gprim.GetDisplayColorAttr().Get()
    if color is not None and len(color):
        first = color[0]
        return [float(first[0]), float(first[1]), float(first[2]), alpha]
    try:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material:
            for context in ("mdl", ""):
                shader, _, _ = material.ComputeSurfaceSource(context)
                if not shader:
                    continue
                for name in SCENE_COLOR_INPUTS:
                    shader_input = shader.GetInput(name)
                    if not shader_input:
                        continue
                    value = shader_input.Get()
                    if value is not None and len(value) >= 3:
                        return [
                            float(value[0]),
                            float(value[1]),
                            float(value[2]),
                            alpha,
                        ]
    except Exception:  # noqa: BLE001 - cosmetic lookup over arbitrary graphs
        pass
    return list(SCENE_FALLBACK_RGBA)


def stage_is_metres_z_up(stage: Any) -> bool:
    """
    The client renders metres, z-up, the frame Isaac physics views report
    poses in. Refusing an oddly authored stage beats exporting a
    mislabelled scene.
    """
    from pxr import UsdGeom

    return (
        UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        and float(UsdGeom.GetStageMetersPerUnit(stage)) == 1.0
    )


class MeshCollector:
    """
    Gather the mesh prims under body roots, one call per body row.

    ``link_roots`` are every body root in the plan: link prims can be
    NESTED (a child link inside a parent link's subtree), and an
    unrestricted walk from the parent would collect the child's meshes
    twice, once against each pose row. ``frame_prim`` is the prim whose
    frame the vertices are baked into; ``None`` bakes world coordinates,
    which is what a static fixture against an identity pose row wants.
    """

    def __init__(self, link_roots: Set[str]) -> None:
        from pxr import Usd, UsdGeom

        self._xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        self._link_roots = set(link_roots)
        self.collected: List[CollectedMesh] = []
        self.skipped: Counter = Counter()

    def collect(self, root_prim: Any, row: int, frame_prim: Any) -> None:
        from pxr import Usd, UsdGeom

        root_path = str(root_prim.GetPath())
        iterator = iter(
            Usd.PrimRange(
                root_prim,
                Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate),
            )
        )
        for prim in iterator:
            prim_path = str(prim.GetPath())
            if prim_path != root_path and prim_path in self._link_roots:
                iterator.PruneChildren()
                continue
            if not prim.IsA(UsdGeom.Mesh):
                if prim.IsA(UsdGeom.Gprim):
                    self.skipped[f"gprim_{prim.GetTypeName()}"] += 1
                continue
            imageable = UsdGeom.Imageable(prim)
            purpose = str(imageable.ComputePurpose())
            if purpose == "guide":
                group = SCENE_COLLISION_GROUP
            elif purpose in SCENE_VISUAL_PURPOSES:
                group = SCENE_VISUAL_GROUP
            else:
                self.skipped[f"purpose_{purpose}"] += 1
                continue
            # Invisibility only vetoes visual meshes: an invisible
            # variant must not be drawn as the scene. Collision hulls
            # are often additionally marked invisible, but they land in
            # the hidden group anyway and hiding them entirely would
            # defeat exporting them.
            if (
                group == SCENE_VISUAL_GROUP
                and str(imageable.ComputeVisibility()) == "invisible"
            ):
                self.skipped["invisible"] += 1
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not points or not counts or not indices:
                self.skipped["empty_mesh"] += 1
                continue
            if frame_prim is not None:
                matrix, _ = self._xform_cache.ComputeRelativeTransform(
                    prim, frame_prim
                )
            else:
                matrix = self._xform_cache.GetLocalToWorldTransform(prim)
            vertices, mirrored = bake_transform(
                np.asarray(points, dtype=np.float64),
                np.asarray(matrix, dtype=np.float64),
            )
            triangles = fan_triangulate(counts, indices)
            left_handed = str(mesh.GetOrientationAttr().Get()) == "leftHanded"
            if mirrored != left_handed:
                triangles = np.ascontiguousarray(triangles[:, ::-1])
            self.collected.append((row, group, mesh_rgba(prim), vertices, triangles))


def count_meshes_outside(stage: Any, env_path: str) -> int:
    """Mesh prims under ``/World`` that are not under this environment's prim."""
    from pxr import Usd, UsdGeom

    world_prim = stage.GetPrimAtPath("/World")
    if not (world_prim and world_prim.IsValid()):
        return 0
    return sum(
        1
        for prim in Usd.PrimRange(
            world_prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        )
        if prim.IsA(UsdGeom.Mesh)
        and not str(prim.GetPath()).startswith(env_path + "/")
    )


def pack_scene(
    bodies: List[Dict[str, Any]],
    collected: List[CollectedMesh],
    skipped: Dict[str, int],
) -> dict:
    """
    Pack collected meshes into the dashboard recording format's geometry
    pair: a body/geom table plus one packed vertex-and-index buffer.
    Geom offsets are identity because every prim-to-body transform is
    already baked into the vertices, so only body poses move per step.
    """
    geoms: List[Dict[str, Any]] = []
    meshes: List[Dict[str, int]] = []
    vertex_blocks: List[np.ndarray] = []
    face_blocks: List[np.ndarray] = []
    vert_offset = 0
    face_offset = 0
    for body_row, group, rgba, vertices, triangles in collected:
        meshes.append(
            {
                "vert_offset": vert_offset,
                "vert_count": int(vertices.shape[0]),
                "face_offset": face_offset,
                "face_count": int(triangles.shape[0]),
            }
        )
        geoms.append(
            {
                "body": body_row,
                "type": "mesh",
                "size": [0.0, 0.0, 0.0],
                "pos": [0.0, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "rgba": rgba,
                "group": group,
                "mesh": len(meshes) - 1,
            }
        )
        vertex_blocks.append(vertices)
        face_blocks.append(triangles)
        vert_offset += int(vertices.shape[0])
        face_offset += int(triangles.shape[0])

    vertex_bytes = (
        np.concatenate(vertex_blocks).astype("<f4").tobytes() if vertex_blocks else b""
    )
    face_bytes = (
        np.concatenate(face_blocks).astype("<u4").tobytes() if face_blocks else b""
    )
    return {
        "json": {
            "format_version": 1,
            "nbody": len(bodies),
            "ngeom": len(geoms),
            "bodies": bodies,
            "geoms": geoms,
            "meshes": meshes,
            "faces_start": len(vertex_bytes),
            "skipped": dict(skipped),
            "visible_groups": [SCENE_VISUAL_GROUP],
        },
        "meshes": vertex_bytes + face_bytes,
    }


def link_prim_paths(
    stage: Any,
    asset: Any,
    names: Tuple[str, ...],
    env_index: int,
) -> Tuple[Optional[str], ...]:
    """
    Per-body USD prim paths of an Isaac Lab articulation, in body order.

    The physics view's ``link_paths`` is authoritative: it is in body
    order and disambiguates nested same-name prims (FORGE's fixed asset
    holds ``forge_hole_8mm/forge_hole_8mm``). The fallback matches body
    names against the asset's subtree and resolves only unique hits: a
    body whose prim cannot be identified gets no geometry rather than
    somebody else's.
    """
    for view_name in ("root_view", "root_physx_view"):
        view = getattr(asset, view_name, None)
        paths = getattr(view, "link_paths", None) if view is not None else None
        if not paths:
            continue
        if isinstance(paths[0], (list, tuple)):
            if 0 <= env_index < len(paths):
                per_env = list(paths[env_index])
                if len(per_env) == len(names):
                    return tuple(str(path) for path in per_env)
        break

    root_path = getattr(getattr(asset, "cfg", None), "prim_path", None)
    if not root_path:
        return tuple(None for _ in names)
    root_prim = stage.GetPrimAtPath(root_path.replace("env_.*", f"env_{env_index}"))
    if not (root_prim and root_prim.IsValid()):
        return tuple(None for _ in names)

    from pxr import Usd

    subtree = list(
        Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
    )
    resolved: List[Optional[str]] = []
    for name in names:
        hits = [str(prim.GetPath()) for prim in subtree if prim.GetName() == name]
        resolved.append(hits[0] if len(hits) == 1 else None)
    return tuple(resolved)


def merge_meshes(
    collected: List[CollectedMesh], row: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    One ``(vertices, triangles)`` per body row for distance queries:
    collision hulls when the body has any, else its visual meshes.
    """
    mine = [entry for entry in collected if entry[0] == row]
    if not mine:
        return None
    hulls = [entry for entry in mine if entry[1] == SCENE_COLLISION_GROUP]
    chosen = hulls or mine
    vertices: List[np.ndarray] = []
    triangles: List[np.ndarray] = []
    offset = 0
    for _, _, _, verts, tris in chosen:
        vertices.append(verts)
        triangles.append(tris.astype(np.int64) + offset)
        offset += int(verts.shape[0])
    return np.concatenate(vertices), np.concatenate(triangles)
