"""
Vertex-sampled nearest distance between rigid triangle meshes, on Warp.

Isaac Sim exposes no mesh-to-mesh distance to Python: PhysX's geometry
queries (``PxGeometryQuery::pointDistance``, ``PxGjkQuery``) are not
bound, ``computePenetration`` refuses mesh/mesh pairs, and the contact
report's ``separation`` only exists inside the contact offset (4 cm by
default). Warp, which ships with every Isaac install, does have a
point-to-mesh closest-point query over a BVH. This module composes it
into a pair distance: every vertex of A against B's surface, every
vertex of B against A's surface, minimum of both.

That is a SAMPLED distance. It is exact whenever the closest approach
touches a vertex on either side, and an over-estimate otherwise (an edge
of B grazing a flat face of A with no B vertex nearby). Consumers report
it as ``mesh_distance_sampled``, never as an exact clearance. A point
inside the other mesh reports zero: penetration depth is not measured.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat_wxyz)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


_KERNEL_CACHE: Dict[str, Any] = {}


def _kernel():
    """Compile the query kernel once per process."""
    if "kernel" in _KERNEL_CACHE:
        return _KERNEL_CACHE["kernel"]
    import warp as wp

    @wp.kernel
    def nearest_to_mesh(
        points: wp.array(dtype=wp.vec3),
        mesh: wp.uint64,
        max_dist: float,
        out: wp.array(dtype=float),
    ):
        i = wp.tid()
        p = points[i]
        q = wp.mesh_query_point(mesh, p, max_dist)
        if q.result:
            if q.sign < 0.0:
                wp.atomic_min(out, 0, 0.0)
            else:
                closest = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
                wp.atomic_min(out, 0, wp.length(p - closest))

    _KERNEL_CACHE["kernel"] = nearest_to_mesh
    return nearest_to_mesh


class SampledMeshDistance:
    """
    Registry of posed meshes with a pairwise sampled-distance query.

    ``add`` registers a mesh in its body frame; ``update`` moves it
    (world pose, quaternion ``(w, x, y, z)``) and refits its BVH;
    ``distance`` answers for a pair. Vertices are transformed on the
    host with NumPy and copied to the device each update, so the cost is
    linear in vertex count per step. Meshes with many thousands of
    vertices are worth decimating before ``add``.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        import warp as wp

        wp.config.quiet = True
        wp.init()
        self._wp = wp
        self._device = device or wp.get_device()
        self._local: Dict[str, np.ndarray] = {}
        self._points: Dict[str, Any] = {}
        self._meshes: Dict[str, Any] = {}
        self._out = wp.zeros(1, dtype=float, device=self._device)

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(self._meshes)

    def add(self, key: str, vertices: np.ndarray, triangles: np.ndarray) -> None:
        wp = self._wp
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        triangles = np.ascontiguousarray(triangles, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            raise ValueError(f"{key}: vertices must be (N, 3) with N > 0")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
            raise ValueError(f"{key}: triangles must be (T, 3) with T > 0")
        if int(triangles.max()) >= vertices.shape[0] or int(triangles.min()) < 0:
            raise ValueError(f"{key}: triangle index out of range")
        points = wp.array(vertices, dtype=wp.vec3, device=self._device)
        indices = wp.array(triangles.reshape(-1), dtype=wp.int32, device=self._device)
        self._local[key] = vertices.astype(np.float64)
        self._points[key] = points
        self._meshes[key] = wp.Mesh(points=points, indices=indices)

    def update(self, key: str, position: np.ndarray, quat_wxyz: np.ndarray) -> None:
        local = self._local[key]
        world = local @ _quat_to_matrix(np.asarray(quat_wxyz)).T + np.asarray(
            position, dtype=np.float64
        )
        self._points[key].assign(world.astype(np.float32))
        self._meshes[key].refit()

    def distance(self, a: str, b: str, max_distance: float) -> Tuple[float, bool]:
        """
        Sampled nearest distance between ``a`` and ``b`` in metres, capped
        at ``max_distance``, and whether the cap was hit (no sampled
        vertex on either side lies within it).
        """
        if max_distance <= 0.0 or not np.isfinite(max_distance):
            raise ValueError("max_distance must be finite and positive")
        wp = self._wp
        kernel = _kernel()
        # The kernel works in float32 and admits a point only strictly
        # inside its search radius. The accumulator starts at infinity so a
        # miss is unmistakable, and the radius is the first float32 at or
        # above the cutoff so every distance below the cutoff is admitted;
        # a hit at or beyond the cutoff (possible when the radius rounded
        # up) counts as capped.
        radius = np.float32(max_distance)
        if float(radius) < max_distance:
            radius = np.nextafter(radius, np.float32(np.inf))
        self._out.fill_(float("inf"))
        for src, dst in ((a, b), (b, a)):
            points = self._points[src]
            wp.launch(
                kernel,
                dim=points.shape[0],
                inputs=[points, self._meshes[dst].id, float(radius), self._out],
                device=self._device,
            )
        value = float(self._out.numpy()[0])
        if not np.isfinite(value) or value >= max_distance:
            return float(max_distance), True
        return value, False
