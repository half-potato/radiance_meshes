"""Warp-BVH boundary entry finder for the adjacency-walking ray tracer.

Replaces the only job the old OptiX tracer did that the slangtorch walker
cannot: locating, for a ray whose origin lies OUTSIDE the tet mesh, the
boundary face (and thus tet) through which it first enters. The volume
integration itself stays in the walker kernel (slang/raytrace_shader.slang);
this module only seeds its ``start_tet_ids``.

Rays originating inside the mesh must keep using dtlookup point location:
for an inside origin the nearest boundary hit is an *exit* face, and seeding
the walker there would skip everything before it.

Uses NVIDIA Warp (``warp-lang``) for the BVH build + ``wp.mesh_query_ray``
traversal -- pip-installable, driver-only, zero-copy with torch CUDA tensors.
"""

import torch
import warp as wp

# Face j is opposite vertex j; ordering must match utils.topo_utils.build_adj
# and kTetFaces in slang/raytrace_shader.slang.
_TET_FACES = ((0, 2, 1), (1, 2, 3), (0, 3, 2), (3, 0, 1))


def extract_boundary_faces(indices: torch.Tensor,
                           tet_adj: torch.Tensor):
    """Boundary triangles of a tet mesh.

    Args:
        indices: [T, 4] tet vertex ids.
        tet_adj: [T, 4] adjacency from ``build_adj`` (-1 marks boundary).

    Returns:
        face_tris: [F, 3] vertex ids of boundary triangles.
        face_tet:  [F] id of the (unique) tet owning each boundary face.
    """
    assert tet_adj.shape == indices.shape
    idx = indices.long()
    faces = torch.stack([idx[:, list(f)] for f in _TET_FACES], dim=1)  # [T,4,3]
    boundary = tet_adj < 0                                             # [T,4]
    face_tris = faces[boundary]
    face_tet = boundary.nonzero(as_tuple=False)[:, 0]
    return face_tris, face_tet


@wp.kernel
def _entry_kernel(mesh_id: wp.uint64,
                  rays: wp.array2d(dtype=wp.float32),    # [N, 6] (o, d)
                  face_tet: wp.array(dtype=wp.int32),
                  t_max: wp.float32,
                  out_tet: wp.array(dtype=wp.int32),
                  out_t: wp.array(dtype=wp.float32)):
    i = wp.tid()
    o = wp.vec3(rays[i, 0], rays[i, 1], rays[i, 2])
    d = wp.vec3(rays[i, 3], rays[i, 4], rays[i, 5])
    q = wp.mesh_query_ray(mesh_id, o, d, t_max)
    if q.result:
        out_tet[i] = face_tet[q.face]
        out_t[i] = q.t
    else:
        out_tet[i] = -1
        out_t[i] = -1.0


class BoundaryEntryFinder:
    """BVH over the boundary faces of a tet mesh, queried per ray.

    The returned start tet feeds ``TetrahedralRayTrace`` unchanged: the walker
    clips the ray against the start tet's four face planes itself, so the ray
    origin does not need to be advanced to the entry point.
    """

    def __init__(self, vertices: torch.Tensor,
                 indices: torch.Tensor,
                 tet_adj: torch.Tensor):
        wp.init()  # idempotent; device_from_torch needs the runtime up
        device = vertices.device
        face_tris, face_tet = extract_boundary_faces(indices, tet_adj)
        self.face_tet = face_tet.to(device=device, dtype=torch.int32).contiguous()
        # Warp views share memory with these tensors; keep them alive.
        self._verts = vertices.detach().to(torch.float32).contiguous()
        self._tris = face_tris.reshape(-1).to(device=device,
                                              dtype=torch.int32).contiguous()
        self._wp_device = wp.device_from_torch(device)
        self.mesh = wp.Mesh(
            points=wp.from_torch(self._verts, dtype=wp.vec3),
            indices=wp.from_torch(self._tris),
        )

    def refit(self, vertices: "torch.Tensor | None" = None):
        """Update the BVH after vertex motion (same topology)."""
        if vertices is not None:
            self._verts.copy_(vertices.detach().to(torch.float32))
        self.mesh.refit()

    def find_entry(self, rays: torch.Tensor, t_max: float = 1e30):
        """First boundary hit per ray.

        Args:
            rays: [N, 6] (origin, direction) tensor; origins must be outside
                the mesh boundary.

        Returns:
            start_tet: [N] int32, tet owning the first-hit boundary face,
                -1 where the ray misses the mesh entirely.
            t_enter: [N] float32 hit distance, -1.0 on miss.
        """
        assert rays.dim() == 2 and rays.shape[1] == 6
        N = rays.shape[0]
        rays_c = rays.detach().to(torch.float32).contiguous()
        start_tet = torch.empty(N, dtype=torch.int32, device=rays.device)
        t_enter = torch.empty(N, dtype=torch.float32, device=rays.device)
        wp.launch(
            _entry_kernel, dim=N,
            inputs=[self.mesh.id, wp.from_torch(rays_c),
                    wp.from_torch(self.face_tet), float(t_max)],
            outputs=[wp.from_torch(start_tet), wp.from_torch(t_enter)],
            device=self._wp_device,
        )
        return start_tet, t_enter
