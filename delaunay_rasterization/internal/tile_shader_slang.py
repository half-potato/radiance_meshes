import torch
from delaunay_rasterization.internal.slang.slang_modules import shader_manager
import math
from delaunay_rasterization.internal.sort_by_keys import sort_by_keys_cub
from icecream import ic
from delaunay_rasterization.internal.util import recombine_tensors, split_tensors

def augment(v):
    return torch.cat([v, torch.ones_like(v[:, :1])], dim=-1)

def augmentT(v):
    return torch.cat([v, torch.ones_like(v[:1])], dim=0)

def point2image(vertices, viewmat, projection_matrix, cam_pos, eps=torch.finfo(torch.float32).eps):
    cam_space_homo = viewmat @ augment(vertices).T
    cam_space_nohomo = cam_space_homo[:2] / (cam_space_homo[2:3].abs() + 1e-10)
    pixel_space = projection_matrix @ augmentT(cam_space_nohomo)
    inv_distance  = 1 / torch.clip(cam_space_homo[2:3].T, eps, None)
    return torch.cat([pixel_space[:2].T, inv_distance], dim=1)

def ceil_div(x, y):
    return (x + y - 1) // y

# Hard limit on (tet, tile) keys per render to prevent OOM / buffer stomps
# when geometry explodes mid-training. 500M keys = ~6GB (int64 key + int32
# id + the sorted copies). Module-level so tests can monkeypatch it.
MAX_KEYS = 500_000_000

# Conservative tile culling: emit a sentinel tile id for AABB-rect tiles that
# provably miss the projected tetrahedron, shrinking each real tile's tet list
# to only overlapping tets. This is BIT-EXACT (skipped tiles contribute nothing
# to the alpha blend) -- the key buffer size is unchanged, only tile membership.
# Module-level so tests can toggle it and assert cull-on == cull-off bitwise.
#
# Default OFF: measured on london (11.8M tets, 546x988, tile=4) the reachable
# key reduction at 4px tile granularity is small (3.5% at the safe 1px margin;
# 11.3% ceiling at 0px) because most tets are tiny (median ~2 tiles) so their
# hull already fills the AABB, and the keys that ARE culled are cheap ray-tet
# "miss" keys. The added tile-stage SAT cost (~+5ms) slightly exceeds the
# alpha-blend saving (~-2ms) there. It is a LARGE win on scenes dominated by
# big near-camera tets (e.g. ~42% key reduction on near-camera slivers), where
# the projected hull fills a small fraction of a wide AABB -- enable it there.
ENABLE_TILE_CULLING = False

# Safety margin (pixels) for the tile/hull overlap test. A tile is culled only
# when its pixel box is separated from the projected tet hull by MORE than this,
# absorbing the sub-pixel float mismatch between the vertex-shader projection
# and the alpha-blend's per-ray intersection. 1px is comfortably conservative
# on the default 4px tiles (validated bitwise on 11.8M real london tets).
CULL_MARGIN = 1.0

# Only run the per-tile overlap test on tets whose screen AABB spans at least
# this many tiles. Tiny tets (median ~2 tiles) have a hull that nearly fills
# their AABB -- no cullable corners -- so testing them is pure tile-stage
# overhead. Gating keeps the culling win (concentrated in large near-camera
# tets) while avoiding a net regression on small-tet-dominated scenes. Bit-exact
# for any threshold; skipped tets emit their full AABB like the un-culled path.
# 4 is free: a tet whose AABB spans <4 tiles has the AABB extremes on the hull,
# so every tile overlaps -- nothing to cull (measured: gate 1 and gate 4 give
# identical culling on london).
CULL_MIN_TILES = 4

def vertex_and_tile_shader(indices,
                           vertices,
                           cam,
                           render_grid,
                           stochastic_subpixel=False,
                           stochastic_seed=0,
                           stochastic_eps=1e-3):
    """Vertex + tile shader.

    Stochastic size-proportional tet sampling (opt-in, default off): each tet
    projects to a max pixel extent s_i; with p_i = min(s_i, 1) we include tet i
    iff a per-call uniform r_i < p_i, else cull it entirely (tiles_touched=0 ->
    no keys, no sort, no alpha-blend). Kept sub-pixel tets are reweighted by
    1/p_i on their DENSITY (done by the caller, differentiably). Russian-roulette
    importance sampling: in the thin/low-optical-depth regime where sub-pixel
    tets live, E[alpha] = p*(1-exp(-tau/p)) ~= tau -> unbiased. Tets with
    s_i >= 1px get p_i = 1: always kept, never reweighted -> bit-exact.

    When ``stochastic_subpixel`` is True the returned mask is the FULL visibility
    mask (pre-cull) so the caller's per-tet network query / geometric normals
    still see every visible tet; a 7th return value ``sample_p`` [T] (the per-tet
    p, 1 for >=1px and non-visible tets) carries the density reweight factor.
    """
    n_tetra = indices.shape[0]
    tiles_touched, rect_tile_space, vs_tetra, circumcenter, proj_verts, extent = VertexShader.apply(
        indices,
        vertices,
        cam,
        render_grid)
    # Boundary clones for the differentiable per-tet outputs: VertexShader
    # saves its outputs for backward, so ANY downstream in-place torch op on
    # the returned tensors bumps their version and poisons every held graph.
    # full_inverse backwards immediately and never notices; rfds_rev holds
    # the graph across its diffusion phases and died with "output 3 of
    # VertexShaderBackward ... version 22". Cloning here hands consumers
    # their own tensors while the saved originals stay at version 0.
    vs_tetra = vs_tetra.clone()
    circumcenter = circumcenter.clone()

    with torch.no_grad():
        min_x = rect_tile_space[..., 0]
        min_y = rect_tile_space[..., 1]
        max_x = rect_tile_space[..., 2]
        max_y = rect_tile_space[..., 3]

        # int64 throughout: each per-tet count fits in int32, but the running
        # total can pass 2^31 long before the MAX_KEYS check, and an int32
        # cumsum overflow hands generate_keys garbage offsets (overlapping /
        # negative write ranges -> corrupted keys).
        tiles_touched = (max_x - min_x).long() * (max_y - min_y).long()

        # Full (pre-sampling) visibility -- the query/normals must see every
        # visible tet even when the sampler culls some from the rasterization.
        visible_mask = tiles_touched > 0

        sample_p = None
        if stochastic_subpixel:
            # p_i = min(s_i, 1), floored at eps to bound the 1/p reweight (a
            # variance knob only -- the thin-regime estimate E=p*(tau/p)=tau is
            # unbiased for ANY p). p=1 for >=1px and non-visible tets: they are
            # always kept and never reweighted.
            p = extent.clamp(min=stochastic_eps, max=1.0)
            p = torch.where(visible_mask, p, torch.ones_like(p))
            g = torch.Generator(device=extent.device)
            g.manual_seed(int(stochastic_seed) & 0x7FFFFFFF)
            r = torch.rand(n_tetra, device=extent.device, generator=g)
            drop = (r >= p) & visible_mask
            # Cull dropped tets BEFORE keygen/sort: zero footprint -> no keys.
            # rect_tile_space is a VertexShader output saved for its backward
            # (circumcenter/vs_tetra ride the same node and DO get a gradient in
            # training), so we must NOT modify it in place -- that would bump its
            # autograd version and fail the backward version check. Clone first.
            # tiles_touched is already a fresh tensor from the multiply above.
            rect_tile_space = rect_tile_space.clone()
            rect_tile_space[drop] = 0
            tiles_touched[drop] = 0
            sample_p = p

        total_size_index_buffer = int(tiles_touched.sum().item()) if n_tetra > 0 else 0

        if total_size_index_buffer > MAX_KEYS:
            # Emergency brake for exploding geometry: drop the tets with the
            # largest tile footprints (the degenerate ones) until the key
            # budget fits. The old check only zeroed tets covering the WHOLE
            # grid, so e.g. 2M tets x 300 tiles each (600M keys) sailed
            # through to a ~7GB allocation.
            print(f"WARNING: Exploding geometry detected! Requested "
                  f"{total_size_index_buffer} keys. Dropping largest tets to fit {MAX_KEYS}.")
            sorted_counts, order = torch.sort(tiles_touched)
            kept_cumsum = torch.cumsum(sorted_counts, dim=0)
            n_keep = int(torch.searchsorted(
                kept_cumsum, torch.tensor(MAX_KEYS, dtype=torch.int64, device=kept_cumsum.device),
                right=True).item())
            bad = order[n_keep:]
            tiles_touched[bad] = 0
            rect_tile_space[bad] = 0
            total_size_index_buffer = int(tiles_touched.sum().item())

        index_buffer_offset = torch.cumsum(tiles_touched, dim=0)
        # generate_keys takes int32 offsets; safe now that the total is
        # guaranteed <= MAX_KEYS < 2^31.
        index_buffer_offset = index_buffer_offset.int()
        unsorted_keys = torch.zeros((total_size_index_buffer,),
                                    device="cuda",
                                    dtype=torch.int64)
        unsorted_tetra_idx = torch.zeros((total_size_index_buffer,),
                                         device="cuda",
                                         dtype=torch.int32)

        shader_manager.tile_shader.generate_keys(xyz_vs=vs_tetra,
                                                rect_tile_space=rect_tile_space,
                                                index_buffer_offset=index_buffer_offset,
                                                proj_verts=proj_verts,
                                                out_unsorted_keys=unsorted_keys,
                                                out_unsorted_gauss_idx=unsorted_tetra_idx,
                                                grid_height=render_grid.grid_height,
                                                grid_width=render_grid.grid_width,
                                                tile_width=render_grid.tile_width,
                                                tile_height=render_grid.tile_height,
                                                cull_enable=1 if ENABLE_TILE_CULLING else 0,
                                                cull_margin=float(CULL_MARGIN),
                                                cull_min_tiles=int(CULL_MIN_TILES)).launchRaw(
              blockSize=(256, 1, 1),
              gridSize=(ceil_div(n_tetra, 256), 1, 1)
        )

        highest_tile_id_msb = (render_grid.grid_width*render_grid.grid_height).bit_length()
        sorted_keys, sorted_tetra_idx = sort_by_keys_cub.sort_by_keys(
            unsorted_keys, unsorted_tetra_idx, highest_tile_id_msb)

        tile_ranges = torch.zeros((render_grid.grid_height*render_grid.grid_width, 2), 
                                  device="cuda",
                                  dtype=torch.int32)
        if total_size_index_buffer > 0:
            shader_manager.tile_shader.compute_tile_ranges(sorted_keys=sorted_keys,
                                                        out_tile_ranges=tile_ranges).launchRaw(
                    blockSize=(256, 1, 1),
                    gridSize=(ceil_div(total_size_index_buffer, 256), 1, 1)
            )

    if stochastic_subpixel:
        # Return the pre-cull visibility so the caller's query/normals cover
        # every visible tet; sample_p carries the 1/p density reweight.
        return (sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter,
                visible_mask, rect_tile_space, sample_p)

    mask = tiles_touched > 0
    return sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, rect_tile_space


class VertexShader(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, vertices, tcam, render_grid, device="cuda"):
        assert not torch.isnan(vertices).any(), "Tensor contains NaN values!"
        n_tetra = indices.shape[0]
        indices = indices.contiguous()
        vertices = vertices.contiguous()
        tiles_touched = torch.zeros((n_tetra), 
                                    device="cuda", 
                                    dtype=torch.int32)
        rect_tile_space = torch.zeros((n_tetra, 4), 
                                      device="cuda", 
                                      dtype=torch.int32)
        
        vs_tetra = torch.zeros((n_tetra, 3),
                               device="cuda",
                               dtype=torch.float)
        circumcenter = torch.zeros((n_tetra, 3),
                                   device="cuda",
                                   dtype=torch.float)
        # Projected pixel positions of the 4 tet verts (cols 0..7) + a cullable
        # flag (col 8), consumed by generate_keys for conservative tile culling.
        proj_verts = torch.zeros((n_tetra, 9),
                                 device="cuda",
                                 dtype=torch.float)
        # Per-tet max projected pixel extent s_i (max of screen-AABB w/h in px).
        # Drives the stochastic size-proportional sampler. 0 for culled tets.
        extent = torch.zeros((n_tetra,),
                             device="cuda",
                             dtype=torch.float)

        cull_enable = 1 if ENABLE_TILE_CULLING else 0
        shader_manager.vertex_shader.vertex_shader(indices=indices,
                                                  vertices=vertices,
                                                  out_tiles_touched=tiles_touched,
                                                  out_rect_tile_space=rect_tile_space,
                                                  out_proj_verts=proj_verts,
                                                  out_extent=extent,
                                                  cull_enable=cull_enable,
                                                  out_vs=vs_tetra,
                                                  out_circumcenter=circumcenter,
                                                  tcam=tcam).launchRaw(
                blockSize=(256, 1, 1),
                gridSize=(ceil_div(n_tetra, 256), 1, 1)
        )

        # inds, = torch.where(vs_tetra[:, 1] == 1)
        # if len(inds) > 1:
        #     j = vs_tetra[inds, 2].argmin()
        #     mask = torch.tensor([ind for i, ind in enumerate(inds) if i != j], device=vs_tetra.device)
        #     tiles_touched[mask] = 0
        #     rect_tile_space[mask] = 0
        #     vs_tetra[mask, 1] = 0

        tensors = [
            indices, vertices,
            tiles_touched, rect_tile_space, vs_tetra, circumcenter, proj_verts, extent
        ]
        non_tensor_data, tensor_data = split_tensors(tcam)
        ctx.non_tensor_data = non_tensor_data
        ctx.len_tensors = len(tensors)
        ctx.save_for_backward(*tensors, *tensor_data)
        ctx.render_grid = render_grid

        return tiles_touched, rect_tile_space, vs_tetra, circumcenter, proj_verts, extent

    @staticmethod
    def backward(ctx, grad_tiles_touched, grad_rect_tile_space, grad_vs_tetra, grad_circumcenter, grad_proj_verts, grad_extent):
        (indices, vertices,
            tiles_touched, rect_tile_space, vs_tetra, circumcenter, proj_verts, extent) = ctx.saved_tensors[:ctx.len_tensors]
        tcam = recombine_tensors(ctx.non_tensor_data, ctx.saved_tensors[ctx.len_tensors:])
        render_grid = ctx.render_grid

        n_tetra = indices.shape[0]

        grad_indices = torch.zeros_like(indices)
        grad_vertices = torch.zeros_like(vertices)

        shader_manager.vertex_shader.vertex_shader.bwd(indices=indices,
                                                      vertices=(vertices, grad_vertices),
                                                      tcam=tcam,
                                                      out_tiles_touched=tiles_touched,
                                                      out_rect_tile_space=rect_tile_space,
                                                      out_proj_verts=proj_verts,
                                                      out_extent=extent,
                                                      cull_enable=0,
                                                      out_vs=(vs_tetra, grad_vs_tetra),
                                                      out_circumcenter=(circumcenter, grad_circumcenter),
                                                      ).launchRaw(
                blockSize=(256, 1, 1),
                gridSize=(ceil_div(n_tetra, 256), 1, 1)
        )
        return grad_indices, grad_vertices, None, None


def vertex_tile_and_interval_shader(indices, vertices, vertex_normals, cell_values,
                                    cam, render_grid, aux_dim):
    """Convenience wrapper: vertex+tile shader + interval generation in one call."""
    from delaunay_rasterization.internal.alphablend_tiled_slang_interval import run_interval_generate

    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, rect_tile_space = \
        vertex_and_tile_shader(indices, vertices, cam, render_grid)

    interval_verts, interval_tet_data, interval_meta = run_interval_generate(
        indices, vertices, vertex_normals, cell_values,
        cam, aux_dim)

    return (sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask,
            rect_tile_space, interval_verts, interval_tet_data, interval_meta)
