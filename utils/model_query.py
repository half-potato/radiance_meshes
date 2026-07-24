"""ModelQuery: one iNGP query per training step, shared by every consumer.

Motivation (profiled on london @3.4M tets, see utils/section_prof.py): the
training step used to run the network more than once per iteration — the
masked cell-value query inside render() plus a FULL-table density pass hidden
in ``model.vertex_normals`` (60% of the step once its hash-grid backward is
counted), and in rfds_rev a third full-table query for the shadow tracer.

``prepare_query`` performs the frustum masking (a byproduct of
``vertex_and_tile_shader``) and the single ``get_cell_values`` call, derives
the geometric vertex normals from the SAME density column, and returns
everything downstream consumers need. The renderers accept ``query=`` and
skip their internal queries; the training loops build the query once and hand
it to the rasterizer, the densification criteria, and (in rfds_rev, with
``full_table=True``) the shadow tracer.

Semantics notes:
* ``cell_values`` is RAW model output (pre ``activate_output`` aux /
  ``activate_aux``); the density column passes through activations untouched,
  so ``cell_values[:, 0]`` equals ``calc_tet_density()`` output exactly.
* With ``full_table=False`` the masked rows are zero. Geometric normals then
  weight frustum-invisible faces by zero — the same approximation
  ``renderer/ref_renderer.py`` has always used for full_inverse.
* ``full_table=True`` queries every tet (needed when shadow rays must
  traverse the whole scene); the rasterizer simply gathers its visible
  subset from the same tensor.
* FrozenTetModel has no ``compute_geometric_vertex_normals`` and a cheap
  ``vertex_normals`` property — the fallback path covers it.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from utils.section_prof import section


@dataclass
class ModelQuery:
    render_grid: RenderGrid
    tcam: dict
    sorted_tetra_idx: torch.Tensor
    tile_ranges: torch.Tensor
    vs_tetra: torch.Tensor
    circumcenter: torch.Tensor
    mask: torch.Tensor                  # [T] bool: tet touches >=1 screen tile
    cell_values: torch.Tensor           # [T, C] raw; masked rows zero unless full_table
    shs: Optional[torch.Tensor]
    vertex_normals: torch.Tensor        # geometric (shared density) + learned offsets
    full_table: bool


def prepare_query(camera, model, *, tile_size=4, min_t=0.1,
                  full_table: bool = False,
                  cell_values: Optional[torch.Tensor] = None,
                  vertex_normals: Optional[torch.Tensor] = None,
                  stochastic_subpixel: bool = False,
                  stochastic_seed: int = 0) -> ModelQuery:
    """Run the tile shader + the single model query for ``camera``.

    ``cell_values`` / ``vertex_normals`` overrides preserve legacy behavior
    for callers that supply their own table (render_debug, parity tests):
    with a caller-supplied table the normals fall back to the model property
    rather than the shared-density derivation, matching the old code path
    bit-for-bit.

    ``stochastic_subpixel`` enables size-proportional tet sampling (unbiased
    LOD): sub-pixel tets are randomly culled with prob ``1 - min(s_i, 1)`` and
    the survivors' DENSITY is reweighted by ``1/min(s_i, 1)`` (differentiably),
    so the expected render equals the full render. ``stochastic_seed`` (e.g. the
    training iteration) makes each call sample independently but reproducibly.
    Tets >= 1px are never culled and never reweighted -> bit-exact at high res.
    The density reweight is applied AFTER geometric normals are derived, so the
    normals still see the true (unsampled) density field.
    """
    device = model.device
    render_grid = RenderGrid(camera.image_height,
                             camera.image_width,
                             tile_height=tile_size,
                             tile_width=tile_size)
    tcam = dict(
        tile_height=tile_size,
        tile_width=tile_size,
        grid_height=render_grid.grid_height,
        grid_width=render_grid.grid_width,
        min_t=min_t,
        **camera.to_dict(device)
    )
    sample_p = None
    with section('render/vertex_tile_shader'):
        if stochastic_subpixel:
            sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _, sample_p = \
                vertex_and_tile_shader(model.indices, model.vertices, tcam,
                                       render_grid, stochastic_subpixel=True,
                                       stochastic_seed=stochastic_seed)
        else:
            sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _ = \
                vertex_and_tile_shader(model.indices, model.vertices, tcam,
                                       render_grid)

    shs = None
    caller_supplied = cell_values is not None
    if not caller_supplied:
        with section('render/cell_values'):
            if full_table:
                shs, cell_values = model.get_cell_values(
                    camera, all_circumcenters=circumcenter)
            else:
                cell_values = torch.zeros(
                    (mask.shape[0], model.feature_dim), device=device)
                if mask.sum() > 0 and model.mask_values:
                    shs, values = model.get_cell_values(
                        camera, mask, circumcenter[mask])
                    cell_values = cell_values.index_put((mask,), values)
                else:
                    shs, cell_values = model.get_cell_values(
                        camera, all_circumcenters=circumcenter)

    with section('render/vertex_normals'):
        if vertex_normals is None:
            if (not caller_supplied
                    and hasattr(model, 'compute_geometric_vertex_normals')):
                # The load-bearing dedup: derive from the SAME density the
                # rasterizer consumes -- no second full-table network pass,
                # and the normals gradient rides the shared autograd node.
                vertex_normals = model.compute_geometric_vertex_normals(
                    density=cell_values[:, 0]) + torch.cat([
                        model.interior_vertex_normals,
                        model.ext_vertex_normals])
            else:
                vertex_normals = model.vertex_normals

    # Russian-roulette density reweight: scale the density column by 1/p for the
    # surviving tets so the alpha-blend is importance-weighted (E == full render)
    # and dL/d(density_raw) picks up the same 1/p factor. Applied strictly AFTER
    # the geometric normals (above) so normals see the true, unsampled density.
    # p == 1 for >=1px / non-visible tets, so this is a no-op there (bit-exact).
    if sample_p is not None:
        density = cell_values[:, :1] / sample_p.unsqueeze(1)
        cell_values = torch.cat([density, cell_values[:, 1:]], dim=1)

    return ModelQuery(
        render_grid=render_grid, tcam=tcam,
        sorted_tetra_idx=sorted_tetra_idx, tile_ranges=tile_ranges,
        vs_tetra=vs_tetra, circumcenter=circumcenter, mask=mask,
        cell_values=cell_values, shs=shs, vertex_normals=vertex_normals,
        full_table=full_table,
    )
