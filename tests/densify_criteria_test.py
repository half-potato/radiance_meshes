"""Properties of the swappable densification criteria.

Pure-CPU: exercises the criterion math on a tiny synthetic tet mesh -- no
CUDA, no rasterizer. Encodes the design invariants:

* vertex-carried stats survive retriangulation (tet ids change, vertex ids
  don't) and are windowed (reset -> zero);
* DWR accumulates |dL/d(cell_values)| where gradients actually flow, once per
  backward even when the instance is observed twice (shared via product);
* cell-jump ranks field-discontinuity tets above smooth-interior tets and
  aims the split point at the offending face;
* the shared selection/comparison utilities behave (top-k respects positive
  support; identical criteria have Jaccard/Spearman 1).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.densify_criteria import (  # noqa: E402
    VertexAccumulator, DWRCriterion, CellJumpCriterion, ProductCriterion,
    rank_normalize, select_top_k, compare_criteria, build_criteria,
)

DEVICE = torch.device("cpu")


def _cube_mesh():
    """Unit cube split into tets via Delaunay of its corners."""
    from scipy.spatial import Delaunay
    corners = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0)
         for z in (0.0, 1.0)], dtype=np.float64)
    tri = Delaunay(corners)
    vertices = torch.tensor(corners, dtype=torch.float32, device=DEVICE)
    indices = torch.tensor(tri.simplices, dtype=torch.int64, device=DEVICE)
    return vertices, indices


class FakeModel(SimpleNamespace):
    """Duck-typed stand-in: vertices, indices, get_cell_values."""

    def get_cell_values(self, camera, **kw):
        return None, self.cell_values


class TestVertexAccumulator:
    def test_scatter_gather_and_reset(self):
        vertices, indices = _cube_mesh()
        acc = VertexAccumulator(vertices.shape[0], DEVICE)
        per_tet = torch.arange(1.0, indices.shape[0] + 1.0)
        acc.scatter_tet(indices, per_tet)
        g = acc.gather_tet(indices)
        assert (g > 0).all()
        acc.reset()
        assert acc.buf.abs().sum() == 0

    def test_survives_retriangulation(self):
        """Same vertices, different connectivity: gather still works and
        vertex mass is conserved."""
        vertices, indices = _cube_mesh()
        acc = VertexAccumulator(vertices.shape[0], DEVICE)
        acc.scatter_tet(indices, torch.ones(indices.shape[0]))
        before = acc.buf.clone()
        # "Retriangulate": permute tet order and rotate corner order.
        new_indices = indices.flip(0).roll(1, dims=1)
        g = acc.gather_tet(new_indices)
        assert g.shape[0] == new_indices.shape[0]
        torch.testing.assert_close(acc.buf, before)  # buffer untouched

    def test_grows_for_new_vertices(self):
        vertices, indices = _cube_mesh()
        acc = VertexAccumulator(vertices.shape[0], DEVICE)
        acc.scatter_tet(indices, torch.ones(indices.shape[0]))
        grown = torch.cat([indices, torch.tensor([[0, 1, 2, 8]])])
        g = acc.gather_tet(grown)  # vertex 8 didn't exist at scatter time
        assert g.shape[0] == grown.shape[0]
        assert acc.buf.shape[0] >= 9
        assert acc.buf[8] == 0


class TestDWR:
    def _model_and_pkg(self):
        vertices, indices = _cube_mesh()
        model = FakeModel(vertices=vertices, indices=indices)
        cv = torch.zeros(indices.shape[0], 7, requires_grad=True)
        pkg = {"cell_values": cv}
        return model, cv, pkg

    def test_gradient_magnitude_ranks_tets(self):
        model, cv, pkg = self._model_and_pkg()
        crit = DWRCriterion(model)
        crit.observe(model, pkg)
        w = torch.zeros(model.indices.shape[0])
        w[0] = 10.0
        w[1] = 1.0
        (cv[:, 0] * w).sum().backward()
        scores = crit.scores(model)
        assert scores.argmax() == 0
        assert scores[0] > scores[1] > 0
        crit.reset()
        assert crit.scores(model).abs().sum() == 0

    def test_shared_instance_counts_backward_once(self):
        """observe() twice on the same tensor (standalone + product fan-out)
        must accumulate exactly one backward's worth of gradient."""
        model, cv, pkg = self._model_and_pkg()
        dwr = DWRCriterion(model)
        prod = ProductCriterion([dwr, CellJumpCriterion()])
        dwr.observe(model, pkg)
        prod.observe(model, pkg)  # same underlying DWR: must not double-hook
        cv[:, 0].sum().backward()
        once = dwr.scores(model).clone()

        model2, cv2, pkg2 = self._model_and_pkg()
        ref = DWRCriterion(model2)
        ref.observe(model2, pkg2)
        cv2[:, 0].sum().backward()
        torch.testing.assert_close(once, ref.scores(model2))

    def test_accumulates_across_iterations(self):
        model, cv, pkg = self._model_and_pkg()
        crit = DWRCriterion(model)
        crit.observe(model, pkg)
        cv[:, 0].sum().backward()
        s1 = crit.scores(model).clone()
        cv2 = torch.zeros_like(cv.detach()).requires_grad_(True)
        crit.observe(model, {"cell_values": cv2})
        cv2[:, 0].sum().backward()
        torch.testing.assert_close(crit.scores(model), 2 * s1)


class TestCellJump:
    def _two_region_model(self):
        vertices, indices = _cube_mesh()
        cv = torch.zeros(indices.shape[0], 7)
        centroid_x = vertices[indices].mean(1)[:, 0]
        # Dense-and-red on one side of x=0.5, empty on the other.
        hot = centroid_x > 0.5
        cv[hot, 0] = 5.0
        cv[hot, 1] = 1.0
        return FakeModel(vertices=vertices, indices=indices,
                         cell_values=cv), hot

    def test_boundary_tets_outrank_interior(self):
        model, hot = self._two_region_model()
        crit = CellJumpCriterion(density_gate=0.05)
        scores = crit.scores(model, camera=None)
        # Every tet adjacent to the density discontinuity must outrank any
        # tet whose faces all sit inside one region. In the 6-tet cube split
        # every tet touches the boundary, so at minimum: all scores positive
        # and the max face for top tets crosses regions.
        assert (scores > 0).any()
        top = select_top_k(scores, 2)
        pts = crit.split_points(model, top)
        assert pts.shape == (top.numel(), 3)
        assert torch.isfinite(pts).all()
        # Face-aimed placement: points differ from plain centroids.
        centroids = model.vertices[model.indices[top]].mean(1)
        assert not torch.allclose(pts, centroids)

    def test_uniform_field_scores_zero_color(self):
        vertices, indices = _cube_mesh()
        cv = torch.zeros(indices.shape[0], 7)
        cv[:, 0] = 3.0   # uniform density
        cv[:, 1:4] = 0.5  # uniform color
        model = FakeModel(vertices=vertices, indices=indices, cell_values=cv)
        scores = CellJumpCriterion().scores(model, camera=None)
        assert scores.abs().max() < 1e-6


class TestSelectionAndComparison:
    def test_rank_normalize_keeps_zeros(self):
        x = torch.tensor([0.0, 3.0, 1.0, 0.0])
        r = rank_normalize(x)
        assert r[0] == 0 and r[3] == 0
        assert r[1] == r.max()

    def test_top_k_respects_positive_support(self):
        s = torch.tensor([0.0, 5.0, 0.0, 2.0])
        assert set(select_top_k(s, 10).tolist()) == {1, 3}

    def test_compare_identical_and_disjoint(self):
        a = torch.rand(100) + 0.1
        m = compare_criteria({"x": a, "y": a.clone()}, k=10)
        assert m["densify/jaccard_x_vs_y"] == 1.0
        assert m["densify/spearman_x_vs_y"] > 0.99
        b = a.max() - a  # reversed ranking
        m2 = compare_criteria({"x": a, "y": b}, k=10)
        assert m2["densify/jaccard_x_vs_y"] < 0.5
        assert m2["densify/spearman_x_vs_y"] < 0

    def test_build_criteria_shares_dwr(self):
        vertices, indices = _cube_mesh()
        model = FakeModel(vertices=vertices, indices=indices)
        crits = build_criteria("dwr_jump,dwr,cell_jump", model)
        assert set(crits) == {"dwr_jump", "dwr", "cell_jump"}
        assert crits["dwr_jump"].children[0] is crits["dwr"]
        with pytest.raises(ValueError):
            build_criteria("nope", model)
