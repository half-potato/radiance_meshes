"""Swappable densification criteria.

Each criterion produces a per-tet refinement score; a shared executor turns
scores into vertex insertions. This decouples the three questions the legacy
path (utils/densification.py) answers monolithically:

    1. WHERE is the discretization failing?   -> criterion.scores()
    2. WHICH tets get this round's budget?    -> select_top_k / legacy threshold
    3. WHERE inside the tet to insert?        -> criterion.split_points()

Criteria (a-posteriori error estimators, borrowed from the AMR/CFD world):

* ``dwr``        -- Dual-Weighted-Residual analog: accumulate |dL/d(cell_values)|
                    from the training backward passes over the densify window.
                    The adjoint weight the trainer already computes and discards;
                    view-importance-weighted for free. Carried on VERTICES so the
                    stats survive the Delaunay retriangulation churn (tet ids are
                    not stable between rounds; vertex ids are). Windowed + reset
                    each round, so there is no stale-error memory and no
                    over-densification ratchet: a fixed region stops scoring
                    the moment its gradients die.
* ``cell_jump``  -- Kelly-estimator analog: the jump of cell values across
                    interior faces. Where the underlying field varies faster
                    than the piecewise-constant tets can represent, adjacent
                    circumcenter samples disagree. View-independent (no camera
                    sampling starvation), stateless, recomputed fresh at each
                    decision from the current field.
* ``dwr_jump``   -- rank-normalized product of the two: DWR knows what matters
                    to the images, the jump knows where the field has structure
                    regardless of whether a camera looked this round.
* ``error_votes``-- adapter over the legacy RenderStats flow so the old
                    behavior participates in side-by-side comparisons.

Comparison tooling: ``compare_criteria`` computes pairwise top-k Jaccard
overlap and Spearman rank correlation between score vectors, so alternative
criteria can be evaluated in shadow mode inside a single run before being
promoted to the active criterion.
"""

from typing import Dict, Optional

import torch

from utils import safe_math
from utils.topo_utils import get_tet_adjacency


# =========================================================================
# Vertex-carried windowed accumulator
# =========================================================================

class VertexAccumulator:
    """Windowed per-vertex scalar buffer that survives retriangulation.

    Delaunay updates (every ``delaunay_interval`` iters) reconnect the SAME
    vertex set -- only splits append vertices. Stats scattered to a tet's four
    corners therefore persist across topology churn; a tet's score is read
    back by gathering its current corners. ``reset()`` at the end of each
    densify round keeps the window stale-free.
    """

    def __init__(self, n_vertices: int, device):
        self.buf = torch.zeros(int(n_vertices), device=device)

    def ensure_size(self, n_vertices: int):
        n = int(n_vertices)
        if n > self.buf.shape[0]:
            pad = torch.zeros(n - self.buf.shape[0], device=self.buf.device)
            self.buf = torch.cat([self.buf, pad])

    @torch.no_grad()
    def scatter_tet(self, indices: torch.Tensor, per_tet: torch.Tensor):
        """Add ``per_tet[t] / 4`` to each of tet t's corner vertices."""
        self.ensure_size(int(indices.max()) + 1)
        q = (per_tet.float() / 4.0)
        for c in range(4):
            self.buf.index_add_(0, indices[:, c].long(), q)

    @torch.no_grad()
    def gather_tet(self, indices: torch.Tensor) -> torch.Tensor:
        """Per-tet mean of the four corner values, sized to current buffer."""
        self.ensure_size(int(indices.max()) + 1)
        return self.buf[indices.long()].mean(dim=1)

    def reset(self):
        self.buf.zero_()


# =========================================================================
# Criteria
# =========================================================================

class DensifyCriterion:
    name = "base"

    def observe(self, model, render_pkg):
        """Called once per training iteration, before backward. Cheap."""

    def scores(self, model, camera=None) -> torch.Tensor:
        raise NotImplementedError

    def split_points(self, model, tet_ids: torch.Tensor) -> torch.Tensor:
        """Default insertion point: tet centroid."""
        return model.vertices[model.indices[tet_ids].long()].mean(dim=1)

    def reset(self):
        """Called after each densify decision (end of window)."""


class DWRCriterion(DensifyCriterion):
    """Accumulated |dL/d(cell_values)| per tet, vertex-carried, windowed.

    ``observe`` registers a tensor hook on ``render_pkg['cell_values']``; when
    the trainer's backward fires, the hook reduces the [T, C] gradient to
    per-tet row norms and scatters them to the corner vertices. No gradient
    tensor is retained -- the reduction happens while autograd already holds
    it. Costs one norm + four index_add per iteration.

    ``channels``: slice of cell-value columns to include in the norm. Default
    covers density + base RGB (columns 0:4), the terms every pipeline trains.
    """

    name = "dwr"

    def __init__(self, model, channels: slice = slice(0, 4)):
        self.channels = channels
        self.acc = VertexAccumulator(model.vertices.shape[0],
                                     model.vertices.device)

    def observe(self, model, render_pkg):
        cv = render_pkg.get("cell_values")
        if cv is None or not cv.requires_grad:
            return
        # Idempotence: a shared DWR instance can be observed twice per
        # iteration (standalone entry + product-criterion fan-out); a second
        # hook on the same tensor would double-count.
        flag = f"_dwr_hooked_{id(self):x}"
        if getattr(cv, flag, False):
            return
        try:
            setattr(cv, flag, True)
        except AttributeError:
            pass
        indices = model.indices  # snapshot valid when backward fires (same iter)
        acc, ch = self.acc, self.channels

        def _hook(grad):
            per_tet = grad[:, ch].norm(dim=1)
            acc.scatter_tet(indices, per_tet)
            return None

        cv.register_hook(_hook)

    def scores(self, model, camera=None) -> torch.Tensor:
        return self.acc.gather_tet(model.indices)

    def reset(self):
        self.acc.reset()


class CellJumpCriterion(DensifyCriterion):
    """Kelly-style inter-tet jump indicator on the current cell-value field.

    For every interior face, the jump is the channel-normalized difference of
    the two adjacent tets' cell values; a tet's score is its max face jump
    scaled by its characteristic length (vol^{1/3}) -- a large tet with the
    same jump is a worse offender. Color jumps between two effectively empty
    tets are meaningless, so they are gated by the pair's density.

    ``split_points`` biases the insertion toward the max-jump face (midpoint
    of face centroid and tet centroid) so the new vertex lands where the
    discretization is failing rather than at the geometric center.
    """

    name = "cell_jump"

    def __init__(self, density_gate: float = 0.05):
        self.density_gate = float(density_gate)
        self._faces_int: Optional[torch.Tensor] = None
        self._best_face: Optional[torch.Tensor] = None

    @torch.no_grad()
    def scores(self, model, camera=None) -> torch.Tensor:
        indices = model.indices.long()
        device = indices.device
        T = indices.shape[0]
        _, cv = model.get_cell_values(camera)

        faces, side = get_tet_adjacency(indices)
        interior = (side[:, 0] >= 0) & (side[:, 1] >= 0)
        a = side[interior, 0].long()
        b = side[interior, 1].long()
        faces_int = faces[interior].long()

        dens = cv[:, 0]
        rgb = cv[:, 1:4]
        d_jump = (dens[a] - dens[b]).abs()
        c_jump = (rgb[a] - rgb[b]).norm(dim=1)
        # Scale-free channel weights: normalize by global std so density and
        # color contribute comparably regardless of activation ranges.
        d_jump = safe_math.safe_div(d_jump, dens.std().clamp_min(1e-8))
        c_jump = safe_math.safe_div(c_jump, rgb.std().clamp_min(1e-8))
        # Gate color jumps by the pair's density: empty-vs-empty faces carry
        # no renderable color discontinuity.
        gate = (torch.maximum(dens[a], dens[b]) > self.density_gate).float()
        face_jump = d_jump + gate * c_jump

        # Per-tet max over incident faces (scatter amax), then a second pass
        # to recover which face achieved it so split_points can aim at it.
        # Duplicate writes on ties pick an arbitrary maximizing face -- fine.
        score = torch.zeros(T, device=device)
        score.scatter_reduce_(0, a, face_jump, reduce="amax", include_self=True)
        score.scatter_reduce_(0, b, face_jump, reduce="amax", include_self=True)
        best_face = torch.full((T,), -1, dtype=torch.long, device=device)
        face_ids = torch.arange(a.shape[0], device=device)
        for s in (a, b):
            hit = face_jump >= (score[s] - 1e-12)
            best_face[s[hit]] = face_ids[hit]

        # Characteristic length: vol^(1/3).
        v = model.vertices[indices]
        vol = torch.linalg.det(v[:, 1:] - v[:, :1]).abs() / 6.0
        score = score * vol.clamp_min(1e-30).pow(1.0 / 3.0)

        self._faces_int = faces_int
        self._best_face = best_face
        return score

    @torch.no_grad()
    def split_points(self, model, tet_ids: torch.Tensor) -> torch.Tensor:
        verts = model.vertices
        centroid = verts[model.indices[tet_ids].long()].mean(dim=1)
        if self._best_face is None:
            return centroid
        bf = self._best_face[tet_ids]
        has_face = bf >= 0
        pts = centroid.clone()
        if has_face.any():
            face_centroid = verts[self._faces_int[bf[has_face]]].mean(dim=1)
            pts[has_face] = 0.5 * centroid[has_face] + 0.5 * face_centroid
        return pts

    def reset(self):
        self._best_face = None
        self._faces_int = None


class ProductCriterion(DensifyCriterion):
    """Rank-normalized product of child criteria (default: dwr x cell_jump)."""

    def __init__(self, children, name="product"):
        self.children = list(children)
        self.name = name

    def observe(self, model, render_pkg):
        for c in self.children:
            c.observe(model, render_pkg)

    def scores(self, model, camera=None) -> torch.Tensor:
        out = None
        for c in self.children:
            s = rank_normalize(c.scores(model, camera=camera))
            out = s if out is None else out * s
        return out

    def split_points(self, model, tet_ids):
        # Prefer the jump child's face-aware placement when present.
        for c in self.children:
            if isinstance(c, CellJumpCriterion):
                return c.split_points(model, tet_ids)
        return self.children[0].split_points(model, tet_ids)

    def reset(self):
        for c in self.children:
            c.reset()


class ErrorVoteCriterion(DensifyCriterion):
    """Adapter over the legacy RenderStats so the old error-vote signal can be
    compared against the new criteria. Call ``set_stats`` with the round's
    collected stats before ``scores``. Score = max of rank-normalized within-
    and total-variance (mirrors the legacy OR of the two threshold masks)."""

    name = "error_votes"

    def __init__(self):
        self._stats = None

    def set_stats(self, stats):
        self._stats = stats

    def scores(self, model, camera=None) -> torch.Tensor:
        assert self._stats is not None, "set_stats() before scores()"
        s = self._stats
        s0, s1, s2 = s.total_var_moments.T
        mu = safe_math.safe_div(s1, s0)
        std = (safe_math.safe_div(s2, s0) - mu ** 2).clip(min=0)
        std[s0 < 1] = 0
        total_var = s0 * std
        total_var[(s.tet_view_count < 2) | (s0 < 1)] = 0
        within_var = s.top_ssim.sum(dim=1)
        return torch.maximum(rank_normalize(within_var),
                             rank_normalize(total_var))

    def reset(self):
        self._stats = None


# =========================================================================
# Selection + comparison
# =========================================================================

def rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Map scores to [0, 1] by rank; ties in zeros stay zero."""
    n = x.numel()
    if n == 0:
        return x
    ranks = torch.empty_like(x)
    order = x.argsort()
    ranks[order] = torch.arange(n, device=x.device, dtype=x.dtype)
    out = ranks / max(n - 1, 1)
    out[x <= 0] = 0.0
    return out


def select_top_k(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k highest strictly-positive scores."""
    k = int(min(k, int((scores > 0).sum())))
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.topk(scores, k).indices


@torch.no_grad()
def compare_criteria(score_map: Dict[str, torch.Tensor], k: int) -> Dict[str, float]:
    """Pairwise top-k Jaccard overlap + Spearman rank correlation.

    Meant for shadow-mode evaluation: run candidate criteria alongside the
    active one and log how differently they would spend the same budget.
    """
    out = {}
    names = sorted(score_map)
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            sa, sb = score_map[na], score_map[nb]
            ka = set(select_top_k(sa, k).tolist())
            kb = set(select_top_k(sb, k).tolist())
            denom = len(ka | kb)
            out[f"densify/jaccard_{na}_vs_{nb}"] = (
                len(ka & kb) / denom if denom else 1.0)
            ra = rank_normalize(sa.float())
            rb = rank_normalize(sb.float())
            ra = ra - ra.mean()
            rb = rb - rb.mean()
            denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
            out[f"densify/spearman_{na}_vs_{nb}"] = float(
                (ra * rb).sum() / denom)
    return out


def apply_scored_densification(model, tet_optim, criterion: DensifyCriterion,
                               scores: torch.Tensor, target_addition: int,
                               args) -> int:
    """Shared executor: top-k select, ask the criterion where to insert, split.

    Returns the number of insertions performed.
    """
    ids = select_top_k(scores, target_addition)
    if ids.numel() == 0:
        return 0
    pts = criterion.split_points(model, ids)
    tet_optim.split(pts, clone_indices=model.indices[ids], **args.as_dict())
    return int(ids.numel())


def build_criteria(spec: str, model) -> Dict[str, DensifyCriterion]:
    """Build criteria from a comma-separated spec, e.g. "dwr_jump,error_votes".

    Returns {name: criterion}. ``dwr_jump`` shares one DWR instance with a
    standalone ``dwr`` entry if both are requested (single hook, no double
    accumulation).
    """
    names = [s.strip() for s in spec.split(",") if s.strip()]
    out: Dict[str, DensifyCriterion] = {}
    dwr = None

    def get_dwr():
        nonlocal dwr
        if dwr is None:
            dwr = DWRCriterion(model)
        return dwr

    for n in names:
        if n == "dwr":
            out[n] = get_dwr()
        elif n == "cell_jump":
            out[n] = CellJumpCriterion()
        elif n == "dwr_jump":
            out[n] = ProductCriterion([get_dwr(), CellJumpCriterion()],
                                      name="dwr_jump")
        elif n == "error_votes":
            out[n] = ErrorVoteCriterion()
        else:
            raise ValueError(f"unknown densify criterion '{n}'")
    return out
