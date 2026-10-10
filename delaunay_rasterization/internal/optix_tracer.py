"""Hardware-RT port of the legacy OptiX spline tracer, via torchoptix.

Same architecture as ``tracers/`` (the 200-fps original): a hardware BVH over
ALL tet faces streams t-ordered face crossings, and a spline state machine
keyed by ``side_index[face][crossing_direction]`` integrates segment by
segment -- no adjacency walking, no point-in-tet probes mid-ray, no dtlookup.
The original kept a 16-hit anyhit k-buffer in OptiX payload registers;
torchoptix only exposes closest-hit queries, so the k-buffer becomes a
**wavefront**: each round advances every still-active ray past its last
crossing and does one hardware traversal for the whole compacted batch.
Cost per crossing is one RT-core traversal (~ns) + its share of a ~45us
launch, instead of the software-BVH root restarts that made the pure Warp
port two orders of magnitude slower.

Each segment (the span between consecutive crossings) is classified by a
batched point-in-tet test of its midpoint against the two ``side_index``
tets of the face that closed it -- the same invariant as the Warp
face-marcher, so edge/vertex hits cannot mislabel anything downstream, and
the first segment of an inside-origin ray needs no dtlookup. The loop is
fully masked and sync-free (fixed-shape ops over all N rays, alive count
read back every 32 rounds); per-round GPU syncs, not traversal, dominated
the naive compacting version.

Segment math is bit-for-bit the slang walker's (and the Warp ports'):
``od = clamp(sigma*dt, MIN_OD, MAX_OD)``, phi/w0/w1 exponential weights,
endpoint-relu'd affine color, VSM m1/m2 moments with the small-od Taylor
correction, t_saturate interpolation, LOG_T_THRESHOLD = -5.54 = log(1/255).

Radiance forward is no-grad, but exact-transmittance shadow visibility has an
analytic density gradient (``vis = exp(-sum od)`` gives ``dvis/dsigma_i =
-L_i * vis``): :meth:`OptixSplineTracer.trace_grad_density` replays the
wavefront and scatters it -- the "recorded hit stream" backward the original
tracer had, re-traced instead of recorded.
"""

import torch

from utils.topo_utils import get_tet_adjacency

# Mirrored from slang/raytrace_shader.slang + safe-math.slang (and the
# original tracer's spline-machine.slang, which uses the same phi weights
# and LOG_CUTOFF = 5.54 with the opposite sign convention).
MIN_OD = 1.0e-8
MAX_OD = 88.0
LOG_T_THRESHOLD = -5.54  # log(1/255)


def _point_in_tets(p: torch.Tensor, tet_ids: torch.Tensor,
                   vertices: torch.Tensor, indices: torch.Tensor,
                   valid: torch.Tensor) -> torch.Tensor:
    """[M] bool: p[i] inside tet tet_ids[i]. ``valid`` masks tet_ids >= 0
    (invalid entries return False). Same-side-as-opposite-vertex test on
    all four faces; orientation-free."""
    ids = tet_ids.clamp_min(0).long()
    v = vertices[indices[ids].long()]                  # [M, 4, 3]
    inside = valid.clone()
    for j in range(4):
        rest = [k for k in range(4) if k != j]
        a, b, c = v[:, rest[0]], v[:, rest[1]], v[:, rest[2]]
        opp = v[:, j]
        n = torch.cross(b - a, c - a, dim=-1)
        s_p = ((p - a) * n).sum(-1)
        s_o = ((opp - a) * n).sum(-1)
        inside &= (s_p * s_o) >= 0.0
    return inside


def _round_math(hit_t_raw, hit_i_raw, launch_t, o_s, d_s, far_s,
                seg_start, log_t, rgb, m1, m2, t_sat, n_steps, prev_face,
                active, cv, tet_v0, verts, tets, side,
                t_max_hw: float, saturate_log_t: float):
    """One wavefront round of segment classification + integration.

    Pure tensor ops (no syncs, no data-dependent control flow) so it can be
    fused by torch.compile -- the eager version launches ~100 tiny kernels
    per round and the launch overhead, not math, dominates."""
    hit_f = hit_i_raw.long()
    miss = hit_t_raw >= t_max_hw * 0.999
    t_hit = launch_t + hit_t_raw
    beyond = t_hit > far_s                     # incl. misses
    seg_end = torch.where(beyond, far_s, t_hit)

    # Classify the segment (seg_start, seg_end) by its midpoint against
    # the two side tets of its END face; on a miss (only relevant when
    # t_far cut inside the volume) fall back to the face that OPENED the
    # segment. Same invariant as the Warp face-marcher: every segment is
    # classified independently, so edge/vertex hits cannot mislabel
    # anything downstream.
    neg1 = torch.full_like(hit_f, -1)
    zeros = torch.zeros_like(seg_start)
    cls_face = torch.where(miss, prev_face, hit_f)
    valid = active & (cls_face >= 0)
    cf = cls_face.clamp_min(0)
    ta = torch.where(valid, side[cf, 0].long(), neg1)
    tb = torch.where(valid, side[cf, 1].long(), neg1)
    mid_t = 0.5 * (seg_start + seg_end)
    mid = o_s + d_s * mid_t.unsqueeze(-1)
    in_a = _point_in_tets(mid, ta, verts, tets, ta >= 0)
    in_b = _point_in_tets(mid, tb, verts, tets, tb >= 0) & ~in_a
    seg_tet = torch.where(in_a, ta, neg1)
    seg_tet = torch.where(in_b, tb, seg_tet)

    do_seg = (seg_tet >= 0) & (seg_end > seg_start)
    # ---- masked segment integration (no data-dependent branches)
    tt = seg_tet.clamp_min(0)
    dens = cv[tt, 0]
    c0 = cv[tt, 1:4]
    g = cv[tt, 4:7]
    base = c0 + ((o_s - tet_v0[tt]) * g).sum(-1, keepdim=True)
    dc = (g * d_s).sum(-1, keepdim=True)
    t0v = seg_start
    t1v = seg_end
    c_s = (base + dc * t0v.unsqueeze(-1)).clamp_min(0.0)
    c_e = (base + dc * t1v.unsqueeze(-1)).clamp_min(0.0)
    od = (dens * (t1v - t0v)).clamp(MIN_OD, MAX_OD)
    od = torch.where(do_seg, od, torch.full_like(od, MIN_OD))
    a_t = torch.exp(-od)
    phi = (1.0 - a_t) / od
    w0 = phi - a_t
    w1 = 1.0 - phi
    c_pm = (c_e * w0.unsqueeze(-1) + c_s * w1.unsqueeze(-1)).clamp_min(0.0)
    T_j = torch.exp(log_t)
    w_seg = do_seg.float()
    rgb = rgb + c_pm * (T_j * w_seg).unsqueeze(-1)
    mom = do_seg & ((1.0 - a_t) > 0.002)
    corr_small = od * (1.0 / 6.0 - od * (1.0 / 12.0 - od / 40.0))
    corr_big = w0 + a_t - 2.0 * w0 / od
    corr = torch.where(od.abs() < 0.02, corr_small, corr_big)
    L = t1v - t0v
    m1 = m1 + torch.where(mom, T_j * (w0 * t1v + w1 * t0v), zeros)
    m2 = m2 + torch.where(
        mom, T_j * (w0 * t1v * t1v + w1 * t0v * t0v - L * L * corr), zeros)
    new_log_t = log_t - torch.where(do_seg, od, zeros)
    crossed = do_seg & (t_sat < 0) & (new_log_t < saturate_log_t)
    frac = (log_t - saturate_log_t) / od.clamp_min(1e-12)
    t_sat = torch.where(crossed, t0v + frac * (t1v - t0v), t_sat)
    log_t = new_log_t
    # ---- advance
    seg_start = torch.where(active, seg_end, seg_start)
    prev_face = torch.where(active & ~miss, hit_f, prev_face)
    n_steps = n_steps + active.int()
    active = active & ~miss & ~beyond & (log_t > LOG_T_THRESHOLD)
    # Park finished rays beyond the scene: their queries miss instantly
    # at the BVH root, keeping fixed-size launches cheap.
    t_march = torch.where(active, t_hit.clamp(max=t_max_hw),
                          torch.full_like(launch_t, t_max_hw))
    return (seg_start, log_t, rgb, m1, m2, t_sat, n_steps, prev_face,
            active, t_march)


def _grad_round_math(hit_t_raw, hit_i_raw, launch_t, o_s, d_s, far_s,
                     seg_start, log_t, prev_face, active, gscale,
                     cv, verts, tets, side, t_max_hw: float,
                     grad_dens):
    """Gradient counterpart of :func:`_round_math` for exact transmittance.

    Same segment classification and termination state, but instead of
    color/moment accumulation it scatters ``-L_seg * gscale`` (with
    ``gscale = grad_vis * vis`` fixed per ray) into ``grad_dens`` --
    the analytic ``dvis/ddens`` of ``vis = exp(-sum od)``, zeroed where
    the ``clamp(dens*L, MIN_OD, MAX_OD)`` pins the segment's od."""
    hit_f = hit_i_raw.long()
    miss = hit_t_raw >= t_max_hw * 0.999
    t_hit = launch_t + hit_t_raw
    beyond = t_hit > far_s
    seg_end = torch.where(beyond, far_s, t_hit)

    neg1 = torch.full_like(hit_f, -1)
    zeros = torch.zeros_like(seg_start)
    cls_face = torch.where(miss, prev_face, hit_f)
    valid = active & (cls_face >= 0)
    cf = cls_face.clamp_min(0)
    ta = torch.where(valid, side[cf, 0].long(), neg1)
    tb = torch.where(valid, side[cf, 1].long(), neg1)
    mid_t = 0.5 * (seg_start + seg_end)
    mid = o_s + d_s * mid_t.unsqueeze(-1)
    in_a = _point_in_tets(mid, ta, verts, tets, ta >= 0)
    in_b = _point_in_tets(mid, tb, verts, tets, tb >= 0) & ~in_a
    seg_tet = torch.where(in_a, ta, neg1)
    seg_tet = torch.where(in_b, tb, seg_tet)

    do_seg = (seg_tet >= 0) & (seg_end > seg_start)
    tt = seg_tet.clamp_min(0)
    dens = cv[tt, 0]
    L = seg_end - seg_start
    raw = dens * L
    od = torch.where(do_seg, raw.clamp(MIN_OD, MAX_OD),
                     torch.full_like(raw, MIN_OD))
    live = do_seg & (raw > MIN_OD) & (raw < MAX_OD)
    # index_add_ (atomic) instead of boolean-mask indexing: no host sync,
    # dead lanes contribute an explicit zero to tet 0.
    grad_dens.index_add_(0, tt, torch.where(live, -L * gscale, zeros))
    log_t = log_t - torch.where(do_seg, od, zeros)
    # ---- advance (identical to _round_math)
    seg_start = torch.where(active, seg_end, seg_start)
    prev_face = torch.where(active & ~miss, hit_f, prev_face)
    active = active & ~miss & ~beyond & (log_t > LOG_T_THRESHOLD)
    t_march = torch.where(active, t_hit.clamp(max=t_max_hw),
                          torch.full_like(launch_t, t_max_hw))
    return seg_start, log_t, prev_face, active, t_march


_ROUND_FN = None


def _get_round_fn():
    """Eager by default. torch.compile fusion measured ~1.1x warm on the
    bicycle scene (the cv/vertex gathers don't fuse, so the round stays
    launch-bound) while costing ~14 s of compile on first call -- set
    DSPLAT_OPTIX_COMPILE=1 to opt in for long runs anyway."""
    global _ROUND_FN
    if _ROUND_FN is None:
        import os
        if os.environ.get("DSPLAT_OPTIX_COMPILE", "0") == "1":
            try:
                _ROUND_FN = torch.compile(_round_math, dynamic=True)
            except Exception:
                _ROUND_FN = _round_math
        else:
            _ROUND_FN = _round_math
    return _ROUND_FN


class OptixSplineTracer:
    """Hardware-RT tet-mesh volume tracer (forward only).

    Args:
        vertices: [V, 3] float32 cuda tensor.
        indices: [T, 4] int cuda tensor (active tets).
    """

    def __init__(self, vertices: torch.Tensor, indices: torch.Tensor):
        import torchoptix  # lazy: module stays importable without it
        self._optix = torchoptix
        device = vertices.device
        self.device = device

        faces, side_index = get_tet_adjacency(indices.long())
        # Tensors referenced by the acceleration structure / kernels must
        # stay alive for the tracer's lifetime.
        self._verts = vertices.detach().to(torch.float32).contiguous()
        self._tets = indices.to(torch.int32).contiguous()
        self._faces = faces.to(torch.int32).contiguous()          # [F, 3]
        self._side = side_index.to(torch.int32).contiguous()      # [F, 2]

        self._handle = self._optix.build(
            self._verts.data_ptr(), self._faces.data_ptr(),
            self._verts.shape[0], self._faces.shape[0])

    def __del__(self):
        try:
            self._optix.release(self._handle)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def trace(self,
              rays: torch.Tensor,          # [N, 6]
              cell_values: torch.Tensor,   # [T, >=7]; only [:7] used
              min_t: float,
              t_far=1e30,                  # float OR [N] tensor
              max_iters: int = 1024,
              step_eps: float = 1e-5,
              saturate_log_t: float = -3.0,
              return_extras: bool = False):
        """Same contract as WarpFaceTracer.trace: returns [N, 4] rgba, and
        with ``return_extras`` a dict with m1/m2 (VSM moments), t_saturate
        and n_steps (crossings processed per ray)."""
        assert rays.dim() == 2 and rays.shape[1] == 6
        assert cell_values.shape[0] == self._tets.shape[0]
        N = rays.shape[0]
        device = rays.device
        o = rays[:, :3].detach().to(torch.float32).contiguous()
        d = rays[:, 3:].detach().to(torch.float32).contiguous()
        cv = cell_values.detach()[:, :7].to(torch.float32).contiguous()
        if torch.is_tensor(t_far):
            assert t_far.shape == (N,)
            t_far_t = t_far.detach().to(device=device, dtype=torch.float32)
        else:
            t_far_t = torch.full((N,), float(t_far), device=device)
        # Global hardware cap: finite per-ray caps + headroom; misses come
        # back with out_t == t_max.
        finite_far = t_far_t[torch.isfinite(t_far_t)]
        scene_span = (self._verts.max() - self._verts.min()).item()
        t_max_hw = float(min(1e12, max(
            finite_far.max().item() if finite_far.numel() else 0.0,
            scene_span * 4.0)) * 1.5 + 1.0)

        tet_v0 = self._verts[self._tets[:, 0].long()]             # [T, 3]

        # Wavefront state. The loop body is masked and sync-free
        # (fixed-shape ops; dead rays are parked beyond t_max so their
        # hardware queries miss instantly at the BVH root). Every
        # CHECK_EVERY rounds we pay one sync to read the alive set and
        # compact: per-round work then tracks the surviving rays, whose
        # count decays quickly (median ray finishes in a few hundred
        # crossings while the cap is max_iters). Per-round syncs -- not
        # RT-core traversal -- dominated the naive implementation.
        FULL = dict(
            log_t=torch.zeros(N, device=device),
            rgb=torch.zeros(N, 3, device=device),
            m1=torch.zeros(N, device=device),
            m2=torch.zeros(N, device=device),
            t_sat=torch.full((N,), -1.0, device=device),
            n_steps=torch.zeros(N, dtype=torch.int32, device=device),
        )
        idx = torch.arange(N, device=device)
        o_s, d_s, far_s = o, d, t_far_t
        seg_start = torch.full((N,), float(min_t), device=device)
        t_march = torch.full((N,), float(min_t), device=device)
        log_t = torch.zeros(N, device=device)
        rgb = torch.zeros(N, 3, device=device)
        m1 = torch.zeros(N, device=device)
        m2 = torch.zeros(N, device=device)
        t_sat = torch.full((N,), -1.0, device=device)
        n_steps = torch.zeros(N, dtype=torch.int32, device=device)
        # Face whose crossing opened the current segment; classifies the
        # final partial segment when t_far cuts inside the volume.
        prev_face = torch.full((N,), -1, dtype=torch.int64, device=device)
        active = torch.ones(N, dtype=torch.bool, device=device)

        out_t = torch.empty(N, dtype=torch.float32, device=device)
        out_i = torch.empty(N, dtype=torch.int32, device=device)
        CHECK_EVERY = 32

        def scatter_back():
            FULL["log_t"][idx] = log_t
            FULL["rgb"][idx] = rgb
            FULL["m1"][idx] = m1
            FULL["m2"][idx] = m2
            FULL["t_sat"][idx] = t_sat
            FULL["n_steps"][idx] = n_steps

        round_fn = _get_round_fn()
        for it in range(int(max_iters)):
            M = idx.numel()
            launch_t = t_march + step_eps
            o_adv = (o_s + d_s * launch_t.unsqueeze(-1)).contiguous()
            self._optix.trace_rays(
                self._handle, o_adv.data_ptr(), d_s.data_ptr(),
                out_t.data_ptr(), out_i.data_ptr(), t_max_hw, M)
            try:
                (seg_start, log_t, rgb, m1, m2, t_sat, n_steps, prev_face,
                 active, t_march) = round_fn(
                    out_t[:M], out_i[:M], launch_t, o_s, d_s, far_s,
                    seg_start, log_t, rgb, m1, m2, t_sat, n_steps,
                    prev_face, active, cv, tet_v0, self._verts, self._tets,
                    self._side, t_max_hw, saturate_log_t)
            except Exception:
                # torch.compile backend failure: fall back to eager for
                # the rest of the process.
                global _ROUND_FN
                _ROUND_FN = _round_math
                round_fn = _round_math
                (seg_start, log_t, rgb, m1, m2, t_sat, n_steps, prev_face,
                 active, t_march) = round_fn(
                    out_t[:M], out_i[:M], launch_t, o_s, d_s, far_s,
                    seg_start, log_t, rgb, m1, m2, t_sat, n_steps,
                    prev_face, active, cv, tet_v0, self._verts, self._tets,
                    self._side, t_max_hw, saturate_log_t)

            if (it + 1) % CHECK_EVERY == 0:
                keep = active.nonzero(as_tuple=True)[0]   # the one sync
                if keep.numel() == 0:
                    break
                if keep.numel() < idx.numel():
                    scatter_back()
                    idx = idx[keep]
                    o_s = o_s[keep].contiguous()
                    d_s = d_s[keep].contiguous()
                    far_s = far_s[keep].contiguous()
                    seg_start = seg_start[keep]
                    t_march = t_march[keep]
                    log_t = log_t[keep]
                    rgb = rgb[keep]
                    m1 = m1[keep]
                    m2 = m2[keep]
                    t_sat = t_sat[keep]
                    n_steps = n_steps[keep]
                    prev_face = prev_face[keep]
                    active = active[keep]

        scatter_back()
        out = torch.zeros((N, 4), dtype=torch.float32, device=device)
        out[:, :3] = FULL["rgb"]
        out[:, 3] = 1.0 - torch.exp(FULL["log_t"])
        if return_extras:
            return out, dict(m1=FULL["m1"], m2=FULL["m2"],
                             t_saturate=FULL["t_sat"],
                             n_steps=FULL["n_steps"])
        return out

    def trace_grad_density(self,
                           rays: torch.Tensor,          # [N, 6]
                           cell_values: torch.Tensor,   # [T, >=7]
                           grad_vis: torch.Tensor,      # [N] upstream dL/dvis
                           vis: torch.Tensor,           # [N] forward vis
                           min_t: float,
                           t_far=1e30,
                           max_iters: int = 1024,
                           step_eps: float = 1e-5) -> torch.Tensor:
        """Density gradient of exact transmittance ``vis = exp(-sum od)``.

        Replays :meth:`trace`'s wavefront (the "recorded hit stream" backward
        the module docstring anticipated -- re-traced instead of recorded)
        and scatters ``-L_seg * vis * grad_vis`` into each classified tet.
        Returns ``grad_density`` [T]. Deterministic given the same rays and
        cell table, so the replayed segments match the forward pass.
        """
        assert rays.dim() == 2 and rays.shape[1] == 6
        assert cell_values.shape[0] == self._tets.shape[0]
        N = rays.shape[0]
        device = rays.device
        o = rays[:, :3].detach().to(torch.float32).contiguous()
        d = rays[:, 3:].detach().to(torch.float32).contiguous()
        cv = cell_values.detach()[:, :7].to(torch.float32).contiguous()
        if torch.is_tensor(t_far):
            assert t_far.shape == (N,)
            t_far_t = t_far.detach().to(device=device, dtype=torch.float32)
        else:
            t_far_t = torch.full((N,), float(t_far), device=device)
        finite_far = t_far_t[torch.isfinite(t_far_t)]
        scene_span = (self._verts.max() - self._verts.min()).item()
        t_max_hw = float(min(1e12, max(
            finite_far.max().item() if finite_far.numel() else 0.0,
            scene_span * 4.0)) * 1.5 + 1.0)

        grad_dens = torch.zeros((self._tets.shape[0],), dtype=torch.float32,
                                device=device)
        gscale_full = (grad_vis.detach() * vis.detach()).to(
            device=device, dtype=torch.float32)

        idx = torch.arange(N, device=device)
        o_s, d_s, far_s, gscale = o, d, t_far_t, gscale_full
        seg_start = torch.full((N,), float(min_t), device=device)
        t_march = torch.full((N,), float(min_t), device=device)
        log_t = torch.zeros(N, device=device)
        prev_face = torch.full((N,), -1, dtype=torch.int64, device=device)
        active = torch.ones(N, dtype=torch.bool, device=device)

        out_t = torch.empty(N, dtype=torch.float32, device=device)
        out_i = torch.empty(N, dtype=torch.int32, device=device)
        CHECK_EVERY = 32

        for it in range(int(max_iters)):
            M = idx.numel()
            launch_t = t_march + step_eps
            o_adv = (o_s + d_s * launch_t.unsqueeze(-1)).contiguous()
            self._optix.trace_rays(
                self._handle, o_adv.data_ptr(), d_s.data_ptr(),
                out_t.data_ptr(), out_i.data_ptr(), t_max_hw, M)
            seg_start, log_t, prev_face, active, t_march = _grad_round_math(
                out_t[:M], out_i[:M], launch_t, o_s, d_s, far_s,
                seg_start, log_t, prev_face, active, gscale,
                cv, self._verts, self._tets, self._side, t_max_hw,
                grad_dens)

            if (it + 1) % CHECK_EVERY == 0:
                keep = active.nonzero(as_tuple=True)[0]   # the one sync
                if keep.numel() == 0:
                    break
                if keep.numel() < idx.numel():
                    idx = idx[keep]
                    o_s = o_s[keep].contiguous()
                    d_s = d_s[keep].contiguous()
                    far_s = far_s[keep].contiguous()
                    gscale = gscale[keep]
                    seg_start = seg_start[keep]
                    t_march = t_march[keep]
                    log_t = log_t[keep]
                    prev_face = prev_face[keep]
                    active = active[keep]

        return grad_dens


def render_rt(camera, model, min_t: float = 0.4, max_iters: int = 1024,
              tracer: "OptixSplineTracer | None" = None):
    """Full-frame traced render (the old evaluate_traced_optix role).

    Returns a dict with 'render' [3, H, W] and 'alpha' [H, W]; pass a
    prebuilt ``tracer`` to amortize the BVH across frames.
    """
    if tracer is None:
        tracer = OptixSplineTracer(model.vertices.detach(), model.indices)
    cds = camera.get_camera_space_directions(model.device)
    rays = camera.get_world_space_rays(cds, model.device)
    with torch.no_grad():
        _, cell_values = model.get_cell_values(camera)
        out = tracer.trace(rays, cell_values, min_t, max_iters=max_iters)
    H, W = camera.image_height, camera.image_width
    img = out.reshape(H, W, 4)
    return {
        'render': img[..., :3].permute(2, 0, 1),
        'alpha': img[..., 3],
        'tracer': tracer,
    }
