"""Warp port of the OptiX spline tracer: BVH face-marching, no adjacency walk.

Replaces ``tracers/`` (the CMake/pybind11 OptiX build). Same algorithm shape
as the OptiX tracer: a BVH over ALL tet faces (interior + boundary), and each
ray marches its face crossings in t-order. The segment between two
consecutive crossings lies inside exactly one tet; we identify it with a
point-in-tet test on the segment midpoint against the two tets sharing the
hit face (``side_index`` from ``get_tet_adjacency``). Compared to the
adjacency walker (raytrace_shader.slang) this has no dependent chain to
corrupt: a sliver can misclassify only its own segment, never derail the
rest of the ray. No start tet / dtlookup needed -- inside and outside
origins both work.

A finite ``t_far`` ending *inside* a tet is handled by classifying the final
partial segment against the last face crossed (we are on its far side), so
shadow rays capped just past the surface integrate the surface volume
correctly.

The per-segment integral is bit-for-bit the walker's math (affine color
along the ray, exact exponential weights w0/w1, same MIN_OD/MAX_OD/LOG_T
constants), and the m1/m2 outputs replicate the walker's volumetric VSM
moments including the small-od Taylor correction -- so both the JAX
quadrature reference and the Chebyshev shadow path apply unchanged.

Radiance forward is no-grad like the walker, but exact-transmittance shadow
visibility has an analytic density gradient: :meth:`WarpFaceTracer.
trace_walk_grad` replays the walk and scatters ``-L_seg * vis * grad_vis``
per crossed tet. The single ray never crossing any face (origin and t_far
inside one tet, no crossing in between) contributes nothing -- there is no
face to classify against.
"""

import torch
import warp as wp

from utils.topo_utils import get_tet_adjacency

# Constants mirrored from slang/raytrace_shader.slang + safe-math.slang.
MIN_OD = wp.constant(1.0e-8)
MAX_OD = wp.constant(88.0)
MAX_VAL = wp.constant(1.0e20)
LOG_T_THRESHOLD = wp.constant(-5.54)  # log(1/255)


@wp.func
def _point_in_tet(p: wp.vec3,
                  tet: wp.int32,
                  verts: wp.array(dtype=wp.vec3),
                  tets: wp.array2d(dtype=wp.int32)) -> wp.bool:
    v0 = verts[tets[tet, 0]]
    v1 = verts[tets[tet, 1]]
    v2 = verts[tets[tet, 2]]
    v3 = verts[tets[tet, 3]]
    # p is inside iff for every face it lies on the same side as the
    # opposite vertex. Orientation-free, no winding assumption.
    n = wp.cross(v2 - v1, v3 - v1)
    if wp.dot(n, p - v1) * wp.dot(n, v0 - v1) < 0.0:
        return False
    n = wp.cross(v2 - v0, v3 - v0)
    if wp.dot(n, p - v0) * wp.dot(n, v1 - v0) < 0.0:
        return False
    n = wp.cross(v1 - v0, v3 - v0)
    if wp.dot(n, p - v0) * wp.dot(n, v2 - v0) < 0.0:
        return False
    n = wp.cross(v1 - v0, v2 - v0)
    if wp.dot(n, p - v0) * wp.dot(n, v3 - v0) < 0.0:
        return False
    return True


@wp.func
def _relu3(v: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.max(v[0], 0.0), wp.max(v[1], 0.0), wp.max(v[2], 0.0))


@wp.kernel
def _trace_kernel(mesh_id: wp.uint64,
                  rays: wp.array2d(dtype=wp.float32),     # [N, 6]
                  verts: wp.array(dtype=wp.vec3),
                  tets: wp.array2d(dtype=wp.int32),       # [T, 4]
                  cell: wp.array2d(dtype=wp.float32),     # [T, 7]
                  side_a: wp.array(dtype=wp.int32),       # [F]
                  side_b: wp.array(dtype=wp.int32),       # [F]
                  min_t: wp.float32,
                  t_far_arr: wp.array(dtype=wp.float32),  # [N]
                  max_iters: wp.int32,
                  step_eps: wp.float32,
                  saturate_log_t: wp.float32,
                  out: wp.array2d(dtype=wp.float32),      # [N, 4] rgba
                  out_m: wp.array2d(dtype=wp.float32),    # [N, 2] m1, m2
                  out_tsat: wp.array(dtype=wp.float32),   # [N]
                  out_steps: wp.array(dtype=wp.int32)):
    i = wp.tid()
    o = wp.vec3(rays[i, 0], rays[i, 1], rays[i, 2])
    d = wp.vec3(rays[i, 3], rays[i, 4], rays[i, 5])
    t_far = t_far_arr[i]

    t_cur = min_t
    log_t = float(0.0)
    rgb = wp.vec3(0.0, 0.0, 0.0)
    m1 = float(0.0)
    m2 = float(0.0)
    t_sat = float(-1.0)
    prev_face = int(-1)
    it = int(0)

    while it < max_iters and log_t > LOG_T_THRESHOLD:
        hit = bool(False)
        seg_end = t_far
        cls_face = prev_face
        t_launch = t_cur + step_eps
        if t_launch < t_far:
            q = wp.mesh_query_ray(mesh_id, o + d * t_launch, d,
                                  t_far - t_launch)
            if q.result:
                hit = True
                seg_end = t_launch + q.t
                cls_face = q.face
        # On a hit, (t_cur, seg_end) is bounded by the face we hit; on a
        # miss the remainder up to t_far still lies on the far side of the
        # face we last crossed -- classify against that instead.

        if cls_face >= 0 and seg_end > t_cur:
            mid = o + d * (0.5 * (t_cur + seg_end))
            tet = int(-1)
            ta = side_a[cls_face]
            tb = side_b[cls_face]
            if ta >= 0 and _point_in_tet(mid, ta, verts, tets):
                tet = ta
            elif tb >= 0 and _point_in_tet(mid, tb, verts, tets):
                tet = tb

            if tet >= 0:
                dens = cell[tet, 0]
                c0 = wp.vec3(cell[tet, 1], cell[tet, 2], cell[tet, 3])
                g = wp.vec3(cell[tet, 4], cell[tet, 5], cell[tet, 6])
                v0 = verts[tets[tet, 0]]
                # Affine color along the ray, identical to eval_tet_rt.
                base_offset = wp.dot(g, o - v0)
                base = c0 + wp.vec3(base_offset, base_offset, base_offset)
                dc_dt = wp.dot(g, d)
                c_start = _relu3(base + wp.vec3(dc_dt * t_cur, dc_dt * t_cur,
                                                dc_dt * t_cur))
                c_end = _relu3(base + wp.vec3(dc_dt * seg_end, dc_dt * seg_end,
                                              dc_dt * seg_end))
                od = wp.clamp(dens * (seg_end - t_cur), MIN_OD, MAX_OD)
                alpha_t = wp.exp(-od)
                phi = (1.0 - alpha_t) / od
                w0 = phi - alpha_t
                w1 = 1.0 - phi
                c_premul = c_end * w0 + c_start * w1
                c_premul = wp.vec3(wp.clamp(c_premul[0], 0.0, MAX_VAL),
                                   wp.clamp(c_premul[1], 0.0, MAX_VAL),
                                   wp.clamp(c_premul[2], 0.0, MAX_VAL))
                T_j = wp.exp(log_t)
                rgb += c_premul * T_j

                # Volumetric VSM moments -- same formulas + thin-segment
                # filter and small-od Taylor correction as the walker.
                alpha = 1.0 - alpha_t
                if alpha > 0.002:
                    m1 += T_j * (w0 * seg_end + w1 * t_cur)
                    correction = float(0.0)
                    if wp.abs(od) < 0.02:
                        correction = od * ((1.0 / 6.0)
                                           - od * ((1.0 / 12.0)
                                                   - od * (1.0 / 40.0)))
                    else:
                        correction = w0 + alpha_t - 2.0 * w0 / od
                    L = seg_end - t_cur
                    m2 += T_j * (w0 * seg_end * seg_end
                                 + w1 * t_cur * t_cur
                                 - L * L * correction)

                # First crossing of the saturation threshold, interpolated
                # within the segment exactly like the walker.
                if t_sat < 0.0 and (log_t - od) < saturate_log_t:
                    frac = (log_t - saturate_log_t) / wp.max(od, 1.0e-12)
                    t_sat = t_cur + frac * (seg_end - t_cur)

                log_t -= od

        if not hit:
            break
        prev_face = cls_face
        t_cur = seg_end
        it += 1

    out[i, 0] = rgb[0]
    out[i, 1] = rgb[1]
    out[i, 2] = rgb[2]
    out[i, 3] = 1.0 - wp.exp(log_t)
    out_m[i, 0] = m1
    out_m[i, 1] = m2
    out_tsat[i] = t_sat
    out_steps[i] = it


@wp.func
def _exit_of(tet: wp.int32,
             o: wp.vec3,
             d: wp.vec3,
             t_cur: wp.float32,
             verts: wp.array(dtype=wp.vec3),
             tets: wp.array2d(dtype=wp.int32)):
    """Exit distance + local face index of the ray leaving ``tet`` after
    ``t_cur``, in build_adj's face order -- (0,2,1), (1,2,3), (0,3,2),
    (3,0,1), i.e. face j is opposite vertex [3,0,1,2][j] -- so the result
    indexes ``tet_adj`` directly. Returns (t_exit, face) with face = -1
    when no well-conditioned exit exists (degenerate tet / parallel ray);
    caller falls back to BVH."""
    v0 = verts[tets[tet, 0]]
    v1 = verts[tets[tet, 1]]
    v2 = verts[tets[tet, 2]]
    v3 = verts[tets[tet, 3]]
    best_t = float(1.0e30)
    best_f = int(-1)
    for j in range(4):
        if j == 0:
            a = v0
            b = v2
            c = v1
            opp = v3
        elif j == 1:
            a = v1
            b = v2
            c = v3
            opp = v0
        elif j == 2:
            a = v0
            b = v3
            c = v2
            opp = v1
        else:
            a = v3
            b = v0
            c = v1
            opp = v2
        n = wp.cross(b - a, c - a)
        if wp.dot(n, opp - a) > 0.0:
            n = -n  # outward
        den = wp.dot(n, d)
        if den > 0.0:  # heading out through this face
            t0 = wp.dot(n, a - o) / den
            if t0 >= t_cur and t0 < best_t:
                best_t = t0
                best_f = j
    return best_t, best_f


@wp.func
def _integrate_seg(tet: wp.int32,
                   t0: wp.float32,
                   t1: wp.float32,
                   o: wp.vec3,
                   d: wp.vec3,
                   rgb: wp.vec3,
                   log_t: wp.float32,
                   m1: wp.float32,
                   m2: wp.float32,
                   t_sat: wp.float32,
                   saturate_log_t: wp.float32,
                   verts: wp.array(dtype=wp.vec3),
                   tets: wp.array2d(dtype=wp.int32),
                   cell: wp.array2d(dtype=wp.float32)):
    """One tet segment of the walker's integral; returns the updated
    (rgb, log_t, m1, m2, t_sat) accumulators."""
    dens = cell[tet, 0]
    c0 = wp.vec3(cell[tet, 1], cell[tet, 2], cell[tet, 3])
    g = wp.vec3(cell[tet, 4], cell[tet, 5], cell[tet, 6])
    v0 = verts[tets[tet, 0]]
    base_offset = wp.dot(g, o - v0)
    base = c0 + wp.vec3(base_offset, base_offset, base_offset)
    dc_dt = wp.dot(g, d)
    c_start = _relu3(base + wp.vec3(dc_dt * t0, dc_dt * t0, dc_dt * t0))
    c_end = _relu3(base + wp.vec3(dc_dt * t1, dc_dt * t1, dc_dt * t1))
    od = wp.clamp(dens * (t1 - t0), MIN_OD, MAX_OD)
    alpha_t = wp.exp(-od)
    phi = (1.0 - alpha_t) / od
    w0 = phi - alpha_t
    w1 = 1.0 - phi
    c_premul = c_end * w0 + c_start * w1
    c_premul = wp.vec3(wp.clamp(c_premul[0], 0.0, MAX_VAL),
                       wp.clamp(c_premul[1], 0.0, MAX_VAL),
                       wp.clamp(c_premul[2], 0.0, MAX_VAL))
    T_j = wp.exp(log_t)
    rgb += c_premul * T_j
    alpha = 1.0 - alpha_t
    if alpha > 0.002:
        m1 += T_j * (w0 * t1 + w1 * t0)
        correction = float(0.0)
        if wp.abs(od) < 0.02:
            correction = od * ((1.0 / 6.0)
                               - od * ((1.0 / 12.0) - od * (1.0 / 40.0)))
        else:
            correction = w0 + alpha_t - 2.0 * w0 / od
        L = t1 - t0
        m2 += T_j * (w0 * t1 * t1 + w1 * t0 * t0 - L * L * correction)
    if t_sat < 0.0 and (log_t - od) < saturate_log_t:
        frac = (log_t - saturate_log_t) / wp.max(od, 1.0e-12)
        t_sat = t0 + frac * (t1 - t0)
    log_t -= od
    return rgb, log_t, m1, m2, t_sat


@wp.func
def _grad_seg(tet: wp.int32,
              t0: wp.float32,
              t1: wp.float32,
              log_t: wp.float32,
              gscale: wp.float32,
              cell: wp.array2d(dtype=wp.float32),
              grad_dens: wp.array(dtype=wp.float32)) -> wp.float32:
    """Gradient counterpart of :func:`_integrate_seg` for exact
    transmittance: ``vis = exp(-sum od)`` gives ``dvis/ddens_i = -L_i *
    vis`` per crossed segment, zero where the ``clamp(dens*L, MIN_OD,
    MAX_OD)`` pins od. ``gscale = grad_vis * vis`` is precomputed per ray;
    returns the updated log_t so termination replays the forward walk."""
    dens = cell[tet, 0]
    L = t1 - t0
    raw = dens * L
    if raw > MIN_OD and raw < MAX_OD:
        wp.atomic_add(grad_dens, tet, -L * gscale)
    log_t -= wp.clamp(raw, MIN_OD, MAX_OD)
    return log_t


@wp.kernel
def _trace_walk_grad_kernel(mesh_id: wp.uint64,
                            rays: wp.array2d(dtype=wp.float32),     # [N, 6]
                            verts: wp.array(dtype=wp.vec3),
                            tets: wp.array2d(dtype=wp.int32),       # [T, 4]
                            tet_adj: wp.array2d(dtype=wp.int32),    # [T, 4]
                            cell: wp.array2d(dtype=wp.float32),     # [T, 7]
                            side_a: wp.array(dtype=wp.int32),       # [F]
                            side_b: wp.array(dtype=wp.int32),       # [F]
                            min_t: wp.float32,
                            t_far_arr: wp.array(dtype=wp.float32),  # [N]
                            max_iters: wp.int32,
                            step_eps: wp.float32,
                            gscale_arr: wp.array(dtype=wp.float32),  # [N]
                            grad_dens: wp.array(dtype=wp.float32)):  # [T]
    """Replay of :func:`_trace_walk_kernel`'s control flow that scatters
    per-segment density gradients instead of accumulating color/moments.
    Must mirror the forward walk exactly (same recovery, same termination)
    so gradient mass lands only on segments the forward pass integrated."""
    i = wp.tid()
    gscale = gscale_arr[i]
    if gscale == 0.0:
        return
    o = wp.vec3(rays[i, 0], rays[i, 1], rays[i, 2])
    d = wp.vec3(rays[i, 3], rays[i, 4], rays[i, 5])
    t_far = t_far_arr[i]

    t_cur = min_t
    log_t = float(0.0)
    cur_tet = int(-1)
    it = int(0)

    while it < max_iters and log_t > LOG_T_THRESHOLD and t_cur < t_far:
        it += 1
        if cur_tet < 0:
            t_launch = t_cur + step_eps
            if t_launch >= t_far:
                break
            q = wp.mesh_query_ray(mesh_id, o + d * t_launch, d,
                                  t_far - t_launch)
            if not q.result:
                break
            t_hit = t_launch + q.t
            ta = side_a[q.face]
            tb = side_b[q.face]
            mid = o + d * (0.5 * (t_cur + t_hit))
            seg_tet = int(-1)
            if ta >= 0 and _point_in_tet(mid, ta, verts, tets):
                seg_tet = ta
            elif tb >= 0 and _point_in_tet(mid, tb, verts, tets):
                seg_tet = tb
            if seg_tet >= 0 and t_hit > t_cur:
                log_t = _grad_seg(seg_tet, t_cur, t_hit, log_t, gscale,
                                  cell, grad_dens)
            p = o + d * (t_hit + step_eps)
            cur_tet = int(-1)
            if ta >= 0 and ta != seg_tet and _point_in_tet(p, ta, verts, tets):
                cur_tet = ta
            elif tb >= 0 and tb != seg_tet and _point_in_tet(p, tb, verts,
                                                             tets):
                cur_tet = tb
            t_cur = t_hit
            continue

        t_exit, f_exit = _exit_of(cur_tet, o, d, t_cur, verts, tets)
        if f_exit < 0:
            cur_tet = int(-1)
            continue
        seg_end = wp.min(t_exit, t_far)
        if seg_end > t_cur:
            log_t = _grad_seg(cur_tet, t_cur, seg_end, log_t, gscale,
                              cell, grad_dens)
        if seg_end >= t_far:
            break

        nxt = int(-1)
        cand = tet_adj[cur_tet, f_exit]
        if cand >= 0:
            p = o + d * (seg_end + step_eps)
            if _point_in_tet(p, cand, verts, tets):
                nxt = cand
        cur_tet = nxt
        t_cur = seg_end


@wp.kernel
def _trace_walk_kernel(mesh_id: wp.uint64,
                       rays: wp.array2d(dtype=wp.float32),     # [N, 6]
                       verts: wp.array(dtype=wp.vec3),
                       tets: wp.array2d(dtype=wp.int32),       # [T, 4]
                       tet_adj: wp.array2d(dtype=wp.int32),    # [T, 4]
                       cell: wp.array2d(dtype=wp.float32),     # [T, 7]
                       side_a: wp.array(dtype=wp.int32),       # [F]
                       side_b: wp.array(dtype=wp.int32),       # [F]
                       min_t: wp.float32,
                       t_far_arr: wp.array(dtype=wp.float32),  # [N]
                       max_iters: wp.int32,
                       step_eps: wp.float32,
                       saturate_log_t: wp.float32,
                       out: wp.array2d(dtype=wp.float32),      # [N, 4]
                       out_m: wp.array2d(dtype=wp.float32),    # [N, 2]
                       out_tsat: wp.array(dtype=wp.float32),   # [N]
                       out_recover: wp.array(dtype=wp.int32)):
    """Hybrid marcher: O(1) adjacency hops verified per step, exact BVH
    recovery on any mismatch. Walker speed, face-marcher robustness."""
    i = wp.tid()
    o = wp.vec3(rays[i, 0], rays[i, 1], rays[i, 2])
    d = wp.vec3(rays[i, 3], rays[i, 4], rays[i, 5])
    t_far = t_far_arr[i]

    t_cur = min_t
    log_t = float(0.0)
    rgb = wp.vec3(0.0, 0.0, 0.0)
    m1 = float(0.0)
    m2 = float(0.0)
    t_sat = float(-1.0)
    n_recover = int(0)
    cur_tet = int(-1)
    it = int(0)

    while it < max_iters and log_t > LOG_T_THRESHOLD and t_cur < t_far:
        it += 1
        if cur_tet < 0:
            # Unknown tet: one BVH query to the next face crossing, then
            # handle the segment exactly like the pure face-marcher --
            # classify it by the midpoint against the hit face's two side
            # tets and integrate. This also covers an origin inside a tet
            # (the segment before the first crossing).
            n_recover += 1
            t_launch = t_cur + step_eps
            if t_launch >= t_far:
                break
            q = wp.mesh_query_ray(mesh_id, o + d * t_launch, d,
                                  t_far - t_launch)
            if not q.result:
                break
            t_hit = t_launch + q.t
            ta = side_a[q.face]
            tb = side_b[q.face]
            mid = o + d * (0.5 * (t_cur + t_hit))
            seg_tet = int(-1)
            if ta >= 0 and _point_in_tet(mid, ta, verts, tets):
                seg_tet = ta
            elif tb >= 0 and _point_in_tet(mid, tb, verts, tets):
                seg_tet = tb
            if seg_tet >= 0 and t_hit > t_cur:
                rgb, log_t, m1, m2, t_sat = _integrate_seg(
                    seg_tet, t_cur, t_hit, o, d, rgb, log_t, m1, m2,
                    t_sat, saturate_log_t, verts, tets, cell)
            # The next segment starts on the far side of the crossed face.
            p = o + d * (t_hit + step_eps)
            cur_tet = int(-1)
            if ta >= 0 and ta != seg_tet and _point_in_tet(p, ta, verts, tets):
                cur_tet = ta
            elif tb >= 0 and tb != seg_tet and _point_in_tet(p, tb, verts,
                                                             tets):
                cur_tet = tb
            t_cur = t_hit
            continue

        # Fast path: integrate the current tet up to its exit (or t_far).
        t_exit, f_exit = _exit_of(cur_tet, o, d, t_cur, verts, tets)
        if f_exit < 0:
            # Degenerate exit -- re-anchor via BVH without integrating.
            cur_tet = int(-1)
            continue
        seg_end = wp.min(t_exit, t_far)
        if seg_end > t_cur:
            rgb, log_t, m1, m2, t_sat = _integrate_seg(
                cur_tet, t_cur, seg_end, o, d, rgb, log_t, m1, m2,
                t_sat, saturate_log_t, verts, tets, cell)
        if seg_end >= t_far:
            break

        # Adjacency hop, verified by a probe just past the crossing.
        # Any mismatch (sliver, grazed edge, boundary re-entry) falls back
        # to the exact BVH branch above on the next iteration.
        nxt = int(-1)
        cand = tet_adj[cur_tet, f_exit]
        if cand >= 0:
            p = o + d * (seg_end + step_eps)
            if _point_in_tet(p, cand, verts, tets):
                nxt = cand
        cur_tet = nxt
        t_cur = seg_end

    out[i, 0] = rgb[0]
    out[i, 1] = rgb[1]
    out[i, 2] = rgb[2]
    out[i, 3] = 1.0 - wp.exp(log_t)
    out_m[i, 0] = m1
    out_m[i, 1] = m2
    out_tsat[i] = t_sat
    out_recover[i] = n_recover


class WarpFaceTracer:
    """BVH face-marching tet tracer (forward only).

    Args:
        vertices: [V, 3] float tensor on cuda.
        indices: [T, 4] int tensor on cuda.
    """

    def __init__(self, vertices: torch.Tensor, indices: torch.Tensor,
                 tet_mask: "torch.Tensor | None" = None):
        wp.init()
        device = vertices.device
        # Keep torch tensors alive: warp views share their memory.
        self._verts = vertices.detach().to(torch.float32).contiguous()
        self._tets = indices.to(torch.int32).contiguous()
        self._wp_device = wp.device_from_torch(device)
        self.mesh = None
        self.rebuild(tet_mask)

    def rebuild(self, tet_mask: "torch.Tensor | None" = None):
        """(Re)build the face BVH, optionally over a subset of tets.

        ``tet_mask`` ([T] bool) restricts the BVH to faces of the selected
        tets. Rays then cross only those faces; segments through unselected
        space classify into no tet and contribute nothing. For occlusion
        queries, masking out near-zero-density tets turns millions of
        empty-space crossings (each a full BVH traversal) into a single
        skip -- the integral changes by at most MIN_OD per skipped tet.
        ``side_index`` is remapped back to GLOBAL tet ids, so cell_values
        stays the full per-tet table.
        """
        idx = self._tets.long()
        device = idx.device
        if tet_mask is None:
            faces, side = get_tet_adjacency(idx)
        else:
            assert tet_mask.shape == (idx.shape[0],)
            ids = tet_mask.nonzero(as_tuple=True)[0]
            if ids.numel() == 0:
                # Empty selection: a single degenerate (zero-area, hence
                # unhittable) triangle keeps wp.Mesh happy.
                faces = torch.zeros((1, 3), dtype=torch.long, device=device)
                side = torch.full((1, 2), -1, dtype=torch.long, device=device)
            else:
                faces, side = get_tet_adjacency(idx[ids])
                side = torch.where(side >= 0, ids[side.clamp(min=0)],
                                   torch.full_like(side, -1))
        self._faces = faces.reshape(-1).to(torch.int32).contiguous()
        self._side_a = side[:, 0].to(torch.int32).contiguous()
        self._side_b = side[:, 1].to(torch.int32).contiguous()
        self.mesh = wp.Mesh(
            points=wp.from_torch(self._verts, dtype=wp.vec3),
            indices=wp.from_torch(self._faces),
        )

    def refit(self, vertices: "torch.Tensor | None" = None):
        """Update the BVH after vertex motion (same topology)."""
        if vertices is not None:
            self._verts.copy_(vertices.detach().to(torch.float32))
        self.mesh.refit()

    def trace(self,
              rays: torch.Tensor,          # [N, 6]
              cell_values: torch.Tensor,   # [T, >=7]; only [:7] used
              min_t: float,
              t_far=1e30,                  # float OR [N] tensor
              max_iters: int = 512,
              step_eps: float = 1e-5,
              saturate_log_t: float = -3.0,
              return_n_steps: bool = False,
              return_extras: bool = False):
        """Trace rays; returns output_img [N, 4] (r, g, b, alpha).

        With ``return_extras=True`` returns ``(output_img, extras)`` where
        extras has ``m1`` / ``m2`` (volumetric VSM moments), ``t_saturate``
        (distance at which alpha first crossed ``1 - exp(saturate_log_t)``,
        -1 if never) and ``n_steps``.
        """
        assert rays.dim() == 2 and rays.shape[1] == 6
        assert cell_values.shape[0] == self._tets.shape[0]
        N = rays.shape[0]
        device = rays.device
        rays_c = rays.detach().to(torch.float32).contiguous()
        cell_c = cell_values.detach()[:, :7].to(torch.float32).contiguous()
        if torch.is_tensor(t_far):
            assert t_far.shape == (N,)
            t_far_t = t_far.detach().to(device=device,
                                        dtype=torch.float32).contiguous()
        else:
            t_far_t = torch.full((N,), float(t_far), dtype=torch.float32,
                                 device=device)
        out = torch.zeros((N, 4), dtype=torch.float32, device=device)
        out_m = torch.zeros((N, 2), dtype=torch.float32, device=device)
        out_tsat = torch.full((N,), -1.0, dtype=torch.float32, device=device)
        n_steps = torch.zeros((N,), dtype=torch.int32, device=device)
        wp.launch(
            _trace_kernel, dim=N,
            inputs=[self.mesh.id, wp.from_torch(rays_c),
                    wp.from_torch(self._verts, dtype=wp.vec3),
                    wp.from_torch(self._tets),
                    wp.from_torch(cell_c),
                    wp.from_torch(self._side_a), wp.from_torch(self._side_b),
                    float(min_t), wp.from_torch(t_far_t), int(max_iters),
                    float(step_eps), float(saturate_log_t)],
            outputs=[wp.from_torch(out), wp.from_torch(out_m),
                     wp.from_torch(out_tsat), wp.from_torch(n_steps)],
            device=self._wp_device,
        )
        if return_extras:
            return out, dict(m1=out_m[:, 0], m2=out_m[:, 1],
                             t_saturate=out_tsat, n_steps=n_steps)
        if return_n_steps:
            return out, n_steps
        return out

    def trace_walk(self,
                   rays: torch.Tensor,          # [N, 6]
                   cell_values: torch.Tensor,   # [T, >=7]; only [:7] used
                   tet_adj: torch.Tensor,       # [T, 4] from build_adj
                   min_t: float,
                   t_far=1e30,                  # float OR [N] tensor
                   max_iters: int = 1024,
                   step_eps: float = 1e-5,
                   saturate_log_t: float = -3.0,
                   return_extras: bool = False):
        """Hybrid trace: adjacency hops (O(1) per face crossing, like the
        slang walker) with every hop verified by a point-in-tet probe and
        exact BVH recovery on mismatch. ~Walker speed, face-marcher
        robustness. Same outputs as :meth:`trace`; extras additionally
        report ``n_recover`` -- BVH queries per ray, which includes the
        initial entry query, so healthy values are ~1-2 (mid-walk
        recoveries are the amount above 1)."""
        assert rays.dim() == 2 and rays.shape[1] == 6
        assert cell_values.shape[0] == self._tets.shape[0]
        assert tet_adj.shape == self._tets.shape
        N = rays.shape[0]
        device = rays.device
        rays_c = rays.detach().to(torch.float32).contiguous()
        cell_c = cell_values.detach()[:, :7].to(torch.float32).contiguous()
        adj_c = tet_adj.to(torch.int32).contiguous()
        if torch.is_tensor(t_far):
            assert t_far.shape == (N,)
            t_far_t = t_far.detach().to(device=device,
                                        dtype=torch.float32).contiguous()
        else:
            t_far_t = torch.full((N,), float(t_far), dtype=torch.float32,
                                 device=device)
        out = torch.zeros((N, 4), dtype=torch.float32, device=device)
        out_m = torch.zeros((N, 2), dtype=torch.float32, device=device)
        out_tsat = torch.full((N,), -1.0, dtype=torch.float32, device=device)
        n_recover = torch.zeros((N,), dtype=torch.int32, device=device)
        wp.launch(
            _trace_walk_kernel, dim=N,
            inputs=[self.mesh.id, wp.from_torch(rays_c),
                    wp.from_torch(self._verts, dtype=wp.vec3),
                    wp.from_torch(self._tets),
                    wp.from_torch(adj_c),
                    wp.from_torch(cell_c),
                    wp.from_torch(self._side_a), wp.from_torch(self._side_b),
                    float(min_t), wp.from_torch(t_far_t), int(max_iters),
                    float(step_eps), float(saturate_log_t)],
            outputs=[wp.from_torch(out), wp.from_torch(out_m),
                     wp.from_torch(out_tsat), wp.from_torch(n_recover)],
            device=self._wp_device,
        )
        if return_extras:
            return out, dict(m1=out_m[:, 0], m2=out_m[:, 1],
                             t_saturate=out_tsat, n_recover=n_recover)
        return out

    def trace_walk_grad(self,
                        rays: torch.Tensor,          # [N, 6]
                        cell_values: torch.Tensor,   # [T, >=7]
                        tet_adj: torch.Tensor,       # [T, 4]
                        grad_vis: torch.Tensor,      # [N] upstream dL/dvis
                        vis: torch.Tensor,           # [N] forward visibility
                        min_t: float,
                        t_far=1e30,
                        max_iters: int = 1024,
                        step_eps: float = 1e-5) -> torch.Tensor:
        """Density gradient of exact transmittance ``vis = exp(-sum od)``.

        Replays the :meth:`trace_walk` walk and scatters ``-L_seg * vis *
        grad_vis`` into each crossed tet (zero where the od clamp pins the
        segment). Returns ``grad_density`` [T]. Only the density column
        participates -- color/moment outputs have no gradient here.
        """
        assert rays.dim() == 2 and rays.shape[1] == 6
        assert cell_values.shape[0] == self._tets.shape[0]
        N = rays.shape[0]
        device = rays.device
        rays_c = rays.detach().to(torch.float32).contiguous()
        cell_c = cell_values.detach()[:, :7].to(torch.float32).contiguous()
        adj_c = tet_adj.to(torch.int32).contiguous()
        if torch.is_tensor(t_far):
            assert t_far.shape == (N,)
            t_far_t = t_far.detach().to(device=device,
                                        dtype=torch.float32).contiguous()
        else:
            t_far_t = torch.full((N,), float(t_far), dtype=torch.float32,
                                 device=device)
        gscale = (grad_vis.detach() * vis.detach()).to(
            torch.float32).contiguous()
        grad_dens = torch.zeros((self._tets.shape[0],), dtype=torch.float32,
                                device=device)
        wp.launch(
            _trace_walk_grad_kernel, dim=N,
            inputs=[self.mesh.id, wp.from_torch(rays_c),
                    wp.from_torch(self._verts, dtype=wp.vec3),
                    wp.from_torch(self._tets),
                    wp.from_torch(adj_c),
                    wp.from_torch(cell_c),
                    wp.from_torch(self._side_a), wp.from_torch(self._side_b),
                    float(min_t), wp.from_torch(t_far_t), int(max_iters),
                    float(step_eps), wp.from_torch(gscale)],
            outputs=[wp.from_torch(grad_dens)],
            device=self._wp_device,
        )
        return grad_dens
