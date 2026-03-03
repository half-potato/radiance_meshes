# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "numpy",
#   "scipy",
#   "pillow",
# ]
# ///

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    from scipy.spatial import Delaunay
    from PIL import Image, ImageDraw, ImageFont
    import io, base64
    return Delaunay, Image, ImageDraw, ImageFont, base64, io, mo, np


@app.cell
def __(mo):
    mo.md(r"""
    # Quadratic Spline Smoothness Over Delaunay Flips

    Each triangle carries a degree-2 Lagrange polynomial determined by **6 control values**:
    3 vertices + 3 edge midpoints. Edge midpoint values are **properties of edges**, shared
    between adjacent triangles.

    When a Delaunay flip replaces diagonal $AC$ with $BD$:
    - Vertices $A,B,C,D$ and boundary edge midpoints survive unchanged
    - The new diagonal $BD$ intersects the old diagonal $AC$ at a point $P$
    - $P$ lies **on** old edge $AC$, so both old triangles agree on its value
    - We fit a 1D quadratic through $(v_B,\, v_P,\, v_D)$ to recover the midpoint of $BD$

    This means no information is invented or discarded — the flip is **lossless**.

    > **Left**: Naive flat shading (average vertex colours). Pops at every flip.
    >
    > **Right**: Quadratic spline with intersection-based flip propagation. Smooth everywhere.
    """)
    return


@app.cell
def __(mo):
    n_pts_sl  = mo.ui.slider(5, 35, value=20,  label="Interior points")
    rad_sl    = mo.ui.slider(10, 70, value=38, label="Orbit radius")
    wire_chk  = mo.ui.checkbox(value=True,     label="Show wireframe + midpoints")
    run_btn   = mo.ui.run_button(label="▶ Compute frames")
    mo.hstack([n_pts_sl, rad_sl, wire_chk, run_btn], gap=3)
    return n_pts_sl, rad_sl, wire_chk, run_btn


@app.cell
def __(mo):
    frame_sl = mo.ui.slider(0, 179, value=0, label="Frame  (scrub after computing)")
    frame_sl
    return (frame_sl,)


# ── Core math functions (pure, no UI deps) ────────────────────────────────────
@app.cell
def __(np):

    def field(x, y):
        """Ground-truth colour field — used only to initialise vertex values."""
        s = 5
        x, y = np.asarray(x, float), np.asarray(y, float)
        r = np.sin(x / (10*s)) * 0.25 + 0.5
        g = np.cos(y / (20*s)) * 0.25 + 0.5
        b = np.sin((x+y) / (30*s)) * 0.25 + 0.5
        return np.stack([r, g, b], axis=-1) * 255   # (..., 3)

    def barycentric(pts, a, b, c):
        """Barycentric coords of pts (M,2) w.r.t. triangle (a,b,c)."""
        v0, v1, v2 = b-a, c-a, pts-a
        d00, d01, d11 = v0@v0, v0@v1, v1@v1
        d20, d21 = v2@v0, v2@v1
        denom = d00*d11 - d01*d01
        if abs(denom) < 1e-14:
            return np.full((len(pts), 3), 1/3)
        vv = (d11*d20 - d01*d21) / denom
        ww = (d00*d21 - d01*d20) / denom
        return np.stack([1-vv-ww, vv, ww], axis=-1)

    def quad_basis(lam):
        """Degree-2 Lagrange basis: [v0,v1,v2, m01,m12,m02]."""
        l0, l1, l2 = lam[:,0], lam[:,1], lam[:,2]
        return np.stack([
            l0*(2*l0-1), l1*(2*l1-1), l2*(2*l2-1),
            4*l0*l1, 4*l1*l2, 4*l0*l2
        ], axis=-1)

    def eval_spline(pt, tri, pts, vert_vals, edge_vals):
        """Evaluate the quadratic spline at a single point pt (2,)."""
        idx = tri.find_simplex(pt.reshape(1, 2))[0]
        if idx < 0:
            # Outside convex hull — fall back to nearest vertex value
            dists = np.linalg.norm(pts - pt, axis=1)
            return vert_vals[np.argmin(dists)]
        i0, i1, i2 = tri.simplices[idx]
        lam   = barycentric(pt.reshape(1, 2), pts[i0], pts[i1], pts[i2])
        basis = quad_basis(lam)   # (1, 6)
        ctrl  = np.stack([
            vert_vals[i0], vert_vals[i1], vert_vals[i2],
            edge_vals[(min(i0,i1), max(i0,i1))],
            edge_vals[(min(i1,i2), max(i1,i2))],
            edge_vals[(min(i0,i2), max(i0,i2))],
        ])                        # (6, 3)
        return (basis @ ctrl)[0]  # (3,)

    def seg_intersect(p1, p2, p3, p4):
        """
        Intersection of open segment p1→p2 with segment p3→p4.
        Returns (t along p1-p2, intersection point) or None.
        t is strictly interior: (eps, 1-eps).
        """
        d1, d2 = p2-p1, p4-p3
        cross = d1[0]*d2[1] - d1[1]*d2[0]
        if abs(cross) < 1e-10:
            return None
        diff = p3-p1
        t = (diff[0]*d2[1] - diff[1]*d2[0]) / cross
        u = (diff[0]*d1[1] - diff[1]*d1[0]) / cross
        eps = 1e-6
        if eps < t < 1-eps and -eps <= u <= 1+eps:
            return t, p1 + t*d1
        return None

    def midpoint_via_intersection(pa, pb, va, vb, old_tri, old_pts, old_vert_vals, old_edge_vals):
        """
        Find the midpoint control value for new edge pa→pb.

        Strategy:
          1. Walk all old edges to find which one the new edge crosses.
          2. P = intersection point — it lies ON that old edge, so the
             old spline is unambiguous there.
          3. Fit a 1-D quadratic through (t=0, va), (t=t_P, v_P), (t=1, vb)
             and evaluate at t=0.5 to get the new midpoint control value.
        """
        seen = set()
        for simplex in old_tri.simplices:
            for i in range(3):
                ei, ej = simplex[i], simplex[(i+1)%3]
                key = (min(ei,ej), max(ei,ej))
                if key in seen:
                    continue
                seen.add(key)
                result = seg_intersect(pa, pb, old_pts[ei], old_pts[ej])
                if result is None:
                    continue
                t_P, P = result
                v_P = eval_spline(P, old_tri, old_pts, old_vert_vals, old_edge_vals)
                # Fit quadratic q(t) = at²+bt+c through:
                #   q(0)   = va   →  c = va
                #   q(1)   = vb   →  a+b = vb-va
                #   q(t_P) = v_P  →  a*t_P²+b*t_P = v_P-va
                # Solve for a, then evaluate at t=0.5:
                #   q(0.5) = 0.25a + 0.5(vb-va-a) + va = -0.25a + 0.5(va+vb)
                denom = t_P*t_P - t_P   # t_P*(t_P - 1), always ≠ 0 for t_P ∈ (0,1)
                a_coef = (v_P - va - t_P*(vb - va)) / denom
                return -0.25*a_coef + 0.5*(va + vb)

        # Fallback: no crossing found (shouldn't happen for a genuine new edge)
        mid = (pa + pb) * 0.5
        return eval_spline(mid, old_tri, old_pts, old_vert_vals, old_edge_vals)

    return (barycentric, eval_spline, field, midpoint_via_intersection,
            quad_basis, seg_intersect)


# ── Point trajectory setup ────────────────────────────────────────────────────
@app.cell
def __(Delaunay, n_pts_sl, np, rad_sl):
    W, H      = 560, 380
    N_FRAMES  = 180
    SEED      = 42
    n_pts     = n_pts_sl.value
    r_val     = rad_sl.value

    rng    = np.random.default_rng(SEED)
    p_in   = (rng.random((n_pts, 2)) * 1.4 - 0.2) * [W, H]
    t_bdr  = np.linspace(0, 2*np.pi, 14, endpoint=False)
    p_bdr  = np.stack([0.54*W*np.cos(t_bdr)+0.5*W,
                       0.54*H*np.sin(t_bdr)+0.5*H], axis=1)
    p0     = np.concatenate([p_in, p_bdr])
    N_mv   = n_pts   # only interior points orbit

    a_r   = rng.uniform(r_val*0.4, r_val, (N_mv, 1))
    b_r   = rng.uniform(r_val*0.4, r_val, (N_mv, 1))
    alpha = rng.uniform(0, 2*np.pi, (N_mv, 1))
    phi   = rng.uniform(0, 2*np.pi, (N_mv, 1))
    sign  = rng.choice([-1., 1.], (N_mv, 1))
    ca, sa = np.cos(alpha), np.sin(alpha)
    R_mat  = np.stack([np.concatenate([ca,-sa],1),
                       np.concatenate([sa, ca],1)], axis=1)   # (N_mv,2,2)
    omega  = 2*np.pi / N_FRAMES

    # Pre-bake all point positions so both render paths use identical geometry
    all_pts = []
    for _f in range(N_FRAMES):
        _theta = phi + sign * omega * _f
        _offs  = np.hstack([a_r*np.cos(_theta), b_r*np.sin(_theta)])
        _pts   = p0.copy()
        _pts[:N_mv] = p0[:N_mv] + np.einsum('nij,nj->ni', R_mat, _offs)
        all_pts.append(_pts)

    # First-frame triangulation (used to initialise edge_vals on frame 0)
    tri0 = Delaunay(all_pts[0])

    return (H, N_FRAMES, N_mv, R_mat, W, tri0, a_r, all_pts, alpha, b_r,
            n_pts, omega, p0, phi, r_val, sign)


# ── Sequential frame precomputation ──────────────────────────────────────────
@app.cell
def __(Delaunay, H, Image, ImageDraw, ImageFont, N_FRAMES, W, tri0,
       all_pts, barycentric, eval_spline, field, midpoint_via_intersection,
       mo, np, quad_basis, run_btn):

    run_btn  # re-run when button pressed

    def pixel_grid(W, H):
        ys, xs = np.mgrid[0:H, 0:W]
        return xs, ys, np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(float)

    def render_naive(W, H, pts, tri):
        xs, ys, pix = pixel_grid(W, H)
        tind = tri.find_simplex(pix)
        vv   = field(pts[:,0], pts[:,1])
        img  = np.zeros((H, W, 3), np.float32)
        for idx, simplex in enumerate(tri.simplices):
            mask = tind == idx
            if not mask.any(): continue
            img[ys.ravel()[mask], xs.ravel()[mask]] = vv[simplex].mean(0)
        return np.clip(img, 0, 255).astype(np.uint8)

    def render_quadratic(W, H, pts, tri, vert_vals, edge_vals):
        xs, ys, pix = pixel_grid(W, H)
        tind = tri.find_simplex(pix)
        img  = np.zeros((H, W, 3), np.float32)
        for idx, simplex in enumerate(tri.simplices):
            i0, i1, i2 = simplex
            mask = tind == idx
            if not mask.any(): continue
            lam   = barycentric(pix[mask], pts[i0], pts[i1], pts[i2])
            basis = quad_basis(lam)
            ctrl  = np.stack([
                vert_vals[i0], vert_vals[i1], vert_vals[i2],
                edge_vals[(min(i0,i1), max(i0,i1))],
                edge_vals[(min(i1,i2), max(i1,i2))],
                edge_vals[(min(i0,i2), max(i0,i2))],
            ])
            img[ys.ravel()[mask], xs.ravel()[mask]] = basis @ ctrl
        return np.clip(img, 0, 255).astype(np.uint8)

    def overlay_mesh(arr, pts, tri, edge_vals):
        base = Image.fromarray(arr).convert('RGBA')
        wire = Image.new('RGBA', base.size, (0,0,0,0))
        d    = ImageDraw.Draw(wire)
        for s in tri.simplices:
            poly = [(float(pts[i,0]), float(pts[i,1])) for i in s]
            poly.append(poly[0])
            d.line(poly, fill=(220,220,220,110), width=1)
        for (ei,ej) in edge_vals:
            mx, my = (pts[ei]+pts[ej])/2
            d.ellipse((mx-2.5, my-2.5, mx+2.5, my+2.5), fill=(255,230,50,220))
        return np.asarray(Image.alpha_composite(base, wire).convert('RGB'))

    def compose(naive_arr, quad_arr, W, H):
        label_h = 34
        out = Image.new('RGB', (W*2, H+label_h), (245,245,245))
        out.paste(Image.fromarray(naive_arr), (0, 0))
        out.paste(Image.fromarray(quad_arr),  (W, 0))
        d = ImageDraw.Draw(out)
        d.line([(W,0),(W,H)], fill=(60,60,60), width=2)
        try:
            fnt = ImageFont.truetype("DejaVuSans.ttf", 14)
        except:
            fnt = ImageFont.load_default()
        for txt, ox in [("Naive (flat per triangle)", 0),
                        ("Quadratic spline — intersection propagation", W)]:
            if hasattr(d,'textbbox'):
                l,t,r,b2 = d.textbbox((0,0), txt, font=fnt); tw,th = r-l,b2-t
            else:
                tw,th = d.textsize(txt, font=fnt)
            d.text((ox+(W-tw)//2, H+(label_h-th)//2), txt, fill=(30,30,30), font=fnt)
        return out

    # ── Sequential computation ────────────────────────────────────────────────
    # Initialise edge values for frame 0 from the ground-truth field
    def init_edge_vals(tri, pts):
        ev = {}
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i+1)%3]
                key = (min(a,b), max(a,b))
                if key not in ev:
                    mid = (pts[a]+pts[b])*0.5
                    ev[key] = field(mid[0], mid[1])
        return ev

    with mo.status.spinner(title="Computing frames…"):
        rendered_frames = []

        prev_tri      = tri0
        prev_pts      = all_pts[0]
        prev_vv       = field(prev_pts[:,0], prev_pts[:,1])
        prev_ev       = init_edge_vals(prev_tri, prev_pts)

        for f in range(N_FRAMES):
            pts = all_pts[f]
            tri = Delaunay(pts)

            # Vertex values: always sample from field at current position
            vv = field(pts[:,0], pts[:,1])

            # Edge midpoint values: propagate through flips
            ev = {}
            new_edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    a, b = simplex[i], simplex[(i+1)%3]
                    new_edges.add((min(a,b), max(a,b)))

            for key in new_edges:
                if key in prev_ev:
                    # Surviving edge — carry value forward unchanged
                    ev[key] = prev_ev[key]
                else:
                    # New edge (created by a flip) — intersection method
                    a, b = key
                    ev[key] = midpoint_via_intersection(
                        pts[a], pts[b], vv[a], vv[b],
                        prev_tri, prev_pts, prev_vv, prev_ev
                    )

            # Render both versions
            naive_arr = render_naive(W, H, pts, tri)
            quad_arr  = render_quadratic(W, H, pts, tri, vv, ev)
            naive_arr = overlay_mesh(naive_arr, pts, tri, ev)
            quad_arr  = overlay_mesh(quad_arr,  pts, tri, ev)

            rendered_frames.append(np.asarray(compose(naive_arr, quad_arr, W, H)))

            prev_tri  = tri
            prev_pts  = pts
            prev_vv   = vv
            prev_ev   = ev

    print(f"Done — {len(rendered_frames)} frames at {W}×{H}")
    return (compose, init_edge_vals, overlay_mesh, render_naive,
            render_quadratic, rendered_frames, pixel_grid)


# ── Display ───────────────────────────────────────────────────────────────────
@app.cell
def __(Image, base64, frame_sl, io, mo, rendered_frames):
    _f   = min(frame_sl.value, len(rendered_frames)-1)
    _img = Image.fromarray(rendered_frames[_f])
    _buf = io.BytesIO()
    _img.save(_buf, format='PNG')
    _b64 = base64.b64encode(_buf.getvalue()).decode()
    mo.Html(
        f'<img src="data:image/png;base64,{_b64}" '
        f'style="width:100%;max-width:1120px;border-radius:6px;'
        f'box-shadow:0 2px 14px #0003">'
    )
    return


@app.cell
def __(mo):
    mo.md(r"""
    ## The intersection method

    When edge $AC$ flips to $BD$, we need a control value at the midpoint $M$ of $BD$.

    The two diagonals intersect at a point $P$ with parameter $t_P$ along $BD$:

    $$P = B + t_P\,(D - B), \qquad t_P \in (0,1)$$

    Because $P$ lies **on** old edge $AC$, both triangles sharing $AC$ agree on $v_P = \text{spline}(P)$.

    We then fit the unique 1-D quadratic through three known points on edge $BD$:

    $$q(0) = v_B, \quad q(t_P) = v_P, \quad q(1) = v_D$$

    Solving for the coefficient $a$ in $q(t) = at^2 + bt + c$:

    $$a = \frac{v_P - v_B - t_P(v_D - v_B)}{t_P^2 - t_P}$$

    The midpoint control value is:

    $$q\!\left(\tfrac{1}{2}\right) = -\tfrac{1}{4}a + \tfrac{1}{2}(v_B + v_D)$$

    This is a **smooth rational function** of the vertex positions, so the spline
    varies as smoothly across the flip as the vertices themselves move.
    """)
    return


if __name__ == "__main__":
    app.run()
