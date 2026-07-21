"""Triton FORWARD kernel for the iNGP multiresolution hash grid.

Bitwise-parity target: `HashEmbedderOptimized.forward` in
`radiance_meshes/utils/hashgrid.py` (PyTorch eager). One kernel launch per level
(tcnn/instant-ngp `kernel_grid` structure) so per-level table pointers/sizes stay
simple; 8 launches cost ~microseconds. No autograd here -- the integration wraps
this pure forward.

Bitwise status (mixed collision-free/hashed config, 8 levels, base 16 / finest
2048, log2_hashmap_size 21, float32, points inside + outside the box):

    torch.equal == True on RTX 3090 / triton 3.6.0 / torch 2.10.

The only floating-point work is `scaled = (clamp(x)-bmin)*scale` and the trilinear
blend. Both are written with the reference's exact op order, and the kernel is
launched with `enable_fp_fusion=False` (ptxas `--fmad=false`) so every `a*w + b*v`
stays a separate rounded mul then add -- matching eager's separate elementwise
kernels rather than fusing into an fma. This is the load-bearing detail for
bitwise parity.

Each corner's F-wide feature row is gathered in one `[BLOCK, F]` load (the F
columns are contiguous in the table). This is a pure load-layout change -- the
per-element arithmetic order is unchanged, so it preserves bitwise parity and
gives a large speedup at F=8 (the training config). See `__main__` for the
self-test that asserts this across both the F=2 and F=8 configs.
"""

import torch
import triton
import triton.language as tl

try:
    from hashgrid import HashEmbedderOptimized          # run as a script (utils/ on path)
except ImportError:                                     # imported as utils.triton_hashgrid_fwd
    from utils.hashgrid import HashEmbedderOptimized


@triton.jit
def _hash_fwd_kernel(
    x_ptr,            # [N, 3] float32, contiguous
    table_ptr,        # [S, F] float32, contiguous (embeddings[i].weight)
    out_ptr,          # [N, L*F] float32; this level writes columns [col:col+F]
    N,
    res_long,         # int, level resolution (int64 arithmetic inside)
    sx, sy, sz,       # float32 per-dim scale = res / (bmax - bmin)
    bmin0, bmin1, bmin2,
    bmax0, bmax1, bmax2,
    hash_mask,        # int, (1<<log2) - 1
    prime1, prime2,   # int, iNGP hash primes (prime0 == 1 implicit)
    row_stride,       # int, == L*F (elements per output row)
    col_offset,       # int, == level_idx * F
    F: tl.constexpr,
    BLOCK: tl.constexpr,
    COLLISION_FREE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # 1. load + clamp to the bounding box (clamp(x, min, max) == min(max(x,lo),hi))
    x = tl.load(x_ptr + offs * 3 + 0, mask=mask, other=0.0)
    y = tl.load(x_ptr + offs * 3 + 1, mask=mask, other=0.0)
    z = tl.load(x_ptr + offs * 3 + 2, mask=mask, other=0.0)
    xc = tl.minimum(tl.maximum(x, bmin0), bmax0)
    yc = tl.minimum(tl.maximum(y, bmin1), bmax1)
    zc = tl.minimum(tl.maximum(z, bmin2), bmax2)

    # 2-4. scale (subtract then multiply), + 0.5, floor -> int64, frac
    ox = (xc - bmin0) * sx + 0.5
    oy = (yc - bmin1) * sy + 0.5
    oz = (zc - bmin2) * sz + 0.5
    blx = tl.floor(ox).to(tl.int64)
    bly = tl.floor(oy).to(tl.int64)
    blz = tl.floor(oz).to(tl.int64)
    fx = ox - blx.to(tl.float32)
    fy = oy - bly.to(tl.float32)
    fz = oz - blz.to(tl.float32)

    res = res_long
    res_sq = res_long * res_long

    # 5. eight corner table indices. iNaming iXYZ = (x-bit, y-bit, z-bit); x is
    #    dim0 (box_offsets outermost bit). Per-axis pieces are factored so each
    #    corner combine is integer-identical to the reference's per-corner form.
    if COLLISION_FREE:
        # wrap each axis coord mod res, then idx = wx + wy*res + wz*res^2
        wx0 = blx % res
        wx1 = (blx + 1) % res
        wy0 = (bly % res) * res
        wy1 = ((bly + 1) % res) * res
        wz0 = (blz % res) * res_sq
        wz1 = ((blz + 1) % res) * res_sq
        i000 = wx0 + wy0 + wz0
        i001 = wx0 + wy0 + wz1
        i010 = wx0 + wy1 + wz0
        i011 = wx0 + wy1 + wz1
        i100 = wx1 + wy0 + wz0
        i101 = wx1 + wy0 + wz1
        i110 = wx1 + wy1 + wz0
        i111 = wx1 + wy1 + wz1
    else:
        # iNGP hash: 0 ^ (cx*1) ^ (cy*prime1) ^ (cz*prime2), then & mask.
        tx0 = blx
        tx1 = blx + 1
        ty0 = bly * prime1
        ty1 = (bly + 1) * prime1
        tz0 = blz * prime2
        tz1 = (blz + 1) * prime2
        i000 = (tx0 ^ ty0 ^ tz0) & hash_mask
        i001 = (tx0 ^ ty0 ^ tz1) & hash_mask
        i010 = (tx0 ^ ty1 ^ tz0) & hash_mask
        i011 = (tx0 ^ ty1 ^ tz1) & hash_mask
        i100 = (tx1 ^ ty0 ^ tz0) & hash_mask
        i101 = (tx1 ^ ty0 ^ tz1) & hash_mask
        i110 = (tx1 ^ ty1 ^ tz0) & hash_mask
        i111 = (tx1 ^ ty1 ^ tz1) & hash_mask

    # weights broadcast over the F feature columns ([BLOCK, 1])
    wx0 = (1.0 - fx)[:, None]
    wx1 = fx[:, None]
    wy0 = (1.0 - fy)[:, None]
    wy1 = fy[:, None]
    wz0 = (1.0 - fz)[:, None]
    wz1 = fz[:, None]

    # 6-8. load each corner's whole F-wide row in one shot ([BLOCK, F]); the F
    #    columns are contiguous in the table so this vectorizes to wide loads.
    #    Arithmetic order is per-element identical to the scalar form, so the
    #    enable_fp_fusion=False bitwise guarantee is preserved by construction.
    foff = tl.arange(0, F)[None, :]
    row_mask = mask[:, None]
    e000 = tl.load(table_ptr + i000[:, None] * F + foff, mask=row_mask, other=0.0)
    e001 = tl.load(table_ptr + i001[:, None] * F + foff, mask=row_mask, other=0.0)
    e010 = tl.load(table_ptr + i010[:, None] * F + foff, mask=row_mask, other=0.0)
    e011 = tl.load(table_ptr + i011[:, None] * F + foff, mask=row_mask, other=0.0)
    e100 = tl.load(table_ptr + i100[:, None] * F + foff, mask=row_mask, other=0.0)
    e101 = tl.load(table_ptr + i101[:, None] * F + foff, mask=row_mask, other=0.0)
    e110 = tl.load(table_ptr + i110[:, None] * F + foff, mask=row_mask, other=0.0)
    e111 = tl.load(table_ptr + i111[:, None] * F + foff, mask=row_mask, other=0.0)

    c0 = e000 * wx0 + e100 * wx1
    c1 = e001 * wx0 + e101 * wx1
    c2 = e010 * wx0 + e110 * wx1
    c3 = e011 * wx0 + e111 * wx1
    c0 = c0 * wy0 + c2 * wy1
    c1 = c1 * wy0 + c3 * wy1
    out = c0 * wz0 + c1 * wz1

    tl.store(out_ptr + offs[:, None] * row_stride + col_offset + foff, out, mask=row_mask)


def build_meta(module: HashEmbedderOptimized):
    """Extract per-level host-side scalars from a reference module (once)."""
    box_min = module.box_min.float()
    box_max = module.box_max.float()
    F = module.n_features_per_level
    L = module.n_levels
    log2 = module.log2_hashmap_size
    hashmap_size = 1 << log2
    hash_mask = hashmap_size - 1
    # primes[1], primes[2] (prime0 == 1 is implicit in the kernel)
    prime1 = int(module.primes[1].item())
    prime2 = int(module.primes[2].item())

    levels = []
    for i in range(L):
        resolution = module.level_resolutions[i]            # float32 scalar tensor
        res_long = int(resolution.long().item())
        # EXACT eager order: resolution / (box_max - box_min), then per-dim.
        scale = (resolution / (box_max - box_min)).float()
        num_voxels = res_long ** 3
        collision_free = bool(module.use_conditional_hashing and (num_voxels <= hashmap_size))
        levels.append(dict(
            res_long=res_long,
            scale=(float(scale[0]), float(scale[1]), float(scale[2])),
            collision_free=collision_free,
        ))

    return dict(
        F=F, L=L, hash_mask=hash_mask, prime1=prime1, prime2=prime2,
        box_min=(float(box_min[0]), float(box_min[1]), float(box_min[2])),
        box_max=(float(box_max[0]), float(box_max[1]), float(box_max[2])),
        levels=levels,
    )


def hash_fwd(x, embeddings_list, meta, BLOCK=256):
    """Forward hash-grid encoding. Returns [N, L*F] float32.

    x: [N, 3] float32 cuda contiguous. embeddings_list: module.embeddings
    (indexable to nn.Embedding per level). meta: build_meta(module) output.
    """
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N = x.shape[0]
    F = meta["F"]
    L = meta["L"]
    row_stride = L * F
    out = torch.empty((N, row_stride), dtype=torch.float32, device=x.device)

    bmin = meta["box_min"]
    bmax = meta["box_max"]
    grid = (triton.cdiv(N, BLOCK),)

    for i in range(L):
        lvl = meta["levels"][i]
        _e = embeddings_list[i]
        table = (_e.weight if hasattr(_e, 'weight') else _e).contiguous()
        sx, sy, sz = lvl["scale"]
        _hash_fwd_kernel[grid](
            x, table, out,
            N,
            lvl["res_long"],
            sx, sy, sz,
            bmin[0], bmin[1], bmin[2],
            bmax[0], bmax[1], bmax[2],
            meta["hash_mask"],
            meta["prime1"], meta["prime2"],
            row_stride,
            i * F,
            F=F,
            BLOCK=BLOCK,
            COLLISION_FREE=lvl["collision_free"],
            # ptxas --fmad=false: no mul+add -> fma contraction, so the trilinear
            # blend matches eager's separate elementwise mul/add bitwise.
            enable_fp_fusion=False,
        )
    return out


def _ulp_diff(a, b):
    ai = a.view(torch.int32).to(torch.int64)
    bi = b.view(torch.int32).to(torch.int64)
    # map to a monotonic ordering so ulp distance is meaningful across sign
    ai = torch.where(ai < 0, torch.tensor(0x80000000, device=a.device, dtype=torch.int64) - ai, ai)
    bi = torch.where(bi < 0, torch.tensor(0x80000000, device=b.device, dtype=torch.int64) - bi, bi)
    return (ai - bi).abs()


def _make_points(N, device):
    """Mixed point set: mostly inside the box, a chunk outside (clamp path),
    a chunk on exact integer corners, and a chunk of exact zeros."""
    x = (torch.rand(N, 3, device=device) * 2.4 - 1.2)                     # some outside [-1,1]
    x[:5000] = torch.randint(-2, 3, (5000, 3), device=device).float()     # exact-corner / OOB
    x[5000:6000] = 0.0
    return x


def _run_selftest(name, *, n_features_per_level, log2_hashmap_size,
                  base_resolution, finest_resolution, n_levels, device, time_it=False):
    print(f"\n===== config: {name} "
          f"(F={n_features_per_level}, log2={log2_hashmap_size}, "
          f"base={base_resolution}, finest={finest_resolution}, L={n_levels}) =====")
    box = (torch.tensor([-1.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0]))
    module = HashEmbedderOptimized(
        bounding_box=box,
        n_levels=n_levels,
        n_features_per_level=n_features_per_level,
        log2_hashmap_size=log2_hashmap_size,
        base_resolution=base_resolution,
        finest_resolution=finest_resolution,
        use_conditional_hashing=True,
    ).to(device)

    for i, lvl_res in enumerate(module.level_resolutions):
        r = int(lvl_res.item())
        cf = (r ** 3) <= (1 << module.log2_hashmap_size)
        print(f"level {i}: res={r:6d} table={module.embeddings[i].weight.shape[0]:9d} "
              f"collision_free={cf}")

    meta = build_meta(module)
    x = _make_points(100_000, device)
    with torch.no_grad():
        ref = module.forward(x)
        tri = hash_fwd(x, module.embeddings, meta)

    equal = torch.equal(ref, tri)
    print(f"torch.equal (bitwise): {equal}")
    if not equal:
        abs_diff = (ref - tri).abs()
        ulp = _ulp_diff(ref, tri)
        print(f"max abs diff: {abs_diff.max().item():.3e}   max ulp diff: {ulp.max().item()}")
        F = meta["F"]
        for i in range(meta["L"]):
            sl = slice(i * F, i * F + F)
            print(f"  level {i}: max_ulp={ulp[:, sl].max().item()} "
                  f"max_abs={abs_diff[:, sl].max().item():.3e}")
    else:
        print("forward matches reference bitwise.")

    if time_it:
        import time
        NB = 3_400_000
        xb = (torch.rand(NB, 3, device=device) * 2.2 - 1.1)
        with torch.no_grad():
            for _ in range(3):
                module.forward(xb); hash_fwd(xb, module.embeddings, meta)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                module.forward(xb)
            torch.cuda.synchronize()
            t_eager = (time.time() - t0) / 20
            t0 = time.time()
            for _ in range(20):
                hash_fwd(xb, module.embeddings, meta)
            torch.cuda.synchronize()
            t_tri = (time.time() - t0) / 20
        print(f"timing @ N={NB:,}: eager {t_eager*1e3:.2f} ms, triton {t_tri*1e3:.2f} ms, "
              f"speedup {t_eager/t_tri:.2f}x")
    return equal


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    ok = []
    # spec self-test config: F=2, log2=21, mixed collision-free (levels 0-3) / hashed
    ok.append(_run_selftest(
        "spec (F=2/log2=21)", n_features_per_level=2, log2_hashmap_size=21,
        base_resolution=16, finest_resolution=2048, n_levels=8,
        device=device, time_it=True))
    # live train.py config: F=8, log2=23, base 64, finest 64*2**8=16384, only
    # levels 0-1 collision-free; verifies int64 hash products at res 16384
    # (coord ~2^14 x prime ~2^31.3 ~ 2^45.3, well within int64).
    ok.append(_run_selftest(
        "train.py (F=8/log2=23)", n_features_per_level=8, log2_hashmap_size=23,
        base_resolution=64, finest_resolution=64 * 2 ** 8, n_levels=8,
        device=device, time_it=True))

    print(f"\nALL BITWISE-EQUAL: {all(ok)}")
