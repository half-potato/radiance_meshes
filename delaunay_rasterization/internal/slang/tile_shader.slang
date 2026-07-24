import utils;
import intersect;

uint32_t float2sortable_key(float value)
{
    uint32_t bits = asuint(value);
    if ((bits & 0x80000000) != 0) {
        bits = ~bits;
    } else {
        bits ^= 0x80000000;
    }
    return bits;
}

// Per-tet separating-axis data, precomputed ONCE (independent of tile) so the
// hot per-tile test is just box projections + compares. The candidate axes are
// the (unit) normals of all 6 vertex-pair segments of the projected tet -- a
// superset of the convex-hull edge normals, which is what SAT requires. The two
// box axes never separate for tiles inside the tet's AABB rect, so they are
// omitted. Any axis on which the projections are provably disjoint is a true
// separator, so testing extra (non-hull) axes can never cause an incorrect
// cull; a positive margin further guarantees we never cull a tile the tet
// actually touches.
struct HullSAT {
    float2 axis[6];   // unit normals (zero for degenerate edges -> never separate)
    float pmin[6];
    float pmax[6];
};

HullSAT build_hull_sat(float2 P[4])
{
    static const int2 pairs[6] = {
        int2(0, 1), int2(0, 2), int2(0, 3),
        int2(1, 2), int2(1, 3), int2(2, 3)
    };
    HullSAT s;
    [unroll]
    for (int a = 0; a < 6; ++a) {
        float2 e = P[pairs[a].y] - P[pairs[a].x];
        float2 n = float2(-e.y, e.x);
        float len = length(n);
        if (len < 1e-8) {
            s.axis[a] = float2(0.0, 0.0);   // degenerate edge -> not a separator
            s.pmin[a] = 0.0;
            s.pmax[a] = 0.0;
            continue;
        }
        n /= len;
        float q0 = dot(n, P[0]);
        float q1 = dot(n, P[1]);
        float q2 = dot(n, P[2]);
        float q3 = dot(n, P[3]);
        s.axis[a] = n;
        s.pmin[a] = min(min(q0, q1), min(q2, q3));
        s.pmax[a] = max(max(q0, q1), max(q2, q3));
    }
    return s;
}

// True if the axis-aligned pixel box [bx0,bx1]x[by0,by1] might overlap the tet
// hull; false only when a separating axis (with margin) is found -> safe cull.
bool box_overlaps_hull(HullSAT s,
                       float bx0, float by0, float bx1, float by1,
                       float margin)
{
    [unroll]
    for (int a = 0; a < 6; ++a) {
        float2 n = s.axis[a];
        // A zero axis (degenerate edge) has bmin==bmax==0 and pmin==pmax==0,
        // so it never separates -- no explicit guard needed.
        float c0 = n.x * bx0 + n.y * by0;
        float c1 = n.x * bx1 + n.y * by0;
        float c2 = n.x * bx0 + n.y * by1;
        float c3 = n.x * bx1 + n.y * by1;
        float bmin = min(min(c0, c1), min(c2, c3));
        float bmax = max(max(c0, c1), max(c2, c3));
        if (s.pmax[a] < bmin - margin || bmax < s.pmin[a] - margin) {
            return false;
        }
    }
    return true;
}

[AutoPyBindCUDA]
[CUDAKernel]
void generate_keys(TensorView<float> xyz_vs,
                   TensorView<int32_t> rect_tile_space,
                   TensorView<int32_t> index_buffer_offset,
                   TensorView<float> proj_verts,
                   TensorView<int64_t> out_unsorted_keys,
                   TensorView<int32_t> out_unsorted_gauss_idx,
                   uint grid_height,
                   uint grid_width,
                   uint tile_width,
                   uint tile_height,
                   uint cull_enable,
                   float cull_margin,
                   uint cull_min_tiles)
{
    int32_t globalIdx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (globalIdx >= xyz_vs.size(0))
        return;

    float3 ndc_xyz = {
        xyz_vs[uint2(globalIdx, 0)],
        xyz_vs[uint2(globalIdx, 1)],
        xyz_vs[uint2(globalIdx, 2)]
    };

    // 1. Calculate the allocated range for this tetrahedron
    int32_t start_offset = (globalIdx == 0) ? 0 : index_buffer_offset[globalIdx - 1];
    int32_t end_offset = index_buffer_offset[globalIdx];
    int32_t allocated_count = end_offset - start_offset;

    // 2. Safety: If 0 space was allocated (e.g. filtered in Python), DO NOT WRITE.
    if (allocated_count <= 0) return;

    // 3. Clamp writing pointer to buffer size
    int32_t buffer_limit = int32_t(out_unsorted_keys.size(0));
    if (start_offset >= buffer_limit) return;

    // Hard clamp the end to the buffer size
    if (end_offset > buffer_limit) end_offset = buffer_limit;

    int32_t rect_min_x = rect_tile_space[uint2(globalIdx, 0)];
    int32_t rect_min_y = rect_tile_space[uint2(globalIdx, 1)];
    int32_t rect_max_x = rect_tile_space[uint2(globalIdx, 2)];
    int32_t rect_max_y = rect_tile_space[uint2(globalIdx, 3)];

    // --- Conservative tile culling setup ---
    // A cullable tet (perspective, undistorted, fully in front of the near
    // plane) exposes its 4 projected pixel positions in proj_verts; tiles that
    // provably miss the projected convex hull get a SENTINEL tile id instead of
    // their real (tile,depth) key. Sentinels sort to the very end and fall
    // outside every real tile's [start,end) range, so the alpha blend never
    // loads them -- yet the key buffer stays exactly the same size, so all
    // offsets are unchanged and the output is bitwise identical.
    // Gate culling on footprint: tiny tets (the vast majority -- median ~2
    // tiles) have a projected hull that nearly fills their AABB, so they have
    // essentially no cullable corner tiles. Running the SAT on them is pure
    // overhead, so we only cull tets whose AABB spans >= cull_min_tiles tiles.
    // This is bit-exact regardless of the threshold (skipped tets just emit
    // their full AABB, exactly like the un-culled path).
    int32_t n_rect_tiles = (rect_max_x - rect_min_x) * (rect_max_y - rect_min_y);
    bool do_cull = (cull_enable != 0)
                   && (proj_verts[uint2(globalIdx, 8)] > 0.5)
                   && (n_rect_tiles >= int32_t(cull_min_tiles));
    HullSAT sat;
    if (do_cull) {
        float2 P[4];
        [unroll]
        for (int i = 0; i < 4; ++i) {
            P[i] = float2(proj_verts[uint2(globalIdx, 2 * i + 0)],
                          proj_verts[uint2(globalIdx, 2 * i + 1)]);
        }
        sat = build_hull_sat(P);   // per-tet, done once
    }
    // Safety margin (pixels, host-supplied) absorbs the sub-pixel float
    // mismatch between this projection and the alpha-blend's per-ray
    // intersection. Never cull a tile whose pixel box is within this margin of
    // the hull.
    // Sentinel tile id == grid_width*grid_height, which is >= any real tile id
    // (0..grid_width*grid_height-1) and equals out_tile_ranges.size(0), so
    // compute_tile_ranges treats it as out-of-range and never assigns it.
    uint64_t sentinel_tile = uint64_t(grid_width) * uint64_t(grid_height);

    int32_t current_write_idx = start_offset;

    for (int32_t y = rect_min_y; y < rect_max_y; y++)
    {
        for (int32_t x = rect_min_x; x < rect_max_x; x++)
        {
            // 4. CRITICAL: Stop writing if we exceed our OWN allocated slice.
            // Even if the rect says "write 100", if allocated_count is 50, we stop at 50.
            if (current_write_idx >= end_offset)
                return;

            uint64_t tile_id;
            if (do_cull) {
                // Full continuous pixel box for this tile (covers all ray
                // jitter in [0,1], jitter-independent by construction).
                float bx0 = float(x) * float(tile_width);
                float bx1 = float(x + 1) * float(tile_width);
                float by0 = float(y) * float(tile_height);
                float by1 = float(y + 1) * float(tile_height);
                bool covered = box_overlaps_hull(sat, bx0, by0, bx1, by1, cull_margin);
                tile_id = covered ? uint64_t(y * grid_width + x) : sentinel_tile;
            } else {
                tile_id = uint64_t(y * grid_width + x);
            }

            uint64_t key = tile_id;
            key <<= 32;
            uint32_t fk = float2sortable_key(ndc_xyz.z);
            key = key | fk;

            out_unsorted_keys[current_write_idx] = key;
            out_unsorted_gauss_idx[current_write_idx] = globalIdx;

            current_write_idx++;
        }
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void compute_tile_ranges(TensorView<int64_t> sorted_keys,
                         TensorView<int32_t> out_tile_ranges)
{
    int32_t globalIdx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (globalIdx >= sorted_keys.size(0))
        return;

    // Read tile ID from key.
    uint32_t currtile = uint32_t(uint64_t(sorted_keys[globalIdx]) >> 32);
    uint32_t num_tiles = out_tile_ranges.size(0);

    // Sentinel (culled) keys carry tile id == num_tiles and always sort AFTER
    // every real tile. Each write below is individually guarded by
    // `< num_tiles`, so a sentinel currtile is never assigned a range -- but a
    // real->sentinel transition STILL closes out the preceding real tile's end
    // (unlike an early `return`, which would leave the last real tile's range
    // open). This also preserves the original out-of-bounds safety for garbage
    // keys.
    if (globalIdx == 0)
    {
        if (currtile < num_tiles)
            out_tile_ranges[uint2(currtile, 0)] = 0;
    }
    else
    {
        uint32_t prevtile = uint32_t(uint64_t(sorted_keys[globalIdx - 1]) >> 32);

        if (currtile != prevtile)
        {
            if (prevtile < num_tiles)
                out_tile_ranges[uint2(prevtile, 1)] = globalIdx;
            if (currtile < num_tiles)
                out_tile_ranges[uint2(currtile, 0)] = globalIdx;
        }
    }

    if (globalIdx == sorted_keys.size(0) - 1) {
        if (currtile < num_tiles)
            out_tile_ranges[uint2(currtile, 1)] = int32_t(sorted_keys.size(0));
    }
}
