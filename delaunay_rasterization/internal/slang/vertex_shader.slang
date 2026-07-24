#define FLT_MAX 1e20
#define FLT_MIN -1e20
import safe_math;
import utils;
import intersect;
import stable_power;
import camera;

void print_vec(float3 vec) {
    printf("(%f, %f, %f),", vec.x, vec.y, vec.z);
}

struct rectangle {
    int32_t min_x;
    int32_t min_y;
    int32_t max_x;
    int32_t max_y;
};

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void vertex_shader(TensorView<int32_t> indices,
                   DiffTensorView vertices,
                   TensorCamera tcam,
                   TensorView<int32_t> out_tiles_touched,
                   TensorView<int32_t> out_rect_tile_space,
                   TensorView<float> out_proj_verts,
                   TensorView<float> out_extent,
                   uint cull_enable,
                   DiffTensorView out_vs,
                   DiffTensorView out_circumcenter)
{
    uint32_t g_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (g_idx >= indices.size(0))
        return;

    Camera cam = no_diff load_tensor_camera(tcam);
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    
    // --- DEFENSE LAYER 1: Vertex Index Bounds ---
    // Approximate triangulation can produce indices pointing to deleted or non-existent vertices.
    // This prevents the "Illegal Memory Access" crash in load_tetrahedra.
    uint32_t num_verts = vertices.size(0);
    if (virtual_tet.x >= num_verts || virtual_tet.y >= num_verts || 
        virtual_tet.z >= num_verts || virtual_tet.w >= num_verts) 
    {
        return;
    }

    Tetrahedra tet = load_tetrahedra(vertices, virtual_tet);

    // --- DEFENSE LAYER 2: NaN Geometry ---
    // If vertices contain NaN, math downstream will explode.
    // Checking one component per vertex is usually sufficient to catch garbage data.
    if (isnan(tet.verts[0].x) || isnan(tet.verts[1].x) || 
        isnan(tet.verts[2].x) || isnan(tet.verts[3].x)) 
    {
        return;
    }

    if (is_planar(tet)) {
        return;
    }

    Tetrahedra camspace_tet = tet2camspace(tet, cam);
    no_diff float3 minv, maxv;
    no_diff find_extent(camspace_tet, minv, maxv, cam);

    // Check if tetrahedron is completely outside any frustum plane
    if (maxv.z <= 0.0) {
        return;
    }
    
    // --- DEFENSE LAYER 3: Finite Projection ---
    // If a point is exactly on the camera plane or W=0, projection can produce Infinity/NaN.
    // This prevents INT_MIN/INT_MAX in the rect calculation below.
    if (!isfinite(minv.x) || !isfinite(minv.y) || !isfinite(maxv.x) || !isfinite(maxv.y)) {
        return;
    }

    rectangle rect_tile_space;

    double3 A = { double(tet.verts[0].x), double(tet.verts[0].y), double(tet.verts[0].z) };
    double3 B = { double(tet.verts[1].x), double(tet.verts[1].y), double(tet.verts[1].z) };
    double3 C = { double(tet.verts[2].x), double(tet.verts[2].y), double(tet.verts[2].z) };
    double3 D = { double(tet.verts[3].x), double(tet.verts[3].y), double(tet.verts[3].z) };
    double3 P = { double(cam.position.x), double(cam.position.y), double(cam.position.z) };
    double3 circumcenter;
    float sort_depth = float(power_of_circumsphere(A, B, C, D, P, circumcenter));
    
    // Catch bad sort depths (usually result of near-degenerate geometry)
    if (sort_depth < -1e19 || isnan(sort_depth)) return;

    bool contains_ray_origin = point_in_tetrahedron(tet, cam.position) || sort_depth < 0;

    rect_tile_space.min_x = int32_t(floor(clip(minv.x / cam.tile_width, 0, cam.grid_width)));
    rect_tile_space.min_y = int32_t(floor(clip(minv.y / cam.tile_height, 0, cam.grid_height)));
    rect_tile_space.max_x = int32_t(ceil(clip(maxv.x / cam.tile_width, 0, cam.grid_width)));
    rect_tile_space.max_y = int32_t(ceil(clip(maxv.y / cam.tile_height, 0, cam.grid_height)));

    int32_t n_tiles = (rect_tile_space.max_x - rect_tile_space.min_x) * (rect_tile_space.max_y - rect_tile_space.min_y);

    if (n_tiles <= 0) {
        return;
    }

    out_tiles_touched[g_idx] = n_tiles;

    // Max projected pixel extent of the tet's screen AABB (in PIXELS, not tiles).
    // minv/maxv come from find_extent (image-space, pre tile division). Consumed
    // by the host stochastic size-proportional sampler: p_i = min(s_i, 1). Left
    // at 0 (host pre-zero) for any tet that returned early (non-visible/culled).
    out_extent[g_idx] = max(maxv.x - minv.x, maxv.y - minv.y);

    out_rect_tile_space[uint2(g_idx, 0)] = rect_tile_space.min_x;
    out_rect_tile_space[uint2(g_idx, 1)] = rect_tile_space.min_y;
    out_rect_tile_space[uint2(g_idx, 2)] = rect_tile_space.max_x;
    out_rect_tile_space[uint2(g_idx, 3)] = rect_tile_space.max_y;

    // --- Conservative tile-culling metadata ---
    // The set of pixels whose (pinhole) camera ray hits a convex tet is exactly
    // the convex hull of the 4 projected vertices. generate_keys uses these 4
    // projected pixel positions to skip AABB tiles the hull does not touch.
    // This is only valid when the projection is a well-defined central
    // projection: perspective camera, no lens distortion, and ALL four verts
    // strictly in front of the near plane (so no vertex was clipped in
    // find_extent). Otherwise we emit a "not cullable" flag and generate_keys
    // falls back to full-AABB emission. cols 0..7 = (x,y) per vert, col 8 = flag.
    // Gated on cull_enable so the default (culling-off) path pays nothing here
    // and stays byte-identical to the pre-culling pipeline (out_proj_verts is
    // pre-zeroed by the host, so a skipped col-8 flag reads back as 0).
    if (cull_enable != 0) {
        float near_eps = max(cam.min_t, 1e-4);
        bool no_distortion = dot(cam.distortion_params, cam.distortion_params) < 1e-9;
        bool all_in_front = (camspace_tet.verts[0].z > near_eps) &&
                            (camspace_tet.verts[1].z > near_eps) &&
                            (camspace_tet.verts[2].z > near_eps) &&
                            (camspace_tet.verts[3].z > near_eps);
        float cull_flag = 0.0;
        if (cam.camera_type == 0 && no_distortion && all_in_front) {
            cull_flag = 1.0;
            [unroll]
            for (int i = 0; i < 4; ++i) {
                float3 pv = camspace_to_image(camspace_tet.verts[i], cam);
                out_proj_verts[uint2(g_idx, 2 * i + 0)] = pv.x;
                out_proj_verts[uint2(g_idx, 2 * i + 1)] = pv.y;
            }
        }
        out_proj_verts[uint2(g_idx, 8)] = cull_flag;
    }

    out_vs.storeOnce(uint2(g_idx, 2), min(max(sort_depth, FLT_MIN), FLT_MAX));

    out_circumcenter.storeOnce(uint2(g_idx, 0), (float)circumcenter.x);
    out_circumcenter.storeOnce(uint2(g_idx, 1), (float)circumcenter.y);
    out_circumcenter.storeOnce(uint2(g_idx, 2), (float)circumcenter.z);
}
