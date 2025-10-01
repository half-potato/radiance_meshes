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
                   DiffTensorView out_vs,
                   DiffTensorView out_circumcenter)
{
    uint32_t g_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (g_idx >= indices.size(0))
        return;

    Camera cam = no_diff load_tensor_camera(tcam);
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet = load_tetrahedra(vertices, virtual_tet);
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

    rectangle rect_tile_space;

    double3 A = { double(tet.verts[0].x), double(tet.verts[0].y), double(tet.verts[0].z) };
    double3 B = { double(tet.verts[1].x), double(tet.verts[1].y), double(tet.verts[1].z) };
    double3 C = { double(tet.verts[2].x), double(tet.verts[2].y), double(tet.verts[2].z) };
    double3 D = { double(tet.verts[3].x), double(tet.verts[3].y), double(tet.verts[3].z) };
    double3 P = { double(cam.position.x), double(cam.position.y), double(cam.position.z) };
    double3 circumcenter;
    float sort_depth = float(power_of_circumsphere(A, B, C, D, P, circumcenter));
    if (sort_depth < -1e19) return;

    bool contains_ray_origin = point_in_tetrahedron(tet, cam.position) || sort_depth < 0;
    // printf("contains: %i\n", contains_ray_origin);

    rect_tile_space.min_x = int32_t(floor(clip(minv.x / cam.tile_width, 0, cam.grid_width)));
    rect_tile_space.min_y = int32_t(floor(clip(minv.y / cam.tile_height, 0, cam.grid_height)));
    rect_tile_space.max_x = int32_t(ceil(clip(maxv.x / cam.tile_width, 0, cam.grid_width)));
    rect_tile_space.max_y = int32_t(ceil(clip(maxv.y / cam.tile_height, 0, cam.grid_height)));

    int32_t n_tiles = (rect_tile_space.max_x - rect_tile_space.min_x) * (rect_tile_space.max_y - rect_tile_space.min_y);

    if (n_tiles == 0) {
        return;
    }

    out_tiles_touched[g_idx] = n_tiles;
    out_rect_tile_space[uint2(g_idx, 0)] = rect_tile_space.min_x;
    out_rect_tile_space[uint2(g_idx, 1)] = rect_tile_space.min_y;
    out_rect_tile_space[uint2(g_idx, 2)] = rect_tile_space.max_x;
    out_rect_tile_space[uint2(g_idx, 3)] = rect_tile_space.max_y;

    out_vs.storeOnce(uint2(g_idx, 2), min(max(sort_depth, FLT_MIN), FLT_MAX));

    out_circumcenter.storeOnce(uint2(g_idx, 0), (float)circumcenter.x);
    out_circumcenter.storeOnce(uint2(g_idx, 1), (float)circumcenter.y);
    out_circumcenter.storeOnce(uint2(g_idx, 2), (float)circumcenter.z);

}
