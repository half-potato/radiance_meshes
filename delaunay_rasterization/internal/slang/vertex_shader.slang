#define FLT_MAX 1e20
#define FLT_MIN -1e20
import utils;
import spherical_harmonics;

struct rectangle {
    int32_t min_x;
    int32_t min_y;
    int32_t max_x;
    int32_t max_y;
};


rectangle get_rectangle_tile_space(
    float2 ndc_xy,
    float height,
    float width,
    uint grid_height,
    uint grid_width,
    uint tile_height,
    uint tile_width) {

    rectangle rect_tile_space;

    rect_tile_space.min_x = int32_t(floor(clip((ndc_xy.x - width) / tile_width, 0, grid_width)));
    rect_tile_space.min_y = int32_t(floor(clip((ndc_xy.y - height) / tile_height, 0, grid_height)));
    rect_tile_space.max_x = int32_t(ceil(clip((ndc_xy.x + width) / tile_width, 0, grid_width)));
    rect_tile_space.max_y = int32_t(ceil(clip((ndc_xy.y + height) / tile_height, 0, grid_height)));

    return rect_tile_space;
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void vertex_shader(TensorView<int32_t> indices,
                   DiffTensorView vertices,
                   TensorView<float> world_view_transform,
                   TensorView<float> K,
                   TensorView<float> cam_pos,
                   TensorView<int32_t> out_tiles_touched,
                   TensorView<int32_t> out_rect_tile_space,
                   TensorView<int32_t> out_radii,
                   DiffTensorView out_vs,
                   DiffTensorView out_circumcenter,
                   no_diff float fovy,
                   no_diff float fovx,
                   uint image_height,
                   uint image_width,
                   uint grid_height,
                   uint grid_width,
                   uint tile_height,
                   uint tile_width)
{
    uint32_t g_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (g_idx >= indices.size(0))
        return;

    Camera cam = no_diff load_camera(world_view_transform, K, cam_pos, fovy, fovx, image_height, image_width);
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet = load_tetrahedra(vertices, virtual_tet);
    Tetrahedra pixel_tet = project_tetrahedra(tet, cam);
    // Tetrahedra pixel_tet = tet_ndc2pix(proj_tet, image_height, image_width);
    float3 circumcenter = calculate_circumcenter(tet);

    float3 minv, maxv;
    minmax_tet(pixel_tet, minv, maxv);
    bool contains_ray_origin = point_in_tetrahedron(tet, cam.position);

    rectangle rect_tile_space;
    if (contains_ray_origin) {
        rect_tile_space.min_x = 0;
        rect_tile_space.max_x = grid_width;
        rect_tile_space.min_y = 0;
        rect_tile_space.max_y = grid_height;
        out_vs.storeOnce(uint2(g_idx, 1), 1);
    } else {
        // out_vs.storeOnce(uint2(g_idx, 0), minv.x);
        // out_vs.storeOnce(uint2(g_idx, 1), minv.y);
        // out_vs.storeOnce(uint2(g_idx, 2), min(max(minv.z, FLT_MIN), FLT_MAX));

        // out_vs.storeOnce(uint2(g_idx, 0), minv.x);
        // out_vs.storeOnce(uint2(g_idx, 1), minv.y);
        // out_vs.storeOnce(uint2(g_idx, 2), min(max(minv.z, FLT_MIN), FLT_MAX));

        if (maxv.z <= 0.0) {
            return;
        }

        rect_tile_space.min_x = int32_t(clip(floor(minv.x / tile_width), 0, grid_width));
        rect_tile_space.max_x = int32_t(ceil(clip((maxv.x) / tile_width, 0, grid_width)));
        rect_tile_space.min_y = int32_t(floor(clip((minv.y) / tile_height, 0, grid_height)));
        rect_tile_space.max_y = int32_t(ceil(clip((maxv.y) / tile_height, 0, grid_height)));
    }
    int32_t n_tiles = (rect_tile_space.max_x - rect_tile_space.min_x) * (rect_tile_space.max_y - rect_tile_space.min_y);

    if (n_tiles == 0) {
        return;
    }
    // 
    // float3 test_pt = tet.verts[0];
    // float3 proj_point = geom_transform_points(test_pt, mul(cam.proj_mat, cam.world_view_transform));
    // // float3 proj_point = geom_transform_points(test_pt, cam.world_view_transform);
    // float4 proj_test = mul(mul(cam.proj_mat, cam.world_view_transform), float4(test_pt, 1.f));

    // printf("[%f, %f], [%f, %f], (%f, %f, %f). (%f, %f, %f, %f)\n",
    //     //    rect_tile_space.min_x, rect_tile_space.max_x, rect_tile_space.min_y, rect_tile_space.max_y,
    //        minv.x, maxv.x, minv.y, maxv.y,
    //     //    proj_tet.verts[0].x, proj_tet.verts[0].y, proj_tet.verts[0].z,
    //     //    test_pt.x, test_pt.y, test_pt.z,
    //     //    proj_point.x, proj_point.y, proj_point.z,
    //        proj_point.x, proj_point.y, proj_point.z,
    //        proj_test.x, proj_test.y, proj_test.z, proj_test.w);
    // printf("%i,[%f, %f], [%f, %f], [%i, %i], [%i, %i]\n", n_tiles,
    //        minv.x, maxv.x, minv.y, maxv.y,
    //        rect_tile_space.min_x, rect_tile_space.max_x, rect_tile_space.min_y, rect_tile_space.max_y,
    // );

    // float3 circumcenter_camspace = project_point_w_depth(circumcenter, cam);

    out_radii[g_idx] = (uint32_t)1;
    out_tiles_touched[g_idx] = n_tiles;
    out_rect_tile_space[uint2(g_idx, 0)] = rect_tile_space.min_x;
    out_rect_tile_space[uint2(g_idx, 1)] = rect_tile_space.min_y;
    out_rect_tile_space[uint2(g_idx, 2)] = rect_tile_space.max_x;
    out_rect_tile_space[uint2(g_idx, 3)] = rect_tile_space.max_y;

    // out_vs.storeOnce(uint2(g_idx, 0), minv.x);
    // out_vs.storeOnce(uint2(g_idx, 1), minv.y);
    out_vs.storeOnce(uint2(g_idx, 2), min(max(minv.z, FLT_MIN), FLT_MAX));
    // out_vs.storeOnce(uint2(g_idx, 2), min(max(maxv.z, FLT_MIN), FLT_MAX));

    out_circumcenter.storeOnce(uint2(g_idx, 0), circumcenter.x);
    out_circumcenter.storeOnce(uint2(g_idx, 1), circumcenter.y);
    out_circumcenter.storeOnce(uint2(g_idx, 2), circumcenter.z);

}
