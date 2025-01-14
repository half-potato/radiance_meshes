#define FLT_MAX 1e20
#define FLT_MIN -1e20
import safe_math;
import utils;
import spherical_harmonics;
import intersect;

struct rectangle {
    int32_t min_x;
    int32_t min_y;
    int32_t max_x;
    int32_t max_y;
};

[Differentiable]
float sigmoid(float x) {
    return safe_div(1.0, (1+safe_exp(-x)));
}

[Differentiable]
float4 make_plane(float3 point, float3 normal) {
    // normalize the normal vector to ensure proper distance calculations
    normal = normalize(normal);
    // plane equation: ax + by + cz + d = 0
    // where (a,b,c) is the normal and d = -(ax0 + by0 + cz0) for point (x0,y0,z0) on plane
    float d = -dot(normal, point);
    return float4(normal, d);
}

bool pointInTriangle(float3 P, float3 A, float3 B, float3 C, out float3 bary) {
    float3 v0 = B - A;
    float3 v1 = C - A;
    float3 v2 = P - A;
    
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    
    float denom = d00 * d11 - d01 * d01;
    bary.y = (d11 * d20 - d01 * d21) / denom;
    bary.z = (d00 * d21 - d01 * d20) / denom;
    bary.x = 1.0f - bary.y - bary.z;
    
    return bary.x >= 0 && bary.y >= 0 && bary.z >= 0 &&
           bary.x <= 1 && bary.y <= 1 && bary.z <= 1;
}

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

bool edge_intersects_frustum(float3 v0, float3 v1, float4 plane) {
    float d0 = dot(float4(v0, 1.0f), plane);
    float d1 = dot(float4(v1, 1.0f), plane);
    // If points are on opposite sides of the plane (different signs)
    // and not both zero, there's an intersection
    return (d0 * d1 < 0);
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
    float radius = length(circumcenter - tet.verts[0]);
    bool contains_ray_origin = point_in_tetrahedron(tet, cam.position);

    Tetrahedra camspace_tet = tet2camspace(tet, cam);

    no_diff float3 minv, maxv;
    no_diff find_extent(camspace_tet, minv, maxv, cam);

    // Check if tetrahedron is completely outside any frustum plane
    if (maxv.z <= 0.0) {
        return;
    }

    rectangle rect_tile_space;
    // if ((minv.z <= 0.0 && maxv.z > 0.0) || contains_ray_origin) {
    if (contains_ray_origin) {
        rect_tile_space.min_x = 0;
        rect_tile_space.max_x = grid_width;
        rect_tile_space.min_y = 0;
        rect_tile_space.max_y = grid_height;
        out_vs.storeOnce(uint2(g_idx, 1), 1);
    // } else if (minv.z >= 0.0) {
    } else {

        rect_tile_space.min_x = int32_t(clip(floor(minv.x / tile_width), 0, grid_width));
        rect_tile_space.max_x = int32_t(ceil(clip((maxv.x) / tile_width, 0, grid_width)));
        rect_tile_space.min_y = int32_t(floor(clip((minv.y) / tile_height, 0, grid_height)));
        rect_tile_space.max_y = int32_t(ceil(clip((maxv.y) / tile_height, 0, grid_height)));
    }
    // else {
    //     // Need to clip the tetrahedra using the camera plane to get the stuff in front
    //     Tetrahedra clipped_tet;
    //     Tetrahedra camspace_tet = tet2camspace(tet, cam);
    //     if (clip_tetrahedra_z0(camspace_tet, clipped_tet)) {
    //         Tetrahedra proj_clipped_tet = tet_camspace2image_w_depth(clipped_tet, cam);
    //         float3 clipped_minv, clipped_maxv;
    //         minmax_tet(proj_clipped_tet, clipped_minv, clipped_maxv);

    //         rect_tile_space.min_x = int32_t(clip(floor(clipped_minv.x / tile_width), 0, grid_width));
    //         rect_tile_space.max_x = int32_t(ceil(clip(clipped_maxv.x / tile_width, 0, grid_width)));
    //         rect_tile_space.min_y = int32_t(floor(clip(clipped_minv.y / tile_height, 0, grid_height)));
    //         rect_tile_space.max_y = int32_t(ceil(clip(clipped_maxv.y / tile_height, 0, grid_height)));
    //     } else {
    //         return;
    //     }
    // }
    int32_t n_tiles = (rect_tile_space.max_x - rect_tile_space.min_x) * (rect_tile_space.max_y - rect_tile_space.min_y);

    if (n_tiles == 0) {
        return;
    }

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

    float circum_dist = length(circumcenter - cam.position);
    float sort_depth = circum_dist*circum_dist - radius*radius;

    // out_vs.storeOnce(uint2(g_idx, 2), min(max(circumcenter_camspace.z, FLT_MIN), FLT_MAX));
    // out_vs.storeOnce(uint2(g_idx, 2), min(max(sort_depth + min_dist, FLT_MIN), FLT_MAX));
    out_vs.storeOnce(uint2(g_idx, 2), min(max(sort_depth, FLT_MIN), FLT_MAX));

    out_circumcenter.storeOnce(uint2(g_idx, 0), circumcenter.x);
    out_circumcenter.storeOnce(uint2(g_idx, 1), circumcenter.y);
    out_circumcenter.storeOnce(uint2(g_idx, 2), circumcenter.z);

}
