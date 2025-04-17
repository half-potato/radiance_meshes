#define FLT_MAX 1e20
#define FLT_MIN -1e20
import safe_math;
import utils;
import spherical_harmonics;
import intersect;
import minkowski;
import tet_area;
import stable_power;

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

float get_max_alpha(float z, uint4 virtual_tet, TensorView<float> densities) {
    float density = (
        densities[virtual_tet.x] + densities[virtual_tet.y] + densities[virtual_tet.z] + densities[virtual_tet.w])/4;
    float max_alpha = 1 - safe_exp(-z * density);
    return max_alpha;
}

int32_t count_tiles(Tetrahedra tet, Camera cam, rectangle rect_tile_space, int32_t tile_width, int32_t tile_height) {
    float2 points2d[4];
    points2d[0] = point2image(tet.verts[0], cam);
    points2d[1] = point2image(tet.verts[1], cam);
    points2d[2] = point2image(tet.verts[2], cam);
    points2d[3] = point2image(tet.verts[3], cam);
    Triangle2D tris[4];
    // float m = 1.1*sqrt(2);
    float m = 0.6;
    tris[0] = grow_triangle({points2d[0], points2d[1], points2d[2]}, m*(float)max(tile_height, tile_width));
    tris[1] = grow_triangle({points2d[0], points2d[1], points2d[3]}, m*(float)max(tile_height, tile_width));
    tris[2] = grow_triangle({points2d[0], points2d[2], points2d[3]}, m*(float)max(tile_height, tile_width));
    tris[3] = grow_triangle( { points2d[1], points2d[2], points2d[3] }, m * (float)max(tile_height, tile_width));
    // Polygon2D poly[4];
    // poly[0] = grow_triangle_linf({points2d[0], points2d[1], points2d[2]}, m*(float)max(tile_height, tile_width));
    // poly[1] = grow_triangle_linf({points2d[0], points2d[1], points2d[3]}, m*(float)max(tile_height, tile_width));
    // poly[2] = grow_triangle_linf({points2d[0], points2d[2], points2d[3]}, m*(float)max(tile_height, tile_width));
    // poly[3] = grow_triangle_linf({ points2d[1], points2d[2], points2d[3]}, m * (float)max(tile_height, tile_width));

    int32_t n_tiles_refine = 0;
    for (int y = rect_tile_space.min_y; y <= rect_tile_space.max_y; y++) {
        for (int x = rect_tile_space.min_x; x <= rect_tile_space.max_x; x++) {
            float2 pixel_center = {(x + 0.5f)*tile_width, (y + 0.5f)*tile_height};
            // for (int i = 0; i < 4; i++) {
            //     if (is_point_inside_triangle(pixel_center, tris[i])) {
            //         n_tiles_refine++;
            //         break;
            //     }
            // }
            bool contained = false;
            for (int i = 0; i < 4; i++) {
                if (is_point_inside_triangle(pixel_center, tris[i])) {
                // if (in_polygon(pixel_center, poly[i])) {
                    contained = true;
                    break;
                }
            }
            if (!contained) continue;
            n_tiles_refine++;
        }
    }
    // int32_t n_tiles = (rect_tile_space.max_x - rect_tile_space.min_x) * (rect_tile_space.max_y - rect_tile_space.min_y);
    // printf("Refined from %i->%i,", n_tiles, n_tiles_refine);
    return n_tiles_refine;
}

void print_vec(float3 vec) {
    printf("(%f, %f, %f),", vec.x, vec.y, vec.z);
}


[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void vertex_shader(TensorView<int32_t> indices,
                   DiffTensorView vertices,
                   TensorView<float> world_view_transform,
                   TensorView<float> K,
                   TensorView<float> cam_pos,
                   TensorView<float> out_tet_area,
                   TensorView<int32_t> out_tiles_touched,
                   TensorView<int32_t> out_rect_tile_space,
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
    if (is_planar(tet)) {
        return;
    }

    Tetrahedra camspace_tet = tet2camspace(tet, cam);
    // printf("world: ");
    // print_vec(tet.verts[0]);
    // print_vec(tet.verts[1]);
    // print_vec(tet.verts[2]);
    // print_vec(tet.verts[3]);
    // printf("\norigin: ");
    // print_vec(cam.position);
    // printf("\ncamspace: ");
    // print_vec(camspace_tet.verts[0]);
    // print_vec(camspace_tet.verts[1]);
    // print_vec(camspace_tet.verts[2]);
    // print_vec(camspace_tet.verts[3]);
    no_diff float3 minv, maxv;
    no_diff find_extent(camspace_tet, minv, maxv, cam);
    // no_diff float max_alpha = get_max_alpha(maxv.z - minv.z, virtual_tet, densities);
    // printf("(%f, %f, %f) - (%f, %f, %f)\n", minv.x, minv.y, minv.z, maxv.x, maxv.y, maxv.z);

    // Check if tetrahedron is completely outside any frustum plane
    if (maxv.z <= 0.0) {// && max_alpha < 0.5/255.f) {
        return;
    }
    // if (max_alpha < 1 / 255.f) {
    //     return;
    // }

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


    if (contains_ray_origin) {
        rect_tile_space.min_x = 0;
        rect_tile_space.max_x = grid_width;
        rect_tile_space.min_y = 0;
        rect_tile_space.max_y = grid_height;
        out_vs.storeOnce(uint2(g_idx, 1), 1);
    } else {
        rect_tile_space.min_x = int32_t(floor(clip(minv.x / tile_width, 0, grid_width)));
        rect_tile_space.min_y = int32_t(floor(clip(minv.y / tile_height, 0, grid_height)));
        rect_tile_space.max_x = int32_t(ceil(clip(maxv.x / tile_width, 0, grid_width)));
        rect_tile_space.max_y = int32_t(ceil(clip(maxv.y / tile_height, 0, grid_height)));
    }
    int32_t n_tiles = (rect_tile_space.max_x - rect_tile_space.min_x) * (rect_tile_space.max_y - rect_tile_space.min_y);

    if (n_tiles == 0) {
        return;
    }

    // if (contains_ray_origin) {
    //     out_tiles_touched[g_idx] = n_tiles;
    // } else {
    //     out_tiles_touched[g_idx] = count_tiles(tet, cam, rect_tile_space, tile_width, tile_height);
    // }
    Ray center_ray = get_ray(cam, { cam.W / 2, cam.H / 2 }, 0);
    Tetrahedra image_tet = camspace_tet2image_w_depth(camspace_tet, cam);
    float area = tet_area(image_tet, center_ray.d, cam);

    // float3 circumcenter = calculate_circumcenter(tet);
    // float radius = length(circumcenter - tet.verts[0]);
    // float circum_dist = length(circumcenter - cam.position);
    // float sort_depth = circum_dist * circum_dist - radius * radius;


    out_tet_area[g_idx] = area;

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
