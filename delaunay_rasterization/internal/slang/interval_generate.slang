// Interval generation compute prepass.
// 1 thread per visible tet: projects 4 vertices to pixel space,
// classifies the silhouette, and writes fan vertex data.

import utils;
import safe_math;
import camera;
import intersect;
import interp_version;

[Differentiable]
float3[4] load_vert_normals(no_diff uint4 vtet, DiffTensorView vertex_normals) {
    float3[4] vn;
    vn[0] = read_t3_float3(vtet.x, vertex_normals);
    vn[1] = read_t3_float3(vtet.y, vertex_normals);
    vn[2] = read_t3_float3(vtet.z, vertex_normals);
    vn[3] = read_t3_float3(vtet.w, vertex_normals);
    return vn;
}

// Classify silhouette: determine how the tet projects onto screen.
// Case 1 (3 fan tris): One vertex projects inside the triangle formed by the
//   other three vertices' projections. That vertex is the "center" and the
//   other 3 form the silhouette boundary.
// Case 2 (4 fan tris): Two opposite edges cross in screen space. The crossing
//   point becomes the center and 4 silhouette vertices form the boundary.
// Returns: num_sil_verts (3 or 4), center_idx, and sil_indices[4].

struct SilhouetteClassification {
    uint num_sil_verts;    // 3 or 4 (0 = degenerate)
    int center_idx;        // Index of interior vertex (Case 1) or -1 (Case 2)
    float2 center_pixel;   // Screen position of center
    uint sil_indices[4];   // Silhouette vertex indices (CCW order)
    float2 crossing_point; // For Case 2: the edge crossing point in pixels
};

// Check if a 2D point is inside a triangle formed by 3 other 2D points
bool is_inside_triangle_2d(float2 p, float2 a, float2 b, float2 c) {
    float2 v0 = c - a;
    float2 v1 = b - a;
    float2 v2 = p - a;

    float dot00 = dot(v0, v0);
    float dot01 = dot(v0, v1);
    float dot02 = dot(v0, v2);
    float dot11 = dot(v1, v1);
    float dot12 = dot(v1, v2);

    float inv_denom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    return (u >= -1e-4f) && (v >= -1e-4f) && (u + v <= 1.0f + 1e-4f);
}

// Check if two 2D line segments (p1-p2) and (p3-p4) intersect.
// Returns the intersection parameter t along (p1-p2) if they do.
bool segments_intersect_2d(float2 p1, float2 p2, float2 p3, float2 p4,
                           out float t_out, out float s_out) {
    float2 d1 = p2 - p1;
    float2 d2 = p4 - p3;
    float denom = d1.x * d2.y - d1.y * d2.x;
    if (abs(denom) < 1e-8f) {
        t_out = 0; s_out = 0;
        return false;
    }
    float2 d3 = p3 - p1;
    float t = (d3.x * d2.y - d3.y * d2.x) / denom;
    float s = (d3.x * d1.y - d3.y * d1.x) / denom;
    t_out = t;
    s_out = s;
    return (t > 1e-4f && t < 1.0f - 1e-4f && s > 1e-4f && s < 1.0f - 1e-4f);
}

// The 6 edges of a tetrahedron as pairs of vertex indices,
// with opposite edge pairs: (0,1)-(2,3), (0,2)-(1,3), (0,3)-(1,2)
static const uint2 kTetEdges[6] = {
    uint2(0, 1), uint2(2, 3),  // opposite pair 0
    uint2(0, 2), uint2(1, 3),  // opposite pair 1
    uint2(0, 3), uint2(1, 2),  // opposite pair 2
};

SilhouetteClassification classify_silhouette(float2 pix[4]) {
    SilhouetteClassification result;
    result.num_sil_verts = 0;
    result.center_idx = -1;
    result.center_pixel = float2(0);
    result.crossing_point = float2(0);

    // Case 1: Check if any vertex projects inside the triangle of the other three
    // Opposite face indices for each vertex
    static const uint3 kOppFace[4] = {
        uint3(1, 2, 3),
        uint3(0, 2, 3),
        uint3(0, 1, 3),
        uint3(0, 1, 2),
    };

    for (int i = 0; i < 4; ++i) {
        uint3 f = kOppFace[i];
        if (is_inside_triangle_2d(pix[i], pix[f.x], pix[f.y], pix[f.z])) {
            result.num_sil_verts = 3;
            result.center_idx = i;
            result.center_pixel = pix[i];
            result.sil_indices[0] = f.x;
            result.sil_indices[1] = f.y;
            result.sil_indices[2] = f.z;
            result.sil_indices[3] = 0; // unused

            // Order silhouette vertices CCW around center
            // Check winding: if cross product is negative, swap to make CCW
            float2 e0 = pix[result.sil_indices[0]] - result.center_pixel;
            float2 e1 = pix[result.sil_indices[1]] - result.center_pixel;
            float cross_val = e0.x * e1.y - e0.y * e1.x;
            if (cross_val < 0) {
                // Swap to fix winding
                uint temp = result.sil_indices[1];
                result.sil_indices[1] = result.sil_indices[2];
                result.sil_indices[2] = temp;
            }
            return result;
        }
    }

    // Case 2: Check opposite edge pairs for crossing
    for (int pair = 0; pair < 3; ++pair) {
        uint2 e0 = kTetEdges[pair * 2];
        uint2 e1 = kTetEdges[pair * 2 + 1];
        float t, s;
        if (segments_intersect_2d(pix[e0.x], pix[e0.y], pix[e1.x], pix[e1.y], t, s)) {
            result.num_sil_verts = 4;
            result.center_idx = -1;
            // The crossing point is the center
            result.crossing_point = lerp(pix[e0.x], pix[e0.y], t);
            result.center_pixel = result.crossing_point;

            // Silhouette: 4 vertices arranged around the crossing
            // Order: e0.x, e1.x, e0.y, e1.y (forming a quad around the crossing)
            result.sil_indices[0] = e0.x;
            result.sil_indices[1] = e1.x;
            result.sil_indices[2] = e0.y;
            result.sil_indices[3] = e1.y;

            // Ensure CCW winding by checking cross product of first two edges
            float2 ea = pix[result.sil_indices[0]] - result.center_pixel;
            float2 eb = pix[result.sil_indices[1]] - result.center_pixel;
            float cross_val = ea.x * eb.y - ea.y * eb.x;
            if (cross_val < 0) {
                // Swap sil_indices[1] and sil_indices[3]
                uint temp = result.sil_indices[1];
                result.sil_indices[1] = result.sil_indices[3];
                result.sil_indices[3] = temp;
            }

            return result;
        }
    }

    // Degenerate: no interior vertex and no crossing edges
    return result;
}

// Compute barycentric coordinates of a point in a tet (world space)
// Returns float4 bary where p = bary[0]*v0 + bary[1]*v1 + bary[2]*v2 + bary[3]*v3
[Differentiable]
float4 tet_barycentric(float3 p, float3 v0, float3 v1, float3 v2, float3 v3) {
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 e3 = v3 - v0;
    float3 pv = p - v0;

    float3 c23 = cross(e2, e3);
    float3 c31 = cross(e3, e1);
    float3 c12 = cross(e1, e2);
    float inv_det = safe_div(1.0f, dot(e1, c23));

    float b1 = dot(pv, c23) * inv_det;
    float b2 = dot(pv, c31) * inv_det;
    float b3 = dot(pv, c12) * inv_det;
    float b0 = 1.0f - b1 - b2 - b3;

    return float4(b0, b1, b2, b3);
}

// Compute interpolated vertex normal at a point inside the tet
[Differentiable]
float3 interp_normal(float4 bary, float3[4] vn) {
    return bary[0] * vn[0] + bary[1] * vn[1] + bary[2] * vn[2] + bary[3] * vn[3];
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void interval_generate(
    TensorView<int32_t> indices,
    DiffTensorView vertices,
    DiffTensorView vertex_normals,
    DiffTensorView cell_values,
    DiffTensorView interval_verts,       // [N, 5, 10]
    DiffTensorView interval_tet_data,    // [N, 7 + AUX_DIM]
    TensorView<int32_t> interval_meta,  // [N]
    TensorCamera tcam)
{
    /*
    uint g_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (g_idx >= indices.size(0))
        return;

    Camera cam = no_diff load_tensor_camera(tcam);

    // Load tet indices and vertices
    no_diff uint4 vtet = load_virtual_tetrahedra(g_idx, indices);
    uint32_t num_verts = vertices.size(0);
    if (vtet.x >= num_verts || vtet.y >= num_verts ||
        vtet.z >= num_verts || vtet.w >= num_verts) {
        interval_meta[g_idx] = 0;
        return;
    }

    Tetrahedra tet = load_tetrahedra(vertices, vtet);

    // Check for NaN
    if (isnan(tet.verts[0].x) || isnan(tet.verts[1].x) ||
        isnan(tet.verts[2].x) || isnan(tet.verts[3].x)) {
        interval_meta[g_idx] = 0;
        return;
    }

    // Project to camera space
    float3 cam_verts[4];
    cam_verts[0] = point2camspace(tet.verts[0], cam);
    cam_verts[1] = point2camspace(tet.verts[1], cam);
    cam_verts[2] = point2camspace(tet.verts[2], cam);
    cam_verts[3] = point2camspace(tet.verts[3], cam);

    // Check if any vertex is behind camera
    bool any_behind = (cam_verts[0].z <= cam.min_t ||
                       cam_verts[1].z <= cam.min_t ||
                       cam_verts[2].z <= cam.min_t ||
                       cam_verts[3].z <= cam.min_t);
    if (any_behind) {
        interval_meta[g_idx] = 0;
        return;
    }

    // Project to pixel space
    float2 pix[4];
    float3 pix_w_depth[4];
    pix_w_depth[0] = camspace_to_image(cam_verts[0], cam);
    pix_w_depth[1] = camspace_to_image(cam_verts[1], cam);
    pix_w_depth[2] = camspace_to_image(cam_verts[2], cam);
    pix_w_depth[3] = camspace_to_image(cam_verts[3], cam);

    pix[0] = pix_w_depth[0].xy;
    pix[1] = pix_w_depth[1].xy;
    pix[2] = pix_w_depth[2].xy;
    pix[3] = pix_w_depth[3].xy;

    // Classify silhouette (non-differentiable topology decision)
    no_diff SilhouetteClassification sil = classify_silhouette(pix);

    if (sil.num_sil_verts == 0) {
        interval_meta[g_idx] = 0;
        return;
    }

    // Load vertex normals
    float3[4] vn = load_vert_normals(vtet, vertex_normals);

    // Load cell values (density, base_color, gradient, aux)
    float density = max(float(cell_values[uint2(g_idx, 0)]), 0.f);
    float3 base_color = float3(
        cell_values[uint2(g_idx, 1)],
        cell_values[uint2(g_idx, 2)],
        cell_values[uint2(g_idx, 3)]);
    float3 grd = float3(
        cell_values[uint2(g_idx, 4)],
        cell_values[uint2(g_idx, 5)],
        cell_values[uint2(g_idx, 6)]);

    // Write per-tet data
    interval_tet_data.storeOnce(uint2(g_idx, 0), density);
    interval_tet_data.storeOnce(uint2(g_idx, 1), base_color.x);
    interval_tet_data.storeOnce(uint2(g_idx, 2), base_color.y);
    interval_tet_data.storeOnce(uint2(g_idx, 3), base_color.z);
    interval_tet_data.storeOnce(uint2(g_idx, 4), grd.x);
    interval_tet_data.storeOnce(uint2(g_idx, 5), grd.y);
    interval_tet_data.storeOnce(uint2(g_idx, 6), grd.z);
    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        interval_tet_data.storeOnce(uint2(g_idx, 7 + i), float(cell_values[uint2(g_idx, 7 + i)]));
    }

    // Write meta
    interval_meta[g_idx] = sil.num_sil_verts;

    // --- Write fan vertices ---
    // Vertex 0 = center vertex
    // For Case 1: center is the interior projected vertex.
    //   z_front and z_back are the camera-space depths through the tet.
    // For Case 2: center is the edge crossing point.

    if (sil.center_idx >= 0) {
        // Case 1: Center is an actual tet vertex
        int ci = sil.center_idx;

        // The center vertex sees the full thickness of the tet along the view direction.
        // z_front and z_back: we need to find the depth interval.
        // The center vertex is "inside" the opposite face's projection.
        // Its z is one endpoint; the opposite face's interpolated z is the other.

        // z of center vertex
        float z_center = cam_verts[ci].z;

        // Compute z of the opposite face at the center's pixel position.
        // The opposite face is formed by the 3 silhouette vertices.
        // We need barycentric coords of center's pixel in that triangle.
        uint si0 = sil.sil_indices[0];
        uint si1 = sil.sil_indices[1];
        uint si2 = sil.sil_indices[2];

        float2 sv0 = pix[si0];
        float2 sv1 = pix[si1];
        float2 sv2 = pix[si2];

        // Barycentric coords of center pixel in silhouette triangle
        float2 e0 = sv1 - sv0;
        float2 e1 = sv2 - sv0;
        float2 v = sil.center_pixel - sv0;
        float d00 = dot(e0, e0);
        float d01 = dot(e0, e1);
        float d02 = dot(e0, v);
        float d11 = dot(e1, e1);
        float d12 = dot(e1, v);
        float inv_d = safe_div(1.0f, d00 * d11 - d01 * d01);
        float u = (d11 * d02 - d01 * d12) * inv_d;
        float w = (d00 * d12 - d01 * d02) * inv_d;
        float z_opp = (1.0f - u - w) * cam_verts[si0].z + u * cam_verts[si1].z + w * cam_verts[si2].z;

        float z_front_c = min(z_center, z_opp);
        float z_back_c = max(z_center, z_opp);

        // Normals at front and back: interpolate using tet barycentric coords
        // Front point: closer to camera; back point: further
        // We need 3D points for barycentric interpolation in the tet
        // For the center vertex, the 3D point is known.
        // For the opposite face point, we reconstruct from the opposite face barycentric coords.
        float3 opp_point_cam = (1.0f - u - w) * cam_verts[si0] + u * cam_verts[si1] + w * cam_verts[si2];
        float3 opp_point_world = (1.0f - u - w) * tet.verts[si0] + u * tet.verts[si1] + w * tet.verts[si2];
        float3 center_world = tet.verts[ci];

        float4 bary_center = tet_barycentric(center_world, tet.verts[0], tet.verts[1], tet.verts[2], tet.verts[3]);
        float4 bary_opp = tet_barycentric(opp_point_world, tet.verts[0], tet.verts[1], tet.verts[2], tet.verts[3]);

        float3 n_center = interp_normal(bary_center, vn);
        float3 n_opp = interp_normal(bary_opp, vn);

        float3 n_front_c, n_back_c;
        if (z_center <= z_opp) {
            n_front_c = n_center;
            n_back_c = n_opp;
        } else {
            n_front_c = n_opp;
            n_back_c = n_center;
        }

        // Store center vertex (index 0)
        interval_verts.storeOnce(uint3(g_idx, 0, 0), sil.center_pixel.x);
        interval_verts.storeOnce(uint3(g_idx, 0, 1), sil.center_pixel.y);
        interval_verts.storeOnce(uint3(g_idx, 0, 2), z_front_c);
        interval_verts.storeOnce(uint3(g_idx, 0, 3), z_back_c);
        interval_verts.storeOnce(uint3(g_idx, 0, 4), n_front_c.x);
        interval_verts.storeOnce(uint3(g_idx, 0, 5), n_front_c.y);
        interval_verts.storeOnce(uint3(g_idx, 0, 6), n_front_c.z);
        interval_verts.storeOnce(uint3(g_idx, 0, 7), n_back_c.x);
        interval_verts.storeOnce(uint3(g_idx, 0, 8), n_back_c.y);
        interval_verts.storeOnce(uint3(g_idx, 0, 9), n_back_c.z);

        // Store silhouette vertices (indices 1,2,3)
        // At silhouette edges, z_front == z_back (zero thickness)
        for (int s = 0; s < 3; ++s) {
            uint vi = sil.sil_indices[s];
            float z_sil = cam_verts[vi].z;
            float3 n_sil = interp_normal(tet_barycentric(tet.verts[vi], tet.verts[0], tet.verts[1], tet.verts[2], tet.verts[3]), vn);

            interval_verts.storeOnce(uint3(g_idx, 1 + s, 0), pix[vi].x);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 1), pix[vi].y);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 2), z_sil);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 3), z_sil);  // zero thickness at silhouette
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 4), n_sil.x);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 5), n_sil.y);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 6), n_sil.z);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 7), n_sil.x);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 8), n_sil.y);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 9), n_sil.z);
        }

        // Vertex 4 unused in Case 1 — write zero
        for (int k = 0; k < 10; ++k) {
            interval_verts.storeOnce(uint3(g_idx, 4, k), 0.0f);
        }

    } else {
        // Case 2: Edge crossing. Center is the crossing point.
        // The crossing point has z_front/z_back interpolated from the two crossing edges.

        // Edges that cross
        // We need to find which pair crossed. Re-derive from sil_indices.
        // sil_indices = {e0.x, e1.x, e0.y, e1.y} — the 4 vertices of the two crossing edges
        uint a0 = sil.sil_indices[0]; // first edge start
        uint a1 = sil.sil_indices[2]; // first edge end
        uint b0 = sil.sil_indices[1]; // second edge start
        uint b1 = sil.sil_indices[3]; // second edge end

        // Compute t,s parameters of crossing
        float t_cross, s_cross;
        segments_intersect_2d(pix[a0], pix[a1], pix[b0], pix[b1], t_cross, s_cross);

        // Interpolate z along each edge at the crossing
        float z_edge_a = lerp(cam_verts[a0].z, cam_verts[a1].z, t_cross);
        float z_edge_b = lerp(cam_verts[b0].z, cam_verts[b1].z, s_cross);

        float z_front_c = min(z_edge_a, z_edge_b);
        float z_back_c = max(z_edge_a, z_edge_b);

        // Normals at crossing: interpolate along each edge
        float3 world_a = lerp(tet.verts[a0], tet.verts[a1], t_cross);
        float3 world_b = lerp(tet.verts[b0], tet.verts[b1], s_cross);

        float4 bary_a = tet_barycentric(world_a, tet.verts[0], tet.verts[1], tet.verts[2], tet.verts[3]);
        float4 bary_b = tet_barycentric(world_b, tet.verts[0], tet.verts[1], tet.verts[2], tet.verts[3]);

        float3 n_a = interp_normal(bary_a, vn);
        float3 n_b = interp_normal(bary_b, vn);

        float3 n_front_c, n_back_c;
        if (z_edge_a <= z_edge_b) {
            n_front_c = n_a;
            n_back_c = n_b;
        } else {
            n_front_c = n_b;
            n_back_c = n_a;
        }

        // Store center vertex (index 0)
        interval_verts.storeOnce(uint3(g_idx, 0, 0), sil.crossing_point.x);
        interval_verts.storeOnce(uint3(g_idx, 0, 1), sil.crossing_point.y);
        interval_verts.storeOnce(uint3(g_idx, 0, 2), z_front_c);
        interval_verts.storeOnce(uint3(g_idx, 0, 3), z_back_c);
        interval_verts.storeOnce(uint3(g_idx, 0, 4), n_front_c.x);
        interval_verts.storeOnce(uint3(g_idx, 0, 5), n_front_c.y);
        interval_verts.storeOnce(uint3(g_idx, 0, 6), n_front_c.z);
        interval_verts.storeOnce(uint3(g_idx, 0, 7), n_back_c.x);
        interval_verts.storeOnce(uint3(g_idx, 0, 8), n_back_c.y);
        interval_verts.storeOnce(uint3(g_idx, 0, 9), n_back_c.z);

        // Store 4 silhouette vertices (indices 1..4)
        for (int s = 0; s < 4; ++s) {
            uint vi = sil.sil_indices[s];
            float z_sil = cam_verts[vi].z;
            float3 n_sil = interp_normal(
                tet_barycentric(tet.verts[vi], tet.verts[0], tet.verts[1], tet.verts[2], tet.verts[3]), vn);

            interval_verts.storeOnce(uint3(g_idx, 1 + s, 0), pix[vi].x);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 1), pix[vi].y);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 2), z_sil);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 3), z_sil);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 4), n_sil.x);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 5), n_sil.y);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 6), n_sil.z);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 7), n_sil.x);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 8), n_sil.y);
            interval_verts.storeOnce(uint3(g_idx, 1 + s, 9), n_sil.z);
        }
    }
    */
}
