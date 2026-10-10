// Interval generation compute prepass.
// 1 thread per visible tet: projects 4 vertices to pixel space,
// classifies the silhouette, and writes fan vertex data.
//
// Each fan vertex carries the depth interval (z_front, z_back) plus the
// interpolated normal and color at the front/back surface points, so the
// fragment shader can reproduce the interp renderer's spatially-varying
// color c(p) = dot(grd, p - v0) + base_color without ray-tet intersection.

import utils;
import safe_math;
import camera;
import intersect;
import interp_version;
import interval_version;

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

            // Silhouette: 4 vertices arranged around the crossing,
            // alternating between the two crossing edges so consecutive
            // fan vertices are adjacent around the quad.
            result.sil_indices[0] = e0.x;
            result.sil_indices[1] = e1.x;
            result.sil_indices[2] = e0.y;
            result.sil_indices[3] = e1.y;
            return result;
        }
    }

    // Degenerate: no interior vertex and no crossing edges
    return result;
}

// Store one fan vertex (16 floats — see IntervalVertex layout).
[Differentiable]
void store_fan_vertex(
    DiffTensorView interval_verts,
    no_diff uint tet_idx, no_diff uint vert_idx,
    float2 pixel_xy, float z_front, float z_back,
    float3 n_front, float3 n_back,
    float3 c_front, float3 c_back)
{
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 0), pixel_xy.x);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 1), pixel_xy.y);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 2), z_front);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 3), z_back);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 4), n_front.x);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 5), n_front.y);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 6), n_front.z);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 7), n_back.x);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 8), n_back.y);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 9), n_back.z);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 10), c_front.x);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 11), c_front.y);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 12), c_front.z);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 13), c_back.x);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 14), c_back.y);
    interval_verts.storeOnce(uint3(tet_idx, vert_idx, 15), c_back.z);
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void interval_generate(
    TensorView<int32_t> indices,
    DiffTensorView vertices,
    DiffTensorView vertex_normals,
    DiffTensorView cell_values,
    DiffTensorView interval_verts,       // [N, 5, 16]
    DiffTensorView interval_tet_data,    // [N, 1 + AUX_DIM]
    TensorView<int32_t> interval_meta,  // [N]
    TensorCamera tcam)
{
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

    // Per-vertex colors: c(p) = dot(grd, p - v0) + base_color is linear in p,
    // so evaluating it at the tet vertices and interpolating with the same
    // weights as positions is exact.
    float3 vc[4];
    vc[0] = base_color;
    vc[1] = dot(grd, tet.verts[1] - tet.verts[0]) + base_color;
    vc[2] = dot(grd, tet.verts[2] - tet.verts[0]) + base_color;
    vc[3] = dot(grd, tet.verts[3] - tet.verts[0]) + base_color;

    // Write per-tet data: density + aux (color now lives in the fan verts)
    interval_tet_data.storeOnce(uint2(g_idx, 0), density);
    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        interval_tet_data.storeOnce(uint2(g_idx, 1 + i), float(cell_values[uint2(g_idx, 7 + i)]));
    }

    // Write meta
    interval_meta[g_idx] = int32_t(sil.num_sil_verts);

    // --- Write fan vertices ---
    // Vertex 0 = center vertex
    // For Case 1: center is the interior projected vertex.
    //   z_front and z_back are the camera-space depths through the tet.
    // For Case 2: center is the edge crossing point.

    if (sil.center_idx >= 0) {
        // Case 1: Center is an actual tet vertex
        int ci = sil.center_idx;

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
        // (screen space — consistent with the fragment shader's screen-space
        // interpolation of the silhouette surface).
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
        float b0 = 1.0f - u - w;
        float z_opp = b0 * cam_verts[si0].z + u * cam_verts[si1].z + w * cam_verts[si2].z;

        // Normal and color at the center vertex and at the opposite-face point
        float3 n_center = vn[ci];
        float3 c_center = vc[ci];
        float3 n_opp = b0 * vn[si0] + u * vn[si1] + w * vn[si2];
        float3 c_opp = b0 * vc[si0] + u * vc[si1] + w * vc[si2];

        float z_front_c, z_back_c;
        float3 n_front_c, n_back_c, c_front_c, c_back_c;
        if (z_center <= z_opp) {
            z_front_c = z_center; z_back_c = z_opp;
            n_front_c = n_center; n_back_c = n_opp;
            c_front_c = c_center; c_back_c = c_opp;
        } else {
            z_front_c = z_opp; z_back_c = z_center;
            n_front_c = n_opp; n_back_c = n_center;
            c_front_c = c_opp; c_back_c = c_center;
        }

        // Store center vertex (index 0)
        store_fan_vertex(interval_verts, g_idx, 0, sil.center_pixel,
                         z_front_c, z_back_c, n_front_c, n_back_c, c_front_c, c_back_c);

        // Store silhouette vertices (indices 1,2,3)
        // At silhouette edges, z_front == z_back (zero thickness)
        for (int s = 0; s < 3; ++s) {
            uint vi = sil.sil_indices[s];
            float z_sil = cam_verts[vi].z;
            store_fan_vertex(interval_verts, g_idx, 1 + s, pix[vi],
                             z_sil, z_sil, vn[vi], vn[vi], vc[vi], vc[vi]);
        }

        // Vertex 4 unused in Case 1 — write zero
        for (int k = 0; k < 16; ++k) {
            interval_verts.storeOnce(uint3(g_idx, 4, k), 0.0f);
        }

    } else {
        // Case 2: Edge crossing. Center is the crossing point.
        // The crossing point has z_front/z_back interpolated from the two crossing edges.

        // Edges that cross — re-derive from sil_indices.
        // sil_indices = {e0.x, e1.x, e0.y, e1.y} — the 4 vertices of the two crossing edges
        uint a0 = sil.sil_indices[0]; // first edge start
        uint a1 = sil.sil_indices[2]; // first edge end
        uint b0 = sil.sil_indices[1]; // second edge start
        uint b1 = sil.sil_indices[3]; // second edge end

        // Compute t,s parameters of crossing
        float t_cross, s_cross;
        no_diff segments_intersect_2d(pix[a0], pix[a1], pix[b0], pix[b1], t_cross, s_cross);

        // Interpolate z along each edge at the crossing
        float z_edge_a = lerp(cam_verts[a0].z, cam_verts[a1].z, t_cross);
        float z_edge_b = lerp(cam_verts[b0].z, cam_verts[b1].z, s_cross);

        // Normals/colors at crossing: interpolate along each edge
        float3 n_a = lerp(vn[a0], vn[a1], t_cross);
        float3 n_b = lerp(vn[b0], vn[b1], s_cross);
        float3 c_a = lerp(vc[a0], vc[a1], t_cross);
        float3 c_b = lerp(vc[b0], vc[b1], s_cross);

        float z_front_c, z_back_c;
        float3 n_front_c, n_back_c, c_front_c, c_back_c;
        if (z_edge_a <= z_edge_b) {
            z_front_c = z_edge_a; z_back_c = z_edge_b;
            n_front_c = n_a; n_back_c = n_b;
            c_front_c = c_a; c_back_c = c_b;
        } else {
            z_front_c = z_edge_b; z_back_c = z_edge_a;
            n_front_c = n_b; n_back_c = n_a;
            c_front_c = c_b; c_back_c = c_a;
        }

        // Store center vertex (index 0)
        store_fan_vertex(interval_verts, g_idx, 0, sil.crossing_point,
                         z_front_c, z_back_c, n_front_c, n_back_c, c_front_c, c_back_c);

        // Store 4 silhouette vertices (indices 1..4)
        for (int s = 0; s < 4; ++s) {
            uint vi = sil.sil_indices[s];
            float z_sil = cam_verts[vi].z;
            store_fan_vertex(interval_verts, g_idx, 1 + s, pix[vi],
                             z_sil, z_sil, vn[vi], vn[vi], vc[vi], vc[vi]);
        }
    }
}
