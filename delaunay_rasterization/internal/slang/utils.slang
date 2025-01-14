// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import spherical_harmonics;
import intersect;
import safe_math;

static const float FLT_MAX = 1e20;
static const float FLT_MIN = -1e20;
static const float eps = 1e-7;

struct Ray : IDifferentiable
{
    float3 o, d;
}

struct Camera : IDifferentiable
{
    float4x4 world_view_transform;
    float4x4 proj_mat;
    float3x3 K;
    float3 position;
    float fovy;
    float fovx;
    int H;
    int W;
}

struct SplatTetrahedra : IDifferentiable
{
    Tetrahedra tet;
    float4 rgbs;
};

struct Gaussian_3D : IDifferentiable
{
    float3 xyz_ws;
    SpherHarmCoeffs sh_coeffs;
    float4 rotations;
    float3 scales;
};

struct Splat_2D_Vertex : IDifferentiable
{
    float3 xyz_vs;
    float3 rgb;
    float2x2 cov_vs;
};

struct Splat_2D_AlphaBlend : IDifferentiable
{
    float3 xyz_vs;
    float3 rgb;
    float opacity;
    float2x2 inv_cov_vs;
};

[Differentiable]
float read_t1_float(uint32_t idx, DiffTensorView t1)
{
    return t1[uint2(idx, 0)];
}

[Differentiable]
float3 read_t3_float3(uint32_t idx, DiffTensorView t3)
{
    return float3(t3[uint2(idx, 0)],
                  t3[uint2(idx, 1)],
                  t3[uint2(idx, 2)]);
}

[Differentiable]
float4 read_t4_float4(uint32_t idx, DiffTensorView t4)
{
    return float4(t4[uint2(idx, 0)],
                  t4[uint2(idx, 1)],
                  t4[uint2(idx, 2)],
                  t4[uint2(idx, 3)]);
}

[Differentiable]
float4 read_t2x2_float4(uint32_t idx, DiffTensorView t22)
{
    return float4(t22[uint3(idx, 0, 0)],
                  t22[uint3(idx, 1, 0)],
                  t22[uint3(idx, 0, 1)],
                  t22[uint3(idx, 1, 1)]);
}

[Differentiable]
float2x2 read_t2x2_float2x2(uint32_t idx, DiffTensorView t2x2)
{
    return float2x2(t2x2[uint3(idx, 0, 0)],
                    t2x2[uint3(idx, 1, 0)],
                    t2x2[uint3(idx, 0, 1)],
                    t2x2[uint3(idx, 1, 1)]);
}

[Differentiable]
float ndc2pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

[Differentiable]
float pix2ndc(float v, int S)
{
    return (2.0 * v + 1.0) / S - 1.0;
}

float clip(float val, float min_val, float max_val)
{
    return max(min_val, min(max_val, val));
}


Camera load_camera(TensorView<float> world_view_transform_t, TensorView<float> K_t, TensorView<float> position_t, no_diff float fovy, no_diff float fovx, uint H, uint W) {
    float4x4 world_view_transform = float4x4(world_view_transform_t[uint2(0, 0)], world_view_transform_t[uint2(0, 1)], world_view_transform_t[uint2(0, 2)], world_view_transform_t[uint2(0, 3)],
                                             world_view_transform_t[uint2(1, 0)], world_view_transform_t[uint2(1, 1)], world_view_transform_t[uint2(1, 2)], world_view_transform_t[uint2(1, 3)],
                                             world_view_transform_t[uint2(2, 0)], world_view_transform_t[uint2(2, 1)], world_view_transform_t[uint2(2, 2)], world_view_transform_t[uint2(2, 3)],
                                             world_view_transform_t[uint2(3, 0)], world_view_transform_t[uint2(3, 1)], world_view_transform_t[uint2(3, 2)], world_view_transform_t[uint2(3, 3)]);

    // float4x4 proj_mat = float4x4(proj_mat_t[uint2(0, 0)], proj_mat_t[uint2(0, 1)], proj_mat_t[uint2(0, 2)], proj_mat_t[uint2(0, 3)],
    //                              proj_mat_t[uint2(1, 0)], proj_mat_t[uint2(1, 1)], proj_mat_t[uint2(1, 2)], proj_mat_t[uint2(1, 3)],
    //                              proj_mat_t[uint2(2, 0)], proj_mat_t[uint2(2, 1)], proj_mat_t[uint2(2, 2)], proj_mat_t[uint2(2, 3)],
    //                              proj_mat_t[uint2(3, 0)], proj_mat_t[uint2(3, 1)], proj_mat_t[uint2(3, 2)], proj_mat_t[uint2(3, 3)]);
    float3x3 K = float3x3(K_t[uint2(0, 0)], K_t[uint2(0, 1)], K_t[uint2(0, 2)],
                          K_t[uint2(1, 0)], K_t[uint2(1, 1)], K_t[uint2(1, 2)],
                          K_t[uint2(2, 0)], K_t[uint2(2, 1)], K_t[uint2(2, 2)]);
    float3 position = float3(position_t[0], position_t[1], position_t[2]);
    float4x4 proj_mat = float4x4(1.0);

    return { world_view_transform, proj_mat, K, position, fovy, fovx, H, W};
}

[Differentiable]
float3 geom_transform_points(float3 point, float4x4 transf_matrix)
{
    float4 p_out = mul(transf_matrix, float4(point, 1.0));
    return p_out.xyz / (p_out.w + eps);
}

[Differentiable]
float3 geom_transform_points2(float3 point, float4x4 transf_matrix)
{
    float4 p_out = mul(transf_matrix, float4(point, 1.0));
    return p_out.xyz;
}

// [Differentiable]
// float3 project_point_w_depth(float3 point, Camera cam) {
//     float3 proj_point = geom_transform_points(point, mul(cam.proj_mat, cam.world_view_transform));
//     float3 view_point = geom_transform_points2(point, cam.world_view_transform);
//     proj_point.z = view_point.z;
//     return proj_point;
// }

[Differentiable]
float3 point2camspace(float3 point, Camera cam) {
    float4 cam_space_homo = mul(cam.world_view_transform, float4(point, 1.f));
    float3 cam_space_nohomo = safe_div(cam_space_homo.xyz, cam_space_homo.w);
    return cam_space_nohomo;
}

float3 camspace2image_w_depth(float3 cam_space_nohomo, Camera cam) {
    float3 pix_space = mul(cam.K, float3(safe_div(cam_space_nohomo.xy, abs(cam_space_nohomo.z) + eps), 1.f));
    pix_space.z = cam_space_nohomo.z;
    return pix_space;
}

[Differentiable]
float3 project_point_w_depth(float3 point, Camera cam) {
    float4 cam_space_homo = mul(cam.world_view_transform, float4(point, 1.f));
    float3 cam_space_nohomo = safe_div(cam_space_homo.xyz, cam_space_homo.w);
    float3 pix_space = mul(cam.K, float3(safe_div(cam_space_homo.xy, abs(cam_space_homo.z)), 1.f));


    pix_space.z = cam_space_nohomo.z;
    return pix_space;
}

[Differentiable]
float3x3 compute_jacobian(float3 xyz_ws, Camera cam) {
    float tan_half_fovx = tan(cam.fovx / 2.0);
    float tan_half_fovy = tan(cam.fovy / 2.0);
    float h_x = cam.W / (2.0 * tan_half_fovx);
    float h_y = cam.H / (2.0 * tan_half_fovy);

    float3 t = geom_transform_points(xyz_ws, cam.world_view_transform);

    // TODO: Clean this up, quite unintuive math and code
    const float limx = 1.3f * tan_half_fovx;
    const float limy = 1.3f * tan_half_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    float3x3 J = float3x3(h_x / t.z, 0.0, -(h_x * t.x) / (t.z * t.z),
                          0.0, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
                          0.0, 0.0, 0.0);

    return J;
}

[Differentiable]
float2x2 covariance_3d_to_2d(Camera cam, float3 xyz_ws, float3x3 cov_ws) {
    float3x3 R = (float3x3)cam.world_view_transform;
    float3x3 J = compute_jacobian(xyz_ws, cam);
    float3x3 cov_vs = mul(J, mul(R, mul(cov_ws, mul(transpose(R), transpose(J)))));
    cov_vs[0][0] += 0.3;
    cov_vs[1][1] += 0.3;

    return float2x2(cov_vs);
}

[Differentiable]
uint4 load_virtual_tetrahedra(int32_t g_idx,
                              TensorView<int32_t> indices)
{
    return { 
        indices[uint2(g_idx, 0)],
        indices[uint2(g_idx, 1)],
        indices[uint2(g_idx, 2)],
        indices[uint2(g_idx, 3)]
    };
}

[Differentiable]
Tetrahedra load_tetrahedra(DiffTensorView vertices,
                           uint4 virtual_tetra)
{
    float3[4] tetra = {
        read_t3_float3(virtual_tetra.x, vertices),
        read_t3_float3(virtual_tetra.y, vertices),
        read_t3_float3(virtual_tetra.z, vertices),
        read_t3_float3(virtual_tetra.w, vertices),
    };
    return { tetra };
}

[Differentiable]
float compute_det(float2x2 M) {
    return M[0][0] * M[1][1] - M[0][1] * M[1][0];
}

[Differentiable]
Tetrahedra project_tetrahedra(Tetrahedra tet, Camera cam) {
    return {
        {
            project_point_w_depth(tet.verts[0], cam),
            project_point_w_depth(tet.verts[1], cam),
            project_point_w_depth(tet.verts[2], cam),
            project_point_w_depth(tet.verts[3], cam)
        }
    };
}

[Differentiable]
Tetrahedra tet2camspace(Tetrahedra tet, Camera cam) {
    return {
        {
            point2camspace(tet.verts[0], cam),
            point2camspace(tet.verts[1], cam),
            point2camspace(tet.verts[2], cam),
            point2camspace(tet.verts[3], cam)
        }
    };
}

[Differentiable]
Tetrahedra tet_ndc2pix(Tetrahedra tet, int image_height, int image_width) {
    return {{
        {
            ndc2pix(tet.verts[0].x, image_width),
            ndc2pix(tet.verts[0].y, image_height),
            tet.verts[0].z
        },
        {
            ndc2pix(tet.verts[1].x, image_width),
            ndc2pix(tet.verts[1].y, image_height),
            tet.verts[1].z
        },
        {
            ndc2pix(tet.verts[2].x, image_width),
            ndc2pix(tet.verts[2].y, image_height),
            tet.verts[2].z
        },
        {
            ndc2pix(tet.verts[3].x, image_width),
            ndc2pix(tet.verts[3].y, image_height),
            tet.verts[3].z
        }
    }};
}

// [Differentiable]
// void minmax_tet(Tetrahedra tet, out float3 minv, out float3 maxv) {
//     minv = {999999, 999999, 999999};
//     maxv = {-999999, -999999, -999999};
//     for (int i=0; i<4; i++) {
//         minv.x = min(tet.verts[i].x, minv.x);
//         maxv.x = max(tet.verts[i].x, maxv.x);
//         minv.y = min(tet.verts[i].y, minv.y);
//         maxv.y = max(tet.verts[i].y, maxv.y);
//         minv.z = min(tet.verts[i].z, minv.z);
//         maxv.z = max(tet.verts[i].z, maxv.z);
//     }
// }

// [Differentiable]
void find_extent(Tetrahedra tet, out float3 minv, out float3 maxv, Camera cam) {
    minv = float3(FLT_MAX, FLT_MAX, FLT_MAX);
    maxv = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = 0; i < 4; i++) {
        if (tet.verts[i].z < 0) continue;
        float3 pixel_vert = camspace2image_w_depth(tet.verts[i], cam);
        minv = min(minv, pixel_vert);
        maxv = max(maxv, pixel_vert);
    }

    // Adjust bounds if tetrahedron straddles z=0
    for (int i = 0; i < 4; i++) {
        [MaxIters(4)]
        for (int j = i + 1; j < 4; j++) {
            float3 vb = tet.verts[i];
            float3 vf = tet.verts[j];
            float3 temp;
            if (vb.z > vf.z) {
                // swap(vb, vf);
                temp = vf;
                vf = vb;
                vb = temp;
            }
            // vb = v back, vf = v front
            if (vf.z > 0 && vb.z < 0) {

                float t = (1e-3 - vb.z) / (vf.z - vb.z);
                float3 v_clip = vb + t * (vf - vb);
                float3 pixel_v_clip = camspace2image_w_depth(v_clip, cam);
                // printf("pixel_v_clip: %f, %f, %f\n", v_clip.x, v_clip.y, v_clip.z);

                // Adjust bounds to include the intersection point
                minv = min(minv, pixel_v_clip);
                maxv = max(maxv, pixel_v_clip);
            }
        }
    }
}

[Differentiable]
float3 SolveLinearSystem(float3x3 A, float3 b)
{
    // Calculate the determinant of A
    float det = det_3x3(A._11_21_31, A._12_22_32, A._13_23_33);
    
    // Check if the system has a unique solution
    if (abs(det) < 1e-20)
    {
        return float3(0, 0, 0); // Return zero vector if no unique solution
    }

    float a11 = A[0][0];
    float a12 = A[0][1];
    float a13 = A[0][2];
    float a21 = A[1][0];
    float a22 = A[1][1];
    float a23 = A[1][2];
    float a31 = A[2][0];
    float a32 = A[2][1];
    float a33 = A[2][2];
    
    // Calculate the inverse of A
    return {
        (a22 * a33 - a23 * a32) / det * b.x + -(a12 * a33 - a13 * a32) / det * b.y + (a12 * a23 - a13 * a22) / det * b.z,
        -(a21 * a33 - a23 * a31) / det * b.x + (a11 * a33 - a13 * a31) / det * b.y + -(a11 * a23 - a13 * a21) / det * b.z,
        (a21 * a32 - a22 * a31) / det * b.x + -(a11 * a32 - a12 * a31) / det * b.y + (a11 * a22 - a12 * a21) / det * b.z
    };
    /*

    float3x3 inv_A = 
    {
        (a22 * A._33 - A._23 * A._32) / det, -(A._12 * A._33 - A._13 * A._32) / det, (A._12 * A._23 - A._13 * a22) / det,
        -(A._21 * A._33 - A._23 * A._31) / det, (A._11 * A._33 - A._13 * A._31) / det, -(A._11 * A._23 - A._13 * A._21) / det,
        (A._21 * A._32 - a22 * A._31) / det, -(A._11 * A._32 - A._12 * A._31) / det, (A._11 * a22 - A._12 * A._21) / det
    };
    
    // Multiply inv_A with b to get x
    return mul(inv_A, b);
    */
}

// Compute the circumcenter of a tetrahedron
[Differentiable]
float3 calculate_circumcenter(Tetrahedra tet)
{
    // Compute vectors from v0 to other vertices
    float3 a = tet.verts[1] - tet.verts[0];
    float3 b = tet.verts[2] - tet.verts[0];
    float3 c = tet.verts[3] - tet.verts[0];
    
    // Compute squares of lengths
    float aa = dot(a, a);
    float bb = dot(b, b);
    float cc = dot(c, c);
    
    // Compute cross products
    float3 cross_bc = cross(b, c);
    float3 cross_ca = cross(c, a);
    float3 cross_ab = cross(a, b);
    
    // Compute denominator
    float denominator = 2.0f * dot(a, cross_bc);
    
    // Guard against division by zero
    /*
    if (abs(denominator) < 1e-6f)
    {
        // Return center of mass as fallback
        return tet.verts[0] + (a + b + c) / 4.0f;
    }
    
    // Compute circumcenter relative to verts[0]
    float3 relative_circumcenter = (
        aa * cross_bc +
        bb * cross_ca +
        cc * cross_ab
    ) / denominator;
    */
    float3 relative_circumcenter = safe_div(
        aa * cross_bc +
        bb * cross_ca +
        cc * cross_ab
    , denominator);
    
    // Return absolute position
    return tet.verts[0] + relative_circumcenter;
}

[Differentiable]
SplatTetrahedra load_tet_alphablend(int32_t g_idx,
                                    DiffTensorView vertices,
                                    TensorView<int32_t> indices,
                                    DiffTensorView rgbs)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet = load_tetrahedra(vertices, virtual_tet);
    return {
        tet,
        read_t4_float4(g_idx, rgbs)
    };
}

[Differentiable]
float4 evaluate_tetra(SplatTetrahedra g, 
                      float3 ray_o,
                      float3 ray_d)
{
    float dist = ray_tetrahedron_intersect(ray_o, ray_d, g.tet);
    if (dist <= 0) {
        return {0.f, 0.f, 0.f, 0.f};
    } else {
        float alpha = safe_clip(1-safe_exp(-g.rgbs.w * dist), 0, 1);
        return {
            alpha * g.rgbs.x, alpha * g.rgbs.y, alpha * g.rgbs.z, alpha};
    }
}

[Differentiable]
float3x3 get_covariance_from_quat_scales(float4 q, float3 s) {
    float r = q[0], x = q[1], y = q[2], z = q[3];

    float3x3 rotation_matrix = float3x3(
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y));

    float3x3 scales_matrix = float3x3(s[0], 0, 0,
                                      0, s[1], 0,
                                      0, 0, s[2]);

    float3x3 L = mul(rotation_matrix, scales_matrix);

    return mul(L, transpose(L));
}

[Differentiable]
float3x3 construct_view_matrix(float4x4 viewmatrix)
{
    return float3x3(
        viewmatrix._11, viewmatrix._21, viewmatrix._31,
        viewmatrix._12, viewmatrix._22, viewmatrix._32,
        viewmatrix._13, viewmatrix._23, viewmatrix._33
    );
}

[Differentiable]
Ray get_ray(Camera cam, float2 center_pix_coord) {
    float3x3 V = construct_view_matrix(cam.world_view_transform);
    Ray ray;
    // Construct pixel ray in camera space
    float fx = cam.K[0][0];
    float fy = cam.K[1][1];
    ray.d = mul(V, normalize(float3(
    // ray.d = normalize(float3(
        (center_pix_coord.x + 0.5 - cam.K[0][2]) / fx,
        (center_pix_coord.y + 0.5 - cam.K[1][2]) / fy,
        1.0
    )));
    
    ray.o = cam.position;
    return ray;
}