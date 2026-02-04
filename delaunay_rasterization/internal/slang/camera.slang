import safe_math;
import intersect;

static const float FLT_MAX = 1e20;
static const float FLT_MIN = -1e20;
static const float eps = 1e-7;

struct Ray : IDifferentiable
{
    float3 o, d;
    float ortho;
}

struct TensorCamera {
     int grid_height;
     int grid_width;

     int image_height;
     int image_width;
     int camera_type;
     TensorView<float> world_view_transform;
     TensorView<float> K;
     TensorView<float> cam_pos;
     TensorView<float> distortion_params;
     no_diff float min_t;
     no_diff float fovy;
     no_diff float fovx;
     int tile_height;
     int tile_width;
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
    int grid_height;
    int grid_width;
    int tile_height;
    int tile_width;
    no_diff float min_t;
    int camera_type;
    float4 distortion_params;
}

Camera load_tensor_camera(TensorCamera cam) {
    float4x4 world_view_transform = float4x4(cam.world_view_transform[uint2(0, 0)], cam.world_view_transform[uint2(0, 1)], cam.world_view_transform[uint2(0, 2)], cam.world_view_transform[uint2(0, 3)],
                                             cam.world_view_transform[uint2(1, 0)], cam.world_view_transform[uint2(1, 1)], cam.world_view_transform[uint2(1, 2)], cam.world_view_transform[uint2(1, 3)],
                                             cam.world_view_transform[uint2(2, 0)], cam.world_view_transform[uint2(2, 1)], cam.world_view_transform[uint2(2, 2)], cam.world_view_transform[uint2(2, 3)],
                                             cam.world_view_transform[uint2(3, 0)], cam.world_view_transform[uint2(3, 1)], cam.world_view_transform[uint2(3, 2)], cam.world_view_transform[uint2(3, 3)]);
    float3x3 K = float3x3(cam.K[uint2(0, 0)], cam.K[uint2(0, 1)], cam.K[uint2(0, 2)],
                          cam.K[uint2(1, 0)], cam.K[uint2(1, 1)], cam.K[uint2(1, 2)],
                          cam.K[uint2(2, 0)], cam.K[uint2(2, 1)], cam.K[uint2(2, 2)]);
    float3 position = float3(cam.cam_pos[0], cam.cam_pos[1], cam.cam_pos[2]);
    float4 distortion_params = float4(cam.distortion_params[0], cam.distortion_params[1], cam.distortion_params[2], cam.distortion_params[3]);
    float4x4 proj_mat = float4x4(1.0);
    return { world_view_transform,
             proj_mat,
             K,
             position,
             cam.fovy,
             cam.fovx,
             cam.image_height,
             cam.image_width,
             cam.grid_height,
             cam.grid_width,
             cam.tile_height,
             cam.tile_width,
             cam.min_t,
             cam.camera_type,
             distortion_params };
}

Camera load_camera(TensorView<float> world_view_transform_t, TensorView<float> K_t, TensorView<float> position_t, no_diff float fovy, no_diff float fovx, uint H, uint W) {
    float4x4 world_view_transform = float4x4(world_view_transform_t[uint2(0, 0)], world_view_transform_t[uint2(0, 1)], world_view_transform_t[uint2(0, 2)], world_view_transform_t[uint2(0, 3)],
                                             world_view_transform_t[uint2(1, 0)], world_view_transform_t[uint2(1, 1)], world_view_transform_t[uint2(1, 2)], world_view_transform_t[uint2(1, 3)],
                                             world_view_transform_t[uint2(2, 0)], world_view_transform_t[uint2(2, 1)], world_view_transform_t[uint2(2, 2)], world_view_transform_t[uint2(2, 3)],
                                             world_view_transform_t[uint2(3, 0)], world_view_transform_t[uint2(3, 1)], world_view_transform_t[uint2(3, 2)], world_view_transform_t[uint2(3, 3)]);
    float3x3 K = float3x3(K_t[uint2(0, 0)], K_t[uint2(0, 1)], K_t[uint2(0, 2)],
                          K_t[uint2(1, 0)], K_t[uint2(1, 1)], K_t[uint2(1, 2)],
                          K_t[uint2(2, 0)], K_t[uint2(2, 1)], K_t[uint2(2, 2)]);
    float3 position = float3(position_t[0], position_t[1], position_t[2]);
    float4x4 proj_mat = float4x4(1.0);
    return { world_view_transform, proj_mat, K, position, fovy, fovx, H, W};
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

[Differentiable]
float3 point2camspace(float3 point, Camera cam) {
    float4 cam_space_homo = mul(cam.world_view_transform, float4(point, 1.f));
    float3 cam_space_nohomo = safe_div(cam_space_homo.xyz, cam_space_homo.w);
    return cam_space_nohomo;
}

[Differentiable]
float3 camspace2image_w_depth(float3 cam_space_nohomo, Camera cam) {
    float3 pix_space = mul(cam.K, float3(safe_div(cam_space_nohomo.xy, abs(cam_space_nohomo.z)), 1.f));
    pix_space.z = cam_space_nohomo.z;
    return pix_space;
}

[Differentiable]
float2 point2image(float3 point, Camera cam) {
    float4 cam_space_homo = mul(cam.world_view_transform, float4(point, 1.f));
    float3 cam_space_nohomo = safe_div(cam_space_homo.xyz, cam_space_homo.w);
    float3 pix_space = mul(cam.K, float3(safe_div(cam_space_nohomo.xy, abs(cam_space_nohomo.z) + eps), 1.f));
    return { pix_space.x, pix_space.y };
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
Tetrahedra camspace_tet2image_w_depth(Tetrahedra camspace_tet, Camera cam) {
    return {
        {
            camspace2image_w_depth(camspace_tet.verts[0], cam),
            camspace2image_w_depth(camspace_tet.verts[1], cam),
            camspace2image_w_depth(camspace_tet.verts[2], cam),
            camspace2image_w_depth(camspace_tet.verts[3], cam)
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


[Differentiable]
float3x3 construct_view_matrix(float4x4 viewmatrix)
{
    return float3x3(
        viewmatrix._11, viewmatrix._21, viewmatrix._31,
        viewmatrix._12, viewmatrix._22, viewmatrix._32,
        viewmatrix._13, viewmatrix._23, viewmatrix._33
    );
}

// // [Differentiable]
// Ray get_ray(Camera cam, float2 center_pix_coord) {
//     float3x3 V = construct_view_matrix(cam.world_view_transform);
//     Ray ray;
//     // Construct pixel ray in camera space
//     float fx = cam.K[0][0];
//     float fy = cam.K[1][1];
//     ray.d = normalize(mul(V, float3(
//         (center_pix_coord.x - cam.K[0][2]) / fx,
//         (center_pix_coord.y - cam.K[1][2]) / fy,
//         1.0
//     )));
    
//     ray.o = cam.position + ray.d * cam.min_t;
//     return ray;
// }

[Differentiable]
float depth2distance(Camera cam, float2 center_pix_coord, float depth) {
    float fx = cam.K[0][0];
    float fy = cam.K[1][1];
    return depth * length(float3(
    // ray.d = normalize(float3(
        (center_pix_coord.x - cam.K[0][2]) / fx,
        (center_pix_coord.y - cam.K[1][2]) / fy,
        1.0
    ));
}


// Takes a normalized, distorted point and iteratively finds the original undistorted point.
// Translated from python: _radial_and_tangential_undistort
float2 undistort(float2 p_distorted, Camera cam)
{
    // Optimization: If there are no distortion params, return immediately.
    if (dot(cam.distortion_params, cam.distortion_params) < 1e-9f) {
        return p_distorted;
    }

    float k1 = cam.distortion_params.x;
    float k2 = cam.distortion_params.y;
    float p1 = cam.distortion_params.z;
    float p2 = cam.distortion_params.w;

    // Initialize the undistorted point guess with the distorted point.
    float2 p_undistorted = p_distorted;

    // Iteratively refine the guess using Newton-Raphson method.
    [loop] [MaxIters(10)]
    for (int i = 0; i < 10; ++i)
    {
        float x = p_undistorted.x;
        float y = p_undistorted.y;

        // --- Start of _compute_residual_and_jacobian translation ---
        float r2 = x * x + y * y;
        float d = 1.0 + r2 * (k1 + r2 * k2);

        // Residuals
        float fx = d * x + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) - p_distorted.x;
        float fy = d * y + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y) - p_distorted.y;

        // Jacobian
        float d_r = k1 + r2 * (2.0 * k2);
        float d_x = 2.0 * x * d_r;
        float d_y = 2.0 * y * d_r;

        float fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x;
        float fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y;
        float fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x;
        float fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y;
        // --- End of _compute_residual_and_jacobian translation ---

        // Solve 2x2 linear system J * step = -F using Cramer's rule to find the step.
        float denominator = fy_x * fx_y - fx_x * fy_y;
        if (abs(denominator) < 1e-9) {
            break;
        }
        
        float x_numerator = fx * fy_y - fy * fx_y;
        float y_numerator = fy * fx_x - fx * fy_x;
        
        float2 step = float2(x_numerator / denominator, y_numerator / denominator);
        p_undistorted += step;
    }

    return p_undistorted;
}

Ray get_ray(Camera cam, float2 center_pix_coord)
{
    // Step 1: Convert pixel coordinates to normalized distorted coordinates (inverse of K)
    float fx = cam.K[0][0];
    float fy = cam.K[1][1];
    float cx = cam.K[0][2];
    float cy = cam.K[1][2];

    float2 p_distorted;
    p_distorted.x = (center_pix_coord.x - cx) / fx;
    p_distorted.y = (center_pix_coord.y - cy) / fy;

    // Step 2: Undistort the normalized point
    float2 p_undistorted = p_distorted;
    if (cam.camera_type == 0 || cam.camera_type == 1) { // PERSPECTIVE or FISHEYE
        p_undistorted = undistort(p_distorted, cam);
    }

    // Step 3: Form the camera-space ray direction based on camera type
    float3 dir_cam;
    if (cam.camera_type == 1) { // FISHEYE
        // Translated from python: pixels_to_rays for FISHEYE
        float theta = length(p_undistorted);
        theta = min(3.14159265, theta); // Clamp theta to pi
        
        // Handle case where theta is very small to avoid division by zero
        float sin_theta_over_theta = (theta < 1e-6) ? 1.0 : sin(theta) / theta;

        dir_cam = float3(
            p_undistorted.x * sin_theta_over_theta,
            p_undistorted.y * sin_theta_over_theta,
            cos(theta)
        );
    } else { // PERSPECTIVE (camera_type == 0) or default
        dir_cam = normalize(float3(p_undistorted.x, p_undistorted.y, 1.0));
    }

    // Step 4: Transform ray direction from camera space to world space
    // We need the rotation part of the world_view_transform's inverse.
    // Assuming world_view_transform is [R|t], its inverse is [R'|-R't]. We only need R'.
    // A 3x3 view matrix from a 4x4 transform is typically the top-left 3x3, transposed.
    float3x3 V_rot = construct_view_matrix(cam.world_view_transform);
    
    float3 raw_ray_d = mul(V_rot, dir_cam);
    Ray ray;
    ray.d = normalize(raw_ray_d);
    ray.o = cam.position + ray.d * cam.min_t; // Start ray slightly in front
    ray.ortho = length(raw_ray_d);
    
    return ray;
}

// Camera model and ray generation functions.
// The distortion/undistortion logic is a C++ shader translation of the
// iterative solver found in Google's Nerfies project:
// https://github.com/google/nerfies/blob/main/nerfies/camera.py
//
// The overall structure for handling multiple camera types (perspective, fisheye)
// is adapted from patterns common in modern NeRF codebases like Zip-NeRF.

// Takes a normalized, undistorted point in camera space (where z=1) and applies distortion.
float2 distort(float2 p_undistorted, Camera cam)
{
    // Optimization: If there are no distortion params, return immediately.
    if (dot(cam.distortion_params, cam.distortion_params) < 1e-9f) {
        return p_undistorted;
    }
    
    float k1 = cam.distortion_params.x;
    float k2 = cam.distortion_params.y;
    float p1 = cam.distortion_params.z;
    float p2 = cam.distortion_params.w;

    float r2 = dot(p_undistorted, p_undistorted);
    float radial_dist = r2 * (k1 + r2 * k2);

    float2 tangential_dist;
    tangential_dist.x = 2.0 * p1 * p_undistorted.x * p_undistorted.y + p2 * (r2 + 2.0 * p_undistorted.x * p_undistorted.x);
    tangential_dist.y = 2.0 * p2 * p_undistorted.x * p_undistorted.y + p1 * (r2 + 2.0 * p_undistorted.y * p_undistorted.y);

    return p_undistorted + p_undistorted * radial_dist + tangential_dist;
}

// Projects a 3D point from camera space to a 2D pixel coordinate (with depth).
// This is the inverse of get_ray.
// MODIFIED: This function now correctly handles fisheye projection.
float3 camspace_to_image(float3 p_cam, Camera cam)
{
    float2 p_undistorted;

    if (cam.camera_type == 1) { // FISHEYE
        // This is the inverse of the fisheye ray generation.
        // It maps a 3D direction to a 2D point on the normalized sensor plane.
        float3 dir = normalize(p_cam);
        // acos is sensitive to values slightly > 1.0 due to floating point error
        float theta = acos(saturate(dir.z)); 
        float sin_theta = sin(theta);
        
        // Handle the case where the direction is straight ahead to avoid division by zero.
        float theta_over_sin_theta = (sin_theta < 1e-6f) ? 1.0f : theta / sin_theta;

        p_undistorted = dir.xy * theta_over_sin_theta;

    } else { // PERSPECTIVE (camera_type == 0) or default
        // Standard perspective division
        p_undistorted = p_cam.xy / p_cam.z;
    }

    // Apply radial and tangential distortion (common to both models)
    float2 p_distorted = distort(p_undistorted, cam);

    // Apply intrinsics matrix K to get pixel coordinates
    float px = p_distorted.x * cam.K[0][0] + cam.K[0][2];
    float py = p_distorted.y * cam.K[1][1] + cam.K[1][2];

    return float3(px, py, p_cam.z);
}


// Helper function to sample an edge and update the bounding box.
void update_bounds_from_edge(float3 v_start, float3 v_end, inout float3 minv, inout float3 maxv, Camera cam)
{
    int n_samples = 2; // Default for standard perspective
    bool is_distorted = dot(cam.distortion_params, cam.distortion_params) > 1e-9f;

    if (cam.camera_type == 1) { // FISHEYE
        n_samples = 4;
    } else if (is_distorted) { // DISTORTED PERSPECTIVE
        n_samples = 3;
    }

    // Loop from 0 to n_samples-1. The endpoints are always included.
    // For n_samples=2, this loop runs for i=0 and i=1.
    [loop]
    for (int i = 0; i < n_samples; ++i) {
        float t = (float)i / (float)(n_samples - 1);
        float3 v_sample = lerp(v_start, v_end, t);
        
        float3 pixel_vert = camspace_to_image(v_sample, cam);

        minv = min(minv, pixel_vert);
        maxv = max(maxv, pixel_vert);
    }
}

// [Differentiable]
void find_extent(Tetrahedra tet, out float3 minv, out float3 maxv, Camera cam) 
{
    minv = float3(FLT_MAX, FLT_MAX, FLT_MAX);
    maxv = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    // CHANGE 1: Use the camera's actual near plane. 
    // If your Camera struct doesn't have min_t, pass it in as an argument.
    // (Based on your previous code, 'cam.min_t' should exist).
    float eps = cam.min_t; 

    // Safety: Ensure we don't divide by zero if min_t is accidentally 0
    if (eps < 1e-4) eps = 1e-4;

    int faces[4][3] = {
        {0, 1, 2},
        {0, 2, 3},
        {0, 3, 1},
        {1, 3, 2}
    };

    [loop]
    for (int i = 0; i < 4; ++i)
    {
        float3 face_verts[3] = {
            tet.verts[faces[i][0]],
            tet.verts[faces[i][1]],
            tet.verts[faces[i][2]]
        };

        float3 final_verts[4]; 
        int final_vert_count = 0;

        [unroll]
        for (int j = 0; j < 3; ++j) {
            float3 v_a = face_verts[j];
            float3 v_b = face_verts[(j + 1) % 3];

            // CHANGE 2: Compare against the actual near plane
            bool a_in_front = v_a.z > eps;
            bool b_in_front = v_b.z > eps;

            if (a_in_front) {
                bool found = false;
                for(int k=0; k < final_vert_count; ++k) if(all(final_verts[k] == v_a)) found = true;
                if(!found) final_verts[final_vert_count++] = v_a;
            }

            if (a_in_front != b_in_front) {
                // CHANGE 3: The intersection point is now at z = min_t (0.1)
                // This guarantees the projected coordinate (x/z) is finite and reasonable.
                float t = (eps - v_a.z) / (v_b.z - v_a.z);
                float3 v_clip = lerp(v_a, v_b, t);
                final_verts[final_vert_count++] = v_clip;
            }
        }

        if (final_vert_count > 1) {
            [loop]
            for (int j = 0; j < final_vert_count; ++j) {
                float3 v_start = final_verts[j];
                float3 v_end = final_verts[(j + 1) % final_vert_count];
                update_bounds_from_edge(v_start, v_end, minv, maxv, cam);
            }
        }
    }
}
