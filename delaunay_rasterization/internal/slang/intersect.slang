import safe_math;
#define epsilon 1e-6
#define FLT_MAX 1e20
#define FLT_MIN -1e20

// Helper function to compute the determinant of a 3x3 matrix
[Differentiable]
float det_3x3(float3 a, float3 b, float3 c) {
    return dot(a, cross(b, c));
}

// bool same_side(float3 v1, float3 v2, float3 v3, float3 v4, float3 p) {
//     float3 normal = cross(v2 - v1, v3 - v1);
//     float dot_v4 = dot(normal, v4 - v1);
//     float dot_p = dot(normal, p - v1);
//     return sign(dot_v4) * sign(dot_p) > 0;
// }

// bool point_in_tetrahedron(Tetrahedra tet, float3 point) {
//     return (same_side(tet.verts[0], tet.verts[1], tet.verts[2], tet.verts[3], point) &&
//             same_side(tet.verts[1], tet.verts[2], tet.verts[3], tet.verts[0], point) &&
//             same_side(tet.verts[2], tet.verts[3], tet.verts[0], tet.verts[1], point) && 
//             same_side(tet.verts[3], tet.verts[0], tet.verts[1], tet.verts[2], point));
// };

inline int sign_with_eps(float x, float eps) {
    if      (x >  eps) return  1;
    else if (x < -eps) return -1;
    else               return  0;
}

bool same_side_eps(
    const float3 v1,
    const float3 v2,
    const float3 v3,
    const float3 v4,
    const float3 p,
    float eps
) {
    float3 normal = cross(v2 - v1, v3 - v1); 
    float   dpV4  = dot(normal, v4 - v1);
    float   dpP   = dot(normal, p  - v1);

    // Compare the "sign" of dpV4 and dpP with a tolerance:
    int sV4 = sign_with_eps(dpV4, eps);
    int sP  = sign_with_eps(dpP,  eps);

    // same_side means they are either the same nonzero sign or at least one is effectively zero
    // So we check sV4*sP >= 0 (which includes both > 0 and == 0):
    return (sV4 * sP) >= 0;
}

bool point_in_tetrahedron(const Tetrahedra tet, const float3 point, float eps=1e-10f)
{
    // Check if the point is extremely close to any vertex

    for (int i = 0; i < 4; ++i) {
        if (length(point - tet.verts[i]) < 1e-6) {
            return true;  // Treat it as inside
        }
    }

    // Same-side tests with relaxed epsilon
    return ( same_side_eps(tet.verts[0], tet.verts[1], tet.verts[2], tet.verts[3], point, eps) &&
             same_side_eps(tet.verts[1], tet.verts[2], tet.verts[3], tet.verts[0], point, eps) &&
             same_side_eps(tet.verts[2], tet.verts[3], tet.verts[0], tet.verts[1], point, eps) &&
             same_side_eps(tet.verts[3], tet.verts[0], tet.verts[1], tet.verts[2], point, eps) );
}


struct Tetrahedra : IDifferentiable
{
    float3[4] verts;
};

bool is_planar(Tetrahedra tetra)
{
    float3 v0 = tetra.verts[0];
    float3 v1 = tetra.verts[1];
    float3 v2 = tetra.verts[2];
    float3 v3 = tetra.verts[3];
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 normal = normalize(cross(edge1, edge2));

    float distance = dot(normal, (v3 - v0));

    return abs(distance) < 1e-3;
}

int max_dim(float3 v) {
    if (v.x > v.y && v.x > v.z) return 0;
    if (v.y > v.z) return 1;
    return 2;
}

void swap(inout int a, inout int b) {
    int temp = a;
    a = b;
    b = temp;
}

[Differentiable]
bool ray_triangle_intersect(
    in float3 ray_origin,
    in float3 ray_dir,
    float3 v0,
    float3 v1,
    float3 v2,
    out float t,
    out float U,
    out float V)
{
    t = -1;
    /* Calculate dimension where the ray direction is maximal */
    int kz = max_dim(abs(ray_dir));
    int kx = kz + 1; 
    if (kx == 3) kx = 0;
    int ky = kx + 1; 
    if (ky == 3) ky = 0;

    /* Swap kx and ky dimensions to preserve winding direction of triangles */
    if (ray_dir[kz] < 0.0f) swap(kx, ky);

    /* Calculate shear constants */
    float Sx = ray_dir[kx] / ray_dir[kz];
    float Sy = ray_dir[ky] / ray_dir[kz];
    float Sz = 1.0f / ray_dir[kz];

    /* Calculate vertices relative to ray origin */
    float3 A = v0 - ray_origin;
    float3 B = v1 - ray_origin;
    float3 C = v2 - ray_origin;

    /* Perform shear and scale of vertices */
    float Ax = A[kx] - Sx * A[kz];
    float Ay = A[ky] - Sy * A[kz];
    float Bx = B[kx] - Sx * B[kz];
    float By = B[ky] - Sy * B[kz];
    float Cx = C[kx] - Sx * C[kz];
    float Cy = C[ky] - Sy * C[kz];

    /* Calculate scaled barycentric coordinates */
    U = Cx * By - Cy * Bx;
    V = Ax * Cy - Ay * Cx;
    float W = Bx * Ay - By * Ax;

    /* Fallback to test against edges using double precision */
    if (U == 0.0f || V == 0.0f || W == 0.0f) {
        double CxBy = (double)Cx * (double)By;
        double CyBx = (double)Cy * (double)Bx;
        U = (float)(CxBy - CyBx);
        
        double AxCy = (double)Ax * (double)Cy;
        double AyCx = (double)Ay * (double)Cx;
        V = (float)(AxCy - AyCx);
        
        double BxAy = (double)Bx * (double)Ay;
        double ByAx = (double)By * (double)Ax;
        W = (float)(BxAy - ByAx);
    }

    /* Perform edge tests */
    #ifdef BACKFACE_CULLING
    if (U < 0.0f || V < 0.0f || W < 0.0f) return false;
    #else
    if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f)) return false;
    #endif

    /* Calculate determinant */
    float det = U + V + W;
    if (det == 0.0f) return false;

    /* Calculate scaled z-coordinates of vertices and use them to calculate the hit distance */
    float Az = Sz * A[kz];
    float Bz = Sz * B[kz];
    float Cz = Sz * C[kz];
    float T = U * Az + V * Bz + W * Cz;

    #ifdef BACKFACE_CULLING
    if (T < 0.0f) return false;
    #else
    int det_sign = sign(det);
    if (asfloat(asuint(T) ^ asuint(det_sign)) < 0.0f) return false;
    #endif

    /* Normalize T */
    t = T / det;
    return true;
}

[Differentiable]
bool ray_triangle_intersect2(in float3 ray_origin, 
                             in float3 ray_vector, 
                             in float3 tri_a,
                             in float3 tri_b,
                             in float3 tri_c,
                             out float t)
{
    t = -1;
    float3 edge1 = tri_b - tri_a;
    float3 edge2 = tri_c - tri_a;
    float3 ray_cross_e2 = cross(ray_vector, edge2);
    float det = dot(edge1, ray_cross_e2);
    float inv_det = safe_div(1.0, det);
    float3 s = ray_origin - tri_a;
    float u = inv_det * dot(s, ray_cross_e2);

    if (u < 0 || u > 1)
        return false;

    float3 s_cross_e1 = cross(s, edge1);
    float v = inv_det * dot(ray_vector, s_cross_e1);

    if (v < 0 || u + v > 1)
        return false;

    // At this stage we can compute t to find out where the intersection point is on the line.
    t = inv_det * dot(edge2, s_cross_e1);

    return true;
}

// Ray-tetrahedron intersection function
[Differentiable]
bool ray_tetrahedron_intersect(float3 orig, float3 dir, Tetrahedra g, float2 dist) {
    float3 v0 = g.verts[0];
    float3 v1 = g.verts[1];
    float3 v2 = g.verts[2];
    float3 v3 = g.verts[3];

    bool hit = false;
    float t_enter = FLT_MAX;
    float t_exit = -FLT_MAX;

    // Check intersection with each face of the tetrahedron
    float t, u, v;
    if (ray_triangle_intersect(orig, dir, v0, v1, v2, t, u, v)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    if (ray_triangle_intersect(orig, dir, v0, v1, v3, t, u, v)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    if (ray_triangle_intersect(orig, dir, v0, v2, v3, t, u, v)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }

    if (ray_triangle_intersect(orig, dir, v1, v2, v3, t, u, v)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }

    if (point_in_tetrahedron(g, orig)) {
        t_enter = 0;
    }

    dist = {t_enter, t_exit};
    return hit;
}

[Differentiable]
bool ray_triangle_intersect_2D(
    float2 pixelPos,                 // The 2D pixel coordinate (screen space)
    float2 t0, float2 t1, float2 t2, // 2D positions of the triangle vertices
    float w0, float w1, float w2,     // 1/distance at each vertex (reciprocal of camera distance)
    out float distance
)
{
    // Edges relative to t0
    float2 e0 = t1 - t0;
    float2 e1 = t2 - t0;
    float2 v = pixelPos - t0;

    // Compute dot products
    float dot00 = dot(e0, e0);
    float dot01 = dot(e0, e1);
    float dot02 = dot(e0, v);
    float dot11 = dot(e1, e1);
    float dot12 = dot(e1, v);

    // Compute barycentric denominator
    float denom = dot00 * dot11 - dot01 * dot01;

    // Compute barycentric coordinates (alpha, beta, gamma)
    float alpha = (dot11 * dot02 - dot01 * dot12) / denom;
    float beta = (dot00 * dot12 - dot01 * dot02) / denom;
    float gamma = 1.0 - alpha - beta;

    // Check if we are inside the triangle
    if (alpha < 0.0f || beta < 0.0f || gamma < 0.0f)
    {
        // Return negative to indicate "outside"
        return false;
    }

    // Interpolate 1/distance
    float reciprocalDist = alpha * w1 + beta * w2 + gamma * w0;

    // Avoid division by zero
    if (reciprocalDist <= 0.0f)
    {
        return false;
    }

    // Final perspective-correct distance
    distance = 1.0f / reciprocalDist;
    return true;
}

[Differentiable]
float2 ray_tetrahedron_intersect_2D(float2 pixelPos, float tmin, Tetrahedra g) {
    float2 v0 = { g.verts[0].x, g.verts[0].y };
    float2 v1 = { g.verts[1].x, g.verts[1].y };
    float2 v2 = { g.verts[2].x, g.verts[2].y };
    float2 v3 = { g.verts[3].x, g.verts[3].y };
    float w0 = g.verts[0].z;
    float w1 = g.verts[1].z;
    float w2 = g.verts[2].z;
    float w3 = g.verts[3].z;

    bool hit = false;
    float t_enter = FLT_MAX;
    float t_exit = -FLT_MAX;

    // Check intersection with each face of the tetrahedron
    float t;
    if (ray_triangle_intersect_2D(pixelPos, v0, v1, v2, w0, w1, w2, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    if (ray_triangle_intersect_2D(pixelPos, v0, v1, v3, w0, w1, w3, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    if (ray_triangle_intersect_2D(pixelPos, v0, v2, v3, w0, w2, w3, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }

    if (ray_triangle_intersect_2D(pixelPos, v1, v2, v3, w1, w2, w3, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }

    if (hit) {
        return {max(t_exit - max(t_enter, tmin), 0), (t_enter+t_exit)/2};
    } else {
        return {-1, 0};
    }
}

[Differentiable]
bool ray_tetrahedron_intersect_fused(
    float3 orig, float3 dir,
    float4x3 verts,
    inout float2 t)
{
    // --------------------------------------------------------------------
    // 1. Compute plane equations for the 4 faces (outward normals)
    //    n[i].x, n[i].y, n[i].z, and d[i]
    // --------------------------------------------------------------------
    static const uint3 kTetTriangles[4] = {
        /* 
        *    y
        *    ^
        *    |
        *    *v2    z
        *    |     /
        *    |    *v3
        *    |   /
        *    |  /
        *    | /          v1
        *    v0------------*--->x
        *                 
        */
        uint3(0, 2, 1),
        uint3(1, 2, 3),
        uint3(0, 3, 2),
        uint3(3, 0, 1),
    };



    float3 v0 = verts[0];
    float3 v1 = verts[1];
    float3 v2 = verts[2];
    float3 v3 = verts[3];    
    // const float3 n0 = cross(
    //     verts[kTetTriangles[0][2]] - verts[kTetTriangles[0][0]],
    //     verts[kTetTriangles[0][1]] - verts[kTetTriangles[0][0]]);
    // const float d0  = -dot(n0, verts[kTetTriangles[0][0]] - orig);
    // const float3 n1 = cross(
    //     verts[kTetTriangles[1][2]] - verts[kTetTriangles[1][0]],
    //     verts[kTetTriangles[1][1]] - verts[kTetTriangles[1][0]]);
    // const float d1  = -dot(n1, verts[kTetTriangles[1][0]] - orig);
    // const float3 n2 = cross(
    //     verts[kTetTriangles[2][2]] - verts[kTetTriangles[2][0]],
    //     verts[kTetTriangles[2][1]] - verts[kTetTriangles[2][0]]);
    // const float d2  = -dot(n2, verts[kTetTriangles[2][0]] - orig);
    // const float3 n3 = cross(
    //     verts[kTetTriangles[3][2]] - verts[kTetTriangles[3][0]],
    //     verts[kTetTriangles[3][1]] - verts[kTetTriangles[3][0]]);
    // const float d3  = -dot(n3, verts[kTetTriangles[3][0]] - orig);


    // Face 0: (v0, v1, v2)
    float3 n0 = cross(v1 - v0, v2 - v0);
    if (dot(n0, v3 - v0) > 0) {
        n0 = -n0;
    }
    float d0 = -dot(n0, v0-orig);

    // Face 1: (v0, v1, v3)
    float3 n1 = cross(v1 - v0, v3 - v0);
    if (dot(n1, v2 - v0) > 0) {
        n1 = -n1;
    }
    float d1 = -dot(n1, v0-orig);

    // Face 2: (v0, v2, v3)
    float3 n2 = cross(v2 - v0, v3 - v0);
    if (dot(n2, v1 - v0) > 0) {
        n2 = -n2;
    }
    float d2 = -dot(n2, v0-orig);

    // Face 3: (v1, v2, v3)
    float3 n3 = cross(v2 - v1, v3 - v1);
    if (dot(n3, v0 - v1) > 0) {
        n3 = -n3;
    }
    float d3 = -dot(n3, v1-orig);

    // --------------------------------------------------------------------
    // 2. Initialize the intersection range [t_enter, t_exit]
    //    You might prefer (-∞, +∞), but this variant starts at [0, big].
    // --------------------------------------------------------------------
    float t_enter = -1e30f;
    float t_exit  = 1e30f;
    float eps     = 1e-10f;

    // --------------------------------------------------------------------
    // 3. Clip against Plane 0
    // --------------------------------------------------------------------
    {
        float dist0  = d0;
        float denom0 = dot(n0, dir);

        if (fabs(denom0) < eps) {
            // Ray is almost parallel to plane
            if (dist0 > 0.0f) {
                // Entire ray is outside
                return false;
            }
        }
        else {
            float tplane = -dist0 / denom0;
            if (denom0 < 0.0f) {
                // Ray entering at this t
                if (tplane > t_enter) {
                    t_enter = tplane;
                }
            }
            else {
                // Ray exiting
                if (tplane < t_exit) {
                    t_exit = tplane;
                }
            }
        }

    }

    // --------------------------------------------------------------------
    // 4. Clip against Plane 1
    // --------------------------------------------------------------------
    {
        float dist1  = d1;
        float denom1 = dot(n1, dir);

        if (fabs(denom1) < eps) {
            if (dist1 > 0.0f) {
                return false;
            }
        }
        else {
            float tplane = -dist1 / denom1;
            if (denom1 < 0.0f) {
                if (tplane > t_enter) {
                    t_enter = tplane;
                }
            }
            else {
                if (tplane < t_exit) {
                    t_exit = tplane;
                }
            }
        }

    }

    // --------------------------------------------------------------------
    // 5. Clip against Plane 2
    // --------------------------------------------------------------------
    {
        float dist2  = d2;
        float denom2 = dot(n2, dir);

        if (fabs(denom2) < eps) {
            if (dist2 > 0.0f) {
                return false;
            }
        }
        else {
            float tplane = -dist2 / denom2;
            if (denom2 < 0.0f) {
                if (tplane > t_enter) {
                    t_enter = tplane;
                }
            }
            else {
                if (tplane < t_exit) {
                    t_exit = tplane;
                }
            }
        }

    }

    // --------------------------------------------------------------------
    // 6. Clip against Plane 3
    // --------------------------------------------------------------------
    {
        float dist3  = d3;
        float denom3 = dot(n3, dir);

        if (fabs(denom3) < eps) {
            if (dist3 > 0.0f) {
                return false;
            }
        }
        else {
            float tplane = -dist3 / denom3;
            if (denom3 < 0.0f) {
                if (tplane > t_enter) {
                    t_enter = tplane;
                }
            }
            else {
                if (tplane < t_exit) {
                    t_exit = tplane;
                }
            }
        }

        if (t_enter > t_exit) {
            return false;
        }
    }

    // --------------------------------------------------------------------
    // 7. Final intersection range
    //    Return the clipped segment length if it's in front of the origin
    // --------------------------------------------------------------------
    if (t_exit <= 0.0f) {
        return false;
    }
    t_enter = max(t_enter, 0.f);
    t = { t_enter, t_exit };
    return true;
}

/*
[Differentiable]
bool ray_tetrahedron_intersect_fused_vectorized(
    float3 orig, float3 dir,
    float4x3 verts,
    inout float2 t)
{
    // These are the specific winding orders that produce inward-facing normals.
    static const uint3 kTetTriangles[4] = {
        uint3(0, 2, 1), uint3(1, 2, 3), uint3(0, 3, 2), uint3(3, 0, 1)
    };

    // 1. Compute the four INWARD-FACING normals in parallel using the fixed indices.
    // The cross product order is chosen to match the known winding.
    // const float4x3 normals = float4x3(
    //     cross(verts[kTetTriangles[0][2]] - verts[0], verts[kTetTriangles[0][1]] - verts[0]), // Face 0: (v1-v0) x (v2-v0)
    //     cross(verts[kTetTriangles[1][2]] - verts[1], verts[kTetTriangles[1][1]] - verts[1]), // Face 1: (v3-v1) x (v2-v1)
    //     cross(verts[kTetTriangles[2][2]] - verts[0], verts[kTetTriangles[2][1]] - verts[0]), // Face 2: (v2-v0) x (v3-v0)
    //     cross(verts[kTetTriangles[3][2]] - verts[3], verts[kTetTriangles[3][1]] - verts[3])  // Face 3: (v1-v3) x (v0-v3)
    // );

    const float3 v0 = verts[0], v1 = verts[1], v2 = verts[2], v3 = verts[3];
    const float4x3 normals = float4x3(
        cross(v1 - v0, v2 - v0),
        cross(v1 - v0, v3 - v0),
        cross(v2 - v0, v3 - v0),
        cross(v2 - v1, v3 - v1)
    );

    // 2. Ensure all normals point outward without branching.
    // We check each normal against the tetrahedron's fourth vertex.
    const float4 dots = float4(
        dot(normals[0], v3 - v0),
        dot(normals[1], v2 - v0),
        dot(normals[2], v1 - v0),
        dot(normals[3], v0 - v1)
    );
    const float4x3 outward_normals = float4x3(
        normals[0] * sign(dots.x),
        normals[1] * sign(dots.y),
        normals[2] * sign(dots.z),
        normals[3] * sign(dots.w),
    );

    // NOTE: The expensive step of checking and flipping normals is now removed.

    // 2. Calculate ray-plane intersection parameters t = N/D.
    // Numerators N = dot(n, v_base - orig)
    const float4 numerators = float4(
        // dot(outward_normals[0], verts[kTetTriangles[0][0]] - orig),
        // dot(outward_normals[1], verts[kTetTriangles[1][0]] - orig),
        // dot(outward_normals[2], verts[kTetTriangles[2][0]] - orig),
        // dot(outward_normals[3], verts[kTetTriangles[3][0]] - orig)
        dot(outward_normals[0], v0 - orig),
        dot(outward_normals[1], v1 - orig),
        dot(outward_normals[2], v2 - orig),
        dot(outward_normals[3], v3 - orig)
    );

    // Denominators D = dot(n, dir)
    const float4 denoms = mul(normals, dir);

    // 3. Handle parallel rays.
    // For an inward normal, the origin is "outside" if dot(n, o-v_base) < 0,
    // which means the numerator is negative.
    const float eps = 1e-10f;
    if (any((abs(denoms) < eps) & (numerators < 0.0f))) {
        return false;
    }

    // 4. Perform the slab test using the INWARD normal convention.
    const float4 t_planes = -numerators / denoms;
    const float t_initial_enter = -1e30f;
    const float t_initial_exit  = 1e30f;

    // With INWARD normals:
    // denom > 0 means the ray is ENTERING.
    // denom < 0 means the ray is EXITING.
    const float4 t_enters = select(denoms < 0.0f, t_planes, float4(t_initial_enter));
    const float4 t_exits  = select(denoms > 0.0f, t_planes, float4(t_initial_exit));

    // 5. Find the final interval [t_enter, t_exit] with horizontal reductions.
    float t_enter = max(max(t_enters.x, t_enters.y), max(t_enters.z, t_enters.w));
    float t_exit  = min(min(t_exits.x, t_exits.y), min(t_exits.z, t_exits.w));

    // 6. Final intersection checks.
    if (t_enter > t_exit || t_exit <= 0.0f) {
        return false;
    }

    t = float2(max(t_enter, 0.f), t_exit);
    return true;
}
*/

[Differentiable]
bool ray_tetrahedron_intersect_fused_vectorized(
    float3 orig, float3 dir,
    float4x3 verts,
    inout float2 t)
{
    static const uint3 kTetTriangles[4] = {
        uint3(0, 2, 1), uint3(1, 2, 3), uint3(0, 3, 2), uint3(3, 0, 1)
    };
    // float4x3 outward_normals;
    // for (int i=0; i<4; i++) {
    //     outward_normals[i] = -cross(
    //         verts[kTetTriangles[i][2]] - verts[kTetTriangles[i][0]],
    //         verts[kTetTriangles[i][1]] - verts[kTetTriangles[i][0]]);
    // }
    const float4x3 outward_normals = float4x3(
        -cross(verts[kTetTriangles[0][2]] - verts[0], verts[kTetTriangles[0][1]] - verts[0]),
        -cross(verts[kTetTriangles[1][2]] - verts[1], verts[kTetTriangles[1][1]] - verts[1]),
        -cross(verts[kTetTriangles[2][2]] - verts[0], verts[kTetTriangles[2][1]] - verts[0]),
        -cross(verts[kTetTriangles[3][2]] - verts[3], verts[kTetTriangles[3][1]] - verts[3])
    );


    // float4x3 outward_normals = {n0, n1, n2, n3};
    const float4 numerators = float4(
        -dot(outward_normals[0], verts[kTetTriangles[0][0]] - orig),
        -dot(outward_normals[1], verts[kTetTriangles[1][0]] - orig),
        -dot(outward_normals[2], verts[kTetTriangles[2][0]] - orig),
        -dot(outward_normals[3], verts[kTetTriangles[3][0]] - orig)
    );

    // float4 numerators = {d0, d1, d2, d3};
    const float4 denoms = mul(outward_normals, dir);

    // --- 4. Handle parallel rays. ---
    // For an OUTWARD normal, the origin is outside if dot(n, o-v_base) > 0.
    // This is equivalent to our numerator > 0.
    const float eps = 1e-10f;
    if (any((abs(denoms) < eps) & (numerators > 0.0f))) {
        return false;
    }

    // --- 5. Perform the slab test using the OUTWARD normal convention. ---
    const float4 t_planes = -numerators / denoms;
    const float t_initial_enter = -1e30f;
    const float t_initial_exit  = 1e30f;

    // FIX: This logic now correctly matches the outward normal convention.
    // We assume a standard ternary/select(condition, if_true, if_false) function.
    // denom < 0 means the ray is ENTERING.
    // denom > 0 means the ray is EXITING.
    const float4 t_enters = select(denoms < 0.0f, t_planes, float4(t_initial_enter));
    const float4 t_exits  = select(denoms > 0.0f, t_planes, float4(t_initial_exit));
    
    // --- 6. Find the final interval with horizontal reductions. ---
    float t_enter = max(max(t_enters.x, t_enters.y), max(t_enters.z, t_enters.w));
    float t_exit  = min(min(t_exits.x, t_exits.y), min(t_exits.z, t_exits.w));

    // --- 7. Final intersection checks, identical to scalar code. ---
    if (t_enter > t_exit || t_exit <= 0.0f) {
        return false;
    }

    t = float2(max(t_enter, 0.f), t_exit);
    return true;
}