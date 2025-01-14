import safe_math;
#define epsilon 1e-6
#define FLT_MAX 1e20
#define FLT_MIN -1e20

// Helper function to compute the determinant of a 3x3 matrix
[Differentiable]
float det_3x3(float3 a, float3 b, float3 c) {
    return dot(a, cross(b, c));
}

bool same_side(float3 v1, float3 v2, float3 v3, float3 v4, float3 p) {
    float3 normal = cross(v2 - v1, v3 - v1);
    float dot_v4 = dot(normal, v4 - v1);
    float dot_p = dot(normal, p - v1);
    return sign(dot_v4) * sign(dot_p) > 0;
}

bool point_in_tetrahedron(Tetrahedra tet, float3 point) {
    return (same_side(tet.verts[0], tet.verts[1], tet.verts[2], tet.verts[3], point) &&
            same_side(tet.verts[1], tet.verts[2], tet.verts[3], tet.verts[0], point) &&
            same_side(tet.verts[2], tet.verts[3], tet.verts[0], tet.verts[1], point) && 
            same_side(tet.verts[3], tet.verts[0], tet.verts[1], tet.verts[2], point));
};

struct Tetrahedra : IDifferentiable
{
    float3[4] verts;
};

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
    out float t)
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
    float U = Cx * By - Cy * Bx;
    float V = Ax * Cy - Ay * Cx;
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
float ray_tetrahedron_intersect(float3 orig, float3 dir, Tetrahedra g) {
    float3 v0 = g.verts[0];
    float3 v1 = g.verts[1];
    float3 v2 = g.verts[2];
    float3 v3 = g.verts[3];
    bool hit = false;
    float t_enter = FLT_MAX;
    float t_exit = -FLT_MAX;
    float3 ndir = normalize(dir);
    float3 off_orig = orig;

    // Check intersection with each face of the tetrahedron
    float t;
    if (ray_triangle_intersect(off_orig, ndir, v0, v1, v2, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    if (ray_triangle_intersect(off_orig, ndir, v0, v1, v3, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    if (ray_triangle_intersect(off_orig, ndir, v0, v2, v3, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    if (ray_triangle_intersect(off_orig, ndir, v1, v2, v3, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    if (point_in_tetrahedron(g, off_orig)) {
        t_enter = 0;
    }

    if (hit) {
        return max(t_exit - max(t_enter, 0), 0);
    } else {
        return -1;
    }
}
