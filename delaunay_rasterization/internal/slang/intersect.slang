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
float2 ray_tetrahedron_intersect(float3 orig, float3 dir, Tetrahedra g) {
    float3 v0 = g.verts[0];
    float3 v1 = g.verts[1];
    float3 v2 = g.verts[2];
    float3 v3 = g.verts[3];

    bool hit = false;
    float t_enter = FLT_MAX;
    float t_exit = -FLT_MAX;

    // Check intersection with each face of the tetrahedron
    float t;
    if (ray_triangle_intersect(orig, dir, v0, v1, v2, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    if (ray_triangle_intersect(orig, dir, v0, v1, v3, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    if (ray_triangle_intersect(orig, dir, v0, v2, v3, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }

    if (ray_triangle_intersect(orig, dir, v1, v2, v3, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }

    if (point_in_tetrahedron(g, orig)) {
        t_enter = 0;
    }

    if (hit) {
        return {max(t_exit - max(t_enter, 0), 0), (t_enter+t_exit)/2};
    } else {
        return {-1, 0};
    }
}

[Differentiable]
bool ray_tetrahedron_intersect_fused(
    float3 orig, float3 dir,
    float3 v0, float3 v1, float3 v2, float3 v3,
    inout float2 t)
{
    // --------------------------------------------------------------------
    // 1. Compute plane equations for the 4 faces (outward normals)
    //    n[i].x, n[i].y, n[i].z, and d[i]
    // --------------------------------------------------------------------

    // Face 0: (v0, v1, v2)
    float3 n0 = cross(v1 - v0, v2 - v0);
    if (dot(n0, v3 - v0) > 0) {
        n0 = -n0;
    }
    float d0 = -dot(n0, v0);

    // Face 1: (v0, v1, v3)
    float3 n1 = cross(v1 - v0, v3 - v0);
    if (dot(n1, v2 - v0) > 0) {
        n1 = -n1;
    }
    float d1 = -dot(n1, v0);

    // Face 2: (v0, v2, v3)
    float3 n2 = cross(v2 - v0, v3 - v0);
    if (dot(n2, v1 - v0) > 0) {
        n2 = -n2;
    }
    float d2 = -dot(n2, v0);

    // Face 3: (v1, v2, v3)
    float3 n3 = cross(v2 - v1, v3 - v1);
    if (dot(n3, v0 - v1) > 0) {
        n3 = -n3;
    }
    float d3 = -dot(n3, v1);

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
        float dist0  = dot(n0, orig) + d0;
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
        float dist1  = dot(n1, orig) + d1;
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
        float dist2  = dot(n2, orig) + d2;
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
        float dist3  = dot(n3, orig) + d3;
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

struct PluckerLine : IDifferentiable {
    float3 U, V;
};

[Differentiable]
PluckerLine make_plucker(float3 origin, float3 dir) {
    return { dir, cross(dir, origin) };
};

[Differentiable]
float pdot(PluckerLine p1, PluckerLine p2) {
    return dot(p1.U, p2.V) + dot(p2.U, p1.V);
}

// Compute the intersection t for a ray with a triangle plane.
// (a, b, c) define the triangle (face) and 'opp' is the opposite vertex.
// This function ensures that the face normal is oriented outward.
[Differentiable]
float intersect_face(float3 orig, float3 dir, float3 a, float3 b, float3 c) {
    float3 normal = cross(b - a, c - a);
    let denom = dot(normal, dir);
    let numer = dot(normal, orig) - dot(normal, a);

    let dist = safe_div(numer, denom);
    return abs(dist);
}

/*
[Differentiable]
float2 ray_tetrahedron_intersect_plucker(
    float3 orig, float3 dir,
    float3 v0, float3 v1, float3 v2, float3 v3)
{
    // Precompute the ray's Plücker coordinate.
    PluckerLine rayPlucker = make_plucker(orig, dir);

    // Pack vertices into an array.
    float3 verts[4] = { v0, v1, v2, v3 };

    // Define each face by an array of indices:
    // The first three indices specify the face vertices,
    // and the fourth index specifies the opposite vertex.
    // Face ordering:
    // F0: {3, 2, 1, 0} => face (v3, v2, v1), opposite: v0.
    // F1: {2, 3, 0, 1} => face (v2, v3, v0), opposite: v1.
    // F2: {1, 0, 3, 2} => face (v1, v0, v3), opposite: v2.
    // F3: {0, 1, 2, 3} => face (v0, v1, v2), opposite: v3.
    int faceIndices[4][4] = {
        { 3, 2, 1, 0 },
        { 2, 3, 0, 1 },
        { 1, 0, 3, 2 },
        { 0, 1, 2, 3 }
    };

    // Initialize face selection and intersection parameters.
    int Fenter = -1;
    int Fleave = -1;
    float t_enter = -1e30f;
    float t_exit = 1e30f;

    // Loop over faces in descending order: i = 3, 2, 1, 0.
    for (int i = 3; i >= 0; i--)
    {
        // Retrieve indices for this face.
        int ia = faceIndices[i][0];
        int ib = faceIndices[i][1];
        int ic = faceIndices[i][2];
        int iopp = faceIndices[i][3];

        // Fetch the vertices.
        float3 a = verts[ia];
        float3 b = verts[ib];
        float3 c = verts[ic];
        float3 opp = verts[iopp];

        // Compute the face normal (using vertex order a, b, c).
        float3 normal = cross(b - a, c - a);
        // The face is correctly oriented if the outward normal points away from the opposing vertex.
        // Since the opposing vertex is inside the tetrahedron, we expect: dot(normal, (opp - a)) < 0.
        bool orientedCorrectly = (dot(normal, (opp - a)) < 0.0f);

        // Compute Plücker coordinates for each edge.
        // Edge e0: from b to c.
        PluckerLine e0 = make_plucker(b, c - b);
        // Edge e1: from c to a.
        PluckerLine e1 = make_plucker(c, a - c);
        // Edge e2: from a to b.
        PluckerLine e2 = make_plucker(a, b - a);

        float sigma0 = sign(pdot(rayPlucker, e0));
        float sigma1 = sign(pdot(rayPlucker, e1));
        float sigma2 = sign(pdot(rayPlucker, e2));

        // If the face is not oriented correctly, flip the sigma values.
        if (!orientedCorrectly)
        {
            sigma0 = -sigma0;
            sigma1 = -sigma1;
            sigma2 = -sigma2;
        }

        // Process the face only if at least one sigma is nonzero.
        if ((sigma0 != 0.0f) || (sigma1 != 0.0f) || (sigma2 != 0.0f))
        {
            // If no entry face has been selected and all sigmas are nonnegative,
            // then the ray is entering the tetrahedron through this face.
            if ((Fenter == -1) && (sigma0 >= 0.0f) && (sigma1 >= 0.0f) && (sigma2 >= 0.0f))
            {
                Fenter = i;
                t_enter = intersect_face(orig, dir, a, b, c);
            }
            // Else if no exit face has been selected and all sigmas are nonpositive,
            // then the ray is leaving the tetrahedron through this face.
            else if ((Fleave == -1) && (sigma0 <= 0.0f) && (sigma1 <= 0.0f) && (sigma2 <= 0.0f))
            {
                Fleave = i;
                t_exit = intersect_face(orig, dir, a, b, c);
            }
        }
    }

    // If the exit intersection is behind the ray origin, return no hit.
    // if (t_exit <= 0.0f)
    if (t_exit <= 0.0f || Fleave == -1)
    {
        return float2(-1.0f, 0.0f);
    }

    t_enter = max(t_enter, 0.0f);
    float intersection_t = (t_enter + t_exit) * 0.5f;
    float dist = max(t_exit - t_enter, 0.0f);
    return float2(dist, intersection_t);
}
*/

/*
[Differentiable]
float2 ray_tetrahedron_intersect_plucker(
    float3 orig, float3 dir,
    float3 v0, float3 v1, float3 v2, float3 v3)
{
    // Precompute the ray's Plücker coordinate.
    PluckerLine rayPlucker = make_plucker(orig, dir);

    // Pack vertices into an array.
    float3 verts[4] = { v0, v1, v2, v3 };

    // Define each face using an array of indices.
    // Each row: first three indices are the face vertices; the fourth is the opposing vertex.
    // Face ordering:
    // F0: {3, 2, 1, 0} => face (v3, v2, v1), opposite: v0.
    // F1: {2, 3, 0, 1} => face (v2, v3, v0), opposite: v1.
    // F2: {1, 0, 3, 2} => face (v1, v0, v3), opposite: v2.
    // F3: {0, 1, 2, 3} => face (v0, v1, v2), opposite: v3.
    int faceIndices[4][4] = {
        { 3, 2, 1, 0 },
        { 2, 3, 0, 1 },
        { 1, 0, 3, 2 },
        { 0, 1, 2, 3 }
    };

    // Initialize entry and exit face indices and their intersection t-values.
    int Fenter = -1;
    int Fleave = -1;
    float t_enter = -1e30f;
    float t_exit  =  1e30f;

    // Loop over faces in descending order: i = 3, 2, 1, 0.
    for (int i = 3; i >= 0; i--)
    {
        // Retrieve the indices for the face and its opposing vertex.
        int ia = faceIndices[i][0];
        int ib = faceIndices[i][1];
        int ic = faceIndices[i][2];
        int iopp = faceIndices[i][3];

        float3 a   = verts[ia];
        float3 b   = verts[ib];
        float3 c   = verts[ic];
        float3 opp = verts[iopp];

        // Compute the face normal using the vertices (a, b, c).
        float3 normal = cross(b - a, c - a);
        // The face is correctly oriented if its outward normal points away from the opposing vertex.
        // Since the opposing vertex is inside the tetrahedron, we expect: dot(normal, (opp - a)) < 0.
        bool orientedCorrectly = (dot(normal, (opp - a)) < 0.0f);

        // Compute Plücker coordinates for the face edges:
        // e0: from b to c, e1: from c to a, e2: from a to b.
        PluckerLine e0 = make_plucker(b, c - b);
        PluckerLine e1 = make_plucker(c, a - c);
        PluckerLine e2 = make_plucker(a, b - a);

        float sigma0 = sign(pdot(rayPlucker, e0));
        float sigma1 = sign(pdot(rayPlucker, e1));
        float sigma2 = sign(pdot(rayPlucker, e2));

        // If the face is not oriented correctly, flip the sigma values.
        if (!orientedCorrectly)
        {
            sigma0 = -sigma0;
            sigma1 = -sigma1;
            sigma2 = -sigma2;
        }

        // Process this face if at least one sigma is nonzero.
        if ((sigma0 != 0.0f) || (sigma1 != 0.0f) || (sigma2 != 0.0f))
        {
            // If no entry face has been selected and all sigma values are nonnegative,
            // then the ray is entering through this face.
            if ((Fenter == -1) && (sigma0 >= 0.0f) && (sigma1 >= 0.0f) && (sigma2 >= 0.0f))
            {
                Fenter = i;
                t_enter = intersect_face(orig, dir, a, b, c);
            }
            // Else if no exit face has been selected and all sigma values are nonpositive,
            // then the ray is leaving through this face.
            else if ((Fleave == -1) && (sigma0 <= 0.0f) && (sigma1 <= 0.0f) && (sigma2 <= 0.0f))
            {
                Fleave = i;
                t_exit = intersect_face(orig, dir, a, b, c);
            }
        }
    }

    // Ensure the exit intersection is valid.
    if (t_exit <= 0.0f || Fleave == -1)
    {
        return float2(-1.0f, 0.0f);
    }

    float intersection_t = (t_enter + t_exit) * 0.5f;
    float dist = max(t_exit - max(t_enter, 0.0f), 0.0f);
    return float2(dist, intersection_t);
}
*/
// You will need a helper for the Plücker dot product (sometimes called "orient" or "edgeAgainstRay").
// Exactly how pdot is implemented depends on your vector math.
// For example, a typical Plücker test for an oriented line (orig,dir) against an oriented segment (a->b)
// might look like: dot( dir, cross( (a - orig), (b - orig) ) ), etc.
// We'll assume you have pdot(...) already declared somewhere.
//
// Likewise, we assume intersectTriangle(...) or similar is available to return a t-parameter for the
// intersection of (orig, dir) with the plane of the given triangle, returning -1 if no hit.
//
// The code below is a template to show how the faces and sigma-tests can fit together in a consistent way.

bool sameSignXYZ(float x, float y, float z)
{
    // true if (x, y, z) are all < 0 or all > 0
    return ((x < 0 && y < 0 && z < 0) ||
            (x > 0 && y > 0 && z > 0));
}

[Differentiable]
float2 ray_tetrahedron_intersect_plucker(
    float3 orig, float3 dir,
    float3 v0, float3 v1,
    float3 v2, float3 v3)
{
    // Build the Plücker line for the ray.
    PluckerLine ray_plucker = make_plucker(orig, dir);
    // 4 faces of the tetrahedron, each holding 3 vertex indices
    int faceIndices[4][3] = {
        { 3, 2, 1 }, // Face 0: (v3, v2, v1)
        { 2, 3, 0 }, // Face 1: (v2, v3, v0)
        { 1, 0, 3 }, // Face 2: (v1, v0, v3)
        { 0, 1, 2 }  // Face 3: (v0, v1, v2)
    };

    // Compute sigma values from ray Plücker to each oriented edge:
    //   s01 = P(v0->v1), s02 = P(v0->v2), s03 = P(v0->v3),
    //   s12 = P(v1->v2), s13 = P(v1->v3), s23 = P(v2->v3).
    float s01 = pdot(ray_plucker, make_plucker(v0, v1 - v0));
    float s02 = pdot(ray_plucker, make_plucker(v0, v2 - v0));
    float s03 = pdot(ray_plucker, make_plucker(v0, v3 - v0));
    float s12 = pdot(ray_plucker, make_plucker(v1, v2 - v1));
    float s13 = pdot(ray_plucker, make_plucker(v1, v3 - v1));
    float s23 = pdot(ray_plucker, make_plucker(v2, v3 - v2));

    // Identify which faces the ray pierces, based on signs
    int F_1 = -1, F_2 = -1;

    // Face 0: (v3, v2, v1) => edges = (v3->v2)=-s23, (v2->v1)=-s12, (v1->v3)=s13
    if (sameSignXYZ(-s23, -s12, s13))
    {
        if (F_1 == -1) F_1 = 0; else F_2 = 0;
    }

    // Face 1: (v2, v3, v0) => edges = (v2->v3)=s23, (v3->v0)=-s03, (v0->v2)=s02
    if (sameSignXYZ(s23, -s03, s02))
    {
        if (F_1 == -1) F_1 = 1; else F_2 = 1;
    }

    // Face 2: (v1, v0, v3) => edges = (v1->v0)=-s01, (v0->v3)=s03, (v3->v1)=-s13
    if (sameSignXYZ(-s01, s03, -s13))
    {
        if (F_1 == -1) F_1 = 2; else F_2 = 2;
    }

    // Face 3: (v0, v1, v2) => edges = (v0->v1)=s01, (v1->v2)=s12, (v2->v0)=-s02
    if (sameSignXYZ(s01, s12, -s02))
    {
        if (F_1 == -1) F_1 = 3; else F_2 = 3;
    }

    // If we don't have two faces, the ray does not pass through
    if (F_1 == -1 || F_2 == -1) {
        return float2(-1.0, 0.0);
    }

    // Use the same faceIndices to retrieve each face's vertices
    float3 vArr[4] = { v0, v1, v2, v3 };

    float3 f1a = vArr[faceIndices[F_1][0]];
    float3 f1b = vArr[faceIndices[F_1][1]];
    float3 f1c = vArr[faceIndices[F_1][2]];

    float3 f2a = vArr[faceIndices[F_2][0]];
    float3 f2b = vArr[faceIndices[F_2][1]];
    float3 f2c = vArr[faceIndices[F_2][2]];

    // Intersect the ray with the two planes
    float t_min = intersect_face(orig, dir, f1a, f1b, f1c);
    float t_max = intersect_face(orig, dir, f2a, f2b, f2c);

    // Ensure t_min <= t_max
    if (t_min > t_max) {
        float ttmp = t_min;
        t_min = t_max;
        t_max = ttmp;
    }

    // Check validity
    if (t_max < 0.0) {
        return float2(-1.0, 0.0);
    }

    // Example of returning (thickness, midpoint)
    float intersection_t = 0.5f * (t_min + t_max);
    float dist = max(t_max - max(t_min, 0.0f), 0.0f);
    return float2(dist, intersection_t);
}

// Pick the dominant axis (kz) where |dir[kz]| is largest.
// Then pick kx, ky in a cyclic manner, optionally swap if dir[kz] < 0.
int3 pickTransformAxes(float3 dir)
{
    float3 ad = abs(dir);
    int kz = 0;
    if (ad.y > ad.x && ad.y > ad.z) kz = 1;
    else if (ad.z > ad.x && ad.z > ad.y) kz = 2;

    int kx = (kz + 1) % 3;
    int ky = (kx + 1) % 3;

    // If the dominant component is negative, swap kx, ky
    // to preserve triangle winding
    if (dir[kz] < 0.0f)
    {
        int tmp = kx;
        kx = ky;
        ky = tmp;
    }

    return int3(kx, ky, kz);
}

// Shear+scale transform a 3D point P into the new space (where the ray
// becomes (0,0,1)), assuming we have already translated so the ray origin is (0,0,0).
[Differentiable]
float3 shearScalePoint(float3 P, float3 S, int3 axes)
{
    int kz = axes.z;
    int kx = axes.x;
    int ky = axes.y;

    float3 outP;
    return {
        P[kx] - S.x * P[kz],
        P[ky] - S.y * P[kz],
        P[kz] * S.z
    };
}

[Differentiable]
float2 cyrusBeckTetrahedronShearScale(
    float3 orig, float3 dir,
    float3 v0, float3 v1, float3 v2, float3 v3)
{
    // Face index array: each face is a triple of vertex indices
    int faceIndices[4][4] = {
        { 3, 2, 1, 0 }, // Face 0: (v3, v2, v1)
        { 2, 3, 0, 1 }, // Face 1: (v2, v3, v0)
        { 1, 0, 3, 2 }, // Face 2: (v1, v0, v3)
        { 0, 1, 2, 3 }  // Face 3: (v0, v1, v2)
    };

    // --------------------------------------------------------------------
    // A) Translate all vertices so 'orig' is the new (0,0,0)
    //    This ensures the param 't' in new space matches 't' in old space.
    // --------------------------------------------------------------------
    float3 V0 = v0 - orig;
    float3 V1 = v1 - orig;
    float3 V2 = v2 - orig;
    float3 V3 = v3 - orig;

    // --------------------------------------------------------------------
    // B) Determine which axis is dominant and compute shear+scale
    // --------------------------------------------------------------------
    int3 axes = pickTransformAxes(dir);
    int kz = axes.z;
    int kx = axes.x;
    int ky = axes.y;

    float3 S = {
        dir[kx] / dir[kz],
        dir[ky] / dir[kz],
        1.0f / dir[kz]};    // Scale in z

    // Transform each vertex into new space
    float3 A0 = shearScalePoint(V0, S, axes);
    float3 A1 = shearScalePoint(V1, S, axes);
    float3 A2 = shearScalePoint(V2, S, axes);
    float3 A3 = shearScalePoint(V3, S, axes);

    // Now, in that new space:
    //   Ray origin => (0,0,0)
    //   Ray direction => (0,0,1)

    // --------------------------------------------------------------------
    // C) We do standard Cyrus–Beck plane-clipping in the new space.
    //    We'll treat each face (A,B,C) => normal, d, and clip t_enter, t_exit
    // --------------------------------------------------------------------
    float t_enter = 0.0f;
    float t_exit = 1e30f;
    float eps = 1e-12f; // small threshold

    // We'll store the transformed vertices in an array for convenience
    float3 tverts[4] = { A0, A1, A2, A3 };

    for (int faceIdx = 0; faceIdx < 4; faceIdx++)
    {
        float3 P0 = tverts[faceIndices[faceIdx][0]];
        float3 P1 = tverts[faceIndices[faceIdx][1]];
        float3 P2 = tverts[faceIndices[faceIdx][2]];
        float3 P3 = tverts[faceIndices[faceIdx][3]];

        // Face normal in the new space. Flip based on opp vec
        float3 n = cross(P1 - P0, P2 - P0);
        if (dot(n, P3 - P0) > 0) {
            n = -n;
        }

        // Plane equation: dot(n, X) + d = 0 -> d = -dot(n, P0)
        float d = -dot(n, P0);

        // Dist at origin (0,0,0) is dot(n, origin) + d => d
        float dist = d;

        // denom = dot(n, (0,0,1)) => just n.z in new space
        float denom = n.z;

        // Clip as usual
        if (abs(denom) < eps)
        {
            // Parallel to plane
            // If dist > 0 => outside, no intersection
            if (dist > 0.0f)
            {
                return float2(-1.0f, 0.0f);
            }
        }
        else
        {
            float tplane = -dist / denom;
            if (denom < 0.0f)
            {
                // entering
                if (tplane > t_enter) t_enter = tplane;
            }
            else
            {
                // exiting
                if (tplane < t_exit) t_exit = tplane;
            }

            if (t_enter > t_exit)
            {
                return float2(-1.0f, 0.0f); // no intersection
            }
        }
    }

    // --------------------------------------------------------------------
    // D) We have t_enter..t_exit in the new space. Because we set 'orig'->(0,0,0)
    //    before the transform, that param range is valid in the *original* space!
    // --------------------------------------------------------------------
    if (t_exit <= 0.0f)
    {
        // Intersection is behind the ray start
        return float2(-1.0f, 0.0f);
    }

    // Clip t_enter to be >= 0
    t_enter = max(0.0f, t_enter);
    float clippedRange = t_exit - t_enter;

    if (clippedRange <= 0.0f)
    {
        // Degenerate or behind
        return float2(-1.0f, 0.0f);
    }

    // Return the param-space length plus some representative t
    float midT = (t_enter + t_exit) * 0.5f;
    return float2(clippedRange, midT);
}
