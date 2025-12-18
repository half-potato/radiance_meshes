#define epsilon 1e-20
import spline_machine;
import safe_math;

struct GTransform {
    float3 scales;
    float3 mean;
    float4 quat;
    float height;
};

[Differentiable]
float3x3 quat2mat(float4 quat)
{
    let r = quat[0];
    let x = quat[1];
    let y = quat[2];
    let z = quat[3];
    float3x3 R = {
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - r * z),
        2.0 * (x * z + r * y),

        2.0 * (x * y + r * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - r * x),

        2.0 * (x * z - r * y),
        2.0 * (y * z + r * x),
        1.0 - 2.0 * (x * x + y * y)
    };
    return R;
}


[Differentiable]
float3x3 diag(float3 vec)
{
    return {
      vec.x, 0, 0,
      0, vec.y, 0,
      0, 0, vec.z,
    };
}

/*
static const float3 BASE_VERTICES[5] = {
  { 0.f, 0.f, 1.f},
  { 0.f, 1.f, 0.f},
  { 0.86602540378f, -0.5f, 0.f},
  {-0.86602540378f, -0.5f, 0.f},
  { 0.f, 0.f,-1.f},
};

static const uint3 INDICES[6] = {
    {0,2,1}, {0,3,2}, {0,1,3},
    {1,2,4}, {2,3,4}, {3,1,4},
};
*/

/*
static const float3 BASE_VERTICES[4] = {
  { 0.f, 0.f, 0.5},
  { 0.f, 1.f, -0.5},
  { 0.86602540378f, -0.5f, -0.5},
  {-0.86602540378f, -0.5f, -0.5},
};

static const uint3 INDICES[4] = {
    {0,2,1}, {0,3,2}, {0,1,3}, {1,2,3}
};
//*/

//*
static const float3 BASE_VERTICES[4] = {
  { 0.f, 0.f, 1.f},
  { 0.f, 1.f, 0.f},
  { 1.f, 0.f, 0.f},
  { 0.f, 0.f, 0.f},
};

static const uint3 INDICES[4] = {
  {0, 2, 1}, {0, 3, 2}, {0, 1, 3}, {1, 2, 3}
};
//*/


bool ray_intersects_triangle(in float3 ray_origin, 
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

    if (det > -epsilon && det < epsilon)
        return false;    // This ray is parallel to this triangle.

    float inv_det = 1.0 / det;
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

    if (t > epsilon) // ray intersection
    {
        return true;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return false;
}

[Differentiable]
float ray_intersect_plane(float3 ray_origin, 
                          float3 ray_vector, 
                          float3 normal,
                          float3 point)
{
    let denom = dot(normal, ray_vector);
    let numer = dot(normal, ray_origin) - dot(normal, point);

    let dist = safe_div(numer, denom);
    return abs(dist);
    //return (abs(denom) > 1e-20) ? abs(dist) : length(point - ray_origin);
}

[Differentiable]
ControlPoint intersect(
    float3 rayo, float3 rayd, float3x3 verts, float3 base_color, float3 grd, float density
) {
    float3 tri_a = verts[0];
    float3 tri_b = verts[1];
    float3 tri_c = verts[2];
    float3 edge1 = tri_b - tri_a;
    float3 edge2 = tri_c - tri_a;
    float3 normal = l2_normalize(cross(edge1, edge2));
    
    float t = ray_intersect_plane(rayo, rayd, normal, tri_a);
    ControlPoint out = {
        t, density, base_color, grd
    };
    return out;
}
