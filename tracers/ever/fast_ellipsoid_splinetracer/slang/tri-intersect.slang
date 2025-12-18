// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define epsilon 1e-20
static const float TINY_VAL = 1.0754944e-20;
#define tri_per_g 2
import spline_machine;
import safe_math;

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


static const float3 BASE_VERTICES[6] = {
  { 0.f, 0.f, 1.f},
  { 0.f, 1.f, 0.f},
  { 1.f, 0.f, 0.f},
  { 0.f,-1.f, 0.f},
  {-1.f, 0.f, 0.f},
  { 0.f, 0.f,-1.f},
};

static const uint3 INDICES[8] = {
    {0,2,1}, {0,3,2},
    {0,4,3}, {0,1,4},
    {5,1,2}, {5,2,3},
    {5,3,4}, {5,4,1},
};

[Differentiable]
float2 ray_intersect_unit_sphere(float3 origin, float3 dir, float3 center) 
{
    float3 L = center - origin; 
    float tca = dot(L, dir);
    float d2 = dot(L, L) - tca * tca;
    float radius = 1;
    float r2 = radius*radius;
    if (d2 > r2) return {-1, -1};
    float thc = sqrt(max(r2 - d2, 0.f));
    float t0 = tca - thc;
    float t1 = tca + thc;
    float2 minmaxt;
    if (t0 > t1) {
        minmaxt.x = t1;
        minmaxt.y = t0;
    } else {
        minmaxt.x = t0;
        minmaxt.y = t1;
    }
    return minmaxt;
}

// ellipsoid centered at the origin with radii ra
half2 half_eliIntersect( in half3 ro, in half3 rd, in half3 ra )
{
    // separate out radius for numerical precision?
    half radius = 1.h;
    half3 ra_norm = ra / radius;
    half3 ocn = ro/ra_norm;
    half3 rdn = rd/ra_norm;
    half a = dot( rdn, rdn );
    half bp = -dot( ocn, rdn );
    half3 l = (ocn + bp / a * rdn);
    half h = a*(radius*radius - dot(l, l));
    half c = dot( ocn, ocn ) - 1.h;
    if( h<0.0 ) return half2(-1.0); //no intersection
    let q = bp + half(hsign(bp)) * hsqrt(h);
    return {c/q, q/a};
}

// ellipsoid centered at the origin with radii ra
[Differentiable]
float2 eliIntersect( in float3 ro, in float3 rd, in float3 ra )
{
    // separate out radius for numerical precision?
    float3 ra_norm = ra;
    float3 ocn = ro/ra_norm;
    float3 rdn = rd/ra_norm;
    float a = dot( rdn, rdn );
    float bp = -dot( ocn, rdn );
    float3 l = (ocn + bp / a * rdn);
    float h = a*(1.f - dot(l, l));
    float c = dot( ocn, ocn ) - 1.f;
    if( h<0.0 ) return float2(-1.0); //no intersection
    let q = bp + sign(bp) * sqrt(h);
    return {c/q, q/a};
}

// ellipsoid centered at the origin with radii ra
[Differentiable]
float2 safe_eliIntersect( in float3 ro, in float3 rd, in float3 ra )
{
    // separate out radius for numerical precision?
    float radius = 1;
    float3 ra_norm = ra;
    float3 ocn = safe_div(ro, ra_norm);
    float3 rdn = safe_div(rd, ra_norm);
    float a = dot( rdn, rdn );
    float bp = -dot( ocn, rdn );
    float3 l = (ocn + safe_div(bp, a) * rdn);
    float h = a*(radius*radius - dot(l, l));
    float c = dot( ocn, ocn ) - radius*radius;
    if( h<0.0 ) return float2(-1.0); //no intersection
    let q = bp + sign(bp) * safe_sqrt(h);
    return {safe_div(c, q), safe_div(q, a)};
}

float2 half_ray_intersect_ellipsoid(
    in half3 rayo, 
    in half3 rayd, 
    in half3 scales, 
    in half4 quat)
{
    let Trayd = rotate_vector(rayd, quat);
    let Trayo = rotate_vector(rayo, quat);

    float2 fminmaxt = eliIntersect(Trayo, Trayd, scales);
    return {min(fminmaxt.x, fminmaxt.y), max(fminmaxt.x, fminmaxt.y)};
}

[Differentiable]
float2 ray_intersect_ellipsoid(
    in float3 rayo, 
    in float3 rayd, 
    in float3 scales, 
    in float4 quat)
{
    let Trayd = rotate_vector(rayd, quat);
    let Trayo = rotate_vector(rayo, quat);

    float2 fminmaxt = eliIntersect(Trayo, Trayd, scales);
    return {min(fminmaxt.x, fminmaxt.y), max(fminmaxt.x, fminmaxt.y)};
}


[Differentiable]
float2 safe_ray_intersect_ellipsoid(
    in float3 rayo, 
    in float3 rayd, 
    in float3 scales, 
    in float4 quat)
{
    let Trayd = l2_normalize(rotate_vector(rayd, quat));
    let Trayo = rotate_vector(rayo, quat);

    float2 fminmaxt = safe_eliIntersect(Trayo, Trayd, scales);
    return {min(fminmaxt.x, fminmaxt.y), max(fminmaxt.x, fminmaxt.y)};
}

[TorchEntryPoint]
float2 ray_intersect_ellipsoid_th(
    in float3 rayo, 
    in float3 rayd, 
    in float3 scales, 
    in float4 quat)
{
    return ray_intersect_ellipsoid(rayo, rayd, scales, quat);
}

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
}

[Differentiable]
float density_multi(
    float3 rayo, float3 rayd, float3 scales, float3 mean, float4 quat
) {
    float3 clip_scale = max(scales, 1e-8);
    let Trayd = safe_div(rotate_vector(rayd, quat), clip_scale);
    let Trayo = safe_div(rotate_vector(rayo - mean, quat), clip_scale);
    let Tortho = Trayd - dot(Trayo, Trayd) / dot(Trayo, Trayo) * Trayo;

    let feather_factor = 1-length(Tortho)*length(Tortho);
    return feather_factor;
}

[Differentiable]
ControlPoint intersect(
    float3 rayo, float3 rayd, float3 scales, float3 mean, float4 quat, float3 color, float density, uint face_id, no_diff bool skip_close
) {
    float2 minmaxt = ray_intersect_ellipsoid(rayo - mean, rayd, scales, quat);
    float t = (face_id == 1) ? minmaxt.x : minmaxt.y;

    float dirac_multi = ((face_id == 1) ? density : -density);

    ControlPoint out = {
        t, dirac_multi, dirac_multi*color.x, dirac_multi*color.y, dirac_multi*color.z 
    };
    return out;
}

[Differentiable]
float3 rotate_vector2(float3 v, float4 q)
{
  return mul(v, quat2mat(q));
}

[Differentiable]
ControlPoint safe_intersect(
    float3 rayo, float3 rayd, float3 scales, float3 mean, float4 quat, float3 color, float density, uint face_id, no_diff bool skip_close
) {
    float2 minmaxt = safe_ray_intersect_ellipsoid(rayo - mean, rayd, scales, quat);

    float t = (face_id == 1) ? minmaxt.x : minmaxt.y;
    float dirac_multi = ((face_id == 1) ? density : -density);

    ControlPoint out = {
        t, dirac_multi, dirac_multi*color.x, dirac_multi*color.y, dirac_multi*color.z 
    };
    return out;
}
