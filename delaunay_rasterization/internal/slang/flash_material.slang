// Reciprocal unit-Fresnel isotropic GGX. Directions point away from the point.
// Shared local evaluator for reflection and mirrored-view backscatter.
[Differentiable]
float flash_ggx_unit(float3 n, float3 l, float3 v, float roughness)
{
    float normalLength = length(n);
    // No surface orientation is defined for numerically empty segments.
    if(normalLength < 0.5) return 0.0;
    n = n / normalLength;
    float nl = dot(n,l);
    float nv = dot(n,v);
    if(nl <= 0.0 || nv <= 0.0) return 0.0;
    float halfLength = length(l+v);
    if(halfLength < 1e-8) return 0.0;
    float3 h = (l+v)/halfLength;
    float nh = clamp(dot(n,h),0.0,1.0);
    float r = clamp(roughness,0.04,1.0);
    float a = r*r;
    float a2 = a*a;
    // Avoid cancellation near the peak (especially mirrored-view n == h).
    float3 tangent = cross(n,h);
    float d = dot(tangent,tangent)+a2*nh*nh;
    float D = a2/(3.141592653589793*d*d);
    float gl = 2.0/(nl+sqrt(a2+(1.0-a2)*nl*nl));
    float gv = 2.0/(nv+sqrt(a2+(1.0-a2)*nv*nv));
    return 0.25*D*gl*gv;
}
