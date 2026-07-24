// Copyright 2026. Hand-written CUDA backward for the Delaunay alphablend
// rasterizer. Replaces ONLY the Slang bwd_alpha_blend reverse pass; the Slang
// forward is unchanged and its saved images (output_img, xyzd_img,
// distortion_img, n_contributors) are read back here and walked in reverse via
// undo_alpha_out. See delaunay_rasterization/internal/slang/*.slang for the
// ground-truth forward math this file differentiates.
//
// Two levers (design doc cuda-alphablend-backward.md):
//   1. register-lean: recompute-don't-save, template<AUX_DIM>, __launch_bounds__.
//   2. warp-reduced scatter: 16 lanes process the same tet at step j, so the
//      per-scalar grads are reduced across lanes 0-15 with __shfl_down_sync
//      before a single atomicAdd per scalar (16x atomic-traffic cut).

#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>

// ------------------------------------------------------------------ constants
#define TINY_VAL 1.0754944e-20f
#define CLIP_MIN (-1e20f)
#define CLIP_MAX (1e20f)
#define LOG_CLIP_MAX 46.051697f   // log(1e20)

// ------------------------------------------------------------------ small vec
struct f3 { float x, y, z; };
__device__ __forceinline__ f3 mkf3(float x, float y, float z){ f3 r; r.x=x; r.y=y; r.z=z; return r; }
__device__ __forceinline__ f3 operator+(f3 a, f3 b){ return mkf3(a.x+b.x,a.y+b.y,a.z+b.z); }
__device__ __forceinline__ f3 operator-(f3 a, f3 b){ return mkf3(a.x-b.x,a.y-b.y,a.z-b.z); }
__device__ __forceinline__ f3 operator*(f3 a, float s){ return mkf3(a.x*s,a.y*s,a.z*s); }
__device__ __forceinline__ f3 operator*(float s, f3 a){ return mkf3(a.x*s,a.y*s,a.z*s); }
__device__ __forceinline__ float dot3(f3 a, f3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ __forceinline__ f3 cross3(f3 a, f3 b){
    return mkf3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__device__ __forceinline__ float len3(f3 a){ return sqrtf(dot3(a,a)); }

// ------------------------------------------------------------------ safe math
__device__ __forceinline__ float clipf(float v, float lo, float hi){ return fmaxf(fminf(v,hi),lo); }
__device__ __forceinline__ float safe_exp(float v){ return expf(clipf(v, CLIP_MIN, LOG_CLIP_MAX)); }
__device__ __forceinline__ float safe_expm1(float v){ return expm1f(v); }
__device__ __forceinline__ float safe_div(float a, float b){
    if (fabsf(b) < TINY_VAL) return clipf(a / TINY_VAL, CLIP_MIN, CLIP_MAX);
    return clipf(a / b, CLIP_MIN, CLIP_MAX);
}

// phi(x) = (1 - e^-x)/x, with the |x|<1e-6 series branch matching interp_version.slang
__device__ __forceinline__ float phi_fwd(float x){
    if (fabsf(x) < 1e-6f) return 1.0f - x * 0.5f;
    return -safe_expm1(-x) / x;
}
// d phi/dx
__device__ __forceinline__ float phi_deriv(float x){
    if (fabsf(x) < 1e-6f) return -0.5f;
    float e = expf(-x);
    return ((x + 1.0f) * e - 1.0f) / (x * x);
}

// tukey_power_ladder(160*x, 0.1) and its derivative w.r.t x  (scale_dist)
__device__ __forceinline__ float scale_dist_fwd(float x){
    const float p = 0.1f;
    float z = 160.0f * x;
    float c = fabsf(p - 1.0f);                 // 0.9
    float xp = fabsf(z);
    float xs = xp / fmaxf(TINY_VAL, c);
    float s = (z > 0.0f) ? 1.0f : ((z < 0.0f) ? -1.0f : 0.0f);
    return s * c / p * (powf(xs + 1.0f, p) - 1.0f);
}
__device__ __forceinline__ float scale_dist_deriv(float x){
    const float p = 0.1f;
    float z = 160.0f * x;
    float c = fmaxf(TINY_VAL, fabsf(p - 1.0f));
    float xs = fabsf(z) / c;
    // d/dz tpl = (|z|/c + 1)^(p-1); chain 160
    return 160.0f * powf(xs + 1.0f, p - 1.0f);
}

// ------------------------------------------------------------------ camera
struct CamRaw {
    float wvt[16];    // row-major 4x4 (already the tcam .T version)
    float K[9];       // row-major 3x3
    float cam_pos[3];
    float dist[4];
    int   camera_type;
    float min_t;
};

struct Ray { f3 o, d; float ortho; };

__device__ f3 undistort_persp(f3 pin, const CamRaw& cam){
    // pin.z unused; operate on (x,y)
    float k1=cam.dist[0], k2=cam.dist[1], p1=cam.dist[2], p2=cam.dist[3];
    if (k1*k1+k2*k2+p1*p1+p2*p2 < 1e-9f) return pin;
    float ux=pin.x, uy=pin.y;
    #pragma unroll 1
    for (int i=0;i<10;i++){
        float x=ux, y=uy;
        float r2=x*x+y*y;
        float d=1.0f + r2*(k1 + r2*k2);
        float fx=d*x + 2.0f*p1*x*y + p2*(r2+2.0f*x*x) - pin.x;
        float fy=d*y + 2.0f*p2*x*y + p1*(r2+2.0f*y*y) - pin.y;
        float d_r=k1 + r2*(2.0f*k2);
        float d_x=2.0f*x*d_r, d_y=2.0f*y*d_r;
        float fx_x=d + d_x*x + 2.0f*p1*y + 6.0f*p2*x;
        float fx_y=d_y*x + 2.0f*p1*x + 2.0f*p2*y;
        float fy_x=d_x*y + 2.0f*p2*y + 2.0f*p1*x;
        float fy_y=d + d_y*y + 2.0f*p2*x + 6.0f*p1*y;
        float den=fy_x*fx_y - fx_x*fy_y;
        if (fabsf(den) < 1e-9f) break;
        float xn=fx*fy_y - fy*fx_y;
        float yn=fy*fx_x - fx*fy_x;
        ux += xn/den; uy += yn/den;
    }
    return mkf3(ux, uy, 0.0f);
}

__device__ f3 fisheye_undistort(f3 pin, const CamRaw& cam){
    float k1=cam.dist[0], k2=cam.dist[1], k3=cam.dist[2], k4=cam.dist[3];
    if (k1*k1+k2*k2+k3*k3+k4*k4 < 1e-9f) return pin;
    float td = sqrtf(pin.x*pin.x+pin.y*pin.y);
    if (td < 1e-8f) return pin;
    float th = td;
    #pragma unroll 1
    for (int i=0;i<10;i++){
        float t2=th*th, t4=t2*t2, t6=t4*t2, t8=t4*t4;
        float fv=th + k1*th*t2 + k2*th*t4 + k3*th*t6 + k4*th*t8 - td;
        float fd=1.0f + 3.0f*k1*t2 + 5.0f*k2*t4 + 7.0f*k3*t6 + 9.0f*k4*t8;
        if (fabsf(fd) < 1e-12f) break;
        float step=fv/fd; th -= step;
        if (fabsf(step) < 1e-10f) break;
    }
    float s = th/td;
    return mkf3(pin.x*s, pin.y*s, 0.0f);
}

__device__ Ray get_ray(const CamRaw& cam, float px, float py){
    float fx=cam.K[0], fy=cam.K[4], cx=cam.K[2], cy=cam.K[5];
    f3 pd = mkf3((px-cx)/fx, (py-cy)/fy, 0.0f);
    f3 pu;
    if (cam.camera_type == 1)      pu = fisheye_undistort(pd, cam);
    else if (cam.camera_type == 0) pu = undistort_persp(pd, cam);
    else                           pu = pd;
    f3 dir_cam;
    if (cam.camera_type == 1) {
        float theta = sqrtf(pu.x*pu.x+pu.y*pu.y);
        theta = fminf(3.14159265f, theta);
        float st = (theta < 1e-6f) ? 1.0f : sinf(theta)/theta;
        dir_cam = mkf3(pu.x*st, pu.y*st, cosf(theta));
    } else {
        f3 v = mkf3(pu.x, pu.y, 1.0f);
        float l = len3(v);
        dir_cam = mkf3(v.x/l, v.y/l, v.z/l);
    }
    // V_rot[i][j] = wvt[j*4+i];  raw = V_rot @ dir_cam
    f3 raw;
    raw.x = cam.wvt[0]*dir_cam.x + cam.wvt[4]*dir_cam.y + cam.wvt[8]*dir_cam.z;
    raw.y = cam.wvt[1]*dir_cam.x + cam.wvt[5]*dir_cam.y + cam.wvt[9]*dir_cam.z;
    raw.z = cam.wvt[2]*dir_cam.x + cam.wvt[6]*dir_cam.y + cam.wvt[10]*dir_cam.z;
    float ortho = len3(raw);
    Ray r;
    r.d = mkf3(raw.x/ortho, raw.y/ortho, raw.z/ortho);
    r.o = mkf3(cam.cam_pos[0]+r.d.x*cam.min_t,
               cam.cam_pos[1]+r.d.y*cam.min_t,
               cam.cam_pos[2]+r.d.z*cam.min_t);
    r.ortho = ortho;
    return r;
}

// ------------------------------------------------------------------ intersect
// Face table: anchor A, edge ends B,C (n = cross(vB-vA, vC-vA)), opposite D.
// constexpr so all indexing is compile-time (keeps grad arrays in registers).
__device__ __forceinline__ int FACE_A(int i){ const int t[4]={0,0,0,1}; return t[i]; }
__device__ __forceinline__ int FACE_B(int i){ const int t[4]={1,1,2,2}; return t[i]; }
__device__ __forceinline__ int FACE_C(int i){ const int t[4]={2,3,3,3}; return t[i]; }
__device__ __forceinline__ int FACE_D(int i){ const int t[4]={3,2,1,0}; return t[i]; }
template<int F> struct Face;
template<> struct Face<0>{ enum{A=0,B=1,C=2,D=3}; };
template<> struct Face<1>{ enum{A=0,B=1,C=3,D=2}; };
template<> struct Face<2>{ enum{A=0,B=2,C=3,D=1}; };
template<> struct Face<3>{ enum{A=1,B=2,C=3,D=0}; };

struct IsectInfo {
    bool  hit;
    float t_enter;      // clamped (final dist.x)
    float t_exit;       // final dist.y
    int   enter_face;   // -1 if none
    int   exit_face;
    bool  enter_clamped;// raw t_enter <= 0 -> grad gated to 0
};

// Forward slab-clip intersection, recording which faces set enter/exit.
__device__ IsectInfo intersect_fwd(f3 orig, f3 dir, const f3 v[4]){
    IsectInfo o; o.hit=false; o.enter_face=-1; o.exit_face=-1; o.enter_clamped=false;
    float t_enter=-1e30f, t_exit=1e30f;
    const float eps=1e-10f;
    #pragma unroll
    for (int i=0;i<4;i++){
        f3 vA=v[FACE_A(i)], vB=v[FACE_B(i)], vC=v[FACE_C(i)], vD=v[FACE_D(i)];
        f3 n = cross3(vB-vA, vC-vA);
        if (dot3(n, vD-vA) > 0.0f) n = mkf3(-n.x,-n.y,-n.z);
        float dd = -dot3(n, vA-orig);
        float denom = dot3(n, dir);
        if (fabsf(denom) < eps){
            if (dd > 0.0f){ o.hit=false; return o; }  // parallel & outside
        } else {
            float tp = -dd/denom;
            if (denom < 0.0f){
                if (tp > t_enter){ t_enter=tp; o.enter_face=i; }
            } else {
                if (tp < t_exit){ t_exit=tp; o.exit_face=i; }
            }
        }
    }
    if (t_enter > t_exit){ o.hit=false; return o; }
    if (t_exit <= 0.0f){ o.hit=false; return o; }
    o.enter_clamped = (t_enter <= 0.0f);
    t_enter = fmaxf(t_enter, 0.0f);
    o.hit=true; o.t_enter=t_enter; o.t_exit=t_exit;
    return o;
}

// VJP of a single selected face's t = num/den, num=dot(n, vA-orig), den=dot(n,dir),
// n = s*cross(vB-vA, vC-vA), s = flip sign (no_diff). Templated on FACE so all
// vertex indices are compile-time (dV[] stays in registers, no local memory).
template<int FACE>
__device__ __forceinline__ void bwd_face_t(const f3 v[4], f3 orig, f3 dir, float d_t, f3 dV[4]){
    const int iA=Face<FACE>::A, iB=Face<FACE>::B, iC=Face<FACE>::C, iD=Face<FACE>::D;
    f3 vA=v[iA], vB=v[iB], vC=v[iC], vD=v[iD];
    f3 ea=vB-vA, eb=vC-vA;
    f3 raw=cross3(ea,eb);
    float s = (dot3(raw, vD-vA) > 0.0f) ? -1.0f : 1.0f;
    f3 n = s*raw;
    float num = dot3(n, vA-orig);
    float den = dot3(n, dir);
    float d_num = d_t / den;
    float d_den = -d_t * num / (den*den);
    f3 d_n = d_num*(vA-orig) + d_den*dir;
    f3 dvA = d_num * n;
    f3 d_raw = s * d_n;
    f3 d_ea = cross3(eb, d_raw);
    f3 d_eb = cross3(d_raw, ea);
    dvA = dvA - d_ea - d_eb;
    dV[iA] = dV[iA] + dvA;
    dV[iB] = dV[iB] + d_ea;
    dV[iC] = dV[iC] + d_eb;
    // iD only affects the no_diff flip sign -> no gradient.
}
__device__ __forceinline__ void bwd_face_t_dispatch(int face, const f3 v[4], f3 orig, f3 dir, float d_t, f3 dV[4]){
    switch(face){
        case 0: bwd_face_t<0>(v,orig,dir,d_t,dV); break;
        case 1: bwd_face_t<1>(v,orig,dir,d_t,dV); break;
        case 2: bwd_face_t<2>(v,orig,dir,d_t,dV); break;
        case 3: bwd_face_t<3>(v,orig,dir,d_t,dV); break;
        default: break;
    }
}

// ------------------------------------------------------------------ moments
// forward moment helpers (dt assumed = end-start, gate applied by caller)
__device__ __forceinline__ float moment0(float dt, float density, float T){
    float alpha = fmaxf(1.0f - safe_exp(-density*dt), 0.0f);
    return T*alpha;
}
__device__ __forceinline__ float moment1(float a, float dt, float density, float T){
    float u=density*dt; float e=safe_exp(-u); float alpha=fmaxf(1.0f-e,0.0f);
    return T*(a*alpha + safe_div(1.0f-(1.0f+u)*e, density));
}
__device__ __forceinline__ float selfdist(float dt, float density, float T){
    float u=density*dt; float e=safe_exp(-u);
    return T*T*safe_div(1.0f - 2.0f*u*e - e*e, density);
}

// ------------------------------------------------------------------ CtrlPt / AlphaOut
template<int AUX>
struct CtrlPt {
    float rgba[4];
    float dist[2];
    float density;
    float xyzd[4];
    float aux[AUX>0?AUX:1];
};
template<int AUX>
struct AlphaOut {
    float rgba[4];   // rgb + logT(a)
    float ds[5];     // distortion state x,y,z,w,v
    float xyzd[4];
    float aux[AUX>0?AUX:1];
};

// ------------------------------------------------------------------ SplatTet (shared)
template<int AUX>
struct SplatTet {
    unsigned int vtet[4];
    f3   verts[4];
    f3   base;
    f3   grd;
    float density;
    float aux[AUX>0?AUX:1];
};

// evaluate forward -> CtrlPt (+ enter/exit/hit). Returns hit&&finite&&a>0.
template<int AUX>
__device__ bool evaluate_fwd(const SplatTet<AUX>& g, const Ray& ray, float min_t,
                             const f3 vn[4], CtrlPt<AUX>& ctrl, IsectInfo& iso){
    (void)min_t;
    iso = intersect_fwd(ray.o, ray.d, g.verts);
    if (!iso.hit){ return false; }
    float enter = iso.t_enter;
    float exit  = fmaxf(iso.t_exit, enter);
    float dt = fmaxf(exit - enter, 0.0f);
    f3 d=ray.d, o=ray.o, v0=g.verts[0];
    f3 v_enter = enter*d + o - v0;
    f3 v_exit  = exit*d  + o - v0;
    float ge = dot3(g.grd, v_enter);
    float gx = dot3(g.grd, v_exit);
    f3 c_enter = mkf3(fmaxf(ge+g.base.x,0.f), fmaxf(ge+g.base.y,0.f), fmaxf(ge+g.base.z,0.f));
    f3 c_exit  = mkf3(fmaxf(gx+g.base.x,0.f), fmaxf(gx+g.base.y,0.f), fmaxf(gx+g.base.z,0.f));
    float ddt = dt*g.density;
    float a_ci = safe_exp(-ddt);
    float phv = phi_fwd(ddt);
    float w0 = phv - a_ci, w1 = 1.0f - phv;
    f3 final_color = w0*c_exit + w1*c_enter;
    float alpha = fmaxf(-safe_expm1(-g.density*dt), 0.0f);

    // Cramer normals
    f3 e1=g.verts[1]-v0, e2=g.verts[2]-v0, e3=g.verts[3]-v0;
    f3 c23=cross3(e2,e3), c31=cross3(e3,e1), c12=cross3(e1,e2);
    float inv_det = safe_div(1.0f, dot3(e1,c23));
    float b1e=dot3(v_enter,c23)*inv_det, b2e=dot3(v_enter,c31)*inv_det, b3e=dot3(v_enter,c12)*inv_det;
    float b0e=1.0f-b1e-b2e-b3e;
    f3 n_enter = b0e*vn[0] + b1e*vn[1] + b2e*vn[2] + b3e*vn[3];
    float b1x=dot3(v_exit,c23)*inv_det, b2x=dot3(v_exit,c31)*inv_det, b3x=dot3(v_exit,c12)*inv_det;
    float b0x=1.0f-b1x-b2x-b3x;
    f3 n_exit = b0x*vn[0] + b1x*vn[1] + b2x*vn[2] + b3x*vn[3];
    f3 normal_contrib = w0*n_exit + w1*n_enter;
    float depth = safe_div(w0*exit + w1*enter, ray.ortho);

    ctrl.rgba[0]=final_color.x; ctrl.rgba[1]=final_color.y; ctrl.rgba[2]=final_color.z;
    ctrl.rgba[3]=g.density*dt;
    ctrl.dist[0]=enter; ctrl.dist[1]=exit;
    ctrl.density=g.density;
    ctrl.xyzd[0]=normal_contrib.x; ctrl.xyzd[1]=normal_contrib.y; ctrl.xyzd[2]=normal_contrib.z; ctrl.xyzd[3]=depth;
    #pragma unroll
    for (int i=0;i<AUX;i++) ctrl.aux[i]=g.aux[i]*alpha;
    // hit filter matching the shader: isfinite(a) && a>0
    if (!isfinite(ctrl.rgba[3]) || ctrl.rgba[3] <= 0.0f) return false;
    return true;
}

// undo_alpha_out: curr, ctrl -> prev (pre-tet). Also returns T (=exp(prev.a)).
template<int AUX>
__device__ float undo_alpha_out(const AlphaOut<AUX>& curr, const CtrlPt<AUX>& ctrl, AlphaOut<AUX>& prev){
    float logT = curr.rgba[3] + ctrl.rgba[3];
    float T = safe_exp(logT);
    prev.rgba[0]=curr.rgba[0]-ctrl.rgba[0]*T;
    prev.rgba[1]=curr.rgba[1]-ctrl.rgba[1]*T;
    prev.rgba[2]=curr.rgba[2]-ctrl.rgba[2]*T;
    prev.rgba[3]=logT;
    float start=scale_dist_fwd(ctrl.dist[0]);
    float end  =scale_dist_fwd(fmaxf(ctrl.dist[0],ctrl.dist[1]));
    float dt=end-start;
    float m0=moment0(dt,ctrl.density,T);
    float m1=moment1(start,dt,ctrl.density,T);
    float sf=selfdist(dt,ctrl.density,T);
    prev.ds[2]=curr.ds[2]-m0;   // z
    prev.ds[3]=curr.ds[3]-m1;   // w
    prev.ds[0]=curr.ds[0]-2.0f*m1*prev.ds[2]; // x
    prev.ds[1]=curr.ds[1]-2.0f*m0*prev.ds[3]; // y
    prev.ds[4]=curr.ds[4]-sf;   // v
    prev.xyzd[0]=curr.xyzd[0]-T*ctrl.xyzd[0];
    prev.xyzd[1]=curr.xyzd[1]-T*ctrl.xyzd[1];
    prev.xyzd[2]=curr.xyzd[2]-T*ctrl.xyzd[2];
    prev.xyzd[3]=curr.xyzd[3]-T*ctrl.xyzd[3];
    #pragma unroll
    for (int i=0;i<AUX;i++) prev.aux[i]=curr.aux[i]-T*ctrl.aux[i];
    return T;
}

// VJP of update_alpha_out at (prev, ctrl) with output cotangent d_ao.
// Updates d_ao in place to d_prev (carry) and writes d_ctrl.
template<int AUX>
__device__ void bwd_update_alpha_out(const AlphaOut<AUX>& prev, const CtrlPt<AUX>& ctrl,
                                     AlphaOut<AUX>& d_ao, CtrlPt<AUX>& d_ctrl){
    float T = safe_exp(prev.rgba[3]);
    // gather output cotangents
    float gc0=d_ao.rgba[0], gc1=d_ao.rgba[1], gc2=d_ao.rgba[2], ga=d_ao.rgba[3];
    float gx0=d_ao.xyzd[0], gx1=d_ao.xyzd[1], gx2=d_ao.xyzd[2], gx3=d_ao.xyzd[3];

    // init d_ctrl accumulators
    #pragma unroll
    for (int i=0;i<4;i++) d_ctrl.rgba[i]=0.f;
    d_ctrl.dist[0]=0.f; d_ctrl.dist[1]=0.f; d_ctrl.density=0.f;
    #pragma unroll
    for (int i=0;i<4;i++) d_ctrl.xyzd[i]=0.f;
    #pragma unroll
    for (int i=0;i<AUX;i++) d_ctrl.aux[i]=0.f;

    float dprev_a = 0.f;

    // A: color / logT
    // d_prev.rgb += gc (leave d_ao.rgb as carry d_prev.rgb)
    d_ctrl.rgba[0]=gc0*T; d_ctrl.rgba[1]=gc1*T; d_ctrl.rgba[2]=gc2*T;
    dprev_a += (gc0*ctrl.rgba[0]+gc1*ctrl.rgba[1]+gc2*ctrl.rgba[2])*T;
    dprev_a += ga;                 // res.a = prev.a - ctrl.a
    d_ctrl.rgba[3] += -ga;

    // B: xyzd
    d_ctrl.xyzd[0]=gx0*T; d_ctrl.xyzd[1]=gx1*T; d_ctrl.xyzd[2]=gx2*T; d_ctrl.xyzd[3]=gx3*T;
    dprev_a += (gx0*ctrl.xyzd[0]+gx1*ctrl.xyzd[1]+gx2*ctrl.xyzd[2]+gx3*ctrl.xyzd[3])*T;

    // C: aux
    #pragma unroll
    for (int i=0;i<AUX;i++){
        float g=d_ao.aux[i];
        d_ctrl.aux[i]=g*T;
        dprev_a += g*ctrl.aux[i]*T;
        // d_prev.aux[i] carry stays = g (d_ao.aux[i] unchanged)
    }

    // E: distortion 5-moment. Recurrence at (old=prev.ds, T, density, ctrl.dist):
    //   new.x=old.x+2*m1*old.z; new.y=old.y+2*m0*old.w; new.z=old.z+m0;
    //   new.w=old.w+m1;         new.v=old.v+self
    float gdx=d_ao.ds[0], gdy=d_ao.ds[1], gdz=d_ao.ds[2], gdw=d_ao.ds[3], gdv=d_ao.ds[4];
    float oz=prev.ds[2], ow=prev.ds[3];
    float start=scale_dist_fwd(ctrl.dist[0]);
    float endarg = fmaxf(ctrl.dist[0], ctrl.dist[1]);
    float end=scale_dist_fwd(endarg);
    float dt=end-start;
    float density=ctrl.density;
    float u=density*dt; float e=safe_exp(-u); float alpha=fmaxf(1.0f-e,0.0f);
    // forward moments (needed for the old.z/old.w carry terms)
    float m0 = T*alpha;
    float g_ = safe_div(1.0f-(1.0f+u)*e, density);
    float m1 = T*(start*alpha + g_);
    // adjoints to m0,m1,self
    float d_m0 = 2.0f*ow*gdy + gdz;
    float d_m1 = 2.0f*oz*gdx + gdw;
    float d_self = gdv;

    // partials of moments wrt (a=start, dt, density, T)
    float dm0_dT  = alpha;
    float dm0_ddt = T*density*e;
    float dm0_dd  = T*dt*e;
    float dm1_dT  = start*alpha + g_;
    float dm1_da  = T*alpha;                       // explicit ∂m1/∂a at fixed dt
    float dm1_ddt = T*(start*density*e + u*e);     // ∂m1/∂dt at fixed a
    float dm1_dd  = T*(start*e*dt + safe_div(u*e*dt, density) - safe_div(g_, density));
    float S_ = safe_div(1.0f - 2.0f*u*e - e*e, density);
    float dself_dT  = 2.0f*T*S_;
    float dself_ddt = T*T * (2.0f*e*(u+e-1.0f));            // = T^2 * ∂S/∂u * density
    float dself_dd  = T*T*( safe_div(2.0f*e*(u+e-1.0f)*dt, density) - safe_div(S_, density) );

    float ddt_adj = d_m0*dm0_ddt + d_m1*dm1_ddt + d_self*dself_ddt;
    float d_start = d_m1*dm1_da - ddt_adj;   // ∂dt/∂start = -1
    float d_end   =              ddt_adj;    // ∂dt/∂end   = +1
    float d_density_dist = d_m0*dm0_dd + d_m1*dm1_dd + d_self*dself_dd;
    float d_T_dist = d_m0*dm0_dT + d_m1*dm1_dT + d_self*dself_dT;

    d_ctrl.dist[0] += d_start * scale_dist_deriv(ctrl.dist[0]);
    if (ctrl.dist[1] >= ctrl.dist[0]) d_ctrl.dist[1] += d_end*scale_dist_deriv(ctrl.dist[1]);
    else                              d_ctrl.dist[0] += d_end*scale_dist_deriv(ctrl.dist[0]);
    d_ctrl.density += d_density_dist;
    dprev_a += d_T_dist * T;

    // carry d_prev.ds
    d_ao.ds[0] = gdx;
    d_ao.ds[1] = gdy;
    d_ao.ds[2] = gdz + 2.0f*m1*gdx;
    d_ao.ds[3] = gdw + 2.0f*m0*gdy;
    d_ao.ds[4] = gdv;

    d_ao.rgba[3] = dprev_a;   // carry d_prev.a
    // d_prev.rgb / d_prev.xyzd / d_prev.aux carries already correct (unchanged).
}

// ---------------------------------------------------------------- bwd_evaluate
// VJP of evaluate_tetra_interp (integrate F + intersect G). Given d_ctrl,
// accumulate grads into the passed accumulators (all pre-zeroed by caller).
template<int AUX>
__device__ void bwd_evaluate(const SplatTet<AUX>& g, const Ray& ray, const IsectInfo& iso,
                             const f3 vn[4], const CtrlPt<AUX>& d_ctrl,
                             f3& d_base, f3& d_grd, float& d_density_out,
                             float* d_aux, f3 d_verts[4], f3 d_vn[4]){
    // ---- forward recompute ----
    float enter = iso.t_enter;
    float exit  = fmaxf(iso.t_exit, enter);
    float dt = fmaxf(exit - enter, 0.0f);
    f3 d=ray.d, o=ray.o, v0=g.verts[0];
    f3 v_enter = enter*d + o - v0;
    f3 v_exit  = exit*d  + o - v0;
    float ge = dot3(g.grd, v_enter);
    float gx = dot3(g.grd, v_exit);
    float density = g.density;
    float ddt = dt*density;
    float a_ci = safe_exp(-ddt);
    float phv = phi_fwd(ddt);
    float phvd = phi_deriv(ddt);
    float w0 = phv - a_ci, w1 = 1.0f - phv;
    // color channel forward + masks
    float ce[3], cx[3]; float me[3], mx[3];
    ce[0]=ge+g.base.x; ce[1]=ge+g.base.y; ce[2]=ge+g.base.z;
    cx[0]=gx+g.base.x; cx[1]=gx+g.base.y; cx[2]=gx+g.base.z;
    f3 c_enter=mkf3(fmaxf(ce[0],0.f),fmaxf(ce[1],0.f),fmaxf(ce[2],0.f));
    f3 c_exit =mkf3(fmaxf(cx[0],0.f),fmaxf(cx[1],0.f),fmaxf(cx[2],0.f));
    me[0]=(ce[0]>0.f); me[1]=(ce[1]>0.f); me[2]=(ce[2]>0.f);
    mx[0]=(cx[0]>0.f); mx[1]=(cx[1]>0.f); mx[2]=(cx[2]>0.f);
    // Cramer
    f3 e1=g.verts[1]-v0, e2=g.verts[2]-v0, e3=g.verts[3]-v0;
    f3 c23=cross3(e2,e3), c31=cross3(e3,e1), c12=cross3(e1,e2);
    float det=dot3(e1,c23);
    float inv_det=safe_div(1.0f, det);
    float dve_c23=dot3(v_enter,c23), dve_c31=dot3(v_enter,c31), dve_c12=dot3(v_enter,c12);
    float dvx_c23=dot3(v_exit,c23),  dvx_c31=dot3(v_exit,c31),  dvx_c12=dot3(v_exit,c12);
    float b1e=dve_c23*inv_det, b2e=dve_c31*inv_det, b3e=dve_c12*inv_det, b0e=1.0f-b1e-b2e-b3e;
    float b1x=dvx_c23*inv_det, b2x=dvx_c31*inv_det, b3x=dvx_c12*inv_det, b0x=1.0f-b1x-b2x-b3x;
    f3 n_enter=b0e*vn[0]+b1e*vn[1]+b2e*vn[2]+b3e*vn[3];
    f3 n_exit =b0x*vn[0]+b1x*vn[1]+b2x*vn[2]+b3x*vn[3];
    float alpha=fmaxf(1.0f-a_ci,0.0f);
    float mask_al=( (density*dt)>0.0f );

    // ---- cotangents ----
    f3 d_rgb   = mkf3(d_ctrl.rgba[0],d_ctrl.rgba[1],d_ctrl.rgba[2]);
    float d_a  = d_ctrl.rgba[3];
    f3 d_normal= mkf3(d_ctrl.xyzd[0],d_ctrl.xyzd[1],d_ctrl.xyzd[2]);
    float d_depth=d_ctrl.xyzd[3];

    // ---- accumulators ----
    float d_enter=0.f, d_exit=0.f, d_ddt=0.f, d_dt=0.f, d_density=0.f;
    f3 d_v_enter=mkf3(0,0,0), d_v_exit=mkf3(0,0,0);
    f3 d_c23=mkf3(0,0,0), d_c31=mkf3(0,0,0), d_c12=mkf3(0,0,0);
    float d_inv_det=0.f;
    d_base=mkf3(0,0,0); d_grd=mkf3(0,0,0);
    for (int k=0;k<4;k++){ d_verts[k]=mkf3(0,0,0); d_vn[k]=mkf3(0,0,0); }

    float dw0_factor = phvd + a_ci;   // dw0/dddt
    float dw1_factor = -phvd;         // dw1/dddt

    // ---- color ----
    {
        f3 d_c_exit  = mkf3(d_rgb.x*w0, d_rgb.y*w0, d_rgb.z*w0);
        f3 d_c_enter = mkf3(d_rgb.x*w1, d_rgb.y*w1, d_rgb.z*w1);
        float d_w0 = dot3(d_rgb, c_exit);
        float d_w1 = dot3(d_rgb, c_enter);
        d_ddt += d_w0*dw0_factor + d_w1*dw1_factor;
        float d_gx = d_c_exit.x*mx[0]+d_c_exit.y*mx[1]+d_c_exit.z*mx[2];
        float d_ge = d_c_enter.x*me[0]+d_c_enter.y*me[1]+d_c_enter.z*me[2];
        d_base.x += d_c_exit.x*mx[0] + d_c_enter.x*me[0];
        d_base.y += d_c_exit.y*mx[1] + d_c_enter.y*me[1];
        d_base.z += d_c_exit.z*mx[2] + d_c_enter.z*me[2];
        d_grd = d_grd + d_ge*v_enter + d_gx*v_exit;
        d_v_enter = d_v_enter + d_ge*g.grd;
        d_v_exit  = d_v_exit  + d_gx*g.grd;
    }
    // rgba.a = density*dt
    d_density += d_a*dt; d_dt += d_a*density;

    // ---- normal ----
    {
        f3 d_n_exit  = mkf3(d_normal.x*w0,d_normal.y*w0,d_normal.z*w0);
        f3 d_n_enter = mkf3(d_normal.x*w1,d_normal.y*w1,d_normal.z*w1);
        float d_w0 = dot3(d_normal, n_exit);
        float d_w1 = dot3(d_normal, n_enter);
        d_ddt += d_w0*dw0_factor + d_w1*dw1_factor;
        // n_exit
        d_vn[0]=d_vn[0]+b0x*d_n_exit; d_vn[1]=d_vn[1]+b1x*d_n_exit;
        d_vn[2]=d_vn[2]+b2x*d_n_exit; d_vn[3]=d_vn[3]+b3x*d_n_exit;
        float d_b0x=dot3(d_n_exit,vn[0]), d_b1x=dot3(d_n_exit,vn[1]),
              d_b2x=dot3(d_n_exit,vn[2]), d_b3x=dot3(d_n_exit,vn[3]);
        d_b1x-=d_b0x; d_b2x-=d_b0x; d_b3x-=d_b0x;
        // n_enter
        d_vn[0]=d_vn[0]+b0e*d_n_enter; d_vn[1]=d_vn[1]+b1e*d_n_enter;
        d_vn[2]=d_vn[2]+b2e*d_n_enter; d_vn[3]=d_vn[3]+b3e*d_n_enter;
        float d_b0e=dot3(d_n_enter,vn[0]), d_b1e=dot3(d_n_enter,vn[1]),
              d_b2e=dot3(d_n_enter,vn[2]), d_b3e=dot3(d_n_enter,vn[3]);
        d_b1e-=d_b0e; d_b2e-=d_b0e; d_b3e-=d_b0e;
        // bary exit: b1x=dot(v_exit,c23)*inv_det ...
        d_v_exit = d_v_exit + (d_b1x*inv_det)*c23 + (d_b2x*inv_det)*c31 + (d_b3x*inv_det)*c12;
        d_c23 = d_c23 + (d_b1x*inv_det)*v_exit; d_c31=d_c31+(d_b2x*inv_det)*v_exit; d_c12=d_c12+(d_b3x*inv_det)*v_exit;
        d_inv_det += d_b1x*dvx_c23 + d_b2x*dvx_c31 + d_b3x*dvx_c12;
        // bary enter
        d_v_enter = d_v_enter + (d_b1e*inv_det)*c23 + (d_b2e*inv_det)*c31 + (d_b3e*inv_det)*c12;
        d_c23 = d_c23 + (d_b1e*inv_det)*v_enter; d_c31=d_c31+(d_b2e*inv_det)*v_enter; d_c12=d_c12+(d_b3e*inv_det)*v_enter;
        d_inv_det += d_b1e*dve_c23 + d_b2e*dve_c31 + d_b3e*dve_c12;
    }

    // ---- depth ----
    {
        float d_int1d = d_depth / ray.ortho;
        float d_w0 = d_int1d*exit;
        float d_w1 = d_int1d*enter;
        d_exit  += d_int1d*w0;
        d_enter += d_int1d*w1;
        d_ddt += d_w0*dw0_factor + d_w1*dw1_factor;
    }

    // ---- dist / density passthrough ----
    d_enter += d_ctrl.dist[0];
    d_exit  += d_ctrl.dist[1];
    d_density += d_ctrl.density;

    // ---- aux ----
    {
        float d_alpha=0.f;
        #pragma unroll
        for (int i=0;i<AUX;i++){
            d_aux[i] += d_ctrl.aux[i]*alpha;   // grad to g.aux[i]
            d_alpha  += d_ctrl.aux[i]*g.aux[i];
        }
        d_ddt += d_alpha*a_ci*mask_al;   // alpha=1-exp(-ddt) -> d/dddt = a_ci
    }

    // ---- Cramer cofactor VJPs ----
    // ddt -> dt, density
    d_dt += d_ddt*density; d_density += d_ddt*dt;
    // inv_det -> det
    float d_det = -d_inv_det*inv_det*inv_det;
    f3 d_e1=mkf3(0,0,0), d_e2=mkf3(0,0,0), d_e3=mkf3(0,0,0);
    // det=dot(e1,c23)
    d_e1 = d_e1 + d_det*c23; d_c23 = d_c23 + d_det*e1;
    // c23=cross(e2,e3)
    d_e2 = d_e2 + cross3(e3,d_c23); d_e3 = d_e3 + cross3(d_c23,e2);
    // c31=cross(e3,e1)
    d_e3 = d_e3 + cross3(e1,d_c31); d_e1 = d_e1 + cross3(d_c31,e3);
    // c12=cross(e1,e2)
    d_e1 = d_e1 + cross3(e2,d_c12); d_e2 = d_e2 + cross3(d_c12,e1);
    // e*=verts*-v0
    d_verts[1]=d_verts[1]+d_e1; d_verts[2]=d_verts[2]+d_e2; d_verts[3]=d_verts[3]+d_e3;
    d_verts[0]=d_verts[0]-(d_e1+d_e2+d_e3);

    // ---- v_enter/v_exit -> enter/exit/verts0 ----
    d_enter += dot3(d_v_enter, ray.d);
    d_exit  += dot3(d_v_exit,  ray.d);
    d_verts[0]=d_verts[0]-(d_v_enter+d_v_exit);

    // ---- dt gate ----
    float mask_dt = ((exit-enter)>0.0f);
    d_exit  += d_dt*mask_dt;
    d_enter += -d_dt*mask_dt;

    d_density_out += d_density;

    // ---- intersect (G) ----
    // exit = max(iso.t_exit, enter)
    float d_texit_raw;
    if (iso.t_exit >= enter){ d_texit_raw = d_exit; }
    else { d_enter += d_exit; d_texit_raw = 0.f; }
    // enter = max(raw_t_enter,0)
    float d_enter_face = iso.enter_clamped ? 0.f : d_enter;
    if (iso.enter_face >= 0 && !iso.enter_clamped)
        bwd_face_t_dispatch(iso.enter_face, g.verts, ray.o, ray.d, d_enter_face, d_verts);
    if (iso.exit_face >= 0)
        bwd_face_t_dispatch(iso.exit_face, g.verts, ray.o, ray.d, d_texit_raw, d_verts);
}

// ---------------------------------------------------------------- update fwd (for tests)
template<int AUX>
__device__ void update_alpha_out_fwd(const AlphaOut<AUX>& prev, const CtrlPt<AUX>& ctrl, AlphaOut<AUX>& res){
    float T = safe_exp(prev.rgba[3]);
    res.rgba[0]=prev.rgba[0]+ctrl.rgba[0]*T;
    res.rgba[1]=prev.rgba[1]+ctrl.rgba[1]*T;
    res.rgba[2]=prev.rgba[2]+ctrl.rgba[2]*T;
    res.rgba[3]=prev.rgba[3]-ctrl.rgba[3];
    // distortion
    float start=scale_dist_fwd(ctrl.dist[0]);
    float end=scale_dist_fwd(fmaxf(ctrl.dist[0],ctrl.dist[1]));
    float dt=end-start;
    float m0=moment0(dt,ctrl.density,T);
    float m1=moment1(start,dt,ctrl.density,T);
    float sf=selfdist(dt,ctrl.density,T);
    res.ds[0]=prev.ds[0]+2.0f*m1*prev.ds[2];
    res.ds[1]=prev.ds[1]+2.0f*m0*prev.ds[3];
    res.ds[2]=prev.ds[2]+m0;
    res.ds[3]=prev.ds[3]+m1;
    res.ds[4]=prev.ds[4]+sf;
    res.xyzd[0]=prev.xyzd[0]+T*ctrl.xyzd[0];
    res.xyzd[1]=prev.xyzd[1]+T*ctrl.xyzd[1];
    res.xyzd[2]=prev.xyzd[2]+T*ctrl.xyzd[2];
    res.xyzd[3]=prev.xyzd[3]+T*ctrl.xyzd[3];
    #pragma unroll
    for (int i=0;i<AUX;i++) res.aux[i]=prev.aux[i]+ctrl.aux[i]*T;
}

// ---------------------------------------------------------------- shared tet load
template<int AUX>
__device__ void load_tet_shared(unsigned int gidx, const int* indices, const float* vertices,
                                const float* cell_values, int cell_stride, bool valid,
                                SplatTet<AUX>& g){
    unsigned int v0=indices[gidx*4+0], v1=indices[gidx*4+1], v2=indices[gidx*4+2], v3=indices[gidx*4+3];
    g.vtet[0]=v0; g.vtet[1]=v1; g.vtet[2]=v2; g.vtet[3]=v3;
    g.verts[0]=mkf3(vertices[v0*3+0],vertices[v0*3+1],vertices[v0*3+2]);
    g.verts[1]=mkf3(vertices[v1*3+0],vertices[v1*3+1],vertices[v1*3+2]);
    g.verts[2]=mkf3(vertices[v2*3+0],vertices[v2*3+1],vertices[v2*3+2]);
    g.verts[3]=mkf3(vertices[v3*3+0],vertices[v3*3+1],vertices[v3*3+2]);
    float dens=fmaxf(cell_values[gidx*cell_stride+0],0.f);
    g.density = valid ? dens : 0.f;
    g.base=mkf3(cell_values[gidx*cell_stride+1],cell_values[gidx*cell_stride+2],cell_values[gidx*cell_stride+3]);
    g.grd =mkf3(cell_values[gidx*cell_stride+4],cell_values[gidx*cell_stride+5],cell_values[gidx*cell_stride+6]);
    #pragma unroll
    for (int i=0;i<AUX;i++) g.aux[i]=cell_values[gidx*cell_stride+7+i];
}

__device__ __forceinline__ float wreduce16(float v){
    v += __shfl_down_sync(0xFFFF, v, 8);
    v += __shfl_down_sync(0xFFFF, v, 4);
    v += __shfl_down_sync(0xFFFF, v, 2);
    v += __shfl_down_sync(0xFFFF, v, 1);
    return v;   // valid at lane 0
}

// ---------------------------------------------------------------- MAIN KERNEL
#ifndef USE_WARP_REDUCE
#define USE_WARP_REDUCE 1
#endif
#ifndef MIN_BLOCKS_PER_SM
#define MIN_BLOCKS_PER_SM 16
#endif
template<int AUX>
__global__ void __launch_bounds__(16, MIN_BLOCKS_PER_SM)
alphablend_bwd_kernel(
    const int* __restrict__ sorted_gauss_idx,
    const int* __restrict__ tile_ranges,
    const int* __restrict__ indices,
    const float* __restrict__ vertices,
    const float* __restrict__ vertex_normals,
    const float* __restrict__ cell_values,
    const float* __restrict__ output_img,
    const float* __restrict__ xyzd_img,
    const float* __restrict__ distortion_img,
    const int* __restrict__ n_contributors,
    const float* __restrict__ ray_jitter,
    const float* __restrict__ grad_output_img,
    const float* __restrict__ grad_xyzd_img,
    const float* __restrict__ grad_distortion_img,
    float* __restrict__ vertices_grad,
    float* __restrict__ vertex_normals_grad,
    float* __restrict__ cell_values_grad,
    CamRaw cam,
    int H, int W, int grid_w, int grid_h, int tile_w, int tile_h,
    int n_tiles, int N_tet, int cell_stride)
{
    const int bs = tile_w*tile_h;   // 16
    __shared__ SplatTet<AUX> collected[16];
    __shared__ unsigned int collected_gidx[16];

    int tile_idx = blockIdx.y*grid_w + blockIdx.x;
    if (tile_idx >= n_tiles) return;
    int start = tile_ranges[tile_idx*2+0];
    int end   = tile_ranges[tile_idx*2+1];

    int px = blockIdx.x*tile_w + threadIdx.x;
    int py = blockIdx.y*tile_h + threadIdx.y;
    int rank = threadIdx.y*tile_w + threadIdx.x;   // 0..15 == lane
    bool is_inside = (px < W && py < H);
    int img_ch = 4 + AUX;

    AlphaOut<AUX> ao, d_ao;
    #pragma unroll
    for (int i=0;i<4;i++){ ao.rgba[i]=0.f; d_ao.rgba[i]=0.f; ao.xyzd[i]=0.f; d_ao.xyzd[i]=0.f; }
    #pragma unroll
    for (int i=0;i<5;i++){ ao.ds[i]=0.f; d_ao.ds[i]=0.f; }
    #pragma unroll
    for (int i=0;i<AUX;i++){ ao.aux[i]=0.f; d_ao.aux[i]=0.f; }

    int n_contrib = 0;
    float rjx=0.f, rjy=0.f;
    if (is_inside){
        int pix = py*W + px;
        #pragma unroll
        for (int c=0;c<4;c++){ ao.rgba[c]=output_img[pix*img_ch+c]; d_ao.rgba[c]=grad_output_img[pix*img_ch+c]; }
        #pragma unroll
        for (int i=0;i<AUX;i++){ ao.aux[i]=output_img[pix*img_ch+4+i]; d_ao.aux[i]=grad_output_img[pix*img_ch+4+i]; }
        #pragma unroll
        for (int c=0;c<4;c++){ ao.xyzd[c]=xyzd_img[pix*4+c]; d_ao.xyzd[c]=grad_xyzd_img[pix*4+c]; }
        #pragma unroll
        for (int c=0;c<5;c++){ ao.ds[c]=distortion_img[pix*5+c]; d_ao.ds[c]=grad_distortion_img[pix*5+c]; }
        n_contrib = n_contributors[pix];
        rjx = ray_jitter[pix*2+0]; rjy = ray_jitter[pix*2+1];
    }

    Ray ray = get_ray(cam, (float)px + rjx, (float)py + rjy);

    int tile_len = end - start;
    int rounds = (tile_len + bs - 1)/bs;
    int splats_left = tile_len;
    long current_offset = tile_len;

    for (int i=0;i<rounds;i++){
        __syncthreads();
        int progress = i*bs + rank;
        if (start + progress < end){
            int coll = sorted_gauss_idx[end - progress - 1];
            bool valid = ((unsigned)coll < (unsigned)N_tet);
            unsigned int gidx = valid ? (unsigned)coll : 0u;
            collected_gidx[rank] = gidx;
            load_tet_shared<AUX>(gidx, indices, vertices, cell_values, cell_stride, valid, collected[rank]);
        }
        __syncthreads();
        int jmax = min(bs, splats_left);
        for (int j=0;j<jmax;j++){
            current_offset--;
            bool active = is_inside && (current_offset < (long)n_contrib) && (current_offset >= 0);

            const SplatTet<AUX>& g = collected[j];   // reference to shared (no local copy)
            unsigned int g_idx = collected_gidx[j];
            // load vertex normals from global
            f3 vn[4];
            #pragma unroll
            for (int k=0;k<4;k++){
                unsigned int vi=g.vtet[k];
                vn[k]=mkf3(vertex_normals[vi*3+0],vertex_normals[vi*3+1],vertex_normals[vi*3+2]);
            }

            // per-lane grads
            float dl_density=0.f; f3 dl_base=mkf3(0,0,0), dl_grd=mkf3(0,0,0);
            float dl_aux[AUX>0?AUX:1];
            #pragma unroll
            for (int a=0;a<AUX;a++) dl_aux[a]=0.f;
            f3 dl_verts[4]={mkf3(0,0,0),mkf3(0,0,0),mkf3(0,0,0),mkf3(0,0,0)};
            f3 dl_vn[4]   ={mkf3(0,0,0),mkf3(0,0,0),mkf3(0,0,0),mkf3(0,0,0)};

            bool contributed = false;
            if (active){
                CtrlPt<AUX> ctrl; IsectInfo iso;
                bool hitok = evaluate_fwd<AUX>(g, ray, cam.min_t, vn, ctrl, iso);
                if (hitok){
                    contributed = true;
                    AlphaOut<AUX> prev;
                    undo_alpha_out<AUX>(ao, ctrl, prev);
                    CtrlPt<AUX> d_ctrl;
                    bwd_update_alpha_out<AUX>(prev, ctrl, d_ao, d_ctrl); // d_ao -> d_prev carry
                    bwd_evaluate<AUX>(g, ray, iso, vn, d_ctrl,
                                      dl_base, dl_grd, dl_density, dl_aux, dl_verts, dl_vn);
                    ao = prev;    // carry pre-tet state
                    float gate = (g.density > 0.f) ? 1.f : 0.f;
                    dl_density *= gate;
                }
            }

            // scatter (only if any lane contributed this j)
            if (__any_sync(0xFFFF, contributed)){
#if USE_WARP_REDUCE
                // lever #2: reduce the 31+AUX scalars across lanes 0-15, then
                // lane 0 issues ONE atomicAdd per scalar (16x atomic-traffic cut).
                float r;
                r=wreduce16(dl_density);  if(rank==0) atomicAdd(&cell_values_grad[g_idx*cell_stride+0], r);
                r=wreduce16(dl_base.x);   if(rank==0) atomicAdd(&cell_values_grad[g_idx*cell_stride+1], r);
                r=wreduce16(dl_base.y);   if(rank==0) atomicAdd(&cell_values_grad[g_idx*cell_stride+2], r);
                r=wreduce16(dl_base.z);   if(rank==0) atomicAdd(&cell_values_grad[g_idx*cell_stride+3], r);
                r=wreduce16(dl_grd.x);    if(rank==0) atomicAdd(&cell_values_grad[g_idx*cell_stride+4], r);
                r=wreduce16(dl_grd.y);    if(rank==0) atomicAdd(&cell_values_grad[g_idx*cell_stride+5], r);
                r=wreduce16(dl_grd.z);    if(rank==0) atomicAdd(&cell_values_grad[g_idx*cell_stride+6], r);
                #pragma unroll
                for (int a=0;a<AUX;a++){ r=wreduce16(dl_aux[a]); if(rank==0) atomicAdd(&cell_values_grad[g_idx*cell_stride+7+a], r); }
                #pragma unroll
                for (int k=0;k<4;k++){
                    unsigned int vi=g.vtet[k];
                    r=wreduce16(dl_verts[k].x); if(rank==0) atomicAdd(&vertices_grad[vi*3+0], r);
                    r=wreduce16(dl_verts[k].y); if(rank==0) atomicAdd(&vertices_grad[vi*3+1], r);
                    r=wreduce16(dl_verts[k].z); if(rank==0) atomicAdd(&vertices_grad[vi*3+2], r);
                    r=wreduce16(dl_vn[k].x);    if(rank==0) atomicAdd(&vertex_normals_grad[vi*3+0], r);
                    r=wreduce16(dl_vn[k].y);    if(rank==0) atomicAdd(&vertex_normals_grad[vi*3+1], r);
                    r=wreduce16(dl_vn[k].z);    if(rank==0) atomicAdd(&vertex_normals_grad[vi*3+2], r);
                }
#else
                // direct per-lane atomics (A/B baseline)
                if (contributed){
                    atomicAdd(&cell_values_grad[g_idx*cell_stride+0], dl_density);
                    atomicAdd(&cell_values_grad[g_idx*cell_stride+1], dl_base.x);
                    atomicAdd(&cell_values_grad[g_idx*cell_stride+2], dl_base.y);
                    atomicAdd(&cell_values_grad[g_idx*cell_stride+3], dl_base.z);
                    atomicAdd(&cell_values_grad[g_idx*cell_stride+4], dl_grd.x);
                    atomicAdd(&cell_values_grad[g_idx*cell_stride+5], dl_grd.y);
                    atomicAdd(&cell_values_grad[g_idx*cell_stride+6], dl_grd.z);
                    #pragma unroll
                    for (int a=0;a<AUX;a++) atomicAdd(&cell_values_grad[g_idx*cell_stride+7+a], dl_aux[a]);
                    #pragma unroll
                    for (int k=0;k<4;k++){
                        unsigned int vi=g.vtet[k];
                        atomicAdd(&vertices_grad[vi*3+0], dl_verts[k].x);
                        atomicAdd(&vertices_grad[vi*3+1], dl_verts[k].y);
                        atomicAdd(&vertices_grad[vi*3+2], dl_verts[k].z);
                        atomicAdd(&vertex_normals_grad[vi*3+0], dl_vn[k].x);
                        atomicAdd(&vertex_normals_grad[vi*3+1], dl_vn[k].y);
                        atomicAdd(&vertex_normals_grad[vi*3+2], dl_vn[k].z);
                    }
                }
#endif
            }
        }
        splats_left -= bs;
    }
}

// ================================================================ TEST KERNELS
template<int AUX>
__global__ void k_update_fwd(int N, const float* prev, const float* ctrl, float* res){
    int i = blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    int PS=13+AUX, CS=11+AUX;
    AlphaOut<AUX> p; CtrlPt<AUX> c; AlphaOut<AUX> r;
    #pragma unroll
    for(int k=0;k<4;k++) p.rgba[k]=prev[i*PS+k];
    #pragma unroll
    for(int k=0;k<5;k++) p.ds[k]=prev[i*PS+4+k];
    #pragma unroll
    for(int k=0;k<4;k++) p.xyzd[k]=prev[i*PS+9+k];
    for(int k=0;k<AUX;k++) p.aux[k]=prev[i*PS+13+k];
    for(int k=0;k<4;k++) c.rgba[k]=ctrl[i*CS+k];
    c.dist[0]=ctrl[i*CS+4]; c.dist[1]=ctrl[i*CS+5]; c.density=ctrl[i*CS+6];
    for(int k=0;k<4;k++) c.xyzd[k]=ctrl[i*CS+7+k];
    for(int k=0;k<AUX;k++) c.aux[k]=ctrl[i*CS+11+k];
    update_alpha_out_fwd<AUX>(p,c,r);
    for(int k=0;k<4;k++) res[i*PS+k]=r.rgba[k];
    for(int k=0;k<5;k++) res[i*PS+4+k]=r.ds[k];
    for(int k=0;k<4;k++) res[i*PS+9+k]=r.xyzd[k];
    for(int k=0;k<AUX;k++) res[i*PS+13+k]=r.aux[k];
}

template<int AUX>
__global__ void k_update_bwd(int N, const float* prev, const float* ctrl, const float* d_res,
                             float* d_prev, float* d_ctrl){
    int i = blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    int PS=13+AUX, CS=11+AUX;
    AlphaOut<AUX> p; CtrlPt<AUX> c;
    for(int k=0;k<4;k++) p.rgba[k]=prev[i*PS+k];
    for(int k=0;k<5;k++) p.ds[k]=prev[i*PS+4+k];
    for(int k=0;k<4;k++) p.xyzd[k]=prev[i*PS+9+k];
    for(int k=0;k<AUX;k++) p.aux[k]=prev[i*PS+13+k];
    for(int k=0;k<4;k++) c.rgba[k]=ctrl[i*CS+k];
    c.dist[0]=ctrl[i*CS+4]; c.dist[1]=ctrl[i*CS+5]; c.density=ctrl[i*CS+6];
    for(int k=0;k<4;k++) c.xyzd[k]=ctrl[i*CS+7+k];
    for(int k=0;k<AUX;k++) c.aux[k]=ctrl[i*CS+11+k];
    // d_ao = d_res (carry), gets overwritten to d_prev
    AlphaOut<AUX> dao;
    for(int k=0;k<4;k++) dao.rgba[k]=d_res[i*PS+k];
    for(int k=0;k<5;k++) dao.ds[k]=d_res[i*PS+4+k];
    for(int k=0;k<4;k++) dao.xyzd[k]=d_res[i*PS+9+k];
    for(int k=0;k<AUX;k++) dao.aux[k]=d_res[i*PS+13+k];
    CtrlPt<AUX> dc;
    bwd_update_alpha_out<AUX>(p,c,dao,dc);
    for(int k=0;k<4;k++) d_prev[i*PS+k]=dao.rgba[k];
    for(int k=0;k<5;k++) d_prev[i*PS+4+k]=dao.ds[k];
    for(int k=0;k<4;k++) d_prev[i*PS+9+k]=dao.xyzd[k];
    for(int k=0;k<AUX;k++) d_prev[i*PS+13+k]=dao.aux[k];
    for(int k=0;k<4;k++) d_ctrl[i*CS+k]=dc.rgba[k];
    d_ctrl[i*CS+4]=dc.dist[0]; d_ctrl[i*CS+5]=dc.dist[1]; d_ctrl[i*CS+6]=dc.density;
    for(int k=0;k<4;k++) d_ctrl[i*CS+7+k]=dc.xyzd[k];
    for(int k=0;k<AUX;k++) d_ctrl[i*CS+11+k]=dc.aux[k];
}

template<int AUX>
__device__ void pack_splat(int i, const float* verts, const float* base, const float* grd,
                           const float* density, const float* aux, SplatTet<AUX>& g){
    for(int k=0;k<4;k++) g.verts[k]=mkf3(verts[(i*4+k)*3+0],verts[(i*4+k)*3+1],verts[(i*4+k)*3+2]);
    g.vtet[0]=0;g.vtet[1]=0;g.vtet[2]=0;g.vtet[3]=0;
    g.base=mkf3(base[i*3+0],base[i*3+1],base[i*3+2]);
    g.grd =mkf3(grd[i*3+0],grd[i*3+1],grd[i*3+2]);
    g.density=density[i];
    for(int k=0;k<AUX;k++) g.aux[k]=aux[i*AUX+k];
}

template<int AUX>
__global__ void k_eval_fwd(int N, const float* verts,const float* base,const float* grd,
                           const float* density,const float* aux,const float* vnrm,
                           float ox,float oy,float oz,float dx,float dy,float dz,float ortho,float min_t,
                           float* out_rgba,float* out_dist,float* out_density,float* out_xyzd,
                           float* out_aux,int* out_hit){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    SplatTet<AUX> g; pack_splat<AUX>(i,verts,base,grd,density,aux,g);
    f3 vn[4];
    for(int k=0;k<4;k++) vn[k]=mkf3(vnrm[(i*4+k)*3+0],vnrm[(i*4+k)*3+1],vnrm[(i*4+k)*3+2]);
    Ray ray; ray.o=mkf3(ox,oy,oz); ray.d=mkf3(dx,dy,dz); ray.ortho=ortho;
    CtrlPt<AUX> ctrl; IsectInfo iso;
    bool hit=evaluate_fwd<AUX>(g,ray,min_t,vn,ctrl,iso);
    out_hit[i]=hit?1:0;
    for(int k=0;k<4;k++) out_rgba[i*4+k]=ctrl.rgba[k];
    out_dist[i*2+0]=ctrl.dist[0]; out_dist[i*2+1]=ctrl.dist[1];
    out_density[i]=ctrl.density;
    for(int k=0;k<4;k++) out_xyzd[i*4+k]=ctrl.xyzd[k];
    for(int k=0;k<AUX;k++) out_aux[i*AUX+k]=ctrl.aux[k];
}

template<int AUX>
__global__ void k_eval_bwd(int N, const float* verts,const float* base,const float* grd,
                           const float* density,const float* aux,const float* vnrm,
                           float ox,float oy,float oz,float dx,float dy,float dz,float ortho,float min_t,
                           const float* d_rgba,const float* d_dist,const float* d_density_c,
                           const float* d_xyzd,const float* d_aux,
                           float* dg_verts,float* dg_base,float* dg_grd,float* dg_density,
                           float* dg_aux,float* dg_vn){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    SplatTet<AUX> g; pack_splat<AUX>(i,verts,base,grd,density,aux,g);
    f3 vn[4];
    for(int k=0;k<4;k++) vn[k]=mkf3(vnrm[(i*4+k)*3+0],vnrm[(i*4+k)*3+1],vnrm[(i*4+k)*3+2]);
    Ray ray; ray.o=mkf3(ox,oy,oz); ray.d=mkf3(dx,dy,dz); ray.ortho=ortho;
    CtrlPt<AUX> ctrl; IsectInfo iso;
    bool hit=evaluate_fwd<AUX>(g,ray,min_t,vn,ctrl,iso);
    // zero outputs
    for(int k=0;k<4;k++){ dg_verts[(i*4+k)*3+0]=0;dg_verts[(i*4+k)*3+1]=0;dg_verts[(i*4+k)*3+2]=0;
                          dg_vn[(i*4+k)*3+0]=0;dg_vn[(i*4+k)*3+1]=0;dg_vn[(i*4+k)*3+2]=0; }
    for(int k=0;k<3;k++){ dg_base[i*3+k]=0; dg_grd[i*3+k]=0; }
    dg_density[i]=0;
    for(int k=0;k<AUX;k++) dg_aux[i*AUX+k]=0;
    if(!hit) return;
    CtrlPt<AUX> dc;
    for(int k=0;k<4;k++) dc.rgba[k]=d_rgba[i*4+k];
    dc.dist[0]=d_dist[i*2+0]; dc.dist[1]=d_dist[i*2+1];
    dc.density=d_density_c[i];
    for(int k=0;k<4;k++) dc.xyzd[k]=d_xyzd[i*4+k];
    for(int k=0;k<AUX;k++) dc.aux[k]=d_aux[i*AUX+k];
    f3 d_base,d_grd; float d_density=0.f; float d_aux_l[AUX>0?AUX:1];
    for(int k=0;k<AUX;k++) d_aux_l[k]=0.f;
    f3 d_verts[4], d_vn[4];
    bwd_evaluate<AUX>(g,ray,iso,vn,dc,d_base,d_grd,d_density,d_aux_l,d_verts,d_vn);
    for(int k=0;k<4;k++){
        dg_verts[(i*4+k)*3+0]=d_verts[k].x; dg_verts[(i*4+k)*3+1]=d_verts[k].y; dg_verts[(i*4+k)*3+2]=d_verts[k].z;
        dg_vn[(i*4+k)*3+0]=d_vn[k].x; dg_vn[(i*4+k)*3+1]=d_vn[k].y; dg_vn[(i*4+k)*3+2]=d_vn[k].z;
    }
    dg_base[i*3+0]=d_base.x; dg_base[i*3+1]=d_base.y; dg_base[i*3+2]=d_base.z;
    dg_grd[i*3+0]=d_grd.x; dg_grd[i*3+1]=d_grd.y; dg_grd[i*3+2]=d_grd.z;
    dg_density[i]=d_density;
    for(int k=0;k<AUX;k++) dg_aux[i*AUX+k]=d_aux_l[k];
}

// ================================================================ HOST
#define CK(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous(), #x " must be cuda+contiguous")

template<int AUX>
static void launch_main(
    const at::Tensor& sti,const at::Tensor& tr,const at::Tensor& indices,
    const at::Tensor& vertices,const at::Tensor& vertex_normals,const at::Tensor& cell_values,
    const at::Tensor& output_img,const at::Tensor& xyzd_img,const at::Tensor& distortion_img,
    const at::Tensor& n_contributors,const at::Tensor& ray_jitter,
    const at::Tensor& g_out,const at::Tensor& g_xyzd,const at::Tensor& g_dist,
    at::Tensor& v_grad,at::Tensor& vn_grad,at::Tensor& cv_grad,
    CamRaw cam,int H,int W,int grid_w,int grid_h,int tile_w,int tile_h,int n_tiles,int N_tet,int cell_stride){
    dim3 grid(grid_w, grid_h, 1);
    dim3 block(tile_w, tile_h, 1);
    alphablend_bwd_kernel<AUX><<<grid, block>>>(
        sti.data_ptr<int>(), tr.data_ptr<int>(), indices.data_ptr<int>(),
        vertices.data_ptr<float>(), vertex_normals.data_ptr<float>(), cell_values.data_ptr<float>(),
        output_img.data_ptr<float>(), xyzd_img.data_ptr<float>(), distortion_img.data_ptr<float>(),
        n_contributors.data_ptr<int>(), ray_jitter.data_ptr<float>(),
        g_out.data_ptr<float>(), g_xyzd.data_ptr<float>(), g_dist.data_ptr<float>(),
        v_grad.data_ptr<float>(), vn_grad.data_ptr<float>(), cv_grad.data_ptr<float>(),
        cam, H, W, grid_w, grid_h, tile_w, tile_h, n_tiles, N_tet, cell_stride);
}

static CamRaw build_cam(const at::Tensor& wvt,const at::Tensor& K,const at::Tensor& cam_pos,
                        const at::Tensor& dist,int camera_type,double min_t){
    CamRaw c;
    auto w=wvt.detach().to(torch::kFloat32).cpu().contiguous();
    auto k=K.detach().to(torch::kFloat32).cpu().contiguous();
    auto p=cam_pos.detach().to(torch::kFloat32).cpu().contiguous();
    auto d=dist.detach().to(torch::kFloat32).cpu().contiguous();
    const float* wp=w.data_ptr<float>(); for(int i=0;i<16;i++) c.wvt[i]=wp[i];
    const float* kp=k.data_ptr<float>(); for(int i=0;i<9;i++) c.K[i]=kp[i];
    const float* pp=p.data_ptr<float>(); for(int i=0;i<3;i++) c.cam_pos[i]=pp[i];
    const float* dp=d.data_ptr<float>(); for(int i=0;i<4;i++) c.dist[i]=dp[i];
    c.camera_type=camera_type; c.min_t=(float)min_t;
    return c;
}

void alphablend_backward(
    at::Tensor sorted_gauss_idx, at::Tensor tile_ranges, at::Tensor indices,
    at::Tensor vertices, at::Tensor vertex_normals, at::Tensor cell_values,
    at::Tensor output_img, at::Tensor xyzd_img, at::Tensor distortion_img,
    at::Tensor n_contributors, at::Tensor ray_jitter,
    at::Tensor grad_output_img, at::Tensor grad_xyzd_img, at::Tensor grad_distortion_img,
    at::Tensor vertices_grad, at::Tensor vertex_normals_grad, at::Tensor cell_values_grad,
    at::Tensor world_view_transform, at::Tensor K, at::Tensor cam_pos, at::Tensor distortion_params,
    int64_t camera_type, double min_t,
    int64_t H, int64_t W, int64_t grid_w, int64_t grid_h, int64_t tile_w, int64_t tile_h)
{
    CK(sorted_gauss_idx); CK(tile_ranges); CK(indices);
    CK(vertices); CK(vertex_normals); CK(cell_values);
    CK(output_img); CK(xyzd_img); CK(distortion_img); CK(n_contributors); CK(ray_jitter);
    CK(grad_output_img); CK(grad_xyzd_img); CK(grad_distortion_img);
    CK(vertices_grad); CK(vertex_normals_grad); CK(cell_values_grad);
    TORCH_CHECK(tile_w*tile_h==16, "CUDA backward requires tile 4x4 (16 threads)");
    int n_tiles=tile_ranges.size(0);
    int N_tet=indices.size(0);
    int cell_stride=cell_values.size(1);
    int aux=cell_stride-7;
    CamRaw cam=build_cam(world_view_transform,K,cam_pos,distortion_params,(int)camera_type,min_t);
    #define CALL(A) launch_main<A>(sorted_gauss_idx,tile_ranges,indices,vertices,vertex_normals,cell_values,\
        output_img,xyzd_img,distortion_img,n_contributors,ray_jitter,grad_output_img,grad_xyzd_img,grad_distortion_img,\
        vertices_grad,vertex_normals_grad,cell_values_grad,cam,(int)H,(int)W,(int)grid_w,(int)grid_h,(int)tile_w,(int)tile_h,n_tiles,N_tet,cell_stride)
    switch(aux){
        case 0: CALL(0); break; case 1: CALL(1); break; case 2: CALL(2); break;
        case 3: CALL(3); break; case 4: CALL(4); break; case 8: CALL(8); break;
        default: TORCH_CHECK(false, "unsupported AUX_DIM ", aux, " (add instantiation)");
    }
    #undef CALL
}

// ------------ test-helper host wrappers
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>
test_eval_fwd(at::Tensor verts,at::Tensor base,at::Tensor grd,at::Tensor density,at::Tensor aux,
              at::Tensor vn,std::vector<double> ray,double min_t){
    int N=verts.size(0); int A=aux.size(1);
    auto opt=verts.options();
    auto rgba=torch::zeros({N,4},opt), dist=torch::zeros({N,2},opt), dens=torch::zeros({N},opt),
         xyzd=torch::zeros({N,4},opt), auxo=torch::zeros({N,A>0?A:1},opt);
    auto hit=torch::zeros({N},verts.options().dtype(torch::kInt32));
    int TH=128, BL=(N+TH-1)/TH;
    #define CALLF(AA) k_eval_fwd<AA><<<BL,TH>>>(N,verts.data_ptr<float>(),base.data_ptr<float>(),grd.data_ptr<float>(),\
        density.data_ptr<float>(),aux.data_ptr<float>(),vn.data_ptr<float>(),\
        (float)ray[0],(float)ray[1],(float)ray[2],(float)ray[3],(float)ray[4],(float)ray[5],(float)ray[6],(float)min_t,\
        rgba.data_ptr<float>(),dist.data_ptr<float>(),dens.data_ptr<float>(),xyzd.data_ptr<float>(),auxo.data_ptr<float>(),hit.data_ptr<int>())
    switch(A){case 0:CALLF(0);break;case 1:CALLF(1);break;case 2:CALLF(2);break;case 3:CALLF(3);break;case 4:CALLF(4);break;case 8:CALLF(8);break;default:TORCH_CHECK(false,"aux");}
    #undef CALLF
    return {rgba,dist,dens,xyzd,auxo,hit};
}

std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>
test_eval_bwd(at::Tensor verts,at::Tensor base,at::Tensor grd,at::Tensor density,at::Tensor aux,
              at::Tensor vn,std::vector<double> ray,double min_t,
              at::Tensor d_rgba,at::Tensor d_dist,at::Tensor d_densc,at::Tensor d_xyzd,at::Tensor d_aux){
    int N=verts.size(0); int A=aux.size(1);
    auto opt=verts.options();
    auto dg_verts=torch::zeros({N,4,3},opt), dg_base=torch::zeros({N,3},opt), dg_grd=torch::zeros({N,3},opt),
         dg_density=torch::zeros({N},opt), dg_aux=torch::zeros({N,A>0?A:1},opt), dg_vn=torch::zeros({N,4,3},opt);
    int TH=128, BL=(N+TH-1)/TH;
    #define CALLB(AA) k_eval_bwd<AA><<<BL,TH>>>(N,verts.data_ptr<float>(),base.data_ptr<float>(),grd.data_ptr<float>(),\
        density.data_ptr<float>(),aux.data_ptr<float>(),vn.data_ptr<float>(),\
        (float)ray[0],(float)ray[1],(float)ray[2],(float)ray[3],(float)ray[4],(float)ray[5],(float)ray[6],(float)min_t,\
        d_rgba.data_ptr<float>(),d_dist.data_ptr<float>(),d_densc.data_ptr<float>(),d_xyzd.data_ptr<float>(),d_aux.data_ptr<float>(),\
        dg_verts.data_ptr<float>(),dg_base.data_ptr<float>(),dg_grd.data_ptr<float>(),dg_density.data_ptr<float>(),dg_aux.data_ptr<float>(),dg_vn.data_ptr<float>())
    switch(A){case 0:CALLB(0);break;case 1:CALLB(1);break;case 2:CALLB(2);break;case 3:CALLB(3);break;case 4:CALLB(4);break;case 8:CALLB(8);break;default:TORCH_CHECK(false,"aux");}
    #undef CALLB
    return {dg_verts,dg_base,dg_grd,dg_density,dg_aux,dg_vn};
}

at::Tensor test_update_fwd(at::Tensor prev,at::Tensor ctrl,int64_t aux){
    int N=prev.size(0); auto res=torch::zeros_like(prev);
    int TH=128,BL=(N+TH-1)/TH;
    #define CU(AA) k_update_fwd<AA><<<BL,TH>>>(N,prev.data_ptr<float>(),ctrl.data_ptr<float>(),res.data_ptr<float>())
    switch(aux){case 0:CU(0);break;case 1:CU(1);break;case 2:CU(2);break;case 3:CU(3);break;case 4:CU(4);break;case 8:CU(8);break;default:TORCH_CHECK(false,"aux");}
    #undef CU
    return res;
}

std::tuple<at::Tensor,at::Tensor> test_update_bwd(at::Tensor prev,at::Tensor ctrl,at::Tensor d_res,int64_t aux){
    int N=prev.size(0); auto d_prev=torch::zeros_like(prev); auto d_ctrl=torch::zeros_like(ctrl);
    int TH=128,BL=(N+TH-1)/TH;
    #define CUB(AA) k_update_bwd<AA><<<BL,TH>>>(N,prev.data_ptr<float>(),ctrl.data_ptr<float>(),d_res.data_ptr<float>(),d_prev.data_ptr<float>(),d_ctrl.data_ptr<float>())
    switch(aux){case 0:CUB(0);break;case 1:CUB(1);break;case 2:CUB(2);break;case 3:CUB(3);break;case 4:CUB(4);break;case 8:CUB(8);break;default:TORCH_CHECK(false,"aux");}
    #undef CUB
    return {d_prev,d_ctrl};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("alphablend_backward", &alphablend_backward, "Delaunay alphablend CUDA backward");
    m.def("test_eval_fwd", &test_eval_fwd);
    m.def("test_eval_bwd", &test_eval_bwd);
    m.def("test_update_fwd", &test_update_fwd);
    m.def("test_update_bwd", &test_update_bwd);
}
