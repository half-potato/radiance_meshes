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

#define RT_EPS 0
#define tri_per_g 2
#define LOG_CUTOFF 5.54
#define BUFFER_SIZE 16
import spline_machine;
import optix;
import optix_intrinsics;
import tri_intersect;
import safe_math;
import sh;

struct Camera {
    float fx, fy;
    int height;
    int width;
    float3 U, V, W;
    float3 eye;
};


// LaunchParams
RWStructuredBuffer<uint>        image;
RWStructuredBuffer<float4>      fimage;
RWStructuredBuffer<uint>        iters;
RWStructuredBuffer<uint>        last_face;
RWStructuredBuffer<uint>        touch_count;
RWStructuredBuffer<float4>      last_dirac;
RWStructuredBuffer<SplineState> last_state;
RWStructuredBuffer<int> tri_collection;
StructuredBuffer<float3>      ray_origins;
StructuredBuffer<float3>      ray_directions;
Camera camera;

RWStructuredBuffer<float> half_attribs;

RWStructuredBuffer<float3> means;
RWStructuredBuffer<float3> scales;
RWStructuredBuffer<float4> quats;
RWStructuredBuffer<float> heights;
RWStructuredBuffer<float> features;

size_t sh_degree;
size_t max_iters;
float tmin;
float tmax;
RWStructuredBuffer<float4> initial_drgb;
float max_prim_size;
RaytracingAccelerationStructure traversable;


ControlPoint get_ctrl_pt(uint tri, float t) {
    ControlPoint ctrl_pt;
    let prim_ind = tri / 2;
    let hitkind = tri % 2;
    let height = heights[prim_ind];
    let dirac_height = height * ((hitkind == 1) ? 1 : -1);
    Features feat;
    SHFeatures sh_feats = {prim_ind, sh_degree, features};
    feat.f0 = get_sh(sh_feats, 0);
    float3 rayd = {0, 0, 1};
    let color = eval_sh_col0(rayd, feat);
    ctrl_pt.t = t;
    ctrl_pt.dirac.x = dirac_height;
    ctrl_pt.dirac.y = dirac_height * color.x;
    ctrl_pt.dirac.z = dirac_height * color.y;
    ctrl_pt.dirac.w = dirac_height * color.z;
    return ctrl_pt;
}

[shader("intersection")]
void ellipsoid() {
    uint prim_ind = PrimitiveIndex();

    float3 rayd = WorldRayDirection();
    float3 rayo = WorldRayOrigin();

    let mean = means[prim_ind];
    let scale = scales[prim_ind];
    let quat = quats[prim_ind];
    float2 minmaxt = ray_intersect_ellipsoid(rayo - mean, rayd, scale, quat);

    float cur_t = RayTMin();
    if ((minmaxt.y < cur_t)) {
        return;
    } else {
        bool use_min = cur_t < minmaxt.x;
        if (use_min) {
            optixReportHit(minmaxt.x, 1u, asuint( minmaxt.y ));
        } else if (cur_t < minmaxt.y){
            optixReportHit(minmaxt.y, 0u, asuint( minmaxt.x ));
        }
    }
}

[shader("closesthit")]
void ch(
    inout uint2 garbage : SV_RayPayload,
    in float2 garbage2,
)
{
}

[shader("anyhit")]
void ah()
{
    float t = RayTCurrent();
    float other_t = asfloat(optixGetAttribute_0());
    uint ind = PrimitiveIndex();
    uint hitkind = optixGetHitKind();
    if (hitkind == 0) {
        float temp_t = other_t;
        other_t = t;
        t = temp_t;
    }

    float cur_t = RayTMin();

    float h_t;
    uint h_i;

    float test_t;
    uint test_i;

    for (int n=0; n<2; n++) {
        if (n == 0) {
            h_t = t;
            h_i = 2 * ind + 1;
        } else {
            h_t = other_t;
            h_i = 2 * ind;
        }
        if (h_t > cur_t) {
            [ForceUnroll]
            for (int i=0; i<BUFFER_SIZE; i++) {
                test_t = asfloat(get_payload(i*2));
                if (h_t < test_t) {
                    set_payload(i*2, asuint(h_t));
                    test_i = get_payload(i*2+1);
                    set_payload(i*2+1, h_i);
                    h_i = test_i;
                    h_t = test_t;
                }
            }
        }
    }

    if (t < asfloat(get_payload(2*(BUFFER_SIZE-1)))) {
        IgnoreHit();
    }
}

[shader("miss")]
void ms()
{
}

[shader("raygeneration")]
void rg_float()
{
    let FAST_MODE = false;

    const uint3 idx = DispatchRaysIndex();
    const uint3 dim = DispatchRaysDimensions();
    float3 direction = l2_normalize(ray_directions[idx.x]);
    float3 origin = ray_origins[idx.x] + tmin * ray_directions[idx.x];
    SplineState state = make_empty_state();
    state.t = 0;
    state.drgb = initial_drgb[idx.x];

    let start_id = idx.x * max_iters;
    
    ControlPoint ctrl_pt = {};
    uint last_tri = -1;
    float prev_t = state.t;
    uint next_tri = -1;
    float next_t = 1e20;

    uint tri;

    int iter = 0;
    while (state.logT < LOG_CUTOFF && iter < max_iters)
    {
        let start_t = abs(state.t);
        uint payload[2*BUFFER_SIZE];
        for (int i=0; i<BUFFER_SIZE; i++) {
            payload[2*i] = asuint(1e10f);
        }

        optixTraceP32(
                traversable,
                origin,
                direction,
                start_t,
                tmax,
                payload);
        bool end = false;
        for (int i=0; i<BUFFER_SIZE; i++) {
            ctrl_pt.t = asfloat(payload[2*i]);
            tri = payload[2*i+1];
            if (ctrl_pt.t > 1e9) {
                end = true;
                break;
            }
            ctrl_pt = get_ctrl_pt(tri, ctrl_pt.t);
            state = update(state, ctrl_pt, tmin, tmax, max_prim_size);
            touch_count[tri / tri_per_g]++;
            tri_collection[idx.x + iter * dim.x] = tri;
            iter++;
            if (!(state.logT < LOG_CUTOFF && iter < max_iters)) break;
        }
        if (end) break;

    }
    let output = extract_color(state, tmin);
    fimage[idx.x] = {output.C.x, output.C.y, output.C.z, output.depth};
    let dual_state = to_dual(state, ctrl_pt);
    last_state[idx.x] = dual_state;
    last_dirac[idx.x] = ctrl_pt.dirac;
    last_face[idx.x] = last_tri;
    iters[idx.x] = iter;
}
