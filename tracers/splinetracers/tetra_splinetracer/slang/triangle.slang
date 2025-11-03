#define RT_EPS 0
#define BUFFER_SIZE 16
#define LOG_CUTOFF 5.54
//#define tri_per_g 8
#define tri_per_g 4
import spline_machine;
import optix;
import optix_intrinsics;
import tri_intersect;
import safe_math;
import sh;

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
StructuredBuffer<float3> ray_directions;
RWStructuredBuffer<int> start_tet_ids;

RWStructuredBuffer<int2>  side_index;
RWStructuredBuffer<float> densities;
RWStructuredBuffer<float> features;
size_t max_iters;
float tmin;
float tmax;
RaytracingAccelerationStructure traversable;

ControlPoint get_ctrl_pt(uint ind, float t) {
    uint face_ind = (uint)floor(ind / 2);
    uint hitkind = mod(ind, 2);
    uint prim_ind = side_index[face_ind][hitkind];

    let height = densities[prim_ind];

    Features feat;
    SHFeatures sh_feats = {prim_ind, 0, features};
    feat.f0 = get_sh(sh_feats, 0);
    float3 rayd = {0, 0, 1};
    float3 color = eval_sh_col0(rayd, feat);
    ControlPoint ctrl_pt;
    ctrl_pt.t = t;
    let dirac_height = height;
    ctrl_pt.dirac.x = dirac_height;
    ctrl_pt.dirac.y = dirac_height * color.x;
    ctrl_pt.dirac.z = dirac_height * color.y;
    ctrl_pt.dirac.w = dirac_height * color.z;
    return ctrl_pt;
}

[shader("closesthit")]
void ch(
    inout uint2 garbage : SV_RayPayload,
    in float2 barycentric,
)
{
    printf("HI");
}



[shader("anyhit")]
void ah()
{
    float t = RayTCurrent();
    uint ind = PrimitiveIndex();
    uint hitKind = (isFrontFaceHit()) ? 0 : 1;

    float cur_t = RayTMin();

    float test_t;
    uint test_i;
    float h_t = t;
    uint h_i = 2*ind+hitKind;

    // sort insert
    if (t > cur_t) {
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

    if (t < asfloat(get_payload(2*(BUFFER_SIZE-1)))) {
        IgnoreHit();
    }
}

[shader("miss")]
void ms()
{
    set_payload0(asuint(1e10));
    set_payload1(asuint(0.0f));
    set_payload2(asuint(0.0f));
    set_payload3(asuint(0.0f));
    set_payload4(asuint(0.0f));
    set_payload5(255u);
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
    uint start_idx = start_tet_ids[idx.x];
    float density = densities[start_idx];
    SHFeatures sh_feats = {start_idx, 0, features};
    Features feat;
    feat.f0 = get_sh(sh_feats, 0);
    float3 rayd = {0, 0, 1};
    float3 color = eval_sh_col0(rayd, feat);
    state.drgb = {density, density * color.x, density * color.y, density * color.z };

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
            tri = payload[2 * i + 1];
            printf("%i: %f, %i\n", iter, ctrl_pt.t, tri);
            if (ctrl_pt.t > 1e9) {
                end = true;
                break;
            }
            ctrl_pt = get_ctrl_pt(tri, ctrl_pt.t);
            state = update(state, ctrl_pt, tmin, tmax);
            iter++;
            if (!(state.logT < LOG_CUTOFF && iter < max_iters)) break;
        }
        if (end) break;

    }
    let output = extract_color(state);
    fimage[idx.x] = {output.C.x, output.C.y, output.C.z, output.depth};
    let dual_state = to_dual(state, ctrl_pt);
    last_state[idx.x] = dual_state;
    last_dirac[idx.x] = ctrl_pt.dirac;
    last_face[idx.x] = last_tri;
    iters[idx.x] = iter;
}
