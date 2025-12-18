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

ControlPoint get_ctrl_pt(int prim_ind, float t) {

    float density = 0;
    float3 color = { 0.f, 0.f, 0.f };
    float3 grd = { 0.f, 0.f, 0.f };
    if (prim_ind >= 0) {
        density = densities[prim_ind];
        color = {
            features[6 * prim_ind + 0],
            features[6 * prim_ind + 1],
            features[6 * prim_ind + 2]
        };
        grd = {
            features[6 * prim_ind + 3],
            features[6 * prim_ind + 4],
            features[6 * prim_ind + 5]
        };
    }

    ControlPoint ctrl_pt;
    ctrl_pt.t = t;
    ctrl_pt.density = density;
    ctrl_pt.base_color = color;
    ctrl_pt.grd = grd;
    return ctrl_pt;
}

[shader("closesthit")]
void ch(
    inout uint2 garbage : SV_RayPayload,
    in float2 barycentric,
)
{
    // float t = RayTCurrent();
    // uint ind = PrimitiveIndex();
    // uint hitKind = (isFrontFaceHit()) ? 0 : 1;
    // printf("CH(%i, %i, %f),", ind, hitKind, t);
}



[shader("anyhit")]
void ah()
{
    float t = RayTCurrent();
    uint ind = PrimitiveIndex();
    uint hitKind = (isFrontFaceHit()) ? 1 : 0;

    float cur_t = RayTMin();

    float test_t;
    uint test_i;
    float h_t = t;
    uint h_i = 2 * ind + hitKind;

    // sort insert
    if (t > cur_t) {
        [ForceUnroll]
        for (int i=0; i<BUFFER_SIZE; i++) {
            test_t = asfloat(get_payload(i * 2));
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
    int prim_ind = start_tet_ids[idx.x];

    if (prim_ind >= 0) {
        state.density = densities[prim_ind];
        state.base_color = {
            features[6 * prim_ind + 0],
            features[6 * prim_ind + 1],
            features[6 * prim_ind + 2]
        };
        state.grd = {
            features[6 * prim_ind + 3],
            features[6 * prim_ind + 4],
            features[6 * prim_ind + 5]
        };
    }

    let start_id = idx.x * max_iters;
    
    ControlPoint ctrl_pt = {};
    int last_prim_ind = -1;
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
            ctrl_pt.t = asfloat(payload[2 * i]);
            tri = payload[2 * i + 1];
            // printf("%i: %f, %u\n", iter, ctrl_pt.t, tri);
            if (ctrl_pt.t > 1e9) {
                end = true;
                break;
            }
            uint face_ind = (uint)floor(tri / 2);
            uint hitkind = mod(tri, 2);
            last_prim_ind = prim_ind;
            prim_ind = side_index[face_ind][hitkind];
            if (prim_ind == last_prim_ind) {
                // prim_ind = side_index[face_ind][(hitkind + 1) % 2];
                continue;
            }
            ctrl_pt = get_ctrl_pt(prim_ind, ctrl_pt.t);
            // if (idx.x == 25) {
            //     uint face_ind = (uint)floor(tri / 2);
            //     uint hitkind = mod(tri, 2);
            //     int prim_ind = side_index[face_ind][hitkind];
            //     printf("(%f-%f) i:%i@%i, p:%i, b(%f, %f, %f), g(%f, %f, %f)\n", 
            //         state.t, ctrl_pt.t,
            //         tri / 2, tri % 2, prim_ind,
            //         state.base_color.x, state.base_color.y, state.base_color.z,
            //         state.grd.x, state.grd.y, state.grd.z,
            //     );
            // }
            state = update(origin, direction, state, ctrl_pt, tmin, tmax);
            // tri_collection[idx.x + iter * dim.x] = tri;
            iter++;
            if (state.logT > LOG_CUTOFF || iter >= max_iters) break;
        }
        if (end) break;
    }
    let output = extract_color(state);
    fimage[idx.x] = {output.C.x, output.C.y, output.C.z, output.depth};
    last_state[idx.x] = state;
    last_dirac[idx.x] = {0.f, 0.f, 0.f, 0.f};
    last_face[idx.x] = last_prim_ind;
    iters[idx.x] = iter;
}
