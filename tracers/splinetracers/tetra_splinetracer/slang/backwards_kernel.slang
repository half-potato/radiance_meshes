#define tri_per_g 4
import spline_machine;
import optix;
import optix_intrinsics;
import tri_intersect;
import safe_math;
import sh;
import tensor_util;

SplineState get_state(TensorView<float> view, uint ind) {
    return {
        float2(view[ind, 0], view[ind, 1]),
        float2(view[ind, 2], view[ind, 3]),
        view[ind, 4],
        float2(view[ind, 5], view[ind, 6]),
        view[ind, 7],
        float4(view[ind, 8], view[ind, 9], view[ind, 10], view[ind, 11]),

        view[ind, 12],
        float3(view[ind, 13], view[ind, 14], view[ind, 15]),
    };
}

struct DualModel {
    TensorView<float> vertices;
    TensorView<int> indices;
    TensorView<int> side_index;
    TensorView<float> densities;
    TensorView<float> features;

    TensorView<float> dL_dvertices;
    TensorView<float> dL_ddensities;
    TensorView<float> dL_dfeatures;
    TensorView<float> dL_drayos;
    TensorView<float> dL_drayds;
    TensorView<float> dL_dmeans2D;
};

DifferentialPair<SplineState>
run_update(in SplineState old_dual_state, in ControlPoint old_ctrl_pt, in ControlPoint ctrl_pt,

           in uint prim_ind, in uint3 tri_inds, in uint ray_ind,

           in DifferentialPair<SplineState> deriv_state,
           in float3 origin, in float3 direction, in float tmin, in float tmax,

           inout DualModel model)
{
    var old_deriv_state = diffPair(from_dual(old_dual_state, old_ctrl_pt), {});
    var deriv_ctrl_pt = diffPair(ctrl_pt, {});
    bwd_diff(update)(old_deriv_state, deriv_ctrl_pt, tmin, tmax, deriv_state.d);
    let density = model.densities[prim_ind];
    Features feat = get_feats(model.features, prim_ind, 0);
    float3 color = eval_color(direction, feat, 0);

    float3x3 verts = {
        { model.vertices[tri_inds.x, 0], model.vertices[tri_inds.x, 1], model.vertices[tri_inds.x, 2] },
        { model.vertices[tri_inds.y, 0], model.vertices[tri_inds.y, 1], model.vertices[tri_inds.y, 2] },
        { model.vertices[tri_inds.z, 0], model.vertices[tri_inds.z, 1], model.vertices[tri_inds.z, 2] },
    };
        

    var deriv_origin = diffPair(origin, {});
    var deriv_direction = diffPair(direction, {});
    var deriv_verts = diffPair(verts, {});
    var deriv_color = diffPair(color, {});
    var deriv_density = diffPair(density, {});
    bwd_diff(intersect)(deriv_origin, deriv_direction,
        deriv_verts, deriv_color, deriv_density, deriv_ctrl_pt.d);
    atomic_add_float3(model.dL_dvertices, tri_inds.x, deriv_verts.d[0]);
    atomic_add_float3(model.dL_dvertices, tri_inds.y, deriv_verts.d[1]);
    atomic_add_float3(model.dL_dvertices, tri_inds.z, deriv_verts.d[2]);
    float temp;
    model.dL_ddensities.InterlockedAdd(prim_ind, deriv_density.d, temp);

    float3 d_rayd = deriv_direction.d;
    deriv_direction = diffPair(direction, {});
    var d_feat = diffPair(feat, {});
    bwd_diff(eval_color)(deriv_direction, d_feat, 0, deriv_color.d);
    d_rayd += deriv_direction.d;

    atomic_add_float3(model.dL_dfeatures, prim_ind, 0u, d_feat.d.f0);

    atomic_add_float3(model.dL_drayos, ray_ind, deriv_origin.d);
    atomic_add_float3(model.dL_drayds, ray_ind, d_rayd);
    return old_deriv_state;
}

ControlPoint load_ctrl_pt(in uint tri_ind, in DualModel model, in float3 origin, float3 direction)
{
    int face_index = tri_ind / 2;
    int hitkind = tri_ind % 2;
    int prim_ind = model.side_index[face_index, hitkind];
    let density = model.densities[prim_ind];

    Features feat = get_feats(model.features, prim_ind, 0);
    float3 color = eval_color(direction, feat, 0);
    uint3 tri_inds = {
        model.indices[face_index, 0],
        model.indices[face_index, 1],
        model.indices[face_index, 2],
    };

    float3x3 verts = {
        { model.vertices[tri_inds.x, 0], model.vertices[tri_inds.x, 1], model.vertices[tri_inds.x, 2] },
        { model.vertices[tri_inds.y, 0], model.vertices[tri_inds.y, 1], model.vertices[tri_inds.y, 2] },
        { model.vertices[tri_inds.z, 0], model.vertices[tri_inds.z, 1], model.vertices[tri_inds.z, 2] },
    };
        
    return intersect(origin, direction,
        verts, color, density);
}

[AutoPyBindCUDA]
[CUDAKernel]
void backwards_kernel(
    TensorView<float> last_state,
    TensorView<float> last_dirac,
    TensorView<int> iters,
    TensorView<int> tri_collection,

    TensorView<float> ray_origins,
    TensorView<float> ray_directions,
    DualModel model,
    TensorView<int32_t> touch_count,

    TensorView<float> dL_doutputs,
    TensorView<int32_t> start_tet_ids,

    float tmin,
    float tmax,
    uint max_iters
    )
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint ray_ind = dispatchIdx.x;
    if (ray_ind >= ray_origins.size(0)) {
        return;
    }
    if (iters[ray_ind] >= max_iters-1) return;
    var dual_state = get_state(last_state, ray_ind);
    //printf("%i, %i, %i\n", last_state.size(0), )
    let direction = get_float3(ray_directions, ray_ind);
    let origin = get_float3(ray_origins, ray_ind) + tmin*direction;
    bool skip_close = tmin > 1e-5;

    var deriv_state = diffPair(dual_state, {});
    float start_t = dual_state.t;
    let bw_origin = origin + dual_state.t * direction;

    uint num_iters = max(min(iters[ray_ind], max_iters), 0);
    var dtmin = diffPair(tmin, {});
    var dtmax = diffPair(tmax, {});

    let dL_dC = get_float4(dL_doutputs, ray_ind);
    let dL_ddistortion_loss = dL_doutputs[ray_ind, 4];
    SplineOutput.Differential dL_doutput;
    dL_doutput.C = {dL_dC.x, dL_dC.y, dL_dC.z};
    dL_doutput.depth = dL_dC.w;
    dL_doutput.distortion_loss = dL_ddistortion_loss;
    bwd_diff(extract_color)(deriv_state, dL_doutput);

    let feature_size = model.features.size(1);

    uint tri_ind = tri_collection[ray_ind + max(num_iters-1, 0) * ray_origins.size(0)];
    ControlPoint ctrl_pt = load_ctrl_pt(tri_ind, model, origin, direction);

    int start_tet_id = start_tet_ids[ray_ind];

    // load old ctrl_pt here because the next loop is about to load the older one instead
    for (int i=num_iters; i-->0; )
    {
        uint old_tri_ind;
        ControlPoint old_ctrl_pt;
        if (i-1 >= 0) {
            old_tri_ind = tri_collection[ray_ind + (i-1) * ray_origins.size(0)];
            old_ctrl_pt = load_ctrl_pt(old_tri_ind, model, origin, direction);
        } else {
            old_tri_ind = start_tet_id;
            old_ctrl_pt = load_ctrl_pt(old_tri_ind, model, origin, direction);
            old_ctrl_pt.t = 0;
        }

        SplineState old_dual_state = inverse_update_dual(dual_state, ctrl_pt, old_ctrl_pt, tmin, tmax);
        int face_index = tri_ind / 2;
        int hitkind = tri_ind % 2;
        int prim_ind = model.side_index[face_index, hitkind];
        uint3 tri_inds = {
            model.indices[face_index, 0],
            model.indices[face_index, 1],
            model.indices[face_index, 2],
        };

        let old_deriv_state = run_update(
                old_dual_state, old_ctrl_pt, ctrl_pt,
                prim_ind, tri_inds, ray_ind,
                deriv_state,
                origin, direction, tmin, tmax,
                model);
        int itemp;
        touch_count.InterlockedAdd((uint)floor(tri_ind / tri_per_g), 1, itemp);

        tri_ind = old_tri_ind;
        dual_state = old_dual_state;
        ctrl_pt = old_ctrl_pt;
        //printf("bwray: %i, t: %f->%f, %i, dirac: %f, logT: %f, d: %f\n", ray_ind, dual_state.t, ctrl_pt.t, tri_ind, ctrl_pt.dirac.x, dual_state.logT, dual_state.drgb.x);

        deriv_state = diffPair(old_dual_state, old_deriv_state.d);

    }
}