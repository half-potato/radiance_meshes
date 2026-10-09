import utils;
import intersect;
import camera;
import safe_math;

static const uint TILE_HEIGHT = PYTHON_TILE_HEIGHT;
static const uint TILE_WIDTH = PYTHON_TILE_WIDTH;
static const int FEATURE_DIM = 128; // Fixed dimension size

// -------------------------------------------------------------------------
// 1. Data Structures for High-Dim Features
// -------------------------------------------------------------------------

struct FeatureVector : IDifferentiable
{
    float val[FEATURE_DIM];

    [Differentiable]
    static FeatureVector zero() {
        FeatureVector f;
        [ForceUnroll] for(int i=0; i<FEATURE_DIM; ++i) f.val[i] = 0.f;
        return f;
    }

    [Differentiable]
    static FeatureVector add(FeatureVector a, FeatureVector b) {
        FeatureVector res;
        [ForceUnroll] for(int i=0; i<FEATURE_DIM; ++i) res.val[i] = a.val[i] + b.val[i];
        return res;
    }

    [Differentiable]
    static FeatureVector scale(FeatureVector a, float b) {
        FeatureVector res;
        [ForceUnroll] for(int i=0; i<FEATURE_DIM; ++i) res.val[i] = a.val[i] * b;
        return res;
    }
};

struct SplatTetrahedra : IDifferentiable
{
    Tetrahedra tet;
    FeatureVector features;
    float density; // 129th channel
};

// -------------------------------------------------------------------------
// 2. Updated Shared Memory & Loaders
// -------------------------------------------------------------------------

groupshared SplatTetrahedra collected_splats[TILE_HEIGHT * TILE_WIDTH];
groupshared uint32_t collected_idx[TILE_HEIGHT * TILE_WIDTH];

#define LOG_EPS 1e-6f

[Differentiable]
SplatTetrahedra load_tet_alphablend(int32_t g_idx,
                                    DiffTensorView vertices,
                                    TensorView<int32_t> indices,
                                    DiffTensorView rgbs) // Assumed shape [N, 129]
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet = load_tetrahedra(vertices, virtual_tet);
    
    FeatureVector f;
    // Load 128 features
    [ForceUnroll]
    for(int i = 0; i < FEATURE_DIM; ++i) {
        f.val[i] = rgbs[uint2(g_idx, i)];
    }
    // Load density from index 128
    float d = rgbs[uint2(g_idx, FEATURE_DIM)];

    return { tet, f, d };
}

struct CtrlPt: IDifferentiable {
    FeatureVector features;
    float alpha;
    float2 dist;
    float density;
};

// -------------------------------------------------------------------------
// 3. Updated Logic (Update/Undo/Evaluate)
// -------------------------------------------------------------------------

struct PixelState : IDifferentiable {
    FeatureVector features;
    float log_transmittance;
};

[Differentiable]
CtrlPt evaluate_tetra(in SplatTetrahedra g, 
                      in float3 ray_o,
                      in float3 ray_d)
{
    float2 dist;
    int exit_face;
    float3 enter_norm, exit_norm;
    float seg_len;

    bool hit = ray_tetrahedron_intersect_fused(
        ray_o, ray_d, float4x3(g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]), dist, seg_len, enter_norm, exit_norm);

    float dt = max(seg_len, 0.0f);

    if (!hit) {
        return { FeatureVector.zero(), 0.f, dist, g.density };
    } else {
        // Calculate alpha based on density (g.density)
        float alpha_val = safe_clip(- safe_expm1(-g.density * dt), 0, 1);
        
        // Premultiply features by alpha
        FeatureVector premul_features = FeatureVector.scale(g.features, alpha_val);
        
        // Note: We return raw 'alpha_val' for the transmittance calc, 
        // but 'premul_features' for the color accumulation.
        // Original code passed 'rgba.a' as the alpha for transmittance update.
        // Here we pass 'alpha_val' explicitly in the struct.
        return { premul_features, g.density * dt, dist, g.density }; // Using g.density*dt roughly maps to 'rgba.a' in original logic context for update_pixel_state usage
    }
}


[Differentiable]
PixelState update_pixel_state(PixelState pixel_state_t_nm1, FeatureVector features_t_n, float alpha_t_n)
{
    float transmittance_t_nm1 = safe_exp(pixel_state_t_nm1.log_transmittance);
    
    FeatureVector weighted_feat = FeatureVector.scale(features_t_n, transmittance_t_nm1);
    FeatureVector features_t_n_out = FeatureVector.add(pixel_state_t_nm1.features, weighted_feat);
    
    float log_transmittance_t_n = pixel_state_t_nm1.log_transmittance - alpha_t_n;
    
    return { features_t_n_out, log_transmittance_t_n };
}

PixelState undo_pixel_state(PixelState pixel_state_t_n, FeatureVector features_t_n, float alpha_t_n)
{
    float log_transmittance_t_nm1 = pixel_state_t_n.log_transmittance + alpha_t_n;
    float transmittance_t_nm1 = safe_exp(log_transmittance_t_nm1);
    
    FeatureVector weighted_feat = FeatureVector.scale(features_t_n, transmittance_t_nm1);
    
    // Manual subtract
    FeatureVector features_t_nm1;
    [ForceUnroll]
    for(int i=0; i<FEATURE_DIM; ++i) {
        features_t_nm1.val[i] = pixel_state_t_n.features.val[i] - weighted_feat.val[i];
    }
    
    return { features_t_nm1, log_transmittance_t_nm1 };
}

struct AlphaOut: IDifferentiable {
    PixelState state;
    float weight_square;
};

[Differentiable]
AlphaOut update_alpha_out(AlphaOut prev, CtrlPt ctrl) {
    // Note: 'weight' calc from original code preserved but unused in return?
    // float weight = prev.state.log_transmittance * ctrl.alpha; 
    float alpha_clamped = max(-safe_expm1(-ctrl.alpha), 0.f);
    float w = safe_exp(prev.state.log_transmittance) * alpha_clamped;
    return {
        update_pixel_state(prev.state, ctrl.features, ctrl.alpha),
        prev.weight_square + w*w,
    };
}

AlphaOut undo_alpha_out(AlphaOut curr, CtrlPt ctrl) {
    float alpha_clamped = max(-safe_expm1(-ctrl.alpha), 0.f);
    float w = safe_exp(curr.state.log_transmittance) * alpha_clamped;
    return {
        undo_pixel_state(curr.state, ctrl.features, ctrl.alpha),
        curr.weight_square - w*w
    };
}
// -------------------------------------------------------------------------
// 4. Main Kernels
// -------------------------------------------------------------------------

[BackwardDerivative(bwd_alpha_blend)]
AlphaOut alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                   TensorView<int32_t> indices,
                   DiffTensorView vertices,
                   DiffTensorView rgbs,
                   DiffTensorView final_pixel_state,
                   TensorView<int32_t> n_contributors,
                   no_diff Camera cam,
                   no_diff float ray_jitter_x,
                   no_diff float ray_jitter_y,
                   uint32_t2 pix_coord,
                   uint32_t tile_idx_start,
                   uint32_t tile_idx_end)
{
    float2 center_pix_coord = {pix_coord.x+ray_jitter_x, pix_coord.y+ray_jitter_y};
    
    // Init with 0 features and 0 log_transmittance
    PixelState curr_state = { FeatureVector.zero(), 0.f }; 
    AlphaOut ao = { curr_state, 0.f };
    
    float t = cam.min_t;
    uint32_t block_size = cam.tile_height * cam.tile_width;
    bool is_inside = (pix_coord.x < cam.W && pix_coord.y < cam.H);
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start) + block_size - 1) / block_size;
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

    Ray ray = get_ray(cam, center_pix_coord);

    int32_t local_n_contrib = 0;
    int splats_left_to_process = tile_idx_end - tile_idx_start;
    for (int i = 0; i < shared_memory_rounds; i++)
    {
        AllMemoryBarrierWithGroupSync();
        int splat_pointer_offset = i * block_size + thread_rank;
        if (tile_idx_start + splat_pointer_offset < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            collected_splats[thread_rank] = load_tet_alphablend(coll_id, vertices, indices, rgbs);
        }
        AllMemoryBarrierWithGroupSync();
        if (thread_active) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                local_n_contrib++;
                SplatTetrahedra g = collected_splats[j];
                CtrlPt tetra_ctrl = evaluate_tetra(g, ray.o, ray.d);

                AlphaOut new_ao = update_alpha_out(ao, tetra_ctrl);
                ao = new_ao;

                // Threshold check: log(1/255) approx -5.54
                if (ao.state.log_transmittance < -5.54f) {
                    thread_active = false;
                    break;
                }
            }
        }
        splats_left_to_process -= block_size;
    }

    if (is_inside)
        n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = local_n_contrib;

    return ao;
}

void bwd_alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                     TensorView<int32_t> indices,
                     DiffTensorView vertices,
                     DiffTensorView rgbs,
                     DiffTensorView final_pixel_state,
                     TensorView<int32_t> n_contributors,
                     no_diff Camera cam,
                     no_diff float ray_jitter_x,
                     no_diff float ray_jitter_y,
                     uint32_t2 pix_coord,
                     uint32_t tile_idx_start,
                     uint32_t tile_idx_end,
                     AlphaOut d_ao)
{
    float2 center_pix_coord = {pix_coord.x+ray_jitter_x, pix_coord.y+ray_jitter_y};
    bool is_inside = (pix_coord.x < cam.W && pix_coord.y < cam.H);
    uint32_t block_size = cam.tile_height * cam.tile_width;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);

    int splats_left_to_process = tile_idx_end - tile_idx_start;
    uint32_t current_splat_offset = tile_idx_end - tile_idx_start;

    PixelState current_pixel_state = { FeatureVector.zero(), 0.f };
    float current_weight_square;
    int32_t n_contrib_fwd = 0;

    if (is_inside) {
        // Load final feature state from global memory
        [ForceUnroll]
        for(int k=0; k<FEATURE_DIM; ++k) {
            current_pixel_state.features.val[k] = final_pixel_state[uint3(pix_coord.y, pix_coord.x, k)];
        }
        // Assuming last channel of final_pixel_state or separate logic for alpha?
        // Original code loaded RGBA, here we likely just need the accumulated features + accumulated alpha.
        // For simplicity, assuming final_pixel_state stores features [0..127] and alpha at [128].
        current_pixel_state.log_transmittance = final_pixel_state[uint3(pix_coord.y, pix_coord.x, FEATURE_DIM)];
        current_weight_square = final_pixel_state[uint3(pix_coord.y, pix_coord.x, FEATURE_DIM+1)];
        // Actually, 'ao' in forward starts at 0. Here we load the FINAL state to Undo.
        // NOTE: Standard practice is to ignore final alpha load if we reconstruct, 
        // but 'undo_pixel_state' needs the 'current' state to strip layers off.
        
        n_contrib_fwd = n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)];
    }
    AlphaOut ao = {current_pixel_state, current_weight_square};

    Ray ray = get_ray(cam, center_pix_coord);
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

    for (int i = 0; i < rounds; i++)
    {
        AllMemoryBarrierWithGroupSync();
        int progress = i * block_size + thread_rank;
        if (tile_idx_start + progress < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_end - progress - 1]);
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_tet_alphablend(coll_id, vertices, indices, rgbs);
        }
        AllMemoryBarrierWithGroupSync();
        if (is_inside) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                current_splat_offset--;
                if (current_splat_offset >= n_contrib_fwd)
                    continue;
                uint32_t g_idx = collected_idx[j];
                SplatTetrahedra g = collected_splats[j];

                CtrlPt tetra_ctrl = evaluate_tetra(g, ray.o, ray.d);

                // The forward pass applied update_alpha_out for EVERY splat,
                // so the undo must run unconditionally to keep the replayed
                // state in sync. Only the gradient work below may be skipped
                // for negligible contributions (near-identity update).
                ao = undo_alpha_out(ao, tetra_ctrl);
                if (tetra_ctrl.alpha < 1/255.f) continue;

                DifferentialPair<SplatTetrahedra> dp_g = diffPair(g);
                DifferentialPair<CtrlPt> dp_tetra_ctrl = diffPair(tetra_ctrl);
                DifferentialPair<AlphaOut> dp_ao = diffPair(ao);

                bwd_diff(update_alpha_out)(dp_ao, dp_tetra_ctrl, d_ao);
                d_ao = dp_ao.getDifferential();
                
                DifferentialPair<float3> dp_ray_o = diffPair(ray.o);
                DifferentialPair<float3> dp_ray_d = diffPair(ray.d);
                
                bwd_diff(evaluate_tetra)(dp_g, dp_ray_o, dp_ray_d, dp_tetra_ctrl.d);
                bwd_diff(load_tet_alphablend)(g_idx, vertices, indices, rgbs, dp_g.d);
            }
        }
        splats_left_to_process -= block_size;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void splat_tiled(TensorView<int32_t> sorted_gauss_idx,
                 TensorView<int32_t> tile_ranges,
                 TensorView<int32_t> indices,
                 DiffTensorView vertices,
                 DiffTensorView rgbs,
                 DiffTensorView output_img,
                 TensorView<int32_t> n_contributors,
                 TensorCamera tcam,
                 TensorView<float> ray_jitter)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    uint32_t2 pix_coord = globalIdx.xy;
    Camera cam = load_tensor_camera(tcam);
    uint32_t tile_idx = cudaBlockIdx().y * cam.grid_width + cudaBlockIdx().x;
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));

    float rjx = 0;
    float rjy = 0;
    if (is_inside) {
        rjx = no_diff ray_jitter[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), uint32_t(0))];
        rjy = no_diff ray_jitter[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), uint32_t(1))];
    }

    AlphaOut pixel_states = alpha_blend(
        sorted_gauss_idx, indices, vertices, rgbs,
        output_img, n_contributors, cam, rjx, rjy,
        pix_coord,
        tile_idx_start, tile_idx_end);
                                      
    if (is_inside) {
        // Store 128 features + 1 alpha? Or just features.
        // Assuming output_img is [H, W, 128]
        [ForceUnroll]
        for(int k=0; k<FEATURE_DIM; ++k) {
            output_img.storeOnce(uint3(pix_coord.y, pix_coord.x, k), pixel_states.state.features.val[k]);
        }
        // If there is a 129th channel for alpha/transmittance, store it here.
        output_img.storeOnce(uint3(pix_coord.y, pix_coord.x, FEATURE_DIM), pixel_states.state.log_transmittance);
        output_img.storeOnce(uint3(pix_coord.y, pix_coord.x, FEATURE_DIM+1), pixel_states.weight_square);
    }
} 