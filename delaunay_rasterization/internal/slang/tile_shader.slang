import utils;
import intersect;

uint32_t float2sortable_key(float value)
{
    uint32_t bits = asuint(value);
    if ((bits & 0x80000000) != 0) {
        bits = ~bits;
    } else {
        bits ^= 0x80000000;
    }
    return bits;
}

[AutoPyBindCUDA]
[CUDAKernel]
void generate_keys(TensorView<float> xyz_vs,
                   TensorView<int32_t> rect_tile_space,
                   TensorView<int32_t> index_buffer_offset,
                   TensorView<int64_t> out_unsorted_keys,
                   TensorView<int32_t> out_unsorted_gauss_idx,
                   uint grid_height,
                   uint grid_width)
{
    int32_t globalIdx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (globalIdx >= xyz_vs.size(0))
        return;

    float3 ndc_xyz = {
        xyz_vs[uint2(globalIdx, 0)],
        xyz_vs[uint2(globalIdx, 1)],
        xyz_vs[uint2(globalIdx, 2)]
    };

    // --- FIX 1: Establish Hard Limits for this Thread ---
    int32_t offset;
    int32_t limit; // The exclusive upper bound for this specific thread

    if (globalIdx == 0) {
        offset = 0;
        // The first thread writes up to the first offset
        limit = index_buffer_offset[0];
    } else {
        offset = index_buffer_offset[globalIdx - 1];
        limit = index_buffer_offset[globalIdx];
    }
    
    // Safety clamp against the global buffer size
    int32_t buffer_size = int32_t(out_unsorted_keys.size(0));
    if (limit > buffer_size) limit = buffer_size;

    int32_t rect_min_x = rect_tile_space[uint2(globalIdx, 0)];
    int32_t rect_min_y = rect_tile_space[uint2(globalIdx, 1)];
    int32_t rect_max_x = rect_tile_space[uint2(globalIdx, 2)];
    int32_t rect_max_y = rect_tile_space[uint2(globalIdx, 3)];

    for (int32_t y = rect_min_y; y < rect_max_y; y++)
    {
        for (int32_t x = rect_min_x; x < rect_max_x; x++)
        {
            // --- FIX 2: Check bounds before writing ---
            // If the rectangle is larger than the allocated 'tiles_touched' count,
            // we stop writing to prevent corrupting the next thread's data or crashing.
            if (offset >= limit) 
                return;

            uint64_t key = y * grid_width + x;
            key <<= 32;
            uint32_t fk = float2sortable_key(ndc_xyz.z);
            key = key | fk;

            out_unsorted_keys[offset] = key;
            out_unsorted_gauss_idx[offset] = globalIdx;
            offset++;
        }
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void compute_tile_ranges(TensorView<int64_t> sorted_keys,
                         TensorView<int32_t> out_tile_ranges)
{
    int32_t globalIdx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (globalIdx >= sorted_keys.size(0))
        return;

    // Read tile ID from key. 
    uint32_t currtile = uint32_t(uint64_t(sorted_keys[globalIdx]) >> 32);
    
    // --- FIX 3: Safety check for tile ID ---
    // If garbage keys were generated, this could be out of bounds
    if (currtile >= out_tile_ranges.size(0)) return;

    if (globalIdx == 0)
    {
        out_tile_ranges[uint2(currtile, 0)] = 0;
    }
    else
    {
        uint32_t prevtile = uint32_t(uint64_t(sorted_keys[globalIdx - 1]) >> 32);
        
        // Safety check for prevtile
        if (prevtile < out_tile_ranges.size(0) && currtile != prevtile)
        {
            out_tile_ranges[uint2(prevtile, 1)] = globalIdx;
            out_tile_ranges[uint2(currtile, 0)] = globalIdx;
        }
    }

    if (globalIdx == sorted_keys.size(0) - 1) {
        out_tile_ranges[uint2(currtile, 1)] = int32_t(sorted_keys.size(0));
    }
}
