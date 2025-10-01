import math

def next_multiple(value, multiple):
    """Round `value` up to the nearest multiple of `multiple`."""
    return ((value + multiple - 1) // multiple) * multiple

def grid_scale(level, per_level_scale, base_resolution):
    return math.ceil(2**(level * math.log2(per_level_scale)) * base_resolution - 1) + 1

def compute_grid_offsets(cfg, N_POS_DIMS=3):
    """
    Translates the C++ snippet's logic into Python, returning:
      - offset_table: list of offsets per level
      - total_params: sum of all params_in_level

    cfg is a dictionary containing:
      - otype: "HashGrid" / "DenseGrid" / "TiledGrid" etc.
      - n_levels
      - n_features_per_level
      - log2_hashmap_size
      - base_resolution
      - per_level_scale
    """

    # Unpack configuration
    otype               = cfg["otype"]  # e.g. "HashGrid"
    n_levels            = cfg["n_levels"]
    n_features_per_level = cfg["n_features_per_level"]
    log2_hashmap_size   = cfg["log2_hashmap_size"]
    base_resolution     = cfg["base_resolution"]
    per_level_scale     = cfg["per_level_scale"]

    # (Optional checks, similar to C++ throws)
    # e.g., check if n_levels <= some MAX_N_LEVELS
    # if n_levels > 16:
    #     raise ValueError(f"n_levels={n_levels} exceeds maximum allowed")

    offset_table = []
    offset = 0

    # Simulate the "max_params" check for 32-bit safety
    # C++ used std::numeric_limits<uint32_t>::max() / 2
    max_params_32 = (1 << 31) - 1

    for level in range(n_levels):
        # 1) Compute resolution for this level
        resolution = grid_scale(level, per_level_scale, base_resolution)

        # 2) params_in_level = resolution^N_POS_DIMS (capped by max_params_32)
        grid_size = resolution ** N_POS_DIMS
        # params_in_level = grid_size if grid_size <= max_params_32 else max_params_32
        params_in_level = min(grid_size, max_params_32)

        # 3) Align to multiple of 8
        # ic(params_in_level, next_multiple(params_in_level, 8), resolution, max_params_32)
        params_in_level = next_multiple(params_in_level, 8)

        # 4) Adjust based on grid type
        if otype == "DenseGrid":
            # No-op
            pass
        elif otype == "TiledGrid":
            # Tiled can’t exceed base_resolution^N_POS_DIMS
            tiled_max = (base_resolution ** N_POS_DIMS)
            params_in_level = min(params_in_level, tiled_max)
        elif otype == "HashGrid":
            # Hash grid can't exceed 2^log2_hashmap_size
            params_in_level = min(params_in_level, (1 << log2_hashmap_size))
        else:
            raise RuntimeError(f"Invalid grid type '{otype}'")

        params_in_level = params_in_level * n_features_per_level
        # 5) Store offset for this level and increment
        offset_table.append(offset)
        offset += params_in_level

        # (Optional debug print)
        # print(f"Level={level}, resolution={resolution}, params_in_level={params_in_level}, offset={offset}")

    # offset now points past the last level’s parameters
    return offset_table, offset
