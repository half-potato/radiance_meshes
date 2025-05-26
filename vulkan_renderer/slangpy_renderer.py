import slangpy as spy
import numpy as np
import torch # Assuming torch is used for camera/model data structures
from pathlib import Path

# --- Shader File Definitions (assuming they are in a 'shaders' subdirectory) ---
SHADER_DIR = Path(__file__).parent / 'shaders'
TET_SCENE_SLANG = SHADER_DIR / 'TetrahedronScene.slang'
SORT_UTILS_SLANG = SHADER_DIR / 'SortUtils.slang' # Imported by others
GEN_SPHERES_SLANG = SHADER_DIR / 'GenSpheres.cs.slang'
TET_SORT_SLANG = SHADER_DIR / 'TetSort.cs.slang'
EVAL_SH_SLANG = SHADER_DIR / 'EvaluateTetSH.cs.slang'
MESH_RENDERER_SLANG = SHADER_DIR / 'MeshShaderRenderer.3d.slang'
INVERT_ALPHA_SLANG = SHADER_DIR / 'InvertAlpha.cs.slang'
# Add Rose paths if they are not found via relative paths in main shaders
# e.g., ROSE_MATH_UTILS_SLANG = SHADER_DIR / 'Rose/Core/MathUtils.slang'

# Placeholder for Camera and Model classes from your project
class Camera: # As defined in your main script
    def __init__(self, image_height, image_width, fovx, fovy, world_view_transform, camera_center):
        self.image_height = image_height
        self.image_width = image_width
        self.fovx = fovx
        self.fovy = fovy
        self.world_view_transform = world_view_transform # Should be FloatTensor[4,4] (view_to_world)
        self.camera_center = camera_center # Should be FloatTensor[3] (world space position)

class Model: # As defined in your main script
    def __init__(self, vertices, indices, min_t_val):
        self.vertices = vertices # FloatTensor[N,3]
        self.indices = indices   # IntTensor[M,4] (tetrahedron vertex indices)
        self.min_t = min_t_val   # float

def fov2focal(fov_rad, pixels):
    return pixels / (2 * np.tan(0.5 * fov_rad))

def init_slangpy_device(enable_debug_layers=True, defines: list[dict[str, str]] | None = None):
    try:
        # Base compiler options
        options_dict = {
            'include_paths': [str(SHADER_DIR)],
            # 'shader_model': spy.ShaderModel.sm_6_6  # Target Shader Model 6.6 (for SPIR-V 1.5+)
        }
        
        # Add preprocessor definitions if provided
        if defines:
            # Key is 'defines' as per sgl/device/shader.h
            # Value should be a map/dict[str, str]
            # The input `defines` is list[dict[str,str]], e.g. [{'name':'MY_DEFINE', 'value':'1'}]
            # We need to convert it to a flat dict for SlangCompilerOptions
            processed_defines = {}
            for define_item in defines:
                if isinstance(define_item, dict) and 'name' in define_item and 'value' in define_item:
                    processed_defines[define_item['name']] = define_item['value']
                else:
                    # Handle case where defines might not be in the expected list-of-dicts format
                    print(f"Warning: Skipping invalid define item: {define_item}")
            if processed_defines:
                options_dict['defines'] = processed_defines
            
        compiler_options = spy.SlangCompilerOptions(options_dict)
        device = spy.Device(
            type=spy.DeviceType.vulkan, # Explicitly set device type
            enable_debug_layers=enable_debug_layers, 
            compiler_options=compiler_options
        )
        
        print("slangpy device created successfully.")
        return device
    except Exception as e:
        print(f"Failed to create slangpy device: {e}")
        # Consider reraising or handling more gracefully
        raise
    return None

def extract_render_data_from_model(
    nerf_model: torch.nn.Module, # The original NeRF model (e.g., IngpColorModel, TetColorModel)
    slang_mesh_model: Model,     # The SlangModel (vertices, indices, min_t)
    camera: Camera,              # SlangCamera
    scene_scaling: float,
    expected_sh_coeffs_count_per_tet: int = 16 # L0-L3 for shaders
    ):
    """
    Extracts necessary data from the PyTorch NeRF model for the Slang renderer.
    """
    print(f"Extracting data from NeRF model type: {type(nerf_model).__name__}")
    
    num_tets = slang_mesh_model.indices.shape[0]
    num_vertices = slang_mesh_model.vertices.shape[0]
    device = slang_mesh_model.vertices.device

    tet_sh_coeffs_np = np.zeros((num_tets * expected_sh_coeffs_count_per_tet, 3), dtype=np.float32)
    actual_num_sh_coeffs_per_tet = 0 # This will be updated based on model
    tet_densities_np = np.zeros(num_tets, dtype=np.float32)
    # For tet_vertex_colors_flat, we expect (num_tets * 4, 3)
    # Each tet has 4 vertices, each vertex has an RGB color.
    tet_vertex_colors_flat_np = np.zeros((num_tets * 4, 3), dtype=np.float32)
    
    max_density_val = 1.0
    density_scale_from_model = 1.0 # Default

    with torch.no_grad():
        if hasattr(nerf_model, 'tet_sh_param') and nerf_model.tet_sh_param is not None: # Likely TetColorModel
            print("Extracting data from TetColorModel compatible model.")
            # SH Coeffs
            # model.tet_sh_param is (num_tets, num_coeffs * 3)
            # model.sh_deg gives the degree. (deg+1)^2 coeffs.
            # We need to map this to (num_tets * 16, 3) for L0-L3 if shader expects that.
            
            # For TetColorModel, sh_deg is max degree. (sh_deg+1)^2 coeffs.
            # tet_sh_param is (num_tets, (sh_deg+1)**2 * 3)
            # The slang shader expects 16 float3s (L0 to L3).
            
            source_sh_coeffs_flat_rgb = nerf_model.tet_sh_param.reshape(num_tets, -1, 3) # (num_tets, num_coeffs_per_color, 3)
            num_source_coeffs_per_color = source_sh_coeffs_flat_rgb.shape[1]
            actual_num_sh_coeffs_per_tet = num_source_coeffs_per_color

            print(f"Source SH: {num_source_coeffs_per_color} coeffs per color component (degree {nerf_model.sh_deg}). Shader expects {expected_sh_coeffs_count_per_tet}.")

            # Reshape and pad/truncate if necessary
            # target_sh_coeffs is (num_tets, expected_sh_coeffs_count_per_tet, 3)
            target_sh_coeffs = torch.zeros((num_tets, expected_sh_coeffs_count_per_tet, 3), dtype=torch.float32, device=device)
            
            coeffs_to_copy = min(num_source_coeffs_per_color, expected_sh_coeffs_count_per_tet)
            target_sh_coeffs[:, :coeffs_to_copy, :] = source_sh_coeffs_flat_rgb[:, :coeffs_to_copy, :]
            
            tet_sh_coeffs_np = target_sh_coeffs.reshape(num_tets * expected_sh_coeffs_count_per_tet, 3).cpu().numpy()
            actual_num_sh_coeffs_per_tet = expected_sh_coeffs_count_per_tet # We are conforming to this

            # Densities: model.tet_rgb_param is (num_tets, 4+others) where 4th channel is density (s)
            if hasattr(nerf_model, 'tet_rgb_param') and nerf_model.tet_rgb_param is not None:
                tet_densities_torch = nerf_model.tet_rgb_param[:, 3]
                tet_densities_np = tet_densities_torch.cpu().numpy()
                
                # Base tet colors: model.tet_rgb_param is (num_tets, 4+others) where first 3 are RGB
                base_tet_colors_torch = nerf_model.tet_rgb_param[:, :3] # (num_tets, 3)
                # Duplicate for each of the 4 vertices of a tet
                tet_vertex_colors_flat_np = base_tet_colors_torch.unsqueeze(1).expand(-1, 4, -1).reshape(num_tets * 4, 3).cpu().numpy()
            else: # Fallback if tet_rgb_param somehow not there
                print("Warning: TetColorModel missing tet_rgb_param for density/color.")
                tet_densities_np = np.random.rand(num_tets).astype(np.float32) * 1.0
                tet_vertex_colors_flat_np = np.random.rand(num_tets * 4, 3).astype(np.float32)

        elif hasattr(nerf_model, 'compute_batch_features'): # Likely IngpColorModel
            print("Extracting data from IngpColorModel compatible model.")
            # IngpColorModel.compute_batch_features returns:
            # circumcenter, normalized, density, rgb, grd, sh
            # We need to call this for all tetrahedra.
            # The `sh` from iNGP is (N, sh_dim), where sh_dim = ((1+max_sh_deg)**2-1)*3
            # This usually means L1 to L_max. L0 (DC) is often separate or part of 'rgb'.
            # The shader expects 16 float3s (L0 to L3).
            
            # We need to iterate or process all tets. `model.indices` is available.
            # Let's assume `nerf_model.vertices` and `nerf_model.indices` are what `compute_batch_features` uses internally.
            all_sh_coeffs = []
            all_densities = []
            all_base_colors_rgb = [] # Per-tet base color

            # Max SH degree from the iNGP model
            ingp_max_sh_deg = getattr(nerf_model, 'max_sh_deg', 0) # default to 0 if not found
            
            # Number of SH coeffs from iNGP model (L1 up to L_max_sh_deg)
            # If sh_dim = ((1+max_sh_deg)**2-1)*3, this is for (max_sh_deg - 0) degrees, i.e. max_sh_deg bands (L1..L_max)
            # L0 is 1 coeff, L1 is 3, L2 is 5, L3 is 7.
            # For max_sh_deg=0 (L0 only): iNGP sh_dim is 0. DC is in rgb.
            # For max_sh_deg=1 (L0,L1): iNGP sh_dim is ( (1+1)^2 -1 ) * 3 = 3*3 = 9. (3 coeffs for L1 * 3 colors)
            # For max_sh_deg=2 (L0,L1,L2): iNGP sh_dim is ( (1+2)^2 -1 ) * 3 = 8*3 = 24. (3 L1 + 5 L2 = 8 coeffs * 3 colors)
            # Slang shader wants 16 coeffs (L0..L3). L0=1, L1=3, L2=5, L3=7. Total 1+3+5+7 = 16.

            chunk_size = getattr(nerf_model, 'chunk_size', 408576)
            for i in range(0, num_tets, chunk_size):
                end = min(i + chunk_size, num_tets)
                # Note: compute_batch_features uses `vertices` and `indices` that are attributes of `nerf_model` itself.
                # The `slang_mesh_model.vertices` might be different if contractions were involved differently.
                # For now, assume `nerf_model.vertices` and `nerf_model.indices` are the source for features.
                # The `slang_mesh_model.indices` should be a subset or identical to `nerf_model.indices` if no topology changes.
                # This part is tricky if the `slang_mesh_model` (for rendering V/I) is decoupled from `nerf_model`'s internal geometry.
                # Assuming `slang_mesh_model.indices` refers to indices into `nerf_model.vertices`.
                
                # We need to select the subset of indices for the current chunk based on slang_mesh_model.indices
                current_indices_chunk = slang_mesh_model.indices[i:end]

                # Call compute_batch_features on the nerf_model, using its own full vertex set,
                # but providing the specific indices for the current chunk of tetrahedra we are processing for the slang_mesh_model.
                # The `pre_calc_cell_values` inside `compute_batch_features` will use `vertices[indices[start:end]]`.
                # So `vertices` should be `nerf_model.vertices` and `indices` should be `slang_mesh_model.indices`
                # This requires `compute_batch_features` to be flexible or called carefully.
                # For now, let's assume `nerf_model.compute_batch_features` is called for indices from `slang_mesh_model`
                
                # The `compute_batch_features` in `ingp_color.py` takes (self, vertices, indices, start, end, circumcenters=None)
                # `vertices` should be `nerf_model.vertices`. `indices` should be `slang_mesh_model.indices`.
                # `start` and `end` refer to rows in the passed `indices` tensor.
                
                # This means we pass `nerf_model.vertices` and `slang_mesh_model.indices` to it for the whole set,
                # and it internally chunks. Or we chunk `slang_mesh_model.indices` and pass chunks.
                # The existing `compute_batch_features` expects to be called with global `start, end` for its *own* `self.indices`.
                # This is problematic if `slang_mesh_model.indices` is different.

                # Let's simplify: assume we get all features for all tets in `slang_mesh_model.indices`
                # This might require a temporary modification or a specific getter in IngpColorModel.
                # For now, this is a conceptual placeholder for how one *would* get features from iNGP.
                # A direct call to `nerf_model.get_cell_values(camera, mask_for_all_slang_mesh_tets)` might be better
                # if `mask_for_all_slang_mesh_tets` can be constructed.

                # Placeholder: Simulating feature extraction for all tets in one go (not chunked here)
                # This part needs robust implementation based on IngpColorModel's API
                print(f"  Conceptual: Calling feature computation for {slang_mesh_model.indices.shape[0]} tets for iNGP.")
                # _, batch_features = nerf_model.get_cell_values(camera) # This returns dvrgbs after activate_output
                # We need raw sh, density, rgb before activate_output.

                # Let's try to reconstruct what we need based on `compute_batch_features` logic
                # This assumes `slang_mesh_model.indices` are valid for `nerf_model.vertices`
                temp_all_sh = []
                temp_all_density = []
                temp_all_rgb = [] # This is the DC / base color from iNGP
                
                for chunk_start in range(0, num_tets, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_tets)
                    # `compute_batch_features` needs (self, vertices, indices, start, end)
                    # where `vertices` are the full set from `nerf_model`
                    # `indices` are the tet indices from `slang_mesh_model`
                    # `start`, `end` are slice indices for `slang_mesh_model.indices`
                    _, _, density_chunk, rgb_chunk, _, sh_chunk = \
                        nerf_model.compute_batch_features(nerf_model.vertices, slang_mesh_model.indices, chunk_start, chunk_end)
                    
                    temp_all_sh.append(sh_chunk)
                    temp_all_density.append(density_chunk.squeeze(-1)) # density is (N,1)
                    temp_all_rgb.append(rgb_chunk)

                sh_coeffs_ingp = torch.cat(temp_all_sh, dim=0) # (num_tets, sh_dim_ingp)
                tet_densities_torch = torch.cat(temp_all_density, dim=0) # (num_tets,)
                base_tet_colors_torch = torch.cat(temp_all_rgb, dim=0) # (num_tets, 3)
                
                tet_densities_np = tet_densities_torch.cpu().numpy()

                # Base tet colors (from iNGP `rgb` which is like DC)
                # Duplicate for each of the 4 vertices of a tet
                tet_vertex_colors_flat_np = base_tet_colors_torch.unsqueeze(1).expand(-1, 4, -1).reshape(num_tets * 4, 3).cpu().numpy()

                # SH Coeffs from iNGP
                # sh_coeffs_ingp is (num_tets, sh_dim_raw)
                # sh_dim_raw = ((1+ingp_max_sh_deg)**2-1)*3 for L1..L_max_sh_deg components, each for R,G,B
                # So it's (num_tets, num_higher_order_coeffs * 3)
                # e.g., max_sh_deg=2 -> L1,L2 -> (3+5)=8 coeffs -> sh_dim_raw = 8*3 = 24
                
                # Reshape to (num_tets, num_higher_order_coeffs, 3)
                num_colors = 3
                num_higher_order_coeffs = sh_coeffs_ingp.shape[1] // num_colors
                sh_coeffs_ingp_reshaped = sh_coeffs_ingp.reshape(num_tets, num_higher_order_coeffs, num_colors)

                # Target wants L0...L3 (16 coeffs per color component)
                # L0 is DC (base_tet_colors_torch)
                # L1..L_max_sh_deg are in sh_coeffs_ingp_reshaped
                
                target_sh_coeffs = torch.zeros((num_tets, expected_sh_coeffs_count_per_tet, 3), dtype=torch.float32, device=device)
                
                # Place L0 (DC term) - already have as base_tet_colors_torch
                target_sh_coeffs[:, 0, :] = base_tet_colors_torch 
                
                # Place L1 and higher (if any, and if shader expects them)
                # `expected_sh_coeffs_count_per_tet` includes L0. So L1 starts at index 1.
                # `num_higher_order_coeffs` is for L1..L_max_sh_deg from iNGP
                
                coeffs_to_copy_higher_order = min(num_higher_order_coeffs, expected_sh_coeffs_count_per_tet - 1)
                if coeffs_to_copy_higher_order > 0:
                    target_sh_coeffs[:, 1 : 1+coeffs_to_copy_higher_order, :] = \
                        sh_coeffs_ingp_reshaped[:, :coeffs_to_copy_higher_order, :]
                
                tet_sh_coeffs_np = target_sh_coeffs.reshape(num_tets * expected_sh_coeffs_count_per_tet, 3).cpu().numpy()
                actual_num_sh_coeffs_per_tet = expected_sh_coeffs_count_per_tet # We are conforming
            
            density_scale_from_model = getattr(nerf_model, 'scene_scaling', 1.0) # Use scene_scaling as density_scale

        else:
            print(f"Warning: Unknown NeRF model type for data extraction: {type(nerf_model).__name__}. Using dummy data.")
            # Fallback to dummy data if model type is unrecognized
            actual_num_sh_coeffs_per_tet = expected_sh_coeffs_count_per_tet
            tet_sh_coeffs_np = np.random.rand(num_tets * actual_num_sh_coeffs_per_tet, 3).astype(np.float32).flatten().reshape(-1,3)
            tet_densities_np = np.random.rand(num_tets).astype(np.float32) * 1.0
            tet_vertex_colors_flat_np = np.random.rand(num_tets * 4, 3).astype(np.float32)


    if num_tets > 0:
        max_density_val = np.max(tet_densities_np) if tet_densities_np.size > 0 else 1.0
    else: # if num_tets is 0
        max_density_val = 1.0
        # Ensure buffers are empty but correctly shaped if num_tets is 0
        tet_sh_coeffs_np = np.zeros((0, 3), dtype=np.float32)
        tet_densities_np = np.zeros(0, dtype=np.float32)
        tet_vertex_colors_flat_np = np.zeros((0, 3), dtype=np.float32)
        actual_num_sh_coeffs_per_tet = expected_sh_coeffs_count_per_tet


    # Scene transform (scale, translation, rotation)
    # For simplicity, using scene_scaling from the NeRF model if available, else input arg.
    # This should form a 4x4 matrix.
    # The slang shaders seem to apply model-to-world via viewProjection * scene.transform
    # If nerf_model.center and nerf_model.scene_scaling define the transform to a normalized space,
    # then the inverse of that is the scene_transform.
    # For now, let's use a simple identity or a single scale factor.
    # The C++ code had a 'transform' in its Scene.
    # The shaders might expect vertices to be in a local model space, and then a scene transform.
    # Or vertices are world, and scene transform is identity.

    # If nerf_model has center and scene_scaling, it implies vertices are already in a kind of world space.
    # Let's assume the scene_transform is primarily for scaling if needed.
    # The existing `evaluate_vulkan.py` doesn't seem to pass a separate scene transform to its original `render`
    
    # Use the scene_scaling passed as argument to this function for the transform matrix for now.
    # This was the original behavior of the dummy function.
    transform_matrix = np.identity(4, dtype=np.float32)
    # If vertices are already scaled, this might be 1.0
    # If slang_mesh_model.vertices are in a unit box, then scene_scaling makes sense.
    # Let's assume slang_mesh_model.vertices are in the final world scale.
    # So, scene_transform here can be identity if VP matrix handles all.
    # However, the original dummy used scene_scaling. Let's keep it for now if it affects ray_origin_scene_np.
    # The ray_origin_scene_np calculation uses inv(scene_transform).
    # If vertices are world, scene_transform is identity. ray_origin_scene == ray_origin_world.
    # Let's make scene_transform identity for now, assuming vertices are already world.
    # This means scene_scaling parameter to this function is not used for matrix here.
    
    # Revisit: The C++ code has scene.transform which is `modelToWorld`.
    # And `rayOrigin = (float3)(worldToScene * float4(renderContext.camera.position, 1));`
    # So, if slang_mesh_model.vertices are in local model space, we need a modelToWorld transform.
    # If they are world space, modelToWorld is Identity.
    # Let's assume `slang_mesh_model.vertices` are in final world coordinates.
    # Then scene_transform is identity.
    transform_matrix = np.identity(4, dtype=np.float32)


    # Scene AABB (min and max vertex positions) - shaders might use this
    # Use vertices from slang_mesh_model as they are the ones being rendered directly.
    if num_vertices > 0:
        # Ensure vertices are on CPU for numpy operations
        vertices_for_aabb_np = slang_mesh_model.vertices.cpu().numpy()
        min_vertex = np.min(vertices_for_aabb_np, axis=0)
        max_vertex = np.max(vertices_for_aabb_np, axis=0)
    else:
        min_vertex = np.array([0,0,0], dtype=np.float32)
        max_vertex = np.array([0,0,0], dtype=np.float32)

    return {
        "tet_sh_coeffs": tet_sh_coeffs_np.flatten(), # Shader expects flat buffer
        "num_sh_coeffs_per_tet": actual_num_sh_coeffs_per_tet, # int (coeffs per color component, e.g. 16 for L0-L3)
        "tet_densities": tet_densities_np,           # float per tet
        "tet_vertex_colors_flat": tet_vertex_colors_flat_np.flatten(), # Shader expects flat buffer of float3
        "scene_transform": transform_matrix,      # 4x4 NumPy float32
        "min_vertex": min_vertex,                 # float3
        "max_vertex": max_vertex,                 # float3
        "max_density": float(max_density_val), # float
        "density_scale_from_model": float(density_scale_from_model) 
    }


def render_with_slangpy(
    camera: Camera,
    nerf_model: torch.nn.Module, # The actual NeRF model (e.g. TetColorModel, IngpColorModel)
    slang_mesh_model: Model,     # The SlangModel instance (vertices, indices, min_t)
    slang_device: spy.Device,
    bg_color_rgb=(0.0, 0.0, 0.0), # Background color
    scene_scaling=1.0,           # Overall scene scaling factor (passed to extract_render_data)
    density_threshold_factor=0.01, 
    mesh_percent_tets_to_draw=1.0,
    expected_sh_coeffs_for_shader: int = 16 # L0-L3 for shaders
):
    if not slang_device:
        print("slangpy device not available. Aborting render.")
        return None

    # --- 1. Extract Data & Prepare Basic Params ---
    # Use slang_mesh_model for basic geometry (vertices, indices, min_t)
    # Ensure data is on CPU and in NumPy format for buffer creation if not already.
    vertices_torch = slang_mesh_model.vertices
    indices_torch = slang_mesh_model.indices

    vertices_np = vertices_torch.cpu().numpy() if isinstance(vertices_torch, torch.Tensor) else np.asarray(vertices_torch)
    indices_np = indices_torch.cpu().numpy() if isinstance(indices_torch, torch.Tensor) else np.asarray(indices_torch)
    
    num_tets = indices_np.shape[0]
    num_vertices = vertices_np.shape[0]

    if num_tets == 0:
        print("No tetrahedra to render.")
        img_h, img_w = camera.image_height, camera.image_width
        final_image_rgb_chw = np.full((3, img_h, img_w), bg_color_rgb[0], dtype=np.float32) # Fill with R, G, B if tuple
        if len(bg_color_rgb) == 3:
             final_image_rgb_chw[0,:,:] = bg_color_rgb[0]
             final_image_rgb_chw[1,:,:] = bg_color_rgb[1]
             final_image_rgb_chw[2,:,:] = bg_color_rgb[2]
        final_image_alpha = np.ones((img_h, img_w), dtype=np.float32) 
        return {"render": final_image_rgb_chw, "alpha": final_image_alpha}

    # Pass the original nerf_model to extract detailed attributes like SH, density
    extracted_data = extract_render_data_from_model(
        nerf_model, slang_mesh_model, camera, scene_scaling, 
        expected_sh_coeffs_count_per_tet=expected_sh_coeffs_for_shader
    )

    # Camera matrices
    fx = fov2focal(camera.fovx, camera.image_width)
    fy = fov2focal(camera.fovy, camera.image_height)
    
    znear = slang_mesh_model.min_t # Use min_t from the SlangModel
    zfar = 1000.0 

    # View matrix (world_to_view)
    # camera.world_view_transform is view_to_world (camera pose)
    view_matrix_np = np.linalg.inv(camera.world_view_transform.cpu().numpy())
    
    # Pre-multiply for view-projection
    view_projection_np = np.array([
        [fx / (camera.image_width / 2), 0, 0, 0],
        [0, fy / (camera.image_height / 2), 0, 0], # Y is often flipped for Vulkan by pre-multiplying view or making height negative
        [0, 0, zfar / (zfar - znear), -(zfar * znear) / (zfar - znear)],
        [0, 0, 1, 0]
    ], dtype=np.float32) @ view_matrix_np
    
    # Ray origin in world space
    cam_pos_world_np = camera.camera_center.cpu().numpy()
    
    # Ray origin in scene/model space (if scene has its own transform)
    # ray_origin_scene_np = np.linalg.inv(extracted_data["scene_transform"]) @ np.append(cam_pos_world_np, 1)
    # ray_origin_scene_np = ray_origin_scene_np[:3]
    # For now, assume ray_origin is needed in world space by shaders if scene_transform is applied via view_projection
    # The C++ code calculates rayOrigin in scene space.
    # rayOrigin = (float3)(worldToScene * float4(renderContext.camera.position, 1));
    # Let's follow that:
    world_to_scene_np = np.linalg.inv(extracted_data["scene_transform"])
    ray_origin_scene_np = (world_to_scene_np @ np.append(cam_pos_world_np, 1.0))[:3].astype(np.float32)


    # --- 2. Load Slang Programs ---
    # Defines are now handled at device initialization via compiler_options
    try:
        # Helper to load a program
        def load_slang_program(slang_device_ref: spy.Device, shader_path_str: str, entry_points=None):
            # `shader_path_str` should be a string path to a single .slang file.
            # `entry_points` is a string for compute (kernel name)
            # or a list of {'stage': spy.ShaderStage.XXX, 'name': 'yyy'} for graphics/mesh,
            # or a list of strings for multiple entry point names from the same file.

            if isinstance(entry_points, str): # Compute shader, single entry point name
                 # device.load_program expects a list of entry point names as the second positional arg
                 return slang_device_ref.load_program(shader_path_str, [entry_points])
            elif isinstance(entry_points, list):
                # Check if it's a list of strings (multiple entry points for a program)
                # or list of dicts (stages for a graphics/mesh program)
                if all(isinstance(ep, str) for ep in entry_points):
                    # This case might be for loading a program with multiple, non-staged entry points
                    # if slangpy supports it directly, or if it's meant for compute.
                    # For now, assume it's a list of names for load_program.
                    return slang_device_ref.load_program(shader_path_str, entry_points)
                elif all(isinstance(ep, dict) for ep in entry_points): # Graphics/Mesh shader
                    # Extract names for device.load_program
                    entry_point_names = [ep_info['name'] for ep_info in entry_points]
                    # The stages are used when creating the pipeline, not directly in load_program if it
                    # just returns a general Program object. However, the example used names.
                    # The Program object itself will contain these entry points.
                    return slang_device_ref.load_program(shader_path_str, entry_point_names)
                else:
                    raise ValueError("Invalid format for entry_points list in load_slang_program")
            else: # Should not happen if called correctly
                 raise ValueError("Invalid entry_points argument for load_slang_program (must be str or list)")

        # Pass the slang_device to the helper
        # For compute shaders, load_program returns a Program, which is then used to create_compute_pipeline
        gen_spheres_program = load_slang_program(slang_device, str(GEN_SPHERES_SLANG), entry_points="main")
        
        tet_sort_create_pairs_program = load_slang_program(slang_device, str(TET_SORT_SLANG), entry_points="createPairs")
        tet_sort_update_pairs_program = load_slang_program(slang_device, str(TET_SORT_SLANG), entry_points="updatePairs")
        
        eval_sh_program = load_slang_program(slang_device, str(EVAL_SH_SLANG), ["main"])
        
        mesh_renderer_program = load_slang_program(
            slang_device,
            str(MESH_RENDERER_SLANG), 
            entry_points=[
                {'stage': spy.ShaderStage.vertex, 'name': 'vsmain'},
                {'stage': spy.ShaderStage.fragment, 'name': 'fsmain'}
            ]
        )
        invert_alpha_program = load_slang_program(slang_device, str(INVERT_ALPHA_SLANG), ["main"])

    except Exception as e:
        print(f"Error loading Slang shaders: {e}")
        raise
        return None

    # --- 3. Create Buffers ---
    # Inputs
    vertex_buffer = slang_device.create_buffer(data=vertices_np, usage=spy.BufferUsage.shader_resource)
    # tetrahedron_indices_buffer requires uint4, ensure numpy array is compatible
    indices_buffer = slang_device.create_buffer(data=indices_np.astype(np.uint32), usage=spy.BufferUsage.shader_resource)
    
    tet_sh_coeffs_buffer = slang_device.create_buffer(data=extracted_data["tet_sh_coeffs"], usage=spy.BufferUsage.shader_resource)
    tet_densities_buffer = slang_device.create_buffer(data=extracted_data["tet_densities"], usage=spy.BufferUsage.shader_resource)
    tet_vertex_colors_buffer = slang_device.create_buffer(data=extracted_data["tet_vertex_colors_flat"], usage=spy.BufferUsage.shader_resource)

    # Intermediates
    tet_circumspheres_buffer = slang_device.create_buffer(size=num_tets * 4 * 4, usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access) # float4 per tet
    sort_pairs_buffer = slang_device.create_buffer(
        size=num_tets * 2 * 4, 
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
    ) # uint2 per tet
    evaluated_tet_colors_buffer = slang_device.create_buffer(size=num_tets * 4 * 4, usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access) # float4 per tet

    # Output
    output_image_format = spy.Format.rgba32_float # Matches C++ intermediate target for blending
    output_texture = slang_device.create_texture(
        format=output_image_format,
        width=camera.image_width,
        height=camera.image_height,
        usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_source
    )

    # --- 4. Create Pipelines ---
    # Compute Pipelines
    # Note: ParameterBlock setup is crucial here. The shaders expect `scene` as a parameter block.
    # `slangpy` handles this by allowing you to pass a dict for the parameter block.
    
    gen_spheres_pipeline = slang_device.create_compute_pipeline(program=gen_spheres_program)
    tet_sort_create_pairs_pipeline = slang_device.create_compute_pipeline(program=tet_sort_create_pairs_program)
    tet_sort_update_pairs_pipeline = slang_device.create_compute_pipeline(program=tet_sort_update_pairs_program)
    eval_sh_pipeline = slang_device.create_compute_pipeline(program=eval_sh_program)
    invert_alpha_pipeline = slang_device.create_compute_pipeline(program=invert_alpha_program)

    # Render Pipeline (Mesh Shader)

    # Define color aspect blend descriptor
    color_aspect_blend_params = {
        'src_factor': spy.BlendFactor.dest_alpha,
        'dst_factor': spy.BlendFactor.one,
        'op': spy.BlendOp.add
    }
    color_aspect_desc = spy.AspectBlendDesc(color_aspect_blend_params)

    # Define alpha aspect blend descriptor
    alpha_aspect_blend_params = {
        'src_factor': spy.BlendFactor.dest_alpha,
        'dst_factor': spy.BlendFactor.zero,
        'op': spy.BlendOp.add
    }
    alpha_aspect_desc = spy.AspectBlendDesc(alpha_aspect_blend_params)

    # Define the color target descriptor
    color_target_params = {
        'format': output_image_format, # spy.Format.rgba32_float
        'color': color_aspect_desc,
        'alpha': alpha_aspect_desc,
        'enable_blend': True,
        'logic_op': spy.LogicOp.no_op, # from C++ header default
        'write_mask': spy.RenderTargetWriteMask.enable_all
    }
    color_target_desc_object = spy.ColorTargetDesc(color_target_params)
    
    mesh_render_pipeline = slang_device.create_render_pipeline(
        program=mesh_renderer_program,
        input_layout=None, # Explicitly None as per user's change and for mesh shaders
        primitive_topology=spy.PrimitiveTopology.triangle_list, # Default, may need adjustment for mesh shaders if applicable
        targets=[color_target_desc_object], # Pass the actual ColorTargetDesc object
        depth_stencil=spy.DepthStencilDesc({'depth_test_enable': False, 'depth_write_enable': False}),
        rasterizer=spy.RasterizerDesc({'cull_mode': spy.CullMode.back}),
        multisample=None # Explicitly None
    )
    
    # --- 5. Parameter Block Setup (ShaderParameter in C++) ---
    # This needs to match `TetrahedronScene.slang`'s `ParameterBlock<TetrahedronScene> scene;`
    # All buffers that are part of this structure need to be wrapped correctly.
    scene_params_dict = {
        "vertices": vertex_buffer,
        "tetSH": tet_sh_coeffs_buffer,
        "tetDensities": tet_densities_buffer, # Or as TexelBufferView if format conversion used
        "tetIndices": indices_buffer,
        "tetColors": tet_vertex_colors_buffer, # Base colors for `load_tet_vertex_color`
        "tetCircumspheres": tet_circumspheres_buffer, # Output of GenSpheres, input to Sort
        # Uniforms
        "aabbMin": extracted_data["min_vertex"],
        "aabbMax": extracted_data["max_vertex"],
        "densityScale": extracted_data["density_scale_from_model"], # This was `densityScale` in C++ struct
        "numTets": np.uint32(num_tets),
        "numVertices": np.uint32(num_vertices),
        "numTetSHCoeffs": np.uint32(extracted_data["num_sh_coeffs_per_tet"])
        # "transform": extracted_data["scene_transform"] # If scene transform is passed directly
    }

    # Buffer for evaluated colors from SH pass (output of eval_sh, input to mesh render)
    # Renaming for clarity from `evaluated_tet_colors_buffer`
    vertex_colors_after_sh_eval_buffer = slang_device.create_buffer(
        size=num_tets * 4 * 4, 
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    ) # float4 per tet vertex


    # --- 6. Command Encoding and Execution ---
    command_encoder = slang_device.create_command_encoder()
    with command_encoder.begin_compute_pass() as pass_encoder:

        # a. GenSpheres
        # Bind the pipeline directly to the command encoder, get a shader object
        gen_spheres_shader_object = pass_encoder.bind_pipeline(gen_spheres_pipeline)
        processor = spy.ShaderCursor(gen_spheres_shader_object)
        processor['scene'] = scene_params_dict
        pass_encoder.dispatch([(num_tets + 63) // 64, 1, 1]) # Assuming numthreads(64,1,1)

        tet_sort_create_shader_object = pass_encoder.bind_pipeline(tet_sort_create_pairs_pipeline)
        processor = spy.ShaderCursor(tet_sort_create_shader_object)
        processor['scene'] = scene_params_dict
        processor['sortPairs'] = sort_pairs_buffer
        pass_encoder.dispatch([(num_tets + 63) // 64, 1, 1]) # Assuming numthreads(64,1,1)

        # d. Evaluate SH Colors
        eval_sh_shader_object = pass_encoder.bind_pipeline(eval_sh_pipeline)
        processor = spy.ShaderCursor(eval_sh_shader_object)
        processor['scene'] = scene_params_dict
        processor['tetColors'] = vertex_colors_after_sh_eval_buffer
        processor['rayOrigin'] = ray_origin_scene_np
        pass_encoder.dispatch([(num_tets + 31) // 32, 1, 1])


    slang_device.submit_command_buffer(command_encoder.finish())
    print("Sorting")

    command_encoder = slang_device.create_command_encoder()
    with command_encoder.begin_compute_pass() as pass_encoder:

        # b. Sort Key Generation (updatePairs)
        tet_sort_update_pairs_shader_object = pass_encoder.bind_pipeline(tet_sort_update_pairs_pipeline)
        processor = spy.ShaderCursor(tet_sort_update_pairs_shader_object)
        processor["scene"] = scene_params_dict
        processor["sortPairs"] = sort_pairs_buffer
        processor["rayOrigin"] = ray_origin_scene_np
        pass_encoder.dispatch([(num_tets + 63) // 64, 1, 1])
    slang_device.submit_command_buffer(command_encoder.finish())
    print("Sorted")

    sort_pairs_np = sort_pairs_buffer.to_numpy().view(np.uint32).reshape((num_tets, 2))
    print(sort_pairs_np)
    
    # Sort by the key (first element of uint2)
    sorted_indices_cpu = np.argsort(sort_pairs_np[:, 0])
    sorted_sort_pairs_np = sort_pairs_np[sorted_indices_cpu]

    # Upload sorted data back (or use sorted indices in shaders if they support indirect lookup)
    # The C++ code sorts `sortPairs` in place using RadixSort. Here we re-upload.
    # TODO: Ensure slangpy allows updating buffer content this way or re-create buffer
    # For now, create a new buffer with sorted data
    del sort_pairs_buffer # Release old one
    sorted_sort_pairs_buffer = slang_device.create_buffer(data=sorted_sort_pairs_np, usage=spy.BufferUsage.shader_resource)
    print("created buffer. starting render pass")

    with command_encoder.begin_render_pass(
        {"color_attachments": [
                {"view": output_texture.create_view({}),
                'load_op': spy.LoadOp.clear, 
                'store_op': spy.StoreOp.store,
                'clear_value': [bg_color_rgb[0], bg_color_rgb[1], bg_color_rgb[2], 1.0] # Clear alpha to 1.0 for transmittance accumulation
        }]}) as pass_encoder:
        mesh_render_shader_object = pass_encoder.bind_pipeline(mesh_render_pipeline)
        processor = spy.ShaderCursor(mesh_render_shader_object)
        processor["scene"] = scene_params_dict
        processor["sortBuffer"] = sorted_sort_pairs_buffer
        processor["rayOrigin"] = ray_origin_scene_np
        processor['tetColors'] = vertex_colors_after_sh_eval_buffer
        processor["viewProjection"] = view_projection_np @ extracted_data["scene_transform"], # Apply scene transform here

        # e. Render Mesh
        pass_encoder.bind_pipeline(mesh_render_pipeline)

        pass_encoder.bind_index_buffer(dummy_index_buffer, spy.Format.r32_uint, 0)
        
        pass_encoder.set_viewport(0, 0, camera.image_width, camera.image_height, 0, 1)
        pass_encoder.set_scissor(0, 0, camera.image_width, camera.image_height)

        # Draw 12 vertices per instance (tet), for num_tets_to_draw instances.
        # VS uses SV_InstanceID to get tet_idx from sortBuffer, and SV_VertexID (0-11) to generate tri verts.
        num_tets_to_draw = int(num_tets * mesh_percent_tets_to_draw)
        pass_encoder.draw_indexed(index_count=12, instance_count=num_tets_to_draw, start_index_location=0, base_vertex_location=0, start_instance_location=0)
        slang_device.submit_command_buffer(pass_encoder.finish())

        # f. Invert Alpha
        # invert_alpha_shader_object = pass_encoder.bind_pipeline(invert_alpha_pipeline)
        # params_invert_alpha = {
        #     "image": output_texture.create_view(format=output_image_format),
        #     "dim": np.array([camera.image_width, camera.image_height], dtype=np.uint32)
        # }
        # invert_alpha_shader_object.set_params(params_invert_alpha)
        # pass_encoder.dispatch_compute((camera.image_width + 7)//8, (camera.image_height + 7)//8, 1)
        #
        # # --- 7. Submit and Retrieve ---
        # slang_device.submit_command_buffer(pass_encoder.finish())
        # slang_device.wait_for_idle()
        #
        # output_texture_data = output_texture.read_data()
        # 
        # # Data is flat, reshape. Format is rgba32_float.
        # image_data_np = np.frombuffer(output_texture_data, dtype=np.float32).reshape((camera.image_height, camera.image_width, 4))
        # 
        # final_image_rgb = image_data_np[:, :, :3]
        # final_image_alpha = image_data_np[:, :, 3]
        #
        # # Convert to CHW for PyTorch if needed by calling code.
        # # render_pkg['render'] expects (3, H, W)
        # final_image_rgb_chw = final_image_rgb.transpose(2,0,1)

    # Clean up slangpy objects if they are not reused
    # (slang_device.close() will do this, or manually del buffers/textures/pipelines)
    
    # Explicitly delete buffers that might be re-created or large and go out of scope
    # to help with resource management if this function is called multiple times.
    del vertex_buffer
    del indices_buffer
    del tet_sh_coeffs_buffer
    del tet_densities_buffer
    del tet_vertex_colors_buffer
    del tet_circumspheres_buffer
    # sort_pairs_buffer was already deleted and replaced by sorted_sort_pairs_buffer
    del sorted_sort_pairs_buffer
    del vertex_colors_after_sh_eval_buffer # formerly evaluated_tet_colors_buffer
    del output_texture

    return {
        "render": final_image_rgb_chw, # RGB, CHW format
        "alpha": final_image_alpha,    # H, W format
        # Add other metrics if computed (e.g., distortion_loss)
        # These would require specific shader outputs and processing.
    }


# --- Example Main (for testing, adapt to your script) ---
if __name__ == '__main__':
    print(f"Shader dir: {SHADER_DIR}")
    # This is a very basic example and needs to be filled out with your project's
    # Camera, Model loading, and main loop.

    slang_dev = None
    try:
        slang_dev = init_slangpy_device()

        if slang_dev:
            # Dummy Camera and Model for illustration
            img_h, img_w = 256, 256
            
            # Create a dummy camera (view_to_world identity, looking down -Z)
            cam_pose_torch = torch.eye(4, dtype=torch.float32)
            cam_center_torch = torch.tensor([0.0, 0.0, 3.0], dtype=torch.float32) # Camera at Z=3 looking at origin

            cam = Camera(
                image_height=img_h, image_width=img_w,
                fovx=np.deg2rad(60), fovy=np.deg2rad(60),
                world_view_transform=cam_pose_torch, # view_to_world
                camera_center=cam_center_torch
            )

            # Dummy model: a single tetrahedron
            verts_torch = torch.tensor([
                [0.0, 1.0, 0.0], [-1.0, -0.5, 0.0], [1.0, -0.5, 0.0], [0.0, -0.5, 1.0] # scaled these down
            ], dtype=torch.float32) * 0.5
            
            # Indices for one tetrahedron
            indices_torch = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

            mod = Model(vertices=verts_torch, indices=indices_torch, min_t_val=0.01)

            print("Attempting to call render_with_slangpy...")
            render_pkg = render_with_slangpy(
                camera=cam,
                nerf_model=mod,
                slang_mesh_model=mod,
                slang_device=slang_dev,
                bg_color_rgb=(0.1, 0.2, 0.3),
                scene_scaling=1.0
            )

            if render_pkg:
                print("render_with_slangpy finished.")
                print("Rendered image shape (RGB CHW):", render_pkg['render'].shape)
                print("Alpha mask shape (HW):", render_pkg['alpha'].shape)
                
                # To display with matplotlib (optional)
                try:
                    import matplotlib.pyplot as plt
                    rgb_hwc = render_pkg['render'].transpose(1,2,0)
                    alpha_hw = render_pkg['alpha']
                    
                    # Clamp values for display if they are hdr
                    rgb_display = np.clip(rgb_hwc, 0, 1)
                    
                    plt.figure(figsize=(10,5))
                    plt.subplot(1,2,1)
                    plt.imshow(rgb_display)
                    plt.title("Rendered RGB")
                    plt.subplot(1,2,2)
                    plt.imshow(alpha_hw, cmap='gray')
                    plt.title("Rendered Alpha (1-T)")
                    plt.show()
                except ImportError:
                    print("matplotlib not installed, skipping image display.")
                except Exception as e:
                    print(f"Error displaying image: {e}")

            else:
                print("render_with_slangpy failed or not fully implemented.")
    
    except Exception as e:
        print(f"An error occurred in the example: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if slang_dev:
            slang_dev.close()
            print("slangpy device closed.")
        else:
            print("Could not initialize slangpy device for example usage.") 
