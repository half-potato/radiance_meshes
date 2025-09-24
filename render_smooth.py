from utils.safe_math import safe_div
from utils.topo_utils import tet_volumes
from utils.train_util import render
# from models.vertex_color import Model
import pickle
import math
import torch
from tqdm import tqdm
from pathlib import Path
import imageio
import numpy as np
from data import loader
from utils import test_util
from utils.args import Args
from utils import cam_util
import mediapy
from icecream import ic
from itertools import combinations
from utils.gradient import calculate_gradient
from utils.graphics_utils import l2_normalize_th

args = Args()
args.tile_size = 4
args.image_folder = "images_4"
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/test/")
args.eval = True
args.use_ply = False
args.render_train = False
args.factor = 1.0
args.smooth_iters = 5
args.mu_val=-0.53
args = Args.from_namespace(args.get_parser().parse_args())

device = torch.device('cuda')
if args.use_ply:
    from models.tet_color import Model
    model = Model.load_ply(args.output_path / "ckpt.ply", device)
else:
    from models.ingp_color import Model
    from models.frozen import FrozenTetModel
    try:
        model = Model.load_ckpt(args.output_path, device)
    except:
        model = FrozenTetModel.load_ckpt(args.output_path, device)

# model.light_offset = -1
train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cpu", eval=args.eval)

def taubin_smooth_hybrid_filtered_ring(
    vertices,
    indices,
    densities,
    iterations=1,
    lambda_val=0.5,
    mu_val=-0.53,
    sigma=1.0,
    max_neighbors=200
):
    """
    Applies a sophisticated Taubin smoothing using a geometrically-filtered
    1-ring neighborhood and a hybrid density/Gaussian weighting scheme.
    (Corrected version to handle neighborhood asymmetry).
    """
    device = vertices.device
    dtype = vertices.dtype
    num_vertices = vertices.shape[0]

    smoothed_vertices = vertices.clone()

    # --- Step 1: Geometrically Prune the 1-Ring Neighborhood (creates asymmetry) ---
    indices_cpu = indices.cpu().numpy()
    vertices_cpu = vertices.cpu().numpy()

    adjacency_1ring = [set() for _ in range(num_vertices)]
    for tet in indices_cpu:
        for v1_idx, v2_idx in combinations(tet, 2):
            adjacency_1ring[v1_idx].add(v2_idx)
            adjacency_1ring[v2_idx].add(v1_idx)

    # This list can be asymmetric
    filtered_adjacency_oneway = [[] for _ in range(num_vertices)]
    for i in range(num_vertices):
        neighbors_of_i = list(adjacency_1ring[i])
        if not neighbors_of_i:
            continue

        pos_i = vertices_cpu[i]
        pos_neighbors = vertices_cpu[neighbors_of_i]
        distances_sq = np.sum((pos_neighbors - pos_i)**2, axis=1)
        sorted_indices = np.argsort(distances_sq)
        num_to_keep = min(len(neighbors_of_i), max_neighbors)
        closest_neighbor_indices = [neighbors_of_i[j] for j in sorted_indices[:num_to_keep]]
        filtered_adjacency_oneway[i] = closest_neighbor_indices

    # --- Step 2: Symmetrize the Adjacency List (THE FIX) ---
    # Create a new symmetric list to ensure consistency.
    symmetric_adjacency = [set() for _ in range(num_vertices)]
    for i, neighbors in enumerate(filtered_adjacency_oneway):
        for j in neighbors:
            symmetric_adjacency[i].add(j)
            symmetric_adjacency[j].add(i)
    # Convert back to lists for indexing
    symmetric_adjacency_list = [list(s) for s in symmetric_adjacency]


    # --- Step 3: Build Edges and Weights from the SYMMETRIC Neighborhood ---
    vertex_max_density = torch.zeros(num_vertices, device='cpu', dtype=densities.dtype)
    indices_flat = indices.cpu().view(-1)
    densities_expanded = densities.cpu().repeat_interleave(4)
    vertex_max_density.index_reduce_(0, indices_flat, densities_expanded, 'amax', include_self=False)
    vertex_max_density_np = vertex_max_density.numpy()

    edge_to_weight = {}
    for i, neighbors in enumerate(symmetric_adjacency_list): # Using the symmetric list
        for j in neighbors:
            if i < j:
                edge = (i, j)
                density_weight = max(vertex_max_density_np[i], vertex_max_density_np[j])
                v1_pos, v2_pos = vertices_cpu[i], vertices_cpu[j]
                distance_sq = ((v1_pos - v2_pos)**2).sum()
                gauss_weight = np.exp(-distance_sq / (2 * sigma**2))
                edge_to_weight[edge] = density_weight * gauss_weight

    # --- Step 4: Convert to Padded Tensors for Direct Vectorized Calculation ---
    max_degree = max(len(s) for s in symmetric_adjacency_list) if num_vertices > 0 else 0
    virtual_vertex_idx = num_vertices

    padded_neighbors = torch.full((num_vertices, max_degree), virtual_vertex_idx, device=device, dtype=torch.long)
    padded_weights = torch.zeros((num_vertices, max_degree), device=device, dtype=dtype)

    for i, neighbors in enumerate(symmetric_adjacency_list): # Using the symmetric list
        if neighbors:
            padded_neighbors[i, :len(neighbors)] = torch.tensor(neighbors, device=device, dtype=torch.long)
            # This lookup is now safe because the dictionary was built from the same symmetric graph
            weights = [edge_to_weight[tuple(sorted((i, n_idx)))] for n_idx in neighbors]
            padded_weights[i, :len(neighbors)] = torch.tensor(weights, device=device, dtype=dtype)

    # --- Step 5: Perform Taubin Smoothing Iterations ---
    for _ in range(iterations):
        for factor in [lambda_val, mu_val]:
            virtual_vertex_coord = torch.zeros((1, 3), device=device, dtype=dtype)
            augmented_vertices = torch.cat([smoothed_vertices, virtual_vertex_coord], dim=0)

            neighbor_coords = augmented_vertices[padded_neighbors]
            weights_broadcast = padded_weights.unsqueeze(-1)
            numerator = (neighbor_coords * weights_broadcast).sum(dim=1)
            denominator = padded_weights.sum(dim=1, keepdim=True)

            safe_denominator = torch.where(denominator == 0, 1.0, denominator)
            centroids = numerator / safe_denominator

            update_vector = centroids - smoothed_vertices
            has_neighbors_mask = (denominator > 1e-6).squeeze()
            smoothed_vertices[has_neighbors_mask] += factor * update_vector[has_neighbors_mask]

    return smoothed_vertices

# =================================================================================
# === MODIFIED SECTION: Iterative Smoothing and Rendering from a Fixed Viewpoint ===
# =================================================================================

print("Preparing for iterative smoothing and rendering...")

# --- 1. Select a fixed viewpoint ---
# We'll use the first camera from the training set as our sample viewpoint.
sample_camera = train_cameras[0]
print(f"Using camera {sample_camera.colmap_id} as the fixed viewpoint.")

# --- 2. Initial setup for smoothing ---
density = model.density.detach()
grads = calculate_gradient(density, model)
v = torch.linalg.norm(grads, dim=1)
v_densities = v.detach().reshape(-1).cpu()
indices_cpu = model.indices.cpu()

# --- 3. Iterative smoothing and rendering loop ---
iterative_frames = []
# Start with the original vertices. We'll update this tensor in each iteration.
current_vertices = model.vertices.clone()

# First, render the initial, unsmoothed state (Iteration 0)
print("Rendering initial state (Iteration 0)...")
with torch.no_grad():
    render_pkg = render(sample_camera, model, tile_size=args.tile_size, tmin=model.min_t)
    initial_image = render_pkg['render'].permute(1, 2, 0).detach().cpu().numpy()
    iterative_frames.append(initial_image)

# The main loop for smoothing and rendering each step
for i in tqdm(range(args.smooth_iters), desc="Smoothing and Rendering Iterations"):
    # Calculate the volume of the tetrahedra before this smoothing step
    old_volume = tet_volumes(current_vertices[model.indices])

    # Apply ONE iteration of smoothing to the current vertex positions
    smoothed_vertices_one_iter = taubin_smooth_hybrid_filtered_ring(
        vertices=current_vertices.cpu(),
        indices=indices_cpu,
        densities=v_densities,
        iterations=1,  # Key change: only one iteration at a time
        lambda_val=args.factor,
        mu_val=args.mu_val,
        sigma=0.010,
    ).cuda()

    # Calculate the volume after smoothing to get the scaling ratio
    new_volume = tet_volumes(smoothed_vertices_one_iter[model.indices])
    ratio = safe_div(old_volume, new_volume)

    # Update the model's state to reflect the result of this single smoothing step
    model.interior_vertices = smoothed_vertices_one_iter.clone()
    model.ext_vertices = torch.empty((0, 3), device='cuda')
    # Scale density to compensate for volume changes and preserve appearance
    model.density.data *= ratio.reshape(model.density.data.shape)

    # Update current_vertices for the *next* loop iteration
    current_vertices = smoothed_vertices_one_iter

    # Render the frame for the current iteration
    with torch.no_grad():
        render_pkg = render(sample_camera, model, tile_size=args.tile_size, tmin=model.min_t)
        image = render_pkg['render'].permute(1, 2, 0).detach().cpu().numpy()
        iterative_frames.append(image)

print("Writing iterative smoothing video...")
# We use a low FPS (e.g., 2 frames per second) to make the changes between steps easy to see.
mediapy.write_video(args.output_path / "iterative_smooth.mp4", iterative_frames, fps=2)

print("Done. Video saved to:", args.output_path / "iterative_smooth.mp4")