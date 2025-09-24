from utils.contraction import l2_normalize_th
from utils.safe_math import safe_div
from utils.topo_utils import tet_volumes
from utils.train_util import render
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
from data import loader
from utils.args import Args
from utils import cam_util
import mediapy
from itertools import combinations
from utils.gradient import calculate_gradient
from icecream import ic

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
args.mu_val = -0.53
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

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cpu", eval=args.eval)

# =================================================================================
# === HELPER FUNCTIONS (Taubin Smoothing and PBD Volume Conservation)
# =================================================================================

def apply_smoothing_step(vertices_in, indices, factor, padded_neighbors, padded_weights, normalize):
    """Applies a single weighted Laplacian smoothing update (one Taubin sub-step)."""
    current_volumes = tet_volumes(vertices_in[indices]).reshape(-1)
    vertices_out = vertices_in.clone()
    dtype = vertices_in.dtype
    device = vertices_in.device
    virtual_vertex_coord = torch.zeros((1, 3), device=device, dtype=dtype)
    augmented_vertices = torch.cat([vertices_out, virtual_vertex_coord], dim=0)
    neighbor_coords = augmented_vertices[padded_neighbors]
    weights_broadcast = padded_weights.unsqueeze(-1)
    numerator = (neighbor_coords * weights_broadcast).sum(dim=1)
    denominator = padded_weights.sum(dim=1, keepdim=True)
    centroids = safe_div(numerator, denominator)
    if normalize:
        update_vector = l2_normalize_th(centroids - vertices_out)
    else:
        update_vector = centroids - vertices_out
    has_neighbors_mask = (denominator > 1e-6).squeeze()
    # v_volumes = torch.zeros_like(denominator.reshape(-1))
    # indices = indices.long()
    # v_volumes.scatter_reduce_(0, indices[:, 0], current_volumes, reduce="amax")
    # v_volumes.scatter_reduce_(0, indices[:, 1], current_volumes, reduce="amax")
    # v_volumes.scatter_reduce_(0, indices[:, 2], current_volumes, reduce="amax")
    # v_volumes.scatter_reduce_(0, indices[:, 3], current_volumes, reduce="amax")
    # mask = v_volumes > 0.00001
    vertices_out[has_neighbors_mask] += factor * update_vector[has_neighbors_mask]
    return vertices_out

def tet_volumes_pbd(vertices, indices):
    """Calculates the signed volume of a batch of tetrahedra for PBD."""
    tets = vertices[indices]
    p0, p1, p2, p3 = tets[:, 0, :], tets[:, 1, :], tets[:, 2, :], tets[:, 3, :]
    v1, v2, v3 = p1 - p0, p2 - p0, p3 - p0
    volume = torch.bmm(torch.cross(v2, v3, dim=1).unsqueeze(1), v1.unsqueeze(2)).squeeze() / 6.0
    return volume

# =================================================================================
# === MAIN SCRIPT LOGIC
# =================================================================================

# --- 1. One-Time Setup and Pre-computation ---
print("Performing one-time setup...")
sample_camera = train_cameras[0]
print(f"Using camera {sample_camera.colmap_id} as the fixed viewpoint.")

# Pre-compute structures for Taubin smoothing
initial_vertices_cpu = model.vertices.cpu()
num_vertices = initial_vertices_cpu.shape[0]
indices_cpu = model.indices.cpu()

sigma, dtype = 0.01, initial_vertices_cpu.dtype
indices_np, vertices_np = indices_cpu.numpy(), initial_vertices_cpu.numpy()
adjacency_1ring = [set() for _ in range(num_vertices)]
for i, tet in enumerate(indices_np):
    for v1_idx, v2_idx in combinations(tet, 2):
        adjacency_1ring[v1_idx].add(v2_idx)
        adjacency_1ring[v2_idx].add(v1_idx)

symmetric_adjacency = [set() for _ in range(num_vertices)]
for i, n in enumerate(adjacency_1ring):
    for j in n:
        symmetric_adjacency[i].add(j)
        symmetric_adjacency[j].add(i)
symmetric_adjacency_list = [list(s) for s in symmetric_adjacency]

density = model.density.detach();
grads = calculate_gradient(density,model)
v = torch.linalg.norm(grads,dim=1)
# v_densities_cpu = v.detach().reshape(-1).cpu()
v_densities_cpu = density.detach().reshape(-1).cpu()
vertex_max_density = torch.zeros(num_vertices, device='cpu', dtype=v_densities_cpu.dtype)
vertex_max_density.index_reduce_(0, indices_cpu.view(-1), v_densities_cpu.repeat_interleave(4), 'amax', include_self=False)
vertex_max_density_np = vertex_max_density.numpy()

edge_to_weight = {}
for i, neighbors in enumerate(symmetric_adjacency_list):
    for j in neighbors:
        if i < j:
            edge = (i, j);
            density_weight = max(vertex_max_density_np[i], vertex_max_density_np[j])
            v1_pos,v2_pos = vertices_np[i],vertices_np[j];
            distance_sq = ((v1_pos-v2_pos)**2).sum()
            gauss_weight = np.exp(-distance_sq/(2*sigma**2));
            edge_to_weight[edge] = gauss_weight

max_degree = max(len(s) for s in symmetric_adjacency_list) if num_vertices > 0 else 0
padded_neighbors = torch.full((num_vertices, max_degree), num_vertices, device=device, dtype=torch.long)
padded_weights = torch.zeros((num_vertices, max_degree), device=device, dtype=dtype)
for i, neighbors in enumerate(symmetric_adjacency_list):
    if neighbors:
        padded_neighbors[i, :len(neighbors)] = torch.tensor(neighbors, device=device, dtype=torch.long)
        weights = [edge_to_weight[tuple(sorted((i, n_idx)))] for n_idx in neighbors]
        padded_weights[i, :len(neighbors)] = torch.tensor(weights, device=device, dtype=dtype)
print("Taubin structures cached.")

# Pre-compute rest volumes for PBD
rest_volumes = tet_volumes_pbd(model.vertices.clone(), model.indices)
print("PBD rest volumes cached.")

# --- 2. Iterative Smoothing, Correction, and Rendering Loop ---
iterative_frames = []
current_vertices = model.vertices.clone()

# Render the initial state (Frame 0)
print("Rendering initial state...")
with torch.no_grad():
    render_pkg = render(sample_camera, model, tile_size=args.tile_size, tmin=model.min_t)
    iterative_frames.append(render_pkg['render'].permute(1, 2, 0).detach().cpu().numpy())

# Main loop
for i in tqdm(range(args.smooth_iters), desc="Smoothing + PBD Correction"):
    volume_before_step = tet_volumes_pbd(current_vertices, model.indices)

    # Step 1: Apply one full Taubin smoothing iteration (shrink + bloat)
    vertices_after_shrink = apply_smoothing_step(
        current_vertices, model.indices, args.factor, padded_neighbors, padded_weights, False)
    vertices_after_smooth = apply_smoothing_step(
        vertices_after_shrink, model.indices, args.mu_val, padded_neighbors, padded_weights, True)

    # Update vertices for the next iteration
    # current_vertices = vertices_after_pbd
    current_vertices = vertices_after_smooth

    # Update model for rendering and density correction
    volume_after_step = tet_volumes_pbd(current_vertices, model.indices)
    # We use the overall volume change for a stable density correction
    ratio = safe_div(volume_before_step.sum(), volume_after_step.sum())
    
    model.interior_vertices = current_vertices.clone()
    # model.density.data *= ratio
    
    # Render the final state of this iteration
    with torch.no_grad():
        render_pkg = render(sample_camera, model, tile_size=args.tile_size, tmin=model.min_t)
        iterative_frames.append(render_pkg['render'].permute(1, 2, 0).detach().cpu().numpy())
        
print("Writing final video...")
mediapy.write_video(args.output_path / "smooth_with_pbd_correction.mp4", iterative_frames, fps=3)

print("Done. Video saved to:", args.output_path / "smooth_with_pbd_correction.mp4")
