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

# =================================================================================
# === ARGUMENTS AND SETUP
# =================================================================================

args = Args()
args.tile_size = 4
args.image_folder = "images_4"
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/test/")
args.eval = True
args.use_ply = False
args.render_train = False
args.factor = 1.0  # Shrink factor for erosion step
args.mu_val = -0.53 # Bloat factor for dilation step

# --- New arguments for pulsating animation ---
args.num_dilation_steps = 15 # Number of pre-computed steps for both dilation and erosion
args.num_frames = 240         # Total frames for the output video
args.num_cycles = 4           # Number of pulsation cycles during the camera rotation

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
# === HELPER FUNCTIONS
# =================================================================================

def apply_smoothing_step(vertices_in, indices, factor, padded_neighbors, padded_weights, normalize):
    """Applies a single weighted Laplacian smoothing update (one Taubin sub-step)."""
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
    vertices_out[has_neighbors_mask] += factor * update_vector[has_neighbors_mask]
    return vertices_out

def smoothstep(x):
    """A smooth transition function from 0 to 1 as x goes from 0 to 1."""
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def lerp(a, b, t):
    """Linear interpolation between two tensors."""
    return a * (1.0 - t) + b * t

# =================================================================================
# === MAIN SCRIPT LOGIC
# =================================================================================

# --- 1. One-Time Setup for Smoothing ---
print("Performing one-time setup for smoothing structures...")
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

density = model.density.detach()
v_densities_cpu = density.detach().reshape(-1).cpu()
vertex_max_density = torch.zeros(num_vertices, device='cpu', dtype=v_densities_cpu.dtype)
vertex_max_density.index_reduce_(0, indices_cpu.view(-1), v_densities_cpu.repeat_interleave(4), 'amax', include_self=False)
vertex_max_density_np = vertex_max_density.numpy()

edge_to_weight = {}
for i, neighbors in enumerate(symmetric_adjacency_list):
    for j in neighbors:
        if i < j:
            edge = (i, j)
            v1_pos,v2_pos = vertices_np[i],vertices_np[j]
            distance_sq = ((v1_pos-v2_pos)**2).sum()
            gauss_weight = np.exp(-distance_sq/(2*sigma**2))
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


# --- 2. Pre-compute Dilation and Erosion States ---
print(f"Pre-computing {args.num_dilation_steps} dilation and erosion steps...")
dilated_states = [model.vertices.clone()]
shrunken_states = [model.vertices.clone()]

# Generate dilated (bloating) states
current_v_dilate = model.vertices.clone()
for _ in tqdm(range(args.num_dilation_steps), desc="Pre-computing dilations"):
    current_v_dilate = apply_smoothing_step(
        current_v_dilate, model.indices, args.mu_val, padded_neighbors, padded_weights, True)
    dilated_states.append(current_v_dilate)

# Generate shrunken (eroding) states
current_v_shrink = model.vertices.clone()
for _ in tqdm(range(args.num_dilation_steps), desc="Pre-computing erosions"):
    current_v_shrink = apply_smoothing_step(
        current_v_shrink, model.indices, args.factor, padded_neighbors, padded_weights, False)
    shrunken_states.append(current_v_shrink)

# Combine into a single list: [most shrunken, ..., original, ..., most dilated]
all_states = list(reversed(shrunken_states[1:])) + dilated_states
num_states = len(all_states)
print(f"Total of {num_states} states pre-computed.")


# --- 3. Generate Camera Path and Render Video ---
print(f"Generating camera path for {args.num_frames} frames...")
render_cameras = cam_util.generate_cam_path(train_cameras, args.num_frames)

video_frames = []
for i in tqdm(range(args.num_frames), desc="Rendering video"):
    # --- Wobble Logic ---
    # 1. Calculate a sinusoidal value between -1 and 1 over the course of the video
    progress = i / args.num_frames
    angle = progress * 2 * np.pi * args.num_cycles
    oscillation = np.sin(angle)  # Value from -1 to 1

    # 2. Map this -1 to 1 value to a continuous index in our `all_states` list
    continuous_index = (oscillation + 1) / 2 * (num_states - 1)

    # 3. Find the two states to interpolate between
    floor_index = int(np.floor(continuous_index))
    ceil_index = min(floor_index + 1, num_states - 1)

    # 4. Calculate the interpolation weight using smoothstep for a smoother transition
    interp_weight = continuous_index - floor_index
    smooth_weight = smoothstep(torch.tensor(interp_weight, device=device, dtype=torch.float32))

    # 5. Linearly interpolate between the two vertex states
    v_start = all_states[floor_index]
    v_end = all_states[ceil_index]
    current_vertices = lerp(v_start, v_end, smooth_weight.item())

    # Update model for rendering
    model.interior_vertices = current_vertices

    # Render the frame with the rotating camera
    with torch.no_grad():
        camera = render_cameras[i]
        camera.to(device)
        render_pkg = render(camera, model, tile_size=args.tile_size, tmin=model.min_t)
        video_frames.append(render_pkg['render'].permute(1, 2, 0).detach().cpu().numpy())

# --- 4. Save Video ---
output_filename = args.output_path / f"pulsating_object_{args.num_cycles}cycles.mp4"
print(f"Writing final video to {output_filename}...")
mediapy.write_video(output_filename, video_frames, fps=30)

print(f"Done. Video saved to: {output_filename}")