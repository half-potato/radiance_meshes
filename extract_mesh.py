import torch
from pathlib import Path
from data import loader
from utils.args import Args
from models.ingp_color import Model

import numpy as np
from icecream import ic
from utils import topo_utils
from utils import mesh_util
from delaunay_rasterization.internal.render_err import render_err
from utils.model_util import compute_vertex_colors_from_field, RGB2SH
from utils.safe_math import safe_div

from utils.topo_utils import build_tv_struct, build_adj_matrix
from sh_slang.eval_sh_py import eval_sh
from utils.topo_utils import calculate_circumcenters_torch
import tinyplypy
from itertools import combinations
from utils.topo_utils import tet_volumes

args = Args()
args.tile_size = 4
args.image_folder = "images_4"
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/test/")
args.eval = True
args.use_ply = False
args.contrib_threshold = 0.1
args.density_threshold = 0.5
args.alpha_threshold = 0.5
args.mu_val = -0.11
args.factor = 0.1
args.iterations = 25
args.freeze_features = True
args = Args.from_namespace(args.get_parser().parse_args())

device = torch.device('cuda')
try:
    model = Model.load_ckpt(args.output_path, device, args)
except:
    # if args.freeze_features:
    #     from models.frozen_features import FrozenTetModel, FrozenTetOptimizer
    # else:
    from models.frozen import FrozenTetModel, FrozenTetOptimizer
    model = FrozenTetModel.load_ckpt(args.output_path, device)

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cpu", eval=False)
# eventually, generate a UV map for each mesh
# organize onto a texture map
# then, optimize the texture for each of these maps

# --- 2. Initial setup for smoothing ---
current_vertices = model.vertices.clone()
density = model.density.detach()
v_densities = density.detach().reshape(-1).cpu()
indices_cpu = model.indices.cpu()

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

old_volume = tet_volumes(current_vertices[model.indices])
smoothed_vertices_one_iter = taubin_smooth_hybrid_filtered_ring(
    vertices=current_vertices.cpu(),
    indices=indices_cpu,
    densities=v_densities,
    iterations=args.iterations,  # Key change: only one iteration at a time
    lambda_val=args.factor,
    mu_val=args.mu_val,
    sigma=0.010,
).cuda()
new_volume = tet_volumes(smoothed_vertices_one_iter[model.indices])
ratio = safe_div(old_volume, new_volume)

model.interior_vertices = smoothed_vertices_one_iter.clone()
model.ext_vertices = torch.empty((0, 3), device='cuda')
model.density.data *= ratio.reshape(model.density.data.shape)

cameras = train_cameras

path = args.output_path / "meshes"
path.mkdir(exist_ok=True, parents=True)
n_tets = model.indices.shape[0]
device = device
top1 = torch.zeros(n_tets, device=device)  # highest seen so far
top2 = torch.zeros(n_tets, device=device)  # second-highest seen so far
camera_center = torch.stack([cam.camera_center for cam in cameras], dim=0).mean(dim=0)

for cam in cameras:
    target = cam.original_image.cuda()
    camera_center = cam.camera_center.to(device)

    image_votes, extras = render_err(
        target, cam, model,
        scene_scaling=model.scene_scaling,
        tile_size=args.tile_size,
        lambda_ssim=0
    )

    tc = extras["tet_count"][..., 0]
    max_T = extras["tet_count"][..., 1].float() / 65535
    
    # --- Create a single mask for valid updates ---
    # Mask for tets that have a reasonable number of samples in the current view
    # --- Moments (s-1: sum of T, s1: sum of err, s2: sum of err^2)
    image_T, image_err, image_err1 = image_votes[:, 0], image_votes[:, 1], image_votes[:, 2]
    # total_T_p, image_err, image_err1 = image_votes[:, 3], image_votes[:, 4], image_votes[:, 5]
    _, image_Terr, image_ssim = image_votes[:, 2], image_votes[:, 4], image_votes[:, 5]
    N = tc

    # contrib = (image_T / N.clip(min=1)).reshape(-1)
    # contrib = (image_T / N.clip(min=1)).reshape(-1)
    prev_top1 = top1
    top1 = torch.maximum(prev_top1, max_T)
    top2 = torch.maximum(top2, torch.minimum(prev_top1, max_T))

inds = model.indices
verts = model.vertices
tets = verts[inds]
circumcenters, tet_density, rgb, grd, sh = model.compute_features()
# tet_alpha = model.calc_tet_alpha(mode="min")
# tet_alpha = model.calc_tet_alpha(mode="min") * max_color.clip(min=1)

v0, v1, v2, v3 = verts[inds[:, 0]], verts[inds[:, 1]], verts[inds[:, 2]], verts[inds[:, 3]]
edge_lengths = torch.stack([
    torch.norm(v0 - v1, dim=1), torch.norm(v0 - v2, dim=1), torch.norm(v0 - v3, dim=1),
    torch.norm(v1 - v2, dim=1), torch.norm(v1 - v3, dim=1), torch.norm(v2 - v3, dim=1)
], dim=0)
aspect = edge_lengths.min(dim=0).values / edge_lengths.max(dim=0).values 
vol = topo_utils.tet_volumes(tets).abs()
tet_alpha = 1 - torch.exp(-tet_density.reshape(-1) * vol.reshape(-1)**(1/3) * aspect)

mask = (top1.reshape(-1) > args.contrib_threshold)
owners, face_areas = build_tv_struct(model.vertices, model.indices, device=device)
adj = build_adj_matrix(inds.shape[0], owners).float()
mask = mask & (torch.sparse.mm(adj, mask.float().unsqueeze(1)).squeeze(1) > 1.1)
mask = (mask.float() + torch.sparse.mm(adj, mask.float().unsqueeze(1)).squeeze(1)) > 2
mask = (mask.float() + torch.sparse.mm(adj, mask.float().unsqueeze(1)).squeeze(1)) > 2
# mask = mask & (torch.sparse.mm(adj, mask.float().unsqueeze(1)).squeeze(1) > 2.1)
# mask = mask & (torch.sparse.mm(adj, mask.float().unsqueeze(1)).squeeze(1) > 3.1)

ic(top1.mean(), tet_density.mean(), mask.sum())
# mask = (tet_density > density_threshold) | (tet_alpha > alpha_threshold)

rgb = rgb[mask].detach()
tets = tets[mask]
circumcenters, radius = calculate_circumcenters_torch(tets.double())
grd = grd[mask].detach()
sh = sh[mask].detach()
normed_grd = grd.reshape(-1, 1, 3) * rgb.reshape(-1, 3, 1).mean(dim=1, keepdim=True).detach()
tet_color_raw = eval_sh(
    tets.mean(dim=1).detach(),
    RGB2SH(rgb),
    sh.reshape(-1, (model.max_sh_deg+1)**2 - 1, 3),
    camera_center,
    model.max_sh_deg).float()
tet_color = torch.nn.functional.softplus(tet_color_raw.reshape(-1, 3, 1), beta=10)
vcolors = compute_vertex_colors_from_field(
    tets.detach(), tet_color.reshape(-1, 3), normed_grd.float(), circumcenters.float().detach()).clip(min=0)
# vcolors = torch.nn.functional.softplus(vcolors, beta=10)

# mesh_util.export_textured_meshes(path, 
#     verts=verts.detach().cpu().numpy(),
#     tets=model.indices[mask].cpu().numpy(),
#     tet_v_rgb=vcolors.detach().cpu().numpy(),
#     min_faces_for_export=10000,
#     texture_size=4096*2**0,
# )

# meshes = mesh_util.extract_meshes_per_face_color(
meshes = mesh_util.extract_meshes(
    vcolors.detach().cpu().numpy(),
    verts.detach().cpu().numpy(),
    model.indices[mask].cpu().numpy())
for i, mesh in enumerate(meshes):
    F = mesh['face']['vertex_indices'].shape[0]
    if F > 1000:
        mpath = path / f"{i}.ply"
        print(f"Saving #F:{F} to {mpath}")
        tinyplypy.write_ply(str(mpath), mesh, is_binary=False)
