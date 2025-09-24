import torch
from pathlib import Path
from data import loader
from utils.args import Args
from models.ingp_color import Model

from icecream import ic
from utils import topo_utils
from utils import mesh_util
from delaunay_rasterization.internal.render_err import render_err
from utils.model_util import compute_vertex_colors_from_field, RGB2SH

from utils.topo_utils import build_tv_struct, build_adj_matrix
from sh_slang.eval_sh_py import eval_sh
from utils.topo_utils import calculate_circumcenters_torch
import tinyplypy

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
# mask = (mask.float() + torch.sparse.mm(adj, mask.float().unsqueeze(1)).squeeze(1)) > 0
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
