# from models.vertex_color import Model
import pickle
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
import time

from utils.model_util import *
from utils.topo_utils import build_adj
from delaunay_rasterization.internal.ray_trace import TetrahedralRayTrace, get_degenerate_tet_mask
from dtlookup import lookup_inds, TetrahedraLookup, TetWalkLookup
from utils.train_util import render

args = Args()
args.tile_size = 4
args.image_folder = "images_4"
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/test/")
args.eval = True
args.use_ply = False
args.render_train = False
args.min_t = 0.2
args.resolution = "test"
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

# ic(model.min_t)
# model.min_t = args.min_t
ic(model.min_t)
if args.render_train:
    splits = zip(['train', 'test'], [train_cameras, test_cameras])
else:
    splits = zip(['test'], [test_cameras])

ic(model.empty_indices.shape)
indices = torch.cat([model.indices, model.empty_indices], dim=0)
vertices = model.vertices
circumcenters, density, rgb, grd, sh = model.compute_batch_features(
    vertices, model.indices, start=0, end=model.indices.shape[0])

tet_adj = build_adj(model.vertices, indices, device='cuda')
tet_adj = tet_adj.int()
# lookup = TetWalkLookup(indices.cpu().numpy(), model.vertices.detach().cpu().numpy())
lookup = TetrahedraLookup(indices, vertices, 256)

def render_rt(camera, model, camera_directions, cell_values, tet_adj, min_t, lookup_tool):
    rays = camera.get_world_space_rays(camera_directions, 'cuda')
    # rays = camera.to_rays().cuda()
    cpos = camera.camera_center
    # start_tet_ids = lookup_inds(indices, model.vertices, cpos.reshape(1, 3).cuda())
    # start_tet_ids = lookup_tool.lookup(cpos.reshape(1, 3).numpy())
    start_tet_ids = lookup_tool.lookup(cpos.reshape(1, 3).cuda())
    cell_values_w_empty = torch.cat([
        cell_values,
        torch.zeros((model.empty_indices.shape[0], cell_values.shape[1]), device='cuda')
    ])
    # ic(cell_values_w_empty.shape, model.empty_indices.shape, model.indices.shape, cell_values.shape)
    # print(start_tet_ids)
    output_img, distortion_img = TetrahedralRayTrace.apply(
            rays,
            indices,
            model.vertices,
            cell_values_w_empty,
            tet_adj,
            start_tet_ids * torch.ones((rays.shape[0]), dtype=torch.int, device='cuda'),
            min_t,
            200,
    )
    return output_img[:, :3].reshape(camera.image_height, camera.image_width, 3)

for split, cameras in splits:
    cds = cameras[0].get_camera_space_directions('cuda')
    for idx, camera in enumerate(tqdm(cameras, desc=f"Rendering {split} set")):
        with torch.no_grad():
            dvrgbs = activate_output(camera.camera_center.to(model.device),
                        density, rgb, grd,
                        sh.reshape(-1, (model.max_sh_deg+1)**2 - 1, 3),
                        model.indices,
                        circumcenters,
                        vertices, model.max_sh_deg, model.max_sh_deg)
            render_pkg = render_rt(camera, model, cds, tet_adj=tet_adj, min_t=args.min_t, cell_values=dvrgbs, lookup_tool=lookup)
            # render_pkg = render(camera, model, tile_size=args.tile_size, min_t=args.min_t, cell_values=dvrgbs)['render'][:3].permute(1, 2, 0)
            imageio.imwrite(f'test/{idx:03d}.png', (render_pkg.detach().cpu().numpy()*255).astype(np.uint8))

    dvrgbs = activate_output(camera.camera_center.to(model.device),
                density, rgb, grd,
                sh.reshape(-1, (model.max_sh_deg+1)**2 - 1, 3),
                model.indices,
                circumcenters,
                vertices, model.max_sh_deg, model.max_sh_deg)
    for idx, camera in enumerate(tqdm(cameras, desc=f"Rendering {split} set")):
        times = []
        with torch.no_grad():
            if args.resolution == "1080p":
                camera.image_width = 1920
                camera.image_height = 1080
            elif args.resolution == "2k":
                camera.image_width = 2560
                camera.image_height = 1440
            elif args.resolution == "4k":
                camera.image_width = 3840
                camera.image_height = 2160

            camera.gt_alpha_mask = torch.ones((1, camera.image_height, camera.image_width), device=camera.data_device)
            start_time = time.time()
            render_pkg = render_rt(camera, model, cds, tet_adj=tet_adj, min_t=args.min_t, cell_values=dvrgbs, lookup_tool=lookup)
            # render_pkg = render(camera, model, tile_size=args.tile_size, min_t=args.min_t, cell_values=dvrgbs)
            dt = time.time() - start_time
            times.append(dt)
    print("Average FPS:", 1/np.mean(times))

with torch.no_grad():
    epath = cam_util.generate_cam_path(train_cameras, 400)
    eimages = []
    for camera in tqdm(epath):
        dvrgbs = activate_output(camera.camera_center.to(model.device),
                    density, rgb, grd,
                    sh.reshape(-1, (model.max_sh_deg+1)**2 - 1, 3),
                    model.indices,
                    circumcenters,
                    vertices, model.max_sh_deg, model.max_sh_deg)
        render_pkg = render_rt(camera, model, cds, tet_adj=tet_adj, min_t=args.min_t, cell_values=dvrgbs, lookup_tool=lookup)
        image = render_pkg
        # image = render_pkg['render']
        # image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        eimages.append(image)

mediapy.write_video(args.output_path / "rotating_rt.mp4", eimages)
