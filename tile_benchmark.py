from utils.train_util import render
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
from utils.model_util import *
import time

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

indices = model.indices
vertices = model.vertices
circumcenters, density, rgb, grd, sh = model.compute_batch_features(
    vertices, indices)

for split, cameras in splits:
    for idx, camera in enumerate(tqdm(cameras, desc=f"Rendering {split} set")):
        with torch.no_grad():
            dvrgbs = activate_output(camera.camera_center.to(model.device),
                        density, rgb, grd,
                        sh.reshape(-1, (model.max_sh_deg+1)**2 - 1, 3),
                        indices,
                        circumcenters,
                        vertices, model.max_sh_deg, model.max_sh_deg)
            render_pkg = render(camera, model, tile_size=args.tile_size, min_t=args.min_t, cell_values=dvrgbs)

    for idx, camera in enumerate(tqdm(cameras, desc=f"Rendering {split} set")):
        times = []
        with torch.no_grad():
            start_time = time.time()
            dvrgbs = activate_output(camera.camera_center.to(model.device),
                        density, rgb, grd,
                        sh.reshape(-1, (model.max_sh_deg+1)**2 - 1, 3),
                        indices,
                        circumcenters,
                        vertices, model.max_sh_deg, model.max_sh_deg)
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
            render_pkg = render(camera, model, tile_size=args.tile_size, min_t=args.min_t, cell_values=dvrgbs)
            dt = time.time() - start_time
            times.append(dt)
    print("Average FPS:", 1/np.mean(times))