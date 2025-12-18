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

args = Args()
args.tile_size = 4
args.image_folder = "images_4"
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/traj/")
args.eval = True
args.use_ply = False
args.render_train = False
args.min_t = 0.2
args = Args.from_namespace(args.get_parser().parse_args())

# model.light_offset = -1
train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cpu", eval=args.eval, apply_pcd = False)

args.output_path.mkdir(exist_ok=True, parents=True)

all_cameras = cam_util.generate_cam_path(train_cameras, 400)
# Write all camera intrinsics to cameras.txt
with open(str(args.output_path / "cameras.txt"), "w") as fid:
    fid.write("# Camera list with one line of data per camera:\n")
    fid.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    
    # Use a set to write each unique intrinsic set only once
    written_camera_ids = set()
    for i, cam in enumerate(all_cameras):
        if cam.uid not in written_camera_ids:
            cam.write_intrinsic(fid, i)
            written_camera_ids.add(cam.uid)

# Write all camera extrinsics to images.txt
with open(str(args.output_path / "images.txt"), "w") as fid:
    fid.write("# Image list with two lines of data per image:\n")
    fid.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    fid.write("#   POINTS2D[]\n")

    for i, cam in enumerate(all_cameras):
        cam.write_extrinsic(fid, i)