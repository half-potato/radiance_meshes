from utils.train_util import render
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
args.tile_size = 16
args.image_folder = "images_4"
args.output_path = Path("output/test/")
args.eval = True
args.use_ply = False
args = Args.from_namespace(args.get_parser().parse_args())

device = torch.device('cuda')
if args.use_ply:
    from models.tet_color import Model
    model = Model.load_ply(args.output_path / "ckpt.ply", device)
else:
    from models.ingp_color import Model
    model = Model.load_ckpt(args.output_path, device)
model.extract_mesh(args.output_path / "0.ply")
# eventually, generate a UV map for each mesh
# organize onto a texture map
# then, optimize the texture for each of these maps