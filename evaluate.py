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
args.tile_size = 16
args.image_folder = "images_4"
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/test/")
args.eval = False
args.use_ply = False
args.render_train = False
args.base_min_t = 0.2
args = Args.from_namespace(args.get_parser().parse_args())

device = torch.device('cuda')
if args.use_ply:
    from models.tet_color import Model
    model = Model.load_ply(args.output_path / "ckpt.ply", device)
else:
    from models.ingp_color import Model
    from models.frozen_features import FrozenTetModel
    try:
        model = Model.load_ckpt(args.output_path, device)
    except:
        model = FrozenTetModel.load_ckpt(args.output_path, device)

# model.light_offset = -1
train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cpu", eval=args.eval)

model.min_t = args.min_t = args.base_min_t * model.scene_scaling.item()

ic(model.min_t)
if args.render_train:
    splits = zip(['train', 'test'], [train_cameras, test_cameras])
else:
    splits = zip(['test'], [test_cameras])
test_util.evaluate_and_save(model, splits, args.output_path, args.tile_size, min_t=model.min_t)
#model.save2ply(Path('test.ply'))

with torch.no_grad():
    epath = cam_util.generate_cam_path(train_cameras, 400)
    eimages = []
    for camera in tqdm(epath):
        render_pkg = render(camera, model, tile_size=args.tile_size, min_t=model.min_t)
        image = render_pkg['render']
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        eimages.append(image)

mediapy.write_video(args.output_path / "rotating.mp4", eimages)
