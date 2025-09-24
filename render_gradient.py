from utils.train_util import render
# from models.vertex_color import Model # Assuming one of the model types will be used
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
from itertools import combinations
from utils.topo_utils import build_tv_struct, tet_volumes
from utils.graphics_utils import l2_normalize_th
import math
from utils.gradient import calculate_gradient

# --- Argument and Model Loading (User's Original Code) ---
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


# Convert RGB to Luminance (Y') using the standard formula (Rec. 709)
# L = 0.2126*R + 0.7152*G + 0.0722*B
original_rgb = model.rgb.to(device)
density = model.density.detach()
luminance = 0.2126 * original_rgb[:, 0] + 0.7152 * original_rgb[:, 1] + 0.0722 * original_rgb[:, 2]
lgrads = calculate_gradient(luminance, model)

grads = calculate_gradient(density, model)

def visualize(grads, name):
    grads = l2_normalize_th(grads)
    grads = grads / torch.quantile(grads.abs().mean(dim=1), 0.99)
    gradient_colors = (grads / math.sqrt(3)).clip(min=-0.5, max=0.5) + 0.5

    print("Step 5: Replacing model colors with gradient visualization.")
    # Backup original colors and assign the new gradient colors to the model for rendering
    original_colors = model.rgb.clone()
    model.rgb.data = gradient_colors
    model.sh.data *= 0


    # --- Rendering (User's Original Code) ---
    print("Rendering gradient visualization")
    with torch.no_grad():
        epath = cam_util.generate_cam_path(train_cameras, 400)
        eimages = []
        for camera in tqdm(epath):
            # The render function will now use model.rgb, which holds our gradient colors
            render_pkg = render(camera, model, tile_size=args.tile_size, tmin=model.min_t)
            image = render_pkg['render']
            image = image.permute(1, 2, 0)
            image = image.detach().cpu().numpy()
            eimages.append(image)

    output_video_path = args.output_path / name
    print(f"Writing video to {output_video_path}")
    mediapy.write_video(output_video_path, eimages, fps=30)

visualize(grads, "rotating_dgradient.mp4")
visualize(lgrads, "rotating_gradient.mp4")
