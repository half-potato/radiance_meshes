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
from utils import camera_path
import mediapy
from icecream import ic
from data.types import ProjectionType

args = Args()
args.tile_size = 4
args.image_folder = "images_4"
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/test/")
args.eval = True
args.use_ply = False
args.render_train = False
args.min_t = 0.2
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
#model.save2ply(Path('test.ply'))

def draw_filled_circle(
    mask_tensor: torch.Tensor,
    radius_percentage: float,
    center_xy: tuple[float, float] | None = None
) -> torch.Tensor:
    """
    Draws a filled circle of 1s on a given mask tensor (in-place).

    Args:
        mask_tensor: The tensor to draw on. Expected shape (1, H, W).
        radius_percentage: The radius as a percentage (0.0 to 1.0) of the
                           image's smaller dimension (so 1.0 fits perfectly).
        center_xy: Optional (x, y) tuple for the circle's center.
                   If None, defaults to the image center.
    Returns:
        The modified mask_tensor (modified in-place).
    """
    # Get mask dimensions and device
    if mask_tensor.dim() != 3 or mask_tensor.shape[0] != 1:
        raise ValueError(f"Expected mask_tensor shape (1, H, W), got {mask_tensor.shape}")
        
    _c, H, W = mask_tensor.shape
    device = mask_tensor.device

    # 1. Determine center coordinates
    if center_xy is None:
        # Default to the center of the image
        center_x = W / 2.0
        center_y = H / 2.0
    else:
        center_x, center_y = center_xy

    # 2. Calculate pixel radius
    # We use the smaller dimension so a 1.0 radius_percentage
    # creates a circle that fits within the image boundaries.
    min_dim = min(H, W)
    radius_pixels = radius_percentage * (min_dim / 2.0)

    # 3. Create coordinate grids
    # We create (H, W) grids for y and x coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'  # 'ij' indexing ensures (H, W) shape
    )

    # 4. Apply circle equation to create a boolean mask
    # (x - cx)^2 + (y - cy)^2 <= r^2
    dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
    circle_mask = dist_sq <= (radius_pixels**2)  # Shape (H, W)

    # 5. Apply the mask to the tensor (in-place)
    # mask_tensor[0] gives us the (H, W) slice
    mask_tensor[0, circle_mask] = 1.0

    return mask_tensor

with torch.no_grad():
    epath = camera_path.generate_cam_path(train_cameras, 400)
    eimages = []
    for camera in tqdm(epath):
        camera.model = ProjectionType.FISHEYE
        camera.fovx = 2.1
        camera.fovy = 2.1
        camera.resize(camera.image_height, camera.image_height)
        camera.gt_alpha_mask = torch.zeros((1, camera.image_height, camera.image_width), device=camera.data_device)
        draw_filled_circle(camera.gt_alpha_mask, radius_percentage=0.85)
        render_pkg = render(camera, model, tile_size=args.tile_size, min_t=model.min_t)
        image = render_pkg['render']
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        eimages.append(image)

mediapy.write_video(args.output_path / "fisheye.mp4", eimages)
