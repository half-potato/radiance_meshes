import torch
from pathlib import Path
from data import loader
from utils.args import Args
from models.ingp_color import Model, TetOptimizer

args = Args()
args.tile_size = 16
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
    if args.freeze_features:
        from models.frozen_features import FrozenTetModel, FrozenTetOptimizer
    else:
        from models.frozen import FrozenTetModel, FrozenTetOptimizer
    model = FrozenTetModel.load_ckpt(args.output_path, device)

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cpu", eval=False)
model.extract_mesh(train_cameras, args.output_path / "meshes", **args.as_dict())
# eventually, generate a UV map for each mesh
# organize onto a texture map
# then, optimize the texture for each of these maps
