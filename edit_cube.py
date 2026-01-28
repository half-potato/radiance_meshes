import pyvista as pv
import tetgen
import torch
import numpy as np

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
from PIL import Image
from dtlookup import TetrahedraLookup
from scipy.spatial import Delaunay
from utils import topo_utils
import gc
from tracers.splinetracers.tetra_splinetracer import render_rt

@torch.no_grad()
def insert_cube_conforming(model, cube_center=(0, 0, 0), cube_size=0.5, resolution=32):
    # 1. Generate Steiner points (Shell sampling)
    c = np.array(cube_center)
    s = cube_size / 2.0
    
    steiner_points = []
    lin = np.linspace(-s, s, resolution)
    grid_x, grid_y = np.meshgrid(lin, lin)
    flat_x, flat_y = grid_x.ravel(), grid_y.ravel()

    for val in [-s, s]:
        steiner_points.append(np.stack([flat_x + c[0], flat_y + c[1], np.full_like(flat_x, val) + c[2]], axis=1))
        steiner_points.append(np.stack([flat_x + c[0], np.full_like(flat_x, val) + c[1], flat_y + c[2]], axis=1))
        steiner_points.append(np.stack([np.full_like(flat_x, val) + c[0], flat_x + c[1], flat_y + c[2]], axis=1))

    stein_np = np.unique(np.concatenate(steiner_points), axis=0)
    stein_torch = torch.from_numpy(stein_np).float().to(model.device)

    # 2. Merge radiance mesh points with cube vertices
    orig_verts = model.vertices
    new_vertices = torch.vstack([stein_torch, orig_verts])

    # 3. Create new Delaunay triangulation
    new_vertices_np = new_vertices.detach().cpu().numpy()
    tri = Delaunay(new_vertices_np)
    new_indices = torch.as_tensor(tri.simplices.astype(np.int32), device=model.device)
    
    vols = topo_utils.tet_volumes(new_vertices[new_indices])
    new_indices[vols < 0] = new_indices[vols < 0][:, [1, 0, 2, 3]]

    # 4. Reverse Lookup: Map old attributes to new tets
    # Only use model.indices (active tets) to match model.density shape
    old_indices = model.indices
    old_centroids = model.vertices[old_indices].mean(dim=1).detach().cpu().numpy()
    
    new_tet_indices = tri.find_simplex(old_centroids)
    # Mask for centroids that actually landed inside a new tetrahedron
    valid_mask = new_tet_indices != -1
    
    num_new_tets = new_indices.shape[0]
    new_density = torch.zeros((num_new_tets, 1), device=model.device)
    new_gradient = torch.zeros((num_new_tets, 1, 3), device=model.device)
    new_rgb = torch.zeros((num_new_tets, 3), device=model.device)
    new_sh = torch.zeros((num_new_tets, model.sh.shape[1]), dtype=torch.half, device=model.device)

    # Map attributes from old tets to the new tets they now reside in
    target_ids = torch.from_numpy(new_tet_indices[valid_mask]).long().to(model.device)
    new_density[target_ids] = model.density[valid_mask]
    new_gradient[target_ids] = model.gradient[valid_mask]
    new_rgb[target_ids] = model.rgb[valid_mask]
    new_sh[target_ids] = model.sh[valid_mask]

    # 5. Masking: Identify tetrahedra inside the cube
    new_centroids = new_vertices[new_indices].mean(dim=1)
    half_size = cube_size / 2.0
    min_bound = torch.tensor(cube_center, device=model.device) - half_size
    max_bound = torch.tensor(cube_center, device=model.device) + half_size

    is_inside = (new_centroids >= min_bound).all(dim=-1) & \
                (new_centroids <= max_bound).all(dim=-1)
    new_density[is_inside] = 0

    # 6. Update Model
    model.interior_vertices = torch.nn.Parameter(new_vertices)
    model.ext_vertices = torch.empty((0, 3), device=model.device)
    model.indices.data = new_indices.int()
    model.density.data = new_density
    model.gradient.data = new_gradient
    model.rgb.data = new_rgb
    model.sh.data = new_sh
        
    gc.collect()
    print(f"Mesh updated: {len(new_vertices)} vertices, {len(new_indices)} tetrahedra.")

def visualize_with_pyvista(nodes, elem):
    """Visualizes the resulting tetrahedral mesh."""
    # Create the PyVista unstructured grid
    # 11 is the cell type for a tetrahedron in VTK
    cells = np.column_stack([np.full(len(elem), 4), elem])
    grid = pv.UnstructuredGrid(cells, np.full(len(elem), 11), nodes)
    
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True, color='white', opacity=0.3)
    plotter.add_mesh(grid.extract_cells(range(min(500, len(elem)))), color='red') # Highlight some interior tets
    plotter.show()

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

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cpu", eval=args.eval)

# --- NEW: Cube Insertion Logic ---
print("Inserting cube and re-triangulating...")
# Position the cube somewhere within your scene bounds
center = model.center.cpu().numpy().flatten()
insert_cube_conforming(model, cube_center=center, cube_size=model.scene_scaling.item() * 0.5)

# --- NEW: Visualization ---
# visualize_with_pyvista(nodes, elem)

# ic(model.min_t)
model.min_t = args.min_t
if args.render_train:
    splits = zip(['train', 'test'], [train_cameras, test_cameras])
else:
    splits = zip(['test'], [test_cameras])
test_util.evaluate_and_save(model, splits, args.output_path / 'cut', args.tile_size, min_t=model.min_t)
model.save2ply(Path('test.ply'))

model.compute_adjacency()

with torch.no_grad():
    epath = cam_util.generate_cam_path(train_cameras, 400)
    eimages = []
    for camera in tqdm(epath):
        render_pkg = render(camera, model, min_t=model.min_t)
        image = render_pkg['render']
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        eimages.append(image)

mediapy.write_video(args.output_path / "rotating_removed.mp4", eimages)
