from utils.train_util import render
# from models.vertex_color import Model
from models.tet_color import Model
# from models.ingp_color import Model
import pickle
import torch
from tqdm import tqdm
from pathlib import Path
import imageio
import numpy as np

print("1")
tile_size = 16
with open('camera.pkl', 'rb') as f:
    camera = pickle.load(f)

device = torch.device('cuda')
print("Loading")
# model = Model.load_ply("ckpt.ply", device)
model = Model.load_ply("output/vld500_vldm0.0001_vl1.00E-04_fvl1.00E-05_b2000000/ckpt.ply", device)
# model = Model.load_ckpt(Path("output/ld0"), device)

print(model.indices.shape)
print("Starting")
for i in tqdm(range(100)):
    render_pkg = render(camera, model, tile_size=tile_size)
print("Done")
render_pkg = render(camera, model, tile_size=tile_size)
image = render_pkg['render']
image = image.permute(1, 2, 0)
image = (image.detach().cpu().numpy() * 255).clip(min=0, max=255).astype(np.uint8)
imageio.imwrite('test.png', image)

# model.save2ply(Path("ckpt2.ply"))