"""Quick VK training convergence test."""
import torch
import numpy as np
import sys
import random

sys.path.insert(0, '.')
from data import loader
from models.ingp_color import Model, TetOptimizer
from utils.render_vk import render_vk, create_vk_renderer
from pathlib import Path
from fused_ssim import fused_ssim
from utils.args import Args

device = torch.device('cuda')
train_cameras, _, scene_info = loader.load_dataset(
    Path('/data/nerf_datasets/360/bicycle'), 'images_8',
    data_device='cpu', eval=True, resolution=1)

args = Args()
args.max_sh_deg = 0
args.use_tcnn = False
args.ablate_circumsphere = True
args.encoding_lr = 3e-3
args.final_encoding_lr = 3e-4
args.network_lr = 3e-3
args.final_network_lr = 3e-4
args.vertices_lr = 1e-4
args.final_vertices_lr = 1e-6
args.vertices_lr_delay_multi = 1e-8
args.delaunay_interval = 10
args.freeze_start = 18000
args.lambda_ssim = 0.2
args.lambda_opacity = 0.0
args.lambda_weight_decay = 1

model = Model.init_from_pcd(
    scene_info.point_cloud, train_cameras, device,
    current_sh_deg=0, **args.as_dict())
tet_optim = TetOptimizer(model, **args.as_dict())

W, H = train_cameras[0].image_width, train_cameras[0].image_height
vk_renderer = create_vk_renderer(model, train_cameras[0], W, H)

inds = list(range(len(train_cameras)))
random.shuffle(inds)

for iteration in range(100):
    camera = train_cameras[inds[iteration % len(inds)]]
    target = camera.original_image.cuda()
    gt_mask = camera.gt_alpha_mask.cuda()

    if camera.image_width != W or camera.image_height != H:
        W, H = camera.image_width, camera.image_height
        vk_renderer = create_vk_renderer(model, camera, W, H)

    render_pkg = render_vk(camera, model, vk_renderer)
    image = render_pkg['render']
    alpha = render_pkg['alpha']

    l1_loss = ((target - image).abs() * gt_mask).mean()
    ssim_loss = (1 - fused_ssim(
        image.unsqueeze(0), target.unsqueeze(0))).clip(min=0, max=1)
    reg = tet_optim.regularizer(render_pkg, **args.as_dict())
    loss = 0.8 * l1_loss + 0.2 * ssim_loss + reg
    loss.backward()

    tet_optim.main_step()
    tet_optim.main_zero_grad()
    if iteration % 10 == 0:
        tet_optim.vertex_optim.step()
        tet_optim.vertex_optim.zero_grad()
    tet_optim.update_learning_rate(iteration)

    if iteration % 10 == 0:
        l2 = ((target - image) ** 2 * gt_mask).mean().item()
        psnr = -10 * np.log10(max(l2, 1e-8))
        nz = (alpha > 0.001).sum().item()
        total = alpha.numel()
        print(f"iter {iteration:3d}: L1={l1_loss.item():.4f} "
              f"PSNR={psnr:.2f} alpha_nz={nz}/{total} "
              f"img_max={image.max():.4f}")
