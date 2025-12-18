import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.camera import Camera
from delaunay_rasterization.internal.alphablend_tiled_slang_interp import AlphaBlendTiledRender
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from utils.eval_sh_py import weigh_degree
import time
from icecream import ic

cmap = plt.get_cmap("jet")

def render(camera: Camera, model, cell_values=None, tile_size=16, min_t=0.1,
           scene_scaling=1, clip_multi=0, ray_jitter=None,
           **kwargs):
    device = model.device
    if ray_jitter is None:
        ray_jitter = 0.5*torch.ones((camera.image_height, camera.image_width, 2), device=device)
    else:
        assert(ray_jitter.shape[0] == camera.image_height)
        assert(ray_jitter.shape[1] == camera.image_width)
        assert(ray_jitter.shape[2] == 2)
    vertices = model.vertices
    
    render_grid = RenderGrid(camera.image_height,
                             camera.image_width,
                             tile_height=tile_size,
                             tile_width=tile_size)
    tcam = dict(
        tile_height=tile_size,
        tile_width=tile_size,
        grid_height=render_grid.grid_height,
        grid_width=render_grid.grid_width,
        min_t=min_t,
        **camera.to_dict(device)
    )
    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _ = vertex_and_tile_shader(
        model.indices,
        vertices,
        tcam,
        render_grid)
    extras = {}
    sh_reg = 0
    if cell_values is None:
        cell_values = torch.zeros((mask.shape[0], model.feature_dim), device=circumcenter.device)
        if mask.sum() > 0 and model.mask_values:
            shs, values = model.get_cell_values(camera, mask, circumcenter[mask])
            cell_values[mask] = values
        else:
            shs, cell_values = model.get_cell_values(camera, all_circumcenters=circumcenter)
        weighting = weigh_degree(shs, [0, 0.1, 0.5, 1])
        sh_reg = ((weighting * shs)**2).mean()

    image_rgb, xyzd_img, distortion_img, tet_alive = AlphaBlendTiledRender.apply(
        sorted_tetra_idx,
        tile_ranges,
        model.indices,
        vertices,
        cell_values,
        render_grid,
        tcam,
        ray_jitter)
    alpha = image_rgb.permute(2,0,1)[3, ...]
    total_density = (distortion_img[:, :, 2]**2).clip(min=1e-6)
    distortion_loss = (((distortion_img[:, :, 0] - distortion_img[:, :, 1]) + distortion_img[:, :, 4]) / total_density).clip(min=0)

    # unrotate the xyz part of the xyzd_img
    rotated = xyzd_img[..., :3].reshape(-1, 3) @ camera.world_view_transform[:3, :3].to(device)
    rxyzd_img = torch.cat([rotated.reshape(xyzd_img[..., :3].shape), xyzd_img[..., 3:]], dim=-1)
    
    render_pkg = {
        'render': image_rgb.permute(2,0,1)[:3, ...] * camera.gt_alpha_mask.to(device),
        'alpha': alpha,
        'distortion_loss': distortion_loss.mean(),
        'mask': mask,
        'xyzd': rxyzd_img,
        'sh_reg': sh_reg,
        **extras
    }
    return render_pkg

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """
    lr_init = max(lr_init, 1e-20)
    lr_final = max(lr_final, 1e-20)

    def helper(step):
        if max_steps == 0:
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def pad_hw2even(h, w):
    return int(math.ceil(h / 2))*2, int(math.ceil(w / 2))*2

def pad_image2even(im, fnp=np):
    h, w = im.shape[:2]
    nh, nw = pad_hw2even(h, w)
    im_full = fnp.zeros((nh, nw, 3), dtype=im.dtype)
    im_full[:h, :w] = im
    return im_full

class SpikingLR:
    def __init__(self, duration, max_steps, base_function,
                 peak_start, peak_interval, peak_end,
                 peak_lr_init, peak_lr_final):
        self.duration = duration
        self.base_function = base_function
        self.max_steps = max_steps

        self.peak_start = peak_start
        self.peak_interval = peak_interval
        self.peak_end = peak_end

        self.peak_lr_init = peak_lr_init
        self.peak_lr_final = peak_lr_final

    def peak_height_fn(self, i):
        return i / self.max_steps * (self.peak_lr_final - self.peak_lr_init) + self.peak_lr_init
        # return self.peak_lr_init
    
    def peak_fn(self, step, height):
        t = np.clip(step / self.duration, 0, 1)
        log_lerp = np.exp(np.log(max(height, 1e-20)) * (1 - t) + np.log(1e-6) * t)
        return log_lerp
        # return height * math.exp(-step * 6/self.duration + 2/self.duration) / math.exp(2/self.duration)

    def __call__(self, iteration):
        base_f = self.base_function(iteration)
        if iteration < self.peak_start:
            return base_f
        elif iteration > self.peak_end:
            last_peak = iteration - self.peak_end
        else:
            last_peak = (iteration - self.peak_start) % self.peak_interval
        peak_ind = iteration - last_peak
        height = self.peak_height_fn(peak_ind) - self.base_function(peak_ind)
        return base_f + self.peak_fn(last_peak, height)
