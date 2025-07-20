import torch
import time
import math
from data.camera import Camera
from delaunay_rasterization.internal.alphablend_tiled_slang_interp import AlphaBlendTiledRender as Render
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
import numpy as np
from utils import topo_utils
from icecream import ic
import math
from utils.graphics_utils import l2_normalize_th
import matplotlib.pyplot as plt
from delaunay_rasterization.internal.alphablend_tiled_slang import render_constant_color
from data.camera import focal2fov

cmap = plt.get_cmap("jet")

class ClippedGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lr_matrix):
        ctx.save_for_backward(lr_matrix)
        return input  # Identity operation

    @staticmethod
    def backward(ctx, grad_output):
        lr_matrix, = ctx.saved_tensors
        grad_norm = torch.linalg.norm(grad_output, dim=-1, keepdim=True)
        # grad_output = torch.maximum(-lr_matrix.abs(), torch.minimum(lr_matrix.abs(), grad_output))
        shape = grad_norm.shape
        clipped_grad_norm = grad_norm.clip(-lr_matrix.abs().reshape(*shape), lr_matrix.abs().reshape(*shape))
        return l2_normalize_th(grad_output) * clipped_grad_norm, None
        # return grad_output, None

class ScaledGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lr_matrix):
        ctx.save_for_backward(lr_matrix)
        return input  # Identity operation

    @staticmethod
    def backward(ctx, grad_output):
        lr_matrix, = ctx.saved_tensors
        return grad_output * lr_matrix, None

def get_slang_projection_matrix(znear, zfar, fy, fx, height, width, device):
    tanHalfFovX = width/(2*fx)
    tanHalfFovY = height/(2*fy)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    z_sign = 1.0

    P = torch.tensor([
       [2.0 * znear / (right - left),     0.0,                          (right + left) / (right - left), 0.0 ],
       [0.0,                              2.0 * znear / (top - bottom), (top + bottom) / (top - bottom), 0.0 ],
       [0.0,                              0.0,                          z_sign * zfar / (zfar - znear),  -(zfar * znear) / (zfar - znear) ],
       [0.0,                              0.0,                          z_sign,                          0.0 ]
    ], device=device)

    return P

def common_camera_properties_from_gsplat(viewmats, Ks, height, width):
    """ Fetches all the Camera properties from the inria defined object"""
    zfar = 100.0
    znear = 0.01
  
    world_view_transform = viewmats
    fx = Ks[0,0]
    fy = Ks[1,1]
    projection_matrix = get_slang_projection_matrix(znear, zfar, fy, fx, height, width, Ks.device)
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)

    cam_pos = viewmats.inverse()[:, 3]

    return world_view_transform, projection_matrix, cam_pos, fovy, fovx

def inverse_sigmoid(y):
    return torch.log(y / (1 - y))

def rgbs_activation(rgbs_raw):
    # rgbs = torch.cat([torch.nn.functional.softplus(1e-1*rgbs_raw[:, :3]), safe_exp(rgbs_raw[:, 3:])], dim=1)
    rgbs = torch.cat([torch.sigmoid(rgbs_raw[:, :3]), safe_exp(rgbs_raw[:, 3:])], dim=1)
    return rgbs

def safe_exp(x):
    return x.clip(max=5).exp()

def safe_trig_helper(x, fn, t=100 * torch.pi):
    """Helper function used by safe_cos/safe_sin: mods x before sin()/cos()."""
    return fn(torch.nan_to_num(torch.where(torch.abs(x) < t, x, x % t)))


def safe_cos(x):
    """jnp.cos() on a TPU may NaN out for large values."""
    return safe_trig_helper(x, torch.cos)


def safe_sin(x):
    """jnp.sin() on a TPU may NaN out for large values."""
    return safe_trig_helper(x, torch.sin)


def render(camera: Camera, model, cell_values=None, tile_size=16, min_t=0.1,
           scene_scaling=1, clip_multi=0, ray_jitter=None, glo=None,
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
    if cell_values is None:
        cell_values = torch.zeros((mask.shape[0], model.feature_dim), device=circumcenter.device)
        if mask.sum() > 0 and model.mask_values:
            normed_cc, cell_values[mask] = model.get_cell_values(camera, mask, circumcenter[mask], glo=glo)
        else:
            normed_cc, cell_values = model.get_cell_values(camera, all_circumcenters=circumcenter, glo=glo)
        if clip_multi > 0 and not model.frozen:
            with torch.no_grad():
                tet_sens, sensitivity = topo_utils.compute_vertex_sensitivity(model.indices[mask],
                                                                            vertices, normed_cc, True)
                scaling = clip_multi*sensitivity.reshape(-1, 1).clip(min=1e-5)
            vertices = ClippedGradients.apply(vertices, scaling)

    image_rgb, distortion_img, tet_alive = Render.apply(
        sorted_tetra_idx,
        tile_ranges,
        model.indices,
        vertices,
        cell_values,
        render_grid,
        tcam,
        ray_jitter)
    alpha = image_rgb.permute(2,0,1)[3, ...]
    # total_density = (distortion_img[:, :, 2]**2).clip(min=1e-6)
    total_density = ((1-alpha) ** 2).clip(min=1e-6)
    distortion_loss = (((distortion_img[:, :, 0] - distortion_img[:, :, 1]) + distortion_img[:, :, 4]) / total_density).clip(min=0)

    
    render_pkg = {
        'render': image_rgb.permute(2,0,1)[:3, ...],
        'alpha': alpha,
        'distortion_img': distortion_img,
        'distortion_loss': distortion_loss.mean(),
        'visibility_filter': mask,
        'circumcenters': circumcenter,
        'density': cell_values[:, 0],
        'mask': mask,
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

    def helper(step):
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
        log_lerp = np.exp(np.log(height) * (1 - t) + np.log(1e-6) * t)
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

class TwoPhaseLR:
    def __init__(self, max_i, start_i, period_i, settle_i, 
                 lr_peak, lr_end_peak, lr_trough, lr_final):
        self.max_i = max_i
        self.start_i = start_i
        self.settle_i = settle_i
        self.period_i = period_i
        self.lr_peak = lr_peak
        self.lr_end_peak = lr_end_peak
        self.lr_trough = lr_trough
        self.lr_final = lr_final

        n_cycles = settle_i / period_i
        self.gamma = (lr_end_peak / lr_peak) ** (1 / n_cycles) if n_cycles > 0 else 1

    def __call__(self, i):
        # Phase 1: Spiking with decaying cosine annealing
        if i < self.start_i:
            return get_expon_lr_func(self.lr_peak, self.lr_trough, max_steps=self.start_i)(i)
        elif self.start_i <= i <= self.settle_i:
            cycle = math.floor((i-self.start_i) / self.period_i)
            t_cycle = (i-self.start_i) % self.period_i
            
            lr_max = self.lr_peak * (self.gamma ** cycle)
            
            height = (lr_max - self.lr_trough)
            # lr = self.lr_trough + 0.5 * height * \
            #      (1 + math.cos(math.pi * t_cycle / self.period_i))
            t = t_cycle / self.period_i
            lr = self.lr_trough + np.exp(np.log(height) * (1 - t) + np.log(1e-6) * t)
            
            return lr

        # Phase 2: Final settling cosine decay
        else:
            if i >= self.max_i:
                return self.lr_final

            t_settle = i - self.settle_i
            d_settle = self.max_i - self.settle_i
            if d_settle <= 0:
                return self.lr_final
            
            lr = self.lr_final + 0.5 * (self.lr_trough - self.lr_final) * \
                 (1 + math.cos(math.pi * t_settle / d_settle))

            return lr

def render_debug(render_tensor, model, camera, density_multi=1):

    # Convert to RGB (NxMx3) using the colormap
    _, features = model.get_cell_values(camera)
    tet_grad_color = torch.zeros((features.shape[0], 4), device=features.device)
    if render_tensor.shape[1] == 1:
        tensor_min, tensor_max = render_tensor.min(), torch.quantile(render_tensor, 0.99)
        normalized_tensor = ((render_tensor - tensor_min) / (tensor_max - tensor_min)).clip(0, 1)
        normalized_tensor = torch.as_tensor(
            cmap(normalized_tensor.reshape(-1).cpu().numpy())).float().cuda()
    else:
        normalized_tensor = render_tensor
    tet_grad_color[:, :normalized_tensor.shape[1]] = normalized_tensor
    if render_tensor.shape[1] < 4:
        tet_grad_color[:, 3] = features[:, 0] * density_multi# * render_tensor.reshape(-1)
    render_pkg = render_constant_color(model.indices, model.vertices, None, camera, cell_values=tet_grad_color)

    image = render_pkg['render']
    image = image.permute(1, 2, 0)
    image = (image.detach().cpu().numpy() * 255).clip(min=0, max=255).astype(np.uint8)

    del render_pkg, render_tensor
    return image

def get_approx_ray_intersections(split_rays_data, epsilon=1e-7):
    """
    Calculates the approximate intersection point for pairs of line segments.

    The intersection is defined as the midpoint of the shortest segment
    connecting the two input line segments.

    Args:
        split_rays_data (torch.Tensor): Tensor of shape (N, 2, 6).
            - N: Number of segment pairs.
            - 2: Represents the two segments in a pair.
            - 6: Contains [Ax, Ay, Az, Bx, By, Bz] for each segment,
                 where A and B are the segment endpoints.
                 Based on current Python code:
                 A = average_P_exit, B = average_P_entry
        epsilon (float): Small value to handle parallel lines and avoid
                         division by zero if a segment has zero length.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) representing the approximate
                      "intersection" points (midpoints of closest approach).
    """
    # Segment 1 endpoints
    p1_a = split_rays_data[:, 0, 0:3]  # Endpoint A of first segments (N, 3)
    p1_b = split_rays_data[:, 0, 3:6]  # Endpoint B of first segments (N, 3)
    # Segment 2 endpoints
    p2_a = split_rays_data[:, 1, 0:3]  # Endpoint A of second segments (N, 3)
    p2_b = split_rays_data[:, 1, 3:6]  # Endpoint B of second segments (N, 3)

    # Define segment origins and direction vectors
    # Segment S1: o1 + s * d1, for s in [0, 1]
    # Segment S2: o2 + t * d2, for t in [0, 1]
    o1 = p1_a
    d1 = p1_b - p1_a  # Direction vector for segment 1 (from A to B)
    o2 = p2_a
    d2 = p2_b - p2_a  # Direction vector for segment 2 (from A to B)

    # Calculate terms for finding closest points on the infinite lines
    # containing the segments (based on standard formulas, e.g., Christer Ericson's "Real-Time Collision Detection")
    v_o = o1 - o2 # Vector from origin of line 2 to origin of line 1

    a = torch.sum(d1 * d1, dim=1)  # Squared length of d1
    b = torch.sum(d1 * d2, dim=1)  # Dot product of d1 and d2
    c = torch.sum(d2 * d2, dim=1)  # Squared length of d2
    d = torch.sum(d1 * v_o, dim=1) # d1 dot (o1 - o2)
    e = torch.sum(d2 * v_o, dim=1) # d2 dot (o1 - o2)

    denom = a * c - b * b
    s_line_num = (b * e) - (c * d)
    t_line_num = (a * e) - (b * d) # This corresponds to t_c = (a*e - b*d)/denom from previous thoughts for P(t) = O2 + tD2

    # Handle near-zero denominator (lines are parallel or one segment is a point)
    # We compute with a safe denominator, then clamp. Clamping is key for segments.
    denom_safe = torch.where(denom.abs() < epsilon, torch.ones_like(denom), denom)
    
    s_line = s_line_num / denom_safe
    t_line = t_line_num / denom_safe # Note: This t_line is for the parameter of d2 (from o2)

    # Clamp parameters to [0, 1] to stay within the segments
    bad_intersect = (s_line < 0) | (t_line < 0) | (s_line > 1) | (t_line > 1)
    s_seg = torch.clamp(s_line, 0.0, 1.0)
    t_seg = torch.clamp(t_line, 0.0, 1.0)

    # Points of closest approach on the segments
    pc1 = o1 + s_seg.unsqueeze(1) * d1
    pc2 = o2 + t_seg.unsqueeze(1) * d2
    
    p_int = (pc1 + pc2) / 2.0
                        
    return p_int, bad_intersect
