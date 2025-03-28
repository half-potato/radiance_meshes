import torch
import time
import math
from data.camera import Camera
from utils import optim
from sh_slang.eval_sh import eval_sh
from delaunay_rasterization.internal.alphablend_tiled_slang_interp import AlphaBlendTiledRender
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader, point2image
import numpy as np
from utils import topo_utils, train_util
from icecream import ic
import math

def sample_uniform_in_sphere(batch_size, dim, radius=1.0, device=None):
    """
    Generate samples uniformly distributed inside a sphere.

    Parameters:
        batch_size (int): Number of samples to generate.
        dim (int): Dimensionality of the sphere.
        radius (float): Radius of the sphere (default is 1.0).
        device (torch.device, optional): Device to perform computation on.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, dim) with samples from inside the sphere.
    """
    if device is None:
        device = torch.device("cpu")

    # Sample from a normal distribution
    samples = torch.randn(batch_size, dim, device=device)
    
    # Normalize each vector to lie on the unit sphere
    samples = samples / samples.norm(dim=1, keepdim=True)
    
    # Sample radii uniformly with proper weighting for volume
    radii = torch.rand(batch_size, device=device).pow(1 / dim) * radius
    
    # Scale samples by the radii
    samples = samples * radii.unsqueeze(1)

    return samples

class ClippedGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lr_matrix):
        ctx.save_for_backward(lr_matrix)
        return input  # Identity operation

    @staticmethod
    def backward(ctx, grad_output):
        lr_matrix, = ctx.saved_tensors
        grad_output = torch.maximum(-lr_matrix.abs(), torch.minimum(lr_matrix.abs(), grad_output))
        return grad_output, None

class ScaledGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lr_matrix):
        ctx.save_for_backward(lr_matrix)
        return input  # Identity operation

    @staticmethod
    def backward(ctx, grad_output):
        lr_matrix, = ctx.saved_tensors
        return grad_output * lr_matrix, None

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

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

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

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


def render(camera: Camera, model, bg=0, cell_values=None, tile_size=16, min_t=0.1,
           pre_multi=500, ladder_p=-0.1, clip_multi=1e-1,
           **kwargs):
    fy = fov2focal(camera.fovy, camera.image_height)
    fx = fov2focal(camera.fovx, camera.image_width)
    K = torch.tensor([
        [fx, 0, camera.image_width/2],
        [0, fy, camera.image_height/2],
        [0, 0, 1],
    ]).to(camera.world_view_transform.device)
    cam_pos = camera.camera_center
    vertices = model.vertices
    world_view_transform = camera.world_view_transform.T

    assert(model.indices.device == model.vertices.device)
    assert(model.indices.device == world_view_transform.device)
    assert(model.indices.device == K.device)
    assert(model.indices.device == cam_pos.device)
    
    render_grid = RenderGrid(camera.image_height,
                             camera.image_width,
                             tile_height=tile_size,
                             tile_width=tile_size)
    # with torch.no_grad():
    #     sensitivity = topo_utils.compute_vertex_sensitivity(model.indices, model.vertices)
    #     scaling = clip_multi/sensitivity.reshape(-1, 1).clip(min=1e-3, max=1000)
    # vertices = train_util.ClippedGradients.apply(model.vertices, scaling)
    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _, tet_area = vertex_and_tile_shader(
        model.indices,
        # scale_vertices,
        vertices,
        world_view_transform,
        K,
        cam_pos,
        camera.fovy,
        camera.fovx,
        render_grid)
    # if cell_values is None:
    #     cell_values = torch.zeros((mask.shape[0]), device=circumcenter.device)
    #     if mask.sum() > 0:
    #         vertex_color, cell_values[mask] = model.get_cell_values(camera, mask)
    #     else:
    #         vertex_color, cell_values = model.get_cell_values(camera)
    vertex_color, cell_values = model.get_cell_values(camera)
    # cell_values = model.get_cell_values(camera)

    # torch.cuda.synchronize()
    # ic("vt", time.time()-st)
    # retain_grad fails if called with torch.no_grad() under evaluation
    try:
        vs_tetra.retain_grad()
    except:
        pass
    # tet_vertices = vertices[model.indices]
    # verts_trans = point2image(model.vertices, world_view_transform, K, cam_pos)
    # st = time.time()
    image_rgb, distortion_img = AlphaBlendTiledRender.apply(
        sorted_tetra_idx,
        tile_ranges,
        model.indices,
        vertices,
        vertex_color,
        cell_values,
        render_grid,
        world_view_transform,
        K,
        cam_pos,
        pre_multi,
        ladder_p,
        min_t,
        camera.fovy,
        camera.fovx)
    # alpha = 1-image_rgb.permute(2,0,1)[3, ...]
    # ic(alpha.min(), alpha.max())
    distortion_loss = ((distortion_img[:, :, 0] - distortion_img[:, :, 1]) + distortion_img[:, :, 4])# / alpha.clip(min=1e-3)
    # ic(distortion_img)
    # torch.cuda.synchronize()
    # dt2 = (time.time() - st)
    # print(dt1, dt2, 1/(dt1+dt2))
    # image_rgb = image_rgb + (1-image_rgb[..., 3:4]) * bg * torch.rand_like(image_rgb)
    # ic(distortion_loss.mean(), distortion_loss.max())
    
    render_pkg = {
        'render': image_rgb.permute(2,0,1)[:3, ...],
        'alpha': image_rgb.permute(2,0,1)[3, ...],
        'distortion_img': distortion_img,
        'distortion_loss': distortion_loss.mean(),
        'viewspace_points': vs_tetra,
        'visibility_filter': mask,
        'circumcenters': circumcenter,
        'tet_area': tet_area,
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