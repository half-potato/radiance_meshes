import torch
import time
import math
from data.camera import Camera
from utils import optim
from sh_slang.eval_sh import eval_sh
from delaunay_rasterization.internal.alphablend_tiled_slang import AlphaBlendTiledRender, render_alpha_blend_tiles_slang_raw
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from gDel3D.build.gdel3d import Del
from icecream import ic

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

class ScaleGradients(torch.autograd.Function):
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


def render(camera: Camera, model, register_tet_hook=False, tile_size=4):
    fy = fov2focal(camera.fovy, camera.image_height)
    fx = fov2focal(camera.fovx, camera.image_width)
    K = torch.tensor([
    [fx, 0, camera.image_width/2],
    [0, fy, camera.image_height/2],
    [0, 0, 1],
    ]).to(camera.world_view_transform.device)

    cam_pos = camera.world_view_transform.T.inverse()[:, 3]

    # render_pkg = render_alpha_blend_tiles_slang_raw(model.indices, model.vertices, None,
    #                                                 camera.world_view_transform.T, K, cam_pos,
    #                                                 camera.fovy, camera.fovx, camera.image_height,
    #                                                 camera.image_width, cell_values=cell_values,
    #                                                 tile_size=tile_size)
    world_view_transform = camera.world_view_transform.T
    torch.cuda.synchronize()
    assert(model.indices.device == model.vertices.device)
    assert(model.indices.device == world_view_transform.device)
    assert(model.indices.device == K.device)
    assert(model.indices.device == cam_pos.device)
    st = time.time()
    
    render_grid = RenderGrid(camera.image_height,
                             camera.image_width,
                             tile_height=tile_size,
                             tile_width=tile_size)
    st = time.time()
    densities = safe_exp(model.vertex_s_param)
    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _ = vertex_and_tile_shader(
        model.indices,
        model.vertices,
        densities,
        world_view_transform,
        K,
        cam_pos,
        camera.fovy,
        camera.fovx,
        render_grid)
    # cell_values = torch.zeros((mask.shape[0], 4), device=circumcenter.device)
    # cell_values[mask] = model.get_cell_values(camera, mask)
    cell_values = model.get_cell_values(camera)
    tet_grads = []
    if register_tet_hook:
        cell_values.register_hook(lambda d: tet_grads.append(d))

    # torch.cuda.synchronize()
    # ic("vt", time.time()-st)
    # retain_grad fails if called with torch.no_grad() under evaluation
    try:
        vs_tetra.retain_grad()
    except:
        pass

    # st = time.time()
    image_rgb = AlphaBlendTiledRender.apply(
        sorted_tetra_idx,
        tile_ranges,
        model.indices,
        model.vertices,
        cell_values,
        render_grid,
        world_view_transform,
        K,
        cam_pos,
        camera.fovy,
        camera.fovx)
    # torch.cuda.synchronize()
    # dt2 = (time.time() - st)
    # print(dt1, dt2, 1/(dt1+dt2))
    
    render_pkg = {
        'render': image_rgb.permute(2,0,1)[:3, ...],
        'viewspace_points': vs_tetra,
        'visibility_filter': mask,
        'circumcenters': circumcenter,
        'tet_grad': tet_grads,
    }
    return render_pkg