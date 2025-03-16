import torch
from utils.graphics_utils import l2_normalize_th
from utils.safe_math import safe_exp, safe_div, safe_sqrt, safe_pow, safe_cos, safe_sin, remove_zero, safe_arctan2
# @torch.jit.script
def to_sphere(coordinates):
    return torch.stack([
        safe_cos(coordinates[..., 0]) * safe_sin(coordinates[..., 1]),
        safe_sin(coordinates[..., 0]) * safe_sin(coordinates[..., 1]),
        safe_cos(coordinates[..., 1]),
    ], dim=-1)

@torch.jit.script
def light_function(base_color, reflection_dirs, light_colors, light_roughness, view_dirs, eps:float=torch.finfo(torch.float32).eps):
    similarity = (reflection_dirs * view_dirs).sum(dim=-1, keepdim=True)
    mask = similarity > 0
    spec_intensity = torch.where(mask, (similarity.clip(min=eps) ** light_roughness), 0)
    spec_color = (light_colors * spec_intensity).sum(dim=1)
    return torch.nn.functional.softplus(base_color + spec_color + 0.5, beta=10)

@torch.jit.script
def activate_lights(base_color_raw, lights, light_offset: float, dir_offset):
    base_color = base_color_raw
    light_colors = lights[:, :, :3]#+light_offset
    # ic(base_color_raw, light_colors)
    light_roughness = 4*safe_exp(lights[:, :, 3:4]).clip(max=100)
    # ic(light_roughness)
    num_lights = lights.shape[1]
    reflection_dirs = 4*lights[:, :, 4:6] + dir_offset.reshape(1, -1, 2)[:, :num_lights]
    return base_color, light_colors, light_roughness, reflection_dirs

@torch.jit.script
def compute_tet_color(base_color_raw, lights, vertices, indices, camera_center, light_offset: float, dir_offset):
    base_color, light_colors, light_roughness, reflection_dirs = activate_lights(
        base_color_raw, lights, light_offset, dir_offset)
    reflection_dirs = to_sphere(reflection_dirs)
    # reflection_dirs = lambert_to_sphere(reflection_dirs)
    barycenters = vertices[indices].mean(dim=1)
    view_dirs = l2_normalize_th(camera_center - barycenters).reshape(-1, 1, 3)
    return light_function(base_color, reflection_dirs, light_colors, light_roughness, view_dirs)

def compute_vert_color(base_color_raw, lights, vertices, camera_center, light_offset: float, dir_offset):
    base_color, light_colors, light_roughness, reflection_dirs = activate_lights(
        base_color_raw, lights, light_offset, dir_offset)
    reflection_dirs = to_sphere(reflection_dirs)
    view_dirs = l2_normalize_th(camera_center - vertices).reshape(-1, 1, 3)
    return light_function(base_color, reflection_dirs, light_colors, light_roughness, view_dirs)
