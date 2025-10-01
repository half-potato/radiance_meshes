import torch
from torch import nn
from utils import hashgrid
import math

from utils.graphics_utils import l2_normalize_th
from utils.topo_utils import calculate_circumcenters_torch, fibonacci_spiral_on_sphere, calc_barycentric, sample_uniform_in_sphere, project_points_to_tetrahedra, contraction_jacobian_d_in_chunks
from utils.safe_math import safe_exp, safe_div, safe_sqrt
from utils.contraction import contract_mean_std
from utils.contraction import contract_points, inv_contract_points
from sh_slang.eval_sh_py import eval_sh
from utils.hashgrid import HashEmbedderOptimized
from icecream import ic
import torch.nn.init as init # Common alias for torch.nn.init
import tinycudann as tcnn

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def gaussian_in_circumsphere(cc: torch.Tensor,       # (T,3)
                             r:  torch.Tensor,       # (T,1)
                             k:  int,
                             trunc_sigma: float = 0.3) -> torch.Tensor:
    """
    Draw `k` 3‑D points from N(cc, (trunc_sigma*r)^2 I), truncated so ‖x‑cc‖≤r.

    Returns: (T,k,3)
    """
    T = cc.shape[0]
    # iid standard normal                                                          (T,k,3)
    x  = torch.randn((T, k, 3), device=cc.device)

    # scale by radius*trunc_sigma
    x  = x * (trunc_sigma * r).unsqueeze(1)

    # rejection‑sampling for the few out‑of‑sphere samples ------------------------
    inside = (x.norm(dim=-1, p=2) <= r.unsqueeze(1)).all(dim=-1)
    while not inside.all():
        # re‑draw only the failed rows (≈ 1 % for σ=0.3)
        mask   = ~inside
        num    = mask.sum()
        x[mask] = torch.randn((num, 3), device=x.device) * (trunc_sigma * r[mask])
        inside  = (x.norm(dim=-1) <= r[mask]).all(dim=-1)

    return cc.unsqueeze(1) + x                 # (T,k,3)

def init_linear(m, gain):
    """Standard Xavier-uniform + zero bias for any Linear layer."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

@torch.jit.script
def pre_calc_cell_values(vertices, indices):
    device = vertices.device
    tets = vertices[indices]
    circumcenter, radius = calculate_circumcenters_torch(tets.double())
    return circumcenter

# @torch.jit.script
def compute_gradient_from_vertex_colors(
    vcolors:        torch.Tensor,   # (T, 4, 3) - (T, V, C)
    element_verts:  torch.Tensor,   # (T, 4, 3) - (T, V, D)
    circumcenters:  torch.Tensor    # (T, 3)    - (T, D)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Returns base (T,C) and gradients (T,C,D)
    """
    Recovers the base color and gradients from vertex colors and geometry.
    Assumes V=4 vertices, D=3 dimensions, C=3 color channels.
    """
    # 1. unit directions from the circum-centre
    # dirs: (T, 4, 3) -> (T, V, D)
    dirs = torch.nn.functional.normalize(
        element_verts - circumcenters[:, None, :], p=2.0, dim=-1, eps=1e-12
    )

    d0, d1, d2, d3 = dirs.unbind(dim=1)
    f0, f1, f2, f3 = vcolors.unbind(dim=1)
    A = torch.stack([d1 - d0, d2 - d0, d3 - d0], dim=1)   # (T, 3, D) = (T, 3, 3)
    
    B = torch.stack([f1 - f0, f2 - f0, f3 - f0], dim=1)   # (T, 3, C)

    solved_G_transposed = torch.linalg.solve(A, B) # Shape (T, D, C)

    gradients_recovered = solved_G_transposed.transpose(1, 2) # Shape (T, C, D) = (T, 3, 3)

    base_recovered = f0 - torch.einsum('tcd,td->tc', gradients_recovered, d0) # Shape (T, C)

    return base_recovered, gradients_recovered

@torch.jit.script
def compute_vertex_colors_from_field(
    element_verts: torch.Tensor,   # (T, 4, 3) - Renamed for clarity (V=4, D=3)
    base:           torch.Tensor,   # (T, 3)    - (T, C)
    gradients:      torch.Tensor,   # (T, 3, 3) - (T, C, D_grad=3)
    circumcenters:  torch.Tensor    # (T, 3)    - (T, D=3)
) -> torch.Tensor: # Returns vertex_colors (T, 4, 3)
    """
    Compute per-vertex colors for each element (e.g., tetrahedron).
    
    For each vertex:
      color = base_for_channel + dot(gradient_for_channel, normalized(vertex - circumcenter))
    """
    offsets = element_verts - circumcenters[:, None, :]
    grad_contrib = torch.einsum('tcd,tvd->tvc', gradients, offsets)
    vertex_colors = base[:, None, :] + grad_contrib 
    
    return vertex_colors

@torch.jit.script
def offset_normalize(rgb, grd, circumcenters, tets):
    grd = grd.reshape(-1, 1, 3) * rgb.reshape(-1, 3, 1).mean(dim=1, keepdim=True).detach()
    radius = torch.linalg.norm(tets[:, 0] - circumcenters, dim=-1, keepdim=True).reshape(-1, 1, 1)
    normed_grd = safe_div(grd, radius)
    vcolors = compute_vertex_colors_from_field(
        tets.detach(), rgb.reshape(-1, 3), normed_grd.float(), circumcenters.float().detach())

    base_color_v0_raw = vcolors[:, 0]
    return base_color_v0_raw, normed_grd

@torch.jit.script
def activate_output(camera_center, density, rgb, grd, sh, indices, circumcenters, vertices, current_sh_deg, max_sh_deg):
    tets = vertices[indices]
    base_color_v0_raw, normed_grd = offset_normalize(rgb, grd, circumcenters, tets)
    tet_color_raw = eval_sh(
        tets.mean(dim=1),
        RGB2SH(base_color_v0_raw),
        sh.reshape(-1, (max_sh_deg+1)**2 - 1, 3),
        camera_center,
        current_sh_deg).float()
    base_color_v0 = torch.nn.functional.softplus(tet_color_raw.reshape(-1, 3, 1), beta=10)
    features = torch.cat([density, base_color_v0.reshape(-1, 3), normed_grd.reshape(-1, 3)], dim=1)
    return features.float()

class iNGPD(nn.Module):
    def __init__(self, 
                 scale_multi=0.5,
                 log2_hashmap_size=16,
                 base_resolution=16,
                 per_level_scale=2,
                 L=10,
                 hashmap_dim=4,
                 hidden_dim=64,
                 g_init=1,
                 s_init=1e-4,
                 d_init=0.1,
                 c_init=0.6,
                 density_offset=-4,
                 ablate_gradient=False,
                 **kwargs):
        super().__init__()
        self.scale_multi = scale_multi
        self.L = L
        self.dim = hashmap_dim
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.density_offset = density_offset
        self.ablate_gradient = ablate_gradient

        self.encoding = hashgrid.HashEmbedderOptimized(
            [torch.zeros((3)), torch.ones((3))],
            self.L, n_features_per_level=self.dim,
            log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution,
            finest_resolution=base_resolution*per_level_scale**self.L)

        offsets, pred_total = compute_grid_offsets(config, 3)
        total = list(self.encoding.parameters())[0].shape[0]
        # ic(offsets, pred_total, total)
        # assert total == pred_total, f"Pred #params: {pred_total} vs {total}"
        resolution = grid_scale(L-1, per_level_scale, base_resolution)
        self.different_size = 0
        self.nominal_offset_size = offsets[-1] - offsets[-2]
        for o1, o2 in zip(offsets[:-1], offsets[1:]):
            if o2 - o1 == self.nominal_offset_size:
                break
            else:
                self.different_size += 1
        self.offsets = offsets

        def mk_head(n):
            network = nn.Sequential(
                nn.Linear(self.encoding.n_output_dims, hidden_dim),
                nn.SELU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SELU(inplace=True),
                nn.Linear(hidden_dim, n)
            )
            gain = nn.init.calculate_gain('relu')  # for example, if using ReLU activations
            network.apply(lambda m: init_linear(m, gain))
            return network

        self.density_net   = mk_head(1)

    def _encode(self, x: torch.Tensor, cr: torch.Tensor):
        x = x.detach()
        output = self.encoding(x)#.float()
        output = output.reshape(-1, self.dim, self.L)
        cr = cr.half().detach() * self.scale_multi
        n = torch.arange(self.L, device=x.device).reshape(1, 1, -1)
        erf_x = safe_div(torch.tensor(1.0, device=x.device),
                         safe_sqrt(self.per_level_scale * 4*n*cr.reshape(-1, 1, 1)))
        scaling = torch.erf(erf_x).half()
        output = output * scaling
        return output


    def forward(self, x, cr):
        output = self._encode(x, cr)

        h = output.reshape(-1, self.L * self.dim)

        sigma = self.density_net(h)
        density = safe_exp(sigma+self.density_offset)
        return density


#"""
class iNGPDW(nn.Module):
    def __init__(self, 
                 sh_dim=0,
                 scale_multi=0.5,
                 log2_hashmap_size=16,
                 base_resolution=16,
                 per_level_scale=2,
                 L=10,
                 hashmap_dim=4,
                 hidden_dim=64,
                 g_init=1,
                 s_init=1e-4,
                 d_init=0.1,
                 c_init=0.6,
                 density_offset=-4,
                 **kwargs):
        super().__init__()
        self.scale_multi = scale_multi
        self.L = L
        self.dim = hashmap_dim
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.density_offset = density_offset

        self.config = dict(
            per_level_scale=per_level_scale,
            n_levels=L,
            otype="HashGrid",
            n_features_per_level=self.dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
        )
        # self.encoding = tcnn.Encoding(3, self.config)

        self.encoding = hashgrid.HashEmbedderOptimized(
            [torch.zeros((3)), torch.ones((3))],
            self.L, n_features_per_level=self.dim,
            log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution,
            finest_resolution=base_resolution*per_level_scale**self.L)


        def mk_head(n):
            network = nn.Sequential(
                nn.Linear(self.encoding.n_output_dims, hidden_dim),
                nn.SELU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SELU(inplace=True),
                nn.Linear(hidden_dim, n)
            )
            gain = nn.init.calculate_gain('relu')  # for example, if using ReLU activations
            network.apply(lambda m: init_linear(m, gain))
            return network

        self.network = mk_head(1+12+sh_dim)

        self.density_net   = mk_head(1)
        self.color_net     = mk_head(3)
        self.gradient_net  = mk_head(3)
        self.sh_net        = mk_head(sh_dim)

        last = self.network[-1]
        with torch.no_grad():
            last.weight[4:, :].zero_()
            last.bias[4:].zero_()
            for network, eps in zip(
                [self.gradient_net, self.sh_net, self.density_net, self.color_net], 
                [g_init, s_init, d_init, c_init]):
                last = network[-1]
                with torch.no_grad():
                    init.uniform_(last.weight.data, a=-eps, b=eps)
                    # nn.init.xavier_uniform_(m.weight, gain)
                    last.bias.zero_()


    def _encode(self, x: torch.Tensor, cr: torch.Tensor):
        x = x.detach()
        output = self.encoding(x)
        output = output.reshape(-1, self.dim, self.L)
        cr = cr.detach() * self.scale_multi
        n = torch.arange(self.L, device=x.device).reshape(1, 1, -1)
        erf_x = safe_div(torch.tensor(1.0, device=x.device),
                         safe_sqrt(self.per_level_scale * 4*n*cr.reshape(-1, 1, 1)))
        scaling = torch.erf(erf_x)
        output = output * scaling
        return output


    def forward(self, x, cr):
        output = self._encode(x, cr)

        h = output.reshape(-1, self.L * self.dim).float()

        sigma = self.density_net(h)
        rgb = self.color_net(h)
        # density_color_output = self.density_color_net(h)
        # sigma = density_color_output[:, :1]
        # rgb = density_color_output[:, 1:]
        field_samples = self.gradient_net(h)
        sh  = self.sh_net(h).half()

        rgb = rgb.reshape(-1, 3, 1) + 0.5
        density = safe_exp(sigma+self.density_offset)
        grd = torch.tanh(field_samples.reshape(-1, 1, 3)) / math.sqrt(3)
        # grd = field_samples.reshape(-1, 1, 3)
        # grd = rgb * torch.tanh(field_samples.reshape(-1, 3, 3))  # shape (T, 3, 3)
        return density, rgb.reshape(-1, 3), grd, sh

"""

class iNGPDW(nn.Module):
    def __init__(self, 
                 sh_dim=0,
                 scale_multi=0.5,
                 log2_hashmap_size=16,
                 base_resolution=16,
                 per_level_scale=2,
                 L=10,
                 hashmap_dim=4,
                 hidden_dim=64,
                 density_offset=-4,
                 **kwargs):
        super().__init__()
        self.scale_multi = scale_multi
        self.L = L
        self.dim = hashmap_dim
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.density_offset = density_offset
        self.sh_dim = sh_dim

        # --- Hash Grid Encoding ---
        self.config = dict(
            otype="HashGrid",
            n_levels=L,
            n_features_per_level=self.dim,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=per_level_scale,
        )
        self.encoding = tcnn.Encoding(3, self.config)
        
        print(f"TCNN supports JIT fusion: {tcnn.supports_jit_fusion()}")

        # --- Network Heads using TinyCUDANN ---
        n_input_dims = self.encoding.n_output_dims

        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": hidden_dim,
            "n_hidden_layers": 1,
        }
        sh_network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": hidden_dim,
            "n_hidden_layers": 1,
        }

        self.density_color_net = tcnn.Network(n_input_dims, 4, network_config)
        self.gradient_net = tcnn.Network(n_input_dims, 3, network_config)
        self.sh_net = tcnn.Network(n_input_dims, self.sh_dim, sh_network_config)


   def _encode(self, x: torch.Tensor, cr: torch.Tensor):
       x = x.detach()
       output = self.encoding(x).float()
       output = output.reshape(-1, self.dim, self.L)

       # Apply coarse-to-fine scaling based on camera ray radius 'cr'
       cr = cr.detach() * self.scale_multi
       n = torch.arange(self.L, device=x.device, dtype=torch.float32).reshape(1, 1, -1)
       erf_x = safe_div(torch.tensor(1.0, device=x.device, dtype=torch.float32),
                         safe_sqrt(self.per_level_scale * 4 * n * cr.reshape(-1, 1, 1)))
        scaling = torch.erf(erf_x)
        output = output * scaling
        return output

    def forward(self, x, cr):
        encoded_output = self._encode(x, cr)

        # Reshape for the network and ensure input is float32 for FullyFusedMLP
        h = encoded_output.reshape(-1, self.L * self.dim).float()

        # Query the network heads
        density_color_output = self.density_color_net(h)
        sigma = density_color_output[:, :1]
        rgb = density_color_output[:, 1:]

        field_samples = self.gradient_net(h)

        sh = self.sh_net(h).half()

        # --- Apply final activations and transformations ---
        density = safe_exp(sigma + self.density_offset)
        rgb = torch.sigmoid(rgb)
        grd = torch.tanh(field_samples.reshape(-1, 1, 3)) / math.sqrt(3)
        
        return density, rgb, grd, sh
# """
