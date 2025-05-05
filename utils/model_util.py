import torch
from torch import nn
from utils import hashgrid

from utils.graphics_utils import l2_normalize_th
from utils.topo_utils import calculate_circumcenters_torch, fibonacci_spiral_on_sphere, calc_barycentric, sample_uniform_in_sphere, project_points_to_tetrahedra, contraction_jacobian_d_in_chunks
from utils.safe_math import safe_exp, safe_div, safe_sqrt
from utils.contraction import contract_mean_std
from utils.contraction import contract_points, inv_contract_points
from sh_slang.eval_sh import eval_sh
from utils.hashgrid import HashEmbedderOptimized

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
def pre_calc_cell_values(vertices, indices, center, scene_scaling: float):
    device = vertices.device
    tets = vertices[indices]
    circumcenter, radius = calculate_circumcenters_torch(tets.double())
    # clipped_circumcenter = project_points_to_tetrahedra(circumcenter.float(), tets)
    clipped_circumcenter = circumcenter
    normalized = (clipped_circumcenter - center) / scene_scaling
    cv, cr = contract_mean_std(normalized, radius / scene_scaling)
    return clipped_circumcenter, cv.float(), cr, normalized

@torch.jit.script
def compute_vertex_colors_from_field(triangle_verts, base, gradients, circumcenters):
    """
    Compute per-vertex colors for each triangle.
    
    For each vertex:
      color = base + dot(gradient, (vertex - circumcenter))
    
    - The first 3 coefficients of field_samples provide the base color.
    - The next 6 coefficients (reshaped as (3,2)) define the gradients.
    """
    offsets = triangle_verts - circumcenters[:, None, :]  # shape (T, 4, 3)
    offsets = l2_normalize_th(offsets)
    grad_contrib = torch.einsum('tcd,tvd->tvc', gradients, offsets)
    vertex_colors = base[:, None, :] + grad_contrib
    return vertex_colors, gradients

def activate_output(camera_center, density, rgb, grd, sh, indices, circumcenters, vertices, current_sh_deg, max_sh_deg, density_offset:float):
    # subtract 0.5 to remove 0th order spherical harmonic
    tet_color_raw = eval_sh(
        circumcenters,
        torch.zeros((rgb.shape[0], 3), device=vertices.device),
        sh.reshape(-1, (max_sh_deg+1)**2 - 1, 3),
        camera_center,
        current_sh_deg) - 0.5
    vcolors, _ = compute_vertex_colors_from_field(
        vertices[indices].detach(), rgb.float(), grd.float(), circumcenters.float().detach())
    vcolors = torch.nn.functional.softplus(vcolors + tet_color_raw.reshape(-1, 1, 3), beta=10)
    vcolors = vcolors.reshape(-1, 12)
    features = torch.cat([density, vcolors], dim=1)
    return features

class iNGPDW(nn.Module):
    """
    Hash‑grid backbone with four independent heads
      • density     (1)
      • base colour (3)
      • colour‑gradient (3×3 = 9)
      • SH coeffs   (sh_dim)
    Multisamples `k_samples` Gaussian jitters inside each circumsphere and
    averages the encoded features (Zip‑NeRF style).
    """
    def __init__(self,
                 sh_dim:        int   = 0,
                 k_samples:     int   = 4,
                 trunc_sigma:   float = 0.30,   # radius·sigma of Gaussian
                 scale_multi:   float = 0.5,
                 log2_hashmap_size: int = 16,
                 base_resolution:    int = 16,
                 per_level_scale:    int = 2,
                 L:              int   = 10,
                 hashmap_dim:    int   = 4,
                 hidden_dim:     int   = 64,
                 density_offset: float = -4,
                 **kwargs):
        super().__init__()

        # ---------------- parameters used in forward -------------------------
        self.k_samples      = k_samples
        self.trunc_sigma    = trunc_sigma
        self.scale_multi    = scale_multi
        self.L              = L
        self.dim            = hashmap_dim
        self.per_level_scale= per_level_scale
        self.base_resolution= base_resolution
        self.density_offset = density_offset

        # ---------------- hash‑grid encoder ----------------------------------
        self.encoding = HashEmbedderOptimized(
            [torch.zeros((3)), torch.ones((3))],
            L, n_features_per_level=hashmap_dim,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=base_resolution * per_level_scale ** L
        )

        # -------------- shared MLP template ----------------------------------
        def mk_head(out_channels: int) -> nn.Sequential:
            head = nn.Sequential(
                nn.Linear(self.encoding.n_output_dims, hidden_dim),
                nn.SELU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SELU(inplace=True),
                nn.Linear(hidden_dim, out_channels)
            )
            head.apply(lambda m: init_linear(m, nn.init.calculate_gain('relu')))
            return head

        # ---------------- four heads -----------------------------------------
        self.density_net   = mk_head(1)
        self.color_net     = mk_head(3)
        self.gradient_net  = mk_head(9)
        self.sh_net        = mk_head(sh_dim)
        self.backup_net        = mk_head(1+3+9+sh_dim)

        # optional: zero‑init non‑density/colour outputs
        with torch.no_grad():
            for net in (self.gradient_net, self.sh_net):
                net[-1].weight.zero_()
                net[-1].bias.zero_()
            self.backup_net[-1].weight[4:].zero_()
            self.backup_net[-1].bias[4:].zero_()

    # ╭────────────────── helper: encode  ────────────────────────────────────╮
    def _encode(self, x: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
        h = self.encoding(x).float()                         # (B, dim, L)
        cr = cr.float()
        h = h.reshape(-1, self.dim, self.L)

        n = torch.arange(self.L, device=x.device).view(1,1,-1)  # (1,1,L)
        erf_x   = safe_div(torch.tensor(1.0, device=x.device),
                           safe_sqrt(self.per_level_scale * 4*n * cr.view(-1,1,1)))
        h       = (h * torch.erf(erf_x)).reshape(-1, self.L * self.dim)
        return h                                             # (B, F)
    # ╰────────────────────────────────────────────────────────────────────────╯

    # ╭────────────────── helper: decode  ────────────────────────────────────╮
    def _decode(self, h: torch.Tensor) -> torch.Tensor:
        output = self.backup_net(h)
        sigma = output[:, :1]
        temp = output[:, 1:12+1]
        rgb = temp[:, :3]
        field_samples = temp[:, 3:12]
        sh = output[:, 13:]

        # sigma   = self.density_net(h)
        # rgb = self.color_net(h)
        # field_samples = self.gradient_net(h)
        # sh  = self.sh_net(h)

        rgb = rgb.reshape(-1, 3, 1) + 0.5
        density = safe_exp(sigma+self.density_offset)
        grd = rgb * torch.tanh(field_samples.reshape(-1, 3, 3))  # shape (T, 3, 3)

        return density, rgb.reshape(-1, 3), grd, sh
    # ╰────────────────────────────────────────────────────────────────────────╯

    # ╭──────────────────────────── forward  ─────────────────────────────────╮
    def forward(self, x: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x  : contracted coords in [0,1]^3   (B,3)   ➊
        cr : contracted radius/std          (B,)    ➋

        ➊ Your caller already uses  x = (cv/2 + 1)/2
        ➋ cr is the same radius that went into contract_mean_std
        """

        if self.k_samples > 1:
            # ------- convert back to contracted space centred at 0 -----------
            cv = 4.0 * x - 2.0                        # (B,3)  in [-1,1]

            # ------- draw Gaussian jitters inside the sphere -----------------
            eps = torch.randn((cv.shape[0], self.k_samples, 3),
                              device=cv.device)
            eps = eps * (self.trunc_sigma * cr).view(-1,1,1)   # scale
            samples_cv = (cv.unsqueeze(1) + eps)               # (B,k,3)
            samples_cv.clamp_(-1.0, 1.0)

            # ------- map back to [0,1]^3 -------------------------------------
            samples_x = samples_cv / 4.0 + 0.5                  # (B,k,3)
            samples_xf = samples_x.view(-1, 3)                  # (B*k,3)
            samples_cr = cr.repeat_interleave(self.k_samples)   # (B*k,)

            # ------- encode every jitter and mean in feature space -----------
            h = self._encode(samples_xf, samples_cr)            # (B*k,F)
            h = h.view(cv.shape[0], self.k_samples, -1).mean(dim=1)
        else:
            h = self._encode(x, cr)

        return self._decode(h)
    # ╰────────────────────────────────────────────────────────────────────────╯


class iNGPDW2(nn.Module):
    def __init__(self, 
                 sh_dim=0,
                 scale_multi=0.5,
                 log2_hashmap_size=16,
                 base_resolution=16,
                 per_level_scale=2,
                 k_samples=1,
                 trunc_sigma=0.3,
                 L=10,
                 hashmap_dim=4,
                 hidden_dim=64,
                 density_offset=-4,
                 **kwargs):
        super().__init__()
        self.k_samples = k_samples
        self.trunc_sigma = trunc_sigma
        self.scale_multi = scale_multi
        self.L = L
        self.dim = hashmap_dim
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.density_offset = density_offset

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
        self.gradient_net  = mk_head(9)
        self.sh_net        = mk_head(sh_dim)

        last = self.network[-1]
        with torch.no_grad():
            last.weight[4:, :].zero_()
            last.bias[4:].zero_()
        for network in [self.gradient_net, self.sh_net]:
            last = network[-1]
            with torch.no_grad():
                last.weight.zero_()
                last.bias.zero_()

    def _encode(self, x: torch.Tensor, cr: torch.Tensor):
        x = x.detach()
        output = self.encoding(x).float()
        output = output.reshape(-1, self.dim, self.L)
        cr = cr.float() * self.scale_multi
        n = torch.arange(self.L, device=x.device).reshape(1, 1, -1)
        erf_x = safe_div(torch.tensor(1.0, device=x.device), safe_sqrt(self.per_level_scale * 4*n*cr.reshape(-1, 1, 1)))
        scaling = torch.erf(erf_x)
        output = output * scaling
        # sphere_area = 4/3*math.pi*cr**3
        # scaling = safe_div(base_resolution * per_level_scale**n, sphere_area.reshape(-1, 1, 1)).clip(max=1)
        return output


    def forward(self, x, cr):
        if self.k_samples == 1:
            output = self._encode(x, cr)
        else:
            cv = 4.0 * x - 2.0                        # (B,3)  in [-1,1]

            # ------- draw Gaussian jitters inside the sphere -----------------
            eps = torch.randn((cv.shape[0], self.k_samples, 3),
                              device=cv.device)
            eps = eps * (self.trunc_sigma * cr).view(-1,1,1)   # scale
            samples_cv = (cv.unsqueeze(1) + eps)               # (B,k,3)
            samples_cv.clamp_(-1.0, 1.0)

            # ------- map back to [0,1]^3 -------------------------------------
            samples_x = samples_cv / 4.0 + 0.5                  # (B,k,3)
            samples_xf = samples_x.view(-1, 3)                  # (B*k,3)
            samples_cr = cr.repeat_interleave(self.k_samples)   # (B*k,)
            output = self._encode(samples_xf, samples_cr)            # (B*k,F)
            output = output.view(cv.shape[0], self.k_samples, -1).mean(dim=1)

        h = output.reshape(-1, self.L * self.dim)
        # output = self.network(h)
        # sigma = output[:, :1]
        # temp = output[:, 1:12+1]
        # rgb = temp[:, :3]
        # field_samples = temp[:, 3:12]
        # sh = output[:, 13:]

        sigma   = self.density_net(h)
        rgb = self.color_net(h)
        field_samples = self.gradient_net(h)
        sh  = self.sh_net(h)

        rgb = rgb.reshape(-1, 3, 1) + 0.5
        density = safe_exp(sigma+self.density_offset)
        grd = rgb * torch.tanh(field_samples.reshape(-1, 3, 3))  # shape (T, 3, 3)
        return density, rgb.reshape(-1, 3), grd, sh
