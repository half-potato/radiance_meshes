"""Debug gradient values: analytical vs numerical side-by-side."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tests.test_vk_backward import make_simple_scene, compute_loss, finite_diff_grad

renderer, cam_pos, vp, inv_vp, vertices, sh_coeffs, densities, color_grads = \
    make_simple_scene(n_verts=8, sh_degree=0, width=32, height=32)

# Get analytical gradients
for param_name in ["densities", "sh_coeffs", "color_grads"]:
    params = {
        "vertices": vertices.clone().requires_grad_(True),
        "sh_coeffs": sh_coeffs.clone().requires_grad_(True),
        "densities": densities.clone().requires_grad_(True),
        "color_grads": color_grads.clone().requires_grad_(True),
    }
    loss = compute_loss(renderer, cam_pos, vp, inv_vp, **params)
    loss.backward()
    analytical = params[param_name].grad

    numerical = finite_diff_grad(
        renderer, cam_pos, vp, inv_vp,
        vertices, sh_coeffs, densities, color_grads,
        param_name, eps=1e-3,
    )

    print(f"\n=== {param_name} ===")
    for i in range(min(analytical.numel(), 20)):
        a = analytical.view(-1)[i].item()
        n = numerical.view(-1)[i].item()
        if abs(a) > 1e-4 or abs(n) > 1e-4:
            ratio = a / n if abs(n) > 1e-6 else float('inf')
            print(f"  [{i:2d}] analytical={a:10.6f}  numerical={n:10.6f}  ratio={ratio:8.4f}")
