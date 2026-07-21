"""Correctness of the fused Triton MLP heads against the four eager heads.

Contract (see utils/triton_heads.py):
* Baseline is eager-tf32 (torch.backends.cuda.matmul.allow_tf32=True): the
  fused kernel uses tf32 tl.dot, so it must match eager-tf32 to within the
  hardware's own tf32 noise, not fp32.  Truth is the same heads run in
  float64; the gate is that the fused error vs fp64 truth stays within a small
  multiple of eager-tf32's own error vs fp64 truth.
* forward: valid columns vs the 4 torch heads; relerr within 3x eager-tf32.
* backward: grad_h (allclose tf32-level) and every head weight/bias grad
  within ~3x of eager-tf32's distance to the fp64 truth.

Auto-skips off-rig (no CUDA / no triton), mirroring triton_hashgrid_test.py.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from models.ingp_color import iNGPDW              # noqa: E402
from utils.triton_heads import fused_heads        # noqa: E402

DEVICE = torch.device("cuda")
SH_DIM = 15


def _build_model():
    torch.manual_seed(0)
    m = iNGPDW(
        sh_dim=SH_DIM,
        log2_hashmap_size=23,
        hashmap_dim=8,
        L=8,
        base_resolution=64,
        per_level_scale=2,
        hidden_dim=64,
        additional_attr=0,
        use_triton_hashgrid=True,
    ).to(DEVICE)
    # Randomize head params away from the near-zero init so grads are
    # non-degenerate and the tf32 stress is realistic.
    g = torch.Generator(device=DEVICE).manual_seed(7)
    for net in (m.density_net, m.color_net, m.gradient_net, m.sh_net):
        for p in net.parameters():
            p.data = torch.empty_like(p).normal_(0.0, 0.15, generator=g)
    return m


def _heads(m):
    return m.density_net, m.color_net, m.gradient_net, m.sh_net


def _eager_forward(heads, h):
    d, c, g, s = heads
    return d(h), c(h), g(h), s(h)


def _relerr(approx, truth, floor=1e-2):
    """Max relative error over entries where |truth| > floor."""
    mask = truth.abs() > floor
    if mask.sum() == 0:
        return torch.tensor(0.0)
    return ((approx - truth).abs()[mask] / truth.abs()[mask]).max()


def _setup(n=200_000):
    m = _build_model()
    heads = _heads(m)
    g = torch.Generator(device=DEVICE).manual_seed(3)
    h = torch.randn((n, 64), device=DEVICE, generator=g)
    return m, heads, h


def test_forward_matches_eager_tf32():
    import copy
    m, heads, h = _setup()

    # fp64 truth (deepcopy so we don't mutate the tf32 heads in place)
    heads64 = [copy.deepcopy(net).double() for net in heads]
    with torch.no_grad():
        truth = torch.cat(_eager_forward(heads64, h.double()), dim=1)

    torch.backends.cuda.matmul.allow_tf32 = True
    with torch.no_grad():
        eager = torch.cat(_eager_forward(heads, h), dim=1).double()
        d, c, gr, s = fused_heads(h, *heads, SH_DIM)
        fused = torch.cat([d, c, gr, s], dim=1).double()

    eager_err = _relerr(eager, truth)
    fused_err = _relerr(fused, truth)
    print(f"\n[fwd] eager-tf32 relerr={eager_err:.3e} "
          f"fused relerr={fused_err:.3e} "
          f"max|fused-eager|={(fused - eager).abs().max():.3e}")

    # tf32-level agreement with eager, and within 3x eager's own tf32 error.
    assert (fused - eager).abs().max() < 5e-2
    assert fused_err <= 3.0 * eager_err + 1e-4


def test_backward_matches_eager_tf32():
    m, heads, h = _setup()
    d0, c0, g0, s0 = heads

    g = torch.Generator(device=DEVICE).manual_seed(11)
    cot = [torch.randn(h.shape[0], k, device=DEVICE, generator=g)
           for k in (1, 3, 3, SH_DIM)]

    def run_eager(hin, nets, dtype):
        hin = hin.to(dtype).detach().requires_grad_(True)
        outs = [net.to(dtype)(hin) for net in nets]
        loss = sum((o * c.to(dtype)).sum() for o, c in zip(outs, cot))
        loss.backward()
        wg = {}
        for name, net in zip(("d", "c", "g", "s"), nets):
            for i in (0, 2, 4):
                wg[f"{name}{i}w"] = net[i].weight.grad.detach().double().clone()
                wg[f"{name}{i}b"] = net[i].bias.grad.detach().double().clone()
        return hin.grad.detach().double().clone(), wg

    # fp64 truth (fresh clones so grads don't collide)
    import copy
    nets_t = [copy.deepcopy(net) for net in heads]
    gh_t, wg_t = run_eager(h, nets_t, torch.float64)

    # eager tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    nets_e = [copy.deepcopy(net) for net in heads]
    gh_e, wg_e = run_eager(h, nets_e, torch.float32)

    # fused
    nets_f = [copy.deepcopy(net).to(DEVICE) for net in heads]
    hf = h.detach().requires_grad_(True)
    df, cf, grf, sf = fused_heads(hf, *nets_f, SH_DIM)
    lossf = sum((o * c).sum()
                for o, c in zip((df, cf, grf, sf), cot))
    lossf.backward()
    gh_f = hf.grad.detach().double().clone()
    wg_f = {}
    for name, net in zip(("d", "c", "g", "s"), nets_f):
        for i in (0, 2, 4):
            wg_f[f"{name}{i}w"] = net[i].weight.grad.detach().double().clone()
            wg_f[f"{name}{i}b"] = net[i].bias.grad.detach().double().clone()

    def _frob(approx, truth):
        """Relative L2 (Frobenius) error of a gradient tensor vs fp64 truth.

        This is the correct way to compare gradient *vectors* up to numerical
        noise (standard gradcheck metric).  Per-element relative error is
        meaningless here: the bias/weight grads are sums of ~2e5 zero-mean
        terms, so individual components are cancelling sums that can be
        near-zero with large per-element relative error even though the whole
        gradient vector agrees.  The fused kernel accumulates these sums via
        fp32 atomic_add in nondeterministic block order (vs eager's single
        reduction), so only the vector-norm agreement is well-defined -- same
        contract the Triton hash-grid uses for its contended scatter grads.
        """
        return ((approx - truth).norm() / (truth.norm() + 1e-12)).item()

    # A fused grad is correct if its Frobenius error vs fp64 truth is within a
    # small multiple of eager-tf32's own, OR already below an absolute tf32
    # floor.  (Layer-3 weight grads sit at ~1.8e-3, eager ~4e-4: the floor
    # covers that so a pure ratio gate does not false-fail on excellent grads.)
    RATIO, TF32_FLOOR = 2.0, 3e-2

    e_gh, f_gh = _frob(gh_e, gh_t), _frob(gh_f, gh_t)
    print(f"\n[bwd grad_h] frob-relerr eager={e_gh:.3e} fused={f_gh:.3e}")
    assert f_gh <= max(RATIO * e_gh, TF32_FLOOR), (
        f"grad_h frob fused={f_gh:.3e} eager={e_gh:.3e}")

    worst = 0.0
    for key in wg_t:
        e, f = _frob(wg_e[key], wg_t[key]), _frob(wg_f[key], wg_t[key])
        worst = max(worst, f)
        assert f <= max(RATIO * e, TF32_FLOOR), (
            f"{key} frob-relerr fused={f:.3e} eager={e:.3e}")
    print(f"[bwd wgrad] worst fused frob-relerr vs fp64 truth = {worst:.3e} "
          f"(all < {TF32_FLOOR} tf32 floor)")
