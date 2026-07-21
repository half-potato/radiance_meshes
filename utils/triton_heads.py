"""Fused Triton kernel for the four small MLP heads of ``iNGPDW``.

The model (models/ingp_color.py, class ``iNGPDW``) runs four independent
3-layer MLP heads on the *same* hash-grid feature ``h = [T, D]`` (D = L*dim =
8*8 = 64 in the train config)::

    density_net(h) -> [T, 1]
    color_net(h)   -> [T, 3]
    gradient_net(h)-> [T, 3]
    sh_net(h)      -> [T, sh_dim]   (sh_dim = 15 for max_sh_deg=3)

Each head is ``Sequential(Linear(64,64), SELU, Linear(64,64), SELU,
Linear(64,n))``.  As eager code this is ~12 cuBLAS GEMMs + SELU launches and
materializes the [T,64] intermediates.  This module fuses all four heads into
a single kernel that keeps every [BLOCK,64] activation in SRAM and never
writes the intermediates back to HBM.  Backward recomputes the intermediates
checkpoint-style (nothing but ``h`` and the packed weights is saved).

All matmuls use ``tl.dot(..., input_precision='tf32')`` -- the eager baseline
runs these in tf32 as well (``torch.backends.cuda.matmul.allow_tf32``), so the
kernel is compared against the eager-tf32 result and must sit within the
hardware's own tf32 noise.

Weight packing: the four heads have output widths ``nout = [1, 3, 3, sh_dim]``.
tl.dot needs N >= 16, so the third layer is padded to ``NPAD = 16`` columns.
The fused output is ``[T, NH*NPAD]`` and the caller slices the valid columns.

Public entry point: :func:`fused_heads`.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# SELU constants (torch.nn.SELU defaults).
_SCALE = 1.0507009873554805
_ALPHA = 1.6732632423543772


@triton.jit
def _selu(x):
    return 1.0507009873554805 * tl.where(
        x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))


@triton.jit
def _selu_grad(x):
    # selu'(x) = scale * (1 if x>0 else alpha*exp(x))
    return tl.where(x > 0, 1.0507009873554805,
                    1.0507009873554805 * 1.6732632423543772 * tl.exp(x))


@triton.jit
def _fused_fwd(h_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr,
               N,
               NH: tl.constexpr, D: tl.constexpr, HID: tl.constexpr,
               NPAD: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    m = rows < N
    dc = tl.arange(0, D)
    hc = tl.arange(0, HID)
    pc = tl.arange(0, NPAD)
    h = tl.load(h_ptr + rows[:, None] * D + dc[None, :], mask=m[:, None],
                other=0.0)
    for hd in tl.static_range(NH):
        w1 = tl.load(w1_ptr + hd * D * HID + dc[:, None] * HID + hc[None, :])
        a = _selu(tl.dot(h, w1, input_precision='tf32')
                  + tl.load(b1_ptr + hd * HID + hc)[None, :])
        w2 = tl.load(w2_ptr + hd * HID * HID + hc[:, None] * HID + hc[None, :])
        a = _selu(tl.dot(a, w2, input_precision='tf32')
                  + tl.load(b2_ptr + hd * HID + hc)[None, :])
        w3 = tl.load(w3_ptr + hd * HID * NPAD + hc[:, None] * NPAD + pc[None, :])
        o = tl.dot(a, w3, input_precision='tf32') \
            + tl.load(b3_ptr + hd * NPAD + pc)[None, :]
        tl.store(out_ptr + rows[:, None] * (NH * NPAD) + hd * NPAD + pc[None, :],
                 o, mask=m[:, None])


@triton.jit
def _fused_bwd(h_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr,
               w1t_ptr, w2t_ptr, w3t_ptr, go_ptr, gh_ptr,
               gw1_ptr, gb1_ptr, gw2_ptr, gb2_ptr, gw3_ptr, gb3_ptr, N,
               NH: tl.constexpr, D: tl.constexpr, HID: tl.constexpr,
               NPAD: tl.constexpr, BLOCK: tl.constexpr):
    # w{1,2,3}t are the *transposed* weights (w1t=[NH,HID,D], w2t=[NH,HID,HID],
    # w3t=[NH,NPAD,HID]).  Preloading them lets the grad-flow matmuls
    # (go@w3^T, gz2@w2^T, gz1@w1^T) skip an in-kernel tl.trans on the weights,
    # which was the dominant backward cost (~40% of the whole backward): a
    # shared-memory shuffle per head that the tf32 MMA otherwise stalls on.
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    m = rows < N
    dc = tl.arange(0, D)
    hc = tl.arange(0, HID)
    pc = tl.arange(0, NPAD)
    h = tl.load(h_ptr + rows[:, None] * D + dc[None, :], mask=m[:, None],
                other=0.0)
    gh = tl.zeros((BLOCK, D), dtype=tl.float32)
    for hd in tl.static_range(NH):
        # --- recompute forward (checkpoint) ---
        w1 = tl.load(w1_ptr + hd * D * HID + dc[:, None] * HID + hc[None, :])
        z1 = tl.dot(h, w1, input_precision='tf32') \
            + tl.load(b1_ptr + hd * HID + hc)[None, :]
        a1 = _selu(z1)
        w2 = tl.load(w2_ptr + hd * HID * HID + hc[:, None] * HID + hc[None, :])
        z2 = tl.dot(a1, w2, input_precision='tf32') \
            + tl.load(b2_ptr + hd * HID + hc)[None, :]
        a2 = _selu(z2)

        # grad_out for this head's columns (0 on masked rows and padded cols).
        go = tl.load(go_ptr + rows[:, None] * (NH * NPAD) + hd * NPAD
                     + pc[None, :], mask=m[:, None], other=0.0)

        # layer 3: o = a2 @ w3 + b3
        gw3 = tl.dot(tl.trans(a2), go, input_precision='tf32')      # [HID, NPAD]
        tl.atomic_add(gw3_ptr + hd * HID * NPAD + hc[:, None] * NPAD
                      + pc[None, :], gw3)
        gb3 = tl.sum(go, axis=0)                                    # [NPAD]
        tl.atomic_add(gb3_ptr + hd * NPAD + pc, gb3)

        w3t = tl.load(w3t_ptr + hd * NPAD * HID + pc[:, None] * HID
                      + hc[None, :])                                # [NPAD, HID]
        ga2 = tl.dot(go, w3t, input_precision='tf32')              # [BLOCK, HID]
        gz2 = ga2 * _selu_grad(z2)

        # layer 2: z2 = a1 @ w2 + b2
        gw2 = tl.dot(tl.trans(a1), gz2, input_precision='tf32')     # [HID, HID]
        tl.atomic_add(gw2_ptr + hd * HID * HID + hc[:, None] * HID
                      + hc[None, :], gw2)
        gb2 = tl.sum(gz2, axis=0)
        tl.atomic_add(gb2_ptr + hd * HID + hc, gb2)

        w2t = tl.load(w2t_ptr + hd * HID * HID + hc[:, None] * HID + hc[None, :])
        ga1 = tl.dot(gz2, w2t, input_precision='tf32')             # [BLOCK, HID]
        gz1 = ga1 * _selu_grad(z1)

        # layer 1: z1 = h @ w1 + b1
        gw1 = tl.dot(tl.trans(h), gz1, input_precision='tf32')      # [D, HID]
        tl.atomic_add(gw1_ptr + hd * D * HID + dc[:, None] * HID
                      + hc[None, :], gw1)
        gb1 = tl.sum(gz1, axis=0)
        tl.atomic_add(gb1_ptr + hd * HID + hc, gb1)

        w1t = tl.load(w1t_ptr + hd * HID * D + hc[:, None] * D + dc[None, :])
        gh += tl.dot(gz1, w1t, input_precision='tf32')             # [BLOCK, D]

    tl.store(gh_ptr + rows[:, None] * D + dc[None, :], gh, mask=m[:, None])


class FusedHeads(torch.autograd.Function):
    """Fused forward/backward for the four packed MLP heads.

    Inputs are already packed (host side, via :func:`fused_heads`):
    ``h`` [T, D], ``W1`` [NH, D, HID], ``B1`` [NH, HID], ``W2`` [NH, HID, HID],
    ``B2`` [NH, HID], ``W3`` [NH, HID, NPAD], ``B3`` [NH, NPAD].
    Returns ``out`` [T, NH*NPAD].
    """

    @staticmethod
    def forward(ctx, h, W1, B1, W2, B2, W3, B3):
        h = h.contiguous()
        W1 = W1.contiguous(); B1 = B1.contiguous()
        W2 = W2.contiguous(); B2 = B2.contiguous()
        W3 = W3.contiguous(); B3 = B3.contiguous()
        NH, D, HID = W1.shape
        NPAD = W3.shape[2]
        T = h.shape[0]
        out = torch.empty((T, NH * NPAD), device=h.device, dtype=torch.float32)
        BLOCK = 64
        grid = (triton.cdiv(T, BLOCK),)
        _fused_fwd[grid](h, W1, B1, W2, B2, W3, B3, out, T,
                         NH=NH, D=D, HID=HID, NPAD=NPAD, BLOCK=BLOCK,
                         num_warps=4)
        ctx.save_for_backward(h, W1, B1, W2, B2, W3, B3)
        ctx.dims = (NH, D, HID, NPAD)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        h, W1, B1, W2, B2, W3, B3 = ctx.saved_tensors
        NH, D, HID, NPAD = ctx.dims
        T = h.shape[0]
        grad_out = grad_out.contiguous()

        grad_h = torch.empty_like(h)
        gW1 = torch.zeros_like(W1); gB1 = torch.zeros_like(B1)
        gW2 = torch.zeros_like(W2); gB2 = torch.zeros_like(B2)
        gW3 = torch.zeros_like(W3); gB3 = torch.zeros_like(B3)

        # Transposed weights (tiny [NH,64,64] tensors) let the grad-flow
        # matmuls skip an in-kernel tl.trans -- the dominant backward cost.
        W1t = W1.transpose(1, 2).contiguous()
        W2t = W2.transpose(1, 2).contiguous()
        W3t = W3.transpose(1, 2).contiguous()

        BLOCK = 64
        grid = (triton.cdiv(T, BLOCK),)
        _fused_bwd[grid](h, W1, B1, W2, B2, W3, W1t, W2t, W3t, grad_out, grad_h,
                         gW1, gB1, gW2, gB2, gW3, gB3, T,
                         NH=NH, D=D, HID=HID, NPAD=NPAD, BLOCK=BLOCK,
                         num_warps=4)
        return grad_h, gW1, gB1, gW2, gB2, gW3, gB3


def _pad_last(x, npad):
    return F.pad(x, (0, npad - x.shape[-1]))


@torch.compiler.disable
def fused_heads(h, density_net, color_net, gradient_net, sh_net, sh_dim,
                npad=16):
    """Drop-in replacement for the four eager heads of ``iNGPDW``.

    ``torch.compiler.disable``: ``iNGPDW.forward`` is wrapped in
    ``torch.compile``; inductor tries to inline the custom Triton kernel and
    crashes on its int/constexpr scalar args (same InductorError class the
    hash-grid hit). Fencing this call gives a clean graph break -- the rest of
    the backbone stays compiled.

    Packs the weights of the four ``nn.Sequential`` heads (differentiably -- the
    real ``net[k].weight``/``bias`` Parameters are routed into the autograd
    graph via ``torch.stack``/``.t()``/``F.pad``, so their ``.grad`` populates
    on backward), runs :class:`FusedHeads`, and returns the *raw* head outputs::

        (density_raw [T,1], color_raw [T,3], grad_raw [T,3], sh [T,sh_dim])

    i.e. the exact tuple the heads produce BEFORE ``iNGPDW.forward``'s
    post-processing (no exp / +0.5 / normalize applied here).
    """
    heads = [density_net, color_net, gradient_net, sh_net]
    nout = [1, 3, 3, sh_dim]

    W1 = torch.stack([net[0].weight.t() for net in heads], 0)   # [NH, D, HID]
    B1 = torch.stack([net[0].bias for net in heads], 0)         # [NH, HID]
    W2 = torch.stack([net[2].weight.t() for net in heads], 0)   # [NH, HID, HID]
    B2 = torch.stack([net[2].bias for net in heads], 0)         # [NH, HID]
    W3 = torch.stack([_pad_last(net[4].weight.t(), npad)
                      for net in heads], 0)                     # [NH, HID, NPAD]
    B3 = torch.stack([_pad_last(net[4].bias, npad)
                      for net in heads], 0)                     # [NH, NPAD]

    out = FusedHeads.apply(h, W1, B1, W2, B2, W3, B3)

    density = out[:, 0:1]
    color = out[:, npad:npad + 3]
    grad = out[:, 2 * npad:2 * npad + 3]
    sh = out[:, 3 * npad:3 * npad + sh_dim]
    return density, color, grad, sh
