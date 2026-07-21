"""Drop-in Triton-accelerated iNGP hash grid.

Wraps the Triton forward/backward kernels (utils/triton_hashgrid_fwd.py,
utils/triton_hashgrid_bwd.py) in a torch.autograd.Function and a module that
subclasses :class:`utils.hashgrid.HashEmbedderOptimized` — identical
parameters, buffers, and state_dict layout, so existing checkpoints load
unchanged and the swap is a construction-time flag.

Correctness contract (see the kernel files' self-tests):
* forward: BITWISE equal to the eager module (enable_fp_fusion=False keeps
  ptxas from fma-fusing the trilinear blend).
* grad_x: allclose vs eager autograd (different op order; rtol 1e-5).
* grad_table: bitwise where no atomic contention exists; allclose + per-table
  sum invariant under contention (eager's scatter is nondeterministic there
  too).

Why not tcnn: no support for torch.utils.checkpoint. This implementation
recomputes everything from ``x`` in backward (nothing saved but inputs), so
it is checkpoint-friendly and never materializes the eager path's
[N, 8, F]-per-level autograd intermediates (~GBs at rasterization scale).
"""

import torch

from utils.hashgrid import HashEmbedderOptimized
from utils.triton_hashgrid_fwd import hash_fwd, build_meta
from utils.triton_hashgrid_bwd import hash_bwd


class _TritonHashEncode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, meta, *tables):
        out = hash_fwd(x.contiguous(), list(tables), meta)
        ctx.save_for_backward(x, *tables)
        ctx.meta = meta
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, *tables = ctx.saved_tensors
        # dL/dx is only built when an upstream actually consumes it. The iNGP
        # feeds the encoder x.detach(), so this is False in all training/eval
        # paths and the kernel skips the whole grad_x tree (~1.25x faster bwd);
        # the gradcheck tests pass a leaf x and take the full path.
        need_grad_x = ctx.needs_input_grad[0]
        grad_x, grad_tables = hash_bwd(
            x.contiguous(), grad_out.contiguous(), list(tables), ctx.meta,
            need_grad_x=need_grad_x)
        grad_tables = [
            g if ctx.needs_input_grad[2 + i] else None
            for i, g in enumerate(grad_tables)
        ]
        return (grad_x, None, *grad_tables)


class TritonHashEmbedder(HashEmbedderOptimized):
    """HashEmbedderOptimized with the forward routed through Triton.

    State-dict compatible with the parent (no new parameters/buffers); the
    per-level launch metadata is rebuilt lazily whenever the device changes.
    """

    def _meta(self, device):
        cached = getattr(self, '_triton_meta', None)
        if cached is None or getattr(self, '_triton_meta_dev', None) != device:
            self._triton_meta = build_meta(self)
            self._triton_meta_dev = device
        return self._triton_meta

    # torch.compiler.disable: inductor tries to inline user-defined Triton
    # kernels into the compiled backbone graph and crashes on our int64
    # scalar args (InductorError 'int' has no attribute 'xreplace'). Fencing
    # the encoding out gives a clean graph break: heads stay compiled, the
    # _orig_mod state-dict prefix is preserved, and the kernels launch as-is.
    @torch.compiler.disable
    def forward(self, x):
        meta = self._meta(x.device)
        return _TritonHashEncode.apply(
            x, meta, *[emb.weight for emb in self.embeddings])

    # forward_in_chunks inherits and just works (calls self.forward), but the
    # Triton path has no [N,8,F] intermediates, so chunking is rarely needed.
