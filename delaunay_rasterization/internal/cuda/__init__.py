import os
import torch
from torch.utils.cpp_extension import load

_root = os.path.dirname(__file__)

_wr = os.environ.get("CUDA_BWD_WARP_REDUCE", "1")
_mb = os.environ.get("CUDA_BWD_MIN_BLOCKS", "16")

# Hand-written CUDA backward for the Delaunay alphablend rasterizer.
# Mirrors delaunay_rasterization/internal/sort_by_keys build pattern.
# Build flags per design doc: -O3, NO --use_fast_math (match Slang IEEE
# exp/log + safe_* rules), -lineinfo, ptxas -v for register reporting.
_name = "alphablend_bwd_cuda" if (_wr == "1" and _mb == "16") else f"alphablend_bwd_cuda_wr{_wr}_mb{_mb}"
alphablend_bwd_cuda = load(
    name=_name,
    sources=[os.path.join(_root, "alphablend_bwd.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-lineinfo",
        "--ptxas-options=-v",
        "-U__CUDA_NO_HALF_OPERATORS__",
        f"-DUSE_WARP_REDUCE={_wr}",
        f"-DMIN_BLOCKS_PER_SM={_mb}",
    ],
    verbose=True,
)
