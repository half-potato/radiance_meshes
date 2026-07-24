"""Shared pytest configuration for the radiance_meshes test suite.

Two jobs:

1.  Put the project root (``radiance_meshes/``) on ``sys.path`` so the tests can
    ``from utils...`` / ``from models...`` exactly like the training scripts do.
    ``[tool.pytest.ini_options] pythonpath`` in ``pyproject.toml`` already does
    this when pytest is launched from the project root; the explicit insert here
    keeps direct ``uv run tests/<file>.py`` invocations working too.

2.  Skip whole test modules *at collection time* when their hard requirements
    (a CUDA device, or the JAX quadrature reference) are unavailable, instead of
    letting the import blow up into a collection error. The kept tests are
    ``absltest`` / ``unittest`` classes, so we can't cheaply decorate individual
    methods -- gating by file is the pragmatic equivalent.
"""

import os
import sys
import importlib.util

# --- 1. make `from utils...` resolve -------------------------------------------------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
for _p in (_PROJECT_ROOT, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --- 2. capability probes ------------------------------------------------------------
def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _jax_available() -> bool:
    # The JAX reference path imports `jaxutil` (the quadrature renderer) and `jax`.
    return (
        importlib.util.find_spec("jax") is not None
        and importlib.util.find_spec("jaxutil") is not None
    )


def _warp_available() -> bool:
    return importlib.util.find_spec("warp") is not None


CUDA_AVAILABLE = _cuda_available()
JAX_AVAILABLE = _jax_available()
WARP_AVAILABLE = _warp_available()

# Per-file requirements. A file absent from this map is always collected, so a new
# test that needs nothing special "just works"; one that needs gpu/jax should be
# listed here so it skips cleanly off-rig rather than erroring at import.
_REQUIREMENTS = {
    "single_tet_test.py": ("gpu", "jax"),
    "multi_tet_test.py": ("gpu", "jax"),
    "raytrace_thin_tet_test.py": ("gpu", "jax"),
    "bvh_entry_test.py": ("gpu", "jax", "warp"),
    "warp_tracer_test.py": ("gpu", "jax", "warp"),
    "freeze_gradient_test.py": ("gpu",),
    "frozen_roundtrip_test.py": ("gpu",),
    "interval_parity_test.py": ("gpu",),
    "tile_overflow_test.py": ("gpu",),
    "tile_culling_test.py": ("gpu",),
    "cuda_alphablend_bwd_test.py": ("gpu",),
    # safe_math_test.py is pure CPU; always collect.
}


def _missing_requirement(filename: str):
    reqs = _REQUIREMENTS.get(filename, ())
    if "gpu" in reqs and not CUDA_AVAILABLE:
        return "no CUDA device available"
    if "jax" in reqs and not JAX_AVAILABLE:
        return "jax / jaxutil not importable"
    if "warp" in reqs and not WARP_AVAILABLE:
        return "warp (warp-lang) not importable"
    return None


def pytest_ignore_collect(collection_path, config):
    reason = _missing_requirement(collection_path.name)
    if reason is not None:
        # Surfaced in `-rs` summary output.
        print(f"[conftest] skipping {collection_path.name}: {reason}")
        return True
    return False
