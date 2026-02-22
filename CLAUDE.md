# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Radiance Meshes / Delaunay Splatting — volumetric reconstruction using Delaunay tetrahedral meshes combined with instant neural graphics primitives (iNGP). An alternative to Gaussian Splatting that produces explicit mesh topology.

- Website: https://half-potato.gitlab.io/rm/
- Viewer: https://github.com/half-potato/vkrm

## Build & Run

Uses `uv` as the package manager. Requires CUDA 12.

```bash
uv pip install setuptools
uv pip install torch
# First run needs --no-build-isolation for slangtorch/tinycudann compilation:
uv run --no-build-isolation train.py -s <dataset_path> -o <output_path>
```

### Data Preparation

```bash
# Perspective cameras:
python convert3.py -s <source_path>
# Fisheye cameras (skips undistortion, preserves OPENCV_FISHEYE model):
python convert3.py -s <source_path> --fisheye
# With hloc/LightGlue matching (better for fisheye/wide-angle):
python convert3.py -s <source_path> --fisheye --matcher lightglue
# Insta360 dual-lens video:
python convert3.py -s <source_path> --fisheye --video input.insv --insta360
```

### Evaluation

```bash
python evaluate.py -o <output_path>
```

## Architecture

### Pipeline: Data → Delaunay Mesh → iNGP → Tiled Rasterization

1. **Data loading** (`data/`): Reads COLMAP or Blender datasets. `dataset_readers.py` maps COLMAP camera models (PINHOLE, OPENCV_FISHEYE, SIMPLE_RADIAL, etc.) to internal `ProjectionType` enum. Camera distortion params flow through to the Slang shaders.

2. **Model** (`models/ingp_color.py`): `iNGPDW` stores Delaunay tetrahedra vertices + per-tet attributes (density, rgb, gradient, SH coefficients). Uses tinycudann hash grid encoding (or fallback PyTorch `HashEmbedderOptimized`). Cell values are queried at circumcenters.

3. **Rendering** (`delaunay_rasterization/`): Three-stage tiled rasterizer written in Slang:
   - **Vertex shader** (`slang/vertex_shader.slang`): Projects tet vertices to screen, computes AABBs and circumcenters
   - **Tile shader** (`slang/tile_shader.slang`): Generates (tile_id, depth) keys for each tet-tile overlap, sorted via CUB
   - **Fragment shader** (`slang/alphablend_shader.slang`): Per-pixel ray-tet intersection + alpha blending. Camera projection/distortion in `slang/camera.slang`

4. **Densification** (`utils/densification.py`): Collects per-tet rendering statistics across views, clones/splits vertices in under-sampled regions. Controlled by `within_thresh`, `total_thresh`, and `budget`.

5. **Freezing** (`models/frozen.py`): At `freeze_start` iteration, topology is fixed (final high-precision Delaunay via scipy), empty tets culled, model transitions to `FrozenTetModel` for fine-tuning without topology changes.

### Troubleshooting

If training hangs during Slang shader compilation (no output after initial warnings), delete the slangtorch cache directories and `.lock` files:
```bash
find . -name ".slangtorch_cache" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.lock" -path "*/slang/*" -delete 2>/dev/null
```
This is needed when cache files become stale (e.g., after modifying `.slang` shader source files).

### Key Architectural Details

- **Delaunay updates** happen every `delaunay_interval` (default 10) iterations via gdel3d (GPU) during training, scipy (CPU) at freeze time.
- **Scene contraction** (`utils/contraction.py`) normalizes unbounded scenes into a unit ball.
- **SH degrees** are progressively enabled during training via `sh_interval`.
- **Slang shaders** are compiled at runtime by `slangtorch.loadModule()` with autodiff support for backprop.
- **Sort** uses CUB radix sort (`sort_by_keys/sort_by_keys_cub.py`) for tile-depth ordering.

### Camera Model Flow

`data/types.py` defines `ProjectionType`: PERSPECTIVE, FISHEYE, SIMPLE_RADIAL, PANORAMIC. These are read from COLMAP camera models in `dataset_readers.py` and passed through `data/camera.py` to the Slang `camera.slang` shader which handles projection and distortion.

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Main training loop |
| `evaluate.py` | Test-set metrics (PSNR, SSIM, LPIPS) |
| `convert3.py` | COLMAP preprocessing (preferred over convert.py) |
| `utils/args.py` | All training hyperparameters |
| `utils/train_util.py` | `render()` function, LR scheduling |
| `utils/densification.py` | Vertex cloning/splitting |
| `models/ingp_color.py` | Main model (`iNGPDW`) and optimizer |
| `models/frozen.py` | Frozen geometry model |
| `delaunay_rasterization/internal/alphablend_tiled_slang.py` | Core renderer entry point |
| `delaunay_rasterization/internal/slang/camera.slang` | Camera projection + distortion |
| `delaunay_rasterization/internal/slang/intersect.slang` | Ray-tet intersection |
| `data/dataset_readers.py` | COLMAP → CameraInfo parsing |

## Training Hyperparameters

Key non-obvious defaults (see `utils/args.py` for full list):
- `budget=2000000`: Target vertex count
- `freeze_start=18000`: When topology freezes
- `densify_start/end=2000/16000`, `densify_interval=500`
- `tile_size=4`: Tile dimensions for rasterizer (4 = 16x16 px blocks)
- `min_t=0.4`: Minimum ray intersection depth
- `lambda_ssim=0.2`: SSIM loss weight vs L1

Mobile/low-poly settings: `--budget 250000 --within_thresh 4 --total_thresh 15 --iterations 10000 --freeze_start 8000`
