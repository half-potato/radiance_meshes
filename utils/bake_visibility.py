"""Offline precompute of per-region occlusion visibility (PVS) for the rmesh_rs viewer.

Clusters the training cameras into K spatial regions, and for each region records
which tets are visible (RadSplat peak_contrib = max over rays of alpha*transmittance)
from any camera assigned to that region. Emits three .npy files that
`rmesh_rs/convert.py --visibility <dir>` bakes into the .rmesh as tagged sections:

  tet_visibility.npy   [K, M] uint8   (1 = tet visible from region k)   -- M = model.indices count
  region_centroids.npy [K, 3] float32 (camera-center centroid, viewer/model frame)
  region_radii.npy     [K]    float32 (coverage radius per region)

The viewer picks the nearest region to the current camera and skips occluded tets.
Camera centers are taken from `camera.camera_center` (the loader's normalized/model
frame), which matches the .rmesh vertex frame the viewer renders in.

Run from radiance_meshes/:
  CUDA_VISIBLE_DEVICES=2 uv run --no-build-isolation utils/bake_visibility.py \
    --ckpt output/<model> --dataset_path /path/to/scene --image_folder images_4 \
    --output_dir output/<model>/vis --num_regions 24
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from data import loader
from models.ingp_color import Model
from models.frozen import FrozenTetModel
from delaunay_rasterization.internal.render_err import render_err


def load_any_ckpt(path: Path, device):
    """Load either a live iNGP Model or a frozen (post-freeze) FrozenTetModel,
    mirroring evaluate.py's fallback."""
    try:
        return Model.load_ckpt(path, device)
    except Exception as e:
        print(f"  (iNGP load failed: {type(e).__name__}; trying FrozenTetModel)")
        return FrozenTetModel.load_ckpt(path, device)


def kmeans(points: np.ndarray, k: int, iters: int = 50, seed: int = 0):
    """Minimal Lloyd's k-means (numpy). points [C,3] -> (labels [C], centroids [k,3])."""
    rng = np.random.default_rng(seed)
    c = points.shape[0]
    k = min(k, c)
    # k-means++ style seeding: first random, rest by farthest.
    centroids = np.empty((k, 3), dtype=np.float64)
    centroids[0] = points[rng.integers(c)]
    d2 = ((points - centroids[0]) ** 2).sum(1)
    for i in range(1, k):
        probs = d2 / max(d2.sum(), 1e-12)
        centroids[i] = points[rng.choice(c, p=probs)]
        d2 = np.minimum(d2, ((points - centroids[i]) ** 2).sum(1))
    labels = np.zeros(c, dtype=np.int64)
    for _ in range(iters):
        dists = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(2)  # [C,k]
        new_labels = dists.argmin(1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for j in range(k):
            m = labels == j
            if m.any():
                centroids[j] = points[m].mean(0)
    return labels, centroids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="Model checkpoint dir (ckpt.pth + config.json)")
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--image_folder", default="images_4")
    ap.add_argument("--output_dir", required=True, help="Where to write the .npy files")
    ap.add_argument("--num_regions", type=int, default=24)
    ap.add_argument("--thresh", type=float, default=0.01,
                    help="peak_contrib (max alpha*T) above which a tet counts as visible")
    ap.add_argument("--min_t", type=float, default=None,
                    help="near-plane for render_err; defaults to the model's config min_t")
    ap.add_argument("--tile_size", type=int, default=16)
    ap.add_argument("--resolution", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading cameras from {args.dataset_path} ({args.image_folder})...")
    # eval=False -> all images returned as train cameras (full coverage for regions).
    train_cameras, _test, _scene = loader.load_dataset(
        args.dataset_path, args.image_folder, data_device="cpu",
        eval=False, resolution=args.resolution)
    print(f"  {len(train_cameras)} cameras")

    print(f"Loading model from {args.ckpt}...")
    model = load_any_ckpt(Path(args.ckpt), device)
    M = int(model.indices.shape[0])
    min_t = args.min_t if args.min_t is not None else float(getattr(model, "min_t", 0.2))
    print(f"  {M} tets, min_t={min_t}")

    # Camera centers in the model/viewer frame.
    centers = np.stack([c.camera_center.detach().cpu().numpy().reshape(3) for c in train_cameras])

    K = min(args.num_regions, len(train_cameras))
    print(f"K-means into {K} regions...")
    labels, centroids = kmeans(centers, K)
    radii = np.zeros(K, dtype=np.float32)
    for j in range(K):
        m = labels == j
        if m.any():
            radii[j] = np.sqrt(((centers[m] - centroids[j]) ** 2).sum(1)).max()
    counts = np.bincount(labels, minlength=K)
    print(f"  region sizes: min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")

    vis = np.zeros((K, M), dtype=np.uint8)
    with torch.no_grad():
        for j in range(K):
            cam_ids = np.nonzero(labels == j)[0]
            peak = torch.zeros(M, device=device)
            for ci in tqdm(cam_ids, desc=f"region {j+1}/{K}", leave=False):
                cam = train_cameras[int(ci)]
                target = cam.original_image.cuda()
                gt_mask = cam.gt_alpha_mask.cuda()
                _, extras = render_err(target, gt_mask, cam, model,
                                       tile_size=args.tile_size, min_t=min_t)
                max_T = extras["tet_count"][..., 1].float() / 65535.0
                peak = torch.maximum(peak, max_T)
            visible = (peak > args.thresh)
            vis[j] = visible.cpu().numpy().astype(np.uint8)
            frac = float(visible.float().mean())
            print(f"  region {j}: {int(visible.sum())}/{M} visible ({100*frac:.1f}%)")

    np.save(out / "tet_visibility.npy", vis)
    np.save(out / "region_centroids.npy", centroids.astype(np.float32))
    np.save(out / "region_radii.npy", radii.astype(np.float32))
    overall = (vis.max(0) > 0).mean()
    print(f"Saved {out}/tet_visibility.npy [{K},{M}] u8, centroids, radii.")
    print(f"Union over all regions covers {100*overall:.1f}% of tets "
          f"(the rest are never visible from any region and are always culled).")


if __name__ == "__main__":
    main()
