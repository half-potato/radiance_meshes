#!/usr/bin/env python
"""Compare per-image render quality across radiance-mesh runs.

Reads the gt/pred PNGs and ``metrics.json`` that ``train.py``'s
``evaluate_and_save`` writes into each output dir, then:

  1. builds a per-image comparison table (PSNR/SSIM/LPIPS per run + deltas),
  2. quantifies *where* error lives -- in particular how concentrated it is in
     bright / specular regions (the shiny table in mip-NeRF 360 ``garden``),
  3. renders diagnostic panels: GT | preds | error maps | improvement map,
     plus an auto-zoomed crop on the highest-error region.

Everything is computed from the PNGs on disk -- no model load, no CUDA.

Example
-------
    uv run python scripts/analyze_renders.py \
        --runs output/garden_ifimages_4:baseline \
               output/garden_sh_fp32_only:fp32 \
        --out analysis/garden
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Rec. 601 luma -- "brightness" proxy for finding specular highlights.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def parse_run(spec: str):
    """``path[:label]`` -> (Path, label)."""
    if ":" in spec:
        path, label = spec.rsplit(":", 1)
    else:
        path, label = spec, Path(spec).name
    return Path(path), label


def load_metrics(run_dir: Path, split: str) -> dict:
    """Return {idx: {PSNR, SSIM, LPIPS}} from a run's metrics.json."""
    data = json.loads((run_dir / "metrics.json").read_text())
    out = {}
    for key, val in data.items():
        prefix = f"{split}_"
        if not key.startswith(prefix):
            continue
        tail = key[len(prefix):]
        if tail == "mean":
            continue
        out[int(tail)] = val
    return out


def img_paths(run_dir: Path, split: str):
    gt_dir = run_dir / "images" / "gt" / split
    pred_dir = run_dir / "images" / "pred" / split
    idxs = sorted(int(p.stem) for p in pred_dir.glob("*.png"))
    return gt_dir, pred_dir, idxs


def load_img(path: Path, max_w: int | None = None) -> np.ndarray:
    """Load PNG as float32 HxWx3 in [0,1], optionally downscaled to max_w wide."""
    arr = imageio.imread(path).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    arr = arr[..., :3]
    if max_w is not None and arr.shape[1] > max_w:
        # cheap box-ish downscale via PIL for clean viz
        from PIL import Image
        h = round(arr.shape[0] * max_w / arr.shape[1])
        arr = np.asarray(
            Image.fromarray((arr * 255).astype(np.uint8)).resize((max_w, h), Image.BILINEAR)
        ).astype(np.float32) / 255.0
    return arr


def err_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-pixel mean-abs RGB error, HxW."""
    return np.abs(gt - pred).mean(-1)


def highlight_stats(gt: np.ndarray, err: np.ndarray, pct: float = 90.0) -> dict:
    """How concentrated is error in the brightest `100-pct`% of GT pixels?"""
    lum = gt @ LUMA
    thr = np.percentile(lum, pct)
    bright = lum >= thr
    bright_err = float(err[bright].mean()) if bright.any() else 0.0
    rest_err = float(err[~bright].mean()) if (~bright).any() else 0.0
    # mean GT brightness of the worst-1% error pixels (are failures on bright stuff?)
    k = max(1, err.size // 100)
    worst = np.argpartition(err.ravel(), -k)[-k:]
    worst_lum = float(lum.ravel()[worst].mean())
    return {
        "bright_err": bright_err,
        "rest_err": rest_err,
        "bright_over_rest": bright_err / (rest_err + 1e-8),
        "worst1pct_lum": worst_lum,
        "mean_lum": float(lum.mean()),
    }


def worst_bbox(err: np.ndarray, pct: float = 99.0, margin: float = 0.08):
    """Bounding box around the highest-error pixels, with a relative margin."""
    thr = np.percentile(err, pct)
    ys, xs = np.where(err >= thr)
    if len(ys) == 0:
        return None
    h, w = err.shape
    my, mx = int(h * margin), int(w * margin)
    y0, y1 = max(0, ys.min() - my), min(h, ys.max() + my)
    x0, x1 = max(0, xs.min() - mx), min(w, xs.max() + mx)
    return y0, y1, x0, x1


def make_panel(idx, gt, preds, labels, metrics_by_run, out_path, err_vmax, crop=None):
    """One figure: row0 = GT + preds, row1 = error maps + improvement map."""
    n = len(preds)
    errs = [err_map(gt, p) for p in preds]
    ncols = n + 1
    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 7.0))

    def show(ax, im, title, **kw):
        ax.imshow(im, **kw)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    show(axes[0, 0], gt, f"GT  (img {idx:04d})")
    axes[1, 0].axis("off")
    for j, (pred, lab, err) in enumerate(zip(preds, labels, errs), start=1):
        m = metrics_by_run[lab].get(idx, {})
        psnr = m.get("PSNR", float("nan"))
        lp = m.get("LPIPS", float("nan"))
        show(axes[0, j], np.clip(pred, 0, 1), f"{lab}\nPSNR {psnr:.2f}  LPIPS {lp:.3f}")
        im = axes[1, j].imshow(err, cmap="inferno", vmin=0, vmax=err_vmax)
        axes[1, j].set_title(f"|err|  (mean {err.mean():.4f})", fontsize=9)
        axes[1, j].axis("off")
    fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01)
    fig.suptitle(f"image {idx:04d}", fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # improvement map: baseline_err - other_err (green = improved), only if >=2 runs
    if n >= 2 and crop is not None:
        y0, y1, x0, x1 = crop
        fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7.0))
        axes = np.atleast_2d(axes)
        base_err = errs[0]
        for j in range(n):
            show(axes[0, j], np.clip(preds[j][y0:y1, x0:x1], 0, 1), f"{labels[j]} (zoom)")
            if j == 0:
                im = axes[1, j].imshow(errs[0][y0:y1, x0:x1], cmap="inferno", vmin=0, vmax=err_vmax)
                axes[1, j].set_title("|err| baseline", fontsize=9)
            else:
                diff = (base_err - errs[j])[y0:y1, x0:x1]
                d = np.abs(diff).max() + 1e-8
                im = axes[1, j].imshow(diff, cmap="RdYlGn", vmin=-d, vmax=d)
                axes[1, j].set_title(f"{labels[0]} - {labels[j]}\n(green=better)", fontsize=9)
            axes[1, j].axis("off")
        fig.suptitle(f"image {idx:04d}  -- zoom on worst-error region", fontsize=11)
        fig.savefig(out_path.with_name(out_path.stem + "_zoom.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs as path[:label]")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, help="output directory for analysis artifacts")
    ap.add_argument("--max-w", type=int, default=1300, help="downscale width for analysis/viz")
    ap.add_argument("--no-panels", action="store_true", help="skip per-image panel rendering")
    args = ap.parse_args()

    runs = [parse_run(s) for s in args.runs]
    labels = [lab for _, lab in runs]
    out = Path(args.out)
    (out / "panels").mkdir(parents=True, exist_ok=True)

    metrics_by_run = {lab: load_metrics(d, args.split) for d, lab in runs}
    base_dir, _, idxs = img_paths(runs[0][0], args.split)

    # ---- per-image metric table + highlight stats -------------------------
    rows = []
    per_run_hl = {lab: [] for lab in labels}
    for idx in idxs:
        gt = load_img(base_dir / f"{idx:04d}.png", args.max_w)
        preds = [load_img(d / "images" / "pred" / args.split / f"{idx:04d}.png", args.max_w) for d, _ in runs]
        errs = [err_map(gt, p) for p in preds]
        row = {"idx": idx}
        for lab in labels:
            m = metrics_by_run[lab].get(idx, {})
            for k in ("PSNR", "SSIM", "LPIPS"):
                row[f"{lab}_{k}"] = m.get(k, float("nan"))
        # deltas vs baseline (run 0)
        for lab in labels[1:]:
            row[f"d{lab}_PSNR"] = row[f"{lab}_PSNR"] - row[f"{labels[0]}_PSNR"]
        for lab, err in zip(labels, errs):
            hl = highlight_stats(gt, err)
            per_run_hl[lab].append(hl)
            row[f"{lab}_hiErr"] = hl["bright_err"]
            row[f"{lab}_loErr"] = hl["rest_err"]
            row[f"{lab}_hi/lo"] = hl["bright_over_rest"]
        rows.append(row)

        if not args.no_panels:
            err_vmax = float(np.percentile(np.concatenate([e.ravel() for e in errs]), 99))
            crop = worst_bbox(errs[0])
            make_panel(idx, gt, preds, labels, metrics_by_run, out / "panels" / f"{idx:04d}.png", err_vmax, crop)

    # ---- write CSV --------------------------------------------------------
    csv_path = out / "per_image_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----------------------------------------------------------
    lines = []
    lines.append(f"runs: {', '.join(labels)}   split: {args.split}   images: {len(idxs)}\n")
    lines.append("=== mean metrics ===")
    for lab in labels:
        ms = metrics_by_run[lab]
        P = np.mean([ms[i]["PSNR"] for i in idxs])
        S = np.mean([ms[i]["SSIM"] for i in idxs])
        L = np.mean([ms[i]["LPIPS"] for i in idxs])
        hl = per_run_hl[lab]
        hi = np.mean([h["bright_err"] for h in hl])
        lo = np.mean([h["rest_err"] for h in hl])
        lines.append(f"  {lab:12s}  PSNR {P:6.3f}  SSIM {S:.4f}  LPIPS {L:.4f}   "
                     f"brightErr {hi:.4f}  restErr {lo:.4f}  ratio {hi/lo:.2f}")
    lines.append("")
    if len(labels) >= 2:
        b, o = labels[0], labels[1]
        dlt = sorted(rows, key=lambda r: r[f"d{o}_PSNR"])
        lines.append(f"=== {o} vs {b}: biggest PSNR regressions (top) / gains (bottom) ===")
        for r in dlt[:5] + ["..."] + dlt[-5:]:
            if r == "...":
                lines.append("  ...")
                continue
            lines.append(f"  img {r['idx']:04d}  {b} {r[f'{b}_PSNR']:6.3f} -> {o} {r[f'{o}_PSNR']:6.3f}"
                         f"  (d {r[f'd{o}_PSNR']:+.3f})   brightErr {b} {r[f'{b}_hiErr']:.4f} {o} {r[f'{o}_hiErr']:.4f}")
        lines.append("")
    lines.append("=== worst images by baseline PSNR ===")
    for r in sorted(rows, key=lambda r: r[f"{labels[0]}_PSNR"])[:8]:
        lines.append(f"  img {r['idx']:04d}  PSNR {r[f'{labels[0]}_PSNR']:6.3f}  "
                     f"hi/lo errRatio {r[f'{labels[0]}_hi/lo']:.2f}")
    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"\nwrote {csv_path}")
    print(f"wrote {out / 'summary.txt'}")
    if not args.no_panels:
        print(f"wrote panels to {out / 'panels'}/")


if __name__ == "__main__":
    main()
