#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Modified to support:
#   - GLOMAP global mapper (--use_glomap)
#   - Fisheye camera model (--fisheye)
#   - Insta360 dual-lens video ingestion (--video / --insta360)
#   - Frame sampling from video (--fps)
#   - hloc + LightGlue/SuperGlue feature pipeline (--matcher lightglue|superglue|sift)
#     with SuperPoint, DISK, or ALIKED extractors (--extractor)
#
# Install hloc:
#   pip install git+https://github.com/cvg/Hierarchical-Localization.git
# Install LightGlue (required for --matcher lightglue):
#   pip install git+https://github.com/cvg/LightGlue.git

import os
import logging
import shutil
import struct
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = ArgumentParser("Colmap/GLOMAP converter")
parser.add_argument("--no_gpu", action="store_true")
parser.add_argument("--skip_matching", action="store_true")
parser.add_argument("--source_path", "-s", required=True, type=str,
                    help="Root working directory. Images should be in <source_path>/input "
                         "(or will be extracted there from video).")
parser.add_argument("--camera", default="OPENCV", type=str,
                    help="COLMAP camera model. Overridden to OPENCV_FISHEYE when --fisheye is set.")
parser.add_argument("--fisheye", action="store_true",
                    help="Use OPENCV_FISHEYE camera model and skip image undistortion.")
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--glomap_executable", default="", type=str)
parser.add_argument("--use_glomap", action="store_true",
                    help="Use GLOMAP global mapper instead of COLMAP incremental mapper.")
parser.add_argument("--resize", action="store_true",
                    help="Also produce images_2, images_4, images_8 downscaled copies.")
parser.add_argument("--magick_executable", default="", type=str)

# --- Feature matching pipeline ----------------------------------------------
parser.add_argument("--matcher", default="sift",
                    choices=["sift", "lightglue", "superglue"],
                    help=(
                        "Feature matching pipeline:\n"
                        "  sift       — classic COLMAP SIFT (default, no extra deps)\n"
                        "  lightglue  — hloc + LightGlue (best quality, recommended for fisheye)\n"
                        "  superglue  — hloc + SuperGlue (older, slower than lightglue)\n"
                    ))
parser.add_argument("--extractor", default="superpoint",
                    choices=["superpoint", "disk", "aliked"],
                    help=(
                        "Local feature extractor for hloc matchers (ignored for --matcher sift):\n"
                        "  superpoint — fast, broadly good (default)\n"
                        "  disk       — more repeatable on fisheye / wide-angle\n"
                        "  aliked     — newer, competitive with disk\n"
                    ))
parser.add_argument("--num_keypoints", default=4096, type=int,
                    help="Max keypoints per image for hloc extractors (default: 4096).")
parser.add_argument("--num_matched", default=0, type=int,
                    help="hloc: pairs per image via NetVLAD retrieval. "
                         "0 (default) = exhaustive (best for <=~200 images). "
                         "Set to 20-50 for larger datasets.")

# --- Video / Insta360 options -----------------------------------------------
parser.add_argument("--video", nargs="+", default=[],
                    help="Path(s) to one or more video files. Frames are extracted to <source_path>/input.")
parser.add_argument("--insta360", action="store_true",
                    help="Treat --video as an Insta360 dual-lens file: splits streams, "
                         "extracts frames from each lens (prefixed lens1_/lens2_).")
parser.add_argument("--fps", default=2.0, type=float,
                    help="Frames per second to sample from video (default: 2).")
parser.add_argument("--max_video_size", default=1920, type=int,
                    help="Scale frames down to this width in pixels on extraction "
                         "(default: 1920). 0 = no scaling.")

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Resolve executables and paths
# ---------------------------------------------------------------------------
colmap_command = f'"{args.colmap_executable}"' if args.colmap_executable else "colmap"
glomap_command = f'"{args.glomap_executable}"' if args.glomap_executable else "glomap"
magick_command = f'"{args.magick_executable}"' if args.magick_executable else "magick"
use_gpu = 0 if args.no_gpu else 1

camera_model = "OPENCV_FISHEYE" if args.fisheye else args.camera
use_hloc     = args.matcher in ("lightglue", "superglue")

source        = Path(args.source_path)
input_dir     = source / "input"
distorted_dir = source / "distorted"
sparse_dir    = distorted_dir / "sparse"
db_path       = distorted_dir / "database.db"
hloc_dir      = source / "hloc_outputs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run(cmd: str, step: str) -> None:
    logging.info(f"[{step}] {cmd}")
    # Clear LD_LIBRARY_PATH so that hloc/torch's injected conda lib path
    # doesn't override system libraries needed by colmap/glomap.
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    exit_code = subprocess.call(cmd, shell=True, env=env)
    if exit_code != 0:
        logging.error(f"{step} failed with exit code {exit_code}.\nCommand: {cmd}")
        sys.exit(exit_code)


def check_tool(name: str) -> None:
    if shutil.which(name) is None:
        logging.error(f"Required tool '{name}' not found on PATH.")
        sys.exit(1)


def find_best_model(sparse_root: Path) -> Path:
    """Return the sub-model directory with the most registered images.

    COLMAP mapper can produce multiple models (0/, 1/, 2/, …).
    We pick the one whose images.bin (or images.txt) is largest,
    which corresponds to the most registered images.
    """
    candidates = sorted(
        [d for d in sparse_root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not candidates:
        # No numbered sub-dirs — model files are directly in sparse_root
        return sparse_root

    best = candidates[0]
    best_size = 0
    for d in candidates:
        for name in ("images.bin", "images.txt"):
            p = d / name
            if p.exists():
                sz = p.stat().st_size
                if sz > best_size:
                    best_size = sz
                    best = d
                break
    print(f"    Selected model {best.name} (out of {len(candidates)}) as largest")
    return best


def require_hloc() -> None:
    try:
        import hloc  # noqa: F401
    except ImportError:
        print(
            "\n[ERROR] hloc is not installed but is required for "
            f"--matcher {args.matcher}.\n\n"
            "Install with:\n"
            "  pip install git+https://github.com/cvg/Hierarchical-Localization.git\n"
            + ("  pip install git+https://github.com/cvg/LightGlue.git\n"
               if args.matcher == "lightglue" else "")
        )
        sys.exit(1)


def rectify_fisheye_images(input_dir: Path, output_dir: Path,
                           fx: float, fy: float, cx: float, cy: float,
                           k1: float, k2: float, k3: float, k4: float,
                           width: int, height: int) -> None:
    """Rectify OPENCV_FISHEYE images from polynomial distortion to pure equidistant.

    Builds a remap grid using the *forward* Kannala-Brandt distortion model:
        theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9

    For each output pixel (pure equidistant), we compute the corresponding
    source pixel (distorted) and sample via cv2.remap().  The output retains
    the full fisheye FOV — only the nonlinear polynomial correction is baked out.
    """
    import cv2
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build remap grid (same for all images — single_camera mode)
    v_coords, u_coords = np.mgrid[0:height, 0:width].astype(np.float64)

    # Normalize to undistorted angular coords
    px = (u_coords - cx) / fx
    py = (v_coords - cy) / fy
    theta = np.sqrt(px * px + py * py)  # incidence angle (equidistant)

    # Forward distortion: theta -> theta_d
    theta2 = theta * theta
    theta_d = theta * (1.0
                       + k1 * theta2
                       + k2 * theta2 * theta2
                       + k3 * theta2 * theta2 * theta2
                       + k4 * theta2 * theta2 * theta2 * theta2)

    # Scale factor: theta_d / theta  (handle theta==0 to avoid division by zero)
    scale = np.where(theta > 1e-9, theta_d / theta, 1.0)

    # Source pixel coordinates in the original (distorted) image
    map_x = (px * scale * fx + cx).astype(np.float32)
    map_y = (py * scale * fy + cy).astype(np.float32)

    # Process each image
    extensions = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in extensions)

    for img_path in images:
        img_pil = Image.open(img_path)
        has_alpha = img_pil.mode == "RGBA"
        img_np = np.array(img_pil)

        if has_alpha:
            # Remap RGB and alpha separately
            rgb = img_np[:, :, :3]
            alpha = img_np[:, :, 3]
            rgb_rect = cv2.remap(rgb, map_x, map_y,
                                 interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
            alpha_rect = cv2.remap(alpha, map_x, map_y,
                                   interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
            out_np = np.dstack([rgb_rect, alpha_rect])
        else:
            img_np = np.array(img_pil.convert("RGB"))
            out_np = cv2.remap(img_np, map_x, map_y,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))

        out_pil = Image.fromarray(out_np)
        out_pil.save(output_dir / img_path.name)

    print(f"    Rectified {len(images)} images -> {output_dir}")


def write_cameras_binary(path: Path, cameras: dict) -> None:
    """Write a COLMAP cameras.bin file.

    *cameras* is a dict of camera_id -> dict with keys:
        camera_id, model_id, width, height, params (list/array of float64)
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras.values():
            f.write(struct.pack("<iiQQ",
                                cam["camera_id"],
                                cam["model_id"],
                                cam["width"],
                                cam["height"]))
            for p in cam["params"]:
                f.write(struct.pack("<d", p))


def read_cameras_binary_simple(path: Path) -> dict:
    """Read COLMAP cameras.bin and return a dict of camera info dicts."""
    from data.colmap_loader import CAMERA_MODEL_IDS

    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = list(struct.unpack("<" + "d" * num_params,
                                        f.read(8 * num_params)))
            cameras[cam_id] = {
                "camera_id": cam_id,
                "model_id": model_id,
                "width": w,
                "height": h,
                "params": params,
            }
    return cameras


# ---------------------------------------------------------------------------
# Step 0 — Video ingestion
# ---------------------------------------------------------------------------
if args.video:
    check_tool("ffmpeg")
    video_paths = [Path(v) for v in args.video]
    for vp in video_paths:
        if not vp.exists():
            logging.error(f"Video file not found: {vp}")
            sys.exit(1)

    multi = len(video_paths) > 1

    input_dir.mkdir(parents=True, exist_ok=True)
    scale_filter = (f"scale='min({args.max_video_size},iw):-2'"
                    if args.max_video_size > 0 else "")

    def build_ffmpeg_extract(video_src: Path, out_dir: Path, prefix: str) -> str:
        out_dir.mkdir(parents=True, exist_ok=True)
        vf_parts = [f"fps={args.fps}"]
        if scale_filter:
            vf_parts.append(scale_filter)
        output_pattern = out_dir / f"{prefix}_%06d.jpg"
        return (
            f'ffmpeg -hide_banner -loglevel error '
            f'-i "{video_src}" '
            f'-vf "{",".join(vf_parts)}" '
            f'-q:v 2 "{output_pattern}"'
        )

    if args.insta360:
        print(f"=== Insta360 mode: splitting {len(video_paths)} dual-lens video(s) ===")
        tmp_dir = source / "tmp_video"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for i, video_path in enumerate(video_paths):
            if multi:
                prefix1, prefix2 = f"v{i}_lens1", f"v{i}_lens2"
                print(f"--- Video {i}: {video_path.name} (prefixes: {prefix1}, {prefix2}) ---")
            else:
                prefix1, prefix2 = "lens1", "lens2"

            lens1_path = tmp_dir / f"{video_path.stem}_lens1.mp4"
            lens2_path = tmp_dir / f"{video_path.stem}_lens2.mp4"

            print("Splitting lens 1...")
            run(f'ffmpeg -hide_banner -loglevel error -i "{video_path}" '
                f'-map 0:v:0 -map 0:a:0? -c:v copy -c:a copy "{lens1_path}"',
                f"Insta360 split lens 1 (video {i})")
            print("Splitting lens 2...")
            run(f'ffmpeg -hide_banner -loglevel error -i "{video_path}" '
                f'-map 0:v:1 -map 0:a:0? -c:v copy -c:a copy "{lens2_path}"',
                f"Insta360 split lens 2 (video {i})")

            print(f"Extracting frames from lens 1 at {args.fps} fps...")
            run(build_ffmpeg_extract(lens1_path, input_dir, prefix1),
                f"Frame extraction lens 1 (video {i})")
            print(f"Extracting frames from lens 2 at {args.fps} fps...")
            run(build_ffmpeg_extract(lens2_path, input_dir, prefix2),
                f"Frame extraction lens 2 (video {i})")

        shutil.rmtree(tmp_dir, ignore_errors=True)

        # --- Apply fisheye masks to extracted frames --------------------------
        print("=== Applying lens masks to Insta360 frames ===")
        from PIL import Image

        script_dir = Path(__file__).resolve().parent
        mask1_path = script_dir / "assets" / "lens1_mask.jpg"
        mask2_path = script_dir / "assets" / "lens2_mask.jpg"

        if not mask1_path.exists() or not mask2_path.exists():
            logging.error(f"Lens mask files not found in {script_dir / 'assets'}")
            sys.exit(1)

        mask1 = Image.open(mask1_path).convert("L")
        mask2 = Image.open(mask2_path).convert("L")

        for img_path in sorted(input_dir.glob("*lens*_*.jpg")):
            img = Image.open(img_path).convert("RGB")
            if "lens1" in img_path.name:
                mask = mask1
            else:
                mask = mask2
            # Resize mask to match frame dimensions
            mask_resized = mask.resize(img.size, Image.LANCZOS)
            # Create RGBA image with mask as alpha
            img_rgba = img.copy()
            img_rgba.putalpha(mask_resized)
            # Save as PNG and remove original JPG
            png_path = img_path.with_suffix(".png")
            img_rgba.save(png_path)
            img_path.unlink()

        print(f"    Masked {len(list(input_dir.glob('*lens*_*.png')))} frames "
              f"(converted to PNG with alpha)")
    else:
        print(f"=== Extracting frames from {len(video_paths)} video(s) at {args.fps} fps ===")
        for i, video_path in enumerate(video_paths):
            prefix = f"v{i}_frame" if multi else "frame"
            if multi:
                print(f"--- Video {i}: {video_path.name} (prefix: {prefix}) ---")
            run(build_ffmpeg_extract(video_path, input_dir, prefix),
                f"Frame extraction (video {i})")

    num_frames = len(list(input_dir.glob('*.jpg'))) + len(list(input_dir.glob('*.png')))
    print(f"Total frames extracted: {num_frames}")


# ---------------------------------------------------------------------------
# Step 1 — Feature extraction & matching
# ---------------------------------------------------------------------------
if not args.skip_matching:
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Branch A: hloc pipeline (LightGlue or SuperGlue)
    # =========================================================================
    if use_hloc:
        require_hloc()

        from hloc import extract_features, match_features
        from hloc import pairs_from_exhaustive, pairs_from_retrieval

        hloc_dir.mkdir(parents=True, exist_ok=True)

        # --- Extractor configs -----------------------------------------------
        extractor_confs = {
            "superpoint": {
                "output": "feats-superpoint",
                "model": {
                    "name": "superpoint",
                    "nms_radius": 3,
                    "max_num_keypoints": args.num_keypoints,
                },
                "preprocessing": {"grayscale": True, "resize_max": 1024},
            },
            "disk": {
                "output": "feats-disk",
                "model": {
                    "name": "disk",
                    "max_num_keypoints": args.num_keypoints,
                },
                "preprocessing": {"grayscale": False, "resize_max": 1024},
            },
            "aliked": {
                "output": "feats-aliked",
                "model": {
                    "name": "aliked",
                    "model_name": "aliked-n16rot",
                    "max_num_keypoints": args.num_keypoints,
                    "detection_threshold": 0.01,
                },
                "preprocessing": {"grayscale": False, "resize_max": 1024},
            },
        }

        # --- Matcher configs -------------------------------------------------
        if args.matcher == "lightglue":
            matcher_conf = {
                "output": f"matches-lightglue-{args.extractor}",
                "model": {
                    "name": "lightglue",
                    "features": args.extractor,
                    # Conservative thresholds — keeps more matches on fisheye
                    "depth_confidence": 0.95,
                    "width_confidence": 0.99,
                    "filter_threshold": 0.1,
                },
            }
        else:  # superglue
            matcher_conf = {
                "output": f"matches-superglue-{args.extractor}",
                "model": {
                    "name": "superglue",
                    "features": args.extractor,
                    "weights": "outdoor",
                    "sinkhorn_iterations": 50,
                },
            }

        # --- Feature extraction ----------------------------------------------
        print(f"=== hloc: extracting {args.extractor} features "
              f"(max {args.num_keypoints} kp/image) ===")
        features_path = extract_features.main(
            extractor_confs[args.extractor],
            input_dir,
            hloc_dir,
        )

        # --- Image pair generation -------------------------------------------
        pairs_path = hloc_dir / "pairs.txt"
        num_images = len(list(input_dir.glob("*.[jJpPjJ][pPnNpP][gGeEgG]")))

        if args.num_matched > 0:
            print(f"=== hloc: NetVLAD retrieval "
                  f"(top-{args.num_matched} pairs per image) ===")
            retrieval_path = extract_features.main(
                extract_features.confs["netvlad"],
                input_dir,
                hloc_dir,
            )
            pairs_from_retrieval.main(
                retrieval_path,
                pairs_path,
                num_matched=args.num_matched,
            )
        else:
            print(f"=== hloc: exhaustive pairing ({num_images} images, "
                  f"{num_images*(num_images-1)//2} pairs) ===")
            pairs_from_exhaustive.main(pairs_path, features=features_path)

        # --- Matching --------------------------------------------------------
        print(f"=== hloc: matching with {args.matcher} + {args.extractor} ===")
        matches_path = match_features.main(
            matcher_conf,
            pairs_path,
            features=extractor_confs[args.extractor]["output"],
            export_dir=hloc_dir,
        )

        # --- Write COLMAP DB + run hloc SfM ----------------------------------
        # Build the COLMAP database from hloc features/matches.
        # We use COLMAP CLI + sqlite3 + h5py directly, because pycolmap's
        # Database class is broken by hloc's LD_LIBRARY_PATH injection.
        import sqlite3
        import h5py
        import numpy as np
        from hloc.utils.io import get_keypoints, get_matches

        print(f"=== hloc: building COLMAP database "
              f"(camera model: {camera_model}) ===")

        hloc_sfm_dir = hloc_dir / "sfm"
        hloc_sfm_dir.mkdir(parents=True, exist_ok=True)
        hloc_db_path = hloc_sfm_dir / "database.db"

        # 1. Create database + register images via COLMAP CLI
        if hloc_db_path.exists():
            hloc_db_path.unlink()
        run(f"{colmap_command} feature_extractor "
            f"--database_path \"{hloc_db_path}\" "
            f"--image_path \"{input_dir}\" "
            f"--ImageReader.single_camera 1 "
            f"--ImageReader.camera_model {camera_model} "
            f"--FeatureExtraction.use_gpu {use_gpu}",
            "Database creation + image registration")

        # 2. Replace SIFT keypoints with hloc keypoints via sqlite3
        print("    Importing hloc keypoints into database...")
        conn = sqlite3.connect(str(hloc_db_path))
        cur = conn.cursor()

        image_ids = {}
        for row in cur.execute("SELECT image_id, name FROM images"):
            image_ids[row[1]] = row[0]

        for image_name, image_id in image_ids.items():
            keypoints = get_keypoints(features_path, image_name)
            keypoints += 0.5  # COLMAP origin convention
            kp_blob = keypoints.astype(np.float32).tobytes()
            cur.execute(
                "UPDATE keypoints SET rows=?, cols=?, data=? WHERE image_id=?",
                (keypoints.shape[0], keypoints.shape[1], kp_blob, image_id),
            )
        # Clear SIFT descriptors (not needed for hloc matches)
        cur.execute("UPDATE descriptors SET rows=0, cols=0, data=X''")

        conn.commit()
        conn.close()

        # 3. Write hloc matches to text file in COLMAP raw format, then
        #    import + geometrically verify via matches_importer --match_type raw
        raw_matches_path = hloc_dir / "raw_matches.txt"
        print("    Writing hloc matches to raw format...")
        pairs_list = []
        with open(str(pairs_path), "r") as f:
            pairs_list = [line.strip().split() for line in f if line.strip()]

        num_pairs = 0
        with open(str(raw_matches_path), "w") as out:
            matched = set()
            for name0, name1 in pairs_list:
                if (name0, name1) in matched or (name1, name0) in matched:
                    continue
                matches, scores = get_matches(matches_path, name0, name1)
                if matches.shape[0] == 0:
                    continue
                out.write(f"{name0} {name1}\n")
                for m in matches:
                    out.write(f"{m[0]} {m[1]}\n")
                out.write("\n")
                matched.add((name0, name1))
                num_pairs += 1
        print(f"    Wrote {num_pairs} match pairs")

        print("    Running match import + geometric verification...")
        run(f"{colmap_command} matches_importer "
            f"--database_path \"{hloc_db_path}\" "
            f"--match_list_path \"{raw_matches_path}\" "
            f"--match_type raw "
            f"--FeatureMatching.use_gpu {use_gpu}",
            "Geometric verification")

        if args.use_glomap:
            # Copy the database so GLOMAP can find it
            print("    Note: hloc SfM skipped — GLOMAP will handle mapping.")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(hloc_db_path, db_path)
            print(f"    Copied hloc DB -> {db_path}")

    # =========================================================================
    # Branch B: classic COLMAP SIFT
    # =========================================================================
    else:
        print(f"=== SIFT feature extraction (camera model: {camera_model}) ===")
        run(
            f"{colmap_command} feature_extractor "
            f"--database_path \"{db_path}\" "
            f"--image_path \"{input_dir}\" "
            f"--ImageReader.single_camera 1 "
            f"--ImageReader.camera_model {camera_model} "
            f"--FeatureExtraction.use_gpu {use_gpu}"
            + (" --SiftExtraction.estimate_affine_shape 1"
               " --SiftExtraction.domain_size_pooling 1"
               if args.fisheye else ""),
            "SIFT feature extraction"
        )

        print("=== SIFT exhaustive matching ===")
        run(
            f"{colmap_command} exhaustive_matcher "
            f"--database_path \"{db_path}\" "
            f"--FeatureMatching.use_gpu {use_gpu}",
            "SIFT feature matching"
        )


# ---------------------------------------------------------------------------
# Step 2 — Mapping (GLOMAP or COLMAP incremental)
# Runs even with --skip_matching so long as sparse_dir is missing but the
# database exists (e.g. user cleaned the sparse output but kept the DB).
# ---------------------------------------------------------------------------
sparse_model_exists = sparse_dir.exists() and any(sparse_dir.iterdir())

if args.use_glomap and not sparse_model_exists:
    print("=== Mapping with GLOMAP (global SfM) ===")
    sparse_dir.mkdir(parents=True, exist_ok=True)
    run(
        f"{colmap_command} global_mapper "
        f"--database_path \"{db_path}\" "
        f"--image_path \"{input_dir}\" "
        f"--output_path \"{sparse_dir}\"",
        "GLOMAP mapping"
    )

elif use_hloc and not sparse_model_exists:
    # Run COLMAP incremental SfM via CLI subprocess.
    # We avoid pycolmap.incremental_mapping() because hloc/torch
    # pollute LD_LIBRARY_PATH with an incompatible SQLite.
    # The run() helper clears LD_LIBRARY_PATH for subprocesses.
    hloc_db = hloc_dir / "sfm" / "database.db"
    if not hloc_db.exists():
        logging.error(f"hloc database not found at {hloc_db}. "
                      "Run without --skip_matching first.")
        sys.exit(1)
    print("=== Mapping with COLMAP (hloc features) ===")
    sparse_dir.mkdir(parents=True, exist_ok=True)
    run(
        f"{colmap_command} mapper "
        f"--database_path \"{hloc_db}\" "
        f"--image_path \"{input_dir}\" "
        f"--output_path \"{sparse_dir}\" "
        f"--Mapper.ba_global_function_tolerance=0.000001",
        "COLMAP incremental SfM (hloc)"
    )

elif not use_hloc and not args.use_glomap and not sparse_model_exists:
    print("=== Mapping with COLMAP (incremental SfM) ===")
    sparse_dir.mkdir(parents=True, exist_ok=True)
    run(
        f"{colmap_command} mapper "
        f"--database_path \"{db_path}\" "
        f"--image_path \"{input_dir}\" "
        f"--output_path \"{sparse_dir}\" "
        f"--Mapper.ba_global_function_tolerance=0.000001",
        "COLMAP mapping"
    )

elif sparse_model_exists:
    print(f"=== Sparse model already exists at {sparse_dir}, skipping mapper ===")


# ---------------------------------------------------------------------------
# Step 3 — Image undistortion  (skipped for fisheye)
# ---------------------------------------------------------------------------
if args.fisheye:
    print("=== Fisheye mode: rectifying to pure equidistant projection ===")
    print("    Copying sparse model directly to <source>/sparse/0 ...")
    final_sparse = source / "sparse" / "0"
    final_sparse.mkdir(parents=True, exist_ok=True)

    src_files = find_best_model(sparse_dir)
    for f in src_files.iterdir():
        if f.is_file():
            shutil.copy2(f, final_sparse / f.name)

    # Read camera intrinsics from the copied sparse model
    cameras_bin_path = final_sparse / "cameras.bin"
    cameras = read_cameras_binary_simple(cameras_bin_path)
    cam = next(iter(cameras.values()))  # single_camera mode
    params = cam["params"]
    # OPENCV_FISHEYE params: [fx, fy, cx, cy, k1, k2, k3, k4]
    cam_fx, cam_fy, cam_cx, cam_cy = params[0], params[1], params[2], params[3]
    cam_k1, cam_k2, cam_k3, cam_k4 = params[4], params[5], params[6], params[7]

    has_distortion = any(abs(k) > 1e-12 for k in [cam_k1, cam_k2, cam_k3, cam_k4])

    images_out = source / "images"
    if has_distortion:
        print(f"    Distortion coefficients: k1={cam_k1:.6f} k2={cam_k2:.6f} "
              f"k3={cam_k3:.6f} k4={cam_k4:.6f}")
        # Remove existing images dir if it's a symlink from a previous run
        if images_out.is_symlink():
            images_out.unlink()
        rectify_fisheye_images(
            input_dir, images_out,
            fx=cam_fx, fy=cam_fy, cx=cam_cx, cy=cam_cy,
            k1=cam_k1, k2=cam_k2, k3=cam_k3, k4=cam_k4,
            width=cam["width"], height=cam["height"],
        )
        # Write updated cameras.bin with zeroed distortion
        cam["params"] = [cam_fx, cam_fy, cam_cx, cam_cy, 0.0, 0.0, 0.0, 0.0]
        write_cameras_binary(cameras_bin_path, cameras)
        print("    Updated cameras.bin with zeroed distortion coefficients")
    else:
        print("    Distortion coefficients already zero, skipping rectification")
        if not images_out.exists():
            print(f"    Symlinking {input_dir} -> {images_out}")
            images_out.symlink_to(input_dir.resolve())

else:
    print("=== Image undistortion ===")
    best_model = find_best_model(sparse_dir)
    run(
        f"{colmap_command} image_undistorter "
        f"--image_path \"{input_dir}\" "
        f"--input_path \"{best_model}\" "
        f"--output_path \"{source}\" "
        f"--output_type COLMAP",
        "Image undistortion"
    )

    out_sparse   = source / "sparse"
    out_sparse_0 = out_sparse / "0"
    out_sparse_0.mkdir(parents=True, exist_ok=True)
    for f in out_sparse.iterdir():
        if f.name == "0":
            continue
        shutil.move(str(f), str(out_sparse_0 / f.name))


# ---------------------------------------------------------------------------
# Step 4 — Optional image resizing
# ---------------------------------------------------------------------------
if args.resize:
    print("=== Resizing images ===")
    check_tool("magick")
    images_dir = source / "images"
    for scale, suffix in [(50, "2"), (25, "4"), (12.5, "8")]:
        scaled_dir = source / f"images_{suffix}"
        scaled_dir.mkdir(parents=True, exist_ok=True)
        for img in images_dir.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            dest = scaled_dir / img.name
            shutil.copy2(img, dest)
            exit_code = os.system(f"{magick_command} mogrify -resize {scale}% \"{dest}\"")
            if exit_code != 0:
                logging.error(f"Resize to {scale}% failed for {img.name}")
                sys.exit(exit_code)
    print("Resizing complete.")

print("\nDone.")
