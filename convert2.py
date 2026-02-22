# modified from INRIA code
import os
import logging
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = ArgumentParser("Colmap/GLOMAP converter")
parser.add_argument("--no_gpu", action="store_true")
parser.add_argument("--skip_matching", action="store_true")
parser.add_argument("--source_path", "-s", required=True, type=str,
                    help="Root working directory. Images should be placed in <source_path>/input "
                         "(or will be extracted there from video).")
parser.add_argument("--camera", default="OPENCV", type=str,
                    help="COLMAP camera model. Overridden to OPENCV_FISHEYE when --fisheye is set.")
parser.add_argument("--fisheye", action="store_true",
                    help="Use OPENCV_FISHEYE camera model and skip image undistortion "
                         "(fisheye undistortion is handled separately or skipped for NeRF).")
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--glomap_executable", default="", type=str)
parser.add_argument("--use_glomap", action="store_true",
                    help="Use GLOMAP global mapper instead of COLMAP incremental mapper.")
parser.add_argument("--resize", action="store_true",
                    help="Also produce images_2, images_4, images_8 downscaled copies.")
parser.add_argument("--magick_executable", default="", type=str)

# --- Video / Insta360 options -----------------------------------------------
parser.add_argument("--video", default="", type=str,
                    help="Path to a single video file. Frames will be extracted to "
                         "<source_path>/input before running reconstruction.")
parser.add_argument("--insta360", action="store_true",
                    help="Treat --video as an Insta360 dual-lens file. Splits into two "
                         "single-lens videos, extracts frames from each lens into separate "
                         "sub-folders, then merges them into <source_path>/input.")
parser.add_argument("--fps", default=2.0, type=float,
                    help="Frames per second to sample from video (default: 2). "
                         "Lower = fewer frames = faster but sparser reconstruction.")
parser.add_argument("--max_video_size", default=1920, type=int,
                    help="Scale video frames down to this width (px) on extraction. "
                         "Useful for very high-res Insta360 footage. 0 = no scaling.")

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Resolve executables
# ---------------------------------------------------------------------------
colmap_command = f'"{args.colmap_executable}"' if args.colmap_executable else "colmap"
glomap_command = f'"{args.glomap_executable}"' if args.glomap_executable else "glomap"
magick_command = f'"{args.magick_executable}"' if args.magick_executable else "magick"
use_gpu = 0 if args.no_gpu else 1

# Fisheye overrides camera model and affects undistortion step
if args.fisheye:
    camera_model = "OPENCV_FISHEYE"
else:
    camera_model = args.camera

source = Path(args.source_path)
input_dir = source / "input"
distorted_dir = source / "distorted"
sparse_dir = distorted_dir / "sparse"
db_path = distorted_dir / "database.db"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run(cmd: str, step: str) -> None:
    """Run a shell command and exit with an error if it fails."""
    logging.info(f"[{step}] {cmd}")
    exit_code = os.system(cmd)
    if exit_code != 0:
        logging.error(f"{step} failed with exit code {exit_code}.\nCommand: {cmd}")
        sys.exit(exit_code)


def check_tool(name: str) -> None:
    if shutil.which(name) is None:
        logging.error(f"Required tool '{name}' not found on PATH. Please install it.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 0 — Video ingestion
# ---------------------------------------------------------------------------
if args.video:
    check_tool("ffmpeg")
    video_path = Path(args.video)
    if not video_path.exists():
        logging.error(f"Video file not found: {video_path}")
        sys.exit(1)

    input_dir.mkdir(parents=True, exist_ok=True)

    # Build a scale filter string (empty string = no scaling)
    if args.max_video_size > 0:
        # Scale width to max_video_size, keep aspect ratio, ensure even dimensions
        scale_filter = f"scale='min({args.max_video_size},iw):-2'"
    else:
        scale_filter = ""

    def build_ffmpeg_extract(video_src: Path, out_dir: Path, prefix: str) -> str:
        """Return an ffmpeg command that extracts frames at --fps from video_src."""
        out_dir.mkdir(parents=True, exist_ok=True)
        vf_parts = [f"fps={args.fps}"]
        if scale_filter:
            vf_parts.append(scale_filter)
        vf = ",".join(vf_parts)
        output_pattern = out_dir / f"{prefix}_%06d.jpg"
        return (
            f'ffmpeg -hide_banner -loglevel error '
            f'-i "{video_src}" '
            f'-vf "{vf}" '
            f'-q:v 2 '          # JPEG quality (2 = near-lossless, fine for SIFT)
            f'"{output_pattern}"'
        )

    if args.insta360:
        print("=== Insta360 mode: splitting dual-lens video ===")
        base = video_path.stem
        tmp_dir = source / "tmp_video"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        lens1_path = tmp_dir / f"{base}_lens1.mp4"
        lens2_path = tmp_dir / f"{base}_lens2.mp4"

        # Split streams
        print("Splitting lens 1...")
        run(
            f'ffmpeg -hide_banner -loglevel error '
            f'-i "{video_path}" '
            f'-map 0:v:0 -map 0:a:0? -c:v copy -c:a copy "{lens1_path}"',
            "Insta360 split lens 1"
        )
        print("Splitting lens 2...")
        run(
            f'ffmpeg -hide_banner -loglevel error '
            f'-i "{video_path}" '
            f'-map 0:v:1 -map 0:a:0? -c:v copy -c:a copy "{lens2_path}"',
            "Insta360 split lens 2"
        )

        # Extract frames from each lens into the shared input folder.
        # Prefixing by lens ensures no filename collisions.
        print(f"Extracting frames from lens 1 at {args.fps} fps...")
        run(build_ffmpeg_extract(lens1_path, input_dir, "lens1"), "Frame extraction lens 1")

        print(f"Extracting frames from lens 2 at {args.fps} fps...")
        run(build_ffmpeg_extract(lens2_path, input_dir, "lens2"), "Frame extraction lens 2")

        # Clean up temporary split videos
        shutil.rmtree(tmp_dir, ignore_errors=True)

        total_frames = len(list(input_dir.glob("*.jpg")))
        print(f"Total frames extracted: {total_frames}")

    else:
        # Single video — just extract frames
        print(f"=== Extracting frames from video at {args.fps} fps ===")
        run(build_ffmpeg_extract(video_path, input_dir, "frame"), "Frame extraction")
        total_frames = len(list(input_dir.glob("*.jpg")))
        print(f"Total frames extracted: {total_frames}")

# ---------------------------------------------------------------------------
# Step 1 — Feature extraction & matching
# ---------------------------------------------------------------------------
if not args.skip_matching:
    sparse_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Feature extraction (camera model: {camera_model}) ===")
    feat_extraction_cmd = (
        f"{colmap_command} feature_extractor "
        f"--database_path \"{db_path}\" "
        f"--image_path \"{input_dir}\" "
        f"--ImageReader.single_camera 1 "
        f"--ImageReader.camera_model {camera_model} "
        f"--SiftExtraction.use_gpu {use_gpu} "
        # Affine shape + domain size pooling help with strong distortion
        + ("--SiftExtraction.estimate_affine_shape 1 "
           "--SiftExtraction.domain_size_pooling 1 "
           if args.fisheye else "")
    )
    run(feat_extraction_cmd, "Feature extraction")

    print("=== Feature matching ===")
    feat_matching_cmd = (
        f"{colmap_command} exhaustive_matcher "
        f"--database_path \"{db_path}\" "
        f"--SiftMatching.use_gpu {use_gpu}"
    )
    run(feat_matching_cmd, "Feature matching")

# ---------------------------------------------------------------------------
# Step 2 — Mapping (GLOMAP or COLMAP)
# ---------------------------------------------------------------------------
if not args.skip_matching:
    if args.use_glomap:
        print("=== Mapping with GLOMAP (global SfM) ===")
        mapper_cmd = (
            f"{glomap_command} mapper "
            f"--database_path \"{db_path}\" "
            f"--image_path \"{input_dir}\" "
            f"--output_path \"{sparse_dir}\""
        )
        if use_gpu:
            mapper_cmd += " --GlobalPositioning.use_gpu 1 --BundleAdjustment.use_gpu 1"
    else:
        print("=== Mapping with COLMAP (incremental SfM) ===")
        mapper_cmd = (
            f"{colmap_command} mapper "
            f"--database_path \"{db_path}\" "
            f"--image_path \"{input_dir}\" "
            f"--output_path \"{sparse_dir}\" "
            f"--Mapper.ba_global_function_tolerance=0.000001"
        )
    run(mapper_cmd, "Mapping")

# ---------------------------------------------------------------------------
# Step 3 — Image undistortion
#
# Fisheye note: colmap image_undistorter does not support fisheye models well.
# For fisheye/OPENCV_FISHEYE we skip undistortion and copy the sparse model
# directly. Downstream tools (e.g. nerfstudio) can read distorted models.
# ---------------------------------------------------------------------------
if args.fisheye:
    print("=== Fisheye mode: skipping COLMAP undistortion ===")
    print("    Copying sparse model directly to <source>/sparse/0 ...")
    final_sparse = source / "sparse" / "0"
    final_sparse.mkdir(parents=True, exist_ok=True)

    # Copy the first (best) model from distorted/sparse to sparse/0
    best_model = sparse_dir / "0"
    if best_model.exists():
        for f in best_model.iterdir():
            shutil.copy2(f, final_sparse / f.name)
    else:
        # GLOMAP sometimes writes directly into sparse/ without a sub-folder
        for f in sparse_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, final_sparse / f.name)

    # Also copy input images to images/ so downstream tools find them
    images_out = source / "images"
    if not images_out.exists():
        print(f"    Symlinking {input_dir} -> {images_out}")
        images_out.symlink_to(input_dir.resolve())

else:
    print("=== Image undistortion ===")
    img_undist_cmd = (
        f"{colmap_command} image_undistorter "
        f"--image_path \"{input_dir}\" "
        f"--input_path \"{sparse_dir / '0'}\" "
        f"--output_path \"{source}\" "
        f"--output_type COLMAP"
    )
    run(img_undist_cmd, "Image undistortion")

    # Move sparse files into sparse/0 (expected layout for 3DGS)
    out_sparse = source / "sparse"
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
