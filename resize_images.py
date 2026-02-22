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

import os
import logging
import shutil
import subprocess
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--num_workers", "-j", default=8, type=int,
                    help="Number of parallel resize workers (default: 8)")
args = parser.parse_args()
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"


def resize_one(source_file, file, source_path):
    """Copy and resize a single image to all three scale levels."""
    for pct, suffix in [("50%", "2"), ("25%", "4"), ("12.5%", "8")]:
        dest = os.path.join(source_path, f"images_{suffix}", file)
        shutil.copy2(source_file, dest)
        ret = subprocess.call(f'{magick_command} mogrify -resize {pct} "{dest}"', shell=True)
        if ret != 0:
            return file, pct, ret
    return file, None, 0


print("Copying and resizing...")

os.makedirs(args.source_path + "/images_2", exist_ok=True)
os.makedirs(args.source_path + "/images_4", exist_ok=True)
os.makedirs(args.source_path + "/images_8", exist_ok=True)

files = os.listdir(args.source_path + "/images")

with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
    futures = {
        pool.submit(resize_one, os.path.join(args.source_path, "images", f), f, args.source_path): f
        for f in files
    }
    done = 0
    for fut in as_completed(futures):
        fname, failed_pct, code = fut.result()
        if code != 0:
            logging.error(f"{failed_pct} resize failed for {fname} with code {code}. Exiting.")
            exit(code)
        done += 1
        if done % 20 == 0 or done == len(files):
            print(f"  [{done}/{len(files)}]")

print("Done.")

