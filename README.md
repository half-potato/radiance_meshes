This is the training code for 3DDR (3D Delaunay Rasterization). The viewing code is [here](https://github.com/Shmaug/DelTetRenderer) and the web viewer source code [here](https://github.com/half-potato/dsplat_web).

# Install
Install UV [here](https://github.com/astral-sh/uv).
```
uv pip install setuptools
uv pip install torch
```
Dependencies:
```
sudo apt install gnuplot
```
Then, it is as simple as running the command `uv run` instead of python to run the code. It installs everything super quickly.
Example running command for bonsai:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run train.py --eval  --dataset_path /data/nerf_datasets/360/bonsai --image_folder images_2 --output_path output/bonsai
```
