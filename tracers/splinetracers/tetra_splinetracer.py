# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from pathlib import Path
from typing import *

import slangtorch
import torch
from torch.autograd import Function
from icecream import ic

import sys
sys.path.append(str(Path(__file__).parent))
# from build.splinetracer.extension import d6_splinetracer_cpp_extension as sp
# kernels = slangpy.loadModule(
#     str(Path(__file__).parent / "d6_splinetracer/slang/backwards_kernel.slang")
# )

from ..build.splinetracer.extension import tetra_splinetracer_cpp_extension as sp
kernels = slangtorch.loadModule(
    str(Path(__file__).parent / "tetra_splinetracer/slang/backwards_kernel.slang"),
    includePaths=[str(Path(__file__).parent / 'slang')]
)


otx = sp.OptixContext(torch.device("cuda:0"))

MAX_ITERS = 500

def get_tet_adjacency(tets: torch.Tensor):
    """
    Takes a tensor of N tetrahedra indices and finds all M unique
    faces, returning the oriented faces and a map of their neighboring tets.

    Args:
        tets: A (N, 4) long tensor of tetrahedra indices.

    Returns:
        A tuple of (faces, side_index):
        - faces: (M, 3) long tensor of unique, oriented face indices.
        - side_index: (M, 2) long tensor.
            - side_index[i, 0] is the index of the tet for the "front"
              face (faces[i]).
            - side_index[i, 1] is the index of the "back" face tet,
              or -1 if it's a boundary face.
    """
    if not (tets.ndim == 2 and tets.shape[1] == 4):
        raise ValueError(f"Input tensor must have shape (N, 4), "
                         f"but got {tets.shape}")
        
    N = tets.shape[0]
    device = tets.device

    # ---
    # 1. Define all 4 faces for all N tets
    # We assume a standard winding order:
    # face 0: [v0, v1, v2] (base)
    # face 1: [v0, v3, v1] (side)
    # face 2: [v1, v3, v2] (side)
    # face 3: [v2, v3, v0] (side)
    # ---
    f0 = tets[:, [0, 1, 2]]
    f1 = tets[:, [0, 3, 1]]
    f2 = tets[:, [1, 3, 2]]
    f3 = tets[:, [2, 3, 0]]
    
    # Stack into a (4*N, 3) tensor
    all_faces = torch.stack([f0, f1, f2, f3], dim=1).reshape(-1, 3)

    # ---
    # 2. Create a (4*N) tensor to map each face back to its tet index
    # ---
    # [0, 0, 0, 0, 1, 1, 1, 1, ... N-1, N-1, N-1, N-1]
    tet_idx_map = torch.arange(N, device=device).unsqueeze(1).expand(N, 4).reshape(-1)

    # ---
    # 3. Create a unique, hashable key for each face by sorting its indices
    # ---
    # (4*N, 3)
    sorted_faces, _ = torch.sort(all_faces, dim=1)

    # ---
    # 4. Find unique sorted faces and group all 4*N faces by them
    # ---
    # unique_sorted_keys: (M, 3) tensor, where M is num unique faces
    # inverse_map: (4*N) tensor mapping each of the 4N faces to its
    #                unique ID (from 0 to M-1)
    unique_sorted_keys, inverse_map = torch.unique(
        sorted_faces, dim=0, return_inverse=True
    )
    
    M = unique_sorted_keys.shape[0]

    # ---
    # 5. Sort by the unique ID. This groups matching faces (pairs) together.
    # ---
    # perm will be a (4*N) tensor of indices [0...4N-1]
    perm = torch.argsort(inverse_map)
    
    # Re-order all our data using this permutation
    sorted_inverse_map = inverse_map[perm]  # [0, 0, 1, 1, 2, 3, 3, ...]
    sorted_tet_idx = tet_idx_map[perm]
    sorted_all_faces = all_faces[perm]      # This has the original winding

    # ---
    # 6. Find the *first* instance of each unique face
    # ---
    # We find where the ID changes in sorted_inverse_map
    # [True, False, True, False, True, True, False, ...]
    change_mask = torch.cat(
        (torch.tensor([True], device=device), 
         sorted_inverse_map[1:] != sorted_inverse_map[:-1])
    )
    # first_indices will be (M) tensor giving the index into the
    # sorted_* tensors for the first instance of each unique face.
    first_indices = torch.where(change_mask)[0]

    # ---
    # 7. Build the final (M, 3) faces and (M, 2) side_index
    # ---
    
    # Initialize outputs
    # The (M, 3) oriented faces
    faces = torch.zeros((M, 3), dtype=torch.long, device=device)
    # The (M, 2) adjacency map, default to -1 (boundary)
    side_index = torch.full((M, 2), -1, dtype=torch.long, device=device)

    # Fill the "front" face (slot 0) for ALL M faces
    # This is the oriented face from the first tet we found
    faces = sorted_all_faces[first_indices]
    side_index[:, 0] = sorted_tet_idx[first_indices]

    # ---
    # 8. Find internal faces (duplicates) and fill the "back" (slot 1)
    # ---
    
    # A count of 2 means it's an internal face
    counts = torch.bincount(sorted_inverse_map, minlength=M)
    internal_mask = (counts == 2)
    
    # Get the M-indices of just the internal faces
    internal_face_indices_M = torch.where(internal_mask)[0]
    
    # The *second* instance of these faces is at `first_indices + 1`
    # in the sorted tensors
    second_indices_sorted = first_indices[internal_mask] + 1
    
    # Get the tet ID for the second face in the pair
    second_tet_idx = sorted_tet_idx[second_indices_sorted]
    
    # Scatter these tet IDs into slot 1 of the side_index
    side_index[internal_face_indices_M, 1] = second_tet_idx

    return faces, side_index

# Inherit from Function
class SplineTracer(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(
        ctx: Any,
        vertices: torch.Tensor,
        face_indices: torch.Tensor,
        side_index: torch.Tensor,
        density: torch.Tensor,
        color: torch.Tensor,
        rayo: torch.Tensor,
        rayd: torch.Tensor,
        tmin: float,
        tmax: float,
        max_iters: int,
        lookup_tool,
        return_extras: bool = False,
    ):
        ctx.device = rayo.device
        st = time.time()
        # otx = sp.OptixContext(ctx.device)
        ctx.prims = sp.Primitives(ctx.device)
        assert density.device == ctx.device
        density = density.contiguous()
        color = color.contiguous()

        # assume all same start. Not strictly necessary
        start_tet_ids = lookup_tool.lookup(rayo[0].reshape(1, 3).cuda())
        start_tet_ids = start_tet_ids * torch.ones((rayo.shape[0]), dtype=torch.int, device='cuda')
        face_indices = face_indices.int()
        side_index = side_index.int()

        ctx.prims.add_primitives(vertices, face_indices, side_index, density, color)

        ctx.gas = sp.GAS(otx, ctx.device, ctx.prims, True, False, True)

        ctx.forward = sp.Forward(otx, ctx.device, ctx.prims, True)
        ctx.max_iters = max_iters
        out = ctx.forward.trace_rays(
            ctx.gas, rayo, rayd, tmin, tmax, ctx.max_iters, start_tet_ids)
        ctx.saved = out["saved"]
        ctx.tmin = tmin
        ctx.tmax = tmax
        tri_collection = out["tri_collection"]

        states = ctx.saved.states.reshape(rayo.shape[0], -1)
        distortion_pt1 = states[:, 0]
        distortion_pt2 = states[:, 1]
        distortion_loss = (distortion_pt1 - distortion_pt2)
        color_and_loss = torch.cat([out["color"], distortion_loss.reshape(-1, 1)], dim=1)

        ctx.save_for_backward(
            vertices, face_indices, side_index, density, color, rayo, rayd, tri_collection, start_tet_ids
        )

        if return_extras:
            return color_and_loss, dict(
                tri_collection=tri_collection,
                iters=ctx.saved.iters,
                opacity=out["color"][:, 3],
                touch_count=ctx.saved.touch_count,
                saved=ctx.saved,
            )
        else:
            return color_and_loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, return_extras=False):
        (
            vertices,
            face_indices,
            side_index,
            density,
            features,
            rayo,
            rayd,
            tri_collection,
            start_tet_ids
        ) = ctx.saved_tensors
        device = ctx.device
        assert(grad_output.shape[1] == 5)

        num_prims = features.shape[0]
        num_rays = rayo.shape[0]
        dL_dvertices = torch.zeros((vertices.shape[0], 3), dtype=torch.float32, device=device)
        dL_ddensities = torch.zeros((num_prims), dtype=torch.float32, device=device)
        dL_dfeatures = torch.zeros_like(features)
        dL_drayo = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)
        dL_drayd = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)


        touch_count = torch.zeros((num_prims), dtype=torch.int32, device=device)


        # block_size = 64
        block_size = 16
        st = time.time()
        # print(tri_collection.max())
        if ctx.saved.iters.sum() > 0:

            dual_model = (
                vertices,
                face_indices,
                side_index,
                density,
                features,
                dL_dvertices,
                dL_ddensities,
                dL_dfeatures,
                dL_drayo,
                dL_drayd,
            )
            kernels.backwards_kernel(
                last_state=ctx.saved.states,
                last_dirac=ctx.saved.diracs,
                iters=ctx.saved.iters,
                tri_collection=tri_collection,
                ray_origins=rayo,
                ray_directions=rayd,
                model=dual_model,
                touch_count=touch_count,
                dL_doutputs=grad_output.contiguous(),
                tmin=ctx.tmin,
                tmax=ctx.tmax,
                max_iters=ctx.max_iters,
                start_tet_ids=start_tet_ids,
            ).launchRaw(
                blockSize=(block_size, 1, 1),
                gridSize=(num_rays // block_size + 1, 1, 1),
            )

        # print('bw:', time.time()-st)
        # print(f"mean grad max: {dL_dmeans.abs().max()}, median: {torch.median(dL_dmeans.abs())}")
        # print(f"scale grad max: {dL_dscales}, median: {torch.median(dL_dscales.abs())}")
        # print(f"scale grad max: {dL_dscales.abs().max()}, median: {torch.median(dL_dscales.abs())}")
        # print(f"density grad max: {dL_ddensities.abs().max()}, median: {torch.median(dL_ddensities.abs())}")
        # print(f"feature grad max: {dL_dfeatures.abs().max()}, median: {torch.median(dL_dfeatures.abs())}")
        v = 1e+3
        mean_v = 1e+3
        return (
            dL_dvertices.clip(min=-mean_v, max=mean_v),
            None,
            None,
            dL_ddensities.clip(min=-v, max=v).reshape(density.shape),
            dL_dfeatures.clip(min=-v, max=v),
            dL_drayo.clip(min=-v, max=v),
            dL_drayd.clip(min=-v, max=v),
            None,
            None,
            None,
            None,
            None
        )


def trace_rays(
    vertices: torch.Tensor,
    face_indices: torch.Tensor,
    side_index: torch.Tensor,
    density: torch.Tensor,
    features: torch.Tensor,
    rayo: torch.Tensor,
    rayd: torch.Tensor,
    tmin: float = 0.2,
    tmax: float = 1000,
    lookup_tool = None,
    max_iters: int = 500,
    return_extras: bool = False,
):
    out = SplineTracer.apply(
        vertices,
        face_indices,
        side_index,
        density,
        features,
        rayo,
        rayd,
        tmin,
        tmax,
        max_iters,
        lookup_tool,
        return_extras,
    )
    return out

trace_rays.uses_density = True