import time
from pathlib import Path
from typing import *

import slangtorch
import torch
from torch.autograd import Function
from icecream import ic
import numpy as np
from utils.model_util import activate_output
from dtlookup import TetrahedraLookup

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

class FastTracer:
    def __init__(self, vertices, face_indices, side_index, density, features, lookup_tool, tmin=0, tmax=1000, max_iters=200):
        self.face_indices = face_indices.to(torch.uint32)
        self.side_index = side_index.int().contiguous()
        # ic(side_index[25], face_indices[25], indices[13], rayo[0], rayd[0])
        # print(time.time()-st)
        self.density = density.contiguous()
        self.face_indices = face_indices.contiguous()
        self.vertices = vertices.contiguous()
        self.features = features.contiguous()

        self.device = vertices.device
        st = time.time()
        # otx = sp.OptixContext(self.device)
        self.prims = sp.Primitives(self.device)
        self.prims.add_primitives(self.vertices, self.face_indices, self.side_index, self.density, self.features)
        self.gas = sp.GAS(otx, self.device, self.prims, True, False, True)

        self.forward = sp.Forward(otx, self.device, self.prims, True)
        self.lookup_tool = lookup_tool
        self.tmin = tmin
        self.tmax = tmax
        self.max_iters = max_iters
        
    def trace(self, features, rayo, rayd):
        start_tet_ids = self.lookup_tool.lookup(rayo[0].reshape(1, 3).cuda())
        start_tet_ids = start_tet_ids * torch.ones((rayo.shape[0]), dtype=torch.int, device='cuda')
        self.prims.set_features(features)
        self.forward.update_model(self.prims)
        out = self.forward.trace_rays(
            self.gas, rayo.contiguous(), rayd.contiguous(), self.tmin, self.tmax, self.max_iters, start_tet_ids.contiguous())
        return out["color"]

def render_rt(camera, model, min_t=0):
    indices = torch.cat([model.indices, model.empty_indices], dim=0)
    # indices = model.indices
    vertices = model.vertices
    circumcenters, density, rgb, base_grd, sh, attr = model.compute_batch_features(
        vertices, model.indices, start=0, end=model.indices.shape[0])

    lookup = TetrahedraLookup(indices, vertices, 256)
    cds = camera.get_camera_space_directions('cuda')

    dvrgbs = activate_output(camera.camera_center.to(model.device),
                density, rgb, base_grd,
                sh.reshape(-1, (model.max_sh_deg+1)**2 - 1, 3),
                attr,
                model.indices, circumcenters, vertices,
                model.max_sh_deg, model.max_sh_deg)

    features = dvrgbs[:, 1:]
    grd = features[..., 3:]
    offset = ((-vertices[model.indices[..., 0]]) * grd).sum(dim=1, keepdim=True)
    base_color = features[..., :3] + offset
    features = torch.cat([base_color, grd], dim=1).contiguous()
    tracer = FastTracer(vertices, model.faces, model.side_index, dvrgbs[:, :1].contiguous(), features, lookup, min_t, 1000)

    rays = camera.get_world_space_rays(cds, 'cuda')
    image =  tracer.trace(
        features,
        rays[:, :3].contiguous(),
        rays[:, 3:].contiguous(),
    ).reshape(camera.image_height, camera.image_width, 4).permute(2, 0, 1)
    return {
        'render': image[:3],
    }

# Inherit from Function
class SplineTracer(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(
        ctx: Any,
        vertices: torch.Tensor,
        indices: torch.Tensor,
        face_indices: torch.Tensor,
        side_index: torch.Tensor,
        density: torch.Tensor,
        features: torch.Tensor,
        rayo: torch.Tensor, rayd: torch.Tensor,
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
        face_indices = face_indices.contiguous()
        vertices = vertices.contiguous()

        # assume all same start. Not strictly necessary
        st = time.time()
        start_tet_ids = lookup_tool.lookup(rayo[0].reshape(1, 3).cuda())
        start_tet_ids = start_tet_ids * torch.ones((rayo.shape[0]), dtype=torch.int, device='cuda')
        face_indices = face_indices.to(torch.uint32)
        side_index = side_index.int()
        # ic(side_index[25], face_indices[25], indices[13], rayo[0], rayd[0])
        # print(time.time()-st)

        st = time.time()
        ctx.prims.add_primitives(vertices, face_indices, side_index, density, features)
        ctx.gas = sp.GAS(otx, ctx.device, ctx.prims, True, False, True)

        ctx.forward = sp.Forward(otx, ctx.device, ctx.prims, True)
        # print(time.time()-st)
        st = time.time()
        ctx.max_iters = max_iters
        out = ctx.forward.trace_rays(
            ctx.gas, rayo, rayd, tmin, tmax, ctx.max_iters, start_tet_ids)
        # print(time.time()-st)
        ctx.saved = out["saved"]
        ctx.tmin = tmin
        ctx.tmax = tmax
        tri_collection = out["tri_collection"]
        # ic(tri_collection.reshape(ctx.max_iters, -1)[:, 0][:8])

        states = ctx.saved.states.reshape(rayo.shape[0], -1)
        distortion_pt1 = states[:, 0]
        distortion_pt2 = states[:, 1]
        distortion_loss = (distortion_pt1 - distortion_pt2)
        color_and_loss = torch.cat([out["color"], distortion_loss.reshape(-1, 1)], dim=1)

        ctx.save_for_backward(
            vertices, face_indices, side_index, density, features, rayo, rayd, tri_collection, start_tet_ids
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
                face_indices.int(),
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
    indices: torch.Tensor,
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
        indices,
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