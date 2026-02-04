import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from data.types import ProjectionType
from icecream import ic
import math
from data.colmap_loader import rotmat2qvec
import torch.nn.functional as F

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, fovx, fovy, image, gt_alpha_mask=None,
                 image_name=None, uid=0, cx=-1, cy=-1,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 model=ProjectionType.PERSPECTIVE, distortion_params=None,
                 exposure=1, iso=100, aperature=1):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.fovx = fovx
        self.fovy = fovy
        self.cx = cx
        self.cy = cy
        self.image_name = image_name
        self.model = model
        self.distortion_params = torch.as_tensor(distortion_params).float().clone() if distortion_params is not None else torch.zeros((4))
        self.glo_vector = None
        self.exposure = exposure
        self.iso = iso
        self.aperature = aperature

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 10.0
        self.znear = 0.5

        self.trans = trans
        self.scale = scale
        self.update()
        self.data_device = self.world_view_transform.device

    def set_size(self, h, w):
        self.image_width = w
        self.image_height = h

        self.gt_alpha_mask = torch.ones(
            (1, self.image_height, self.image_width), device=self.data_device
        )

    def resize(self, h, w):
        self.original_image = F.interpolate(self.original_image[None], size=(h, w), mode='bilinear')[0]
        self.set_size(h, w)

    def resize_multiple(self, multiple):
        pad_h = (multiple - (self.image_height % multiple)) % multiple
        pad_w = (multiple - (self.image_width % multiple)) % multiple
        self.resize(self.image_height + pad_h, self.image_width + pad_w)

    @property
    def fx(self):
        return fov2focal(self.fovx, self.image_width)

    @property
    def fy(self):
        return fov2focal(self.fovy, self.image_height)

    def update(self):
        # world_view_transform takes world to view
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.fovx, fovY=self.fovy).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def to_dict(self, device):
        return dict(
            world_view_transform=self.world_view_transform.T.to(device),
            cam_pos=self.camera_center.to(device),
            # fx=fx,
            # fy=fy,
            K = torch.tensor([
                [self.fx, 0, self.image_width/2],
                [0, self.fy, self.image_height/2],
                [0, 0, 1],
            ]).to(device),
            image_height=self.image_height,
            image_width=self.image_width,
            fovy=self.fovy,
            fovx=self.fovx,
            distortion_params=self.distortion_params.to(device),
            camera_type=self.model.value,
        )

    @torch.no_grad()
    def _get_undistorted_coords(self, p_distorted: torch.Tensor, dist_params: torch.Tensor) -> torch.Tensor:
        """
        Iteratively computes undistorted coordinates from distorted ones.
        This is a vectorized PyTorch port of the 'undistort' Slang function.
        """
        # If no distortion, return immediately
        if torch.dot(dist_params, dist_params) < 1e-9:
            return p_distorted

        k1, k2, p1, p2 = dist_params[0], dist_params[1], dist_params[2], dist_params[3]
        
        # Initialize the guess for undistorted points
        p_undistorted = p_distorted.clone()

        # Iteratively refine the guess (Newton-Raphson method)
        for _ in range(10):
            x = p_undistorted[..., 0]
            y = p_undistorted[..., 1]

            # --- Compute residual and Jacobian ---
            r2 = x * x + y * y
            d = 1.0 + r2 * (k1 + r2 * k2)

            # Residuals (fx, fy)
            fx = d * x + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) - p_distorted[..., 0]
            fy = d * y + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y) - p_distorted[..., 1]

            # Jacobian
            d_r = k1 + r2 * (2.0 * k2)
            d_x = 2.0 * x * d_r
            d_y = 2.0 * y * d_r

            fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
            fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y
            fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
            fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y
            
            # --- Solve 2x2 linear system J * step = -F ---
            denominator = fy_x * fx_y - fx_x * fy_y
            x_numerator = fx * fy_y - fy * fx_y
            y_numerator = fy * fx_x - fx * fy_x

            # Create a mask for near-zero denominators to avoid division by zero
            denom_mask = denominator.abs() < 1e-9
            
            # Calculate step, avoiding NaNs
            step_x = x_numerator / denominator
            step_y = y_numerator / denominator
            step_x[denom_mask] = 0.0
            step_y[denom_mask] = 0.0
            
            # Update the guess
            p_undistorted[..., 0] += step_x
            p_undistorted[..., 1] += step_y

        return p_undistorted

    @torch.no_grad()
    def get_camera_space_directions(self, device=None, ray_jitter=None) -> torch.Tensor:
        """
        Computes the camera-space ray directions for every pixel.
        
        This is the "heavy" part of ray generation and depends only on
        intrinsics (K, distortion, H, W) and the camera model.
        
        The output of this function should be cached externally.

        Returns:
            torch.Tensor: A tensor of shape [H*W, 3] containing
                          the (x, y, z) direction vectors in
                          camera space.
        """
        if device is None:
            device = self.data_device

        H, W = self.image_height, self.image_width
        
        # --- Step 0: Get Intrinsic Parameters ---
        cam_dict = self.to_dict(device)
        K = cam_dict['K']
        dist_params = cam_dict['distortion_params']
        if ray_jitter is None:
            ray_jitter = 0.5*torch.ones((self.image_height, self.image_width, 2), device=device)
        else:
            assert(ray_jitter.shape[0] == self.image_height)
            assert(ray_jitter.shape[1] == self.image_width)
            assert(ray_jitter.shape[2] == 2)

        # --- Step 1: Create Pixel Grid & Normalized Distorted Coords ---
        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij"
        )
        Y = y + ray_jitter[..., 1]
        X = x + ray_jitter[..., 0]
        
        p_distorted_x = (X - K[0, 2]) / K[0, 0]
        p_distorted_y = (Y - K[1, 2]) / K[1, 1]
        p_distorted = torch.stack([p_distorted_x, p_distorted_y], dim=-1) # [H, W, 2]

        # --- Step 2: Undistort ---
        p_undistorted = self._get_undistorted_coords(p_distorted, dist_params) # [H, W, 2]

        # --- Step 3: Form Camera-Space Ray Direction ---
        px_u = p_undistorted[..., 0]
        py_u = p_undistorted[..., 1]
        
        if self.model == ProjectionType.FISHEYE:
            theta = torch.linalg.norm(p_undistorted, dim=-1)
            theta = torch.clamp_max(theta, 3.14159265)
            
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            
            sin_theta_over_theta = torch.ones_like(theta)
            mask = theta >= 1e-6
            sin_theta_over_theta[mask] = sin_theta[mask] / theta[mask]
            
            dir_cam = torch.stack([
                px_u * sin_theta_over_theta,
                py_u * sin_theta_over_theta,
                cos_theta
            ], dim=-1) # [H, W, 3]
            
        else: # Perspective
            dir_cam = torch.stack([
                px_u,
                py_u,
                torch.ones_like(px_u)
            ], dim=-1) # [H, W, 3]
            dir_cam = dir_cam / torch.linalg.norm(dir_cam, dim=-1, keepdim=True)
        
        # --- Step 4: Flatten ---
        # dir_cam shape is [H, W, 3] -> [H*W, 3]
        dir_cam_flat = dir_cam.reshape(-1, 3)
        
        return dir_cam_flat    

    @torch.no_grad()
    def get_world_space_rays(
        self,
        cam_space_dirs: torch.Tensor,
        device=None
    ) -> torch.Tensor:
        """
        Transforms pre-computed camera-space ray directions into
        world-space rays (origins and directions) using the
        camera's current pose (extrinsics).

        Args:
            cam_space_dirs (torch.Tensor): The [H*W, 3] tensor of
                camera-space directions, typically from
                get_camera_space_directions().
            device (torch.device, optional): The target device.

        Returns:
            torch.Tensor: A tensor of shape [H*W, 6]
                          (ray_origins, ray_directions).
        """
        if device is None:
            device = self.data_device

        # Ensure directions are on the correct device
        if cam_space_dirs.device != device:
            cam_space_dirs = cam_space_dirs.to(device)

        # --- Step 0: Get Extrinsic Parameters ---
        cam_dict = self.to_dict(device) 
        
        # View Matrix V = [R|T] (World-to-Cam)
        V = cam_dict['world_view_transform']
        # R_world2cam
        R_view = V[:3, :3]
        # R_cam2world (for transforming directions from cam to world)
        R_world = R_view.T
        # World-space camera center
        cam_center = cam_dict['cam_pos']

        # --- Step 1: Transform to World Space ---
        # (H*W, 3) @ (3, 3) -> (H*W, 3)
        ray_d_world = cam_space_dirs @ R_world.T
        
        # Ray origins are the camera center, repeated
        # (H*W, 3)
        ray_o_world = cam_center.reshape(1, 3).expand(ray_d_world.shape)
        
        # --- Step 2: Combine ---
        rays_tensor = torch.cat([ray_o_world, ray_d_world], dim=1)
        
        return rays_tensor

    @torch.no_grad()
    def to_rays(self, device=None) -> torch.Tensor:
        """
        Generates a tensor of rays for every pixel in the camera.
        
        Matches the logic of the Slang 'get_ray' function, including
        fisheye and perspective models with distortion.

        Returns:
            torch.Tensor: A tensor of shape [N_rays, 6], where N_rays = H * W.
                          Each row is [ox, oy, oz, dx, dy, dz].
        """
        if device is None:
            device = self.data_device

        H, W = self.image_height, self.image_width
        
        # --- Step 0: Get Camera Parameters ---
        cam_dict = self.to_dict(device)
        K = cam_dict['K']
        dist_params = cam_dict['distortion_params']
        
        # View Matrix V = [R|T] (World-to-Cam)
        V = cam_dict['world_view_transform'] 
        # R_world2cam
        R_view = V[:3, :3]
        # R_cam2world (for transforming directions from cam to world)
        R_world = R_view.T 
        # World-space camera center
        cam_center = cam_dict['cam_pos']

        # --- Step 1: Create Pixel Grid & Normalized Distorted Coords ---
        # Create a grid of pixel coordinates (at pixel centers)
        Y, X = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32) + 0.5,
            torch.arange(W, device=device, dtype=torch.float32) + 0.5,
            indexing="ij"
        )
        
        # p_distorted.x = (X - cx) / fx
        # p_distorted.y = (Y - cy) / fy
        p_distorted_x = (X - K[0, 2]) / K[0, 0]
        p_distorted_y = (Y - K[1, 2]) / K[1, 1]
        p_distorted = torch.stack([p_distorted_x, p_distorted_y], dim=-1) # Shape [H, W, 2]

        # --- Step 2: Undistort ---
        p_undistorted = self._get_undistorted_coords(p_distorted, dist_params) # Shape [H, W, 2]

        # --- Step 3: Form Camera-Space Ray Direction ---
        px_u = p_undistorted[..., 0]
        py_u = p_undistorted[..., 1]
        
        if self.model == ProjectionType.FISHEYE:
            # Fisheye model
            theta = torch.linalg.norm(p_undistorted, dim=-1)
            theta = torch.clamp_max(theta, 3.14159265) # Clamp to pi
            
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            
            # Handle theta ~= 0 to avoid 0/0
            sin_theta_over_theta = torch.ones_like(theta)
            mask = theta >= 1e-6
            sin_theta_over_theta[mask] = sin_theta[mask] / theta[mask]
            
            dir_cam = torch.stack([
                px_u * sin_theta_over_theta,
                py_u * sin_theta_over_theta,
                cos_theta
            ], dim=-1) # Shape [H, W, 3]
            
        else:
            # Perspective model (default)
            dir_cam = torch.stack([
                px_u,
                py_u,
                torch.ones_like(px_u)
            ], dim=-1) # Shape [H, W, 3]
            
            # Normalize
            dir_cam = dir_cam / torch.linalg.norm(dir_cam, dim=-1, keepdim=True)
            
        # --- Step 4: Transform to World Space & Format ---
        
        # Flatten directions for matrix multiplication
        # dir_cam shape is [H, W, 3] -> [H*W, 3]
        dir_cam_flat = dir_cam.reshape(-1, 3)
        
        # Transform directions from camera space to world space
        # ray_d = R_world @ dir_cam.T
        # (H*W, 3) @ (3, 3) -> (H*W, 3)
        ray_d_world = dir_cam_flat @ R_world 
        
        # Ray origins are just the camera center, repeated
        # (H*W, 3)
        ray_o_world = cam_center.reshape(1, 3).expand(ray_d_world.shape)
        
        # Combine into the final [N_rays, 6] tensor
        rays_tensor = torch.cat([ray_o_world, ray_d_world], dim=1)
        
        return rays_tensor

    def write_extrinsic(self, fid, i):
        """
        Writes the camera's extrinsics to a file object in
        COLMAP images.txt format.

        Args:
            fid: A file object opened in write mode.
        """
        # self.R is R_world2cam (3x3)
        # We need R_cam2world for qvec, which is the transpose
        R_cam2world = self.R.T
        qvec = rotmat2qvec(R_cam2world) # rotmat2qvec is assumed to be in scope
        
        # self.T is T_world2cam (3,)
        tvec = self.T
        
        # Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        fid.write(f"{i} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                  f"{tvec[0]} {tvec[1]} {tvec[2]} {self.uid} {self.image_name}\n")
        
        # Write the second line (2D keypoints), which we don't store.
        # An empty line is required for a valid COLMAP images.txt entry.

    def write_intrinsic(self, fid, i):
        """
        Writes the camera's intrinsics to a file object in
        COLMAP cameras.txt format.
        
        Tries to reverse-map the stored FoV and distortion
        parameters back to a compatible COLMAP model.

        Args:
            fid: A file object opened in write mode.
        """
        
        # Get focal lengths from FoV
        fx = fov2focal(self.fovx, self.image_width) # fov2focal is assumed to be in scope
        fy = fov2focal(self.fovy, self.image_height)
        
        # Get principal point, using image center as fallback
        cx = self.cx if self.cx != -1 else self.image_width / 2.0
        cy = self.cy if self.cy != -1 else self.image_height / 2.0
        
        # Get distortion parameters from the tensor
        # Based on _get_undistorted_coords, self.distortion_params is [k1, k2, p1, p2]
        dist_params = self.distortion_params.cpu().numpy()
        k1, k2, p1, p2 = dist_params[0], dist_params[1], dist_params[2], dist_params[3]
        
        model_name = ""
        params = []

        if self.model == ProjectionType.FISHEYE:
            # Map to SIMPLE_RADIAL_FISHEYE (f, cx, cy, k)
            # This is an approximation, assuming f=fx and k=k1
            model_name = "SIMPLE_RADIAL_FISHEYE"
            params = [fx, cx, cy, k1]
        
        elif self.model == ProjectionType.SIMPLE_RADIAL:
            # Map to SIMPLE_RADIAL (f, cx, cy, k)
            # This assumes f=fx and k=k1
            model_name = "SIMPLE_RADIAL"
            params = [fx, cx, cy, k1]

        else: # Default to PERSPECTIVE models
            if k2 != 0.0 or p1 != 0.0 or p2 != 0.0:
                # Map to OPENCV (fx, fy, cx, cy, k1, k2, p1, p2)
                model_name = "OPENCV"
                params = [fx, fy, cx, cy, k1, k2, p1, p2]
            elif k1 != 0.0:
                # Map to SIMPLE_RADIAL (f, cx, cy, k)
                # This assumes f=fx.
                model_name = "SIMPLE_RADIAL"
                params = [fx, cx, cy, k1]
            else:
                # Map to PINHOLE (fx, fy, cx, cy)
                model_name = "PINHOLE"
                params = [fx, fy, cx, cy]
                
        # Format: CAMERA_ID, MODEL_NAME, WIDTH, HEIGHT, PARAMS...
        param_str = " ".join(map(str, params))
        fid.write(f"{i} {model_name} {self.image_width} {self.image_height} {param_str}\n")
