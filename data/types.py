from typing import NamedTuple, Optional
import enum
import numpy as np

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

class ProjectionType(enum.Enum):
  """Camera projection type (perspective pinhole, fisheye, or 360 pano)."""

  PERSPECTIVE = 0
  FISHEYE = 1
  PANORAMIC = 2
  SIMPLE_RADIAL = 3


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    fovy: np.array
    fovx: np.array
    cx: Optional[float]
    cy: Optional[float]
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    model: ProjectionType
    distortion_params: Optional[dict]
    exposure: float
    iso: float
    aperature: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
