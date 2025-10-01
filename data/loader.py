import os
import torch
import numpy as np
from PIL import Image

from data.dataset_readers import sceneLoadTypeCallbacks
from data.camera import Camera
from data.types import BasicPointCloud
from tqdm import tqdm

WARNED = False

def transform_poses_pca(poses):
    """
    Transforms poses so principal components lie on XYZ axes.

    Args:
        poses: A (N, 3, 4) array of camera-to-world transforms.

    Returns:
        A tuple (poses, transform) with the transformed poses and the
        applied 4x4 transformation matrix.
    """
    def pad(p):
        return np.concatenate([p, np.tile(np.eye(4)[3:, :], (p.shape[0], 1, 1))], axis=1)

    def unpad(p):
        return p[:, :3, :]

    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad(transform @ pad(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    return poses_recentered, transform

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def load_cam(data_device, resolution, id, cam_info, resolution_scale):
    image = Image.open(cam_info.image_path)
    orig_w, orig_h = image.size

    if resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * resolution)), round(orig_h/(resolution_scale * resolution))
    else:  # should be a type that converts to float
        if resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  fovx=cam_info.fovx, fovy=cam_info.fovy, cx=cam_info.cx, cy=cam_info.cy, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=data_device,
                  model=cam_info.model, distortion_params=cam_info.distortion_params,
                  exposure=cam_info.exposure, aperature=cam_info.aperature, iso=cam_info.iso)

def load_cameras(cam_infos, resolution_scale, resolution, data_device):
    camera_list = []
    for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos), desc="Loading cameras"):
        camera_list.append(load_cam(data_device, resolution, id, c, resolution_scale))
    if len(cam_infos) > 0:
        print(cam_infos[0])
    return camera_list

def transform_cameras_pca(cameras):
    if len(cameras) == 0:
        return cameras, np.eye(4)
    poses = np.stack([
        np.linalg.inv(view.world_view_transform.T.cpu().numpy())[:3]
        for view in cameras], axis=0)
    new_poses, transform = transform_poses_pca(poses)
    for i, cam in enumerate(cameras):
        T = np.eye(4)
        T[:3] = new_poses[i][:3]
        T = torch.linalg.inv(torch.tensor(T).float()).to(cam.world_view_transform.device)
        T[:3, 0] = T[:3, 0]*torch.linalg.det(T[:3, :3])
        cameras[i] = set_pose(cam, T)
    return cameras, transform

def set_pose(camera, T):
    # camera.world_view_transform = T.T
    # camera.full_proj_transform = (
    #     camera.world_view_transform.unsqueeze(0).bmm(
    #         camera.projection_matrix.unsqueeze(0))).squeeze(0)
    # camera.camera_center = camera.world_view_transform.inverse()[3, :3]
    camera.R = T[:3, :3].T.numpy()
    camera.T = T[:3, 3].numpy()
    camera.update()
    return camera


def load_dataset(source_path, images_folder, data_device, eval, white_background=True, resolution_scale=1.0, resolution=-1):
    if os.path.exists(os.path.join(source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, images_folder, eval)
    elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](source_path, white_background, eval)
    else:
        assert False, "Could not recognize scene type!"


    train_cameras = load_cameras(scene_info.train_cameras, resolution_scale, resolution, data_device)
    print(f"Loaded Train Cameras: {len(train_cameras)}")
    test_cameras = load_cameras(scene_info.test_cameras, resolution_scale, resolution, data_device)
    print(f"Loaded Test Cameras: {len(test_cameras)}")

    print("Transforming poses")
    _, pca_transform = transform_cameras_pca(train_cameras + test_cameras)
    xyz = scene_info.point_cloud.points
    xyz_hom = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    xyz_transformed_hom = (pca_transform @ xyz_hom.T).T
    transformed_pcd = scene_info.point_cloud._replace(points=xyz_transformed_hom[:, :3])
    scene_info = scene_info._replace(
        point_cloud=transformed_pcd,
        transform=pca_transform,
    )


    return train_cameras, test_cameras, scene_info
