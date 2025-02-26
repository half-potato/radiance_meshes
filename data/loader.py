import os
import torch
import numpy as np
from PIL import Image

from data.dataset_readers import sceneLoadTypeCallbacks
from data.camera import Camera

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

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  fovx=cam_info.fovx, fovy=cam_info.fovy, cx=cam_info.cx, cy=cam_info.cy, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=data_device,
                  model=cam_info.model, distortion_params=cam_info.distortion_params,
                  exposure=cam_info.exposure, aperature=cam_info.aperature, iso=cam_info.iso)

def load_cameras(cam_infos, resolution_scale, resolution, data_device):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(load_cam(data_device, resolution, id, c, resolution_scale))

    return camera_list

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
    return train_cameras, test_cameras, scene_info
