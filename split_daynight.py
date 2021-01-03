import os
import shutil
import time
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.splits import create_splits_logs



if __name__ == "__main__":
    nusc_version = "v1.0-trainval"
    nusc_kitti_dir = "/home/developer/nuscenes/nusc_kitti"
    split = "val"

    nusc = NuScenes(version=nusc_version, dataroot='/home/developer/nuscenes')
    
    label_folder_raw = os.path.join(nusc_kitti_dir, split, 'label_2')
    calib_folder_raw = os.path.join(nusc_kitti_dir, split, 'calib')
    image_folder_raw = os.path.join(nusc_kitti_dir, split, 'image_2')
    lidar_folder_raw = os.path.join(nusc_kitti_dir, split, 'velodyne')

    split_postfix = "_daytime"
    label_folder_daytime = os.path.join(nusc_kitti_dir, split + split_postfix, 'label_2')
    calib_folder_daytime = os.path.join(nusc_kitti_dir, split + split_postfix, 'calib')
    image_folder_daytime = os.path.join(nusc_kitti_dir, split + split_postfix, 'image_2')
    lidar_folder_daytime = os.path.join(nusc_kitti_dir, split + split_postfix, 'velodyne')
    for folder in [label_folder_daytime, calib_folder_daytime, image_folder_daytime, lidar_folder_daytime]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    split_postfix = "_night"
    label_folder_night = os.path.join(nusc_kitti_dir, split + split_postfix, 'label_2')
    calib_folder_night = os.path.join(nusc_kitti_dir, split + split_postfix, 'calib')
    image_folder_night = os.path.join(nusc_kitti_dir, split + split_postfix, 'image_2')
    lidar_folder_night = os.path.join(nusc_kitti_dir, split + split_postfix, 'velodyne')
    for folder in [label_folder_night, calib_folder_night, image_folder_night, lidar_folder_night]:
        if not os.path.isdir(folder):
            os.makedirs(folder)
            
    sample_tokens_list = [filename[:-4] for filename in os.listdir(os.path.join(nusc_kitti_dir, split, 'calib')) if filename.lower().endswith(("txt", "jpg", "png", "pcd", "bin"))]
    print(len(sample_tokens_list))

    for sample_token in sample_tokens_list:
        print("process file: {}".format(sample_token))
        sample_instance = nusc.get('sample', sample_token)
        scene_instance = nusc.get('scene', sample_instance['scene_token'])
        scene_description = scene_instance['description']
        flag_is_night_scene = True if "night" in scene_description.lower() else False
        if flag_is_night_scene:
            shutil.copy2(str(os.path.join(label_folder_raw, sample_token + ".txt")), str(os.path.join(label_folder_night, sample_token + ".txt")))
            shutil.copy2(str(os.path.join(calib_folder_raw, sample_token + ".txt")), str(os.path.join(calib_folder_night, sample_token + ".txt")))
            shutil.copy2(str(os.path.join(image_folder_raw, sample_token + ".png")), str(os.path.join(image_folder_night, sample_token + ".png")))
        else:
            shutil.copy2(str(os.path.join(label_folder_raw, sample_token + ".txt")), str(os.path.join(label_folder_daytime, sample_token + ".txt")))
            shutil.copy2(str(os.path.join(calib_folder_raw, sample_token + ".txt")), str(os.path.join(calib_folder_daytime, sample_token + ".txt")))
            shutil.copy2(str(os.path.join(image_folder_raw, sample_token + ".png")), str(os.path.join(image_folder_daytime, sample_token + ".png")))
        time.sleep(0.1)
            