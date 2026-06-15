import os.path

import numpy as np
from typing import List, Union
import torch
from PIL import Image
import glob
from tqdm import tqdm


def calculate_mean_std(image_paths):
    means = []
    stds = []

    for image_path in tqdm(image_paths, desc="Calculating mean and std"):
        image = Image.open(image_path)
        image = np.array(image)
        means.append(np.mean(image, axis=(0, 1)))
        stds.append(np.std(image, axis=(0, 1)))

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return mean, std


if __name__ == "__main__":
    # MFNet
    # rgb_folder = "/data1/cjp/Datasets/MFNetDataset/train_rgb"
    # t_folder = "/data1/cjp/Datasets/MFNetDataset/train_t"
    # rgb_paths = glob.glob(f"{rgb_folder}/*.png")
    # t_paths = glob.glob(f"{t_folder}/*.png")

    # PST900
    # rgb_folder = "/data1/cjp/Datasets/PST900/train_rgb"
    # t_folder = "/data1/cjp/Datasets/PST900/train_t"
    # rgb_paths = glob.glob(f"{rgb_folder}/*.png")
    # t_paths = glob.glob(f"{t_folder}/*.png")

    # NYU_Depth_V2
    # rgb_folder = "/data1/cjp/Datasets/NYU_Depth_V2/train_rgb"
    # d_folder = "/data1/cjp/Datasets/NYU_Depth_V2/train_depth"
    # hha_folder = "/data1/cjp/Datasets/NYU_Depth_V2/train_hha"
    # rgb_paths = glob.glob(f"{rgb_folder}/*.png")
    # d_paths = glob.glob(f"{d_folder}/*.png")
    # hha_paths = glob.glob(f"{hha_folder}/*.png")

    # SUN-RGBD
    # rgb_folder = "/data1/cjp/Datasets/SUN-RGBD/train_rgb"
    # d_folder = "/data1/cjp/Datasets/SUN-RGBD/train_depth"
    # hha_folder = "/data1/cjp/Datasets/SUN-RGBD/train_hha"
    # rgb_paths = glob.glob(f"{rgb_folder}/*.jpg")
    # d_paths = glob.glob(f"{d_folder}/*.png")
    # hha_paths = glob.glob(f"{hha_folder}/*.png")

    # Cityscapes
    # rgb_folder = "/data1/cjp/Datasets/Cityscapes/train_rgb"
    # d_folder = "/data1/cjp/Datasets/Cityscapes/train_depth"
    # hha_folder = "/data1/cjp/Datasets/Cityscapes/train_hha"
    # rgb_paths = glob.glob(f"{rgb_folder}/*.png")
    # d_paths = glob.glob(f"{d_folder}/*.png")
    # hha_paths = glob.glob(f"{hha_folder}/*.png")

    # ZJU-RGB-P
    # rgb_folder = "/data1/cjp/Datasets/ZJU-RGB-P/train_rgb"
    # aolp_folder = "/data1/cjp/Datasets/ZJU-RGB-P/train_aolp"
    # dolp_folder = "/data1/cjp/Datasets/ZJU-RGB-P/train_dolp"
    # rgb_paths = glob.glob(f"{rgb_folder}/*.png")
    # aolp_paths = glob.glob(f"{aolp_folder}/*.png")
    # dolp_paths = glob.glob(f"{dolp_folder}/*.png")

    # EventScape
    # rgb_folder = "/data1/cjp/Datasets/EventScape/train_rgb"
    # e_folder = "/data1/cjp/Datasets/EventScape/train_event"
    # rgb_paths = glob.glob(f"{rgb_folder}/*.png")
    # e_paths = glob.glob(f"{e_folder}/*.png")

    # DELIVER
    rgb_folder = "/data1/cjp/Datasets/DELIVER/train_rgb"
    # hha_folder = "/data1/cjp/Datasets/DELIVER/train_hha"
    # e_folder = "/data1/cjp/Datasets/DELIVER/train_event"
    l_folder = "/data1/cjp/Datasets/DELIVER/train_lidar"
    rgb_paths = glob.glob(f"{rgb_folder}/*.png")
    # hha_paths = glob.glob(f"{hha_folder}/*.png")
    # e_paths = glob.glob(f"{e_folder}/*.png")
    # l_paths = glob.glob(f"{l_folder}/*_color.png")
    l_paths = glob.glob(f"{l_folder}/*front.png")

    mean_rgb, std_rgb = calculate_mean_std(rgb_paths)
    # mean_t, std_t = calculate_mean_std(t_paths)
    # mean_d, std_d = calculate_mean_std(d_paths)
    # mean_hha, std_hha = calculate_mean_std(hha_paths)
    # mean_aolp, std_aolp = calculate_mean_std(aolp_paths)
    # mean_dolp, std_dolp = calculate_mean_std(dolp_paths)
    # mean_e, std_e = calculate_mean_std(e_paths)
    mean_l, std_l = calculate_mean_std(l_paths)

    print(f"mean_rgb: {mean_rgb}, std_rgb: {std_rgb}")
    # print(f"mean_t: {mean_t}, std_t: {std_t}")
    # print(f"mean_d: {mean_d}, std_d: {std_d}")
    # print(f"mean_hha: {mean_hha}, std_hha: {std_hha}")
    # print(f"mean_aolp: {mean_aolp}, std_aolp: {std_aolp}")
    # print(f"mean_dolp: {mean_dolp}, std_dolp: {std_dolp}")
    # print(f"mean_e: {mean_e}, std_e: {std_e}")
    print(f"mean_l: {mean_l}, std_l: {std_l}")

    # MFNetDataset
    # mean_rgb: [56.499, 65.976, 58.657], std_rgb: [42.671, 43.113, 42.843]
    # mean_t: [100.830, 100.830, 100.830], std_t: [19.323, 19.323, 19.323]

    # PST900
    # mean_rgb: [87.896, 88.792, 84.435], std_rgb: [59.174, 60.869, 60.293]
    # mean_t: [63.230, 63.230, 63.230], std_t: [14.613, 14.613, 14.613]

    # NYU_Depth_V2
    # mean_rgb: [120.224, 101.929, 95.899], std_rgb: [62.039, 62.126, 63.925]
    # mean_d: [26798.298, 26798.298, 26798.298], std_d: [8812.980, 8812.980, 8812.980]
    # mean_hha: [140.704, 110.851, 113.044], std_hha: [42.979, 60.620, 33.753]

    # SUN RGB-D
    # mean_rgb: [125.912, 116.549, 110.438], std_rgb: [65.313, 66.655, 67.422]
    # mean_d: [19074.531, 19074.531, 19074.531], std_hha-raw: [7116.634, 7116.634, 7116.634]
    # mean_hha: [182.915, 91.306, 104.844], std_hha: [45.315, 40.044, 36.519]

    # Cityscapes
    # mean_rgb: [73.158, 82.909, 72.392], std_rgb: [44.915, 46.153, 45.319]
    # mean_d: [33502.961, 33502.961, 33502.961], std_d: [24159.327, 24159.327, 24159.327]
    # mean_hha: [], std_hha: []

    # ZJU-RGB-P
    # mean_rgb: [70.206, 70.331, 69.142], std_rgb: [63.893, 64.148, 64.577]
    # mean_aolp: [123.467, 123.467, 123.467], std_aolp: [46.702, 46.702, 46.702]
    # mean_dolp: [37.628, 37.628, 37.628], std_dolp: [30.675, 30.675, 30.675]

    # EventScape
    # mean_rgb: [94.130, 94.951, 97.051], std_rgb: [51.293, 49.680, 50.927]
    # mean_e: [9.644, 0.0, 10.724], std_e: [45.399, 0.0, 47.629]

    # DELIVER
    # mean_rgb:[86.065, 84.333, 82.844], std_rgb: [39.256, 37.491, 36.251]
    # mean_hha: [118.145, 66.146, 110.639], std_hha: [105.124, 77.536, 43.666]
    # mean_e: [15.675, 0. , 16.476], std_e: [54.403, 0. ,55.628]
    # mean_l (color): [236.389, 238.331, 252.892], std_l: [55.146, 52.924, 13.570]
    # mean_l (gray): [1.733, 1.733, 1.733], std_: [7.389, 7.389, 7.389]
