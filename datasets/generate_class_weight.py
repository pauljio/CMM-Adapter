import os
import glob
import numpy as np
import cv2
import json
from typing import List
from tqdm import tqdm


def calculate_class_weights(
    dataset_path: str,
    output_file: str,
    num_classes: int,
    label_dirs: List[str] = None,
    ignore_index: int = 255,
    eps: float = 1e-12,
    normalize: bool = True,
    max_weight: float = None,
) -> List[float]:
    """
    Calculate class weights using a mild sqrt inverse-frequency scheme.

    Args:
        dataset_path (str): Path to the dataset directory.
        output_file (str): Path to save the class weights.
        num_classes (int): Number of classes in the dataset (excluding no-object class).
        label_dirs (List[str]): Label sub-directories to scan. Defaults to ["labels_train"].
        ignore_index (int): Label value to ignore when counting pixels.
        eps (float): Small value to avoid division-by-zero.
        normalize (bool): Whether to normalize non-zero weights to mean 1.
        max_weight (float): Optional upper bound to clip large weights.

    Returns:
        List[float]: Calculated weights for each class.
    """
    if label_dirs is None:
        label_dirs = ["labels_train"]

    class_counts = np.zeros(num_classes, dtype=np.int64)
    valid_pixels = 0
    total_files = 0

    # Count total files first for progress bar
    for label_dir in label_dirs:
        label_path = os.path.join(dataset_path, label_dir)
        if os.path.exists(label_path):
            label_files = glob.glob(os.path.join(label_path, '*.png'))
            total_files += len(label_files)

    print(f"Found {total_files} label files in total.")
    
    # Initialize progress bar
    progress_bar = tqdm(total=total_files, desc="Processing label files")

    # Check selected label directories
    for label_dir in label_dirs:
        label_path = os.path.join(dataset_path, label_dir)
        if os.path.exists(label_path):
            # Get all PNG label files
            label_files = glob.glob(os.path.join(label_path, '*.png'))
            for label_file in label_files:
                # Read label image as grayscale
                label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
                if label is not None:
                    valid_mask = label != ignore_index
                    valid_label = label[valid_mask]
                    if valid_label.size > 0:
                        valid_pixels += valid_label.size
                        bincount = np.bincount(valid_label, minlength=num_classes)
                        class_counts += bincount[:num_classes]
                progress_bar.update(1)
    
    progress_bar.close()

    if valid_pixels == 0:
        raise ValueError(f"No label images found in the dataset at {dataset_path}.")

    print("Calculating frequencies from valid (non-ignore) pixels...")

    # Calculate class frequencies
    p_class = class_counts.astype(np.float64) / float(valid_pixels)

    # Mild scheme: inverse sqrt of class frequency
    weights = np.zeros(num_classes, dtype=np.float64)
    present_mask = class_counts > 0
    weights[present_mask] = 1.0 / np.sqrt(p_class[present_mask] + eps)

    if normalize and np.any(present_mask):
        weights[present_mask] = weights[present_mask] / np.mean(weights[present_mask])

    if max_weight is not None:
        weights = np.clip(weights, a_min=0.0, a_max=max_weight)

    # Save weights to JSON file
    with open(output_file, 'w') as f:
        json.dump(weights.astype(np.float32).tolist(), f, indent=4)

    return weights.astype(np.float32).tolist()


if __name__ == "__main__":
    dataset_path = "/data1/cjp/Datasets/MFNet"
    # dataset_path = "/data1/cjp/Datasets/PST900"
    # dataset_path = "/data1/cjp/Datasets/NYU_Depth_V2"
    # dataset_path = "/data1/cjp/Datasets/SUN-RGBD""
    # dataset_path = "/data1/cjp/Datasets/ZJU-RGB-P"
    # dataset_path = "/data1/cjp/Datasets/EventScape"
    # dataset_path = "/data1/cjp/Datasets/DELIVER"

    num_classes = 9  # MFNet
    # num_classes = 5  # PST900
    # num_classes = 40  # NYU_Depth_V2
    # num_classes = 37  # SUN-RGBD
    # num_classes = 8  # ZJU-RGB-P
    # num_classes = 12  # EventScape
    # num_classes = 25  # DELIVER


    output_file = os.path.join(dataset_path, "class_weights.json")

    weights = calculate_class_weights(
        dataset_path=dataset_path,
        output_file=output_file,
        num_classes=num_classes,
        label_dirs=["labels_train"],
        ignore_index=255,
        normalize=True,
        max_weight=None,
    )
    print(f"Class weights calculated using sqrt scheme and saved to {output_file}: {weights}")
