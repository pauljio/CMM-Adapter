#!/usr/bin/env python3
"""Compute per-channel mean/std for MFNet training RGB and X(Thermal) modalities.

默认目录结构（由 prepare_mfnet.py 生成）：

MFNet/
  train_rgb/
  train_t/

示例：
  python datasets/MFNetDataset/compute_train_mean_std.py \
    --dataset-root /data1/cjp/Datasets/MFNet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 MFNet 训练集 RGB 与 X(Thermal) 模态的通道均值与标准差"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data1/cjp/Datasets/MFNet"),
        help="数据集根目录（应包含 train_rgb/ 和 train_x/）",
    )
    parser.add_argument(
        "--rgb",
        type=str,
        default="train_rgb",
        help="RGB 训练集目录名（相对 dataset-root）",
    )
    parser.add_argument(
        "--x",
        type=str,
        default="train_t",
        help="X 模态训练集目录名（相对 dataset-root）",
    )
    return parser.parse_args()


def collect_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"目录不存在: {folder}")

    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    files.sort()

    if not files:
        raise RuntimeError(f"目录中未找到图像文件: {folder}")

    return files


def compute_mean_std(image_paths: Iterable[Path], desc: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    sum_c: np.ndarray | None = None
    sumsq_c: np.ndarray | None = None
    total_pixels = 0
    num_images = 0

    for path in tqdm(list(image_paths), desc=desc, unit="img", dynamic_ncols=True):
        # 保持原始通道组织：RGB 为 3 通道；热图常见为单通道
        img = Image.open(path)
        arr = np.asarray(img, dtype=np.float64)

        if arr.ndim == 2:
            arr = arr[:, :, None]  # H x W -> H x W x 1
        elif arr.ndim == 3:
            pass
        else:
            raise ValueError(f"不支持的图像维度 {arr.ndim}: {path}")

        h, w, c = arr.shape
        pixels = h * w
        flat = arr.reshape(-1, c)

        cur_sum = flat.sum(axis=0)
        cur_sumsq = np.square(flat).sum(axis=0)

        if sum_c is None:
            sum_c = cur_sum
            sumsq_c = cur_sumsq
        else:
            if c != sum_c.shape[0]:
                raise ValueError(
                    f"通道数不一致: {path} 是 {c} 通道，但此前为 {sum_c.shape[0]} 通道"
                )
            sum_c += cur_sum
            sumsq_c += cur_sumsq

        total_pixels += pixels
        num_images += 1

    assert sum_c is not None and sumsq_c is not None
    mean = sum_c / total_pixels
    var = sumsq_c / total_pixels - np.square(mean)
    var = np.clip(var, a_min=0.0, a_max=None)
    std = np.sqrt(var)

    return mean, std, num_images, total_pixels


def format_array(arr: np.ndarray, precision: int = 6) -> str:
    return "[" + ", ".join(f"{v:.{precision}f}" for v in arr.tolist()) + "]"


def print_stats(name: str, mean_255: np.ndarray, std_255: np.ndarray, n: int, total_pixels: int) -> None:
    mean_01 = mean_255 / 255.0
    std_01 = std_255 / 255.0

    print(f"\n{name}:")
    print(f"  images      : {n}")
    print(f"  total pixels: {total_pixels}")
    print(f"  mean (0-255): {format_array(mean_255)}")
    print(f"  std  (0-255): {format_array(std_255)}")
    print(f"  mean (0-1)  : {format_array(mean_01)}")
    print(f"  std  (0-1)  : {format_array(std_01)}")


def main() -> None:
    args = parse_args()

    rgb_dir = args.root / args.rgb
    x_dir = args.root / args.x

    print("=" * 80)
    print("Train set mean/std statistics: ")
    print(f"dataset root: {args.root}")
    print(f"rgb folder  : {rgb_dir}")
    print(f"x folder    : {x_dir}")
    print("=" * 80)

    rgb_paths = collect_images(rgb_dir)
    x_paths = collect_images(x_dir)

    rgb_mean, rgb_std, rgb_n, rgb_pixels = compute_mean_std(rgb_paths, desc="RGB statistics")
    x_mean, x_std, x_n, x_pixels = compute_mean_std(x_paths, desc="X statistics")

    print_stats("RGB", rgb_mean, rgb_std, rgb_n, rgb_pixels)
    print_stats("X", x_mean, x_std, x_n, x_pixels)

    print("\nDone.")


if __name__ == "__main__":
    main()
