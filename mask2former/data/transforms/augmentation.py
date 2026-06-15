import random
import torch
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image


__all__ = [
    "RandomColorJitter",
    "RandomHorizontalFlip",
    "RandomResizedCrop",
    "RandomGaussianBlur",
    "Augmentation",
]


class RandomColorJitter:
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p

    def __call__(self, image_r, image_x, sem_seg):
        if random.random() < self.p:
            fn_idx = torch.randperm(3)
            b = float(torch.empty(1).uniform_(max(0, 1 - self.brightness), 1 + self.brightness))
            c = float(torch.empty(1).uniform_(max(0, 1 - self.contrast), 1 + self.contrast))
            s = float(torch.empty(1).uniform_(max(0, 1 - self.saturation), 1 + self.saturation))

            for fn_id in fn_idx:
                if fn_id == 0 and self.brightness > 0:
                    image_r = TF.adjust_brightness(image_r, b)
                elif fn_id == 1 and self.contrast > 0:
                    image_r = TF.adjust_contrast(image_r, c)
                elif fn_id == 2 and self.saturation > 0:
                    image_r = TF.adjust_saturation(image_r, s)
        return image_r, image_x, sem_seg

    def __repr__(self):
        return f"RandomColorJitter(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, p={self.p})"


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image_r, image_x, sem_seg):
        if random.random() < self.p:
            image_r = TF.hflip(image_r)
            image_x = TF.hflip(image_x)
            sem_seg = TF.hflip(sem_seg)
        return image_r, image_x, sem_seg

    def __repr__(self):
        return f"RandomHorizontalFlip(p={self.p})"


class RandomResizedCrop:
    def __init__(self, size, scale=(0.5, 2.0), seg_fill=255, modal=None):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.scale = scale
        self.seg_fill = seg_fill
        self.modal = modal

    def __call__(self, image_r, image_x, sem_seg):
        H, W = image_r.size[1], image_r.size[0]  # PIL Image: (W, H)
        tH, tW = self.size

        # get the scale
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        scale_h, scale_w = int(tH * ratio), int(tW * 4 * ratio)

        # scale the image 
        scale_factor = min(max(scale_h, scale_w) / max(H, W), min(scale_h, scale_w) / min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)

        if self.modal in ["RGB-T", "RGB-P"]:
            interp_x = TF.InterpolationMode.BILINEAR
        elif self.modal in ["RGB-D", "RGB-E", "RGB-L"]:
            interp_x = TF.InterpolationMode.NEAREST
        else:
            interp_x = TF.InterpolationMode.BILINEAR

        image_r = TF.resize(image_r, (nH, nW), TF.InterpolationMode.BILINEAR)
        image_x = TF.resize(image_x, (nH, nW), interp_x)
        sem_seg = TF.resize(sem_seg, (nH, nW), TF.InterpolationMode.NEAREST)

        # random crop
        margin_h = max(nH - tH, 0)
        margin_w = max(nW - tW, 0)
        y1 = random.randint(0, margin_h)
        x1 = random.randint(0, margin_w)

        crop_h = min(nH, tH)
        crop_w = min(nW, tW)

        image_r = TF.crop(image_r, y1, x1, crop_h, crop_w)
        image_x = TF.crop(image_x, y1, x1, crop_h, crop_w)
        sem_seg = TF.crop(sem_seg, y1, x1, crop_h, crop_w)

        # pad the image
        pad_h = tH - crop_h
        pad_w = tW - crop_w
        if pad_h > 0 or pad_w > 0:
            padding = [0, 0, pad_w, pad_h]
            image_r = TF.pad(image_r, padding, fill=0)
            image_x = TF.pad(image_x, padding, fill=0)
            sem_seg = TF.pad(sem_seg, padding, fill=self.seg_fill)

        return image_r, image_x, sem_seg

    def __repr__(self):
        return f"RandomResizedCrop(size={self.size}, scale={self.scale}, seg_fill={self.seg_fill})"


class RandomGaussianBlur:
    def __init__(self, kernel_size=None, sigma=(0.1, 2.0), p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, image_r, image_x, sem_seg):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            kernel_size = self.kernel_size
            if kernel_size is None:
                kernel_size = int(3 * sigma)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel_size = max(3, kernel_size)
            image_r = TF.gaussian_blur(image_r, kernel_size, [sigma, sigma])
        return image_r, image_x, sem_seg

    def __repr__(self):
        return f"RandomGaussianBlur(kernel_size={self.kernel_size}, sigma={self.sigma}, p={self.p})"


class Augmentation:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image_r, image_x, sem_seg):
        is_np = isinstance(image_r, np.ndarray)
        if is_np:
            image_r = Image.fromarray(image_r)
            image_x = Image.fromarray(image_x)
            sem_seg = Image.fromarray(sem_seg)

        for t in self.transforms:
            image_r, image_x, sem_seg = t(image_r, image_x, sem_seg)
            
        if is_np:
            image_r = np.array(image_r)
            image_x = np.array(image_x)
            sem_seg = np.array(sem_seg)
            
        return image_r, image_x, sem_seg

    def __repr__(self):
        return f"Augmentation(transforms={self.transforms})"
