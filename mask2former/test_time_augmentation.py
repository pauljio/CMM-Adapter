# Copyright (c) Facebook, Inc. and its affiliates.
import copy

import numpy as np
import torch
from fvcore.transforms import HFlipTransform
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from detectron2.data.detection_utils import read_image
from detectron2.modeling import DatasetMapperTTA


__all__ = [
    "SemanticSegmentorWithTTA",
]


class SemanticSegmentorTTAMapper:
    """Build test-time augmented inputs for both single-image and RGB-X inputs."""

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.default_mapper = DatasetMapperTTA(cfg)
        self.modal = cfg.DATASETS.MODAL

        resize_cfg = getattr(cfg.TEST, "RESIZE", None)
        self.flat_resize_enabled = bool(
            getattr(cfg.TEST, "AUG_ENABLED", False)
            and resize_cfg is not None
            and getattr(resize_cfg, "ENABLED", False)
        )
        self.flat_scales = list(getattr(resize_cfg, "SCALE", [])) if resize_cfg is not None else []

        self.aug_enabled = getattr(cfg.TEST.AUG, "ENABLED", False)
        self.aug_min_sizes = list(getattr(cfg.TEST.AUG, "MIN_SIZES", []))
        self.aug_max_size = getattr(cfg.TEST.AUG, "MAX_SIZE", 0)

        self.flip = bool(getattr(cfg.TEST, "FLIP", False) or getattr(cfg.TEST.AUG, "FLIP", False))

    def __call__(self, dataset_dict):
        if self._is_rgbx_input(dataset_dict):
            return self._call_rgbx(dataset_dict)
        return self.default_mapper(dataset_dict)

    def _is_rgbx_input(self, dataset_dict):
        return any(
            key in dataset_dict
            for key in [
                "image_r",
                "image_x",
                "file_name_rgb",
                "file_name_t",
                "file_name_d",
                "file_name_p",
                "file_name_e",
                "file_name_l",
            ]
        )

    def _call_rgbx(self, dataset_dict):
        image_r, image_x = self._get_source_images(dataset_dict)
        original_height = dataset_dict.get("height", image_r.shape[1])
        original_width = dataset_dict.get("width", image_r.shape[2])

        augmented_inputs = []
        for target_height, target_width in self._get_target_sizes(image_r.shape[1], image_r.shape[2]):
            resized_r = self._resize_tensor(image_r, target_height, target_width, mode="bilinear")
            resized_x = self._resize_tensor(image_x, target_height, target_width, mode=self._x_interp_mode())

            augmented_inputs.append(self._make_input(dataset_dict, resized_r, resized_x, original_height, original_width, False))
            if self.flip:
                augmented_inputs.append(
                    self._make_input(
                        dataset_dict,
                        torch.flip(resized_r, dims=[2]),
                        torch.flip(resized_x, dims=[2]),
                        original_height,
                        original_width,
                        True,
                    )
                )

        return augmented_inputs

    def _make_input(self, dataset_dict, image_r, image_x, height, width, flip):
        augmented = copy.copy(dataset_dict)
        augmented["image_r"] = image_r.contiguous()
        augmented["image_x"] = image_x.contiguous()
        augmented["height"] = height
        augmented["width"] = width
        augmented["transforms"] = {"flip": flip}
        return augmented

    def _get_source_images(self, dataset_dict):
        if "image_r" in dataset_dict and "image_x" in dataset_dict:
            image_r = self._to_float_tensor(dataset_dict["image_r"])
            image_x = self._to_float_tensor(dataset_dict["image_x"])
        else:
            image_r = self._read_rgb_image(dataset_dict)
            image_x = self._read_x_image(dataset_dict)

        height = dataset_dict.get("height")
        width = dataset_dict.get("width")
        if height is not None and width is not None:
            image_r = image_r[:, :height, :width]
            image_x = image_x[:, :height, :width]
        return image_r, image_x

    def _read_rgb_image(self, dataset_dict):
        image = read_image(dataset_dict["file_name_rgb"], "RGB")
        return self._to_float_tensor(image)

    def _read_x_image(self, dataset_dict):
        if self.modal == "RGB-T" and "file_name_t" in dataset_dict:
            image = read_image(dataset_dict["file_name_t"], "L")
            image = image.repeat(3, axis=2)
        elif self.modal == "RGB-D" and "file_name_d" in dataset_dict:
            image = read_image(dataset_dict["file_name_d"], "RGB")
        elif self.modal == "RGB-P" and "file_name_p" in dataset_dict:
            image = read_image(dataset_dict["file_name_p"], "L")
            image = image.repeat(3, axis=2)
        elif self.modal == "RGB-E" and "file_name_e" in dataset_dict:
            image = read_image(dataset_dict["file_name_e"], "RGB")
        elif self.modal == "RGB-L" and "file_name_l" in dataset_dict:
            image = read_image(dataset_dict["file_name_l"], "RGB")
            # image = image.repeat(3, axis=2)
        else:
            raise ValueError(
                "RGB-X TTA expects one of 'file_name_t', 'file_name_d', 'file_name_p', 'file_name_e' or 'file_name_l'."
            )
        return self._to_float_tensor(image)

    def _to_float_tensor(self, image):
        if isinstance(image, torch.Tensor):
            tensor = image.detach().clone()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            return tensor.float()

        array = np.asarray(image)
        if array.ndim == 2:
            array = array[:, :, None]
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()

    def _get_target_sizes(self, height, width):
        sizes = []
        if self.flat_resize_enabled and self.flat_scales:
            for scale in self.flat_scales:
                target_height = max(int(round(height * float(scale))), 1)
                target_width = max(int(round(width * float(scale))), 1)
                sizes.append((target_height, target_width))
        elif self.aug_enabled and self.aug_min_sizes:
            for min_size in self.aug_min_sizes:
                target_height, target_width = self._short_edge_resize_size(height, width, int(min_size), int(self.aug_max_size))
                sizes.append((target_height, target_width))
        else:
            sizes.append((height, width))

        unique_sizes = []
        seen = set()
        for size in sizes:
            if size not in seen:
                unique_sizes.append(size)
                seen.add(size)
        return unique_sizes

    def _short_edge_resize_size(self, height, width, min_size, max_size):
        short_edge = min(height, width)
        long_edge = max(height, width)
        scale = float(min_size) / float(short_edge)
        if max_size > 0 and scale * long_edge > max_size:
            scale = float(max_size) / float(long_edge)
        target_height = max(int(round(height * scale)), 1)
        target_width = max(int(round(width * scale)), 1)
        return target_height, target_width

    def _resize_tensor(self, image, target_height, target_width, mode):
        if image.shape[-2:] == (target_height, target_width):
            return image.float()

        kwargs = {}
        if mode in ["bilinear", "bicubic"]:
            kwargs["align_corners"] = False

        return F.interpolate(
            image.unsqueeze(0).float(),
            size=(target_height, target_width),
            mode=mode,
            **kwargs,
        ).squeeze(0)

    def _x_interp_mode(self):
        if self.modal in ["RGB-T", "RGB-P"]:
            return "bilinear"
        if self.modal in ["RGB-D", "RGB-E", "RGB-L"]:
            return "nearest"
        return "bilinear"


class SemanticSegmentorWithTTA(nn.Module):
    """
    A SemanticSegmentor with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`SemanticSegmentor.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=1):
        """
        Args:
            cfg (CfgNode):
            model (SemanticSegmentor): a SemanticSegmentor to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg.clone()

        self.model = model

        if tta_mapper is None:
            tta_mapper = SemanticSegmentorTTAMapper(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`SemanticSegmentor.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret and not self._is_rgbx_input(ret):
                image = read_image(ret.pop("file_name"), self.model.input_format)
                image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                if "image_r" in ret:
                    image = ret["image_r"]
                elif "image" in ret:
                    image = ret["image"]
                else:
                    raise ValueError("TTA input must contain either 'image' or 'image_r'.")
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        processed_results = []
        for x in batched_inputs:
            result = self._inference_one_image(_maybe_read_image(x))
            processed_results.append(result)
        return processed_results

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor
        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)

        final_predictions = None
        count_predictions = 0
        for input, tfm in zip(augmented_inputs, tfms):
            count_predictions += 1
            with torch.no_grad():
                is_flipped = self._is_hflip_transform(tfm)
                if final_predictions is None:
                    if is_flipped:
                        final_predictions = self.model([input])[0].pop("sem_seg").flip(dims=[2])
                    else:
                        final_predictions = self.model([input])[0].pop("sem_seg")
                else:
                    if is_flipped:
                        final_predictions += self.model([input])[0].pop("sem_seg").flip(dims=[2])
                    else:
                        final_predictions += self.model([input])[0].pop("sem_seg")

        final_predictions = final_predictions / count_predictions
        return {"sem_seg": final_predictions}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def _is_rgbx_input(self, dataset_dict):
        return "image_r" in dataset_dict or "file_name_rgb" in dataset_dict

    def _is_hflip_transform(self, tfm):
        if isinstance(tfm, dict):
            return tfm.get("flip", False)
        if hasattr(tfm, "flip"):
            return tfm.flip
        if hasattr(tfm, "transforms"):
            return any(isinstance(t, HFlipTransform) for t in tfm.transforms)
        return False
