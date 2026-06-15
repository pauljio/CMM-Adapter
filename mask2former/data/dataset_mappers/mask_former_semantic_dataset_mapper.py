# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.structures import BitMasks, Instances
from torch.nn import functional as F

from mask2former.data import transforms as T

__all__ = ["MaskFormerSemanticDatasetMapper"]


class MaskFormerSemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        aug_enabled=True,
        *,
        augmentations,
        transforms,
        image_format,
        ignore_label,
        size_divisibility,
        modal,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            aug_enabled: whether to apply augmentation
            augmentations: a list of augmentations or deterministic transforms to apply
            transforms: apply transforms to the image
            image_format: an image format supported by :func:`detection_utils.read_image`
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
            modal: modalities of input images (e.g. "RGB-T")
        """
        self.is_train = is_train
        self.aug_enabled = aug_enabled
        self.augmentations = augmentations
        self.tfms = transforms
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.modal = modal

        logger = logging.getLogger(__name__)
        if is_train:
            logger.info(f"Augmentations used in training: {self.tfms}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        aug_enabled = cfg.INPUT.AUG_ENABLED
        # min_sizes = cfg.INPUT.MIN_SIZE_TRAIN
        # max_size = cfg.INPUT.MAX_SIZE_TRAIN
        crop_enabled = cfg.INPUT.CROP.ENABLED
        crop_size = cfg.INPUT.CROP.SIZE
        modal = cfg.DATASETS.MODAL
        color_aug_ssd = cfg.INPUT.COLOR_AUG_SSD
        flip = cfg.INPUT.FLIP

        dataset_names = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        augs = []
        if crop_enabled:
            # We replace RandomCrop with RandomResizedCrop here if desired
            # But the config gives crop_size as absolute size.
            augs.append(T.RandomResizedCrop(crop_size, seg_fill=ignore_label, modal=modal))
        if color_aug_ssd:
            augs.append(T.RandomColorJitter(p=0.2))
        if flip:
            augs.append(T.RandomHorizontalFlip(p=0.5))

        tfms = T.Augmentation(augs)

        ret = {
            "is_train": is_train,
            "aug_enabled": aug_enabled,
            "augmentations": augs,
            "transforms": tfms,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "modal": cfg.DATASETS.MODAL
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # read RGB images
        image_r = utils.read_image(dataset_dict["file_name_rgb"], format="RGB")  # format: RGB
        utils.check_image_size(dataset_dict, image_r)

        # read X images
        if self.modal == "RGB-T" and "file_name_t" in dataset_dict:
            image_x = utils.read_image(dataset_dict["file_name_t"], format="L")  # Gray
            image_x = image_x.repeat(3, axis=2)  # convert gray similar to RGB
        elif self.modal == "RGB-D" and "file_name_d" in dataset_dict:
            image_x = utils.read_image(dataset_dict["file_name_d"], format="RGB")  # HHA
            # image_x = utils.read_image(dataset_dict["file_name_d"], format="L")  # raw depth image
            # image_x = image_x.repeat(3, axis=2)  # convert gray similar to RGB
        elif self.modal == "RGB-P" and "file_name_p" in dataset_dict:
            image_x = utils.read_image(dataset_dict["file_name_p"], format="L")  # Gray
            image_x = image_x.repeat(3, axis=2)  # convert gray similar to RGB
        elif self.modal == "RGB-E" and "file_name_e" in dataset_dict:
            image_x = utils.read_image(dataset_dict["file_name_e"], format="RGB")  # RGB
        elif self.modal == "RGB-L" and "file_name_l" in dataset_dict:
            image_x = utils.read_image(dataset_dict["file_name_l"], format="RGB")  # RGB
            # image_x = utils.read_image(dataset_dict["file_name_l"], format="L")  # Gray
            # image_x = image_x.repeat(3, axis=2)  # convert gray similar to RGB
        else:
            raise ValueError(f"dataset_dict should have 'file_name_t', 'file_name_d', 'file_name_p', 'file_name_e' or"
                             f" 'file_name_l' key, but got {dataset_dict.keys()}")
        utils.check_image_size(dataset_dict, image_x)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            # read segmentation mask
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                ))

        # DEBUG
        # print(f"Before augmentation: image_r: {image_r.shape}, image_x: {image_x.shape}, sem_seg_gt: {sem_seg_gt.shape}")

        # Apply augmentation
        if self.is_train and self.aug_enabled:
            image_r, image_x, sem_seg_gt = self.tfms(image_r, image_x, sem_seg_gt)

        # DEBUG
        # print(f"After augmentation: image_r: {image_r.shape}, image_x: {image_x.shape}, sem_seg_gt: {sem_seg_gt.shape}")

        # Pad image and segmentation label here!
        # convert image and segmentation mask to Tensor
        image_r = torch.as_tensor(np.ascontiguousarray(image_r.transpose(2, 0, 1)).copy())  # C H W
        image_x = torch.as_tensor(np.ascontiguousarray(image_x.transpose(2, 0, 1)).copy())  # C H W
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        # adjust the size of the image to be divisible by self.size_divisibility
        # cfg.INPUT.SIZE_DIVISIBILITY = -1
        if self.size_divisibility > 0:
            image_size = (image_r.shape[-2], image_r.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image_r = F.pad(image_r, padding_size, value=128).contiguous()
            image_x = F.pad(image_x, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image_r.shape[-2], image_r.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore, it's important to use torch.Tensor.
        dataset_dict["image_r"] = image_r
        dataset_dict["image_x"] = image_x

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict
