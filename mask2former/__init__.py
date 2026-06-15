# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling
from .config import add_fine_tuning_config
# config
from .config import add_maskformer2_config
# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
from .data.datasets.register_rgbx_sem_seg import RegisterRGBXSemSeg
from .evaluation.instance_evaluation import InstanceSegEvaluator
# evaluation
from .evaluation.semantic_evaluation import SemanticSegEvaluator
# models
# from .maskformer_model import MaskFormer
from .maskformer_model import MaskFormerRGBX
from .test_time_augmentation import SemanticSegmentorWithTTA
