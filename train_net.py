# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import numpy as np
from collections import OrderedDict
from typing import Any, Dict, List, Set

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader
)
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    SemanticSegEvaluator,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    RegisterRGBXSemSeg,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    add_fine_tuning_config,
)


DEFAULT_SEED = 42


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def __init__(self, cfg):
        super().__init__(cfg)

    def train(self):
        """
        Run training, and print parameter counts at the start.
        """
        self.print_param_counts()

        super().train()

    def build_hooks(self):
        hooks_list = super().build_hooks()

        best_cfg = getattr(self.cfg.TEST, "BEST_CHECKPOINTER", None)
        if (
            best_cfg is not None
            and getattr(best_cfg, "ENABLED", False)
            and self.cfg.TEST.EVAL_PERIOD > 0
        ):
            eval_hook_idx = next(
                (i for i, h in enumerate(hooks_list) if isinstance(h, hooks.EvalHook)),
                None,
            )
            if eval_hook_idx is not None:
                hooks_list.insert(
                    eval_hook_idx + 1,
                    hooks.BestCheckpointer(
                        self.cfg.TEST.EVAL_PERIOD,
                        self.checkpointer,
                        best_cfg.METRIC,
                        best_cfg.MODE,
                        best_cfg.FILE_PREFIX,
                    ),
                )

        return hooks_list
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        res = DefaultTrainer.test.__func__(cls, cfg, model, evaluators)
        if getattr(cfg.TEST, "AUG_ENABLED", False) and not getattr(model, "_mask2former_tta_model", False):
            res.update(cls.test_with_TTA(cfg, model))
        return res
    
    def print_param_counts(self):
        total_params = 0
        trainable_params = 0
        module_total_params = OrderedDict(
            (name, 0) for name in ["backbone", "pixel_decoder", "predictor", "other"]
        )
        module_trainable_params = OrderedDict(
            (name, 0) for name in ["backbone", "pixel_decoder", "predictor", "other"]
        )
        
        def _categorize_param(name):
            if name.startswith("backbone."):
                return "backbone"
            if name.startswith("sem_seg_head.pixel_decoder."):
                return "pixel_decoder"
            if name.startswith("sem_seg_head.predictor."):
                return "predictor"
            return "other"

        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            module_name = _categorize_param(name)
            module_total_params[module_name] += param_count

            if param.requires_grad:
                trainable_params += param_count
                module_trainable_params[module_name] += param_count

        logger = logging.getLogger("detectron2.trainer")
        logger.info(
            f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}, "
            f"Trainable ratio: {trainable_params / total_params * 100:.2f}%"
        )
        logger.info("Trainable parameter breakdown by module:")
        for module_name, total_count in module_total_params.items():
            trainable_count = module_trainable_params[module_name]
            trainable_ratio = (trainable_count / total_count * 100.0) if total_count > 0 else 0.0
            logger.info(
                f"  - {module_name}: total = {total_count:,}, trainable = {trainable_count:,}, "
                f"trainable_ratio = {trainable_ratio:.2f}%"
            )

    def load_filtered_weights(self, model, weights_path, skip_params=None):
        """
        Load pretrained weights while skipping specified parameters.

        Args:
            model: The model to load weights into
            weights_path: Path to the weights file
            skip_params: List of parameter names to skip
        """
        skip_params = skip_params or []
        checkpoint = DetectionCheckpointer(model)

        # Load checkpoint to memory
        loaded = checkpoint._load_file(weights_path)

        logger = logging.getLogger("detectron2.checkpoint")
        # If checkpoint has a model key
        if "model" in loaded:
            # Make a list of parameters to skip
            for param_name in skip_params:
                if param_name in loaded["model"]:
                    logger.info(f"Skipping loading parameter: {param_name}")
                    del loaded["model"][param_name]

            # Convert numpy arrays to tensors
            for k in loaded["model"]:
                if isinstance(loaded["model"][k], np.ndarray):
                    loaded["model"][k] = torch.from_numpy(loaded["model"][k])

        # Load the filtered checkpoint with strict=False
        incompatible = model.load_state_dict(loaded["model"], strict=False)
        # if incompatible.missing_keys:
        #     logger.info(f"Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.info(f"Unexpected keys: {incompatible.unexpected_keys}")

        return model

    def resume_or_load(self, resume=True):
        """
        Override to filter out specific parameters during loading.
        """
        if resume and self.checkpointer.has_checkpoint():
            super().resume_or_load(resume=True)

        else:
            if not self.cfg.MODEL.WEIGHTS:
                return

            # Skip loading specific parameters
            skip_params = [
                "sem_seg_head.predictor.static_query.weight",
                "sem_seg_head.predictor.query_embed.weight",
                "sem_seg_head.predictor.class_embed.weight",
                "sem_seg_head.predictor.class_embed.bias",
                "criterion.empty_weight",
            ]
            # Load weights with filtering
            self.load_filtered_weights(self.model, self.cfg.MODEL.WEIGHTS, skip_params)
            self.start_iter = 0  # Always start from iteration 0

    @staticmethod
    def unfreeze_params(cfg, model):
        """
        Unfreeze parts of the model based on the configuration.

        Strategy: freeze ALL parameters first, then selectively unfreeze adapter
        parameters based on which components are configured with adapters.
        This ensures only adapter parameters are trained during fine-tuning.

        Args:
        - cfg: Configuration object specifying which parts to freeze.
        - model: The model whose parameters will be frozen.
        """
        if not cfg.FINE_TUNING.ENABLE:
            return

        logger = logging.getLogger("detectron2.trainer")
        
        # Step 1: Freeze ALL parameters
        for param in model.parameters():
            param.requires_grad = False

        # Step 2: Unfreeze backbone adapter parameters
        backbone_trained_params = []
        if cfg.FINE_TUNING.BACKBONE.ADAPTER:
            for name, param in model.backbone.named_parameters():
                if "adapter" in name.lower():
                    param.requires_grad = True
                    backbone_trained_params.append(name)
            # DEBUG
            logger.info(f"Unfreezed `backbone` parameters: {backbone_trained_params}")

        # Step 3: Unfreeze pixel_decoder adapter parameters
        pixel_trained_params = []
        if cfg.FINE_TUNING.SEM_SEG_HEAD.PIXEL_DECODER.ADAPTER:
            for name, param in model.sem_seg_head.pixel_decoder.named_parameters():
                if "experts" in name.lower() or "router" in name.lower():
                    param.requires_grad = True
                    pixel_trained_params.append(name)
            # DEBUG
            logger.info(f"Unfreezed `pixel_decoder` parameters: {pixel_trained_params}")

        # Step 4: Unfreeze predictor (transformer_decoder) adapter parameters
        predictor_trained_params = []
        if cfg.FINE_TUNING.SEM_SEG_HEAD.PREDICTOR.ADAPTER:
            for name, param in model.sem_seg_head.predictor.named_parameters():
                if (
                    "adapter" in name.lower() or 
                    "query_feat" in name.lower() or 
                    "query_embed" in name.lower() or 
                    "class_embed" in name.lower() or
                    "mask_embed" in name.lower()
                ):
                    param.requires_grad = True
                    predictor_trained_params.append(name)
            # DEBUG
            logger.info(f"Unfreezed `predictor` parameters: {predictor_trained_params}")

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemanticSegEvaluator(
                    dataset_name,
                    # distributed=True,
                    distributed=False,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

        else:
            mapper = None
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        cls.unfreeze_params(cfg, model)  # unfreeze params before optimizer building

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        # select trainable params
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    if "adapter" not in module_name.lower():
                        hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        model._mask2former_tta_model = True
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fine_tuning_config(cfg)  # fine-tuning configs
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    seed_specified_in_cli = "SEED" in args.opts
    if cfg.SEED < 0 and not seed_specified_in_cli:
        cfg.SEED = DEFAULT_SEED

    cfg.freeze()
    default_setup(cfg, args)

    logger = logging.getLogger("detectron2")
    logger.info(f"Using training seed: {cfg.SEED}")

    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)
    # register custom dataset
    RegisterRGBXSemSeg(
        cfg.DATASETS.DATASET_ROOT,
        modal=cfg.DATASETS.MODAL,
        dataset_name=cfg.DATASETS.DATASET_NAME,
    ).register_dataset()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    # distributed training
    launch(
        main_func=main,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
