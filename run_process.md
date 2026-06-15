### Run Mask2Former

### 1. 环境配置

1. 创建conda环境。

    ```bash
    conda create -n mask2former python=3.8
    # 激活环境
    conda activate mask2former
    ```

2. 安装 torch-1.10.1 和 opencv 。

    ```bash
    # CUDA 11.3
    conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
    
    # OpenCV
    pip install -U opencv-python
    
    ```
    - pip 换源
    ```bash
    # 升级pip
    python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --upgrade pip
    # 清华源
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    # 默认源
    pip config set global.index-url https://pypi.org/simple
    ```

3. 安装 Detectron2 。

    ```bash
    # under your working directory
    git clone https://github.com/facebookresearch/detectron2.git
    
    cd detectron2
    
    # rebuild detectron2 should do this:
    rm -rf build/ **/*.so
    
    # build detectron2
    pip install -e .
    ```
    
4. clone 项目并安装其他依赖。

    ```bash
    cd ..
    git clone https://github.com/facebookresearch/Mask2Former.git
    cd Mask2Former
    pip install -r requirements.txt
    
    pip install git+https://github.com/cocodataset/panopticapi.git
    pip install git+https://github.com/mcordts/cityscapesScripts.git
    ```

5. 为 MSDeformAttn 编译 CUDA 内核。

    ```bash
    cd mask2former/modeling/pixel_decoder/ops
    
    # rebuild should do this:
    rm -rf build/ dist/ *.egg-info
    
    # build
    bash make.sh
    # 或者执行
    python setup.py build install
    ```


### 2. 预训练模型推理 demo

1. 从 [model_zoo](MODEL_ZOO.md#semantic-segmentation) 下载预训练模型。

    ```bash
    # 下载 maskformer2_cityscapes_semantic_R101_bs16_90k.pkl
    wget https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_R101_bs16_90k/model_final_257ce8.pkl
    ```

2. 选择 config 配置文件并进行推理。

    ```bash
    # 推理 
    cd demo/
    
    CUDA_VISIBLE_DEVICES=2 python demo.py --config-file ../configs/cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k.yaml --input images/00295D.png --output output/ --opts MODEL.WEIGHTS ../model_zoo/maskformer2_cityscapes_semantic_R101_bs16_90k.pkl 
    ```

### 3. 数据集准备

#### RGB-T

##### MFNetDataset 1568/392/393

1. 使用 `make_flip.py` 制作左右翻转图像供训练使用。

2. 使用 `rgb_t_split.py` 将数据集中的 RGBA 图片通过拆分为 RGB 与 Thermal 图片。

3. 使用 `train_test_val_split.py` 将数据集的 RGB 图片以及 label 灰度图片根据 `train.txt`，`test.txt`，`val.txt` 拆分为训练集，测试集，验证集。

4. 使用 `json2coco.py` 将数据集标签 `json` 文件转换为 COCO 格式，标签格式详见 [Detectron2官方文档](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html# "https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#") 。最后生成 `semantic_train.json` 和 `semantic_val.json` 。

5. 数据集文件组织如下：

    ```
    datasets/
    └──MFNetDataset/
        ├──annotations/
        |  ├──semantic_train.json
        |  └──semantic_val.json
        ├──train_rgb/				# image files that are mentioned in the corresponding json
        ├──train_t/
        ├──val_rgb/
        ├──val_t/
        ├──test_rgb/				# image files for model testing
        ├──test_t/
        ├──labels_train/			# png annotations
        ├──labels_val/
        └──labels_test/
    ```

##### PST900 597/288



#### RGB-D

##### NYU Depth V2 795/654

##### SUN RGB-D 5285/5050

##### Cityscapes 2975/500/1525

##### DELIVER 2882/1733/1270



#### RGB-P

##### ZJU 344/50

#### RGB-E

##### EventScape 4409/810

##### DELIVER 2882/1733/1270



#### RGB-L

##### DELIVER 2882/1733/1270





### 4. 模型训练

1. 修改 `train_net.py` 训练脚本。

    - 添加 `Register` 类：

        ```python
        from detectron2.data import DatasetCatalog, MetadataCatalog
        import json
        
        
        class Register:
            """
            Register custom dataset and metadata
            """
        
            ## class names
            # MFNet Dataset
            stuff_classes = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
        
            ## class colors
            # MFNet Dataset
            stuff_colors = [
                (128, 128, 128),  # unlabeled: 灰色
                (255, 0, 0),      # car: 亮红色
                (0, 255, 0),      # person: 亮绿色
                (0, 0, 255),      # bike: 亮蓝色
                (255, 255, 0),    # curve: 黄色
                (255, 255, 255),  # car_stop: 白色
                (255, 20, 147),   # guardrail: 玫瑰红
                (255, 165, 0),    # color_cone: 橙色
                (0, 255, 255)     # bump: 青色
            ]
        
            dataset_root = "/data1/cjp/Projects/Mask2Former/datasets/MFNetDataset"
        
            def __init__(self):
                self.stuff_classes = Register.stuff_classes
                self.dataset_root = Register.dataset_root
        
                self.anno_root = os.path.join(self.dataset_root, "annotations")
        
                # TODO: add RGB-Thermal image path
                self.train_image_path = os.path.join(self.dataset_root, "train")
                self.val_image_path = os.path.join(self.dataset_root, "val")
        
                self.train_label_path = os.path.join(self.dataset_root, "labels_train")
                self.val_label_path = os.path.join(self.dataset_root, "labels_val")
        
                self.train_json = os.path.join(self.anno_root, "semantic_train.json")
                self.val_json = os.path.join(self.anno_root, "semantic_val.json")
        
                # split dataset into train and val by json file
                self.train_val_dataset = {
                    "MFNetDataset_train": (self.train_image_path, self.train_json),
                    "MFNetDataset_val": (self.val_image_path, self.val_json),
                }
        
            def get_dataset_dicts(self, json_file):
                if not os.path.exists(json_file):
                    raise FileNotFoundError(f"File '{json_file}' does not exist")
        
                with open(json_file, "r") as f:
                    dataset_dicts = json.load(f)
        
                # ensure the image path is full path
                # TODO: add RGB-Thermal image path
                for item in dataset_dicts:
                    if "train" in json_file:
                        item["file_name"] = os.path.join(self.train_image_path, item["file_name"])
                        item["sem_seg_file_name"] = os.path.join(self.train_label_path, item["sem_seg_file_name"])
        
                    elif "val" in json_file:
                        item["file_name"] = os.path.join(self.val_image_path, item["file_name"])
                        item["sem_seg_file_name"] = os.path.join(self.val_label_path, item["sem_seg_file_name"])
        
                return dataset_dicts
        
            def register_semantic_dataset(self, name, json_file, image_root):
                """
                Register a dataset with semantic segmentation annotations.
                """
        
                DatasetCatalog.register(name, lambda: self.get_dataset_dicts(json_file))
                MetadataCatalog.get(name).set(
                    json_file=json_file,
                    image_root=image_root,
                    evaluator_type="sem_seg",
                    ignore_label=255,
                    stuff_classes=self.stuff_classes,
                    stuff_colors=self.stuff_colors,
                )
        
            def register_dataset(self):
                """
                Register all custom datasets.
                """
        
                for key, (image_root, json_file) in self.train_val_dataset.items():
                    self.register_semantic_dataset(
                        name=key,
                        json_file=json_file,
                        image_root=image_root,
                    )
        ```

    - 在 `main` 方法中添加 `Register().register_dataset()`：

        ```python
        def main(args):
            Register().register_dataset()  # 注册自己的数据集
            cfg = setup(args)
            
            # 其余部分...
        ```

2. 修改模型文件 `mask2former/maskformer_model.py`

    - 添加自定义模型：

        ```python
        @META_ARCH_REGISTRY.register()
        class MaskFormerRGBT(MaskFormer):
            """
            Main class for mask classification RGB-T semantic segmentation architectures.
            """
        
            @configurable
            def __init__(
                self,
                *,
                backbone: Backbone,
                sem_seg_head: nn.Module,
                criterion: nn.Module,
                num_queries: int,
                object_mask_threshold: float,
                overlap_threshold: float,
                metadata,
                size_divisibility: int,
                sem_seg_postprocess_before_inference: bool,
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                # inference
                semantic_on: bool,
                panoptic_on: bool,
                instance_on: bool,
                test_topk_per_image: int,
            ):
                """
                Args:
                    backbone: a backbone module, must follow detectron2's backbone interface
                    sem_seg_head: a module that predicts semantic segmentation from backbone features
                    criterion: a module that defines the loss
                    num_queries: int, number of queries
                    object_mask_threshold: float, threshold to filter query based on classification score
                        for panoptic segmentation inference
                    overlap_threshold: overlap threshold used in general inference for panoptic segmentation
                    metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                        segmentation inference
                    size_divisibility: Some backbones require the input height and width to be divisible by a
                        specific integer. We can use this to override such requirement.
                    sem_seg_postprocess_before_inference: whether to resize the prediction back
                        to original input size before semantic segmentation inference or after.
                        For high-resolution dataset like Mapillary, resizing predictions before
                        inference will cause OOM error.
                    pixel_mean, pixel_std: list or tuple with #channels element, representing
                        the per-channel mean and std to be used to normalize the input image
                    semantic_on: bool, whether to output semantic segmentation prediction
                    instance_on: bool, whether to output instance segmentation prediction
                    panoptic_on: bool, whether to output panoptic segmentation prediction
                    test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
                """
                super().__init__(
                    backbone=backbone,
                    sem_seg_head=sem_seg_head,
                    criterion=criterion,
                    num_queries=num_queries,
                    object_mask_threshold=object_mask_threshold,
                    overlap_threshold=overlap_threshold,
                    metadata=metadata,
                    size_divisibility=size_divisibility,
                    sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
                    pixel_mean=pixel_mean,
                    pixel_std=pixel_std,
                    semantic_on=semantic_on,
                    panoptic_on=panoptic_on,
                    instance_on=instance_on,
                    test_topk_per_image=test_topk_per_image,
                )
        
                # TODO: pixel mean and std for RGB-Thermal images
                self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
                self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
                embed_dim = backbone.embed_dim
        
                self.conv_fuse_res2 = nn.Conv2d(in_channels= embed_dim * 2, out_channels=embed_dim, kernel_size=1)
                self.conv_fuse_res3 = nn.Conv2d(in_channels= embed_dim * 4, out_channels=embed_dim * 2, kernel_size=1)
                self.conv_fuse_res4 = nn.Conv2d(in_channels= embed_dim * 8, out_channels=embed_dim * 4, kernel_size=1)
                self.conv_fuse_res5 = nn.Conv2d(in_channels= embed_dim * 16 , out_channels=embed_dim * 8, kernel_size=1)
        
            @classmethod
            def from_config(cls, cfg):
                # TODO: build backbone and head for 3 channel images
                backbone = build_backbone(cfg)
                sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        
                # Loss parameters:
                deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
                no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        
                # loss weights
                class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
                dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
                mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        
                # building criterion
                matcher = HungarianMatcher(
                    cost_class=class_weight,
                    cost_mask=mask_weight,
                    cost_dice=dice_weight,
                    num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                )
        
                weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        
                if deep_supervision:
                    dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
                    aux_weight_dict = {}
                    for i in range(dec_layers - 1):
                        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                    weight_dict.update(aux_weight_dict)
        
                losses = ["labels", "masks"]
        
                criterion = SetCriterion(
                    sem_seg_head.num_classes,
                    matcher=matcher,
                    weight_dict=weight_dict,
                    eos_coef=no_object_weight,
                    losses=losses,
                    num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                    oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                    importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
                )
        
                return {
                    "backbone": backbone,
                    "sem_seg_head": sem_seg_head,
                    "criterion": criterion,
                    "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                    "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                    "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                    "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                    "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                    "sem_seg_postprocess_before_inference": (
                            cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                            or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                            or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                    ),
                    "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                    "pixel_std": cfg.MODEL.PIXEL_STD,
                    # inference
                    "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                    "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                    "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                    "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
                }
        
            def forward(self, batched_inputs):
                """
                Args:
                    batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                        Each item in the list contains the inputs for one RGB-Thermal image.
                        For now, each item in the list is a dict that contains:
                           * "image": Tensor, image in (C, H, W) format.
                           * "instances": per-region ground truth
                           * Other information that's included in the original dicts, such as:
                             "height", "width" (int): the output resolution of the model (may be different
                             from input resolution), used in inference.
                Returns:
                    list[dict]:
                        each dict has the results for one image. The dict contains the following keys:
        
                        * "sem_seg":
                            A Tensor that represents the
                            per-pixel segmentation prediced by the head.
                            The prediction has shape KxHxW that represents the logits of
                            each class for each pixel.
                        * "panoptic_seg":
                            A tuple that represent panoptic output
                            panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                            segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                                Each dict contains keys "id", "category_id", "isthing".
                """
                # TODO: add thermal image to the input
                images= [x["image"].to(self.device) for x in batched_inputs]
        
                # TODO: add mean and std for RGB-Thermal image
                images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        
                images = ImageList.from_tensors(images, self.size_divisibility)  # (B, 4, H, W)
        
                # TODO: get RGB and Thermal features separately
                # separate RGB and Thermal images
                images_rgb = images.tensor[:, :3, :, :]
                images_t = images.tensor[:, 3:, :, :].repeat(1, 3, 1, 1)  # repeat to create a 3-channel image
        
                # TODO: add backbone for Thermal images
                features_rgb = self.backbone(images_rgb)
                features_t = self.backbone(images_t)
        
                # fuse RGB and Thermal features
                # TODO: add more fusion methods
                # Concat & Conv
                features_fusion = {
                    "res2": self.conv_fuse_res2(torch.cat((features_rgb["res2"], features_t["res2"]), dim=1)),
                    "res3": self.conv_fuse_res3(torch.cat((features_rgb["res3"], features_t["res3"]), dim=1)),
                    "res4": self.conv_fuse_res4(torch.cat((features_rgb["res4"], features_t["res4"]), dim=1)),
                    "res5": self.conv_fuse_res5(torch.cat((features_rgb["res5"], features_t["res5"]), dim=1)),
                }
        
                outputs = self.sem_seg_head(features_fusion)
        
                if self.training:
                    # mask classification target
                    if "instances" in batched_inputs[0]:
                        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                        targets = self.prepare_targets(gt_instances, images)
                    else:
                        targets = None
        
                    # bipartite matching-based loss
                    losses = self.criterion(outputs, targets)
        
                    for k in list(losses.keys()):
                        if k in self.criterion.weight_dict:
                            losses[k] *= self.criterion.weight_dict[k]
                        else:
                            # remove this loss if not specified in `weight_dict`
                            losses.pop(k)
                    return losses
                else:
                    mask_cls_results = outputs["pred_logits"]
                    mask_pred_results = outputs["pred_masks"]
                    # upsample masks
                    mask_pred_results = F.interpolate(
                        mask_pred_results,
                        size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    )
        
                    del outputs
        
                    processed_results = []
                    for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                        mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                    ):
                        height = input_per_image.get("height", image_size[0])
                        width = input_per_image.get("width", image_size[1])
                        processed_results.append({})
        
                        if self.sem_seg_postprocess_before_inference:
                            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                                mask_pred_result, image_size, height, width
                            )
                            mask_cls_result = mask_cls_result.to(mask_pred_result)
        
                        # semantic segmentation inference
                        if self.semantic_on:
                            r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                            if not self.sem_seg_postprocess_before_inference:
                                r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                            processed_results[-1]["sem_seg"] = r
        
                        # panoptic segmentation inference
                        if self.panoptic_on:
                            panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                            processed_results[-1]["panoptic_seg"] = panoptic_r
        
                        # instance segmentation inference
                        if self.instance_on:
                            instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                            processed_results[-1]["instances"] = instance_r
        
                    return processed_results
        ```

3. 修改 `detectron2/detectron2/modeling/backbone/build.py` 的 `build_backbone` 方法，使得 backbone 只处理 3 通道的输入。

    ```python
    # input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    input_shape = ShapeSpec(channels=3)
    ```

4. 修改配置文件中的相关配置。

    - `Base-Cityscapes-SemanticSegmentation.yaml` ：

        ```yaml
        MODEL:
          PIXEL_MEAN: [56.499, 65.976, 58.657, 100.830]  # for MFNet RGB-T images
          PIXEL_STD: [42.671, 43.113, 42.843, 19.323]
          
        DATASETS:
          TRAIN: ("MFNetDataset_train",)  # my custom dataset
          TEST: ("MFNetDataset_val",)
          
        SOLVER:
          IMS_PER_BATCH: 8  # batch size
          BASE_LR: 0.0001   # learning rate
          MAX_ITER: 20000   # epoch: 100 = max iteration / data elements(1568) * batch size
          
        INPUT:
          MIN_SIZE_TRAIN: !!python/object/apply:eval ["[int(x * 0.1 * 640) for x in range(5, 21)]"]
          MIN_SIZE_TRAIN_SAMPLING: "choice"
          MIN_SIZE_TEST: 640
          MAX_SIZE_TRAIN: 2560
          MAX_SIZE_TEST: 1280
          CROP:
            SIZE: (640, 640)
          COLOR_AUG_SSD: False
          FORMAT: "RGBA"
        
        TEST:
          EVAL_PERIOD: 2000  # every 10 epoch
        ```
        
    - `maskformer2_R50_bs16_90k.yaml` ：

        ```yaml
        MODEL:
          META_ARCHITECTURE: "MaskFormerRGBT"  # my sem_seg_head
          NUM_CLASSES: 9  # 自己数据集中目标类别个数
          
        OUTPUT_DIR: "output/cityscapes/maskformer2_R50_bs16_90k"  # 输出目录
        ```
        
    - `maskformer2_R101_bs16_90k.yaml` ：

        ```yaml
        OUTPUT_DIR: "output/cityscapes/maskformer2_R101_bs16_90k"  # 输出目录
        ```

    - `maskformer2_swin_tiny_bs16_90k.yaml` ：

        ```yaml
        MODEL:  
          PIXEL_MEAN: [56.499, 65.976, 58.657, 100.830]
          PIXEL_STD: [42.671, 43.113, 42.843, 19.323]
          
        OUTPUT_DIR: "output/cityscapes/maskformer2_swin_tiny_bs16_90k"
        ```

    - `maskformer2_swin_small_bs16_90k.yaml` ：

        ```yaml
        MODEL:  
          PIXEL_MEAN: [56.499, 65.976, 58.657, 100.830]
          PIXEL_STD: [42.671, 43.113, 42.843, 19.323]
          
        OUTPUT_DIR: "output/cityscapes/maskformer2_swin_small_bs16_90k"
        ```

    - `maskformer2_swin_base_IN21k_384_bs16_90k.yaml` ：

        ```yaml
        MODEL:  
          PIXEL_MEAN: [56.499, 65.976, 58.657, 100.830]
          PIXEL_STD: [42.671, 43.113, 42.843, 19.323]
          
        OUTPUT_DIR: "output/cityscapes/maskformer2_swin_base_IN21k_384_bs16_90k"
        ```

    - `maskformer2_swin_large_IN21k_384_bs16_90k.yaml` ：

        ```yaml
        MODEL:  
          PIXEL_MEAN: [56.499, 65.976, 58.657, 100.830]
          PIXEL_STD: [42.671, 43.113, 42.843, 19.323]
          
        OUTPUT_DIR: "output/cityscapes/maskformer2_swin_large_IN21k_384_bs16_90k"
        ```

5. 运行 `train_net.py` 。

    - 所有命令行参数：

        ```bash
        usage: train_net.py [-h] [--config-file FILE] [--resume] [--eval-only] [--num-gpus NUM_GPUS] [--num-machines NUM_MACHINES] [--machine-rank MACHINE_RANK] [--dist-url DIST_URL]
        ...
        
        positional arguments:
          opts                  Modify config options at the end of the command. For Yacs configs, use space-separated "PATH.KEY VALUE" pairs. For python-based
                                LazyConfig, use "path.key=value".
        
        optional arguments:
          -h, --help            show this help message and exit
          --config-file FILE    path to config file
          --resume              Whether to attempt to resume from the checkpoint directory. See documentation of `DefaultTrainer.resume_or_load()` for what it means.
          --eval-only           perform evaluation only
          --num-gpus NUM_GPUS   number of gpus *per machine*
          --num-machines NUM_MACHINES
                                total number of machines
          --machine-rank MACHINE_RANK
                                the rank of this machine (unique per machine)
          --dist-url DIST_URL   initialization URL for pytorch distributed backend. See https://pytorch.org/docs/stable/distributed.html for details.
        
        Examples:
        
        Run on single machine:
            $ train_net.py --num-gpus 8 --config-file cfg.yaml
        
        Change some config options:
            $ train_net.py --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
        
        Run on multiple machines:
            (machine0)$ train_net.py --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine1)$ train_net.py --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        ```

    - 其他超参数：

        - `SOLVER.IMS_PER_BATCH`：批大小
        - `SOLVER.BASE_LR`：初始学习率
        - `SOLVER.MAX_ITER`：最大迭代次数
        - `TEST.EVAL_PERIOD`：评估周期
        - `FINE_TUNING.ENABLE`：是否启用微调
        - `OUTPUT_DIR`：输出保存目录

#### 命令行运行：

```bash
# 1gpu-r50
python train_net.py --num-gpus 1 \ 
--config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \ 
MODEL.WEIGHTS "model_zoo/maskformer2_cityscapes_semantic_r50_bs16_90k.pkl"

# 2gpu-r50
python train_net.py --num-gpus 2 \
--config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \
MODEL.WEIGHTS "model_zoo/maskformer2_cityscapes_semantic_r50_bs16_90k.pkl"

# 3gpu-r50
python train_net.py --num-gpus 3 \
--config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \
MODEL.WEIGHTS "model_zoo/maskformer2_cityscapes_semantic_r50_bs16_90k.pkl"

# 1gpu-r50-resume
python train_net.py --num-gpus 3 \
--config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \
MODEL.WEIGHTS "model_zoo/maskformer2_cityscapes_semantic_r50_bs16_90k.pkl" \
--resume  # 从checkpoint恢复训练
```

- 浏览器[打开](http://localhost:6006/ "http://localhost:6006/")，即可使用 Tensorboard 观察训练过程。
  
    ```bash
    # 启动tensorboard（本地运行）
    tensorboard --logdir output\cityscapes\maskformer2_R50_bs16_90k  # events.out.tfevents文件目录
    ```
    
#### bugs

- `UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').`

    - 在 `mask2former/modeling/transformer_decoder/position_encoding.py` 41 行 `forward` 方法中：

        ```python
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        ```

- `AttributeError: module 'distutils' has no attribute 'version'.` 

    ```bash
    pip install setuptools==59.5.0
    ```

- `RuntimeError: Default process group has not been initialized, please maske sure to call init_process_group.`

    - 在 `configs/cityscapes/semantic-segmentation/Base-Cityscapes-SemanticSegmentation.yaml` 和 `configs/citycapes/semantic-segmentation/maskformer2_R101_bs16_90k.yaml` 中：

        ```bash
        # NORM: "SyncBN"  # 用于分布式训练
        NORM: "BN"
        ```

- `Segmentation fault(core dumped). 段错误（核心已转储） `
  
    - 重新编译 `detectron2` 和 `MSDeformAttn` ，详见[环境配置](#1. 环境配置)。

### 5. 模型评估

- 命令行运行 `train_net.py` ，注意添加 `--eval-only` 参数。

    ```bash
    CUDA_VISIBLE_DEVICES=2 python train_net.py \ 
    --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \  # 模型配置
    --eval-only \
    MODEL.WEIGHTS output/cityscapes/maskformer2_R50_bs16_90k/model_final.pth                # 模型权重
    ```

### 6. 模型预测

1. 修改 `predict.py` 文件。

    - 添加 `Register` 类：

        ```python
        class Register:
            """注册自己的测试数据集"""
        
            # 类别名称
            stuff_classes = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
            # 类别配色
            stuff_colors = [
                (128, 128, 128),  # unlabeled: 灰色
                (255, 0, 0),  # car: 亮红色
                (0, 255, 0),  # person: 亮绿色
                (0, 0, 255),  # bike: 亮蓝色
                (255, 255, 0),  # curve: 黄色
                (255, 255, 255),  # car_stop: 白色
                (255, 20, 147),  # guardrail: 玫瑰红
                (255, 165, 0),  # color_cone: 橙色
                (0, 255, 255)  # bump: 青色
            ]
            # 数据集路径
            dataset_root = "/data1/cjp/Projects/Mask2Former/datasets/MFNetDataset"
        
            def __init__(self):
                self.stuff_classes = Register.stuff_classes
                self.dataset_root = Register.dataset_root
        
                self.test_image_path = os.path.join(self.dataset_root, "test")
        
            def get_dataset_dicts(self):
                """获取测试集的图像字典，不需要标签"""
        
                dataset_dicts = []
        
                for filename in os.listdir(self.test_image_path):
                    if filename.endswith((".jpg", ".png", ".jpeg")):  # 检查图像格式
                        file_path = os.path.join(self.test_image_path, filename)
                        image = cv2.imread(file_path)
                        height, width = image.shape[:2] if image is not None else (480, 640)
        
                        dataset_dicts.append({
                            "file_name": file_path,
                            "height": height,
                            "width": width,
                            "image_id": filename.split('.')[0],  # 使用文件名作为图像 ID
                            "sem_seg_file_name": None,  # 没有标签，则设置为 None
                        })
        
                return dataset_dicts
        
            def register_semantic_dataset(self, name):
                """注册数据集实例，加载数据集中的对象实例"""
        
                DatasetCatalog.register(name, self.get_dataset_dicts)
                MetadataCatalog.get(name).set(
                    image_root=self.test_image_path,
                    evaluator_type="sem_seg",
                    ignore_label=255,
                    stuff_classes=self.stuff_classes,
                    stuff_colors=self.stuff_colors,
                )
        
            def register_dataset(self, name):
                """将自定义数据集注册进 detectron2"""
        
                dataset_name = "MFNetDataset_test"
                self.register_semantic_dataset(
                    name=dataset_name,
                )
        
                return dataset_name

    - 修改 `Predictor` 类：

        ```python
        class Predictor:
            def __init__(self, cfg, dataset_name):
                self.predictor = DefaultPredictor(cfg)
                # self.coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
                self.mfnet_metadata = MetadataCatalog.get(dataset_name)
        
            def predict_single_image(self, image_path):
                im = cv2.imread(str(image_path))
                outputs = self.predictor(im)
                v = Visualizer(im[:, :, ::-1], self.mfnet_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
                semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
                return semantic_result
        
            def predict(self, test_dir, output_dir):
                test_dir = Path(test_dir)
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
        
                image_path = list(test_dir.glob("*.png"))  # 测试图像为.png格式
        
                for image in tqdm(image_path, desc="Processing images"):  # 测试图像为.png格式
                    result = self.predict_single_image(image)
                    out_path = output_dir / f"{image.stem}_sem_seg.png"
                    cv2.imwrite(str(out_path), result)
        
                return output_dir
        ```

    - 添加参数解析和程序入口：

        ```python
        def setup(args):
            """
            load config from file and command-line arguments
            """
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_maskformer2_config(cfg)
            cfg.merge_from_file(args.config_file)
            cfg.merge_from_list(args.opts)
            cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True   # 语义分割
            cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
            cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
            cfg.freeze()
        
            return cfg
        
        
        def get_parser():
            parser = argparse.ArgumentParser(description="Semantic Segmentation Predictor")
            parser.add_argument(
                "--config-file",
                type=str,
                required=True,
                help="Path to the config file."
            )
        
            parser.add_argument(
                "--test-dir",
                type=str,
                default="datasets/MFNetDataset/test",
                help="Directory containing test images."
            )
        
            parser.add_argument(
                "--output-dir",
                type=str,
                default="output/visualize",
                help="Directory to save the output results."
            )
        
            parser.add_argument(
                "--opts",
                help="Modify config options using the command-line 'KEY VALUE' pairs",
                default=[],
                nargs=argparse.REMAINDER,
            )
        
            return parser
        
        
        def main(args):
            cfg = setup(args)
        
            # 注册数据集
            register = Register()
            dataset_name = register.register_dataset(name="MFNetDataset_test")  # 数据集名称
        
            predictor = Predictor(cfg, dataset_name)
            output_dir = predictor.predict(args.test_dir, args.output_dir)
            print(f"[predict INFO] All semantic segmentation results saved to {output_dir}")
        
        
        if __name__ == "__main__":
            args = get_parser().parse_args()
            print("Command Line Args:", args)
        
            main(args)
        ```

2. 运行 `predict.py` 文件。

    ```bash
    CUDA_VISIBLE_DEVICES=2 python predict.py \
    --config-file output/cityscapes/maskformer2_R50_bs16_90k/config.yaml \           # 模型配置文件
    --test-dir datasets/MFNetDataset/test \                           	             # 测试集数据目录
    --output-dir output/cityscapes/maskformer2_R50_bs16_90k/visualize \              # 可视化结果目录
    --opts MODEL.WEIGHTS output/cityscapes/maskformer2_R50_bs16_90k/model_final.pth  # 模型权重
    ```

### 7. 模型微调

1. 修改 `mask2former/config.py` ，添加微调配置的方法：

    ```python
    def add_fine_tuning_config(cfg):
        """
        Add config for fine-tuning including 2 backbone: ResNet and SwinTransformer.
        """
    
        cfg.FINE_TUNING = CN()
        cfg.FINE_TUNING.ENABLE = False
        cfg.FINE_TUNING.FREEZE_AT = 0  # freeze at stage 0
    
        # backbone configs
        cfg.FINE_TUNING.BACKBONE = CN()
        cfg.FINE_TUNING.BACKBONE.NAME = "ResNet"  # or "Swin"
    
        # ResNet configs
        cfg.FINE_TUNING.RESNET = CN()
        cfg.FINE_TUNING.RESNET.STEM = False  # whether to freeze stem
        cfg.FINE_TUNING.RESNET.RES = []  # freeze res layers' list e.g. ["res2", "res3", "res4", "res5"]
    
        # SwinTransformer configs
        cfg.FINE_TUNING.SWIN = CN()
        cfg.FINE_TUNING.SWIN.PATCH_EMBED = False
        cfg.FINE_TUNING.SWIN.POS_DROP = False
        cfg.FINE_TUNING.SWIN.LAYERS = []  # freeze layers' list e.g. [0, 1, 2, 3]
        cfg.FINE_TUNING.SWIN.NORMS = []  # freeze norm layers' list e.g. ["norm0", "norm1", "norm2", "norm3"]
    
        # sem_seg_head configs
        cfg.FINE_TUNING.SEM_SEG_HEAD = CN()
        cfg.FINE_TUNING.SEM_SEG_HEAD.PIXEL_DECODER = False
        cfg.FINE_TUNING.SEM_SEG_HEAD.PREDICTOR = False
    ```

    

2. 修改 `mask2former/__init__.py` ，添加 `add_fine_tuning_config` 方法：

    ```python
    # config
    from .config import add_maskformer2_config
    from .config import add_fine_tuning_config  # 添加这一行
    ```

3. 修改 `train_net.py` 。

    - 添加 `add_fine_tuning_config` ：

        ```python
        # MaskFormer
        from mask2former import (
            COCOInstanceNewBaselineDatasetMapper,
            COCOPanopticNewBaselineDatasetMapper,
            InstanceSegEvaluator,
            MaskFormerInstanceDatasetMapper,
            MaskFormerPanopticDatasetMapper,
            MaskFormerSemanticDatasetMapper,
            SemanticSegmentorWithTTA,
            add_maskformer2_config,
            add_fine_tuning_config,  # 添加这一行
        )
        ```

    - 在 `Trainer` 类中重写父类 `train` 方法，打印全部参数和可训练的参数数量：

        ```python
        class Trainer(DefaultTrainer):
            """
            Extension of the Trainer class adapted to MaskFormer.
            """
        	def __init__(self, cfg):
                super().__init__(cfg)
        
            def print_param_counts(self):
                total_params = 0
                trainable_params = 0
                for name, param in self.model.named_parameters():
                    total_params += param.numel()
                    if param.requires_grad:
                        trainable_params += param.numel()
                print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
                logger = logging.getLogger("detectron2.trainer")
                logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
        
            def train(self):
                """
                Run training, and print parameter counts at the start.
                """
                # 打印参数信息
                self.print_param_counts()
        
                # 调用父类的 train 方法
                super().train()
        ```

    - 在 `Trainer` 类中添加固定参数方法 `freeze_params` ：

        ```python
        class Trainer(DefaultTrainer):
            # ......
            
            @staticmethod
            def freeze_params(cfg, model):
                """
                Freeze parts of the model based on the configuration.
        
                Args:
                - cfg: Configuration object specifying which parts to freeze.
                - model: The model whose parameters will be frozen.
                """
                if not cfg.FINE_TUNING.ENABLE:
                    return
        
                # default train all params
                for param in model.parameters():
                    param.requires_grad = True
        
                # Freeze backbone parameters
                if cfg.FINE_TUNING.BACKBONE.NAME == "ResNet":
                    if cfg.FINE_TUNING.RESNET.STEM:
                        for param in model.backbone.stem.parameters():
                            param.requires_grad = False
                    for res_layer in cfg.FINE_TUNING.RESNET.RES:
                        layer = getattr(model.backbone, res_layer, None)
                        if layer:
                            for param in layer.parameters():
                                param.requires_grad = False
        
                elif cfg.FINE_TUNING.BACKBONE.NAME == "Swin":
                    if cfg.FINE_TUNING.SWIN.PATCH_EMBED:
                        for param in model.backbone.patch_embed.parameters():
                            param.requires_grad = False
                    for layer_idx in cfg.FINE_TUNING.SWIN.LAYERS:
                        layer = model.backbone.layers[layer_idx]
                        for param in layer.parameters():
                            param.requires_grad = False
                    for norm_idx in cfg.FINE_TUNING.SWIN.NORMS:
                        norm_layer = getattr(model.backbone, f"norm{norm_idx}", None)
                        if norm_layer:
                            for param in norm_layer.parameters():
                                param.requires_grad = False
        
                # Freeze semantic segmentation head parameters
                if cfg.FINE_TUNING.SEM_SEG_HEAD.PIXEL_DECODER:
                    for param in model.sem_seg_head.pixel_decoder.parameters():
                        param.requires_grad = False
                if cfg.FINE_TUNING.SEM_SEG_HEAD.PREDICTOR:
                    for param in model.sem_seg_head.predictor.parameters():
                        param.requires_grad = False
        ```

        

    - 在 `Trainer` 类的 `build_optimizer` 方法中添加参数固定逻辑：

        ```python
        class Trainer(DefaultTrainer):
            # ......
            
            def build_optimizer(cls, cfg, model):
                # ......
                
                cls.freeze_params(cfg, model)  # freeze params before optimizer building
                
            	params: List[Dict[str, Any]] = []
                memo: Set[torch.nn.parameter.Parameter] = set()
        ```

        

### 8. 实验结果

#### 实验一：全参数训练

##### 实验 1：

使用 `maskformer2_cityscapes_semantic_R50_bs16_90k` 预训练模型。


##### 实验 2：

使用 `maskformer2_cityscapes_semantic_R101_bs16_90k` 预训练模型。

##### 实验 3：

使用 `maskformer2_cityscapes_semantic_swin_tiny_bs16_90k` 预训练模型。

##### 实验 4：

使用 `maskformer2_cityscapes_semantic_swin_small_bs16_90k` 预训练模型。

##### 实验 5：

使用 `maskformer2_cityscapes_semantic_swin_base_IN21k_384_bs16_90k` 预训练模型。

##### 实验 6：

使用 `maskformer2_cityscapes_semantic_swin_large_IN21k_384_bs16_90k` 预训练模型。

#### 实验二：微调模型

##### 实验 1：

在 `maskformer2_cityscapes_semantic_R50_bs16_90k` 预训练模型上微调，固定 `backbone`只训练 `sem_seg_head`。

##### 实验 2：

在 `maskformer2_cityscapes_semantic_R101_bs16_90k` 预训练模型上微调，固定 `backbone`只训练 `sem_seg_head`。

##### 实验 3：

在 `maskformer2_cityscapes_semantic_swin_tiny_bs16_90k` 预训练模型上微调，固定 `backbone`只训练 `sem_seg_head`。

##### 实验 4：

在 `maskformer2_cityscapes_semantic_swin_small_bs16_90k` 预训练模型上微调，固定 `backbone`只训练 `sem_seg_head`。

##### 实验 5：

在`maskformer2_cityscapes_semantic_swin_base_IN21k_384_bs16_90k` 预训练模型上微调，固定 `backbone`只训练 `sem_seg_head`。

##### 实验 6：

在`maskformer2_cityscapes_semantic_swin_large_IN21k_384_bs16_90k` 预训练模型上微调，固定 `backbone`只训练 `sem_seg_head`。

#### 全参数评估结果汇总

|                     Method                      |   Modal   | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :---------------------------------------------: | :-------: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
|         maskformer2_R50_bs8_20k_lr0001          |    RGB    | 55.82 |   97.46   | 84.06 | 58.76  | 71.99 | 34.47 |  49.86   |    0.00    |   60.85    | 44.97 |
|         maskformer2_R101_bs8_20k_lr0001         |    RGB    | 56.40 |   97.54   | 84.75 | 58.63  | 72.60 | 34.06 |  51.36   |    0.00    |   60.81    | 47.84 |
|      maskformer2_swin_tiny_bs8_20k_lr0001       |    RGB    | 56.54 |   97.44   | 82.54 | 57.47  | 71.96 | 33.91 |  45.39   |    0.00    |   60.28    | 59.90 |
|      maskformer2_swin_tiny_bs8_20k_lr0001       | **RGB-T** | 56.83 |   97.58   | 84.16 | 70.38  | 72.3  | 34.42 |  47.84   |    0.00    |   52.31    | 52.50 |
|      maskformer2_swin_small_bs8_20k_lr0001      |    RGB    | 58.79 |   97.64   | 85.21 | 59.26  | 73.02 | 36.97 |  48.25   |    0.00    |   60.08    | 68.70 |
|      maskformer2_swin_small_bs4_40k_lr0001      | **RGB-T** | 58.98 |   97.76   | 87.23 | 71.79  | 73.20 | 36.74 |  47.16   |    0.00    |   53.19    | 63.78 |
| maskformer2_swin_base_IN21k_384_bs4_40k_lr0001  |    RGB    | 58.49 |   97.77   | 87.17 | 59.09  | 72.48 | 33.72 |  45.15   |    0.00    |   61.91    | 69.13 |
| maskformer2_swin_base_IN21k_384_bs2_40k_lr0001  | **RGB-T** | 62.39 |   97.97   | 88.55 | 71.73  | 73.75 | 38.05 |  51.37   |    0.00    |   66.25    | 73.79 |
| maskformer2_swin_large_IN21k_384_bs4_40k_lr0001 |    RGB    | 58.51 |   97.78   | 86.35 | 59.59  | 73.05 | 36.96 |  46.76   |    0.00    |   65.12    | 60.97 |
| maskformer2_swin_large_IN21k_384_bs2_40k_lr0001 | **RGB-T** | 62.24 |   97.99   | 88.26 | 72.27  | 73.68 | 39.65 |  47.06   |    0.00    |   68.11    | 73.18 |

#### 微调评估结果汇总

固定 `backbone` 和 `sem_seg_head.pixel_decoder` ，只训练 `predictor`：**模型无法拟合！**

固定 `backbone`，只训练 `sem_seg_head`：

|                     Method                      |   Modal   | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :---------------------------------------------: | :-------: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
|         maskformer2_R50_bs4_40k_lr0001          |    RGB    | 53.78 |   97.25   | 81.73 | 57.27  | 71.04 | 29.91 |  43.33   |    0.00    |   60.98    | 42.53 |
|         maskformer2_R101_bs4_40k_lr0001         |    RGB    | 56.18 |   97.26   | 80.79 | 55.42  | 70.54 | 30.28 |  49.31   |    0.00    |   60.65    | 61.39 |
|      maskformer2_swin_tiny_bs4_40k_lr0001       |    RGB    | 54.07 |   97.22   | 80.30 | 56.99  | 71.45 | 32.43 |  41.77   |    0.00    |   58.27    | 48.16 |
|      maskformer2_swin_tiny_bs8_20k_lr0001       | **RGB-T** | 56.84 |   97.60   | 81.90 | 71.15  | 69.16 | 33.43 |  40.54   |    0.00    |   54.49    | 63.29 |
|      maskformer2_swin_small_bs4_40k_lr0001      |    RGB    | 55.92 |   97.41   | 83.82 | 57.25  | 71.59 | 29.53 |  41.18   |    0.00    |   62.18    | 60.34 |
|      maskformer2_swin_small_bs8_20k_lr0001      | **RGB-T** | 58.81 |   97.74   | 86.18 | 71.62  | 72.94 | 34.29 |  48.23   |    0.00    |   61.39    | 56.94 |
| maskformer2_swin_base_IN21k_384_bs4_40k_lr0001  |    RGB    | 58.60 |   97.63   | 86.01 | 59.49  | 71.59 | 31.31 |  53.00   |    0.00    |   62.03    | 66.83 |
| maskformer2_swin_base_IN21k_384_bs8_20k_lr0001  | **RGB-T** | 61.07 |   97.81   | 86.82 | 71.47  | 72.51 | 37.89 |  48.44   |    0.00    |   63.49    | 71.16 |
| maskformer2_swin_large_IN21k_384_bs4_40k_lr0001 |    RGB    | 60.10 |   97.71   | 86.00 | 59.66  | 71.96 | 37.69 |  54.97   |    0.00    |   63.72    | 69.21 |
| maskformer2_swin_large_IN21k_384_bs8_20k_lr0001 | **RGB-T** | 60.80 |   97.85   | 86.88 | 71.41  | 73.39 | 36.09 |  51.51   |    0.00    |   63.58    | 66.49 |

#### Adapter实验结果汇总

#### 使用 cityscapes 预训练参数

##### adapter

固定 `backbone`，固定 `pixel_decoder`，只训练其中的 `Adapter` （只在每个层的 `FFN` 后串联）：

|                         Method                         | Modal |   mIoU    | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :----------------------------------------------------: | :---: | :-------: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
|       maskformer2_swin_tiny_bs8_20k_lr0001_mid32       | RGB-T |   53.31   |   97.38   | 85.43 | 71.04  | 71.87 | 36.11 |  32.33   |    0.00    |   58.80    | 26.84 |
|       maskformer2_swin_tiny_bs8_20k_lr0001_mid64       | RGB-T | **56.20** |   97.53   | 84.92 | 70.57  | 72.11 | 34.07 |  43.17   |    0.00    |   59.62    | 43.84 |
|      maskformer2_swin_tiny_bs8_20k_lr0001_mid128       | RGB-T |   56.08   |   97.53   | 85.99 | 71.32  | 72.70 | 35.61 |  39.92   |    0.00    |   59.91    | 41.78 |
|      maskformer2_swin_small_bs8_20k_lr0001_mid32       | RGB-T |   56.08   |   97.54   | 85.55 | 70.87  | 71.89 | 38.12 |  44.62   |    0.00    |   62.67    | 33.45 |
|      maskformer2_swin_small_bs8_20k_lr0001_mid64       | RGB-T | **59.26** |   97.72   | 86.99 | 70.82  | 72.48 | 42.38 |  47.77   |    0.00    |   62.96    | 52.19 |
|      maskformer2_swin_small_bs8_20k_lr0001_mid128      | RGB-T |   57.25   |   97.58   | 85.63 | 70.72  | 72.56 | 36.86 |  46.30   |    0.00    |   61.26    | 44.31 |
|  maskformer2_swin_base_IN21k_384_bs8_20k_lr0001_mid32  | RGB-T |   56.99   |   97.71   | 87.89 | 71.52  | 72.98 | 42.89 |  45.58   |    0.00    |   56.84    | 37.54 |
|  maskformer2_swin_base_IN21k_384_bs8_20k_lr0001_mid64  | RGB-T | **59.42** |   97.78   | 87.57 | 71.69  | 72.24 | 45.58 |  49.23   |    0.00    |   57.93    | 52.75 |
| maskformer2_swin_base_IN21k_384_bs8_20k_lr0001_mid128  | RGB-T |   59.49   |   97.81   | 88.27 | 71.68  | 72.82 | 43.28 |  51.74   |    0.00    |   62.50    | 47.34 |
| maskformer2_swin_large_IN21k_384_bs8_20k_lr0001_mid32  | RGB-T |   59.11   |   97.76   | 87.40 | 71.81  | 72.06 | 46.79 |  45.78   |    0.00    |   61.87    | 39.57 |
| maskformer2_swin_large_IN21k_384_bs8_20k_lr0001_mid64  | RGB-T | **58.21** |   97.69   | 88.20 | 72.10  | 73.16 | 44.14 |  44.98   |    0.00    |   63.99    | 35.46 |
| maskformer2_swin_large_IN21k_384_bs8_20k_lr0001_mid128 | RGB-T |   59.07   |   97.76   | 87.66 | 71.58  | 72.92 | 40.44 |  48.76   |    0.00    |   65.91    | 46.63 |

##### adapter1

去掉 residual connection ，并在下投影增加 ReLU 激活，每个 `encoder` 层并联一整个 CrossModalAdapter（adapter1）：

|      Method       | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :---------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr0001 | val  | 60.76 |   97.81   | 87.26 | 71.36  | 73.41 | 43.22 |  48.33   |    0.00    |   63.99    | 61.49 |

~~将 Adapter 的输入改为位置嵌入后的输入（adapter1-1）~~：

|      Method       | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :---------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr0001 | val  | 60.02 |   97.80   | 87.92 | 72.20  | 71.55 | 44.80 |  46.57   |    0.00    |   62.42    | 56.94 |

去掉 Adapter 中间的线性层（adapter1-2）： 

|      Method       | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :---------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr0001 | val  | 58.31 |   97.75   | 88.09 | 72.14  | 73.21 | 39.27 |  51.55   |    0.00    |   64.78    | 38.05 |

增加 dropout=0.1（adapter1-3）：

|      Method       | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :---------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr0001 | val  | 61.25 |   97.79   | 88.26 | 71.32  | 73.26 | 41.68 |  50.14   |    0.00    |   64.35    | 64.47 |

##### adapter2

在 `pixel_decoder` 的 `encoder` 层的 attention 和 FFN 都并联 CrossModalAdapter（adapter2）：

|      Method       | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :---------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr0001 | val  | 60.58 |   97.81   | 87.74 | 71.76  | 72.85 | 42.87 |  49.54   |    0.00    |   62.95    | 59.69 |

##### adapter3

在 `pixel_decoder` 的 `encoder` 层并联一整个 CrossModalAdapter，在 `predictor` 的 `decoder` 层的 FFN 后串联 SelfAdapter（adapter3），调整 `lr=0.001 `：

|      Method      | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :--------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr001 | val  | 60.97 |   97.96   | 88.46 | 71.63  | 73.59 | 43.56 |  48.54   |    0.00    |   63.83    | 61.13 |
|                  | test | 57.82 |   98.19   | 89.88 | 75.82  | 64.89 | 46.51 |  41.62   |    0.00    |   52.06    | 51.39 |

余弦衰减，FFN 后串联**共享**的 SelfAdapter（adapter3-1）：

|       Method        | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :-----------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_coslr001 | val  | 62.14 |   97.93   | 87.65 | 72.21  | 73.34 | 42.71 |  54.47   |    0.00    |   64.08    | 66.90 |
|                     | test | 56.74 |   98.24   | 89.82 | 75.59  | 64.37 | 45.33 |  33.03   |    0.00    |   55.43    | 48.81 |

##### adapter4

在 `pixel_decoder` 的 `encoder` 层并联一整个 CrossModalAdapter，在 `predictor` 的 `decoder` 层的 MHSA 和 FFN 后各串联一个 SelfAdapter（adapter4），调整 `lr=0.005`：

|      Method      | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :--------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr005 | val  | 59.79 |   97.92   | 88.39 | 72.15  | 72.99 | 45.63 |  47.73   |    0.00    |   65.52    | 47.78 |
|                  | test | 56.37 |   98.27   | 89.96 | 76.24  | 66.55 | 47.52 |  31.86   |    0.00    |   53.62    | 43.31 |

余弦衰减，MHSA 和 FFN 后串联**共享**的 SelfAdapter（adapter4-1）：

|       Method        | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :-----------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | ----- |
| swin_large_coslr001 | val  | 62.32 |   97.90   | 88.12 | 72.26  | 73.64 | 39.92 |  54.52   |    0.00    |   65.21    | 69.28 |
|                     | test | 57.51 |   98.24   | 89.33 | 75.50  | 65.31 | 46.13 |  32.74   |    0.00    |   56.38    | 53.96 |

##### adapter5

在 `pixel_decoder` 的 `encoder` 层并联一整个 CrossModalAdapter，在 `predictor` 的 `decoder` 层并联一整个 SelfAdapter（adapter5），调整 `lr=0.005`：

|      Method      | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :--------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr005 | val  | 56.33 |   97.91   | 89.04 | 72.38  | 73.84 | 44.78 |  38.79   |    0.00    |   66.10    | 24.12 |

在 `pixel_decoder` 的 `encoder` 层并联一整个 CrossModalAdapter，在 `predictor` 的 `decoder` 层并联一整个**共享**的 SelfAdapter（adapter5-1），调整 `lr=0.001`：

|      Method      | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :--------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr001 | val  | 62.14 |   97.93   | 88.20 | 71.89  | 73.56 | 42.68 |  54.27   |    0.00    |   66.10    | 64.57 |
|                  | test | 58.59 |   98.27   | 89.86 | 75.60  | 65.32 | 47.58 |  43.84   |    0.00    |   52.36    | 54.53 |

学习率使用余弦衰减 `WarmupCosineLR`：

|       Method        | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :-----------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_coslr001 | val  | 61.80 |   97.95   | 88.43 | 72.17  | 73.50 | 41.83 |  53.74   |    0.00    |   64.95    | 63.61 |
| swin_large_coslr005 | val  | 60.78 |   97.91   | 88.29 | 71.50  | 72.99 | 39.73 |  51.02   |    0.00    |   65.15    | 60.38 |
| swin_large_coslr001 | test | 57.49 |   98.18   | 89.29 | 75.19  | 64.46 | 44.90 |  42.42   |    0.00    |   52.53    | 50.49 |

##### moe1：

`pixel decoder` 的 `encoder` 层共享专家（adapters），并联于 FFN，每层每个模态都有特定的路由（router），`ADAPTER_MID_DIM: 64`，`NUM_EXPERTS: 4`，`TOP_K: 2`，`PATCHES: 1`。

|            Method             | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :---------------------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr001_moe-64-4-2-1 | val  | 60.78 |   97.92   | 87.63 | 71.38  | 72.91 | 38.07 |  49.78   |    0.00    |   65.75    | 63.60 |
|                               | test | 57.59 |   98.26   | 89.46 | 75.06  | 66.87 | 47.41 |  39.25   |    0.00    |   53.99    | 47.97 |

##### moe4：

更改 patches 的值。

|             Method             | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :----------------------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr001_moe-64-4-2-16 | val  | 53.59 |   97.71   | 86.71 | 70.85  | 72.68 | 39.03 |  37.28   |    0.00    |   24.42    | 53.60 |
|                                | test | 53.72 |   98.17   | 89.02 | 74.27  | 64.79 | 43.69 |  37.48   |    0.00    |   44.05    | 32.00 |



---

#### **使用 coco 预训练参数**

##### adapter5-1:

|      Method      | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :--------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr001 | val  | 62.75 |   97.98   | 88.53 | 72.92  | 73.90 | 43.48 |  53.65   |    0.00    |   63.37    | 70.93 |
| swin_large_lr001 | test | 59.16 |   98.34   | 90.82 | 76.08  | 67.35 | 47.14 |  44.86   |    0.00    |   50.83    | 57.05 |

##### moe1：

初始设置。

|             Method              | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone |  Bump  |
| :-----------------------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :----: |
|  swin_large_lr001_moe-64-4-2-1  | val  | 61.62 |   97.97   | 87.03 | 72.31  | 73.44 | 38.83 |  57.00   |    0.00    |   66.06    | 61.92  |
|                                 | test | 58.41 |   98.33   | 89.48 | 76.05  | 65.53 | 46.81 |  40.89   |    0.00    |   52.68    | 55.940 |
| swin_large_lr001_moe-64-12-12-1 | val  | 61.03 |   97.92   | 87.03 | 72.01  | 73.61 | 43.57 |  51.48   |    0.00    |   67.35    | 56.31  |
|                                 | test | 59.22 |   98.33   | 89.98 | 75.59  | 66.09 | 46.21 |  41.06   |    0.00    |   54.16    | 61.59  |

##### moe2：

更改 adapter 的维度为128, 32。

|             Method             | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :----------------------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr001_moe-128-4-2-1 | val  | 62.33 |   97.85   | 86.84 | 72.07  | 73.41 | 41.84 |  58.64   |    0.00    |   63.24    | 67.08 |
|                                | test | 58.95 |   98.23   | 89.64 | 75.33  | 69.17 | 45.74 |  50.15   |    0.00    |   50.25    | 52.07 |
| swin_large_lr001_moe-32-4-2-1  | val  | 61.80 |   97.90   | 86.73 | 71.99  | 73.59 | 40.63 |  55.58   |    0.00    |   65.62    | 64.15 |
|                                | test | 56.97 |   98.29   | 89.46 | 76.05  | 66.82 | 42.86 |  38.93   |    0.00    |   53.59    | 46.73 |

##### moe3：

更改专家数量为6, 8, 10, 12，topk 的值为2, 3, 4, 8。

|             Method              | Mode |   mIoU    | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :-----------------------------: | :--: | :-------: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
|  swin_large_lr001_moe-64-6-2-1  | val  |   63.17   |   97.95   | 86.93 | 72.05  | 73.40 | 44.70 |  58.99   |    0.00    |   65.23    | 69.24 |
|                                 | test |   59.38   |   98.31   | 90.12 | 75.99  | 67.51 | 45.64 |  41.53   |    0.00    |   55.02    | 60.27 |
|  swin_large_lr001_moe-64-6-3-1  | val  |   61.98   |   97.87   | 86.98 | 71.68  | 73.74 | 37.48 |  56.58   |    0.00    |   65.83    | 67.70 |
|                                 | test |   58.54   |   98.29   | 89.93 | 76.11  | 68.88 | 44.58 |  45.05   |    0.00    |   57.08    | 46.90 |
|  swin_large_lr001_moe-64-6-4-1  | val  |   61.40   |   97.91   | 86.82 | 71.99  | 73.39 | 42.99 |  54.06   |    0.00    |   66.47    | 58.99 |
|                                 | test |   57.60   |   98.31   | 90.24 | 75.86  | 67.66 | 47.27 |  34.62   |    0.00    |   57.09    | 47.34 |
|  swin_large_lr001_moe-64-8-2-1  | val  |   61.78   |   97.96   | 86.87 | 72.69  | 73.44 | 42.46 |  54.28   |    0.00    |   65.81    | 62.52 |
|                                 | test |   59.08   |   98.38   | 89.82 | 76.12  | 67.95 | 46.37 |  42.26   |    0.00    |   58.13    | 52.66 |
|  swin_large_lr001_moe-64-8-4-1  | val  |   60.05   |   97.92   | 86.33 | 72.34  | 74.02 | 39.54 |  51.74   |    0.00    |   66.22    | 52.36 |
|                                 | test |   58.32   |   98.36   | 90.19 | 75.89  | 68.04 | 45.69 |  43.51   |    0.00    |   54.84    | 48.36 |
|  swin_large_lr001_moe-64-8-6-1  | val  |   63.18   |   97.95   | 87.04 | 71.43  | 73.72 | 43.50 |  58.78   |    0.00    |   66.38    | 66.38 |
|                                 | test |   59.91   |   98.31   | 89.63 | 75.66  | 68.83 | 46.84 |  51.03   |    0.00    |   54.04    | 54.87 |
| swin_large_lr001_moe-64-10-2-1  | val  |   61.49   |   97.86   | 86.38 | 72.08  | 73.45 | 42.89 |  53.19   |    0.00    |   64.74    | 62.84 |
|                                 | test |   58.57   |   98.23   | 89.46 | 75.70  | 68.44 | 45.47 |  45.36   |    0.00    |   49.17    | 55.33 |
| swin_large_lr001_moe-64-10-4-1  | val  |   60.53   |   97.90   | 86.83 | 72.30  | 73.56 | 42.89 |  50.38   |    0.00    |   66.22    | 54.73 |
|                                 | test |   59.04   |   98.31   | 89.90 | 76.00  | 66.95 | 47.47 |  41.86   |    0.00    |   53.05    | 57.86 |
| swin_large_lr001_moe-64-10-6-1  | val  |   61.90   |   97.96   | 86.85 | 72.26  | 73.38 | 43.71 |  54.43   |    0.00    |   65.95    | 62.58 |
|                                 | test |   58.99   |   98.30   | 90.10 | 75.63  | 67.34 | 46.31 |  46.84   |    0.00    |   54.33    | 52.11 |
| swin_large_lr001_moe-64-10-8-1  | val  |   63.02   |   97.97   | 86.98 | 72.21  | 74.08 | 43.87 |  58.43   |    0.00    |   65.91    | 67.72 |
|                                 | test | **60.76** |   98.36   | 89.91 | 75.75  | 68.71 | 48.06 |  48.08   |    0.00    |   54.37    | 63.63 |
| swin_large_lr001_moe-64-12-2-1  | val  |   62.95   |   97.89   | 86.87 | 72.13  | 73.53 | 41.18 |  60.79   |    0.00    |   67.52    | 66.64 |
|                                 | test | **60.72** |   98.32   | 89.67 | 75.55  | 68.89 | 46.86 |  50.39   |    0.00    |   54.36    | 62.48 |
| swin_large_lr001_moe-64-12-4-1  | val  |   63.65   |   97.95   | 86.59 | 71.83  | 73.49 | 44.59 |  60.99   |    0.00    |   65.73    | 71.70 |
|                                 | test | **60.87** |   98.34   | 90.01 | 75.58  | 68.04 | 44.77 |  49.22   |    0.00    |   55.50    | 66.33 |
| swin_large_lr001_moe-64-12-6-1  | val  |   63.76   |   97.98   | 87.15 | 71.90  | 73.77 | 43.76 |  61.18   |    0.00    |   67.47    | 70.62 |
|                                 | test | **60.46** |   98.33   | 90.16 | 75.34  | 67.55 | 46.14 |  52.42   |    0.00    |   53.91    | 60.32 |
| swin_large_lr001_moe-64-12-8-1  | val  |   60.58   |   97.87   | 87.25 | 71.91  | 73.43 | 41.09 |  54.14   |    0.00    |   66.34    | 53.24 |
|                                 | test |   59.82   |   98.36   | 90.35 | 76.06  | 69.03 | 46.75 |  49.59   |    0.00    |   54.36    | 53.92 |
| swin_large_lr001_moe-64-12-10-1 | val  |   62.44   |   97.97   | 87.31 | 71.53  | 74.03 | 42.79 |  55.34   |    0.00    |   65.39    | 67.61 |
|                                 | test |   58.86   |   98.29   | 90.30 | 75.02  | 67.11 | 46.27 |  42.59   |    0.00    |   54.75    | 55.43 |

##### moe4：

更改 patches 的值为4, 9, 16, 25。

|                 Method                  | Mode |   mIoU    | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :-------------------------------------: | :--: | :-------: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
|      swin_large_lr001_moe-64-4-2-4      | val  |   63.22   |   97.95   | 87.15 | 72.22  | 74.08 | 41.33 |  59.61   |    0.00    |   67.55    | 69.10 |
|                                         | test |   59.62   |   98.30   | 89.36 | 75.67  | 66.21 | 46.36 |  44.03   |    0.00    |   57.48    | 59.13 |
|      swin_large_lr001_moe-64-4-2-9      | val  |   62.14   |   97.86   | 86.79 | 71.62  | 73.60 | 38.87 |  56.90   |    0.00    |   66.82    | 66.80 |
|                                         | test |   58.51   |   98.30   | 89.93 | 75.99  | 67.80 | 46.23 |  42.33   |    0.00    |   55.81    | 50.19 |
| swin_large_lr001_moe-64-4-2-16(bs4_20k) | val  |   62.64   |   97.90   | 86.96 | 71.84  | 73.42 | 42.65 |  57.38   |    0.00    |   67.23    | 66.34 |
|                                         | test | **60.11** |   98.36   | 90.48 | 75.99  | 69.27 | 47.18 |  51.15   |    0.00    |   52.86    | 55.70 |
| swin_large_lr001_moe-64-4-2-16_newloss  | val  |   63.28   |   97.97   | 87.15 | 71.88  | 73.82 | 43.55 |  58.76   |    0.00    |   65.68    | 70.74 |
|                                         | test |   59.14   |   98.34   | 89.50 | 75.79  | 68.89 | 46.88 |  40.85   |    4.60    |   52.01    | 55.39 |
| swin_large_lr001_moe-64-4-2-16(bs8_20k) | val  |   63.24   |   97.96   | 87.27 | 72.06  | 73.50 | 39.26 |  62.68   |    0.00    |   66.17    | 70.24 |
|                                         | test | **60.24** |   98.35   | 90.24 | 76.25  | 67.92 | 45.19 |  51.67   |    0.00    |   53.95    | 58.56 |
|     swin_large_lr001_moe-64-4-2-25      | val  |   62.78   |   97.95   | 86.50 | 71.77  | 74.04 | 41.62 |  58.59   |    0.00    |   66.70    | 67.86 |
|                                         | test |   57.99   |   98.30   | 90.01 | 75.84  | 66.88 | 47.50 |  40.00   |    0.00    |   54.64    | 48.75 |

##### moe5：

64-4-2-1 调优。

|              Method              | Mode |   mIoU    | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :------------------------------: | :--: | :-------: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr001_moe-128-4-2-16  | val  |   62.83   |   97.93   | 87.06 | 71.62  | 73.48 | 42.58 |  60.17   |    0.00    |   65.91    | 66.74 |
|                                  | test |   59.51   |   98.31   | 89.69 | 76.15  | 68.57 | 48.07 |  50.57   |    0.00    |   56.05    | 48.18 |
| swin_large_lr001_moe-128-6-2-16  | val  |   62.44   |   97.93   | 86.72 | 71.88  | 73.94 | 39.60 |  59.66   |    0.00    |   66.76    | 65.48 |
|                                  | test |   59.77   |   98.34   | 89.81 | 76.26  | 67.87 | 38.88 |  38.88   |    0.00    |   53.66    | 68.15 |
| swin_large_lr001_moe-128-6-2-25  | val  |   62.08   |   97.87   | 86.51 | 71.85  | 73.95 | 40.65 |  57.32   |    0.00    |   67.77    | 62.78 |
|                                  | test |   58.56   |   98.29   | 89.67 | 76.15  | 66.39 | 44.89 |  40.67   |    0.00    |   54.17    | 56.78 |
| swin_large_lr001_moe-128-12-2-16 | val  |   62.29   |   97.92   | 86.75 | 71.64  | 73.47 | 40.90 |  59.03   |    0.00    |   66.87    | 64.07 |
|                                  | test |   56.49   |   98.25   | 90.04 | 75.80  | 67.37 | 48.18 |  42.14   |    0.00    |   58.10    | 28.51 |
| swin_large_lr001_moe-128-12-4-16 | val  |   62.94   |   97.96   | 87.25 | 72.37  | 74.15 | 41.60 |  58.54   |    0.00    |   68.37    | 66.17 |
|                                  | test |   58.13   |   98.32   | 90.14 | 75.84  | 68.47 | 48.14 |  38.35   |    0.00    |   56.57    | 47.34 |
| swin_large_lr001_moe-128-12-6-16 | val  |   63.43   |   97.96   | 86.65 | 71.58  | 74.12 | 42.90 |  62.49   |    0.00    |   65.97    | 69.21 |
|                                  | test | **60.56** |   98.37   | 90.15 | 75.86  | 69.18 | 47.91 |  52.41   |    0.00    |   53.59    | 57.60 |
|  swin_large_lr001_moe-64-6-2-16  | val  |   63.79   |   97.99   | 86.89 | 72.36  | 74.09 | 43.24 |  60.35   |    0.00    |   67.06    | 72.12 |
|                                  | test |   59.03   |   98.36   | 90.43 | 76.09  | 65.37 | 46.96 |  39.64   |    0.00    |   54.02    | 60.37 |
|  swin_large_lr001_moe-64-8-2-16  | val  |   62.88   |   97.95   | 86.83 | 71.74  | 73.25 | 44.03 |  57.98   |    0.00    |   66.34    | 67.79 |
|                                  | test |   57.95   |   98.28   | 90.04 | 75.94  | 67.72 | 46.53 |  43.98   |    0.00    |   53.96    | 45.11 |
| swin_large_lr001_moe-64-10-2-16  | val  |   60.86   |   97.93   | 87.12 | 71.96  | 73.32 | 39.95 |  52.12   |    0.00    |   67.46    | 57.88 |
|                                  | test |   58.09   |   98.29   | 89.47 | 76.13  | 66.26 | 47.20 |  37.01   |    0.00    |   53.92    | 54.51 |
| swin_large_lr001_moe-64-12-2-16  | val  |   62.47   |   97.95   | 87.10 | 72.03  | 73.86 | 39.56 |  55.32   |    0.00    |   66.05    | 70.37 |
|                                  | test |   58.97   |   98.35   | 89.83 | 76.24  | 65.65 | 45.87 |  35.79   |    0.00    |   57.70    | 61.30 |
| swin_large_lr001_moe-64-12-4-16  | val  |   62.88   |   97.92   | 86.94 | 71.85  | 73.57 | 40.15 |  59.29   |    0.00    |   66.72    | 69.48 |
|                                  | test | **61.45** |   98.36   | 89.91 | 76.09  | 69.07 | 47.17 |  52.30   |    0.00    |   55.75    | 64.38 |
| swin_large_lr001_moe-64-12-6-16  | val  |   63.51   |   97.96   | 86.64 | 72.28  | 73.72 | 41.12 |  61.37   |    0.00    |   66.84    | 71.66 |
|                                  | test |   59.93   |   98.34   | 90.08 | 75.55  | 68.88 | 46.46 |  49.18   |    0.00    |   53.91    | 56.96 |

 修改 `Guardrail` 类损失权重。

|                 Method                  | Mode |   mIoU    | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :-------------------------------------: | :--: | :-------: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
|  swin_large_lr001_moe-64-12-4-16(base)  | test | **61.45** |   98.36   | 89.91 | 76.09  | 69.07 | 47.17 |  52.30   |    0.00    |   55.75    | 64.38 |
| swin_large_lr001_moe-64-12-4-16_loss10  | test |   59.00   |   98.29   | 89.45 | 75.67  | 68.26 | 45.72 |  53.57   |    0.20    |   52.90    | 47.00 |
| swin_large_lr001_moe-64-12-4-16_loss20  | test |   57.87   |   98.27   | 89.66 | 75.69  | 68.28 | 46.89 |  37.88   |    0.00    |   53.37    | 50.78 |
| swin_large_lr001_moe-64-12-4-16_loss30  | test | **60.09** |   98.30   | 90.02 | 76.14  | 68.37 | 47.22 |  46.65   |    4.87    |   53.66    | 55.62 |
| swin_large_lr001_moe-64-12-4-16_loss40  | test |   58.80   |   98.32   | 90.06 | 76.18  | 67.26 | 47.11 |  40.14   |    0.00    |   56.11    | 54.01 |
| swin_large_lr001_moe-64-12-4-16_loss50  | test |   57.80   |   98.27   | 89.12 | 75.74  | 67.64 | 45.46 |  46.63   |    1.70    |   52.37    | 43.25 |
| swin_large_lr001_moe-64-12-4-16_loss60  | test |   60.20   |   98.38   | 90.36 | 76.15  | 67.27 | 46.98 |  44.25   |    0.00    |   57.47    | 60.97 |
| swin_large_lr001_moe-64-12-4-16_loss70  | test |   58.02   |   98.34   | 90.09 | 75.81  | 69.79 | 46.57 |  27.82   |    4.09    |   56.33    | 53.34 |
| swin_large_lr001_moe-64-12-4-16_loss80  | test |   58.81   |   98.29   | 89.75 | 75.85  | 69.07 | 45.35 |  34.16   |    5.54    |   55.01    | 56.24 |
| swin_large_lr001_moe-64-12-4-16_loss100 | test |   56.16   |   98.36   | 89.84 | 76.17  | 69.31 | 46.65 |  34.07   |    2.44    |   56.99    | 31.59 |

**参考标准：**

|                 Method                  | Mode |   mIoU    | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :-------------------------------------: | :--: | :-------: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr001_moe-64-4-2-16(bs4_20k) | val  |   62.64   |   97.90   | 86.96 | 71.84  | 73.42 | 42.65 |  57.38   |    0.00    |   67.23    | 66.34 |
|                                         | test | **60.11** |   98.36   | 90.48 | 75.99  | 69.27 | 47.18 |  51.15   |    0.00    |   52.86    | 55.70 |

##### moe6-1：

Adapter 增加中间的线性层。

|             Method             | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :----------------------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr001_moe-64-4-2-16 | val  | 63.84 |   97.96   | 87.09 | 72.41  | 73.85 | 45.02 |  60.31   |    0.00    |   67.13    | 70.77 |
|                                | test | 59.68 |   98.32   | 89.84 | 76.12  | 69.22 | 47.58 |  49.99   |    0.00    |   55.23    | 50.83 |

##### moe6-2：

给路由后的结果添加 noise 噪声。

|             Method             | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :----------------------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr001_moe-64-4-2-16 | val  | 62.75 |   97.92   | 86.63 | 71.58  | 73.33 | 42.70 |  57.42   |    0.00    |   65.77    | 69.41 |
|                                | test | 58.97 |   98.27   | 89.69 | 75.64  | 69.30 | 43.92 |  48.19   |    0.00    |   53.86    | 51.88 |

##### moe6-3：

路由前加入可学习的 query。

|                 Method                 | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :------------------------------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
|     swin_large_lr001_moe-64-4-2-16     | val  | 62.62 |   97.96   | 87.00 | 71.81  | 73.84 | 41.40 |  55.20   |    0.00    |   67.25    | 69.10 |
|                                        | test | 59.14 |   98.30   | 90.02 | 76.26  | 67.95 | 44.48 |  48.44   |    0.00    |   54.04    | 52.75 |
| swin_large_lr001_moe-64-4-2-16(shared) | val  | 63.75 |   97.98   | 86.94 | 72.07  | 73.83 | 42.00 |  61.28   |    0.00    |   66.43    | 73.18 |
|                                        | test | 60.03 |   98.30   | 89.56 | 75.87  | 67.61 | 47.10 |  50.73   |    0.00    |   53.34    | 57.79 |

##### moe7：

选取一个特征图只计算一次路由。

|             Method             | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :----------------------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr001_moe-64-4-2-16 | val  | 61.80 |   97.92   | 86.70 | 72.14  | 73.29 | 42.38 |  55.66   |    0.00    |   64.92    | 63.23 |
|                                | test | 58.23 |   98.32   | 89.61 | 75.90  | 69.59 | 44.24 |  43.85   |    0.00    |   47.56    | 55.02 |

##### moe8：

MOSA加在整层上。

|             Method              | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :-----------------------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
| swin_large_lr001_moe-64-12-4-16 | val  | 62.43 |   97.91   | 87.02 | 71.83  | 73.61 | 42.24 |  57.51   |    0.00    |   66.44    | 65.33 |
|                                 | test | 57.04 |   98.35   | 89.86 | 76.07  | 66.94 | 47.27 |  37.46   |    0.02    |   52.18    | 45.25 |

##### moe9：

transformer decoder 也使用 MoE 结构的 Adapter 。

|               Method                | Mode | mIoU  | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :---------------------------------: | :--: | :---: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
|    swin_large_lr001_moe-64-12-4     | test | 58.10 |   98.27   | 89.73 | 76.12  | 68.58 | 45.13 |  40.89   |    0.00    |   54.21    | 50.00 |
| swin_large_lr001_moe-64-12-4_loss30 | test | 59.25 |   98.36   | 90.26 | 75.58  | 69.22 | 44.76 |  39.24   |    0.68    |   54.63    | 60.55 |
| swin_large_lr001_moe-64-12-4_loss50 | test | 58.13 |   98.33   | 89.87 | 76.11  | 69.14 | 44.35 |  44.88   |    0.00    |   54.29    | 46.23 |
|     swin_large_lr001_moe-64-9-3     | test | 59.36 |   98.30   | 89.72 | 75.87  | 66.18 | 43.31 |  43.80   |    0.00    |   56.90    | 60.18 |
|    swin_large_lr001_moe-64-18-6     | test | 57.57 |   98.28   | 90.17 | 75.47  | 67.48 | 46.93 |  36.61   |    0.00    |   53.65    | 49.51 |
|    swin_large_lr001_moe-64-27-9     | test | 57.59 |   98.31   | 89.82 | 76.09  | 68.66 | 43.89 |  39.71   |    0.00    |   54.21    | 47.64 |

---

#### RGB-T

##### MFNet 

|                      Method                       | Mode |   mIoU    | Unlabeled |  Car  | Person | Bike  | Curve | Car Stop | Guard-rail | Color Cone | Bump  |
| :-----------------------------------------------: | :--: | :-------: | :-------: | :---: | :----: | :---: | :---: | :------: | :--------: | :--------: | :---: |
|                   CMX (MiT-B4)                    | test |   59.7    |   98.3    | 90.1  |  75.2  | 64.5  | 50.2  |   35.3   |    8.5     |    54.2    | 60.6  |
|                  CMNeXt (MiT-B4)                  | test |   59.9    |           |       |        |       |       |          |            |            |       |
|   swin-l_bs4_20k_lr001_moe-64-12-4-16 (default)   | test | **60.11** |   98.36   | 90.48 | 75.99  | 69.27 | 47.18 |  51.15   |    0.00    |   52.86    | 55.70 |
|   swin-l_bs4_20k_lr001_moe-64-12-4-16 (no_aug)    | test |   58.38   |   98.28   | 89.56 | 74.23  | 66.87 | 45.85 |  44.20   |    0.00    |   53.35    | 53.07 |
|     swin-l_bs4_20k_lr001_moe-64-12-4-16 (480)     | test |   59.62   |   98.37   | 90.61 | 76.37  | 66.84 | 45.73 |  47.70   |    0.00    |   51.82    | 59.15 |
|  swin-l_bs2_20k_lr001_moe-64-12-2-4-16 (no_aug)   | test |           |           |       |        |       |       |          |            |            |       |
| swin-l_bs2_20k_lr001_moe-64-12-2-4-16 (ms_weight) | test |   58.34   |           |       |        |       |       |          |            |            |       |
|                                                   |      |           |           |       |        |       |       |          |            |            |       |

##### PST900

|                      Method                      | Mode | mIoU  | mAcc  | Background | Fire-Extinguisher | Hand-Drill | Backpack | Survivor |
| :----------------------------------------------: | :--: | :---: | ----- | :--------: | :---------------: | :--------: | :------: | :------: |
|                      MiLNet                      | test | 85.11 | 92.99 |   99.54    |       82.15       |   76.25    |  89.64   |  78.69   |
|                   symmetrical                    | test | 88.7  |       |            |                   |            |          |          |
|   swin-l_bs2_10k_lr001_moe-64-12-4-16 (no_aug)   | test | 53.31 | 61.34 |   98.92    |       32.46       |   26.88    |  61.70   |  46.60   |
| swin-l_bs2_8k_lr0001_moe-64-12-4-16 (aug_weight) | test | 89.47 | 93.86 |   99.67    |       82.47       |   88.80    |  92.40   |  81.53   |



---

#### RGB-D

##### NYU Depth V2

- 测试是使用多尺度 {0.5, 0.75, 1, 1.25, 1.5} + flip，共 12 组。

|                  Method                   | Mode | mIoU  |  Acc  |
| :---------------------------------------: | :--: | :---: | :---: |
|               CMX (MiT-B5)                | test | 56.9  | 80.1  |
| swin-l_bs8_20k_lr001_moe-64-12-4-16 (aug) | test | 49.43 | 63.84 |
|               0.5 (320*240)               | test |       |       |
|                 0.5_flip                  | test |       |       |
|              0.75 (480*360)               | test |       |       |
|                 0.75_flip                 | test |       |       |
|                1 (640*480)                | test | 47.99 | 62.39 |
|                  1_flip                   | test |       |       |
|              1.25 (800*600)               | test |       |       |
|                 1.25_flip                 | test |       |       |
|               1.5 (960*720)               | test |       |       |
|                 1.5_flip                  | test |       |       |



##### SUN-RGBD

|               Method                | Mode | mIoU  | Acc  |
| :---------------------------------: | :--: | :---: | :--: |
|            CMX (MiT-B5)             | test | 52.4  | 83.8 |
| swin-l_bs4_20k_lr001_moe-64-12-4-16 | test | 15.57 |      |

##### Cityscapes

|                         Method                          | Mode | mIoU  |
| :-----------------------------------------------------: | :--: | :---: |
|                      CMX (MiT-B4)                       | val  | 82.6  |
|     swin-l_bs4_10k_lr001_moe-64-12-4-16 (512x1024)      | val  | 72.73 |
|     swin-l_bs4_20k_lr0001_moe-64-12-4-16 (512x1024)     | val  | 75.16 |
| swin-l_bs4_10k_lr001_moe-64-12-4-16 (RGB-RGB)(512x1024) | val  | 71.63 |
|     swin-l_bs2_10k_lr001_moe-64-12-4-16 (1024x2048)     | val  | 76.16 |
|     swin-l_bs2_20k_lr001_moe-64-12-4-16 (1024x2048)     | val  | 81.80 |
|    swin-l_bs2_20k_lr0001_moe-64-12-4-16 (1024x2048)     | val  | 81.01 |
|                                                         |      |       |

##### DELIVER

|                     Method                      | Mode |   mIoU    |
| :---------------------------------------------: | :--: | :-------: |
|                  CMX (MiT-B2)                   | test |   62.67   |
|                 CMNeXt (MiT-B2)                 | test |   63.58   |
|  swin-l_bs4_20k_lr001_moe-64-12-4-16 (no_aug)   | val  |   66.45   |
|                                                 | test |   56.03   |
|   swin-l_bs4_20k_lr001_moe-64-12-4-16 (flip)    | val  |   65.65   |
|                                                 | test |   57.80   |
|   swin-l_bs4_20k_lr001_moe-64-12-4-16 (color)   | val  |   66.80   |
|                                                 | test |   56.50   |
| swin-l_bs4_20k_lr001_moe-64-12-4-16 (crop_flip) | val  |   66.40   |
|                                                 | test |   56.43   |
|  swin-l_bs4_40k_lr001_moe-64-12-4-16 (no_aug)   | val  | **68.95** |
|                                                 | test |   57.61   |



---

#### RGB-P

##### ZJU-RGB-P

AoLP (Trichromatic)

|                     Method                     | Mode | mIoU  | Building | Glass |  Car  | Road  | Tree  |  Sky  | Pedestrian | Bicycle |
| :--------------------------------------------: | :--: | :---: | :------: | :---: | :---: | :---: | :---: | :---: | :--------: | :-----: |
|               CMX (SegFormer-B2)               | test | 92.0  |   91.5   | 87.3  | 95.8  | 98.2  | 96.6  | 89.3  |    85.6    |  91.9   |
|               CMX (SegFormer-B4)               | test | 92.6  |   91.6   | 88.8  | 96.3  | 98.3  | 96.8  | 89.7  |    86.2    |  92.8   |
|  swin-l_bs2_10k_lr001_moe-64-12-4-16 (no_aug)  | test | 93.58 |  92.71   | 89.06 | 96.38 | 98.30 | 97.04 | 90.71 |   91.25    |  93.19  |
| swin-l_bs2_10k_lr001_moe-64-12-4-16 (ms_flip)  | test | 93.89 |  93.78   | 89.51 | 96.45 | 98.31 | 97.27 | 91.01 |   91.55    |  93.24  |
| swin-l_bs2_20k_lr0001_moe-64-12-4-16 (ms_flip) | test | 94.00 |  92.87   | 90.85 | 96.67 | 98.32 | 97.14 | 91.08 |   91.68    |  93.41  |

DoLP (Trichromatic)

|                     Method                     | Mode | mIoU  | Building | Glass |  Car  | Road  | Tree  |  Sky  | Pedestrian | Bicycle |
| :--------------------------------------------: | :--: | :---: | :------: | :---: | :---: | :---: | :---: | :---: | :--------: | :-----: |
|              CMX ( SegFormer-B2)               | test | 92.2  |   91.8   | 87.8  | 96.1  | 98.2  | 96.7  | 89.4  |    86.1    |  91.8   |
|              CMX ( SegFormer-B4)               | test | 92.5  |   91.6   | 88.6  | 96.3  | 98.3  | 96.7  | 89.5  |    86.5    |  92.2   |
|  swin-l_bs2_10k_lr001_moe-64-12-4-16 (no_aug)  | test | 93.55 |  92.63   | 89.30 | 96.28 | 98.15 | 96.98 | 90.73 |   91.22    |  93.13  |
| swin-l_bs2_10k_lr001_moe-64-12-4-16 (ms_flip)  | test | 93.59 |  92.94   | 88.89 | 96.27 | 98.26 | 97.10 | 91.05 |   91.21    |  93.02  |
| swin-l_bs2_20k_lr0001_moe-64-12-4-16 (ms_flip) | test | 93.87 |  92.73   | 90.85 | 96.74 | 98.38 | 97.04 | 90.72 |   91.26    |  93.25  |



---

#### RGB-E

##### EventScape

|                          Method                           | Mode | mIoU  | pAcc  |
| :-------------------------------------------------------: | :--: | :---: | :---: |
|                       CMX (Swin-b)                        | val  | 61.21 | 91.61 |
|                    CMX (SegFormer-B4)                     | val  | 64.28 | 92.60 |
|           swin-l_bs16_10k_lr001_moe-64-12-4-16            | val  | 69.20 | 92.20 |
|         **swin-b**_bs16_10k_lr001_moe-64-12-4-16          | val  | 68.01 | 91.87 |
|   swin-l_bs16_10k_lr001_moe-64-12-4-16 (直接train/test)   | test | 69.69 | 90.72 |
| **swin-b**_bs16_10k_lr001_moe-64-12-4-16 (直接train/test) | test | 69.45 | 90.91 |

swin-b_loss-class-mask-dice

|                       Method                        | Mode | mIoU  |   pAcc    |
| :-------------------------------------------------: | :--: | :---: | :-------: |
|                    CMX (Swin-b)                     | val  | 61.21 |   91.61   |
| swin-b_bs16_10k_lr001_moe-best_loss-2-5-5 (default) | val  | 68.01 |   91.87   |
|      swin-b_bs16_10k_lr001_moe-best_loss-2-5-8      | val  | 68.04 |   91.99   |
|      swin-b_bs16_10k_lr001_moe-best_loss-2-8-5      | val  | 68.16 |   91.96   |
|      swin-b_bs16_10k_lr001_moe-best_loss-2-8-8      | val  | 68.22 |   91.85   |
|      swin-b_bs16_10k_lr001_moe-best_loss-5-8-5      | val  | 68.46 |   92.00   |
|      swin-b_bs16_10k_lr001_moe-best_loss-5-5-8      | val  | 68.42 |   91.96   |
|      swin-b_bs16_10k_lr001_moe-best_loss-5-8-8      | val  | 68.53 | **92.14** |
|     swin-b_bs16_10k_lr001_moe-best_loss-5-10-8      | val  | 68.74 |   92.02   |
|     swin-b_bs16_10k_lr001_moe-best_loss-5-8-10      | val  | 68.44 |   91.90   |
|     swin-b_bs16_10k_lr001_moe-best_loss-5-10-10     | val  | 68.19 |   91.91   |
|      swin-b_bs16_10k_lr001_moe-best_loss-8-5-5      | val  | 68.55 |   91.98   |
| swin-l_bs16_10k_lr001_moe-best_loss-2-5-5 (default) | val  | 69.20 |   92.20   |
|      swin-l_bs16_10k_lr001_moe-best_loss-8-5-5      | val  | 69.30 |   92.08   |

swin_query

|                      Method                       | Mode |   mIoU    |   pAcc    |
| :-----------------------------------------------: | :--: | :-------: | :-------: |
|                   CMX (Swin-b)                    | val  |   61.21   |   91.61   |
| swin-b_bs16_10k_lr001_moe-best_query100 (default) | val  |   68.01   |   91.87   |
|      swin-b_bs16_10k_lr001_moe-best_query25       | val  |   68.23   |   92.03   |
|      swin-b_bs16_10k_lr001_moe-best_query50       | val  | **68.80** | **92.18** |
|    swin-b_384_bs16_10k_lr001_moe-best_query50     | val  |   67.95   |   91.81   |
|      swin-b_bs16_10k_lr001_moe-best_query75       | val  |   68.77   |   91.93   |
| swin-l_bs16_10k_lr001_moe-best_query100 (default) | val  |   69.20   |   92.20   |
|      swin-l_bs16_10k_lr001_moe-best_query50       | val  |   68.29   |   91.88   |
|      swin_s_bs16_10k_lr001_moe-best_query50       | val  |   67.77   |   91.73   |
|      swin_t_bs16_10k_lr001_moe-best_query50       | val  |   67.53   |   91.64   |

##### DELIVER

|                    Method                    | Mode |   mIoU    |
| :------------------------------------------: | :--: | :-------: |
|                 CMX (MiT-B2)                 | test |   56.52   |
|               CMNeXt (MiT-B2)                | test |   57.48   |
| swin-l_bs4_20k_lr001_moe-64-12-4-16 (no_aug) | val  |   56.65   |
|                                              | test |   51.79   |
| swin-l_bs4_40k_lr001_moe-64-12-4-16 (no_aug) | val  | **57.60** |
|                                              | test |   51.59   |
|                                              |      |           |



---

#### RGB-L

##### DELIVER

color

|                      Method                      | Mode | mIoU  |
| :----------------------------------------------: | :--: | :---: |
|                   CMX (MiT-B2)                   | test | 56.37 |
|                 CMNeXt (MiT-B2)                  | test | 58.04 |
|   swin-l_bs4_20k_lr001_moe-64-12-4-16 (no_aug)   | val  | 56.39 |
|                                                  | test | 52.87 |
|   swin-l_bs4_40k_lr001_moe-64-12-4-16 (no_aug)   | val  | 55.55 |
|                                                  | test | 51.91 |
| swin-l_bs4_40k_lr001_moe-64-12-4-16 (crop_flip)  | val  | 56.35 |
|                                                  | test | 51.93 |
|  swin-l_bs4_40k_lr0001_moe-64-12-4-16 (no_aug)   | val  | 57.49 |
|                                                  | test | 51.50 |
| swin-l_bs4_40k_lr0001_moe-64-12-4-16 (crop_flip) | val  | 57.67 |
|                                                  | test | 51.50 |

gray

|                      Method                      | Mode | mIoU  |
| :----------------------------------------------: | :--: | :---: |
|                   CMX (MiT-B2)                   | test | 56.37 |
|                 CMNeXt (MiT-B2)                  | test | 58.04 |
|   swin-l_bs4_20k_lr001_moe-64-12-4-16 (no_aug)   | val  | 56.35 |
|                                                  | test | 52.14 |
| swin-l_bs4_20k_lr001_moe-64-12-4-16 (crop_flip)  | val  | 55.91 |
|                                                  | test | 51.47 |
|  swin-l_bs4_40k_lr0001_moe-64-12-4-16 (no_aug)   | val  | 57.12 |
|                                                  | test | 52.17 |
| swin-l_bs4_40k_lr0001_moe-64-12-4-16 (crop_flip) | val  | 57.25 |
|                                                  | test | 51.94 |



---

### Add Adapter

##### Cross Modal Adapter

```python
class CrossModalAdapter(nn.Module):
    def __init__(self, d_model, d_mid, dropout):
        """
        Args:
            d_model (int): the dimension of input tokens
            d_mid (int): the dimension after the down projection
        """
        super().__init__()
        self.down_proj = nn.Linear(d_model, d_mid)
        self.linea_proj = nn.Linear(d_mid, d_mid)
        self.up_proj = nn.Linear(d_mid, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_down = self.dropout(self.activation(self.down_proj(x)))
        # x_linear = self.activation(self.linea_proj(x_down))
        x_up = self.up_proj(x_down)

        return x_up
```

##### Self Adapter

```python
class SelfAdapter(nn.Module):
    def __init__(self, d_model, d_mid, dropout):
        super().__init__()
        self.down_proj = nn.Linear(d_model, d_mid)
        self.up_proj = nn.Linear(d_mid, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)

        return x
```

##### Sharing MoE Cross Modal Adapter

```python
class Expert(nn.Module):
    def __init__(self, d_model, d_mid, dropout):
        super().__init__()
        self.down_proj = nn.Linear(d_model, d_mid)
        self.up_proj = nn.Linear(d_mid, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_down = self.activation(self.down_proj(x))
        x_down = self.dropout(x_down)
        x_up = self.up_proj(x_down)

        return x_up


class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k, pool_type="avg", query=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.pool_type = pool_type

        if query:
            self.query = nn.Parameter(torch.randn(1, 1, 1, d_model))  # learnable query vector
            self.routing = nn.Linear(d_model * 2, num_experts)  # input dim is doubled due to concat with query
        else:
            self.routing = nn.Linear(d_model, num_experts)

    def forward(self, x, spatial_shapes, level_start_index, patches):
        bs, _, c = x.shape

        kernel_size = int(math.sqrt(patches))
        stride = kernel_size
        assert kernel_size * kernel_size == patches, "The patch size should be a square number."

        # for h, w in spatial_shapes:
        #     assert h % kernel_size == 0 and w % kernel_size == 0, \
        #         f"The spatial shape {h, w} should be divisible by the kernel size {kernel_size}."

        # [bs, h*w, c] -> [bs, h, w, c]
        num_features = len(level_start_index)
        split_size_or_sections = [None] * num_features
        for i in range(num_features):
            if i < num_features - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = x.shape[1] - level_start_index[i]
        x_split = torch.split(x, split_size_or_sections, dim=1)

        patch_probs_list = []
        patch_indices_list = []
        for i, (h, w) in enumerate(spatial_shapes):
            # Get the feature map of the current scale
            x_level = x_split[i]  # [bs, h, w, c]
            x_level = x_level.view(bs, h, w, c).permute(0, 3, 1, 2)  # [bs, c, h, w]

            # Pooling the feature map
            if self.pool_type == "avg":
                x_pooled = F.avg_pool2d(x_level, kernel_size=kernel_size, stride=stride)  # [bs, c, h//ks, w//ks]

            elif self.pool_type == "max":
                x_pooled = F.max_pool2d(x_level, kernel_size=kernel_size, stride=stride)  # [bs, c, h//ks, w//ks]

            else:
                raise ValueError(f"Unsupported pool type: {self.pool_type}")

            x_pooled = x_pooled.permute(0, 2, 3, 1)  # [bs, h//ks, w//ks, c]

            # Calculate the routing probabilities
            routing_logits = self.routing(x_pooled)                                     # [bs, h//ks, w//ks, n_e]
            top_k_logits, top_k_indices = routing_logits.topk(self.top_k, dim=-1)       # [bs, h//ks, w//ks, top_k]

            infs = torch.full_like(routing_logits, float("-inf"))
            sparse_logits = infs.scatter(dim=-1, index=top_k_indices, src=top_k_logits)
            probs = F.softmax(sparse_logits, dim=-1)                                    # [bs, h//ks, w//ks, n_e]

            # Upsample probs and indices to the original spatial shapes
            probs = F.interpolate(
                probs.permute(0, 3, 1, 2),                                        # [bs, n_e, h//ks, w//ks]
                size=(h, w),
                mode="nearest"  # Nearest neighbor interpolation to copy values
            ).permute(0, 2, 3, 1)                                                       # [bs, h, w, n_e]

            top_k_indices = F.interpolate(
                top_k_indices.float().permute(0, 3, 1, 2),                              # [bs, top_k, h//ks, w//ks]
                size=(h, w),
                mode="nearest"
            ).long().permute(0, 2, 3, 1)                                                # [bs, h, w, top_k]

            probs = probs.view(bs, -1, self.num_experts)                                # [bs, h*w, n_e]
            top_k_indices = top_k_indices.view(bs, -1, self.top_k)                      # [bs, h*w, top_k]

            patch_probs_list.append(probs)
            patch_indices_list.append(top_k_indices)

        # Concatenate the routing probabilities and indices
        probs = torch.cat(patch_probs_list, dim=1)                                      # [bs, H*W, n_e]
        top_k_indices = torch.cat(patch_indices_list, dim=1)                            # [bs, H*W, top_k]

        return probs, top_k_indices
```

```python
    def forward_moe(self, src, router, shared_experts, spatial_shapes, level_start_index, patches):
        probs, top_k_indices = router(src, spatial_shapes, level_start_index, patches)
        # flatten to concat every batch
        src_flatten = src.view(-1, src.shape[-1])
        probs_flatten = probs.view(-1, probs.shape[-1])

        final_output = torch.zeros_like(src)
        for i, expert in enumerate(shared_experts):
            expert_mask = (top_k_indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_output = expert(src_flatten[flat_mask])
                gating_scores = probs_flatten[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output
```

##### Sharing Self MoE Adapter (deprecated)

```python
class Expert(nn.Module):
    def __init__(self, d_model, d_mid, dropout):
        super().__init__()
        self.down_proj = nn.Linear(d_model, d_mid)
        self.up_proj = nn.Linear(d_mid, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_down = self.activation(self.down_proj(x))
        x_down = self.dropout(x_down)
        x_up = self.up_proj(x_down)

        return x_up


class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.routing = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # Calculate the routing probabilities
        routing_logits = self.routing(x)                                            
        top_k_logits, top_k_indices = routing_logits.topk(self.top_k, dim=-1)       

        infs = torch.full_like(routing_logits, float("-inf"))
        sparse_logits = infs.scatter(dim=-1, index=top_k_indices, src=top_k_logits)
        probs = F.softmax(sparse_logits, dim=-1)                                   

        return probs, top_k_indices


class SelfMoEAdapter(nn.Module):
    def __init__(self, d_model, d_mid, num_experts, top_k):
        """
        Args:
            d_model (int): the dimension of input tokens
            d_mid (int): the dimension of adapter hidden layer
            num_experts (int): the number of experts
            top_k (int): the number of experts to select
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(Expert(d_model, d_mid, dropout=0.0) for _ in range(num_experts))

    def forward(self, x):

        probs, top_k_indices = self.router(x)
        # flatten to concat every batch
        x_flatten = x.view(-1, x.shape[-1])
        probs_flatten = probs.view(-1, probs.shape[-1])

        final_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_mask = (top_k_indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_output = expert(x_flatten[flat_mask])
                gating_scores = probs_flatten[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output
```

---

### 损失函数

#### 总损失函数：

$$
\mathcal{L}_{total} = \lambda_{ce}\mathcal{L}_{ce} + \lambda_{mask}\mathcal{L}_{mask} + \lambda_{dice}\mathcal{L}_{dice}
$$

#### 分类损失：

$$
\mathcal{L}_{ce} = \frac{1}{N} \sum_{i=1}^{N} w_{c_i} \cdot \left(-\log\left( \frac{e^{s_{c_i}}}{\sum_{j=1}^{C+1}e^{s_j}} \right) \right)
$$

- $w_{c_i}$ 为类别权重
- $C + 1$ 为总类别数（包含可学习的query）

#### 掩码损失（二值交叉熵损失）：

$$
\mathcal{L}_{mask} = \frac{1}{K} \sum_{k=1}^{K} \left[ y_k \cdot \log (\sigma(p_k)) + (1 - y_k) \cdot \log(1 - \sigma(p_k)) \right]
$$

- $\sigma$ 为 sigmoid 函数
- 使用点采样策略——每个 mask 采样 `num_points` 个点计算

#### dice损失：

$$
\mathcal{L}_{dice} = 1 - \frac{2 \sum_{k=1}^{K}{p_k y_k} + \epsilon}{\sum_{k=1}^{K}p_k + \sum_{k=1}^{K}y_k + \epsilon}
$$

- $\epsilon$ 为平滑系数
- 同样基于点采样计算
