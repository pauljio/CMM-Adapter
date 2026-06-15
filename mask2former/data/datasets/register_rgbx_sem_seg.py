import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog


class RegisterRGBXSemSeg:
    """
    Register custom dataset and metadata
    """

    ## class names
    # MFNet Dataset - 9 classes
    mfnet_stuff_classes = [
        "unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"
    ]

    # PST900 - 5 classes
    pst900_stuff_classes = [
        "background", "fire_extinguisher", "backpack", "hand-drill", "survivor"
    ]

    # NYU_Depth_V2 - 40 classes
    nyu_depth_v2_stuff_classes = [
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
        "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelf", "curtain",
        "dresser", "pillow", "mirror", "floor_mat", "clothes", "ceiling", "book", "refrigerator",
        "television", "paper", "towel", "shower_curtain", "box", "whiteboard", "person", "nightstand",
        "toilet", "sink", "lamp", "bathtub", "bag", "other-structure", "other-furniture", "other-prop"
    ]

    # SUN RGB-D - 37 classes
    sun_rgbd_stuff_classes = [
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
        "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor_mat",
        "clothes", "ceiling", "books", "fridge", "tv", "paper", "towel", "shower_curtain", "box", "whiteboard",
        "person", "night_stand", "toilet", "sink", "lamp", "bathtub", "bag"
    ]

    # Cityscapes - 19 classes
    cityscapes_stuff_classes = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign", "vegetation",
        "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
    ]

    # ZJU-RGB-P - 8 classes
    zju_rgbp_stuff_classes = [
        "building", "glass", "car", "road", "tree", "sky", "pedestrian", "bicycle"
    ]

    # EventScape - 12 classes
    event_scape_stuff_classes = [
        "vehicle", "building", "wall", "vegetation", "road", "pole", "road_lines",
        "fences", "pedestrian", "traffic_sign", "sidewalk", "trafficlight"
    ]

    # DELIVER - 25 classes
    deliver_stuff_classes = [
        "Building", "Fence", "Other", "Pedestrian", "Pole",
        "RoadLine", "Road", "SideWalk", "Vegetation", "Cars",
        "Wall", "TrafficSign", "Sky", "Ground", "Bridge",
        "RailTrack", "GroundRail", "TrafficLight", "Static", "Dynamic",
        "Water", "Terrain", "TwoWheeler", "Bus", "Truck"
    ]

    ## class colors
    # MFNet Dataset
    mfnet_stuff_colors = [
        [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 192], [128, 128, 0], [64, 64, 128], [192, 128, 128], [192, 64, 0]
    ]

    # PST900
    pst900_stuff_colors = [
        [0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 255]
    ]

    # NYU_Depth_V2
    nyu_depth_v2_stuff_colors = [

    ]

    # SUN-RGBD
    sun_rgbd_stuff_colors = [

    ]

    # Cityscapes
    cityscapes_stuff_colors = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
        [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ]

    # ZJU-RGB-P
    zju_rgbp_stuff_colors = [
        [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0]
    ]

    # EventScape
    event_scape_stuff_colors = [

    ]

    # DELIVER
    deliver_stuff_colors = [
        [70, 70, 70], [100, 40, 40], [55, 90, 80], [220, 20, 60], [153, 153, 153],
        [157, 234, 50], [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142],
        [102, 102, 156], [220, 220, 0], [70, 130, 180], [81, 0, 81], [150, 100, 100],
        [230, 150, 140], [180, 165, 180], [250, 170, 30], [110, 190, 160], [170, 120, 50],
        [45, 60, 150], [145, 170, 100], [0, 0, 230], [0, 60, 100], [0, 0, 70]
    ]

    def __init__(self, dataset_root, modal=None, dataset_name=None):
        """
        Args:
            dataset_root (str): root path of all the dataset
        """
        assert os.path.exists(dataset_root), f"Dataset root '{dataset_root}' does not exist"

        self.modal = modal
        self.dataset_name = dataset_name.strip() if isinstance(dataset_name, str) else dataset_name

        if self.modal == "RGB-T":
            self._init_rgb_t(dataset_root)
        elif self.modal == "RGB-D":
            self._init_rgb_d(dataset_root)
        elif self.modal == "RGB-P":
            self._init_rgb_p(dataset_root)
        elif self.modal == "RGB-E":
            self._init_rgb_e(dataset_root)
        elif self.modal == "RGB-L":
            self._init_rgb_l(dataset_root)
        else:
            raise ValueError(f"Invalid modal '{modal}', must be 'RGB-T', 'RGB-D', 'RGB-P', 'RGB-E', 'RGB-L'")

    def _resolve_dataset_name(self, default_name, supported_names):
        dataset_name = self.dataset_name or default_name
        if dataset_name not in supported_names:
            raise ValueError(
                f"Unsupported dataset_name '{dataset_name}' for modal '{self.modal}'. "
                f"Supported: {supported_names}"
            )
        return dataset_name

    def _set_common_paths(self):
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")

        self.train_image_rgb = os.path.join(self.dataset_dir, "train_rgb")
        self.train_label = os.path.join(self.dataset_dir, "labels_train")
        self.train_json = os.path.join(self.anno_dir, "semantic_train.json")

        self.val_image_rgb = os.path.join(self.dataset_dir, "val_rgb")
        self.val_label = os.path.join(self.dataset_dir, "labels_val")
        self.val_json = os.path.join(self.anno_dir, "semantic_val.json")

        self.test_image_rgb = os.path.join(self.dataset_dir, "test_rgb")
        self.test_label_path = os.path.join(self.dataset_dir, "labels_test")
        self.test_json = os.path.join(self.anno_dir, "semantic_test.json")

    def _init_rgb_t(self, dataset_root):
        dataset_name = self._resolve_dataset_name("MFNet", ["MFNet", "PST900"])
        self.dataset_dir = os.path.join(dataset_root, dataset_name)
        assert os.path.exists(self.dataset_dir), f"Dataset dir '{self.dataset_dir}' does not exist"
        self._set_common_paths()

        self.train_image_t = os.path.join(self.dataset_dir, "train_t")
        self.val_image_t = os.path.join(self.dataset_dir, "val_t")
        self.test_image_t = os.path.join(self.dataset_dir, "test_t")

        if dataset_name == "MFNet":
            self.stuff_classes = RegisterRGBXSemSeg.mfnet_stuff_classes
            self.stuff_colors = RegisterRGBXSemSeg.mfnet_stuff_colors
            self.train_val_dataset = {
                "MFNet_train": (self.train_image_rgb, self.train_image_t, self.train_json),
                "MFNet_val": (self.val_image_rgb, self.val_image_t, self.val_json),
                "MFNet_test": (self.test_image_rgb, self.test_image_t, self.test_json),
            }
        else:
            self.stuff_classes = RegisterRGBXSemSeg.pst900_stuff_classes
            self.stuff_colors = RegisterRGBXSemSeg.pst900_stuff_colors
            self.train_val_dataset = {
                "PST900_train": (self.train_image_rgb, self.train_image_t, self.train_json),
                "PST900_test": (self.test_image_rgb, self.test_image_t, self.test_json),
            }

    def _init_rgb_d(self, dataset_root):
        dataset_name = self._resolve_dataset_name("DELIVER", ["NYU_Depth_V2", "SUN-RGBD", "Cityscapes", "DELIVER"])
        self.dataset_dir = os.path.join(dataset_root, dataset_name)
        assert os.path.exists(self.dataset_dir), f"Dataset dir '{self.dataset_dir}' does not exist"
        self._set_common_paths()

        self.train_image_d = os.path.join(self.dataset_dir, "train_hha")
        self.val_image_d = os.path.join(self.dataset_dir, "val_hha")
        self.test_image_d = os.path.join(self.dataset_dir, "test_hha")

        if dataset_name == "NYU_Depth_V2":
            self.stuff_classes = RegisterRGBXSemSeg.nyu_depth_v2_stuff_classes
            self.stuff_colors = RegisterRGBXSemSeg.nyu_depth_v2_stuff_colors
            self.train_val_dataset = {
                "NYU_Depth_V2_train": (self.train_image_rgb, self.train_image_d, self.train_json),
                "NYU_Depth_V2_test": (self.test_image_rgb, self.test_image_d, self.test_json),
            }
        elif dataset_name == "SUN-RGBD":
            self.stuff_classes = RegisterRGBXSemSeg.sun_rgbd_stuff_classes
            self.stuff_colors = RegisterRGBXSemSeg.sun_rgbd_stuff_colors
            self.train_val_dataset = {
                "SUN-RGBD_train": (self.train_image_rgb, self.train_image_d, self.train_json),
                "SUN-RGBD_test": (self.test_image_rgb, self.test_image_d, self.test_json),
            }
        elif dataset_name == "Cityscapes":
            self.stuff_classes = RegisterRGBXSemSeg.cityscapes_stuff_classes
            self.stuff_colors = RegisterRGBXSemSeg.cityscapes_stuff_colors
            self.train_val_dataset = {
                "Cityscapes_train": (self.train_image_rgb, self.train_image_d, self.train_json),
                "Cityscapes_val": (self.val_image_rgb, self.val_image_d, self.val_json),
                "Cityscapes_test": (self.test_image_rgb, self.test_image_d, self.test_json),
            }
        else:
            self.stuff_classes = RegisterRGBXSemSeg.deliver_stuff_classes
            self.stuff_colors = RegisterRGBXSemSeg.deliver_stuff_colors
            self.train_val_dataset = {
                "DELIVER_train": (self.train_image_rgb, self.train_image_d, self.train_json),
                "DELIVER_val": (self.val_image_rgb, self.val_image_d, self.val_json),
                "DELIVER_test": (self.test_image_rgb, self.test_image_d, self.test_json),
            }

    def _init_rgb_p(self, dataset_root):
        dataset_name = self._resolve_dataset_name("ZJU-RGB-P", ["ZJU-RGB-P"])
        self.dataset_dir = os.path.join(dataset_root, dataset_name)
        assert os.path.exists(self.dataset_dir), f"Dataset dir '{self.dataset_dir}' does not exist"
        self._set_common_paths()

        self.stuff_classes = RegisterRGBXSemSeg.zju_rgbp_stuff_classes
        self.stuff_colors = RegisterRGBXSemSeg.zju_rgbp_stuff_colors

        self.train_image_p = os.path.join(self.dataset_dir, "train_dolp")
        self.val_image_p = os.path.join(self.dataset_dir, "val_dolp")
        self.test_image_p = os.path.join(self.dataset_dir, "test_dolp")

        self.train_val_dataset = {
            "ZJU-RGB-P_train": (self.train_image_rgb, self.train_image_p, self.train_json),
            "ZJU-RGB-P_val": (self.val_image_rgb, self.val_image_p, self.val_json),
        }

    def _init_rgb_e(self, dataset_root):
        dataset_name = self._resolve_dataset_name("DELIVER", ["EventScape", "DELIVER"])
        self.dataset_dir = os.path.join(dataset_root, dataset_name)
        assert os.path.exists(self.dataset_dir), f"Dataset dir '{self.dataset_dir}' does not exist"
        self._set_common_paths()

        self.train_image_e = os.path.join(self.dataset_dir, "train_event")
        self.val_image_e = os.path.join(self.dataset_dir, "val_event")
        self.test_image_e = os.path.join(self.dataset_dir, "test_event")

        if dataset_name == "EventScape":
            self.stuff_classes = RegisterRGBXSemSeg.event_scape_stuff_classes
            self.stuff_colors = RegisterRGBXSemSeg.event_scape_stuff_colors
            self.train_val_dataset = {
                "EventScape_train": (self.train_image_rgb, self.train_image_e, self.train_json),
                "EventScape_val": (self.val_image_rgb, self.val_image_e, self.val_json),
                "EventScape_test": (self.test_image_rgb, self.test_image_e, self.test_json),
            }
        else:
            self.stuff_classes = RegisterRGBXSemSeg.deliver_stuff_classes
            self.stuff_colors = RegisterRGBXSemSeg.deliver_stuff_colors
            self.train_val_dataset = {
                "DELIVER_train": (self.train_image_rgb, self.train_image_e, self.train_json),
                "DELIVER_val": (self.val_image_rgb, self.val_image_e, self.val_json),
                "DELIVER_test": (self.test_image_rgb, self.test_image_e, self.test_json),
            }

    def _init_rgb_l(self, dataset_root):
        dataset_name = self._resolve_dataset_name("DELIVER", ["DELIVER"])
        self.dataset_dir = os.path.join(dataset_root, dataset_name)
        assert os.path.exists(self.dataset_dir), f"Dataset dir '{self.dataset_dir}' does not exist"
        self._set_common_paths()

        self.stuff_classes = RegisterRGBXSemSeg.deliver_stuff_classes
        self.stuff_colors = RegisterRGBXSemSeg.deliver_stuff_colors

        self.train_image_l = os.path.join(self.dataset_dir, "train_lidar")
        self.val_image_l = os.path.join(self.dataset_dir, "val_lidar")
        self.test_image_l = os.path.join(self.dataset_dir, "test_lidar")

        self.train_val_dataset = {
            "DELIVER_train": (self.train_image_rgb, self.train_image_l, self.train_json),
            "DELIVER_val": (self.val_image_rgb, self.val_image_l, self.val_json),
            "DELIVER_test": (self.test_image_rgb, self.test_image_l, self.test_json),
        }

    def get_dataset_dicts(self, json_file):
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"File '{json_file}' does not exist")

        with open(json_file, "r") as f:
            dataset_dicts = json.load(f)

        # ensure the image path is full path
        for item in dataset_dicts:
            if "train" in json_file:
                item["file_name_rgb"] = os.path.join(self.train_image_rgb, item["file_name_rgb"])

                if self.modal == "RGB-T":
                    item["file_name_t"] = os.path.join(self.train_image_t, item["file_name_t"])
                elif self.modal == "RGB-D":
                    item["file_name_d"] = os.path.join(self.train_image_d, item["file_name_d"])
                elif self.modal == "RGB-P":
                    item["file_name_p"] = os.path.join(self.train_image_p, item["file_name_p"])
                elif self.modal == "RGB-E":
                    item["file_name_e"] = os.path.join(self.train_image_e, item["file_name_e"])
                elif self.modal == "RGB-L":
                    item["file_name_l"] = os.path.join(self.train_image_l, item["file_name_l"])

                item["sem_seg_file_name"] = os.path.join(self.train_label, item["sem_seg_file_name"])

            elif "val" in json_file:
                item["file_name_rgb"] = os.path.join(self.val_image_rgb, item["file_name_rgb"])

                if self.modal == "RGB-T":
                    item["file_name_t"] = os.path.join(self.val_image_t, item["file_name_t"])
                elif self.modal == "RGB-D":  # SUN-RGBD does not have "val"
                    item["file_name_d"] = os.path.join(self.val_image_d, item["file_name_d"])
                elif self.modal == "RGB-P":
                    item["file_name_p"] = os.path.join(self.val_image_p, item["file_name_p"])
                elif self.modal == "RGB-E":
                    item["file_name_e"] = os.path.join(self.val_image_e, item["file_name_e"])
                elif self.modal == "RGB-L":
                    item["file_name_l"] = os.path.join(self.val_image_l, item["file_name_l"])

                item["sem_seg_file_name"] = os.path.join(self.val_label, item["sem_seg_file_name"])

            elif "test" in json_file:
                item["file_name_rgb"] = os.path.join(self.test_image_rgb, item["file_name_rgb"])

                if self.modal == "RGB-T":
                    item["file_name_t"] = os.path.join(self.test_image_t, item["file_name_t"])
                elif self.modal == "RGB-D":
                    item["file_name_d"] = os.path.join(self.test_image_d, item["file_name_d"])
                elif self.modal == "RGB-P":
                    item["file_name_p"] = os.path.join(self.test_image_p, item["file_name_p"])
                elif self.modal == "RGB-E":
                    item["file_name_e"] = os.path.join(self.test_image_e, item["file_name_e"])
                elif self.modal == "RGB-L":
                    item["file_name_l"] = os.path.join(self.test_image_l, item["file_name_l"])

                item["sem_seg_file_name"] = os.path.join(self.test_label_path, item["sem_seg_file_name"])

            else:
                raise ValueError(f"Invalid json file: {json_file}, must contain 'train', 'val' or 'test'")

        return dataset_dicts

    def register_semantic_dataset(self, name, json_file, image_rgb_path, image_x_path):
        """
        Register a dataset with semantic segmentation annotations.
        """
        DatasetCatalog.register(name, lambda: self.get_dataset_dicts(json_file))
        MetadataCatalog.get(name).set(
            json_file=json_file,
            image_rgb_path=image_rgb_path,
            image_x_path=image_x_path,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_classes=self.stuff_classes,
            stuff_colors=self.stuff_colors,
        )

    def _log_dataset_split_statistics(self):
        """
        Log image counts for train/val/test splits in the current dataset.
        """
        logger = logging.getLogger(__name__)
        split_counts = {}

        for dataset_name, (_, _, json_file) in self.train_val_dataset.items():
            try:
                with open(json_file, "r") as f:
                    dataset_dicts = json.load(f)
            except FileNotFoundError:
                logger.warning(f"Skip counting for '{dataset_name}': json file not found -> {json_file}")
                continue
            except json.JSONDecodeError as e:
                logger.warning(f"Skip counting for '{dataset_name}': invalid json -> {json_file}, error: {e}")
                continue

            if "_train" in dataset_name:
                split = "train"
            elif "_val" in dataset_name:
                split = "val"
            elif "_test" in dataset_name:
                split = "test"
            else:
                split = dataset_name

            split_counts[split] = len(dataset_dicts)

        if split_counts:
            logger.info(
                "Dataset image counts (modal=%s): train=%s, val=%s, test=%s",
                self.modal,
                split_counts.get("train", "N/A"),
                split_counts.get("val", "N/A"),
                split_counts.get("test", "N/A"),
            )

    def register_dataset(self):
        """
        Register all custom datasets.
        """
        for key, (image_rgb_path, image_x_path, json_file) in self.train_val_dataset.items():
            self.register_semantic_dataset(
                name=key,
                json_file=json_file,
                image_rgb_path=image_rgb_path,
                image_x_path=image_x_path,
            )

        self._log_dataset_split_statistics()
