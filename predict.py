import sys
sys.path.insert(0, "Mask2Former")
# import tempfile
from pathlib import Path
import numpy as np
import cv2
import cog
import argparse
import os
import json
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

# import Mask2Former project
from mask2former import add_maskformer2_config


class Register:
    """
    Register the test dataset
    """

    # class names
    stuff_classes = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]

    # class colors
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

    dataset_root = "/data1/cjp/Projects/Mask2Former/datasets/MFNetDataset"

    def __init__(self):
        self.stuff_classes = Register.stuff_classes
        self.dataset_root = Register.dataset_root

        self.test_image_path = os.path.join(self.dataset_root, "test")

    def get_dataset_dicts(self):
        dataset_dicts = []

        for filename in os.listdir(self.test_image_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                file_path = os.path.join(self.test_image_path, filename)
                image = cv2.imread(file_path)
                height, width = image.shape[:2] if image is not None else (480, 640)

                dataset_dicts.append({
                    "file_name": file_path,
                    "height": height,
                    "width": width,
                    "image_id": filename.split('.')[0],
                    "sem_seg_file_name": None,
                })

        return dataset_dicts

    def register_semantic_dataset(self, name):
        """
        Register a dataset with semantic segmentation annotations.
        """

        DatasetCatalog.register(name, self.get_dataset_dicts)
        MetadataCatalog.get(name).set(
            image_root=self.test_image_path,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_classes=self.stuff_classes,
            stuff_colors=self.stuff_colors,
        )

    def register_dataset(self, name):
        """
        Register all custom datasets.
        """

        dataset_name = "MFNetDataset_test"
        self.register_semantic_dataset(
            name=dataset_name,
        )

        return dataset_name

# TODO: Visualize the semantic segmentation results
class Predictor:
    def __init__(self, cfg, dataset_name):
        self.predictor = DefaultPredictor(cfg)
        # self.coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
        self.mfnet_metadata = MetadataCatalog.get(dataset_name)

    # @cog.input(
    #     "image",
    #     type=Path,
    #     help="Input image for segmentation. Output will be the concatenation of Panoptic segmentation (top), "
    #          "instance segmentation (middle), and semantic segmentation (bottom).",
    # )

    def predict_single_image(self, image_path):
        im = cv2.imread(str(image_path))
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1], self.mfnet_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
        return semantic_result

    # def predict(self, image):
    #     im = cv2.imread(str(image))
    #     outputs = self.predictor(im)
    #     v = Visualizer(im[:, :, ::-1], self.mfnet_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    #     panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"),
    #                                           outputs["panoptic_seg"][1]).get_image()
    #     v = Visualizer(im[:, :, ::-1], self.mfnet_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    #     instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
    #     v = Visualizer(im[:, :, ::-1], self.mfnet_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    #     semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
    #     result = np.concatenate((panoptic_result, instance_result, semantic_result), axis=0)[:, :, ::-1]
    #     out_path = Path(tempfile.mkdtemp()) / "out.png"
    #     cv2.imwrite(str(out_path), result)
    #     return out_path

    def predict(self, test_dir, output_dir):
        test_dir = Path(test_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_path = list(test_dir.glob("*.png"))

        for image in tqdm(image_path, desc="Processing images"):
            result = self.predict_single_image(image)
            out_path = output_dir / f"{image.stem}_sem_seg.png"
            cv2.imwrite(str(out_path), result)

        return output_dir


def setup(args):
    """
    load config from file and command-line arguments
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
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

    register = Register()
    dataset_name = register.register_dataset(name="MFNetDataset_test")

    predictor = Predictor(cfg, dataset_name)
    output_dir = predictor.predict(args.test_dir, args.output_dir)
    print(f"[predict INFO] All semantic segmentation results saved to {output_dir}")


if __name__ == "__main__":
    args = get_parser().parse_args()
    print("Command Line Args:", args)

    main(args)