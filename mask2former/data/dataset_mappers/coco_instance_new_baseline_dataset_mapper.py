# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances

from pycocotools import mask as coco_mask

__all__ = ["COCOInstanceNewBaselineDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    OPTIMIZED: Crop first at full resolution, then resize for maximum detail preservation.
    Returns:
        list[Augmentation]
    """
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    # Only add random augmentations during training
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    # OPTIMIZED ORDER: Crop first at full resolution, then resize
    # This preserves maximum detail for high-resolution myotube images
    
    # For high-res images, crop a large region first, then scale down
    if hasattr(cfg.INPUT, 'CROP') and cfg.INPUT.CROP.ENABLED:
        # Use crop settings from config if available
        if is_train:
            # Training: Random crop
            augmentation.append(
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            )
        else:
            # Evaluation: Center crop for consistency
            # Use the same crop size but center crop instead of random
            if cfg.INPUT.CROP.TYPE == "relative_range":
                # For relative_range, use the average of min and max
                crop_size = cfg.INPUT.CROP.SIZE
                if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
                    avg_crop_size = (crop_size[0] + crop_size[1]) / 2
                else:
                    avg_crop_size = crop_size
                augmentation.append(
                    T.CenterCrop(crop_size=(avg_crop_size, avg_crop_size))
                )
            else:
                # For absolute size, use center crop with same size
                augmentation.append(
                    T.CenterCrop(crop_size=cfg.INPUT.CROP.SIZE)
                )
        
        # Then resize the cropped region to target size
        augmentation.append(
            T.ResizeShortestEdge(image_size, image_size)
        )
    else:
        # Fallback: Traditional approach for backward compatibility
        if is_train:
            augmentation.extend([
                T.ResizeScale(
                    min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
                ),
                T.FixedSizeCrop(crop_size=(image_size, image_size)),
            ])
        else:
            # For evaluation without explicit crop config, just resize
            augmentation.append(
                T.ResizeShortestEdge(image_size, image_size)
            )

    return augmentation


# This is specifically designed for the COCO dataset.
class COCOInstanceNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

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
        *,
        tfm_gens,
        image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        mode = "training" if is_train else "evaluation"
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] Full TransformGens used in {}: {}".format(mode, str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation - now supports both training and evaluation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            # For evaluation: apply transforms (especially cropping) and transform annotations
            if "annotations" in dataset_dict:
                # Transform annotations to match the transformed image
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                # Convert to instances for evaluation
                instances = utils.annotations_to_instances(annos, image_shape)
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = utils.filter_empty_instances(instances)
                
                # Generate masks from polygon
                h, w = instances.image_size
                if hasattr(instances, 'gt_masks'):
                    gt_masks = instances.gt_masks
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                    instances.gt_masks = gt_masks
                dataset_dict["instances"] = instances
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            dataset_dict["instances"] = instances

        return dataset_dict
