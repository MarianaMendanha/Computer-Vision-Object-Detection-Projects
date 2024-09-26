#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""Finetuning any 🤗 Transformers model supported by AutoModelForObjectDetection for object detection leveraging the Trainer API."""
"""
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.60it/s]
***** test metrics *****
  epoch                    =      100.0
  test_loss                =     1.0359
  test_map                 =     0.4073
  test_map_50              =     0.7499
  test_map_75              =     0.3809
  test_map_Coverall        =     0.4984
  test_map_Face_Shield     =     0.5242
  test_map_Gloves          =     0.3512
  test_map_Goggles         =     0.2417
  test_map_Mask            =     0.4213
  test_map_large           =     0.5704
  test_map_Mask            =     0.4213
  test_map_large           =     0.5704
  test_map_medium          =     0.3809
  test_map_medium          =     0.3809
  test_map_small           =     0.1856
  test_mar_1               =     0.3373
  test_mar_1               =     0.3373
  test_mar_10              =     0.5663
  test_mar_100             =     0.5791
  test_mar_100_Coverall    =     0.7538
  test_mar_100_Coverall    =     0.7538
  test_mar_100_Face_Shield =     0.6588
  test_mar_100_Gloves      =     0.4525
  test_mar_100_Goggles     =     0.5207
  test_mar_100_Mask        =     0.5098
  test_mar_large           =     0.7258
  test_mar_medium          =     0.4987
  test_mar_small           =     0.3171
  test_runtime             = 0:00:11.87
  test_samples_per_second  =      2.442
  test_steps_per_second    =      0.337
[INFO|trainer.py:3600] 2024-08-20 11:19:59,041 >> Saving model checkpoint to detr-finetuned-cppe-5-10k-steps
[INFO|configuration_utils.py:472] 2024-08-20 11:19:59,054 >> Configuration saved in detr-finetuned-cppe-5-10k-steps\config.json
[INFO|modeling_utils.py:2800] 2024-08-20 11:19:59,223 >> Model weights saved in detr-finetuned-cppe-5-10k-steps\model.safetensors
[INFO|image_processing_base.py:258] 2024-08-20 11:19:59,224 >> Image processor saved in detr-finetuned-cppe-5-10k-steps\preprocessor_config.json
[INFO|modelcard.py:449] 2024-08-20 11:19:59,383 >> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Object Detection', 'type': 'object-detection'}}
"""
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Mapping, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from datasets import DatasetDict, Dataset, Features, Value, Sequence, ClassLabel, Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import center_to_corners_format
from transformers.trainer import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from pprint import pprint
import json
import random

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.45.0.dev0")

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/object-detection/requirements.txt")


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


























# Função para adicionar novas features ao dataset
def add_features(example, idx):
    example['image_id'] = idx + 1
    example['width'] = 640
    example['height'] = 640
    print("aAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",example['objects'])
    for i in range(len(example['objects']['bbox'])):
        example['objects']['bbox'][i] = [float(coord) for coord in example['objects']['bbox'][i]]
    return example

def format_image_annotations_as_coco(
    image_id: str, categories: List[int], areas: List[float], bboxes: List[Tuple[float]]
) -> dict:
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }

# Função para renomear a chave 'categories' para 'category'
def rename_categories_to_category(example):
    example['objects']['category'] = example['objects'].pop('categories')
    return example




def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    if len(image_size) != 2:
        raise ValueError(f"Expected image_size to have 2 values (height, width), but got {len(image_size)}")
    
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes


def augment_and_transform_batch(
    examples: Mapping[str, Any],
    transform: A.Compose,
    image_processor: AutoImageProcessor,
    return_pixel_mask: bool = False,
) -> BatchFeature:
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])

        # print("---------------------------------->>")
        # print()
        # print(type(output))
        # pprint(output)
        # print("---------------------------------->>")
        
        # convert bounding boxes to [x0, y0, x1, y1] format -> PPEv5-17 dataset
        # convert bounding boxes from YOLO to Pascal VOC format
        # image_size = image.shape[:2]  # (height, width)
        # converted_bboxes = convert_bbox_yolo_to_pascal(torch.tensor(output["bboxes"]), image_size).tolist()
        
        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)
        

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    
    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def collate_fn(batch: List[BatchFeature]) -> Mapping[str, Union[torch.Tensor, List[Any]]]:
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


@torch.no_grad()
def compute_metrics(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    threshold: float = 0.0,
    id2label: Optional[Mapping[int, str]] = None,
) -> Mapping[str, float]:
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor([x["orig_size"] for x in batch])
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: str = field(
        default="cppe-5",
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    image_square_size: Optional[int] = field(
        default=600,
        metadata={"help": "Image longest size will be resized to this value, then image will be padded to square."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/detr-resnet-50",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to raise an error if some of the weights from the checkpoint do not have the same size as the weights of the model (if for instance, you are instantiating a model with 10 labels from a checkpoint with 3 labels)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_object_detection", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # ------------------------------------------------------------------------------------------------
    # Load dataset, prepare splits
    # ------------------------------------------------------------------------------------------------

    dataset = load_dataset(
        data_args.dataset_name, cache_dir=model_args.cache_dir, trust_remote_code=model_args.trust_remote_code, 
    )
    
    """
    DATASET CUSTOMIZADO
    """
    # dataset = load_dataset("imagefolder", data_dir="PPEv5-17/", cache_dir=model_args.cache_dir, trust_remote_code=model_args.trust_remote_code)
    # for aux in dataset_original['train']:
    #     print(aux["objects"]["category"])
    
    
    # print(dataset["train"]["objects"][0])
    # print(dataset["train"][0])

    # # Carregar o dataset original
    # dataset = DatasetDict({
    #     "train": Dataset.from_json("path/to/train/metadata.jsonl"),
    #     "validation": Dataset.from_json("path/to/validation/metadata.jsonl")
    # })

    # Definir as novas features
    # new_features = Features({
    #     'image_id': Value('int64'),
    #     'image': dataset['train'].features['image'],
    #     'width': Value('int32'),
    #     'height': Value('int32'),
    #     'objects': dataset['train'].features['objects']
    # })
    
    # print(dataset['train'].features["objects"]['bbox'])
    # original_objects = dataset['train'].features["objects"]
    
    # desired_structure = {
    #     'image_id': Value(dtype='int64', id=None),
    #     'image': dataset['train'].features['image'],
    #     'width': Value(dtype='int32', id=None),
    #     'height': Value(dtype='int32', id=None),
    #     'objects': Sequence(
    #         feature={
    #             'id': original_objects['id'].feature,
    #             'area': original_objects['area'].feature,
    #             'bbox': Sequence(feature=Value(dtype='float32', id=None), length=4, id=None),
    #             'category': ClassLabel(names=['ppe', 'boots', 'glasses', 'gloves', 'helmet', 'no boots', 'no glasses', 'no gloves', 'no helmet', 'no vest', 'person', 'vest'], id=None)
    #         },
    #         length=-1,
    #         id=None
    #     )
    # }
    # print(original_objects['id'].feature)
    # dataset = dataset.map(add_features, with_indices=True)
    # print(dataset['train'].features['objects'][0])
    
    """# Função para reformatar os bbox
    def reformat_bbox(example):
        for obj in example['objects']:
            obj['bbox'] = [float(coord) for coord in obj['bbox']]
        return example

    # Aplicar a função para reformatar os bbox
    dataset = dataset.map(reformat_bbox)"""
    
    # Criando o dataset com a estrutura desejada
    # reformatted_data = []
    # for train in dataset['train']:
    #     objects = []
    #     for item in train['objects']:
    #         if isinstance(item['bbox'], list) and all(isinstance(bbox, list) for bbox in item['bbox']):
    #             item['bbox'] = [[float(coord) for coord in bbox] for bbox in item['bbox']]
    #         objects.append(item)
    #     reformatted_data.append({
    #         'image_id': train['image_id'],
    #         'image': train['image'],
    #         'width': train['width'],
    #         'height': train['height'],
    #         'objects': objects
    #     })


    # dataset = Dataset.from_dict(reformatted_data, features=desired_structure)
    
    
    
    # # Aplicar a função para adicionar as novas features
    # dataset = dataset.map(add_features, with_indices=True, features=new_features)
    print(dataset['train'].features["objects"]['bbox'])

    # dataset = dataset.map(rename_categories_to_category)
    print(dataset)

    # Salvar o dataset modificado
    # dataset.save_to_disk("path/to/new_dataset")

    # print(dataset['train']['objects'][0])
   
    # id2label = {
    #                 0: "ppe",
    #                 1: "boots",
    #                 2: "glasses",
    #                 3: "gloves",
    #                 4: "helmet",
    #                 5: "no boots",
    #                 6: "no glasses",
    #                 7: "no gloves",
    #                 8: "no helmet",
    #                 9: "no vest",
    #                 10: "person",
    #                 11: "vest"
    #             }
    print("-----------------------------------------------------------------------")
    # If we don't have a validation split, split off a percentage of train as validation
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split, seed=training_args.seed)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Get dataset categories and prepare mappings for label_name <-> label_id
    categories = dataset["train"].features["objects"].feature["category"].names
    print(categories)
    id2label = dict(enumerate(categories))
    
    label2id = {v: k for k, v in id2label.items()}
    pprint(label2id)
    sys.exit()

    # ------------------------------------------------------------------------------------------------
    # Load pretrained config, model and image processor
    # ------------------------------------------------------------------------------------------------

    common_pretrained_args = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        **common_pretrained_args,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        **common_pretrained_args,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        do_resize=True,
        size={"max_height": data_args.image_square_size, "max_width": data_args.image_square_size},
        do_pad=True,
        pad_size={"height": data_args.image_square_size, "width": data_args.image_square_size},
        **common_pretrained_args,
    )

    # ------------------------------------------------------------------------------------------------
    # Define image augmentations and dataset transforms
    # ------------------------------------------------------------------------------------------------
    max_size = data_args.image_square_size
    train_augment_and_transform = A.Compose(
        [
            A.Compose(
                [
                    A.SmallestMaxSize(max_size=max_size, p=1.0),
                    A.RandomSizedBBoxSafeCrop(height=max_size, width=max_size, p=1.0),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=7, p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                ],
                p=0.1,
            ),
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )
    validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

    # Make transform functions for batch and apply for dataset splits
    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
    )

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
    # dataset["test"] = dataset["test"].with_transform(validation_transform_batch)

    # ------------------------------------------------------------------------------------------------
    # Model training and evaluation with Trainer API
    # ------------------------------------------------------------------------------------------------

    eval_compute_metrics_fn = partial(
        compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        tokenizer=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Final evaluation
    # if training_args.do_eval:
    #     metrics = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
    #     trainer.log_metrics("test", metrics)
    #     trainer.save_metrics("test", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
        "tags": ["object-detection", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()