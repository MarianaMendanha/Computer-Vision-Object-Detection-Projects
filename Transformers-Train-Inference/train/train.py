import os
import sys
from transformers import pipeline, AutoImageProcessor, AutoModelForObjectDetection, TrainingArguments, Trainer
from transformers.image_transforms import center_to_corners_format

from evaluate import load
from PIL import Image, ImageDraw
from datasets import load_dataset
import albumentations as A
from functools import partial
import tensorflow as tf
import torch
import requests
import numpy as np
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
import logging

res = torch.device("cuda")
print(res)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())
print(device)

# image_path = "C:/Users/MarianaCruz/OneDrive - CrazyTechLabs/Documentos/tasks/IA-ImageProcessing/CTL-Framework-AI/extras/sample-images/0.JPG"
# model = pipeline("object-detection")
# result = model(image_path)

# print(result)
# for res in result:
#     print(res)

# # Abrir a imagem
# image = Image.open(image_path)
# draw = ImageDraw.Draw(image)

# # Desenhar bounding boxes
# for res in result:
#     box = res['box']
#     draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline="green", width=2)
#     draw.text((box['xmin'], box['ymin']), res['label'], fill="red")

# # Salvar a imagem resultante
# image.save("result_image_0.jpg")
# image.show()

model_name = "facebook/detr-resnet-50"
# model_name = "microsoft/conditional-detr-resnet-50"
image_size = 480
max_size = image_size

ds = load_dataset("Francesco/construction-safety-gsnvb")

if "validation" not in ds:
    split = ds["train"].train_test_split(0.15, seed=1337)
    ds["train"] = split["train"]
    ds["validation"] = split["test"]

if "test" not in ds:
    split = ds["train"].train_test_split(0.1, seed=1337)  # 10% para teste
    ds["train"] = split["train"]
    ds["test"] = split["test"]

print (ds)
print ("Image 0 Info")
pprint (ds["train"][0])

""" class_label:
                '0': construction-safety
                '1': helmet
                '2': no-helmet
                '3': no-vest
                '4': person
                '5': vest """

image = ds["train"][3]["image"]
annotations = ds["train"][3]["objects"]
draw = ImageDraw.Draw(image)
print ("IMAGE & ANNOTATIONS")
pprint (image)
pprint (annotations)

categories = ds["train"].features["objects"].feature["category"].names
print ("CATEGORIES")
pprint (categories)

id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}
pprint ("LABELS")
pprint (id2label)
pprint (label2id)

width=512
height=512
for i in range(len(annotations["id"])):
    box = annotations["bbox"][i]
    class_idx = annotations["category"][i]
    x, y, w, h = tuple(box)
    # Check if coordinates are normalized or not
    if max(box) > 1.0:
        # Coordinates are un-normalized, no need to re-scale them
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
    else:
        # Coordinates are normalized, re-scale them
        x1 = int(x * width)
        y1 = int(y * height)
        x2 = int((x + w) * width)
        y2 = int((y + h) * height)
    draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
    draw.text((x, y), id2label[class_idx], fill="white")

# image.save("result_image_1.jpg")
# image.show()

# Augmentation
train_augment_and_transform = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], min_area=25),
)

# Reformat Annotations
validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"]),
)

# Processing
image_processor = AutoImageProcessor.from_pretrained(
    model_name,
    do_resize=True,
    size={"height": max_size, "width": max_size},
    do_pad=True,
    pad_size={"height": max_size, "width": max_size},
)

# image_processor = VITFeatureExtractor.from_pretrained(
#     model_name
# )


def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
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


def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False):
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])

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


# Make transform functions for batch and apply for dataset splits
train_transform_batch = partial(
    augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
)
validation_transform_batch = partial(
    augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
)

ds["train"] = ds["train"].with_transform(train_transform_batch)
ds["validation"] = ds["validation"].with_transform(validation_transform_batch)
ds["test"] = ds["test"].with_transform(validation_transform_batch)

pprint(ds["train"][15])

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    print(image_size[:2])
    height, width = image_size[:2]
    boxes = boxes * torch.tensor([[width, height, width, height]]).to(device)

    return boxes

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor

@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
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
    print(predictions, "--------------------", targets)
    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        print("BATCH ", batch)
        batch_image_sizes = torch.tensor(batch['orig_size']).to(device)
        image_sizes.append(batch_image_sizes)
        print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
        print(image_sizes)
        
        boxes = torch.tensor(batch["boxes"]).to(device)
        print("BOXES TYPE->>>>>>>", type(boxes))
        print("SIZE TYPE->>>>>>>", type(batch["orig_size"]))
        
        len(batch["image_id"])
        boxes = convert_bbox_yolo_to_pascal(boxes, batch["orig_size"])
        labels = torch.tensor(batch["class_labels"]).to(device)
        post_processed_targets.append({"boxes": boxes, "labels": labels})
        
        # for image_target in range(batch['image_id']):
        #     print("IMAGE TARGET ", image_target)
        #     boxes = torch.tensor(image_target["boxes"]).to(device)
        #     boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
        #     labels = torch.tensor(image_target["class_labels"]).to(device)
        #     post_processed_targets.append({"boxes": boxes, "labels": labels})

    for batch, target_sizes in zip(predictions, image_sizes):
        print("Estrutura do batch: ", type(batch))
        if isinstance(batch, np.ndarray):
            batch_logits = batch[1]
            batch_boxes = batch[2]
        else:
            print("Tipo inesperado para batch")
            continue
        output = ModelOutput(logits=torch.tensor(batch_logits).to(device), pred_boxes=torch.tensor(batch_boxes).to(device))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

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

eval_compute_metrics_fn = partial(
    compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
)


""" 
Training involves the following steps:

1.Load the model with AutoModelForObjectDetection using the same checkpoint as in the preprocessing.
2.Define your training hyperparameters in TrainingArguments.
3.Pass the training arguments to Trainer along with the model, dataset, image processor, and data collator.
4.Call train() to finetune your model. 
"""

print(f"GPUs disponíveis: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs disponíveis: {len(gpus)}")
    for gpu in gpus:
        print(gpu)
else:
    print("Nenhuma GPU disponível.")
    
# sys.exit()
model = AutoModelForObjectDetection.from_pretrained(
    model_name,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
).to(device)


training_args = TrainingArguments(
    output_dir="./modelo-treinado-1epoch",
    num_train_epochs=1,
    fp16=False,
    per_device_train_batch_size=8,
    dataloader_num_workers=0,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    weight_decay=1e-4,
    max_grad_norm=0.01,
    # metric_for_best_model="eval_runtime",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    # eval_do_concat_batches=False,
    push_to_hub=False,
)

metric = load("accuracy", trust_remote_code=True)
def try_compute_metrics(p):
    print("============================================")
    pprint(p.predictions)
    print("============================================")
    print(f"Predictions shape: {p.predictions.shape}")
    predictions = np.array(p.predictions)
    for i, pred in enumerate(predictions):
        print(f"Prediction {i}: {pred.shape}")
    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=1)
    else:
        raise ValueError("Predictions array has an unexpected shape.")
    return metric.compute(
        predictions=predictions,
        references=p.label_ids
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=image_processor,
    data_collator=collate_fn,
    # compute_metrics=eval_compute_metrics_fn,
    # compute_metrics=try_compute_metrics,
)

# Train
# adicionar resume from checkpoint no train() -> usar de referencia o run_object_detection.py
# adicionar opção push to hub -> usar de referencia o run_object_detection.py
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
#trainer.push_to_hub()

""" Para Inferir:
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch

# Caminho onde o modelo e o tokenizer foram salvos
model_path = "caminho/para/o/modelo/salvo"

# Carregar o processador de imagens
image_processor = AutoImageProcessor.from_pretrained(model_path)

# Carregar o modelo de detecção de objetos
model = AutoModelForObjectDetection.from_pretrained(model_path)

# Mover o modelo para a GPU se disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
"""


# with torch.no_grad():
#     logits = model_finetuned(**inputs).logits

# Evaluate
# metrics = trainer.evaluate(eval_dataset=ds["test"], metric_key_prefix="test")
# pprint(metrics)

# INFERENCE
# url = "https://images.pexels.com/photos/8413299/pexels-photo-8413299.jpeg?auto=compress&cs=tinysrgb&w=630&h=375&dpr=2"
# image = Image.open(requests.get(url, stream=True).raw)

# device = "cuda"
# model_repo = "qubvel-hf/detr_finetuned_cppe5"

# image_processor = AutoImageProcessor.from_pretrained(model_repo)
# model = AutoModelForObjectDetection.from_pretrained(model_repo)
# model = model.to(device)

# with torch.no_grad():
#     inputs = image_processor(images=[image], return_tensors="pt")
#     outputs = model(**inputs.to(device))
#     target_sizes = torch.tensor([[image.size[1], image.size[0]]])
#     results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]

# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     print(
#         f"Detected {model.config.id2label[label.item()]} with confidence "
#         f"{round(score.item(), 3)} at location {box}"
#     )

# draw = ImageDraw.Draw(image)

# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     x, y, x2, y2 = tuple(box)
#     draw.rectangle((x, y, x2, y2), outline="red", width=1)
#     draw.text((x, y), model.config.id2label[label.item()], fill="white")

# image
