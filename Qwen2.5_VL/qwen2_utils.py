import os
import pandas as pd
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
import shutil
import supervision as sv

from supervision.geometry.core import Position
from supervision.metrics import MeanAveragePrecision, MetricTarget

from functools import partial
from maestro.trainer.common.datasets.coco import COCODataset, COCOVLMAdapter
from maestro.trainer.models.qwen_2_5_vl.detection import detections_to_prefix_formatter, detections_to_suffix_formatter
from maestro.trainer.models.qwen_2_5_vl.inference import predict, predict_with_inputs

import supervision as sv
from qwen_vl_utils import smart_resize

DEVICE = torch.device('cuda')

def create_train_directory(dynamic_dir, training_set_path, num_non_bg_image, num_bg_image, type):
    # type : 'png' ot 'tif', GMS imagery have 'png' type while Planet imagery have 'tif' type.

    all_images = sorted(glob(f'{training_set_path}/images/*'))
    train_labels = sorted(glob(f'{training_set_path}/labels/*'))
    non_bg_images = sorted([os.path.basename(train_label)[:-4]+f'.{type}' for train_label in train_labels])
    if num_bg_image:
        bg_images = sorted([os.path.basename(image_name) for image_name in all_images if os.path.basename(image_name) not in non_bg_images])

    dynamic_imagesdir = f'{dynamic_dir}/train'
    source_path = f"{training_set_path}/images"

    if os.path.exists(dynamic_imagesdir):
        shutil.rmtree(dynamic_imagesdir)

    if not os.path.exists(dynamic_imagesdir):
        os.makedirs(dynamic_imagesdir)

    # creating symlink to non background images
    for image_name in sorted(non_bg_images)[:num_non_bg_image]:
        source = os.path.join(source_path, image_name) 
        destination = os.path.join(dynamic_imagesdir, image_name)
        os.symlink(src = source, dst = destination)

    # creating symlink to background images
    if num_bg_image:
        for image_name in sorted(bg_images)[:num_bg_image]:
            source = os.path.join(source_path, image_name) 
            destination = os.path.join(dynamic_imagesdir, image_name)
            os.symlink(src = source, dst = destination)

    # symlink annotations.coco.json
    os.symlink(f'{training_set_path}/_annotations.coco.json',
           f'{dynamic_dir}/train/_annotations.coco.json')

def create_test_directory(new_test_path, test_set_path):
    all_images = sorted(glob(f'{test_set_path}/images/*'))

    imagesdir = f'{new_test_path}/test'

    if os.path.exists(imagesdir):
        shutil.rmtree(imagesdir)

    if not os.path.exists(imagesdir):
        os.makedirs(imagesdir)

    # creating symlink to images
    for image_path in sorted(all_images):
        image_name = os.path.basename(image_path)
        destination = os.path.join(imagesdir, image_name)
        os.symlink(src = image_path, dst = destination)
    
    # symlink annotations.coco.json
    os.symlink(f'{test_set_path}/_annotations.coco.json',
           f'{new_test_path}/test/_annotations.coco.json')
    
def create_valid_directory(new_test_path, test_set_path):
    all_images = sorted(glob(f'{test_set_path}/images/*'))

    imagesdir = f'{new_test_path}/valid'

    if os.path.exists(imagesdir):
        shutil.rmtree(imagesdir)

    if not os.path.exists(imagesdir):
        os.makedirs(imagesdir)

    # creating symlink to images
    for image_path in sorted(all_images):
        image_name = os.path.basename(image_path)
        destination = os.path.join(imagesdir, image_name)
        os.symlink(src = image_path, dst = destination)
    
    # symlink annotations.coco.json
    os.symlink(f'{test_set_path}/_annotations.coco.json',
           f'{new_test_path}/valid/_annotations.coco.json')

def sample_test(model, processor, idx, directory, MIN_PIXELS, MAX_PIXELS):

    coco_dataset = COCODataset(
        annotations_path = f'{directory}/_annotations.coco.json', 
        images_directory_path = directory
    )

    qwen2_5_dataset = COCOVLMAdapter(
        coco_dataset=coco_dataset,
        prefix_formatter=detections_to_prefix_formatter,
        suffix_formatter=partial(detections_to_suffix_formatter, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS),
    )
    image, entry = qwen2_5_dataset[idx]

    generated_suffix = predict(model=model, processor=processor, image=image, prefix=entry["prefix"], system_message=None)

    image_w, image_h = image.size
    input_h, input_w = smart_resize(height=image_h, width=image_w, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    predictions = sv.Detections.from_vlm(
        vlm=sv.VLM.QWEN_2_5_VL,
        result=generated_suffix,
        input_wh=(input_w, input_h),
        resolution_wh=(image_w, image_h),
        classes=coco_dataset.classes,
    )

    annotated_frame = image.copy()

    color_annotator = sv.ColorAnnotator(opacity=0.3)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

    annotated_frame = color_annotator.annotate(scene=annotated_frame, detections=predictions)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=predictions)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=predictions)

    annotated_frame

    return annotated_frame

def create_class_mapping(test_dataset, is_dota_dataset):

    if is_dota_dataset:
        class_mapping = {
                        'airplane': 'plane',
                        'baseballdiamond': 'baseball-diamond',
                        'basebase-ball-field': 'baseball-diamond',
                        'basediamond': 'baseball-diamond',
                        'basketball-track-court': 'basketball-court',
                        'large-tankvehicle': 'large-vehicle',
                        'small-track-field': 'ground-track-field',
                        'soccer-ball': 'soccer-ball-field',
                        'soccer-field':'soccer-ball-field',
                        'ten-basketball-court': 'tennis-court',
                        'ten-court': 'tennis-court',
                        'tenniscourt': 'tennis-court',
                        'tennis court': 'tennis-court',
                        'tennis-court-court': 'tennis-court',
                        'tennis racket': 'tennis-court',
                        'tennis tennis-court': 'tennis-court',
                        'tennis table': 'tennis-court',
                        'tennis net': 'tennis-court', 
                        'tennis -court': 'tennis-court',
                        'vehicle': 'small-vehicle',
                        'land vehicle': 'large-vehicle',
                        'swimming pool': 'swimming-pool',
                        }
    else:
        class_mapping = {'brick kilns with chimney': 'brick kilns with chimney',
                        'brick kilns': 'brick kilns with chimney'}

    if not all([value in test_dataset.classes for value in class_mapping.values()]):
        raise ValueError("All mapped values must be in dataset classes")

    return class_mapping

def add_class_ids_and_confidence(detection, test_dataset, class_mapping):

    if 'class_name' in detection.data:
        detection['class_name'] = list(map(lambda name: class_mapping[name] if name in class_mapping else name, detection['class_name']))

        # remove predicted classes not in the dataset
        detection = detection[np.isin(detection['class_name'], test_dataset.classes)]

        # remap Class IDs based on Class names
        detection.class_id = np.array([test_dataset.classes.index(name) for name in detection['class_name']])

        # add confidence, without confidence confusion matrix cannot be computed
        detection.confidence = np.ones(shape=(len(detection.xyxy)))
    
    return detection

def evaluate_finetuned_qwen2_5_vl_model(model, processor, test_loader, class_mapping):

    predictions_list = []
    targets_list = []
    images_list = []

    for (input_ids, attention_mask, pixel_values, image_grid_thw, images, prefixes, suffixes) in tqdm(test_loader):
        image_grid_thw_cpu = image_grid_thw.cpu()
        generated_suffixes = predict_with_inputs(
        model=model,
        processor=processor,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        device=DEVICE,
        )

        for i, image in enumerate(images):
            images_list.append(image)
            image_w, image_h = image.size
            input_h = image_grid_thw_cpu[i][1] * 14
            input_w = image_grid_thw_cpu[i][2] * 14
            print(f'Generated text {generated_suffixes[i]}')
            predictions = sv.Detections.from_vlm(
                vlm=sv.VLM.QWEN_2_5_VL,
                result=generated_suffixes[i],
                input_wh=(input_w, input_h),
                resolution_wh=(image_w, image_h),
            )
            predictions = add_class_ids_and_confidence(predictions, test_loader.dataset.coco_dataset, class_mapping)

            targets = sv.Detections.from_vlm(
                vlm=sv.VLM.QWEN_2_5_VL,
                result=suffixes[i],
                input_wh=(input_w, input_h),
                resolution_wh=(image_w, image_h),
            )
            targets = add_class_ids_and_confidence(targets, test_loader.dataset.coco_dataset, class_mapping)

            predictions_list.append(predictions)
            targets_list.append(targets)

    return predictions_list, targets_list, images_list


def calculate_map(predictions, targets):
    map_metric = MeanAveragePrecision(metric_target=MetricTarget.BOXES)
    map_result = map_metric.update(predictions, targets).compute()
    return map_result, map_result.map50, map_result.map50_95

def calculate_confusion_matrix(predictions, targets, CLASSES, map_result):

    df = pd.DataFrame({}, columns = ['IoU', 'Precision', 'Recall', 'F1 score', 'TP', 'FP', 'FN', 'Kiln instances', 'mAP@50'])
    for iou in [0.1,0.3,0.5,0.7]:
        confusion_matrix = sv.ConfusionMatrix.from_detections(
            predictions=predictions,
            targets=targets,
            classes=CLASSES,
            conf_threshold = 0.25, # paligemma2 does not give confidence score so no use of conf_threshold  
            iou_threshold=iou
        )

        # calculate precision recall and f1-score
        cm = confusion_matrix.matrix
        tp = cm.diagonal().sum() - cm[-1][-1]
        predicted_positives = cm[:,:-1].sum()
        actual_positives = cm[:-1, :].sum()
        precision = tp/ (predicted_positives + 1e-9)
        recall = tp/ (actual_positives + 1e-9)
        f1_score = 2*precision*recall / (precision + recall + 1e-9)
        false_positives = predicted_positives - tp

        df = pd.concat([df, pd.DataFrame({'IoU': iou, 
                                          'Precision': precision, 
                                          'Recall': recall, 
                                          'F1 score': f1_score, 
                                          'TP': tp, 
                                          'FP': false_positives, 
                                          'FN': actual_positives - tp, 
                                          'Kiln instances': actual_positives, 
                                          'mAP@50': map_result.map50}, index = [0])])

    print(f'\n\nPlot of Confusion matrix at IoU {iou}')
    _ = confusion_matrix.plot()

    return df

def visualize_predictions(predictions_list, targets_list, images_list, start, end, rows, cols):

    annotated_images = []

    for i in range(start, end):
        image = images_list[i]
        annotated_image = image.copy()
        annotated_image = sv.BoxAnnotator(thickness = 2, color=sv.Color(r=255, g=0, b=0), color_lookup=sv.ColorLookup.INDEX).annotate(
            scene=annotated_image, detections=predictions_list[i]
        )
        annotated_image = sv.BoxAnnotator(thickness = 4, color=sv.Color(r=0, g=255, b=0), color_lookup=sv.ColorLookup.INDEX).annotate(
            scene=annotated_image, detections=targets_list[i]
        )
        # annotated_image = sv.LabelAnnotator(text_scale=1, text_thickness=1, text_position= Position.TOP_CENTER ,smart_position=True, color_lookup=sv.ColorLookup.INDEX).annotate(
        #     scene=annotated_image, detections=predictions_list[i]
        # )
        
        annotated_images.append(annotated_image)

    sv.plot_images_grid(annotated_images, (rows, cols))