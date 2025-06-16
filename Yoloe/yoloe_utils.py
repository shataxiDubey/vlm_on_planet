from supervision.geometry.core import Position
from supervision.metrics import MeanAveragePrecision, MetricTarget
import supervision as sv
import pandas as pd
import numpy as np
from glob import glob
import os, shutil
from PIL import Image
import torch
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor


def create_train_directory(dynamic_dir, training_set_path, num_non_bg_image, num_bg_image, type):

    all_images = sorted(glob(f'{training_set_path}/images/*'))
    train_labels = sorted(glob(f'{training_set_path}/labels/*'))
    non_bg_images = sorted([os.path.basename(train_label)[:-4]+f'.{type}' for train_label in train_labels])

    if num_bg_image:
        bg_images = sorted([os.path.basename(image_name) for image_name in all_images if os.path.basename(image_name) not in non_bg_images])

    dynamic_imagesdir = f'{dynamic_dir}/images'
    source_path = f'{training_set_path}/images'

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
            destination = os.path.join(dynamic_dir, image_name)
            os.symlink(src = source, dst = destination)

    dynamic_labelsdir = f'{dynamic_dir}/labels'
    source_path = f'{training_set_path}/labels'

    if os.path.exists(dynamic_labelsdir):
        shutil.rmtree(dynamic_labelsdir)

    if not os.path.exists(dynamic_labelsdir):
        os.makedirs(dynamic_labelsdir)

    # creating symlink to non background labels
    for image_name in sorted(non_bg_images)[:num_non_bg_image]:
        source = os.path.join(source_path, image_name[:-4]+'.txt') 
        destination = os.path.join(dynamic_labelsdir, image_name[:-4]+'.txt')
        os.symlink(src = source, dst = destination)

def visual_prompting(model_id, source_image_paths, label_dir):

    source_images = []
    boxes = []
    classes = []
    NAMES = ['brick kilns with chimney']

    for SOURCE_IMAGE_PATH in source_image_paths:
        src_img_name = os.path.basename(SOURCE_IMAGE_PATH)
        img = Image.open(SOURCE_IMAGE_PATH)
        size = img.size[0]

        label_path = f'{label_dir}/{src_img_name[:-4]}.txt'
        if os.path.exists(label_path):
            bboxes = np.loadtxt(label_path, ndmin = 2)
            cls = bboxes[:, 0]
            # print(cls)
            xmin = np.min(bboxes[:,[1,3,5,7]], axis = 1)
            xmax = np.max(bboxes[:,[1,3,5,7]], axis = 1)
            ymin = np.min(bboxes[:,[2,4,6,8]], axis = 1)
            ymax = np.max(bboxes[:,[2,4,6,8]], axis = 1)

            bboxes = np.array([xmin, ymin, xmax, ymax])
            bboxes = bboxes.T
            bboxes = bboxes * size
            # print(bboxes)
            cls = np.array([0 for _ in cls], dtype=np.int32)
            source_images.append(img)
            boxes.append(bboxes)
            classes.append(cls)
        

    model = YOLOE(model_id).cuda()
    prompts = dict(bboxes=boxes, cls=classes) 
    model.predict(source_images, prompts=prompts, predictor=YOLOEVPSegPredictor, return_vpe = True)
    model.set_classes(NAMES, torch.nn.functional.normalize(model.predictor.vpe.mean(dim=0, keepdim=True), dim=-1, p=2)) # while passing k multiple visual prompts, mean of k embeddings is taken and then normalized 
    model.predictor = None

    return model, prompts


def create_class_mapping(CLASSES, is_dota_dataset):

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
                        'brick kilns': 'brick kilns with chimney'
                        }

    if not all([value in CLASSES for value in class_mapping.values()]):
        raise ValueError("All mapped values must be in dataset classes")

    return class_mapping

def add_class_ids(detection, CLASSES, class_mapping):

    if 'class_name' in detection.data:
        detection['class_name'] = list(map(lambda name: class_mapping[name] if name in class_mapping else name, detection['class_name']))

        # remove predicted classes not in the dataset
        detection = detection[np.isin(detection['class_name'], CLASSES)]

        # remap Class IDs based on Class names
        detection.class_id = np.array([CLASSES.index(name) for name in detection['class_name']])

        # add confidence, without confidence confusion matrix cannot be computed, YOLOe gives confidence scores
        # detection.confidence = np.ones(shape=(len(detection.xyxy)))
    
    return detection

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
            conf_threshold = 0.25,  
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


def visualize_predictions(images, predictions, targets, start, end, rows, cols):

    annotated_images = []

    for i in range(start, end):
        image = images[i]
        detections = predictions[i]
        target = targets[i]
        annotated_image = image.copy()
        annotated_image = sv.BoxAnnotator(thickness=8, color=sv.Color(r=255, g=0, b=0)).annotate(annotated_image, detections)
        annotated_image = sv.BoxAnnotator(thickness=4, color=sv.Color(r=0, g=255, b=0)).annotate(annotated_image, target)
        annotated_image = sv.LabelAnnotator(text_scale=1, text_thickness=1, smart_position=True).annotate(annotated_image, detections)
        annotated_images.append(annotated_image)

    sv.plot_images_grid(annotated_images, (rows, cols))