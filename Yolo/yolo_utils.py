import os
import shutil
from glob import glob
import numpy as np

from ultralytics import YOLO
import supervision as sv
from tempfile import mkdtemp
import pandas as pd

from supervision.metrics import MeanAveragePrecision, MetricTarget

def create_train_directory(dynamic_dir, training_set_path, num_non_bg_image, num_bg_image, type):
    all_images = sorted(glob(f'{training_set_path}/images/*'))
    train_labels = sorted(glob(f'{training_set_path}/labels/*'))
    non_bg_images = sorted([os.path.basename(train_label)[:-4]+f'.{type}' for train_label in train_labels])
    if num_bg_image:
        bg_images = sorted([os.path.basename(image_name) for image_name in all_images if os.path.basename(image_name) not in non_bg_images])

    dynamic_imagesdir = f'{dynamic_dir}/images'
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

    dynamic_labelsdir = f'{dynamic_dir}/labels'
    source_path = f"{training_set_path}/labels"

    if os.path.exists(dynamic_labelsdir):
        shutil.rmtree(dynamic_labelsdir)

    if not os.path.exists(dynamic_labelsdir):
        os.makedirs(dynamic_labelsdir)

    # creating symlink to non background labels
    for image_name in sorted(non_bg_images)[:num_non_bg_image]:
        source = os.path.join(source_path, image_name[:-4]+'.txt') 
        destination = os.path.join(dynamic_labelsdir, image_name[:-4]+'.txt')
        os.symlink(src = source, dst = destination)

def create_yaml_file(yaml_path, dynamic_dir, val_path):
    data_yml = f"""
train: {dynamic_dir}
val: {val_path}
nc: 1
names: ["brick kilns with chimney"]
    """
    data_yml_path = f"{yaml_path}"
    with open(data_yml_path, "w") as f:
        f.write(data_yml)

def train_yolo(model_id, yaml_path, epochs, image_size, batch_size, checkpoint_dir):
    # Load a pretrained model
    model = YOLO(model_id)

    # Train the model on your custom dataset
    model.train(data=yaml_path, epochs=epochs, batch = batch_size, imgsz=image_size, name = checkpoint_dir, save_conf=True, save_txt=True, val = False)

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
                        'brick kilns': 'brick kilns with chimney'}

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
    
    return detection

def inference(model_path, sv_dataset, is_dota_dataset):

    CLASSES = sv_dataset.classes
    class_mapping = create_class_mapping(CLASSES, is_dota_dataset)

    model = YOLO(model_path)

    targets = []
    predictions = []
    images = []

    for classes, image, gt_detection in sv_dataset:
        result = model.predict(image, conf = 0.25)
        detections = sv.Detections.from_ultralytics(result[0])
        detections = add_class_ids(detections, CLASSES, class_mapping)
        
        targets.append(gt_detection)
        predictions.append(detections)
        images.append(image)

    return targets, predictions, images

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