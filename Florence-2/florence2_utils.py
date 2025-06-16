import os
import pandas as pd
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
import shutil
import supervision as sv

from maestro.trainer.models.florence_2.inference import predict_with_inputs
from maestro.trainer.models.florence_2.inference import predict

from supervision.geometry.core import Position
from supervision.metrics import MeanAveragePrecision, MetricTarget

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

def evaluate_finetuned_florence2_model(model, processor, test_loader, class_mapping):
    predictions = []
    targets = []
    images_list = []

    for input_ids, pixel_values, images, prefixes, suffixes in tqdm(test_loader):
        generated_texts = predict_with_inputs(model=model, processor=processor, input_ids=input_ids, pixel_values=pixel_values, device = DEVICE)

        for idx, generated_text in enumerate(generated_texts):
            predicted_result = processor.post_process_generation(text=generated_text, task="<OD>", image_size=(images[idx].width, images[idx].height))

            predicted_result = sv.Detections.from_vlm(vlm='florence_2',
                    result=predicted_result,
                    resolution_wh=(images[idx].width, images[idx].height))

            predicted_result = add_class_ids_and_confidence(predicted_result, test_loader.dataset.coco_dataset, class_mapping)
            predictions.append(predicted_result)

        for idx, suffix in enumerate(suffixes):
            target_result = processor.post_process_generation(text=suffix, task="<OD>", image_size=(images[idx].width, images[idx].height))
            target_result = sv.Detections.from_vlm(vlm='florence_2',
                    result=target_result,
                    resolution_wh=(images[idx].width, images[idx].height))

            target_result = add_class_ids_and_confidence(target_result, test_loader.dataset.coco_dataset, class_mapping)
            targets.append(target_result)
        images_list += list(images)

    return predictions, targets, images_list


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

# def visualize_predictions(model, processor, test_dataset):

#     annotated_images = []

#     for i in range(0,25):
#         image, annotation = test_dataset[i]
#         generated_text = predict(model=model, processor=processor, image=image, prefix=f"<OD>")
#         result = processor.post_process_generation(text=generated_text, task="<OD>", image_size=(image.width, image.height))
#         detections = sv.Detections.from_vlm(vlm="florence_2", result=result, resolution_wh=image.size)
#         # print(detections['class_name'])
#         annotated_image = image.copy()
#         annotated_image = sv.BoxAnnotator(thickness = 4, color=sv.Color(r=255, g=0, b=0), color_lookup=sv.ColorLookup.INDEX).annotate(
#             scene=annotated_image, detections=detections
#         )
#         annotated_image = sv.BoxAnnotator(thickness = 2, color=sv.Color(r=0, g=255, b=0), color_lookup=sv.ColorLookup.INDEX).annotate(
#             scene=annotated_image, detections=annotation
#         )
#         annotated_image = sv.LabelAnnotator(text_scale=1, text_thickness=0, text_position= Position.TOP_CENTER ,smart_position=True, color_lookup=sv.ColorLookup.INDEX).annotate(
#             scene=annotated_image, detections=detections
#         )

#         annotated_images.append(annotated_image)

#     sv.plot_images_grid(annotated_images, (5,5))


