import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from glob import glob
import torch
import supervision as sv
import json
import shutil
import re
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from math import ceil

from supervision.metrics import MeanAveragePrecision, MetricTarget

from transformers import BitsAndBytesConfig, PaliGemmaProcessor, PaliGemmaForConditionalGeneration, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_train_directory(dynamic_dir, training_set_path, num_non_bg_image, num_bg_image, type):

    # planet imagery
    # all_images = sorted(glob('/home/shataxi.dubey/shataxi_work/vlm_on_planet/lucknow_train_test_split/train/images/*'))
    # train_labels = sorted(glob('/home/shataxi.dubey/shataxi_work/vlm_on_planet/lucknow_train_test_split/train/labels/*'))
    # non_bg_images = sorted([os.path.basename(train_label)[:-4]+'.tif' for train_label in train_labels])
    # bg_images = sorted([os.path.basename(image_name) for image_name in all_images if os.path.basename(image_name) not in non_bg_images])

    # gms imagery
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
    non_bg_images = np.random.choice(non_bg_images, size = num_non_bg_image)
    for image_name in non_bg_images:
        source = os.path.join(source_path, image_name) 
        destination = os.path.join(dynamic_imagesdir, image_name)
        os.symlink(src = source, dst = destination)

    # creating symlink to background images
    if num_bg_image:
        bg_images = np.random.choice(bg_images, size = num_bg_image)
        for image_name in bg_images:
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
    
    total_train_images = len(glob(f'{dynamic_imagesdir}/*'))
    print(f'Total number of images in the training set {total_train_images}')

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self):
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path)
        return image, entry
    
def create_jsonl_dataset(dynamic_dir, validation_set_path, test_set_path):

    train_dataset = JSONLDataset(
        jsonl_file_path=f'{dynamic_dir}/paligemma2_annotations.jsonl',
        image_directory_path=f"{dynamic_dir}/images",
    )
    valid_dataset = JSONLDataset(
        jsonl_file_path=f"{validation_set_path}/paligemma2_annotations.jsonl",
        image_directory_path=f"{validation_set_path}/images",
    )
    test_dataset = JSONLDataset(
        jsonl_file_path=f"{test_set_path}/paligemma2_annotations.jsonl",
        image_directory_path=f"{test_set_path}/images",
    )

    return train_dataset, valid_dataset, test_dataset

def parse_bbox_and_labels(detokenized_output: str):
    matches = re.finditer(
        '<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)>'
        ' (?P<label>(\w+\s?)+\w)',
        detokenized_output,
    )
    labels, boxes = [], []
    fmt = lambda x: float(x) / 1024.0
    for m in matches:
        d = m.groupdict()
        # print(d)
        boxes.append([fmt(d['y0']), fmt(d['x0']), fmt(d['y1']), fmt(d['x1'])])
        labels.append(d['label'])
    return np.array(boxes), np.array(labels)

def num_kilns_in_dataset(train_dataset):
    sum = 0
    for i in range(len(train_dataset)):
        box, label = parse_bbox_and_labels(train_dataset.entries[i]['suffix'])
        for classname in label:
            if 'brick kilns with chimney' in classname:
                sum += 1
    print(f'Total number of kilns in the dataset {sum}')


def load_model(MODEL_ID, model_checkpoint):

    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)

    if model_checkpoint != '':
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_checkpoint, device_map="auto")
        TORCH_DTYPE = model.dtype
        return model, processor, TORCH_DTYPE

    # Fine-tune the entire model with LoRA and QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto", quantization_config= bnb_config, torch_dtype= torch.bfloat16)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    TORCH_DTYPE = model.dtype

    return model, processor, TORCH_DTYPE

def set_trainer(model, processor, train_dataset, valid_dataset, num_train_epochs, batch_size, gradient_update_steps, output_dir, TORCH_DTYPE):
        
    def augment_suffix(suffix):
        parts = suffix.split(' ; ')
        random.shuffle(parts)
        return ' ; '.join(parts)


    def collate_fn(batch):
        images, labels = zip(*batch)

        paths = [label["image"] for label in labels]
        prefixes = ["<image>" + label["prefix"] for label in labels]
        suffixes = [augment_suffix(label["suffix"]) for label in labels]

        inputs = processor(
            text=prefixes,
            images=images,
            return_tensors="pt",
            suffix=suffixes,
            padding="longest"
        ).to(TORCH_DTYPE).to(DEVICE)

        return inputs
        
    steps = ceil(len(train_dataset) / (batch_size * gradient_update_steps)) * num_train_epochs
    # Note: trainer uses max_steps, it calculates the num_train_epochs automatically when max_steps is provided explicitly. num_train_epochs = (max_steps // updates_per_epoch) 
    # Refer: https://github.com/huggingface/transformers/blob/772307be7649e1333a933cfaa229dc0dec2fd331/src/transformers/trainer.py#L1585
    args = TrainingArguments(
        # num_train_epochs=100,
        max_steps = steps,
        remove_unused_columns=False,
        per_device_train_batch_size= batch_size,
        gradient_accumulation_steps= gradient_update_steps,
        seed = 2,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=1, # weights and biases logging step
        logging_strategy = 'epoch',
        optim="adamw_hf",
        save_strategy="steps", # checkpoint is saved after steps
        save_steps=1000,  # checkpoint is saved after every 1000 steps
        save_total_limit=1,
        output_dir = output_dir, # model predictions and checkpoint will be present
        bf16=True,
        report_to="none", # disconnects wandb
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        args=args
    )

    return trainer, steps


def sample_test(dataset, model, processor, CLASSES, TORCH_DTYPE):
    image, label = dataset[0]
    prefix = "<image>" + label["prefix"]
    suffix = label["suffix"]
    print(f'Suffix {suffix}')
    inputs = processor(
        text=prefix,
        images=image,
        return_tensors="pt"
    ).to(TORCH_DTYPE).to(DEVICE)

    prefix_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generation = generation[0][prefix_length:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        print(f'Predicted {decoded}')

    w, h = image.size
    detections = sv.Detections.from_vlm(
        vlm='paligemma',
        result=decoded,
        resolution_wh=(w, h),
        classes=CLASSES)

    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator(thickness = 4, color = sv.Color(r=255, g= 0, b= 0)).annotate(annotated_image, detections)
    # annotated_image = sv.LabelAnnotator(smart_position=True).annotate(annotated_image, detections)
    detections = sv.Detections.from_vlm(
        vlm='paligemma',
        result=suffix,
        resolution_wh=(w, h),
        classes=CLASSES)
    annotated_image = sv.BoxAnnotator(thickness = 2, color = sv.Color(r=0, g= 255, b= 0)).annotate(annotated_image, detections)
    return annotated_image


def evaluate_finetuned_paligemma_model(dataset, processor, model, CLASSES, TORCH_DTYPE):

    def collate_test_fn(batch):
        images, labels = zip(*batch)

        prefixes = ["<image>" + label["prefix"] for label in labels]
        suffixes = [label["suffix"] for label in labels]
        inputs = processor(
            text=prefixes,
            images=images,
            return_tensors="pt",
            padding="longest"
        ).to(TORCH_DTYPE).to(DEVICE)

        return images, inputs, suffixes

    test_dataloader = DataLoader(dataset, batch_size=4, collate_fn= collate_test_fn, shuffle=False)

    images = []
    targets = []
    predictions = []

    with torch.inference_mode():
        for imgs, test_inputs, suffixes in tqdm(test_dataloader):
            # print(test_inputs['input_ids'])
            prefix_length = test_inputs["input_ids"].shape[-1]

            generation = model.generate(**test_inputs, max_new_tokens=256, do_sample=False)
            generation = generation[:, prefix_length:]
            generated_texts = processor.batch_decode(generation, skip_special_tokens=True)
            w, h = imgs[0].size
            for generated_text in generated_texts:
                prediction = sv.Detections.from_vlm(
                    vlm='paligemma',
                    result=generated_text,
                    resolution_wh=(w, h),
                    # classes=CLASSES
                    )

                prediction.class_id = np.array([CLASSES.index(class_name) for class_name in prediction['class_name']])
                prediction.confidence = np.ones(len(prediction))
                predictions.append(prediction)

            for suffix in suffixes:
                target = sv.Detections.from_vlm(
                    vlm='paligemma',
                    result=suffix,
                    resolution_wh=(w, h),
                    # classes=CLASSES
                    )

                target.class_id = np.array([CLASSES.index(class_name) for class_name in target['class_name']])
                targets.append(target)
            images += list(imgs)
    
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