import os
id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(id)

import gc

from maestro.trainer.models.florence_2.core import train, create_data_loaders
from maestro.trainer.models.florence_2.checkpoints import load_model
from maestro.trainer.models.florence_2.loaders import evaluation_collate_fn, train_collate_fn
from maestro.trainer.models.florence_2.detection import (
    detections_to_prefix_formatter,
    detections_to_suffix_formatter,
)

import supervision as sv
from functools import partial

from florence2_utils import *


#### Set Parameters before training ===========================================================================================
want_to_train = False
want_to_infer = True

dynamic_dir = './dynamic_lucknow_coco_train_test'
training_set_path = "/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/lucknow_small_600_sq_km/kiln_images/train"

num_bg_image = 0
type = 'png'

epochs = 50
MODEL_ID = "microsoft/Florence-2-large-ft"

advanced_params = {
    # "r": 16,
    # "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    # "task_type": "CAUSAL_LM"
    "r":8,
    "lora_alpha":8,
    "lora_dropout":0.05,
    "inference_mode":False,
    "use_rslora":True,
    "init_lora_weights":"gaussian",
}

#### Training =================================================================================================================
if want_to_train:
    with pd.ExcelWriter(f'Florence2_{region}.xlsx', engine = 'xlsxwriter') as writer:
        row = 0
        for num_non_bg_image in [4]:
            create_train_directory(dynamic_dir, training_set_path, num_non_bg_image, num_bg_image, type)

            config = {
                "model_id": f"{MODEL_ID}",
                "revision": "refs/heads/main",
                "dataset": f'{dynamic_dir}',
                "epochs": epochs,
                "lr": 5e-6,
                "batch_size": 2,
                "val_batch_size": 4,
                "accumulate_grad_batches": 1,
                "num_workers": 10,
                "optimization_strategy": "lora",
                "metrics": ["edit_distance", "mean_average_precision"],
                "peft_advanced_params": advanced_params,
                "device": 'cuda',
                "max_new_tokens": 1024,
                "output_dir": 'gms_training',
                "log_every_n_steps" : 4,
                }
            
            train(config)

            processor, model = load_model(
                model_id_or_path=f"./gms_training/1/checkpoints/latest",
                revision = "refs/heads/main",
                )
            
            os.rename("./gms_training/1", f"./gms_training/wb_{epochs}_epochs_{num_non_bg_image}_non_bg_images")
            print(f'Checkpoint saved at ./gms_training/{epochs}_epochs_{num_non_bg_image}_non_bg_images')

            train_loader, valid_loader, test_loader = create_data_loaders(
                            dataset_location = f'{dynamic_dir}',
                            train_batch_size= 32,
                            train_collect_fn= partial(train_collate_fn, processor=processor),
                            test_batch_size= 4,
                            test_collect_fn= partial(evaluation_collate_fn, processor=processor),
                            detections_to_prefix_formatter=detections_to_prefix_formatter,
                            detections_to_suffix_formatter=detections_to_suffix_formatter,
                            )
            
            CLASSES = test_loader.dataset.coco_dataset.classes
            test_dataset = test_loader.dataset.coco_dataset
            class_mapping = create_class_mapping(test_dataset, is_dota_dataset = False)
            predictions, targets, images_list = evaluate_finetuned_florence2_model(model, processor, test_loader, class_mapping)
            map_result, map_result_50, map_result_50_95 = calculate_map(predictions, targets)
            print(f'Metrics from model trained on {num_non_bg_image} kiln images and {num_bg_image} background images')
            df = calculate_confusion_matrix(predictions, targets, CLASSES, map_result)
            df.index = [f'{param}B_{num_non_bg_image}_non_bg_{num_bg_image}_bg']*len(df)
            df.to_excel(writer, sheet_name="Sheet1", startrow = row)
            row += len(df) + 2
            del model
            torch.cuda.empty_cache()
            gc.collect()
            # visualize_predictions(images_list, predictions, targets, start=0, end=25, rows=5, cols=5)

####Inference =================================================================================================================
if want_to_infer:
    dynamic_dir = './dynamic_lucknow_coco_train_test'
    test_both_kilns_background_images = True
    train_region = 'west_bengal' # lucknow, west_bengal
    test_region = 'west_bengal' # lucknow, west_bengal

    wb_model_checkpoints = [
                    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/wb_50_epochs_1_non_bg_images/checkpoints/latest',
                    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/wb_50_epochs_2_non_bg_images/checkpoints/latest',
                    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/wb_50_epochs_3_non_bg_images/checkpoints/latest',
                    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/wb_50_epochs_4_non_bg_images/checkpoints/latest',
                    '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/wb_50_epochs_5_non_bg_images/checkpoints/latest',
                    '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/wb_50_epochs_7_non_bg_images/checkpoints/latest',
                    '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/wb_50_epochs_10_non_bg_images/checkpoints/latest',
                    '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/wb_50_epochs_20_non_bg_images/checkpoints/latest',
                    '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/wb_50_epochs_30_non_bg_images/checkpoints/latest',
                    '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/wb_50_epochs_40_non_bg_images/checkpoints/latest',
                    '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/wb_50_epochs_60_non_bg_images/checkpoints/latest',
                    '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/wb_50_epochs_98_non_bg_images/checkpoints/latest',
                    ]

    lucknow_model_checkpoints = [
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/50_epochs_5_non_bg_images/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/50_epochs_7_non_bg_images/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/50_epochs_10_non_bg_images/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/50_epochs_20_non_bg_images/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/50_epochs_30_non_bg_images/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/50_epochs_40_non_bg_images/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/50_epochs_60_non_bg_images/checkpoints/latest',
        # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/gms_training/50_epochs_98_non_bg_images/checkpoints/latest',
    ]

    if train_region == 'lucknow':
        checkpoints = lucknow_model_checkpoints
    if train_region == 'west_bengal':
        checkpoints = wb_model_checkpoints

    if test_region == 'lucknow':
        if test_both_kilns_background_images:
            test_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/lucknow_small_600_sq_km/kiln_images/test_with_background_images'
            create_test_directory(dynamic_dir, test_set_path)
        else:
            test_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/lucknow_small_600_sq_km/kiln_images/test'
            create_test_directory(dynamic_dir, test_set_path)

    if test_region == 'west_bengal':
        if test_both_kilns_background_images:
            test_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/west_bengal_small_639_sq_km/kiln_images/test_with_background_images'
            create_test_directory(dynamic_dir, test_set_path)
        else:
            test_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/west_bengal_small_639_sq_km/kiln_images/test'
            create_test_directory(dynamic_dir, test_set_path)

    with pd.ExcelWriter(f'Florence2_train_{train_region}_test_{test_region}.xlsx', engine = 'xlsxwriter') as writer:
        row = 0
        for model_checkpoint in checkpoints:

            processor, model = load_model(
                model_id_or_path=model_checkpoint,
                revision = "refs/heads/main",
                )

            train_loader, valid_loader, test_loader = create_data_loaders(
                            dataset_location = f'{dynamic_dir}',
                            train_batch_size= 32,
                            train_collect_fn= partial(train_collate_fn, processor=processor),
                            test_batch_size= 4,
                            test_collect_fn= partial(evaluation_collate_fn, processor=processor),
                            detections_to_prefix_formatter=detections_to_prefix_formatter,
                            detections_to_suffix_formatter=detections_to_suffix_formatter,
                            )
            
            CLASSES = test_loader.dataset.coco_dataset.classes
            test_dataset = test_loader.dataset.coco_dataset
            class_mapping = create_class_mapping(test_dataset, is_dota_dataset = False)
            predictions, targets, images_list = evaluate_finetuned_florence2_model(model, processor, test_loader, class_mapping)
            map_result, map_result_50, map_result_50_95 = calculate_map(predictions, targets)
            print(model_checkpoint)
            df = calculate_confusion_matrix(predictions, targets, CLASSES, map_result)
            df.index = [model_checkpoint]*len(df)
            df.to_excel(writer, sheet_name="Sheet1", startrow = row)
            row += len(df) + 2
            del model
            torch.cuda.empty_cache()
            gc.collect()