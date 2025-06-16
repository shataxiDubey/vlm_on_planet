import os, sys
id = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(id)

from qwen2_utils import *
import torch
import gc
import pandas as pd
from functools import partial

from maestro.trainer.models.qwen_2_5_vl.core import train
from maestro.trainer.models.qwen_2_5_vl.checkpoints import load_model
from maestro.trainer.models.qwen_2_5_vl.loaders import evaluation_collate_fn, train_collate_fn
from maestro.trainer.models.qwen_2_5_vl.detection import (
    detections_to_prefix_formatter,
    detections_to_suffix_formatter,
)
from maestro.trainer.common.datasets.core import create_data_loaders
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda')

sys.path.append('/home/shataxi.dubey/shataxi_work/VLM_high_res')

want_to_train = False
want_to_infer = True


#### Set parameters before training

region = 'west_bengal' # west_bengal, lucknow

type = 'png'
epochs = 50
batch_size = 4

param = 3 # 3, 7
model_id = f"Qwen/Qwen2.5-VL-{param}B-Instruct"
SYSTEM_MESSAGE = "You are a helpful assistant."
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

if region == 'lucknow':
    dynamic_dir = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/dynamic_lucknow_coco_train_test'
    training_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/lucknow_small_600_sq_km/kiln_images/train'
    test_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/lucknow_small_600_sq_km/kiln_images/test'
    valid_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/lucknow_small_600_sq_km/kiln_images/valid'

if region == 'west_bengal':
    dynamic_dir = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/dynamic_lucknow_coco_train_test'
    training_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/west_bengal_small_639_sq_km/kiln_images/train'
    test_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/west_bengal_small_639_sq_km/kiln_images/test'
    valid_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/west_bengal_small_639_sq_km/kiln_images/valid'

# ======================================================================================================================
#### training
if want_to_train:
    with pd.ExcelWriter(f'Qwen2_5_VL_{param}B_Instruct.xlsx', engine = 'xlsxwriter') as writer:
        row = 0

        num_bg_image = 0
        for num_non_bg_image in [2]:
            create_train_directory(dynamic_dir, training_set_path, num_non_bg_image, num_bg_image, type)
            create_test_directory(dynamic_dir, test_set_path)
            create_valid_directory(dynamic_dir, test_set_path)

            advanced_params = {
                "r": 8,
            }

            config = {
                "model_id": model_id,
                "dataset": '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/dynamic_lucknow_coco_train_test',
                "system_message": SYSTEM_MESSAGE,
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
                "epochs": epochs,
                "batch_size": batch_size,
                "accumulate_grad_batches": 1,
                # "num_workers": 10,
                "optimization_strategy": "qlora",
                "metrics": ["edit_distance", "mean_average_precision"],
                # "peft_advanced_params": advanced_params,
                "output_dir": 'gms_training',
            }

            train(config)

            # processor, model = load_model(
            #     model_id_or_path=f"./gms_training/1/checkpoints/latest",
            #     min_pixels=MIN_PIXELS,
            #     max_pixels=MAX_PIXELS,
            #     )
            
            os.rename("./gms_training/1", f"./gms_training/{param}B_{region}_{epochs}_epochs_{num_non_bg_image}_non_bg_{num_bg_image}_bg_batch_size_{batch_size}")
            print(f'Checkpoint saved at ./gms_training/{param}B_{region}_{epochs}_epochs_{num_non_bg_image}_non_bg_images')

            # idx = 0
            # directory = dynamic_dir + '/train'
            # annotated_image = sample_test(model, processor, 0, directory, MIN_PIXELS, MAX_PIXELS)
            # plt.imshow(annotated_image)
            # plt.gca().set_axis_off()
            # plt.title(f'one sample from train set')

            train_loader, valid_loader, test_loader = create_data_loaders(
                            dataset_location= dynamic_dir,
                            train_batch_size= 4,
                            train_collect_fn= partial(train_collate_fn, processor=processor, system_message=SYSTEM_MESSAGE),
                            # train_num_workers=10,
                            test_batch_size= 4,
                            test_collect_fn= partial(evaluation_collate_fn, processor=processor, system_message=SYSTEM_MESSAGE),
                            detections_to_prefix_formatter=detections_to_prefix_formatter,
                            detections_to_suffix_formatter=partial(
                                        detections_to_suffix_formatter, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
                                        ),
                            )
            CLASSES = test_loader.dataset.coco_dataset.classes
            test_dataset = test_loader.dataset.coco_dataset
            class_mapping = create_class_mapping(test_dataset, is_dota_dataset = False)

            predictions, targets, images_list = evaluate_finetuned_qwen2_5_vl_model(model, processor, test_loader, class_mapping)
            map_result, map_result_50, map_result_50_95 = calculate_map(predictions, targets)
            print(f'Metrics from model trained on {num_non_bg_image} kiln images and {num_bg_image} background images')
            df = calculate_confusion_matrix(predictions, targets, CLASSES, map_result)
            df.index = [f'{param}B_{num_non_bg_image}_non_bg_{num_bg_image}_bg']*len(df)
            df.to_excel(writer, sheet_name="Sheet1", startrow = row)
            row += len(df) + 2
            del model
            torch.cuda.empty_cache()
            gc.collect()
            # visualize_predictions(predictions, targets, images_list, start=0, end=25, rows=5, cols=5)



# =========================================================================================================================
#### inference

if want_to_infer:
    dynamic_dir = './dynamic_lucknow_coco_train_test'
    test_both_kilns_background_images = False
    train_region = 'west_bengal' # lucknow, west_bengal
    test_region = 'west_bengal' # lucknow, west_bengal
    default = True # using default model checkpoints

    wb_model_checkpoints = [
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_west_bengal_50_epochs_1_non_bg_0_bg_batch_size_4/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_west_bengal_50_epochs_2_non_bg_0_bg_batch_size_4/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_west_bengal_50_epochs_3_non_bg_0_bg_batch_size_4/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_west_bengal_50_epochs_4_non_bg_0_bg_batch_size_4/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_west_bengal_50_epochs_5_non_bg_0_bg_batch_size_4/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_west_bengal_50_epochs_7_non_bg_0_bg_batch_size_4/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_west_bengal_50_epochs_10_non_bg_0_bg_batch_size_4/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_west_bengal_50_epochs_20_non_bg_0_bg_batch_size_4/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_west_bengal_50_epochs_30_non_bg_0_bg_batch_size_4/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_west_bengal_50_epochs_40_non_bg_0_bg_batch_size_4/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_west_bengal_50_epochs_60_non_bg_0_bg_batch_size_4/checkpoints/latest',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_west_bengal_50_epochs_98_non_bg_0_bg_batch_size_4/checkpoints/latest',
        # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_west_bengal_50_epochs_1_non_bg_0_bg_batch_size_4/checkpoints/latest',
        # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_west_bengal_50_epochs_2_non_bg_0_bg_batch_size_4/checkpoints/latest',
        # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_west_bengal_50_epochs_3_non_bg_0_bg_batch_size_4/checkpoints/latest',
        # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_west_bengal_50_epochs_4_non_bg_0_bg_batch_size_4/checkpoints/latest',
        # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_west_bengal_50_epochs_5_non_bg_0_bg_batch_size_4/checkpoints/latest',
        # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_west_bengal_50_epochs_98_non_bg_0_bg_batch_size_4/checkpoints/latest',
                    ]

    lucknow_model_checkpoints = [
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_lucknow_50_epochs_1_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_lucknow_50_epochs_2_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_lucknow_50_epochs_3_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_lucknow_50_epochs_4_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_lucknow_50_epochs_5_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_lucknow_50_epochs_7_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_lucknow_50_epochs_10_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_lucknow_50_epochs_20_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_lucknow_50_epochs_30_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_lucknow_50_epochs_40_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/3B_lucknow_50_epochs_60_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/qwen_2_5_vl/3B_lucknow_50_epochs_98_non_bg_0_bg_batch_size_4/checkpoints/latest',
    '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_lucknow_50_epochs_1_non_bg_0_bg_batch_size_4/checkpoints/latest',
    '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_lucknow_50_epochs_2_non_bg_0_bg_batch_size_4/checkpoints/latest',
    '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_lucknow_50_epochs_3_non_bg_0_bg_batch_size_4/checkpoints/latest',
    '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_lucknow_50_epochs_4_non_bg_0_bg_batch_size_4/checkpoints/latest',
    '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_lucknow_50_epochs_5_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_lucknow_50_epochs_7_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_lucknow_50_epochs_10_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_lucknow_50_epochs_20_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_lucknow_50_epochs_30_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_lucknow_50_epochs_40_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/7B_lucknow_50_epochs_60_non_bg_0_bg_batch_size_4/checkpoints/latest',
    # '/home/shataxi.dubey/shataxi_work/vlm_on_planet/Qwen2.5_VL/gms_training/qwen_2_5_vl/7B_lucknow_50_epochs_98_non_bg_0_bg_batch_size_4/checkpoints/latest',

    ]

    
    if train_region == 'lucknow':
        checkpoints = lucknow_model_checkpoints
    if train_region == 'west_bengal':
        checkpoints = wb_model_checkpoints

    if default:
        checkpoints = [model_id]    

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

    with pd.ExcelWriter(f'Qwen2_5_VL_train_{train_region}_test_{test_region}_{param}B_Instruct.xlsx', engine = 'xlsxwriter') as writer:
        row = 0
        checkpoints.reverse()
        for model_checkpoint in checkpoints:
            print(model_checkpoint)
            processor, model = load_model(
                model_id_or_path= model_checkpoint,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
                )

            train_loader, valid_loader, test_loader = create_data_loaders(
                            dataset_location= dynamic_dir,
                            train_batch_size= 4,
                            train_collect_fn= partial(train_collate_fn, processor=processor, system_message=SYSTEM_MESSAGE),
                            # train_num_workers=10,
                            test_batch_size= 4,
                            test_collect_fn= partial(evaluation_collate_fn, processor=processor, system_message=SYSTEM_MESSAGE),
                            detections_to_prefix_formatter=detections_to_prefix_formatter,
                            detections_to_suffix_formatter=partial(
                    detections_to_suffix_formatter, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
                ),
                            )
            
            CLASSES = test_loader.dataset.coco_dataset.classes
            test_dataset = test_loader.dataset.coco_dataset
            class_mapping = create_class_mapping(test_dataset, is_dota_dataset = False)
            predictions, targets, images_list = evaluate_finetuned_qwen2_5_vl_model(model, processor, test_loader, class_mapping)
            map_result, map_result_50, map_result_50_95 = calculate_map(predictions, targets)
            print(model_checkpoint)
            df = calculate_confusion_matrix(predictions, targets, CLASSES, map_result)
            df.index = [model_checkpoint]*len(df)
            df.to_excel(writer, sheet_name="Sheet1", startrow = row)
            row += len(df) + 2
            # visualize_predictions(images_list, predictions, targets, start=0, end=25, rows=5, cols=5)
            del model
            torch.cuda.empty_cache()
            gc.collect()

