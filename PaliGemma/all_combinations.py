import os, sys
id = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(id)

sys.path.append('/home/shataxi.dubey/shataxi_work/VLM_high_res')
from json_format import *
import gc
import matplotlib.pyplot as plt
from paligemma2_utils import *


#### Set Parameters before training ====================================================================================

want_to_train = True
want_to_infer = False

dynamic_dir = '../dynamic_train'
param = 3
region = "lucknow"

if region == "wb":
    source_training_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/west_bengal_small_639_sq_km/kiln_images/train' 
    validation_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/west_bengal_small_639_sq_km/kiln_images/valid' 
    test_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/west_bengal_small_639_sq_km/kiln_images/test' 
if region == 'lucknow':
    source_training_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/lucknow_small_600_sq_km/kiln_images/train' 
    validation_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/lucknow_small_600_sq_km/kiln_images/valid' 
    test_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/lucknow_small_600_sq_km/kiln_images/test' 
    
label_files_path = f'{dynamic_dir}/labels'
json_file_path = f'{dynamic_dir}/paligemma2_annotations.jsonl'
is_dota_dataset = False
task = 'detect brick kilns with chimney'
model_name = 'paligemma2'
obb = True
image_type = 'png'
annotation_format = 'yolo'


num_train_epochs = 50
batch_size = 4
gradient_update_steps = 8

#### Training ===========================================================================================================
if want_to_train:
    with pd.ExcelWriter(f'PaliGemma_{region}.xlsx', engine = 'xlsxwriter') as writer:
        row = 0

        num_bg_image = 0
        for num_non_bg_image in [5]:
            for iteration in range(5):
                MODEL_ID ="google/paligemma2-3b-pt-448"
                model_checkpoint = ''
                create_train_directory(dynamic_dir, source_training_set_path, num_non_bg_image, num_bg_image, type = 'png')
                print(glob(f'{dynamic_dir}/images/*.png'))
                images_file_path = f'{dynamic_dir}/images'
                create_jsonl_file(images_file_path, label_files_path, json_file_path, is_dota_dataset, task, model_name, obb, image_type, annotation_format)
                train_dataset, valid_dataset, test_dataset = create_jsonl_dataset(dynamic_dir, validation_set_path, test_set_path)
                print(f'train dataset length {len(train_dataset)} test dataset length {len(test_dataset)}')
                num_kilns_in_dataset(train_dataset)
                num_kilns_in_dataset(test_dataset)
                CLASSES = test_dataset[0][1]['prefix'].replace("detect ", "").split(" ; ")
                model, processor, TORCH_DTYPE = load_model(MODEL_ID, model_checkpoint)
                output_dir = f"{region}_paligemma2_object_detection_{num_non_bg_image}_{num_bg_image}_itr_{iteration}_r_8_alpha_8_gms"
                trainer, steps = set_trainer(model, processor, train_dataset, valid_dataset, num_train_epochs, batch_size, gradient_update_steps, output_dir, TORCH_DTYPE)
                trainer.train()

                model_checkpoint = f'/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/{output_dir}/checkpoint-{steps}'
                model, processor, TORCH_DTYPE = load_model(MODEL_ID, model_checkpoint)
                annotated_image = sample_test(train_dataset, model, processor, CLASSES, TORCH_DTYPE)
                plt.imshow(annotated_image)
                plt.gca().set_axis_off()
                plt.title(f'one sample from train set')

                targets, predictions, images = evaluate_finetuned_paligemma_model(test_dataset, processor, model, CLASSES, TORCH_DTYPE)
                map_result, map50, map50_95 = calculate_map(predictions, targets)
                print(f'Metrics from model trained on {num_non_bg_image} kiln images and {num_bg_image} background images')
                df = calculate_confusion_matrix(predictions, targets, CLASSES, map_result)
                df.index = [f'{param}B_{num_non_bg_image}_non_bg_{num_bg_image}_bg_{iteration}_itr']*len(df)
                df.to_excel(writer, sheet_name="Sheet1", startrow = row)
                row += len(df) + 2
                del model
                torch.cuda.empty_cache()
                gc.collect()
                # visualize_predictions(images, predictions, targets, start = 0, end = 25, rows = 5, cols = 5)

#### Inference ===========================================================================================================

if want_to_infer:
    test_both_kilns_background_images = True
    train_region = 'west_bengal' # lucknow, west_bengal
    test_region = 'west_bengal' # lucknow, west_bengal

    lucknow_model_checkpoints = [
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/paligemma2_object_detection_1_0_r_8_alpha_8_gms/checkpoint-50',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/paligemma2_object_detection_2_0_r_8_alpha_8_gms/checkpoint-50',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/paligemma2_object_detection_3_0_r_8_alpha_8_gms/checkpoint-50',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/paligemma2_object_detection_4_0_r_8_alpha_8_gms/checkpoint-50',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/paligemma2_object_detection_5_0_r_8_alpha_8_gms/checkpoint-100',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/paligemma2_object_detection_7_0_r_8_alpha_8_gms/checkpoint-100',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/paligemma2_object_detection_10_0_r_8_alpha_8_gms/checkpoint-150',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/paligemma2_object_detection_20_0_r_8_alpha_8_gms/checkpoint-100',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/paligemma2_object_detection_30_0_r_8_alpha_8_gms/checkpoint-50',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/paligemma2_object_detection_40_0_r_8_alpha_8_gms/checkpoint-100',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/paligemma2_object_detection_60_0_r_8_alpha_8_gms/checkpoint-100',
        '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/paligemma2_object_detection_98_0_r_8_alpha_8_gms/checkpoint-200',
    ]


    wb_model_checkpoints = ['/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/wb_paligemma2_object_detection_1_0_r_8_alpha_8_gms/checkpoint-50',
                            '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/wb_paligemma2_object_detection_2_0_r_8_alpha_8_gms/checkpoint-50',
                            '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/wb_paligemma2_object_detection_3_0_r_8_alpha_8_gms/checkpoint-50',
                            '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/wb_paligemma2_object_detection_4_0_r_8_alpha_8_gms/checkpoint-50',
                            '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/wb_paligemma2_object_detection_5_0_r_8_alpha_8_gms/checkpoint-100',
                            '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/wb_paligemma2_object_detection_7_0_r_8_alpha_8_gms/checkpoint-100',
                            '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/wb_paligemma2_object_detection_10_0_r_8_alpha_8_gms/checkpoint-150',
                            '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/wb_paligemma2_object_detection_20_0_r_8_alpha_8_gms/checkpoint-100',
                            '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/wb_paligemma2_object_detection_30_0_r_8_alpha_8_gms/checkpoint-50',
                            '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/wb_paligemma2_object_detection_40_0_r_8_alpha_8_gms/checkpoint-100',
                            '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/wb_paligemma2_object_detection_60_0_r_8_alpha_8_gms/checkpoint-100',
                            '/home/shataxi.dubey/shataxi_work/vlm_on_planet/PaliGemma/wb_paligemma2_object_detection_98_0_r_8_alpha_8_gms/checkpoint-200',           
    ]   

    if train_region == 'lucknow':
        checkpoints = lucknow_model_checkpoints
    if train_region == 'west_bengal':
        checkpoints = wb_model_checkpoints

    if test_region == 'lucknow':
        training_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/lucknow_small_600_sq_km/kiln_images/train'
        validation_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/lucknow_small_600_sq_km/kiln_images/valid'        
        if test_both_kilns_background_images:
            test_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/lucknow_small_600_sq_km/kiln_images/test_with_background_images'
        else:
            test_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/lucknow_small_600_sq_km/kiln_images/test'
    

    if test_region == 'west_bengal':
        training_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/west_bengal_small_639_sq_km/kiln_images/train'
        validation_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/west_bengal_small_639_sq_km/kiln_images/valid'
        if test_both_kilns_background_images:
            test_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/west_bengal_small_639_sq_km/kiln_images/test_with_background_images'
        else:
            test_set_path = '/home/shataxi.dubey/shataxi_work/vlm_on_planet/gms/west_bengal_small_639_sq_km/kiln_images/test'

    with pd.ExcelWriter(f'PaliGemma_train_{train_region}_test_{test_region}.xlsx', engine = 'xlsxwriter') as writer:
        row = 0

        for model_checkpoint in lucknow_model_checkpoints:
            MODEL_ID ="google/paligemma2-3b-pt-448"

            train_dataset, valid_dataset, test_dataset = create_jsonl_dataset(training_set_path, validation_set_path, test_set_path)

            CLASSES = test_dataset[0][1]['prefix'].replace("detect ", "").split(" ; ")

            model, processor, TORCH_DTYPE = load_model(MODEL_ID, model_checkpoint)

            targets, predictions, images = evaluate_finetuned_paligemma_model(test_dataset, processor, model, CLASSES, TORCH_DTYPE)
            map_result, map50, map50_95 = calculate_map(predictions, targets)
            print(f'{model_checkpoint}')
            df = calculate_confusion_matrix(predictions, targets, CLASSES, map_result)
            df.index = [model_checkpoint]*len(df)
            df.to_excel(writer, sheet_name="Sheet1", startrow = row)
            row += len(df) + 2
            del model
            torch.cuda.empty_cache()
            gc.collect()
            # visualize_predictions(images, predictions, targets, start = 0, end = 25, rows = 5, cols = 5)