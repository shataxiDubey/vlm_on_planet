import numpy as np
import glob
import json
import os
from PIL import Image


def dota_to_jsonl(ann, img_name, dota_anns, task, model, CLASSES):
    image_path = ann.replace('labelTxt', 'images')
    image_path = image_path[:-4]+'.png'
    img = Image.open(image_path)
    img_width = img.size[0]
    img_height = img.size[1]
    suffix = ''
    for dota_ann in dota_anns:
        
        x1, y1, x2, y2, x3, y3, x4, y4, classname, level = dota_ann[0]
        x1, x2, x3, x4 = np.array([x1, x2, x3, x4]) / img_width
        y1, y2, y3, y4 = np.array([y1, y2, y3, y4]) / img_height
    
        if model == 'florence_2':
            x1, y1, x2, y2, x3, y3, x4, y4 = round(x1*1000), round(y1*1000), round(x2*1000), round(y2*1000), round(x3*1000), round(y3*1000), round(x4*1000), round(y4*1000)
            suffix += classname+'<loc_'+str(x1)+'>'+'<loc_'+str(y1)+'>'+'<loc_'+str(x2)+'>'+'<loc_'+str(y2)+'>'+'<loc_'+str(x3)+'>'+'<loc_'+str(y3)+'>'+'<loc_'+str(x4)+'>'+'<loc_'+str(y4)+'>'
            
    # convert to json
    json_ann = json.dumps({"image": img_name, "prefix": task, "suffix": suffix})
    return json_ann

def yolo_to_jsonl(img_name, obb, yolo_anns, task, model, CLASSES):
    suffix = ''
    for yolo_ann in yolo_anns:
        if obb:
            classname, x1, y1, x2, y2, x3, y3, x4, y4 = yolo_ann
            x_top = min(x1, x2, x3, x4)
            x_bottom = max(x1, x2, x3, x4)
            y_top = min(y1, y2, y3, y4)
            y_bottom = max(y1, y2, y3, y4)

        else:
            classname, xc, yc, width, height = yolo_ann
            x_top, y_top = xc - (width / 2), yc - (height / 2)
            x_bottom, y_bottom = xc + (width / 2), yc + (height / 2)

        # if classname == 1:
        #     classname = 'zigzag'
        # if classname == 0:
        #     classname = 'fcbk'

        # change prompt to 'brick kiln with chimney', 'brick kiln', 'brick kilns', 'rectangular brick kiln with chimney', 'brick kiln or brick kilns with chimney or chimney stack','object with chimney'
        # classname = 'brick kilns' 
        classname = CLASSES[int(classname)]
    
        if model == 'florence_2':
            x_top, y_top, x_bottom, y_bottom = round(x_top*1000), round(y_top*1000), round(x_bottom*1000), round(y_bottom*1000)
            suffix += classname+'<loc_'+str(x_top)+'>'+'<loc_'+str(y_top)+'>'+'<loc_'+str(x_bottom)+'>'+'<loc_'+str(y_bottom)+'>'
        elif model == 'paligemma2':
            x_top, y_top, x_bottom, y_bottom = round(x_top*1024), round(y_top*1024), round(x_bottom*1024), round(y_bottom*1024)
            y_top, x_top, y_bottom, x_bottom = ['0'*(4-len(x))+str(x) if len(x) < 4 else x for x in [str(int(y_top)), str(int(x_top)), str(int(y_bottom)), str(int(x_bottom))] ]
            if suffix != '':
                suffix += ' ; '
            suffix += '<loc'+str(y_top)+'>'+'<loc'+str(x_top)+'>'+'<loc'+str(y_bottom)+'>'+'<loc'+str(x_bottom)+'>'+f' {classname}'
    
    # print(f'Suffix is {suffix}')

    # convert to json
    json_ann = json.dumps({"image": img_name, "prefix": task, "suffix": suffix})
    return json_ann


def create_jsonl_file(images_file_path, labels_file_path, json_file_path, is_dota_dataset, task, model_name, obb, image_type, annotation_format):

    images_path = glob.glob(f'{images_file_path}/*')
    # annotations = glob.glob(f'{labels_file_path}/*')
    json_file = open(f'{json_file_path}', 'w+')

    if is_dota_dataset:
        CLASSES = ['plane',
        'baseball-diamond',
        'bridge',
        'ground-track-field',
        'small-vehicle',
        'large-vehicle',
        'ship',
        'tennis-court',
        'basketball-court',
        'storage-tank',
        'soccer-ball-field',
        'roundabout',
        'harbor',
        'swimming-pool',
        'helicopter']
    else:
        CLASSES = ['brick kilns with chimney']
    
    for image_path in images_path:
        img_name = os.path.basename(image_path)
        ann = os.path.join(labels_file_path, img_name[:-3]+'txt')

        # if image_type == 'png':
        #     img_name = img[:-3] + 'png'
        # else:
        #     img_name = img[:-3] + 'tif'

        if is_dota_dataset and not os.path.exists(f'/home/shataxi.dubey/shataxi_work/vlm_on_planet/Florence-2/dota_coco_train_test_640px/valid/{img_name}'):
            continue
        
        if annotation_format == 'yolo':
            if os.path.exists(ann):
                # print(f'{ann} exists')
                yolo_anns = np.loadtxt(ann, ndmin = 2)
            else:
                # print(f'{ann} does not exist')
                yolo_anns = []
            json_ann = yolo_to_jsonl(img_name, obb, yolo_anns, task, model_name, CLASSES = CLASSES)

        elif annotation_format == 'dota':
            if os.path.exists(ann):
                dota_anns = np.loadtxt(ann, dtype = {'names': ('x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'classname', 'level'), 'formats': (float, float, float, float, float, float, float, float, 'U30', int)}, ndmin =2)
            else:
                dota_anns = []
            if len(dota_anns):
                json_ann = dota_to_jsonl(ann, img_name, dota_anns, task, model_name, CLASSES = CLASSES)

        json_file.write(json_ann)
        json_file.write('\n')

    json_file.close()
    print(f'JSONL file created at location {json_file_path}')
