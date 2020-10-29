import json
import os

coco_json = {}
coco_json['info'] = {
        "year": 2020,
        "version": "1",
        "description": "Exported using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)",
        "contributor": "",
        "url": "http://www.robots.ox.ac.uk/~vgg/software/via/",
        "date_created": "Wed Oct 28 2020 09:20:29 GMT+0700 (Indochina Time)"
    }
coco_json['images'] = []
coco_json['annotations'] = []
coco_json['licenses'] = [
        {
            "id": 1,
            "name": "Unknown",
            "url": ""
        }
    ]
coco_json['categories'] = [
        {
            "id": 1,
            "name": "thyroid",
            "supercategory": "region"
        }
]

with open('./train.json', 'r+') as json_file:
    via = json.load(json_file)
    
ann_id = 0

for i, k in enumerate(sorted(list(via.keys()))):
    coco_json['images'].append({
        "id": i,
        "width": 800,
        "height": 600,
        "file_name": via[k]['filename'],
        "license": 1,
        "date_captured": ""
    })
    
    for region in via[k]['regions']:
        coco_json['annotations'].append({
            "id": ann_id,
            "image_id": i,
            "category_id": 1,
            "segmentation": [[val for pair in zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y']) for val in pair]],
            "area": 1,
            "bbox": [],
            "iscrowd": 0
        })

        ann_id += 1
with open('./train_coco.json', 'w+') as json_file:
    json.dump(coco_json, json_file)