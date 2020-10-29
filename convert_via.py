import os 
from glob import glob
import json

with open('/data/Download/via.json') as json_file:
    via_data = json.load(json_file)
keys = via_data.keys()
keys_dict = {}
for k in keys:
    name = k.split('png')[0] + 'png'
    size = int(k.split('png')[-1])
    keys_dict[name] = size

all_data = {}

for label_path in glob('./labels/val/*'):
    if 'JSON' in label_path:
        continue
    with open(label_path) as json_file:
        data = json.load(json_file)
        # print(data)
        data['filename'] = data['filename'].split('/')[-1]
        data['size'] = keys_dict[data['filename']]
        for r in data['regions']:
            try:
                r['region_attributes']['region'] = r['shape_attributes']['huycat'].lower()
                del r['shape_attributes']['huycat']
            except:
                pass
            try:
                del r['note']
                print('del')
            except:
                pass
        try:
            del data['note']
            del data['reviewed']
            del data['isSaved']
        except: 
            pass
        all_data[data['filename'] + str(data['size'])] = data
        # print(all_data)
        # break
        
with open('val.json', 'w') as outfile:
    json.dump(all_data, outfile)