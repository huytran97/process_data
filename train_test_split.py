import os 
import glob
import random

root_dir = './Training'
list_image = os.listdir(root_dir)
random.shuffle(list_image)
total = len(list_image)

valid = open('valid.txt', 'w+')
test = open('test.txt', 'w+')

for i, image_name in enumerate(list_image):
    image_path = os.path.join(root_dir, image_name)
    if i / total <= 0.1:
        valid.write(image_path + '\n')
    elif 0.1 < i / total <= 0.2:
        test.write(image_path + '\n')
    else:
        break
