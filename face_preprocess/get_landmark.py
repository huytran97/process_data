import os
from glob import glob
import cv2
import sys
import numpy as np
import pandas as pd
from PIL import Image
import dlib
import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb


# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('/data/Face/data/face/shape_predictor_68_face_landmarks.dat')
# fa = FaceAligner(predictor, desiredFaceWidth=128)

train = pd.read_csv('/data/Face/data/RAW/Manually_Annotated_file_lists/training.csv')
valid = pd.read_csv('/data/Face/data/RAW/Manually_Annotated_file_lists/validation.csv', 
                    names=['subDirectory_filePath','face_x','face_y','face_width','face_height','facial_landmarks','expression','valence','arousal'])

all_data = pd.concat([train, valid])

# for i, row in all_data.iterrows():
#     all_data.loc[i, 'subDirectory_filePath'] = row['subDirectory_filePath'].split('/')[-1]

# print(all_data)

# # sys.exit()

name_column = all_data['subDirectory_filePath'].to_numpy()
change_name = lambda name: name.split('/')[-1]

name_column = np.array([change_name(name) for name in name_column])
name_column = pd.DataFrame(name_column)
all_data['subDirectory_filePath'] = name_column
all_data.reset_index()

# print(all_data)

facial_landmarks = []

for image_path in glob('./*/*', recursive=True):
    # try:
    image = cv2.imread(image_path)
    _, class_id, image_name = image_path.split('/')
    # try:
        # print(all_data.loc[all_data['subDirectory_filePath'] == image_name, ['subDirectory_filePath','facial_landmarks']].values[0])
    print('image_name : ' + image_name)
    print(all_data.loc[all_data['subDirectory_filePath'] == image_name, ['subDirectory_filePath','facial_landmarks']].values)
    facial_landmarks.append(all_data.loc[all_data['subDirectory_filePath'] == image_name, ['subDirectory_filePath','facial_landmarks']].values[0])
    # except Exception as e:
    #     print(e)

# np.save('facial_infomation.npy', np.array(facial_landmarks))