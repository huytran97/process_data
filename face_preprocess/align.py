import os
from glob import glob
import cv2
import sys
import numpy as np
from _collections import OrderedDict
import pandas as pd
from PIL import Image
import dlib
from imutils import face_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', help='Path to root folder', required=True)
parser.add_argument('--epath', help='Path to emotion folder', required=True)
parser.add_argument('--save_path', help='Path to save cropped images', required=True)

args = parser.parse_args()

FACIAL_LANDMARKS_68_IDXS = OrderedDict(
    [("mouth", (48, 68)), ("inner_mouth", (60, 68)), ("right_eyebrow", (17, 22)),
     ("left_eyebrow", (22, 27)), ("right_eye", (36, 42)), ("left_eye", (42, 48)),
     ("nose", (27, 36)), ("jaw", (0, 17))])


# fa = FaceAligner(predictor, desiredFaceWidth=128)
def FaceAligner(shape):
    shape = face_utils.shape_to_np(shape)
    # print(shape)
    # extract the left and right eye (x, y)-coordinates
    (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    return angle

#Read landmark from file
# facial_info = np.load('/data/Face/data/relabeled_data/18-06/Emotions/facial_information.npy',
#                       allow_pickle=True)
# df = pd.DataFrame(facial_info, columns=['name', 'landmark'])

predictor = dlib.shape_predictor('/data/Face/data/face/shape_predictor_68_face_landmarks.dat')

root = args.root
folder = args.epath
desdir = os.path.join('crop', args.save_path)
os.makedirs(os.path.join(root, 'crop'), exist_ok=True)
os.makedirs(os.path.join(root, desdir), exist_ok=True)

for image_path in glob(os.path.join(root, folder) + '/*', recursive=True):
    # try:
    #Parse landmark
    # print(image_path)
    # landmarks = df.loc[df['name'] == image_path.split('/')[-1], ['landmark']].values[0]
    # landmarks = landmarks[0].split(';')
    # landmarks = [float(l) for l in landmarks]
    # landmarks = np.array(landmarks).reshape(68, 2)

    # Using dlib to get lanmark 
    save_path = image_path.replace(folder, desdir)
    image = cv2.imread(image_path)

    # image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # image
    x, y, w, h = 0, 0, image.shape[1], image.shape[0]

    landmarks = predictor(gray, dlib.rectangle(left=0, top=0, right=image.shape[1], bottom=image.shape[0]))

    a = FaceAligner(landmarks)

    faceAligned = Image.fromarray(image).rotate(a)
    print(image_path.replace(folder, desdir))
    cv2.imwrite(save_path, np.array(faceAligned))
    # if cv2.waitKey(0) == ord('q'):
    #     break
# cv2.imshow('image', image)
# cv2.imshow('image_a', np.array(faceAligned))
# except Exception as e:
#     print(e)

# cv2.destroyAllWindows()