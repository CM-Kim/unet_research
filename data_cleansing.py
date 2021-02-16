# 쓸모없는 사진을 지워보아요~
import cv2
import numpy as np
import os

def is_img_empty(img):
    if np.any(img >= 1):
        return False
    return True

PATH = r"dataset\Mask_brain_pic"
file_list = os.listdir(PATH)

delete_list = []

for img in file_list:
    if img.endswith('_ROI.png'):
        IMG_PATH = PATH + img
        image = cv2.imread(IMG_PATH)

        if is_img_empty(image):
            delete_list.append(img[:24] + "ROI.png")
            delete_list.append(img[:24] + "T1W.png")
            delete_list.append(img[:24] + "T2F.png")
            delete_list.append(img[:24] + "T2W.png")

for target in delete_list:
    target_path = PATH + target
    os.remove(target_path)