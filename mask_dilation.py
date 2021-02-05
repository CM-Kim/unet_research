import os
import cv2
import numpy as np

PATH = './MRI_T1T2T2F_ROI/'
file_path = os.listdir(PATH)

DST_PATH = './dilationed_ROI/'

for img in file_path:
    if img.endswith('ROI.png'):
        img_path = PATH + img
        image = cv2.imread(img_path)
        kernel = np.ones((5,5), np.uint8)
        result = cv2.dilate(image, kernel, iterations=1)

        cv2.imwrite(DST_PATH + img, result)