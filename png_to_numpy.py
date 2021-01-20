import numpy as np
from PIL import Image
import os
import cv2

# TODO
# 1. mask 이용해서 label 행렬 만들기
# 2. mat_to_numpy 참고해서 labels / images / masks 만들기
# 3. 아 전처리 어케하지 ㅋㅋ;

path_dir = 'C:/MRI_Brain/dataset/mask/'
file_list = os.listdir(path_dir)

for png in file_list:
    image = Image.open(path_dir + png)
    pixel = np.array(image)
    png = png.split('.')[0]
    np.save("C:/MRI_Brain/dataset/numpy_mask/" + png, pixel)