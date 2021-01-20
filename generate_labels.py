import numpy as np
import cv2
import os

# mask 이미지를 이용해 질병 유무를 labeling

def is_img_empty(img):
    if np.any(img >= 1):
        return 2    # 질병 존재

    return 1        # 질병 미 존재

labels = []

path = "C:/MRI_Brain/dataset/mask/"
file_list = os.listdir(path)

i = 0

for img in file_list:
    img_path = path + img
    i += 1
    print(str(i) + ' / ' + str(len(file_list)))
    image = cv2.imread(img_path)
    labels.append(is_img_empty(image))

labels = np.array(labels)
print(labels)

np.save("C:/MRI_Brain/dataset/numpy/labels.npy", labels)