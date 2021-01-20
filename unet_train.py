import os
import numpy as np
import cv2
from skimage.transform import resize

integer_to_class = { '1': 'no_lesion (1)', '2': 'lesion (2)' }

# npy load
path = "C:/MRI_Brain/dataset/numpy"

labels = np.load(path + '/labels.npy')
t1w_images = np.clip((np.load(path + '/t1w_images.npy') / 12728), 0, 1)
t2f_images = np.clip((np.load(path + '/t2f_images.npy') / 12728), 0, 1)
t2w_images = np.clip((np.load(path + '/t2w_images.npy') / 12728), 0, 1)
masks = np.load(path + '/masks.npy') * 1

# resizing
img_size_ori = 512
img_size_target = 128

t1w_images = np.expand_dims(t1w_images,axis=-1)
t2f_images = np.expand_dims(t2f_images,axis=-1)
t2w_images = np.expand_dims(t2w_images,axis=-1)
masks = np.expand_dims(masks,axis=-1)

def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True,)
    
def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
  
t1w_images = np.array([ downsample(image) for image in t1w_images ])
t2f_images = np.array([ downsample(image) for image in t2f_images ])
t2w_images = np.array([ downsample(image) for image in t2w_images ])
masks = (np.array([ downsample(mask) for mask in masks ])>0)*1

print(t1w_images.shape)
print(masks.shape)