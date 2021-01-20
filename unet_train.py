import os
import numpy as np

integer_to_class = { '1': 'no_lesion (1)', '2': 'lesion (2)' }

# npy load
path = "C:/MRI_Brain/dataset/numpy"

labels = np.load(path + '/labels.npy')
t1w_images = np.clip((np.load(path + '/t1w_images.npy') / 12728), 0, 1)
t2f_images = np.clip((np.load(path + '/t2f_images.npy') / 12728), 0, 1)
t2w_images = np.clip((np.load(path + '/t2w_images.npy') / 12728), 0, 1)
masks = np.load(path + '/masks.npy') * 1

print(labels.shape)
print(t1w_images.shape)
print(masks.shape)