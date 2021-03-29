import nibabel as nib
import numpy as np
import glob
import re
import cv2

def resize(img):
    target_size = 128
    result_img = np.zeros((target_size, target_size, 48))

    for i in range(48):
        each_slice = img[:, :, i]
        resized_slice = cv2.resize(each_slice, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        result_img[:, :, i] = resized_slice
    
    return result_img

flair = glob.glob('*/*/*T2FLAIR_to_MAG.nii')
seg = glob.glob('*/*/*T2FLAIR_to_MAG_ROI.nii')

t2f_img = []
masks = []

pat = re.compile('.*\.nii')
for items in list(zip(flair, seg)):
    print(items)
    for item in items:
        img = nib.load(item)
        img = img.get_fdata()
        
        if(img.shape[2] != 48):
            over = int((img.shape[2] - 48) / 2)
            img = img[:, :, over : img.shape[2] - over]

        img = resize(img)
        print(img.shape)
        is_flair = item.endswith('MAG.nii')

        if is_flair:
            t2f_img.append(img)
        else:
            masks.append(img)
        
t2f_img = np.array(t2f_img)
masks = np.array(masks)

print(t2f_img.shape)
print(masks.shape)

np.save('./dataset/images.npy', t2f_img)
np.save('./dataset/masks.npy', masks)