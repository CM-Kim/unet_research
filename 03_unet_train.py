# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import os
import numpy as np
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.transform import resize

# %% [markdown]
# # Load image, mask, label

# %%
##Load images, labels, masks
NPY_PATH = r"dataset\numpy"

labels = np.load(os.path.join(NPY_PATH,'labels.npy'))
t1w_images = (np.load(os.path.join(NPY_PATH,'T1W_images.npy')) / 255.0).astype(np.float32)
t2f_images = (np.load(os.path.join(NPY_PATH,'T2F_images.npy')) / 255.0).astype(np.float32)
t2w_images = (np.load(os.path.join(NPY_PATH,'T2W_images.npy')) / 255.0).astype(np.float32)
masks = (np.load(os.path.join(NPY_PATH,'Mask_images.npy')) / 255.0).astype(np.float32)

print(masks.shape)

# %% [markdown]
# # Resize data
# 
# 원본 데이터 width : 260 , height : 320 
# 결과 데이터 width : 256 , height : 256
# 
# 

# %%
# 모델의 입력 형태에 맞추기 위해 차원을 확장합니다.
# example. (,260,320) -> (,260,320,1)
t1w_images = np.expand_dims(t1w_images,axis=-1)
t2f_images = np.expand_dims(t2f_images,axis=-1)
t2w_images = np.expand_dims(t2w_images,axis=-1)
masks = np.expand_dims(masks,axis=-1)


# %%
from util import downsample

img_size_ori = (260,320)
img_size_target = (256,320)

t1w_images = np.array([ downsample(image,img_size_ori,img_size_target) for image in t1w_images ])
t2f_images = np.array([ downsample(image,img_size_ori,img_size_target) for image in t2f_images ])
t2w_images = np.array([ downsample(image,img_size_ori,img_size_target) for image in t2w_images ])
masks = (np.array([ downsample(mask,img_size_ori,img_size_target) for mask in masks ])>0)*1.0


# %%
integer_to_class = {'0': 'no ms lesion', '1': 'ms lesion'}

classes, counts = np.unique(labels,return_counts=True)
plt.bar(classes,counts,tick_label=list(integer_to_class.values()))


# %%
from sklearn.model_selection import train_test_split
import gc

image_datasets = np.concatenate((t1w_images,t2f_images,t2w_images),axis=0)
mask_datasets = np.concatenate((masks,masks,masks),axis =0)
labels_datasets = np.concatenate((labels,labels,labels),axis =0)

print(image_datasets.shape)
print(mask_datasets.shape)
print(labels_datasets.shape)

# %%
from sklearn.model_selection import train_test_split
import gc
X,X_v,Y,Y_v = train_test_split( image_datasets,mask_datasets,test_size=0.2,stratify=labels_datasets,random_state=444, shuffle=True)

del image_datasets
del mask_datasets
del labels_datasets

gc.collect()

X.shape,X_v.shape

# %% [markdown]
# ### Augmentation

# %%
from model import baseModel, unet_v1, unet_v2, unet_v3

model_list = [baseModel,unet_v1,unet_v2,unet_v3]
model_name = ['base_320','v1','v2','v3']

for name, model_arch in zip(model_name,model_list):
    unet = model_arch(input_shape=(256,320,1))

    # %% [markdown]
    # ## compile Model

    # %%
    from keras import optimizers
    from util import bce_dice_loss
    from keras.metrics import MeanIoU

    unet.compile(optimizer=optimizers.Adam(lr=1e-3), 
                loss=bce_dice_loss, metrics=['accuracy',MeanIoU(num_classes=2,name="mean_iou")])


    # %%
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
    from keras.models import load_model

    model_name = name
    checkpoint_name = f'model_{model_name}_best_checkpoint.h5'
    history_path = f'model_{model_name}_history.json'

    model_checkpoint  = ModelCheckpoint(checkpoint_name, save_best_only=True, 
                                        monitor='val_loss', mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min' , baseline=None)
    reduceLR = ReduceLROnPlateau(patience=4, verbose=2, monitor='val_loss',min_lr=1e-4, mode='min')

    callback_list = [early_stopping, reduceLR, model_checkpoint]

    hist = unet.fit(
                        X,
                        Y,
                        batch_size=16,
                        epochs=1,
                        validation_data=(X_v,Y_v),
                        verbose=1,
                        callbacks= callback_list
            )

    # %% [markdown]
    # # Save History

    # %%
    import json
    import numpy as np

    history_dict = hist.history

    class NumpyEncoder(json.JSONEncoder):
        def default(self,obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj,np.ndarray):
                return obj.tolist()
            else:
                return super(NumpyEncoder, self).default(obj)

    with open(history_path,'w') as json_file:
        json.dump(history_dict, json_file, cls=NumpyEncoder)


