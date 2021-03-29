from keras.losses import binary_crossentropy
from keras import backend as K
import tensorflow as tf
import numpy as  np
import json
from numpyEncoder import NumpyEncoder
import matplotlib.pyplot as plt
import gc

from keras.metrics import MeanIoU

def dice_loss(y_true, y_pred):
    smooth=1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

### bce_dice_loss = binary_crossentropy_loss + dice_loss
def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def mean_iou(A,B):
    m = MeanIoU(num_classes=2)
    m.update_state(A,B)
    return m.result().numpy()

def get_iou_vector(A, B):
    t = A
    p = B
    intersection = np.logical_and(t,p)
    union = np.logical_or(t,p)
    iou = (np.sum(intersection) + 1e-10 )/ (np.sum(union) + 1e-10)
    return iou

def iou_metric(label, pred):
    return tf.py_function(mean_iou, [label, pred], tf.float64)

def getIOUCurve(mask_org,predicted):
    thresholds = np.linspace(0, 1, 100)

    m = MeanIoU(num_classes=2)

    ious = []

    for threshold in thresholds:
        m.update_state(mask_org, (predicted > threshold) * 1.0)
        ious.append(m.result().numpy())

    ious = np.array(ious)

    thres_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[thres_best_index]
    thres_best = thresholds[thres_best_index]

    return thresholds,ious,iou_best,thres_best

from skimage.transform import resize
  # 이미지의 사이즈를 줄입니다.
def downsample(img, img_size_ori, img_size_target):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target[0], img_size_target[1]), mode='constant', preserve_range=True,)

def downsample(img, img_size_ori, img_size_target):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target[0], img_size_target[1]), mode='constant', preserve_range=True,)
    
def upsample(img, img_size_ori, img_size_target):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori[0], img_size_ori[1]), mode='constant', preserve_range=True)

def historyToJson(history, save_path):
    with open(save_path,'w') as json_file:
        json.dump(history, json_file, cls=NumpyEncoder)

def historyToPng(history, save_path ,hasLoss=True, hasAcc=True, hasIOU=True):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    t = f.suptitle('Unet Performance in Segmenting Tumors', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)
    epoch_list = range(len(history['loss']))
    
    if hasAcc:
        ax1.plot(epoch_list, history['accuracy'], label='Train Accuracy')
        ax1.plot(epoch_list, history['val_accuracy'], label='Validation Accuracy')
        ax1.set_xticks(np.arange(0, epoch_list[-1], 5))
        ax1.set_ylabel('Accuracy Value');ax1.set_xlabel('Epoch');ax1.set_title('Accuracy')
        ax1.legend(loc="best");ax1.grid(color='gray', linestyle='-', linewidth=0.5)

    if hasLoss:
        ax2.plot(epoch_list, history['loss'], label='Train Loss')
        ax2.plot(epoch_list, history['val_loss'], label='Validation Loss')
        ax2.set_xticks(np.arange(0, epoch_list[-1], 5))
        ax2.set_ylabel('Loss Value');ax2.set_xlabel('Epoch');ax2.set_title('Loss')
        ax2.legend(loc="best");ax2.grid(color='gray', linestyle='-', linewidth=0.5)

    if hasIOU:
        ax3.plot(epoch_list, history['mean_iou'], label='Train IOU metric')
        ax3.plot(epoch_list, history['val_mean_iou'], label='Validation IOU metric')
        ax3.set_xticks(np.arange(0, epoch_list[-1], 5))
        ax3.set_ylabel('IOU metric');ax3.set_xlabel('Epoch');ax3.set_title('IOU metric')
        ax3.legend(loc="best");ax3.grid(color='gray', linestyle='-', linewidth=0.5)
    
    plt.savefig(save_path)

def displayHistory(history):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    t = f.suptitle('Unet Performance in Segmenting Tumors', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)
    epoch_list = history.epoch

    ax1.plot(epoch_list, history['accuracy'], label='Train Accuracy')
    ax1.plot(epoch_list, history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(0, epoch_list[-1], 5))
    ax1.set_ylabel('Accuracy Value');ax1.set_xlabel('Epoch');ax1.set_title('Accuracy')
    ax1.legend(loc="best");ax1.grid(color='gray', linestyle='-', linewidth=0.5)

    ax2.plot(epoch_list, history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(0, epoch_list[-1], 5))
    ax2.set_ylabel('Loss Value');ax2.set_xlabel('Epoch');ax2.set_title('Loss')
    ax2.legend(loc="best");ax2.grid(color='gray', linestyle='-', linewidth=0.5)

    ax3.plot(epoch_list, history['mean_iou'], label='Train IOU metric')
    ax3.plot(epoch_list, history['val_mean_iou'], label='Validation IOU metric')
    ax3.set_xticks(np.arange(0, epoch_list[-1], 5))
    ax3.set_ylabel('IOU metric');ax3.set_xlabel('Epoch');ax3.set_title('IOU metric')
    ax3.legend(loc="best");ax3.grid(color='gray', linestyle='-', linewidth=0.5)