from keras.losses import binary_crossentropy
from keras import backend as K
import tensorflow as tf
import numpy as  np

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

def iou_metric(label, pred):
    return tf.py_function(mean_iou, [label, pred > 0.5], tf.float64)

def getIOUCurve(mask_org,predicted):
  thresholds = np.linspace(0, 1, 100)
  ious = np.array([mean_iou(mask_org, predicted > threshold) for threshold in thresholds])
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
    
def upsample(img, img_size_ori, img_size_target):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori[0], img_size_ori[1]), mode='constant', preserve_range=True)
