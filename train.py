import numpy as np

PATH = './dataset/'

images = np.load(PATH + 'images.npy')
masks = np.load(PATH + 'masks.npy')

images = np.expand_dims(images, axis=-1)
masks = np.expand_dims(masks, axis=-1)

from sklearn.model_selection import train_test_split
import gc

X, X_v, Y, Y_v = train_test_split(images, masks, test_size=0.2)

del images
del masks

gc.collect()

from another_unet import unet3d
import metrics
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

u3d = unet3d((128, 128, 48, 1))
u3d.summary()

u3d.compile(optimizer=Adam(lr=1e-3), loss=metrics.dice_coefficient_loss, metrics=metrics.dice_coefficient)

hist = u3d.fit(X, Y, batch_size=1, epochs=30, validation_data=(X_v, Y_v), verbose=1)