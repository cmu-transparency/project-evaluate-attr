# %%
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.datasets import cifar10
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from time import time
import sys
sys.path.append("../")

from densenet import densenet121_model

# %%
DATAPATH = "./"
ROOTPATH = "/home/zifanw/project-deep-vision-explanations/densenet/"
img_rows, img_cols, img_channel = 224, 224, 3

batch_size = 16
nb_classes = 257
epochs = 10

img_rows, img_cols = 224, 224
img_channels = 3

# Parameters for the DenseNet model builder
if K.image_data_format() == 'channels_first':
    img_dim = (img_channels, img_rows, img_cols)
else:
    img_dim = (img_rows, img_cols, img_channels)

dropout_rate = 0.0  # 0.0 for data augmentation
color_mean = [123.68, 116.779, 103.939]

#%%
model = densenet121_model(img_rows=img_rows,
                          img_cols=img_cols,
                          color_type=img_channel,
                          num_classes=nb_classes,
                          weight_path_prefix="../")

print('Model is created')

# sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = Adam()  # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])
print('Finished compiling')

X_train = np.load(DATAPATH + "train_x.npy").astype('float')
Y_train = np.load(
    DATAPATH + "train_y.npy"
)  # The index starts from 1. We substract 1 from it to left shift it to 0.
X_test = np.load(DATAPATH + "test_x.npy").astype('float')
Y_test = np.load(
    DATAPATH + "test_y.npy"
)  # The index starts from 1. We substract 1 from it to left shift it to 0.

for i in range(3):
    X_train[:, :, :, i] -= color_mean[i]
    X_test[:, :, :, i] -= color_mean[i]

Y_train = np_utils.to_categorical(Y_train, num_classes=nb_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes=nb_classes)

#X_train /= 255.
#X_test /= 255.

# %%
generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5. / 32,
                               height_shift_range=5. / 32)

generator.fit(X_train, seed=7)

weights_file = ROOTPATH + "weights/DenseNetCaltech_" + str(time()) + ".h5"

lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10,
                               min_lr=5e-6)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=20)
model_checkpoint = ModelCheckpoint(weights_file,
                                   monitor="val_acc",
                                   save_best_only=True,
                                   save_weights_only=True,
                                   verbose=1)

callbacks = [lr_reducer, early_stopper, model_checkpoint]

#model.load_weights(ROOTPATH + "weights/DenseNetCaltech.h5")

# %%
model.fit_generator(generator.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=(X_test, Y_test),
                    verbose=1)

# %%
scores = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Test loss : ', scores[0])
print('Test accuracy : ', scores[1])