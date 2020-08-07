import numpy as np
import pandas as pd
import collections
import json
import glob
import matplotlib.pyplot as plt
import os
import shutil
import keras

from keras_video import VideoFrameGenerator
import keras_video.utils

#Use subdir names as classes
classes = [i.split(os.path.sep)[1] for i in glob.glob('data/*')]
classes.sort()
print(classes)

#Global Params
SIZE = (256, 256)
CHANNELS = 3
NBFRAME = 5
BS = 8

#pattern to get videos and classes
glob_pattern = 'data/{classname}/*.mp4'

#Create video frame generator
train = VideoFrameGenerator(
    classes=classes,
    glob_pattern=glob_pattern,
    nb_frames=NBFRAME,
    split_val=.33,
    shuffle=False,
    batch_size=BS,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    use_frame_cache=False)

valid = train.get_validation_generator()
print(valid)

import keras_video.utils
keras_video.utils.show_sample(train)

# Model
from keras.layers import Conv2D, ConvLSTM2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D
def build_convnet(shape=(256, 256, 3)):
    momentum = .85
    model = keras.Sequential()
    model.add(Conv2D(64, (3,3), input_shape=shape,
        padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())
    model.add(Dropout(.55))

    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())
    model.add(Dropout(.55))

    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())
    model.add(Dropout(.55))

    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    # flatten...
    model.add(GlobalMaxPool2D())
    model.add(Dropout(.55))

    return model

from keras.layers import TimeDistributed, GRU, LSTM, Dense, Dropout
def action_model(shape=(5, 256, 256, 3), nbout=3):
    # Create our convnet with (256, 256, 3) input shape
    convnet = build_convnet(shape[1:])

    # then create our final model
    model = keras.Sequential()
    # add the convnet with (5, 256, 256, 3) shape
    model.add(TimeDistributed(convnet, input_shape=shape))
    # here, you can also use GRU or LSTM
    model.add(LSTM(64))
    # and finally, we make a decision network
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.55))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.55))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.55))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.55))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    return model

INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) # (5, 256, 256, 3)
model = action_model(INSHAPE, len(classes))
optimizer = keras.optimizers.Adam(0.001)
print(model.summary())
model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

EPOCHS=50
# create a "chkp" directory before to run that
# because ModelCheckpoint will write models inside
callbacks = [
    keras.callbacks.ReduceLROnPlateau(verbose=1),
    keras.callbacks.ModelCheckpoint(
        'chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        verbose=1),
]
model.fit_generator(
    train,
    validation_data=valid,
    verbose=1,
    epochs=EPOCHS,
    callbacks=callbacks
)
