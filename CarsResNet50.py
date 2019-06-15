from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.backend import set_session
from utils import *

import os
import tensorflow as tf


class CarsResNet50(object):
    def __init__(self, learning_rate):
        self.model = self.build_model()

        if os.path.isfile(RESNET_CKPT):
            print('Loading model from checkpoint...')
            self.model = load_model(RESNET_CKPT)

        self.model.compile(
            optimizer=Adam(lr=RESNET_LR),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def build_model(self):
        print('\n----------BUILD MODEL----------\n')
        resnet50 = ResNet50(include_top=False, weights='imagenet', pooling='avg',
                            input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL))

        for layer in resnet50.layers:
            layer.trainable = False

        out = resnet50.output
        print('ResNet50:', out.get_shape().as_list())
        # out = Flatten()(out)
        # print('Flatten:', out.get_shape().as_list())
        out = GlobalAveragePooling2D()(out)
        print('GAP:', out.get_shape().as_list())
        out = Dense(units=512, activation='relu')(out)
        print('Dense:', out.get_shape().as_list())
        out = Dropout(rate=0.5)(out)
        print('Dropout:', out.get_shape().as_list())
        out = Dense(units=N_CLASSES, activation='softmax')(out)
        print('Softmax:', out.get_shape().as_list())

        cars_resnet = Model(inputs=resnet50.input, outputs=out)
        return cars_resnet

    def fit(self, train_x, train_y, val_x, val_y, epochs, batch_size):
        checkpoint = ModelCheckpoint(filepath=RESNET_CKPT, monitor='val_acc',
                                     verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=RESNET_LOG)
        callbacks = [checkpoint, tensorboard]

        self.model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                       validation_data=(val_x, val_y))