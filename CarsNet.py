import os
import utils

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model


class CarsNet(object):
    def __init__(self, learning_rate):
        self.model = self.build_model()

        if os.path.isfile(utils.CHECKPOINT_DIR):
            print('Loading model from checkpoint...')
            self.model = load_model(utils.CHECKPOINT_DIR)
        else:
            self.model = self.build_model()

        self.model.compile(
            optimizer=Adam(lr=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def build_model(self):
        print('\n----------BUILD MODEL----------\n')
        model = Sequential()

        # ------------Layer 1------------
        print('\n---Layer 1---')
        model.add(Conv2D(input_shape=(utils.IMG_WIDTH, utils.IMG_HEIGHT, utils.IMG_CHANNEL),
                         filters=32, kernel_size=(3, 3),
                         padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        print('MaxPool2D:', model.output_shape)

        # ------------Layer 2------------
        print('\n---Layer 2---')
        model.add(Conv2D(filters=64, kernel_size=(3, 3),
                         padding='same', activation='relu'))
        print('Conv2D:', model.output_shape)
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        print('MaxPool2D:', model.output_shape)
        model.add(BatchNormalization())

        # ------------Layer 3------------
        print('\n---Layer 3---')
        model.add(Conv2D(filters=128, kernel_size=(3, 3),
                         padding='same', activation='relu'))
        print('Conv2D:', model.output_shape)
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        print('MaxPool2D:', model.output_shape)

        # ------------Fully connected------------
        print('\n---Fully connected---')
        model.add(GlobalAveragePooling2D())
        print('GAP:', model.output_shape)
        model.add(Flatten())
        print('Flatten:', model.output_shape)
        model.add(Dense(units=512, activation='relu'))
        print('Dense:', model.output_shape)

        # ------------Softmax output------------
        model.add(Dense(units=utils.N_CLASSES, activation='softmax'))
        print('Softmax:', model.output_shape)

        return model

    def fit(self, train_x, train_y, val_x, val_y, epochs, batch_size):
        checkpoint = ModelCheckpoint(filepath=utils.CHECKPOINT_DIR, monitor='val_acc',
                                     verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir='./logs')
        callbacks = [checkpoint, tensorboard]

        self.model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                       validation_data=(val_x, val_y))

