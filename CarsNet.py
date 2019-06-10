import tensorflow as tf
import utils

from tensorflow._api.v1.keras import Sequential
from tensorflow._api.v1.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense


class CarsNet(object):
    def __init__(self, learning_rate):
        self.model = self.build_model()
        self.model.compile(
            optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def build_model(self):
        print('\n----------BUILD MODEL----------\n')
        model = Sequential()

        # ------------Layer 1------------
        print('\n---Layer 1---')
        model.add(Conv2D(input_shape=(utils.IMG_WIDTH, utils.IMG_HEIGHT, utils.IMG_CHANNEL),
                         filters=30, kernel_size=(3, 3), strides=4,
                         padding='same', activation='relu'))
        print('Conv2D:', model.output_shape)
        model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        print('MaxPool2D:', model.output_shape)

        # ------------Layer 2------------
        print('\n---Layer 2---')
        model.add(Conv2D(filters=60, kernel_size=(3, 3), strides=2,
                         padding='same', activation='relu'))
        print('Conv2D:', model.output_shape)
        model.add(Conv2D(filters=60, kernel_size=(3, 3), strides=2,
                         padding='same', activation='relu'))
        print('Conv2D:', model.output_shape)
        model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        print('MaxPool2D:', model.output_shape)

        # ------------Layer 3------------
        print('\n---Layer 3---')
        model.add(Conv2D(filters=120, kernel_size=(3, 3), strides=2,
                         padding='same', activation='relu'))
        print('Conv2D:', model.output_shape)
        model.add(Conv2D(filters=120, kernel_size=(3, 3), strides=2,
                         padding='same', activation='relu'))
        print('Conv2D:', model.output_shape)
        model.add(Conv2D(filters=120, kernel_size=(3, 3), strides=2,
                         padding='same', activation='relu'))
        print('Conv2D:', model.output_shape)
        model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        print('MaxPool2D:', model.output_shape)

        # ------------Layer 4------------
        print('\n---Layer 4---')
        model.add(Conv2D(filters=240, kernel_size=(3, 3), strides=2,
                         padding='same', activation='relu'))
        print('Conv2D:', model.output_shape)
        model.add(Conv2D(filters=240, kernel_size=(3, 3), strides=2,
                         padding='same', activation='relu'))
        print('Conv2D:', model.output_shape)
        model.add(Conv2D(filters=240, kernel_size=(3, 3), strides=2,
                         padding='same', activation='relu'))
        print('Conv2D:', model.output_shape)
        model.add(Conv2D(filters=240, kernel_size=(3, 3), strides=2,
                         padding='same', activation='relu'))
        print('Conv2D:', model.output_shape)
        model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        print('MaxPool2D:', model.output_shape)
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))

        # ------------Layer 5------------
        model.add(Dense(units=4096, activation='relu'))
        print('Dense:', model.output_shape)
        model.add(Dense(units=4096, activation='relu'))
        print('Dense:', model.output_shape)
        model.add(Dense(units=1000, activation='relu'))
        print('Dense:', model.output_shape)
        model.add(Dense(units=utils.NUM_CLASSES, activation='softmax'))
        print('Softmax:', model.output_shape)

        return model

    def fit(self, data, labels, epochs, batch_size):
        self.model.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs)

