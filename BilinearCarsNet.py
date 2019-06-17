from tensorflow.python.keras import backend as Backend
from tensorflow.python.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.python.keras.layers import Reshape, Dense, Lambda
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from utils import *

import os


def _outer_product(x):
    batch_dot = Backend.batch_dot(x=x[0], y=x[1], axes=[1, 1])
    return batch_dot / x[0].get_shape().as_list()[1]


def _signed_sqrt(x):
    sign = Backend.sign(x)
    sqrt = Backend.sqrt(Backend.abs(x) + 1e-9)
    return sign * sqrt


def _l2_normalise(x, axis=1):
    return Backend.l2_normalize(x, axis=axis)


class BilinearCarsNet(object):
    def __init__(self, learning_rate=LEARNING_RATE, use_augmentation=False):
        if use_augmentation:
            ckpt = AUG_CKPT
        else:
            ckpt = CHECKPOINT_DIR

        if os.path.isfile(ckpt):
            print('Loading model from checkpoint...')
            self.model = load_model(ckpt)
        else:
            self.model = self.build_model()

        self.model.compile(
            optimizer=Adam(lr=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def build_model(self):
        print('\n----------BUILD MODEL----------\n')
        inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL))

        # ----------Block 1-----------
        print('\n---Block 1---')
        x = Conv2D(filters=32, kernel_size=(3, 3),
                   padding='same', activation='relu')(inputs)
        print('Conv2D:', x.get_shape().as_list())
        x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
        print('MaxPool2D:', x.get_shape().as_list())

        # ----------Block 2-----------
        print('\n---Block 2---')
        x = Conv2D(filters=64, kernel_size=(3, 3),
                   padding='same', activation='relu')(x)
        print('Conv2D:', x.get_shape().as_list())
        x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
        print('MaxPool2D:', x.get_shape().as_list())
        x = BatchNormalization()(x)

        # ----------Block 3-----------
        print('\n---Block 3---')
        x = Conv2D(filters=64, kernel_size=(3, 3),
                   padding='same', activation='relu')(x)
        print('Conv2D:', x.get_shape().as_list())
        x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
        print('MaxPool2D:', x.get_shape().as_list())

        # ----------Merge 2 CNNs----------
        print('\n---Merge 2 CNNs---')
        detector = x
        detector_shape = detector.get_shape().as_list()
        extractor = x
        extractor_shape = extractor.get_shape().as_list()

        detector = Reshape([detector_shape[1] * detector_shape[2], detector_shape[3]])(detector)
        print('Detector:', detector.get_shape().as_list())
        extractor = Reshape([extractor_shape[1] * extractor_shape[2], extractor_shape[3]])(extractor)
        print('Extractor:', extractor.get_shape().as_list())

        bcnn = Lambda(_outer_product)([detector, extractor])
        print('Outer product:', bcnn.get_shape().as_list())

        bcnn = Reshape([detector_shape[3] * extractor_shape[3]])(bcnn)
        print('Reshape:', bcnn.get_shape().as_list())
        bcnn = Lambda(_signed_sqrt)(bcnn)
        print('Signed square root:', bcnn.get_shape().as_list())
        bcnn = Lambda(_l2_normalise)(bcnn)
        print('L2 normalisation:', bcnn.get_shape().as_list())

        # ----------Fully Connected----------
        bcnn = Dense(units=N_CLASSES, activation='softmax')(bcnn)
        print('Softmax:', bcnn.get_shape().as_list())

        bcnn_model = Model(inputs=[inputs], outputs=[bcnn])

        return bcnn_model

    def fit(self, train_flow, val_flow, augment=False):
        if augment:
            ckpt = AUG_CKPT
            log = AUG_LOG
        else:
            ckpt = CHECKPOINT_DIR
            log = LOG_DIR

        checkpoint = ModelCheckpoint(filepath=ckpt, monitor='val_acc',
                                     verbose=1, save_best_only=False)
        tensorboard = TensorBoard(log_dir=log)
        callbacks = [checkpoint, tensorboard]

        self.model.fit_generator(generator=train_flow, epochs=N_EPOCHS, callbacks=callbacks,
                                 validation_data=val_flow, workers=4)

    def predict(self, data_flow):
        return self.model.predict(data_flow)
