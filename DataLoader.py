import numpy as np
import pandas as pd
import os
import utils

from PIL import Image
from random import shuffle


class DataLoader(object):
    def __init__(self, train=True):
        self.ids = []
        self.data = []
        self.labels = []
        self.data_size = 0

        self.load_data(train=train)

        if train:
            print('\n----------TRAIN LOADER----------\n')
        else:
            print('\n----------VAL LOADER----------\n')

        print('Data:', self.data_size)
        print('Labels:', len(self.labels))
        print('Data instance:', self.data[0].shape)

    def load_data(self, train=True):
        train_annotations = pd.read_csv(os.path.join('devkit', 'train_annotations.csv'))
        train_annotations = train_annotations[train_annotations['train'] == train]

        for _, row in train_annotations.iterrows():
            if train:
                image_arr = np.array(Image.open(fp=os.path.join('cars_train_prepped', 'train', row['fname'])))
            else:
                image_arr = np.array(Image.open(fp=os.path.join('cars_train_prepped', 'val', row['fname'])))

            image_arr = image_arr / 255
            self.data.append(image_arr)

            self.ids.append(row['fname'])

            label = [0] * utils.N_CLASSES
            label[row['class']] = 1
            self.labels.append(label)

        self.data_size = len(self.data)
        self.data = self.reshape_input(self.data)
        self.labels = np.array(self.labels)

    def reshape_input(self, arr, train=True):
        np_arr = np.array(arr)
        np_arr = np_arr.reshape(self.data_size, utils.IMG_WIDTH, utils.IMG_HEIGHT, utils.IMG_CHANNEL)

        return np_arr
