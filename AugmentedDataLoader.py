from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


class AugmentedDataLoader(object):
    def __init__(self):
        self.ids = []
        self.train_flow = None
        self.val_flow = None

        self.augment_data()

    def augment_data(self):
        datagen = ImageDataGenerator(
            brightness_range=[0.5, 1.5],
            horizontal_flip=True,
            shear_range=0.2,
            zoom_range=0.2,
            validation_split=0.2,
        )

        self.train_flow = datagen.flow_from_directory(
            directory='cars_augmented/',
            target_size=(224, 224),
            batch_size=8,
            subset='training'
        )

        self.val_flow = datagen.flow_from_directory(
            directory='cars_augmented/',
            target_size=(224, 224),
            batch_size=8,
            subset='validation'
        )
