import utils
from AugmentedDataLoader import AugmentedDataLoader
from BilinearCarsNet import BilinearCarsNet

cars_net = BilinearCarsNet(learning_rate=utils.LEARNING_RATE, use_augmentation=True)

augmented_loader = AugmentedDataLoader()
train_flow = augmented_loader.train_flow
val_flow = augmented_loader.val_flow

print('\n----------MODEL FITTING----------\n')
cars_net.fit(train_flow=train_flow, val_flow=val_flow, augment=True)
