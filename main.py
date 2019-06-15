import utils

from DataLoader import DataLoader
from CarsNet import CarsNet
from BilinearCarsNet import BilinearCarsNet
from CarsResNet50 import CarsResNet50


cars_net = BilinearCarsNet(learning_rate=utils.LEARNING_RATE)
train_loader = DataLoader(train=True)
val_loader = DataLoader(train=False)

train_x = train_loader.data
train_y = train_loader.labels
print('\ntrain_x: {} - train_y: {}'.format(train_x.shape, train_y.shape))

val_x = val_loader.data
val_y = val_loader.labels
print('val_x: {} - val_y: {}'.format(val_x.shape, val_y.shape))

print('\n----------MODEL FITTING----------\n')
cars_net.fit(train_x=train_x, train_y=train_y,
             epochs=utils.N_EPOCHS, batch_size=utils.BATCH_SIZE,
             val_x=val_x, val_y=val_y)
