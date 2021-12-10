from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

from src.saved_models.fashionmnist_dataset import fashionmnist_dataset
from src.saved_models.mnist_dataset import mnist_dataset

import os
'''
Overall training score: 0.012811495922505856
Accuracy on train set: 0.9957833290100098
Overall test score: 0.9800440669059753
Accuracy on test set: 0.8952000141143799
'''
os.environ['KMP_DUPLICATE_LIB_OK']='True'
class FASHIONMNIST_SIMARD(fashionmnist_dataset):

    def __init__(self):
        super(FASHIONMNIST_SIMARD, self).__init__()

    def create_model(self, input_shape):
        model = Sequential()

        model.add(Dense(128, name='dense_1', input_dim=input_shape))
        model.add(Activation('relu', name='relu_1'))

        model.add(Dense(64, name='dense_2'))
        model.add(Activation('relu', name='relu_2'))

        model.add(Dense(32, name='dense_3'))
        model.add(Activation('relu', name='relu_3'))

        model.add(Dense(16, name='dense_4'))
        model.add(Activation('relu', name='relu_4'))

        model.add(Dense(self.get_num_classes(), name='dense_n'))
        model.add(Activation('softmax', name='softmax'))

        self.set_model(model)
        return model


if __name__ == '__main__':
    # train model
    mnist = FASHIONMNIST_SIMARD()
    mnist.set_num_classes(10)

    mnist.train_model(train=True,
                      kernel_path='/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/fashionmnist_simard.h5',
                      model_path='/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/fashionmnist_simard.json',
                      training_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/fashion-mnist/fashion-mnist_train.csv',
                      testing_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/fashion-mnist/fashion-mnist_test.csv',
                      nb_epoch=300)

    #
    # # plot an observation
    # import matplotlib.pyplot as plt
    # x_train, y_train = mnist.get_an_observation(index=516)
    # img = x_train.reshape(28, 28)
    # plt.imshow(img, cmap='gray')
    # plt.title(f'A sample')
    # plt.show()
