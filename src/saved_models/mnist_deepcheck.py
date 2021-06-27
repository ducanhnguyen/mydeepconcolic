import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential

from src.mnist_dataset import mnist_dataset

'''
Overall training score: 0.10900241881608963
Accuracy on train set: 0.9676166772842407
Overall test score: 0.22557860612869263
Accuracy on test set: 0.9440000057220459
'''
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
class MNIST_DEEPCHECK(mnist_dataset):

    def __init__(self):
        super(MNIST_DEEPCHECK, self).__init__()

    def create_model(self, input_shape):
        model = Sequential()

        model.add(Dense(10, name='dense_1', input_dim=input_shape))
        model.add(Activation('relu', name='relu_1'))

        model.add(Dense(10, name='dense_2'))
        model.add(Activation('relu', name='relu_2'))

        model.add(Dense(10, name='dense_3'))
        model.add(Activation('relu', name='relu_3'))

        model.add(Dense(self.get_num_classes(), name='dense_n'))
        model.add(Activation('softmax', name='softmax'))

        self.set_model(model)
        return model


if __name__ == '__main__':
    # train model
    mnist = MNIST_DEEPCHECK()
    mnist.set_num_classes(10)

    mnist.train_model(train=True,
                      kernel_path='/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/mnist_deepcheck.h5',
                      model_path='/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/mnist_deepcheck.json',
                      training_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/train.csv',
                      testing_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/test.csv',
                      nb_epoch=100)

    #
    # # plot an observation
    # import matplotlib.pyplot as plt
    # x_train, y_train = mnist.get_an_observation(index=516)
    # img = x_train.reshape(28, 28)
    # plt.imshow(img, cmap='gray')
    # plt.title(f'A sample')
    # plt.show()
