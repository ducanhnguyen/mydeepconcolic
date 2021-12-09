'''
Overall training score: 0.0019111856818199158
Accuracy on train set: 0.9994166493415833
Overall test score: 0.06345111131668091
Accuracy on test set: 0.9865000247955322
'''
from keras.layers import Dense, Activation
from keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.optimizer_v1 import SGD

from src.saved_models.mnist_dataset import mnist_dataset

import pandas as pd
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MNIST_CNN_MONDAY(mnist_dataset):

    def __init__(self):
        super(MNIST_CNN_MONDAY, self).__init__()

    def create_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))

        self.set_model(model)
        return model

    def read_data(self, trainset_path, testset_path):
        X = pd.read_csv(trainset_path).to_numpy()

        self.set_Xtrain(X[:, 1:].reshape(-1, 28, 28, 1) / self.NORMALIZATION_FACTOR)
        self.set_ytrain(X[:, 0].reshape(-1))
        print('training set: ' + str(len(X)) + ' samples')

        X = pd.read_csv(testset_path).to_numpy()
        self.set_Xtest(X[:, 1:].reshape(-1, 28, 28, 1)  / self.NORMALIZATION_FACTOR)
        self.set_ytest(X[:, 0].reshape(-1))
        print('test set: ' + str(len(X)) + ' samples')

        return self.get_Xtrain(), self.get_ytrain(), self.get_Xtest(), self.get_ytest()


if __name__ == '__main__':
    # train model
    mnist = MNIST_CNN_MONDAY()
    mnist.set_num_classes(10)

    mnist.train_model(train=True,
                      kernel_path='/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/mnist_cnn_monday.h5',
                      model_path='/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/mnist_cnn_monday.json',
                      training_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/train.csv',
                      testing_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/test.csv',
                      nb_epoch=20)

    #
    # # plot an observation
    # import matplotlib.pyplot as plt
    # x_train, y_train = mnist.get_an_observation(index=516)
    # img = x_train.reshape(28, 28)
    # plt.imshow(img, cmap='gray')
    # plt.title(f'A sample')
    # plt.show()
