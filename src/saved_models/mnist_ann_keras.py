from keras.layers import Dense, Activation
from keras.models import Sequential

from src.abstract_dataset import *
import pandas as pd

class MNIST(abstract_dataset):

    def __init__(self):
        super(MNIST, self).__init__()

    def read_data(self, trainset_path, testset_path):
        X = pd.read_csv(trainset_path).to_numpy()
        limit = 40000

        self.set_Xtrain(X[:limit, 1:]/255)
        self.set_ytrain(X[:limit, 0].reshape(-1))

        self.set_Xtest(X[limit:, 1:]/255)
        self.set_ytest(X[limit:, 0].reshape(-1))

        return self.get_Xtrain(), self.get_ytrain(), self.get_Xtest(), self.get_ytest()

    def create_model(self, input_shape):
        model = Sequential()

        model.add(Dense(32, name='dense_1', input_dim=input_shape))
        model.add(Activation('relu', name='relu_1'))

        model.add(Dense(16, name='dense_2'))
        model.add(Activation('relu', name='relu_2'))

        model.add(Dense(self.get_num_classes(), name='dense_n'))
        model.add(Activation('softmax', name='softmax'))

        self.set_model(model)
        return model


if __name__ == '__main__':
    # train model
    mnist = MNIST()
    mnist.set_num_classes(10)

    '''
    mnist.train_model(train=True, kernel_path='../saved_models/mnist_ann_keras.h5',  model_path='../saved_models/mnist_ann_keras.json',
                      training_path='../../digit-recognizer/train.csv', testing_path='../../digit-recognizer/test.csv')
    '''
    # load model
    model = mnist.load_model(weight_path='../saved_models/mnist_ann_keras.h5', structure_path='../saved_models/mnist_ann_keras.json',
                             trainset_path='../../digit-recognizer/train.csv')
    mnist.read_data(trainset_path='../../digit-recognizer/train.csv', testset_path='../../digit-recognizer/test.csv')
    print(model.layers)

    # plot an observation
    import matplotlib.pyplot as plt
    x_train, y_train = mnist.get_an_observation(index=516)
    img = x_train.reshape(28, 28)
    print(y_train)
    plt.imshow(img, cmap='gray')
    plt.show()