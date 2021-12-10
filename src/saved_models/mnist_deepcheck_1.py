import pandas as pd
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

from src.saved_models.mnist_dataset import mnist_dataset

'''
Overall training score: 0.13488441705703735
Accuracy on train set: 0.9609000086784363
Overall test score: 0.2398766577243805
Accuracy on test set: 0.9399999976158142

'''
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
class MNIST_DEEPCHECK_1(mnist_dataset):

    def __init__(self):
        super(MNIST_DEEPCHECK_1, self).__init__()

    def create_model(self, input_shape):
        model = Sequential()

        model.add(Dense(10, name='dense_1', input_dim=input_shape))
        model.add(Activation('relu', name='relu_1'))



        model.add(Dense(self.get_num_classes(), name='dense_n'))
        model.add(Activation('softmax', name='softmax'))

        self.set_model(model)
        return model


if __name__ == '__main__':
    # train model
    mnist = MNIST_DEEPCHECK_1()
    mnist.set_num_classes(10)
    import os
    mnist.train_model(train=True,
                      kernel_path='../../src/saved_models/mnist_deepcheck_1.h5',
                      model_path='../../src/saved_models/mnist_deepcheck_1.json',
                      training_path='../../dataset/digit-recognizer/train.csv',
                      testing_path='../../dataset/digit-recognizer/test.csv',
                      nb_epoch=100)

    #
    # # plot an observation
    # import matplotlib.pyplot as plt
    # x_train, y_train = mnist.get_an_observation(index=516)
    # img = x_train.reshape(28, 28)
    # plt.imshow(img, cmap='gray')
    # plt.title(f'A sample')
    # plt.show()
