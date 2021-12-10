from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

from src.saved_models.mnist_dataset import mnist_dataset

'''
'''
class MNIST_SIMPLE(mnist_dataset):

    def __init__(self):
        super(MNIST_SIMPLE, self).__init__()

    def create_model(self, input_shape):
        model = Sequential()

        model.add(Dense(16, name='dense_1', input_dim=input_shape))
        model.add(Activation('relu', name='relu_1'))

        model.add(Dense(self.get_num_classes(), name='dense_n'))
        model.add(Activation('softmax', name='softmax'))

        self.set_model(model)
        return model


if __name__ == '__main__':
    # train model
    mnist = MNIST_SIMPLE()
    mnist.set_num_classes(10)

    mnist.train_model(train=True,
                      kernel_path='/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/mnist_simple.h5',
                      model_path='/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/mnist_simple.json',
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
