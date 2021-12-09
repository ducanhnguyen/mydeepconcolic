from keras.layers import Dense, Activation
from keras.models import Sequential

from src.saved_models.mnist_dataset import mnist_dataset

'''
Model mnist_ann_keras:
    Overall training score: 0.0032980279065668583
    Accuracy on train set: 0.9990333318710327
    Overall test score: 0.28833839297294617
    Accuracy on test set: 0.9664999842643738
Model mnist_ann_keras_10k_first_samples_v2:
    Overall training score: 0.003136696992442012
    Accuracy on train set: 0.9990333318710327
    Overall test score: 0.3277113735675812
    Accuracy on test set: 0.9635000228881836

Model mnist_ann_keras_10k_first_samples_v3:
    Overall training score: 0.004398046527057886
    Accuracy on train set: 0.9984833598136902
    Overall test score: 0.3382343649864197
    Accuracy on test set: 0.9635999798774719

'''


class MNIST_ANN_KERAS(mnist_dataset):

    def __init__(self):
        super(MNIST_ANN_KERAS, self).__init__()

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
    mnist = MNIST_ANN_KERAS()
    mnist.set_num_classes(10)

    mnist.train_model(train=True,
                      kernel_path='/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/mnist_ann_keras.h5',
                      model_path='/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/mnist_ann_keras.json',
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
