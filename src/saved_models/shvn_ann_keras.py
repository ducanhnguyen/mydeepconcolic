import scipy.io
from keras.layers import Dense, Activation
from keras.models import Sequential
from skimage import color

from src.abstract_dataset import *


class SHVN(ABSTRACT_DATASET):

    def __init__(self):
        super(SHVN, self).__init__()

    def create_model(self, input_shape):
        assert (input_shape > 0)
        model = Sequential()

        # model.add(Dense(256, name='dense_1', input_dim=input_shape))
        # model.add(Activation('relu', name='relu_1'))

        model.add(Dense(128, name='dense_2'))
        model.add(Activation('relu', name='relu_2'))

        model.add(Dense(64, name='dense_3'))
        model.add(Activation('relu', name='relu_3'))

        model.add(Dense(32, name='dense_4'))
        model.add(Activation('relu', name='relu_4'))

        model.add(Dense(16, name='dense_5'))
        model.add(Activation('relu', name='relu_5'))

        model.add(Dense(self.get_num_classes(), name='dense_n'))
        model.add(Activation('softmax', name='softmax'))

        self.set_model(model)
        return model

    def read_data(self, training_path, testing_path):
        if training_path != None:
            mat = scipy.io.loadmat(training_path)
            y_train = mat['y'].reshape(-1)
            y_train[y_train == 10] = 0
            self.set_ytrain(y_train)

            Xtrain_bw = self.flatten_rgb(mat['X']).astype('float32')
            self.set_Xtrain(Xtrain_bw)

        if testing_path != None:
            mat = scipy.io.loadmat(testing_path)
            y_test = mat['y'].reshape(-1)
            y_test[y_test == 10] = 0  # digit 10 is zero
            self.set_ytest(y_test)

            Xtest_bw = self.flatten_rgb(mat['X']).astype('float32')
            self.set_Xtest(Xtest_bw)

        return self.get_Xtrain(), self.get_ytrain(), self.get_Xtest(), self.get_ytest()

    def flatten_bw(self, X):
        W, H, C, N = X.shape
        Xbw = np.zeros((N, W * H))

        for i in range(N):
            bw = np.array(color.rgb2gray(X[:, :, :, i]))
            Xbw[i] = bw.flatten()

        return Xbw

    def flatten_rgb(self, X):
        W, H, C, N = X.shape
        Xbw = np.zeros((N, W * H * C))

        for i in range(N):
            bw = np.array(X[:, :, :, i])
            Xbw[i] = np.reshape(a=bw, newshape=(W * H * C))

        Xbw /= 255

        return Xbw


if __name__ == '__main__':
    shvn = SHVN()
    shvn.set_num_classes(10)

    shvn.train_model(train=True, kernel_path='./shvn_ann_keras.h5',
                     model_path='./shvn_ann_keras.json',
                     training_path='../SVHN/train_32x32.mat', testing_path='../SVHN/test_32x32.mat')
    '''

    model = shvn.load_model(kernel_path='./shvn_ann_keras_1.h5', model_path='./shvn_ann_keras_1.json',
                            training_path='../SVHN/train_32x32.mat')
    shvn.read_data(training_path='../SVHN/train_32x32.mat', testing_path='../SVHN/test_32x32.mat')

    import matplotlib.pyplot as plt
    x_train, y_train = shvn.get_an_observation(index=13)
    img = x_train.reshape(32, 32,3)
    plt.imshow(img)
    plt.show()
    '''