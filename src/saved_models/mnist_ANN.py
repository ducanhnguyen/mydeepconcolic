from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import to_categorical


def Model1(train=False, path=''):
    nb_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the result, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)
    # create ANN model
    model = Sequential()

    model.add(Dense(20, name='dense_1', input_dim=img_rows * img_cols))
    model.add(Activation('relu', name='relu_1'))

    model.add(Dense(15, name='dense_2'))
    model.add(Activation('relu', name='relu_2'))

    model.add(Dense(nb_classes, name='dense_n'))
    model.add(Activation('softmax', name='softmax'))

    # train ANN model
    if train:
        batch_size = 128
        nb_epoch = 50

        # normalize training set and test set
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        #x_train /= 255
        #x_test /= 255

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        # compiling
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        # training
        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)

        # save model
        model.save_weights(path)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])

    else:
        # Load model from file
        model.load_weights(path)

        print_ = False
        if print_:
            print('Model mnist_LeNet1 loaded from file')
            print(f'Model {model}: {model.summary()}')

    return model


if __name__ == '__main__':
    Model1(train=True, path='./mnist_ANN.h5')
