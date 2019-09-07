'''
LeNet-1: https://medium.com/@sh.tsang/paper-brief-review-of-lenet-1-lenet-4-lenet-5-boosted-lenet-4-image-classification-1f5f809dbf17
'''

# usage: python MNISTModel1.py - train the model

from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Activation, Flatten
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras import backend as K

def Model1(train=False, path=''):
    nb_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    input_shape = (img_rows, img_cols, 1)
    input_tensor = Input(shape=input_shape)

    # the result, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # create CNN model

    model = Sequential()

    model.add(Conv2D(4, kernel_size=(5, 5), input_shape=input_shape, name='conv1'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))

    model.add(Conv2D(12, (5, 5), name='conv2'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(100, name='dense_x'))
    model.add(Activation('relu'))

    model.add(Dense(nb_classes, name='before_softmax'))
    model.add(Activation('softmax'))

    # train CNN model
    if train:
        batch_size = 128
        nb_epoch = 2

        # normalize training set and test set
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

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
        if print:
            print('Model mnist_LeNet1 loaded from file')
            print(f'Model {model}: {model.summary()}')

            print()
            # get weights of a conv layer
            first_conv_layer = model.get_layer(name='conv1')
            weights = first_conv_layer.get_weights()
            print(
                f'first layer - weights (filter_width, filter_height, channel_in, channel_out) = {weights[0].shape}')
            print(f'first layer - biases (channel_out, 1) = {weights[1].shape}')

            # get output of a layer - way 1
            intermediate_layer_name = 'before_softmax'
            intermediate_layer_model = Model(inputs=model.inputs, outputs=model.get_layer(intermediate_layer_name).output)
            prediction = intermediate_layer_model.predict(x_test[0:1])
            print(f'output of layer - way 1 = {prediction}')

            # get output of a layer - way 2
            get_3rd_layer_output = K.function(inputs = [model.layers[0].input], outputs = [model.get_layer(intermediate_layer_name).output])
            prediction = get_3rd_layer_output([x_test[0:1]])[0]
            print(f'output of layer - way 2= {prediction}')

            # get weights of a conv layer
            dense_layer = model.get_layer(name='before_softmax')
            weights = dense_layer.get_weights()
            print(
                f'dense layer = {weights[0].shape}')

    return model


if __name__ == '__main__':
    Model1(train=False, path = './mnist_LeNet1.h5')
