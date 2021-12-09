import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical


def get_original_lenet5():
    # original lenet-5:
    # http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
    # https://en.wikipedia.org/wiki/LeNet#/media/File:Comparison_image_neural_networks.svg

    model = models.Sequential()
    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5),
                            activation='sigmoid', padding='same', input_shape=(28, 28, 1)))  # original version: sigmoid
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='sigmoid'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='sigmoid'))
    model.add(layers.Dense(84, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def get_modified_lenet5():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5),
                            activation='relu', padding='same', input_shape=(28, 28, 1)))  # original version: sigmoid
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def load_model(path):
    new_model = tf.keras.models.load_model(path)
    new_model.summary()


def train(model, modelpath, x_train, y_train, x_test, y_test):
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=5e-4))
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=300, callbacks=[callback], batch_size=512)
    model.save(filepath=modelpath)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train, num_classes=10)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_test = to_categorical(y_test, num_classes=10)

    train(get_original_lenet5(),
          '/Users/ducanhnguyen/Documents/mydeepconcolic/src/c_code/alexnet_origin',
          x_train, y_train, x_test, y_test)
