import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential

from src.deepconcolic import logger
from src.model_loader import initialize_dnn_model_from_name
from src.utils import utilities

epsilon = 0.000001


def loss(classifier: Sequential, tf_x, tf_w, tf_y_true, c):
    tf_adv = 1 / 2 * (tf.math.tanh(tf_w + epsilon) + 1)
    tf_prediction = tf.keras.losses.categorical_crossentropy(
        classifier(tf_adv)[0],
        tf_y_true)
    tf_dist = tf.keras.losses.mean_squared_error(tf_adv, tf_x)
    return tf_dist - c * tf_prediction


if __name__ == '__main__':
    name_model = "mnist_deepcheck"
    logger.debug("initialize_dnn_model")
    model_object = initialize_dnn_model_from_name(name_model)
    classifier = model_object.get_model()
    trainX = model_object.get_Xtrain()
    trainy = utilities.category2indicator(model_object.get_ytrain())

    x_train = trainX[1020].reshape(1, 784)
    y_true = trainy[1020]

    losses = []
    iters = []
    lr = 0.5

    w = tf.math.atanh((np.clip(x_train, 0.00001, 0.99999)) * 2 - 1).numpy()
    for iter in range(0, 10001):
        # gradient
        tf_w = tf.convert_to_tensor(w, dtype='float64')
        with tf.GradientTape() as tape:
            tape.watch(tf_w)
            final_loss = loss(classifier, x_train, tf_w, y_true, c=0.5)
            gradient = tape.gradient(final_loss, tf_w)

        # update
        loss_value = final_loss.numpy()[0]

        w = w - gradient.numpy() * lr
        x_adv = 1 / 2 * (tf.math.tanh(w) + 1)
        x_adv = x_adv.numpy()[0]
        pred = classifier(x_adv.reshape(1, 784))
        pred = pred.numpy()[0]

        if iter % 100 == 0:
            print(f'iter = {iter}: loss = {loss_value}')
            iters.append(iter)
            losses.append(loss_value)
            print(f'adv label = {np.argmax(pred)}')
            print(f'true label = {np.argmax(y_true)}')

        if np.argmax(pred) != np.argmax(y_true):
            break

    plt.plot(iters, losses, label='Training Loss')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    # plt.legend()
    plt.show()

    utilities.show_two_images(x_28_28_left=x_train.reshape(28, 28),
                              x_28_28_right=x_adv.reshape(28, 28),
                              display=True)
