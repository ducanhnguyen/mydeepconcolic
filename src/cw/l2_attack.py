import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential

from src.deepconcolic import logger
from src.model_loader import initialize_dnn_model_from_name
from src.utils import utilities


def cw_loss(classifier: Sequential, tf_x, tf_w, tf_y_true, c):
    tf_adv = 1 / 2 * (tf.math.tanh(tf_w) + 1)
    tf_prediction = tf.keras.losses.categorical_crossentropy(
        classifier(tf_adv)[0],
        tf_y_true)
    tf_dist = tf.keras.losses.mean_squared_error(tf_adv, tf_x)
    return tf_dist - c * tf_prediction


if __name__ == '__main__':
    # Configure
    ATTACKED_MODEL = "mnist_deepcheck"
    N_FEATURES = 784

    N_MAX_ITER = 10001
    ATTACKING_SEED_INDEX = 88
    SGD_LEARNING_RATE = 0.5

    # Attack
    logger.debug("initialize_dnn_model")
    model_object = initialize_dnn_model_from_name(ATTACKED_MODEL)
    classifier = model_object.get_model()
    trainX = model_object.get_Xtrain()
    trainy = utilities.category2indicator(model_object.get_ytrain())

    x_train = trainX[ATTACKING_SEED_INDEX].reshape(1, N_FEATURES)
    x_train_normalized = np.clip(x_train, 0, 0.99999)  # we are unable to atanh(-1) and atanh(1)
    w = tf.math.atanh(x_train_normalized * 2 - 1).numpy()  # apply `change of variable'

    y_true = trainy[ATTACKING_SEED_INDEX]

    losses = []
    iters = []
    final_found_adv = None
    final_label_adv = None
    for iter in range(0, N_MAX_ITER):
        # gradient
        tf_w = tf.convert_to_tensor(w, dtype='float64')
        with tf.GradientTape() as tape:
            tape.watch(tf_w)
            loss = cw_loss(classifier, x_train, tf_w, y_true, c=0.5)
            grad = tape.gradient(loss, tf_w)

        # update
        w = w - grad.numpy() * SGD_LEARNING_RATE
        x_adv = (1 / 2 * (tf.math.tanh(w) + 1)).numpy()[0]
        x_adv = np.round(x_adv * 255) / 255
        pred = classifier(x_adv.reshape(1, N_FEATURES)).numpy()[0]

        if iter % 1 == 0:
            loss_value = loss.numpy()[0]
            print(
                f'iter = {iter}: loss = {loss_value}, adv label = {np.argmax(pred)}, true label = {np.argmax(y_true)}')
            iters.append(iter)
            losses.append(loss_value)

        adv_label = np.argmax(y_true)
        true_label = np.argmax(pred)
        if adv_label != true_label:
            print('Found adv')
            final_found_adv = x_adv
            final_label_adv = adv_label
            # the label of adv is different from the true label
            # we can get a better quality adversary but I stop here
            # to reduce the computational cost
            break

    # visualize
    # plt.plot(iters, losses, label='Loss')
    # plt.xlabel('Iter')
    # plt.ylabel('Loss')
    # plt.show()

    if final_found_adv is not None:
        l2 = utilities.compute_l2(x_train, final_found_adv)
        l0 = utilities.compute_l0(x_train.reshape(-1), final_found_adv.reshape(-1), normalized=False)
        utilities.show_two_images(x_28_28_left=x_train.reshape(28, 28),
                                  x_28_28_right=final_found_adv.reshape(28, 28),
                                  right_title=f'adv label = {final_label_adv}\nl0 = {l0}\nl2 = {l2}',
                                  display=True)
