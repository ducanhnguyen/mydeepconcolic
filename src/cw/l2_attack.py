import csv as csv

import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, model_from_json

from src.mnist_dataset import mnist_dataset
from src.utils import utilities

# Configure

N_ATTACKING_SEED = 10000

# ATTACKED_MODEL_H5 = "/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/mnist_simard.h5"
# ATTACKED_MODEL_JSON = "/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/mnist_simard.json"
# shape = (1, 784)
# OUT_FOLDER = '/Users/ducanhnguyen/Documents/mydeepconcolic/result/cw/mnist_simard/c=0.5, 500 iters, sdg'

ATTACKED_MODEL_H5 = "/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/rivf/pretrained_mnist_cnn1.h5"
ATTACKED_MODEL_JSON = None
shape = (1, 28, 28, 1)
OUT_FOLDER = '/Users/ducanhnguyen/Documents/mydeepconcolic/result/cw/rivf'

LOSS = "CW_LOSS"  # "CW_LOSS", "TRADITIONAL_LOSS"

# For SGD
if LOSS == "TRADITIONAL_LOSS":
    C = 10000
    SGD_LEARNING_RATE = 0.5
    N_MAX_ITER = 500
elif LOSS == "CW_LOSS":
    C = 0.5
    SGD_LEARNING_RATE = 0.25
    N_MAX_ITER = 500


def get_presoftmax_classifier(classifier: Sequential):
    before_softmax = classifier.layers[-2]
    presoftmax_classifier = Model(inputs=classifier.inputs,
                                  outputs=before_softmax.output)
    return presoftmax_classifier


def cw_loss(presoftmax_classifier: Sequential, tf_x, tf_w, tf_y_true, c):
    tf_adv = 1 / 2 * (tf.math.tanh(tf_w) + 1)
    presoftmax = presoftmax_classifier(tf_adv)[0]
    true_label = np.argmax(tf_y_true)

    max = presoftmax[0]
    for idx in range(tf_y_true.shape[0]):
        if idx != true_label:
            if presoftmax[idx] > max:
                max = presoftmax[idx]

    # tf_prediction = tf.math.maximum(max - presoftmax[true_label], 0)
    tf_prediction = max - presoftmax[true_label]

    tf_prediction = tf.cast(tf_prediction, dtype='float64')
    tf_dist = tf.keras.losses.mean_squared_error(tf_adv, tf_x)
    return tf_dist - c * tf_prediction


def traditional_loss(aftersoftmax: Sequential, tf_x, tf_w, tf_y_true, c):
    tf_adv = 1 / 2 * (tf.math.tanh(tf_w) + 1)
    tf_prediction = tf.keras.losses.categorical_crossentropy(
        aftersoftmax(tf_adv)[0],
        tf_y_true)
    tf_dist = tf.keras.losses.mean_squared_error(tf_adv, tf_x)
    return tf_dist - c * tf_prediction


def attack(x_train_normalized, y_true, seed_idx, classifier, LOSS):
    w = tf.math.atanh(x_train_normalized * 2 - 1).numpy()  # apply `change of variable'

    losses = []
    iters = []
    final_found_adv = None
    final_label_adv = None
    for iter in range(0, N_MAX_ITER):
        # gradient
        tf_w = tf.convert_to_tensor(w, dtype='float64')
        with tf.GradientTape() as tape:
            tape.watch(tf_w)
            if LOSS == "CW_LOSS":
                loss = cw_loss(classifier, x_train, tf_w, y_true, c=C)
            elif LOSS == "TRADITIONAL_LOSS":
                loss = traditional_loss(classifier, x_train, tf_w, y_true, c=C)
            grad = tape.gradient(loss, tf_w)

        # update
        w = w - grad.numpy() * SGD_LEARNING_RATE
        x_adv = (1 / 2 * (tf.math.tanh(w) + 1)).numpy()[0]
        # x_adv = np.round(x_adv * 255) / 255
        pred = classifier(x_adv.reshape(shape)).numpy()[0]

        if iter % 1 == 0:
            loss_value = loss.numpy()[0]
            print(
                f'iter = {iter}: loss = {loss_value}, adv label = {np.argmax(pred)}, true label = {np.argmax(y_true)}')
            iters.append(iter)
            losses.append(loss_value)

        adv_label = np.argmax(pred)
        true_label = np.argmax(y_true)
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
        linf = utilities.compute_linf(x_train.reshape(-1), final_found_adv.reshape(-1))
        minimum_change = utilities.compute_minimum_change(x_train.reshape(-1), final_found_adv.reshape(-1))

        # export 1
        utilities.show_two_images(x_28_28_left=x_train.reshape(28, 28),
                                  x_28_28_right=final_found_adv.reshape(28, 28),
                                  left_title=f'index = {seed_idx}\ntrue label = {true_label}',
                                  right_title=f'adv label = {final_label_adv}\nl0 = {l0}\nl2 = {l2}',
                                  display=False,
                                  path=OUT_FOLDER + '/' + str(seed_idx) + "_comparison")
        # export 2
        summary_path = OUT_FOLDER + '/summary.csv'
        with open(summary_path, mode='a') as f:
            seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            seed.writerow([seed_idx, l0, l2, linf, minimum_change, true_label,
                           adv_label, None, None,
                           None, None, None,
                           None,
                           None])

        # export 3
        candidate_adv_csv_path = OUT_FOLDER + f'/{seed_idx}.csv'
        with open(candidate_adv_csv_path, mode='w') as f:
            seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # seed.writerow(np.round(final_found_adv*255).astype(dtype=int))
            seed.writerow(final_found_adv * 255)


def load_model(ATTACKED_MODEL_JSON, ATTACKED_MODEL_H5):
    if ATTACKED_MODEL_JSON is not None:
        # Method 1
        json_file = open(ATTACKED_MODEL_JSON, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_object = model_from_json(loaded_model_json)
        model_object.load_weights(ATTACKED_MODEL_H5)
    else:
        # Method 2
        model_object = keras.models.load_model(filepath=ATTACKED_MODEL_H5)
    return model_object


def load_dataset():
    trainX, trainy, _, _ = mnist_dataset().read_data(
        trainset_path="/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/train.csv",
        testset_path="/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/test.csv")
    trainy = utilities.category2indicator(trainy)
    return trainX, trainy


if __name__ == '__main__':
    # Load model
    classifier = load_model(ATTACKED_MODEL_JSON, ATTACKED_MODEL_H5)
    classifier.summary()
    trainX, trainy = load_dataset()

    if LOSS == "CW_LOSS":
        classifier = get_presoftmax_classifier(classifier)

    with open(OUT_FOLDER + '/summary.csv', mode='w') as f:
        seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        seed.writerow(['seed', 'l0', 'l2', 'l_inf', 'minimum_change', 'true_label', 'adv_label',
                       'position_adv_label_in_original_pred', 'first_largest',
                       'second_largest', 'third_largest', 'fourth_largest', 'fifth_largest', 'last_largest'])

    for idx in range(0, N_ATTACKING_SEED):
        print(f'Attacking seed {idx}')
        x_train = trainX[idx].reshape(shape)
        y_true = trainy[idx]

        pred = classifier(x_train.reshape(shape))[0]
        if np.argmax(pred) == np.argmax(y_true):
            x_train_normalized = np.clip(x_train, 0.00001, 0.99999)  # we are unable to atanh(-1) and atanh(1)
            attack(x_train_normalized, y_true, idx, classifier, LOSS)
        else:
            print('Ignore')
