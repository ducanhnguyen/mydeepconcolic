import keras
import numpy as np
import tensorflow as tf

from src.ae_attack_border.ae_reader import get_X_attack
from src.ae_attack_border.data import wrongseeds_AlexNet, AlexNet_0_5
import matplotlib.pyplot as plt
(X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
X_train = X_train / 255
N_ATTACKING_SAMPLES = 2000
N_CLASSES = 10


def rank_a(dnn, X_attack, adv_arr):
    Y_pred = dnn.predict(X_attack.reshape(-1, 28, 28, 1))
    sum = np.sum(Y_pred, axis=0)

    sum, labels_sorted = zip(*sorted(zip(sum, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), reverse=True))

    adv_sort = []
    for item in np.asarray(labels_sorted):
        adv_sort.append(adv_arr[item])

    # labels_sorted meaning: specific label
    return labels_sorted, adv_sort


def rank_b(dnn, X_attack, adv_arr):
    Y_pred = dnn.predict(X_attack.reshape(-1, 28, 28, 1))

    labels_sorted_arr = []
    for prob in Y_pred:
        _, labels_sorted = zip(*sorted(zip(prob, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), reverse=True))
        labels_sorted_arr.append(labels_sorted)

    labels_sorted_arr = np.asarray(labels_sorted_arr)


    label_matrix = np.zeros(shape=(N_CLASSES, N_CLASSES))
    label_matrix_acc = np.zeros(shape=(N_CLASSES, N_CLASSES))
    for idx in range(N_CLASSES):
        col = labels_sorted_arr[:, idx]
        for jdx in range(N_CLASSES):
            label_matrix[jdx, idx] = np.sum(col == jdx)
            if idx >= 1:
                label_matrix_acc[jdx, idx] = label_matrix_acc[jdx, idx - 1] + np.sum(col == jdx)
            else:
                label_matrix_acc[jdx, idx] = np.sum(col == jdx)

    labels_sorted = []
    for idx in range(N_CLASSES):
        col = label_matrix_acc[:, idx]
        best_label = 0
        largest_count = 0
        for k, v in enumerate(col):
            if k not in labels_sorted:
                if v > largest_count:
                    largest_count = v
                    best_label = k
        labels_sorted.append(best_label)

    labels_sorted = np.asarray(labels_sorted)

    adv_sort = []
    for idx in labels_sorted:
        adv_sort.append(adv_arr[idx])

    # labels_sorted meaning: 1st, 2nd, 3rd, etc.
    return labels_sorted, adv_sort


if __name__ == '__main__':
    ATTACKED_MODEL_H5 = f"/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/model/Alexnet.h5"
    arr = AlexNet_0_5
    WRONGSEED = wrongseeds_AlexNet
    dnn = keras.models.load_model(filepath=ATTACKED_MODEL_H5, compile=False)

    for ori in range(N_CLASSES):
        print(f"\nori label {ori}")
        X_attack = get_X_attack(X_train, y_train, WRONGSEED, ori, N_ATTACKING_SAMPLES)

        labels_sorted, adv_sort = rank_b(dnn, X_attack, arr[ori])
        print(f'label = {labels_sorted}')
        print(f'adv_arr = {adv_sort}')

        tmp_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        plt.bar(tmp_labels, adv_sort, color='maroon', width=0.4)
        plt.title(f'Alexnet (epsilon = 0.5, ori {ori})')
        plt.xticks(tmp_labels)
        plt.xlabel('tmp_labels')
        plt.ylabel('#adv')
        plt.show()
