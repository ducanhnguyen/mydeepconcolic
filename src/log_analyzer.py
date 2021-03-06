import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from src.deepconcolic import initialize_dnn_model
from src.utils.utilities import compute_l0, compute_l2, compute_linf, compute_minimum_change


def is_int(s: str):
    try:
        int(s)
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    DIRECTORY = '/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_1'

    # get path of all adv files
    adv_arr_path = []
    for filename in os.listdir(DIRECTORY):
        if filename.endswith(".csv"):
            adv_arr_path.append(os.path.join(DIRECTORY, filename))
        else:
            continue

    # load pixel
    adv_dict = {} # key: seed index, value: array of pixels
    seed_index_arr = []
    for idx in range(len(adv_arr_path)):
        seed_index = os.path.basename(adv_arr_path[idx]).replace(".csv", "")
        if not is_int(seed_index):
            continue
        seed_index = int(seed_index)
        seed_index_arr.append(seed_index)

        with open(adv_arr_path[idx]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            line_count = 0
            for row in csv_reader:
                adv_dict[seed_index] = np.asarray(row).astype(int)

    # load model
    model_object = initialize_dnn_model()
    X_train = model_object.get_Xtrain()  # for MNIST: shape = (42000, 784)
    l0_arr = []
    l2_arr = []
    linf_arr = []
    seed_arr = []
    minimum_change_arr = []
    true_label_arr = []
    adv_label_arr = []
    position_adv_arr = []

    # kk = 0
    for seed_index in seed_index_arr:
        # kk += 1
        # if kk == 10:
        #     break

        seed_arr.append(seed_index)

        ori = X_train[seed_index]  # [0..1]
        adv = adv_dict[seed_index]  # [0..255]

        # compute distance of adv and its ori
        l0 = compute_l0((ori * 255).astype(int), adv)
        l0_arr.append(l0)

        l2 = compute_l2(ori, adv / 255)
        l2_arr.append(l2)

        linf = compute_linf(ori, adv / 255)
        linf_arr.append(linf)

        minimum_change = compute_minimum_change(ori, adv / 255)
        minimum_change_arr.append(minimum_change)

        # compute prediction
        true_pred = model_object.get_model().predict(ori.reshape(-1, 784))[0]
        true_label = np.argmax(true_pred)
        true_label_arr.append(true_label)

        adv_pred = model_object.get_model().predict(adv.reshape(-1, 784))[0]
        adv_label = np.argmax(adv_pred)
        adv_label_arr.append(adv_label)
        if true_label == adv_label: # just confirm
            print("PROBLEM!")
            exit

        # position of adv in the probability of original prediction
        position_adv = -9999999999999
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        _, labels_sorted_by_prob = zip(*sorted(zip(true_pred, labels), reverse=True))
        for j in range(len(labels_sorted_by_prob)):
            if labels_sorted_by_prob[j] == adv_label:
                position_adv = j + 1  # start from 1
                break
        position_adv_arr.append(position_adv)

        # export image comparison
        fig = plt.figure()
        nrow = 1
        ncol = 2
        ori = ori.reshape(28, 28)
        fig1 = fig.add_subplot(nrow, ncol, 1)
        fig1.title.set_text(f'origin \nindex = {seed_index},\nlabel {true_label}, acc = {true_pred[true_label]}')
        plt.imshow(ori, cmap="gray")

        adv = adv.reshape(28, 28)
        fig2 = fig.add_subplot(nrow, ncol, 2)
        fig2.title.set_text(
            f'adv\nlabel {adv_label}, acc = {adv_pred[adv_label]}\n l0 = {l0}, l2 = ~{np.round(l2, 2)}')
        plt.imshow(adv, cmap="gray")

        png_comparison_image_path = DIRECTORY + f'/{seed_index}_comparison.png'
        plt.savefig(png_comparison_image_path, pad_inches=0, bbox_inches='tight')

    # export to csv
    summary_path = DIRECTORY + '/summary.csv'
    with open(summary_path, mode='w') as f:
        seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        seed.writerow(['seed', 'l0', 'l2', 'l_inf', 'minimum_change', 'true_label', 'adv_label',
                       'position_adv_label_in_original_pred'])
        for i in range(len(l0_arr)):
            seed.writerow([seed_arr[i], l0_arr[i], l2_arr[i], linf_arr[i], minimum_change_arr[i], true_label_arr[i],
                           adv_label_arr[i], position_adv_arr[i]])
