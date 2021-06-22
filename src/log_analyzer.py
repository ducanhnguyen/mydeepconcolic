import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def is_int(s: str):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_float(s: str):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_data(path):
    arr = []

    if os.path.exists(path):
        with open(path, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            next(csv_reader, None)

            for row in csv_reader:
                arr_row = []
                for item in row:
                    if is_float(item):
                        arr_row.append(float(item))
                    else:
                        arr_row.append(None)
                arr.append(arr_row)
            # print(arr)
        csv_file.close()
    return arr


def analyze_by_threshold(path):
    print(f'analyze {path}')
    arr = read_data(path)
    IDX_first_largest_prob = 8
    IDX_second_largest_prob = 9
    IDX_third_largest_prob = 10
    IDX_fourth_largest_prob = 11
    IDX_fifth_largest_prob = 12
    IDX_last_largest_prob = 13

    IDX_seed = 0
    IDX_adv_label = 6
    num_adv_arr = []
    total_samples = []

    threshold_arr = np.arange(0, 100, 0.1)
    for threshold in threshold_arr:
        num_adv = 0
        n_sample = 0

        for row in arr:
            # score = 1 * np.abs(row[IDX_first_largest_prob] - row[IDX_second_largest_prob]) + \
            #         1* np.abs(row[IDX_second_largest_prob] - row[IDX_third_largest_prob]) + \
            #         1 * np.abs(row[IDX_third_largest_prob] - row[IDX_third_largest_prob])
            score = np.abs(row[IDX_first_largest_prob] - row[IDX_second_largest_prob])
            if score <= threshold:
                if row[IDX_adv_label] is not None:
                    num_adv += 1
                n_sample += 1
        # print(f'threshold = {threshold}:  #adv = {num_adv}, # samples = {n_sample}')
        num_adv_arr.append(num_adv)
        total_samples.append(n_sample)
        if n_sample == 10000:
            break

    num_adv_arr = np.asarray(num_adv_arr)
    # num_adv_arr = np.round(num_adv_arr / num_adv_arr[len(num_adv_arr) - 1] * 100, 1)

    total_samples = np.asarray(total_samples)
    total_samples = np.round(total_samples / total_samples[len(total_samples) - 1] * 100, 1)
    return num_adv_arr, total_samples, threshold_arr


def analyze_randomly(path):
    print(f'analyze {path}')
    arr = read_data(path)
    IDX_adv_label = 6
    num_adv_arr = []
    total_samples = []

    import random
    random.shuffle(arr)
    num_adv = 0
    n_sample = 0

    for row in arr:
        if row[IDX_adv_label] is not None:
            num_adv += 1
        n_sample += 1
        # print(f'threshold = {threshold}:  #adv = {num_adv}, # samples = {n_sample}')
        num_adv_arr.append(num_adv)
        total_samples.append(n_sample)

    num_adv_arr = np.asarray(num_adv_arr)
    # num_adv_arr = np.round(num_adv_arr / num_adv_arr[len(num_adv_arr) - 1] * 100, 1)

    total_samples = np.asarray(total_samples)

    if len(total_samples) >= 1:
        total_samples = np.round(total_samples / total_samples[len(total_samples) - 1] * 100, 1)
    else:
        total_samples = None
        num_adv_arr = None
    return num_adv_arr, total_samples


def plot_n_pixel_attack_randomly_vs_directly(type_of_attack):  # simard = M1, ann-keras = m2, simple= m3, deepcheck = m4
    # when not using sample ranking algorithm
    num_adv_arr, total_samples = analyze_randomly(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_simard/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '-k', linewidth=1, label='M1 (random)')

    num_adv_arr, total_samples = analyze_randomly(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_ann_keras/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '-sk', linewidth=1, markevery=250, markersize=3, label='M2 (random)')

    num_adv_arr, total_samples = analyze_randomly(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_simple/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '->k', linewidth=1, markevery=250, markersize=3, label='M3 (random)')

    num_adv_arr, total_samples = analyze_randomly(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_deepcheck/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '-.k', linewidth=1, label='M4 (random)')

    # when using sample ranking algorithm
    num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_simard/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '-g', linewidth=1, label='M1 (ranking)')

    num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_ann_keras/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '-sg', linewidth=1, markevery=0.05, markersize=3, label='M2 (ranking)')

    num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_simple/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '->g', linewidth=1, markevery=0.05, markersize=3, label='M3 (ranking)')

    num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_deepcheck/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '-.g', linewidth=1, label='M4 (ranking)')

    plt.xlabel('% of attacking samples')
    plt.ylabel('# adversaries')
    plt.tight_layout()
    # plt.legend(fontsize=10, ncol=2,handleheight=2, labelspacing=0.05, loc='lower right')
    plt.legend('',frameon=False)
    plt.show()


if __name__ == '__main__':
    plot_n_pixel_attack_randomly_vs_directly("bestFeatureAttack_delta255_secondLabelTarget")
