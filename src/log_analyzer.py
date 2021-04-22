import csv

import matplotlib.pyplot as plt
import numpy as np
import os

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
    total_samples = np.round(total_samples / total_samples[len(total_samples) - 1] * 100, 1)
    return num_adv_arr, total_samples


def plot_n_pixel_attack_randomly_vs_directly():
    # when not using sample ranking algorithm
    num_adv_arr, total_samples = analyze_randomly(
        '/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_secondLabelTarget/result/mnist_simard/summary.csv')
    plt.plot(total_samples, num_adv_arr, '--b', label='mnist_simard')

    num_adv_arr, total_samples = analyze_randomly(
        '/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_secondLabelTarget/result/mnist_simple/summary.csv')
    plt.plot(total_samples, num_adv_arr, '--g', label='mnist_simple')

    num_adv_arr, total_samples = analyze_randomly(
        '/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_secondLabelTarget/result/mnist_deepcheck/summary.csv')
    plt.plot(total_samples, num_adv_arr, '--r', label='mnist_deepcheck')

    num_adv_arr, total_samples = analyze_randomly(
        '/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_secondLabelTarget/result/mnist_ann_keras/summary.csv')
    plt.plot(total_samples, num_adv_arr, '--k', label='mnist_ann_keras')

    # when using sample ranking algorithm
    num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
        '/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_secondLabelTarget/result/mnist_simard/summary.csv')
    plt.plot(total_samples, num_adv_arr, 'b')

    num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
        '/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_secondLabelTarget/result/mnist_simple/summary.csv')
    plt.plot(total_samples, num_adv_arr, 'g')

    num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
        '/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_secondLabelTarget/result/mnist_deepcheck/summary.csv')
    plt.plot(total_samples, num_adv_arr, 'r')

    num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
        '/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_secondLabelTarget/result/mnist_ann_keras/summary.csv')
    plt.plot(total_samples, num_adv_arr, 'k')


    #
    plt.xlabel('% of attacking samples')
    plt.ylabel('# adversaries')
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_n_pixel_attack_randomly_vs_directly()
    # num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
    #     '/Users/ducanhnguyen/Documents/mydeepconcolic/result/changeOnEdge_delta100_upperBound/result/mnist_deepcheck/summary.csv')
    # for u, v in zip(num_adv_arr, total_samples):
    #     print(f'%adv = {u}, %total = {v}')
