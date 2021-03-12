import csv

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


def get_analysis(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        arr = []
        for row in csv_reader:
            arr_row = []
            for item in row:
                if is_float(item):
                    arr_row.append(float(item))
                else:
                    arr_row.append(None)
            arr.append(arr_row)
        # print(arr)

    IDX_delta_first_prod_vs_second_prob = 8
    IDX_seed = 0
    IDX_adv_label = 6
    threshold_arr = np.arange(0, 30, 0.1)
    num_adv_arr = []
    total_samples = []
    for threshold in threshold_arr:
        num_adv = 0
        n_sample = 0
        for row in arr:
            if row[IDX_delta_first_prod_vs_second_prob] <= threshold:
                if row[IDX_adv_label] is not None:
                    num_adv += 1
                n_sample += 1
        print(f'threshold = {threshold}:  #adv = {num_adv}, # samples = {n_sample}')
        num_adv_arr.append(num_adv)
        total_samples.append(n_sample)

    num_adv_arr = np.asarray(num_adv_arr)
    num_adv_arr = np.round(num_adv_arr / num_adv_arr[len(num_adv_arr) - 1] * 100, 1)

    total_samples = np.asarray(total_samples)
    total_samples = np.round(total_samples / total_samples[len(total_samples) - 1] * 100, 1)
    return num_adv_arr, total_samples, threshold_arr

    # plt.plot(threshold_arr, num_adv_arr, '--b', label='percentage of adversaries')
    # plt.plot(threshold_arr, total_samples, 'b', label='percentage of samples used to attack\n(10k first samples on MNSIT)')

if __name__ == '__main__':
    num_adv_arr, total_samples, threshold_arr = get_analysis('/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_simard_10k_first_samples/summary.csv')
    plt.plot(threshold_arr, num_adv_arr, '--b')
    plt.plot(threshold_arr, total_samples, 'b', label = 'mnist_simard (11 adv)')

    num_adv_arr, total_samples, threshold_arr = get_analysis('/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_simple_10k_first_samples/summary.csv')
    plt.plot(threshold_arr, num_adv_arr, '--g')
    plt.plot(threshold_arr, total_samples, 'g', label = 'mnist_simple (426 adv)')

    num_adv_arr, total_samples, threshold_arr = get_analysis('/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_deepcheck_10k_first_samples/summary.csv')
    plt.plot(threshold_arr, num_adv_arr, '--r')
    plt.plot(threshold_arr, total_samples, 'r', label = 'mnist_deepcheck (771 adv)')

    num_adv_arr, total_samples, threshold_arr = get_analysis('/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_ann_keras_10k_first_samples/summary.csv')
    plt.plot(threshold_arr, num_adv_arr, '--c')
    plt.plot(threshold_arr, total_samples, 'c', label = 'mnist_ann_keras (502 adv)')

    # num_adv_arr, total_samples, threshold_arr = get_analysis('/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_deepcheck_10k_1_pixel_attack/summary.csv')
    # plt.plot(threshold_arr, num_adv_arr, '--c')
    # plt.plot(threshold_arr, total_samples, 'c', label = 'mnist_deepcheck (#adv = 219)')
    #
    # num_adv_arr, total_samples, threshold_arr = get_analysis('/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_ann_keras_1_pixel_attack/summary.csv')
    # plt.plot(threshold_arr, num_adv_arr, '--r')
    # plt.plot(threshold_arr, total_samples, 'r', label = 'mnist_ann_keras (#adv = 131)')
    #
    # num_adv_arr, total_samples, threshold_arr = get_analysis('/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_simple_10k_1_pixel_attack/summary.csv')
    # plt.plot(threshold_arr, num_adv_arr, '--b')
    # plt.plot(threshold_arr, total_samples, 'b', label = 'mnist_simple (#adv = 244)')

    # plt.title('The summary of attacked models in terms of percentage\n(solid line style: the percentage of attacked samples, '
    #           '\ndashed line style: the percentage of generated adversaries)')
    plt.xlabel('threshold')
    plt.ylabel('percentage')
    plt.tight_layout()
    plt.legend()
    plt.show()
