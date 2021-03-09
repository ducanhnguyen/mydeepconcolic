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


if __name__ == '__main__':
    with open('/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_ann_keras/summary.csv') as csv_file:
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
    threshold_arr = np.arange(0, 100, 0.1)
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

    # plot
    num_adv_arr = np.asarray(num_adv_arr)
    num_adv_arr = np.round(num_adv_arr/num_adv_arr[len(num_adv_arr)-1] * 100,1)
    plt.plot(threshold_arr, num_adv_arr, label = 'percentage of adversaries')

    total_samples = np.asarray(total_samples)
    total_samples = np.round(total_samples/total_samples[len(total_samples)-1] * 100,1)
    plt.plot(threshold_arr, total_samples, label = 'percentage of samples used to attack\n(10k first samples on MNSIT)')

    plt.xlabel('threshold')
    plt.ylabel('percentage')
    plt.tight_layout()
    plt.legend()
    plt.show()