import csv
import os
from pylab import *
import numpy as np


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


def remove_outlier(data, idx):
    distance = []
    for row in data:
        distance.append(row[idx])
    # print(distance)
    q25, q75 = percentile(distance, 25), percentile(distance, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower = q25 - cut_off
    upper = q75 + cut_off

    new = []
    for item in distance:
        if item >= lower and item <= upper:
            new.append(item)
    avg = np.average(new)
    return lower, avg, upper, distance


if __name__ == '__main__':
    #
    IDX_L0_before = 3
    IDX_L0_after = 4
    IDX_L2_before = 5
    IDX_L2_after = 6

    data = read_data(
        "/Users/ducanhnguyen/Documents/mydeepconcolic/optimization_batch3/Alexnet/ae_saliency_Alexnet_9to7_rankingJSMA_step6/out.csv")
    l0_min, l0_avg, l0_max, l0_distance = remove_outlier(data, IDX_L0_after)
    l2_min, l2_avg, l2_max, l2_distance = remove_outlier(data, IDX_L2_after)

    if l0_min < 0:
        l0_min = min(l0_distance)
    print(
        f"{int(np.round(l0_min))} & {int(np.round(l0_avg))} & {int(np.round(l0_max))} & {np.round(l2_min, 2)} & {np.round(l2_avg, 2)} & {np.round(l2_max, 2)}")
