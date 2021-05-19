from os import listdir
from os.path import isfile, join
import numpy as np
from numpy import load

from src.utils import utilities

if __name__ == '__main__':
    BASE_PATH = '/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/Alexnet/ae_border/autoencoder_models/OUT'
    onlyfiles = [f for f in listdir(BASE_PATH) if isfile(join(BASE_PATH, f))]
    per_pixel_by_prediction_arr = []

    for file in onlyfiles:

        if file.endswith('.npy'):
            full_path = BASE_PATH + f'/{file}'
            per_pixel_by_prediction = load(full_path)
            per_pixel_by_prediction_arr.append(per_pixel_by_prediction)

    per_pixel_by_prediction_arr = np.asarray(per_pixel_by_prediction_arr)
    avg_arr = []

    min_pred = 999
    max_pred = 0
    for row in per_pixel_by_prediction_arr:
        if min_pred > len(row):
            min_pred = len(row)
        if max_pred < len(row):
            max_pred = len(row)

    idxes = np.arange(0, max_pred)
    for idx in idxes:
        print(f'index {idx}')
        tmp = []
        for row in per_pixel_by_prediction_arr:
            if idx < len(row):
                tmp.append(row[idx])
            else:
                tmp.append(row[-1])
        if len(tmp) == 0:
            avg_arr.append(None)
        else:
            tmp = np.asarray(tmp)
            avg = np.average(tmp)
            avg_arr.append(avg)

    print(avg_arr)

    utilities.plot_line_chart(idxes, avg_arr, x_title="#number of prediction", y_title="#restored pixel/#different pixel (%)",
                              title=f"#adv = {len(per_pixel_by_prediction_arr)}; min predictin = {min_pred}, max prediction = {max_pred}")

