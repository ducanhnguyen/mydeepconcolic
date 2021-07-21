from os import listdir
from os.path import isfile, join

import numpy as np
from numpy import load

if __name__ == '__main__':
    BASE_PATH = '/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/Lenet_v2/allfeature/tmp_S3step6'
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
    pred_arr = []
    for row in per_pixel_by_prediction_arr:
        pred_arr.append(len(row))
    pred_arr = np.asarray(pred_arr)

    # line chart
    idxes = np.arange(0, 784)
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

    # print(avg_arr)

    # utilities.plot_line_chart(idxes, avg_arr, x_title="#number of prediction",
    #                           y_title="#restored pixel/#different pixel (%)",
    #                           title=f"#adv = {len(per_pixel_by_prediction_arr)}; min prediction = {np.min(pred_arr)}"
    #                                 f", max prediction = {np.max(pred_arr)}")

    # print(np.average(avg_arr))
    import csv
    with open('/Users/ducanhnguyen/Documents/mydeepconcolic/Lenet_v2_all_feature_S3step6.csv', mode='w') as f:
        seed = csv.writer(f)
        for value in avg_arr:
            seed.writerow([str(np.round(value, 5))])
        f.close()
    #
    # # boxplot
    # import matplotlib.pyplot as plt
    #
    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Box plot of changed pixels')
    # ax1.boxplot(pred_arr)
    # plt.show()